"""
Test data persistence and file operations
"""

import os

import numpy as np
import pytest

from vectorlitedb import VectorLiteDB


def get_wal_path(db_path):
    """Return the expected WAL path for a database file."""
    return db_path + ".wal"


class TestBasicPersistence:
    def test_data_persists_after_close(self, temp_db_path, sample_vectors):
        """Test that data persists after closing and reopening database."""
        db1 = VectorLiteDB(temp_db_path, dimension=3, distance_metric="l2")

        for vec_id, vector in sample_vectors.items():
            metadata = {"id": vec_id, "persisted": True}
            db1.insert(vec_id, vector, metadata)

        initial_count = len(db1)
        db1.close()

        db2 = VectorLiteDB(temp_db_path)

        assert len(db2) == initial_count
        assert db2.dimension == 3
        assert db2.distance_metric == "l2"

        for vec_id, expected_vector in sample_vectors.items():
            vector, metadata = db2.get(vec_id)
            assert vector == expected_vector
            assert metadata["id"] == vec_id
            assert metadata["persisted"] is True

        db2.close()

    def test_search_works_after_reopen(self, temp_db_path, sample_vectors):
        """Test that search functionality works after reopening database."""
        db1 = VectorLiteDB(temp_db_path, dimension=3)
        for vec_id, vector in sample_vectors.items():
            db1.insert(vec_id, vector)
        db1.close()

        db2 = VectorLiteDB(temp_db_path)
        results = db2.search([1.0, 0.0, 0.0], top_k=3)

        assert len(results) > 0
        assert results[0]["id"] == "vec1"
        assert results[0]["similarity"] > 0.99

        db2.close()

    def test_multiple_close_calls(self, temp_db_path, sample_vectors):
        """Test that multiple close() calls don't cause issues."""
        db = VectorLiteDB(temp_db_path, dimension=3)
        db.insert("test", sample_vectors["vec1"])

        db.close()
        db.close()
        db.close()

        assert not os.path.exists(get_wal_path(temp_db_path))

        db2 = VectorLiteDB(temp_db_path)
        vector, _ = db2.get("test")
        assert vector == sample_vectors["vec1"]
        db2.close()

    def test_insert_persists_across_reopen_without_close(
        self, temp_db_path, sample_vectors
    ):
        """Test that inserts are visible through WAL replay before close."""
        db = VectorLiteDB(temp_db_path, dimension=3)
        db.insert("auto_save", sample_vectors["vec1"])

        assert os.path.exists(get_wal_path(temp_db_path))

        db2 = VectorLiteDB(temp_db_path)
        vector, _ = db2.get("auto_save")
        assert vector == sample_vectors["vec1"]

        db.close()
        db2.close()

    def test_delete_persists_across_reopen_without_close(
        self, temp_db_path, sample_vectors
    ):
        """Test that deletes are visible through WAL replay before close."""
        db = VectorLiteDB(temp_db_path, dimension=3)
        db.insert("delete_me", sample_vectors["vec1"])
        db.close()

        db = VectorLiteDB(temp_db_path)
        db.delete("delete_me")

        db2 = VectorLiteDB(temp_db_path)
        with pytest.raises(KeyError):
            db2.get("delete_me")

        db.close()
        db2.close()


class TestWALPersistence:
    def test_existing_snapshot_without_wal_opens_unchanged(
        self, temp_db_path, sample_vectors
    ):
        """Test that a plain snapshot file still opens without WAL involvement."""
        db = VectorLiteDB(temp_db_path, dimension=3)
        db.insert("snap", sample_vectors["vec1"], {"source": "snapshot"})
        db.close()

        wal_path = get_wal_path(temp_db_path)
        assert not os.path.exists(wal_path)

        reopened = VectorLiteDB(temp_db_path)
        vector, metadata = reopened.get("snap")
        assert vector == sample_vectors["vec1"]
        assert metadata == {"source": "snapshot"}
        reopened.close()

    def test_wal_grows_with_data_before_checkpoint(self, temp_db_path):
        """Test that writes accumulate in WAL until a checkpoint happens."""
        db = VectorLiteDB(temp_db_path, dimension=100)
        wal_path = get_wal_path(temp_db_path)
        initial_db_size = os.path.getsize(temp_db_path)

        for i in range(50):
            vector = [float(i)] * 100
            metadata = {"index": i, "data": "x" * 100}
            db.insert(f"vec_{i}", vector, metadata)

        assert os.path.exists(wal_path)
        assert os.path.getsize(wal_path) > 1000
        assert os.path.getsize(temp_db_path) == initial_db_size

        db.close()

        assert not os.path.exists(wal_path)
        assert os.path.getsize(temp_db_path) > initial_db_size

    def test_checkpoint_on_close_compacts_state_and_clears_wal(
        self, temp_db_path, sample_vectors
    ):
        """Test that close checkpoints the latest state into the main snapshot."""
        db = VectorLiteDB(temp_db_path, dimension=3)
        wal_path = get_wal_path(temp_db_path)
        for vec_id, vector in sample_vectors.items():
            db.insert(vec_id, vector, {"data": "x" * 1000})

        snapshot_size_before_close = os.path.getsize(temp_db_path)
        wal_size_before_close = os.path.getsize(wal_path)
        assert wal_size_before_close > 0

        for vec_id in list(sample_vectors.keys())[:-1]:
            db.delete(vec_id)

        assert os.path.getsize(wal_path) > wal_size_before_close

        db.close()

        assert not os.path.exists(wal_path)
        assert os.path.getsize(temp_db_path) > snapshot_size_before_close

        reopened = VectorLiteDB(temp_db_path)
        assert len(reopened) == 1
        vector, metadata = reopened.get("vec5")
        assert vector == sample_vectors["vec5"]
        assert metadata["data"] == "x" * 1000
        reopened.close()

    def test_threshold_checkpoint_clears_wal_during_long_running_session(
        self, temp_db_path, sample_vectors, monkeypatch
    ):
        """Test that the op threshold triggers a checkpoint before close."""
        monkeypatch.setattr(VectorLiteDB, "_CHECKPOINT_INTERVAL", 2)

        db = VectorLiteDB(temp_db_path, dimension=3)
        wal_path = get_wal_path(temp_db_path)

        db.insert("vec1", sample_vectors["vec1"])
        assert os.path.exists(wal_path)

        db.insert("vec2", sample_vectors["vec2"])
        assert not os.path.exists(wal_path)

        reopened = VectorLiteDB(temp_db_path)
        assert len(reopened) == 2
        reopened.close()
        db.close()

    def test_duplicate_insert_and_missing_delete_do_not_append_invalid_wal_entries(
        self, temp_db_path, sample_vectors
    ):
        """Test that failing writes leave the WAL unchanged."""
        db = VectorLiteDB(temp_db_path, dimension=3)
        wal_path = get_wal_path(temp_db_path)

        with pytest.raises(KeyError, match="ID not found: missing"):
            db.delete("missing")
        assert not os.path.exists(wal_path)

        db.insert("vec1", sample_vectors["vec1"])
        wal_size = os.path.getsize(wal_path)

        with pytest.raises(ValueError, match="ID already exists: vec1"):
            db.insert("vec1", sample_vectors["vec2"])
        assert os.path.getsize(wal_path) == wal_size

        with pytest.raises(KeyError, match="ID not found: missing"):
            db.delete("missing")
        assert os.path.getsize(wal_path) == wal_size

        db.close()

    def test_startup_recovery_replays_latest_wal_state(
        self, temp_db_path, sample_vectors
    ):
        """Test crash-style recovery by reopening without checkpointing first."""
        db = VectorLiteDB(temp_db_path, dimension=3)
        db.insert("vec1", sample_vectors["vec1"])
        db.insert("vec2", sample_vectors["vec2"])
        db.delete("vec1")

        recovered = VectorLiteDB(temp_db_path)
        with pytest.raises(KeyError):
            recovered.get("vec1")

        vector, _ = recovered.get("vec2")
        assert vector == sample_vectors["vec2"]

        recovered.close()
        db.close()

    def test_truncated_final_wal_record_is_ignored(self, temp_db_path, sample_vectors):
        """Test that a partial final WAL line does not corrupt valid prior records."""
        db = VectorLiteDB(temp_db_path, dimension=3)
        db.insert("vec1", sample_vectors["vec1"])
        wal_path = get_wal_path(temp_db_path)

        with open(wal_path, "a", encoding="utf-8") as wal_file:
            wal_file.write('{"op":"insert","id":"broken"')

        recovered = VectorLiteDB(temp_db_path)
        vector, _ = recovered.get("vec1")
        assert vector == sample_vectors["vec1"]

        with pytest.raises(KeyError):
            recovered.get("broken")

        recovered.close()
        db.close()


class TestFileFormat:
    def test_file_created_on_init(self, temp_db_path):
        """Test that database file is created on initialization."""
        assert not os.path.exists(temp_db_path)

        db = VectorLiteDB(temp_db_path, dimension=5)

        assert os.path.exists(temp_db_path)
        assert os.path.isfile(temp_db_path)
        assert os.path.getsize(temp_db_path) > 0
        assert not os.path.exists(get_wal_path(temp_db_path))

        db.close()


class TestFileSystemEdgeCases:
    def test_readonly_file_fails_gracefully(self, temp_db_path, sample_vectors):
        """Test behavior when file becomes read-only."""
        db = VectorLiteDB(temp_db_path, dimension=3)
        db.insert("test", sample_vectors["vec1"])
        db.close()

        os.chmod(temp_db_path, 0o444)

        try:
            db2 = VectorLiteDB(temp_db_path)
            vector, _ = db2.get("test")
            assert vector == sample_vectors["vec1"]

            with pytest.raises(PermissionError):
                db2.insert("new", sample_vectors["vec2"])

            db2.close()
        finally:
            os.chmod(temp_db_path, 0o644)

    def test_disk_full_simulation(self, temp_db_path):
        """Test behavior when disk is full (simulated)."""
        db = VectorLiteDB(temp_db_path, dimension=1000000)

        try:
            huge_vector = [1.0] * 1000000
            db.insert("huge", huge_vector)
        except (MemoryError, OSError):
            pass

        db.close()

    def test_concurrent_file_access(self, temp_db_path, sample_vectors):
        """Test that concurrent access is handled appropriately."""
        db1 = VectorLiteDB(temp_db_path, dimension=3)
        db1.insert("from_db1", sample_vectors["vec1"])

        db2 = VectorLiteDB(temp_db_path)

        vector1, _ = db1.get("from_db1")
        vector2, _ = db2.get("from_db1")
        assert vector1 == vector2

        try:
            db1.insert("from_db1_again", sample_vectors["vec2"])
            db2.insert("from_db2", sample_vectors["vec3"])
        except Exception:
            pass

        db1.close()
        db2.close()


class TestDirectoryHandling:
    def test_nested_directory_creation(self, sample_vectors):
        """Test database creation in nested directories."""
        nested_path = "test_nested/deep/very/deep/database.db"

        try:
            db = VectorLiteDB(nested_path, dimension=3)
            db.insert("test", sample_vectors["vec1"])
            db.close()

            assert os.path.exists(nested_path)
            assert os.path.isfile(nested_path)

            db2 = VectorLiteDB(nested_path)
            vector, _ = db2.get("test")
            assert vector == sample_vectors["vec1"]
            db2.close()
        finally:
            for path in [nested_path, nested_path + ".wal", nested_path + ".tmp"]:
                if os.path.exists(path):
                    os.remove(path)

            dirs_to_remove = [
                "test_nested/deep/very/deep",
                "test_nested/deep/very",
                "test_nested/deep",
                "test_nested",
            ]
            for dir_path in dirs_to_remove:
                if os.path.exists(dir_path):
                    os.rmdir(dir_path)

    def test_relative_vs_absolute_paths(self, sample_vectors):
        """Test that relative and absolute paths work correctly."""
        rel_path = "relative_test.db"

        try:
            db1 = VectorLiteDB(rel_path, dimension=3)
            db1.insert("test", sample_vectors["vec1"])
            db1.close()

            abs_path = os.path.abspath(rel_path)
            db2 = VectorLiteDB(abs_path)
            vector, _ = db2.get("test")
            assert vector == sample_vectors["vec1"]
            db2.close()
        finally:
            for path in [
                rel_path,
                rel_path + ".wal",
                rel_path + ".tmp",
                os.path.abspath(rel_path),
                os.path.abspath(rel_path) + ".wal",
                os.path.abspath(rel_path) + ".tmp",
            ]:
                if os.path.exists(path):
                    os.remove(path)


class TestBackupAndRestore:
    def test_manual_file_copy_backup(self, temp_db_path, sample_vectors):
        """Test that manually copying database file works as backup."""
        db = VectorLiteDB(temp_db_path, dimension=3)
        for vec_id, vector in sample_vectors.items():
            db.insert(vec_id, vector, {"backup_test": True})
        db.close()

        backup_path = temp_db_path + ".backup"
        with open(temp_db_path, "rb") as src, open(backup_path, "wb") as dst:
            dst.write(src.read())

        try:
            db = VectorLiteDB(temp_db_path)
            db.delete("vec1")
            db.insert("new_vec", [9, 9, 9], {"modified": True})
            db.close()

            os.remove(temp_db_path)
            os.rename(backup_path, temp_db_path)

            db_restored = VectorLiteDB(temp_db_path)
            assert len(db_restored) == len(sample_vectors)

            vector, metadata = db_restored.get("vec1")
            assert vector == sample_vectors["vec1"]
            assert metadata["backup_test"] is True

            with pytest.raises(KeyError):
                db_restored.get("new_vec")

            db_restored.close()
        finally:
            if os.path.exists(backup_path):
                os.remove(backup_path)


class TestLargeDataPersistence:
    def test_large_database_persistence(self, temp_db_path):
        """Test persistence with large amounts of data."""
        db = VectorLiteDB(temp_db_path, dimension=128)

        np.random.seed(42)
        vectors_to_insert = {}

        for i in range(1000):
            vector = np.random.rand(128).tolist()
            vectors_to_insert[f"vec_{i:04d}"] = vector
            metadata = {"index": i, "batch": i // 100}
            db.insert(f"vec_{i:04d}", vector, metadata)

        db.close()

        db2 = VectorLiteDB(temp_db_path)
        assert len(db2) == 1000

        for i in [0, 100, 500, 999]:
            vec_id = f"vec_{i:04d}"
            vector, metadata = db2.get(vec_id)
            assert vector == vectors_to_insert[vec_id]
            assert metadata["index"] == i

        query = np.random.rand(128).tolist()
        results = db2.search(query, top_k=10)
        assert len(results) == 10

        db2.close()

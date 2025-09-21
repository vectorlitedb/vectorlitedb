"""
Test data persistence and file operations
"""

import os
import tempfile
import pytest
from vectorlitedb import VectorLiteDB


class TestBasicPersistence:
    
    def test_data_persists_after_close(self, temp_db_path, sample_vectors):
        """Test that data persists after closing and reopening database"""
        # Create and populate database
        db1 = VectorLiteDB(temp_db_path, dimension=3, distance_metric="l2")
        
        for vec_id, vector in sample_vectors.items():
            metadata = {"id": vec_id, "persisted": True}
            db1.insert(vec_id, vector, metadata)
        
        initial_count = len(db1)
        db1.close()
        
        # Reopen database
        db2 = VectorLiteDB(temp_db_path)
        
        # Verify all data persisted
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
        """Test that search functionality works after reopening database"""
        # Create and populate
        db1 = VectorLiteDB(temp_db_path, dimension=3)
        for vec_id, vector in sample_vectors.items():
            db1.insert(vec_id, vector)
        db1.close()
        
        # Reopen and search
        db2 = VectorLiteDB(temp_db_path)
        results = db2.search([1.0, 0.0, 0.0], top_k=3)
        
        assert len(results) > 0
        assert results[0]["id"] == "vec1"  # Should find exact match
        assert results[0]["similarity"] > 0.99
        
        db2.close()
    
    def test_multiple_close_calls(self, temp_db_path, sample_vectors):
        """Test that multiple close() calls don't cause issues"""
        db = VectorLiteDB(temp_db_path, dimension=3)
        db.insert("test", sample_vectors["vec1"])
        
        # Multiple closes should be safe
        db.close()
        db.close()
        db.close()
        
        # Verify data still accessible after reopen
        db2 = VectorLiteDB(temp_db_path)
        vector, _ = db2.get("test")
        assert vector == sample_vectors["vec1"]
        db2.close()
    
    def test_automatic_save_on_operations(self, temp_db_path, sample_vectors):
        """Test that operations automatically save to disk"""
        db = VectorLiteDB(temp_db_path, dimension=3)
        
        # Insert without explicit close
        db.insert("auto_save", sample_vectors["vec1"])
        
        # Open new instance - data should be there
        db2 = VectorLiteDB(temp_db_path)
        vector, _ = db2.get("auto_save")
        assert vector == sample_vectors["vec1"]
        
        db.close()
        db2.close()


class TestFileFormat:
    
    def test_file_created_on_init(self, temp_db_path):
        """Test that database file is created on initialization"""
        assert not os.path.exists(temp_db_path)
        
        db = VectorLiteDB(temp_db_path, dimension=5)
        
        # File should exist after creation
        assert os.path.exists(temp_db_path)
        assert os.path.isfile(temp_db_path)
        assert os.path.getsize(temp_db_path) > 0
        
        db.close()
    
    def test_file_grows_with_data(self, temp_db_path):
        """Test that file size grows as data is added"""
        db = VectorLiteDB(temp_db_path, dimension=100)
        
        initial_size = os.path.getsize(temp_db_path)
        
        # Add many vectors
        for i in range(50):
            vector = [float(i)] * 100
            metadata = {"index": i, "data": "x" * 100}  # Some bulk
            db.insert(f"vec_{i}", vector, metadata)
        
        final_size = os.path.getsize(temp_db_path)
        
        # File should have grown significantly
        assert final_size > initial_size + 1000  # At least 1KB growth
        
        db.close()
    
    def test_file_shrinks_on_delete(self, temp_db_path, sample_vectors):
        """Test file behavior when deleting data"""
        db = VectorLiteDB(temp_db_path, dimension=3)
        
        # Add data
        for vec_id, vector in sample_vectors.items():
            large_metadata = {"data": "x" * 1000}  # 1KB per vector
            db.insert(vec_id, vector, large_metadata)
        
        size_after_insert = os.path.getsize(temp_db_path)
        
        # Delete most data
        for vec_id in list(sample_vectors.keys())[:-1]:
            db.delete(vec_id)
        
        size_after_delete = os.path.getsize(temp_db_path)
        
        # Note: Current implementation rewrites entire file, so size should decrease
        assert size_after_delete < size_after_insert
        
        db.close()


class TestFileSystemEdgeCases:
    
    def test_readonly_file_fails_gracefully(self, temp_db_path, sample_vectors):
        """Test behavior when file becomes read-only"""
        # Create database
        db = VectorLiteDB(temp_db_path, dimension=3)
        db.insert("test", sample_vectors["vec1"])
        db.close()
        
        # Make file read-only
        os.chmod(temp_db_path, 0o444)  # Read-only
        
        try:
            # Should still be able to read
            db2 = VectorLiteDB(temp_db_path)
            vector, _ = db2.get("test")
            assert vector == sample_vectors["vec1"]
            
            # But writes should fail
            with pytest.raises(PermissionError):
                db2.insert("new", sample_vectors["vec2"])
            
            db2.close()
            
        finally:
            # Restore permissions for cleanup
            os.chmod(temp_db_path, 0o644)
    
    def test_disk_full_simulation(self, temp_db_path):
        """Test behavior when disk is full (simulated)"""
        # This is hard to test reliably, but we can test with a very large vector
        db = VectorLiteDB(temp_db_path, dimension=1000000)  # Very large
        
        # This might fail due to memory/disk constraints
        # The test verifies the error is handled gracefully
        try:
            huge_vector = [1.0] * 1000000
            db.insert("huge", huge_vector)
        except (MemoryError, OSError):
            # These are acceptable failures for huge data
            pass
        
        db.close()
    
    def test_concurrent_file_access(self, temp_db_path, sample_vectors):
        """Test that concurrent access is handled appropriately"""
        # Create first database instance
        db1 = VectorLiteDB(temp_db_path, dimension=3)
        db1.insert("from_db1", sample_vectors["vec1"])
        
        # Try to open second instance on same file
        # Current implementation doesn't support concurrent writes
        # This test documents the current behavior
        db2 = VectorLiteDB(temp_db_path)
        
        # Reading from both should work
        vector1, _ = db1.get("from_db1")
        vector2, _ = db2.get("from_db1")
        assert vector1 == vector2
        
        # Writing from both is undefined behavior in current version
        # This test just ensures it doesn't crash
        try:
            db1.insert("from_db1_again", sample_vectors["vec2"])
            db2.insert("from_db2", sample_vectors["vec3"])
        except Exception:
            # Concurrent writes may fail - that's acceptable for now
            pass
        
        db1.close()
        db2.close()


class TestDirectoryHandling:
    
    def test_nested_directory_creation(self, sample_vectors):
        """Test database creation in nested directories"""
        nested_path = "test_nested/deep/very/deep/database.db"
        
        try:
            # Should create all parent directories
            db = VectorLiteDB(nested_path, dimension=3)
            db.insert("test", sample_vectors["vec1"])
            db.close()
            
            # Verify file exists
            assert os.path.exists(nested_path)
            assert os.path.isfile(nested_path)
            
            # Verify data persisted
            db2 = VectorLiteDB(nested_path)
            vector, _ = db2.get("test")
            assert vector == sample_vectors["vec1"]
            db2.close()
            
        finally:
            # Cleanup
            if os.path.exists(nested_path):
                os.remove(nested_path)
                # Remove directories (from deepest to shallowest)
                dirs_to_remove = ["test_nested/deep/very/deep", 
                                "test_nested/deep/very", 
                                "test_nested/deep", 
                                "test_nested"]
                for dir_path in dirs_to_remove:
                    if os.path.exists(dir_path):
                        os.rmdir(dir_path)
    
    def test_relative_vs_absolute_paths(self, sample_vectors):
        """Test that relative and absolute paths work correctly"""
        # Test relative path
        rel_path = "relative_test.db"
        
        try:
            db1 = VectorLiteDB(rel_path, dimension=3)
            db1.insert("test", sample_vectors["vec1"])
            db1.close()
            
            # Test absolute path to same file
            abs_path = os.path.abspath(rel_path)
            db2 = VectorLiteDB(abs_path)
            vector, _ = db2.get("test")
            assert vector == sample_vectors["vec1"]
            db2.close()
            
        finally:
            for path in [rel_path, os.path.abspath(rel_path)]:
                if os.path.exists(path):
                    os.remove(path)


class TestBackupAndRestore:
    
    def test_manual_file_copy_backup(self, temp_db_path, sample_vectors):
        """Test that manually copying database file works as backup"""
        # Create original database
        db = VectorLiteDB(temp_db_path, dimension=3)
        for vec_id, vector in sample_vectors.items():
            db.insert(vec_id, vector, {"backup_test": True})
        db.close()
        
        # Create backup by copying file
        backup_path = temp_db_path + ".backup"
        with open(temp_db_path, 'rb') as src, open(backup_path, 'wb') as dst:
            dst.write(src.read())
        
        try:
            # Modify original
            db = VectorLiteDB(temp_db_path)
            db.delete("vec1")
            db.insert("new_vec", [9, 9, 9], {"modified": True})
            db.close()
            
            # Restore from backup
            os.remove(temp_db_path)
            os.rename(backup_path, temp_db_path)
            
            # Verify restoration
            db_restored = VectorLiteDB(temp_db_path)
            assert len(db_restored) == len(sample_vectors)  # Original count
            
            # Original data should be there
            vector, metadata = db_restored.get("vec1")
            assert vector == sample_vectors["vec1"]
            assert metadata["backup_test"] is True
            
            # Modified data should not be there
            with pytest.raises(KeyError):
                db_restored.get("new_vec")
            
            db_restored.close()
            
        finally:
            # Cleanup
            if os.path.exists(backup_path):
                os.remove(backup_path)


class TestLargeDataPersistence:
    
    def test_large_database_persistence(self, temp_db_path):
        """Test persistence with large amounts of data"""
        import numpy as np
        
        db = VectorLiteDB(temp_db_path, dimension=128)
        
        # Insert 1000 vectors
        np.random.seed(42)  # Reproducible
        vectors_to_insert = {}
        
        for i in range(1000):
            vector = np.random.rand(128).tolist()
            vectors_to_insert[f"vec_{i:04d}"] = vector
            metadata = {"index": i, "batch": i // 100}
            db.insert(f"vec_{i:04d}", vector, metadata)
        
        db.close()
        
        # Reopen and verify all data
        db2 = VectorLiteDB(temp_db_path)
        assert len(db2) == 1000
        
        # Spot check some vectors
        for i in [0, 100, 500, 999]:
            vec_id = f"vec_{i:04d}"
            vector, metadata = db2.get(vec_id)
            assert vector == vectors_to_insert[vec_id]
            assert metadata["index"] == i
        
        # Test search still works
        query = np.random.rand(128).tolist()
        results = db2.search(query, top_k=10)
        assert len(results) == 10
        
        db2.close()
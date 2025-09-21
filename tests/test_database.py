"""
Test database creation, initialization, and configuration
"""

import os
import pytest
from vectorlitedb import VectorLiteDB


class TestDatabaseCreation:
    
    def test_create_new_database_with_dimension(self, temp_db_path):
        """Test creating a new database with specified dimension"""
        db = VectorLiteDB(temp_db_path, dimension=128)
        
        assert len(db) == 0
        assert db.dimension == 128
        assert db.distance_metric == "cosine"  # Default
        assert os.path.exists(temp_db_path)
        
        db.close()
    
    def test_create_database_with_different_metrics(self, temp_db_path):
        """Test creating database with different distance metrics"""
        metrics = ["cosine", "l2", "dot"]
        
        for metric in metrics:
            db_path = f"{temp_db_path}_{metric}"
            db = VectorLiteDB(db_path, dimension=64, distance_metric=metric)
            
            assert db.distance_metric == metric
            assert db.dimension == 64
            
            db.close()
            os.remove(db_path)
    
    def test_dimension_required_for_new_database(self, temp_db_path):
        """Test that dimension is required when creating new database"""
        with pytest.raises(ValueError, match="Dimension required for new database"):
            VectorLiteDB(temp_db_path)
    
    def test_invalid_distance_metric_fails(self, temp_db_path):
        """Test that invalid distance metric raises appropriate error"""
        # This test expects the validation to be added to the code
        db = VectorLiteDB(temp_db_path, dimension=10, distance_metric="invalid")
        
        # Should fail when trying to use invalid metric in search
        db.insert("test", [1.0] * 10)
        with pytest.raises(ValueError, match="Unknown distance metric"):
            db.search([1.0] * 10)
        
        db.close()


class TestDatabaseLoading:
    
    def test_open_existing_database(self, temp_db_path):
        """Test opening an existing database file"""
        # Create and populate database
        db1 = VectorLiteDB(temp_db_path, dimension=5, distance_metric="l2")
        db1.insert("test_vec", [1, 2, 3, 4, 5], {"created": "first"})
        db1.close()
        
        # Reopen database
        db2 = VectorLiteDB(temp_db_path)
        
        assert len(db2) == 1
        assert db2.dimension == 5
        assert db2.distance_metric == "l2"
        
        vector, metadata = db2.get("test_vec")
        assert vector == [1, 2, 3, 4, 5]
        assert metadata == {"created": "first"}
        
        db2.close()
    
    def test_open_empty_existing_file(self, temp_db_path):
        """Test opening an empty file (should fail gracefully)"""
        # Create empty file
        with open(temp_db_path, 'w') as f:
            pass
        
        with pytest.raises((ValueError, EOFError, KeyError)):
            VectorLiteDB(temp_db_path)
    
    def test_open_corrupted_database(self, temp_db_path):
        """Test opening a corrupted database file"""
        # Create corrupted file
        with open(temp_db_path, 'wb') as f:
            f.write(b"corrupted data")
        
        with pytest.raises((ValueError, EOFError, KeyError)):
            VectorLiteDB(temp_db_path)


class TestDatabaseProperties:
    
    def test_database_repr(self, empty_db, temp_db_path):
        """Test database string representation"""
        repr_str = repr(empty_db)
        
        assert "VectorLiteDB" in repr_str
        assert "vectors=0" in repr_str
        assert "dim=3" in repr_str
        assert temp_db_path.split('/')[-1] in repr_str
    
    def test_database_len(self, empty_db, sample_vectors):
        """Test database length tracking"""
        assert len(empty_db) == 0
        
        empty_db.insert("vec1", sample_vectors["vec1"])
        assert len(empty_db) == 1
        
        empty_db.insert("vec2", sample_vectors["vec2"])
        assert len(empty_db) == 2
        
        empty_db.delete("vec1")
        assert len(empty_db) == 1
    
    def test_context_manager(self, temp_db_path, sample_vectors):
        """Test database as context manager"""
        # Use database in context manager
        with VectorLiteDB(temp_db_path, dimension=3) as db:
            db.insert("test", sample_vectors["vec1"], {"context": "manager"})
            assert len(db) == 1
        
        # Verify data persisted after auto-close
        with VectorLiteDB(temp_db_path) as db:
            assert len(db) == 1
            vector, metadata = db.get("test")
            assert vector == sample_vectors["vec1"]
            assert metadata == {"context": "manager"}


class TestFilePaths:
    
    def test_relative_path(self, sample_vectors):
        """Test database with relative path"""
        db_path = "test_relative.db"
        
        try:
            db = VectorLiteDB(db_path, dimension=2)
            db.insert("test", sample_vectors["vec1"][:2])
            db.close()
            
            # Should create file in current directory
            assert os.path.exists(db_path)
            
            # Reopen and verify
            db2 = VectorLiteDB(db_path)
            assert len(db2) == 1
            db2.close()
            
        finally:
            if os.path.exists(db_path):
                os.remove(db_path)
    
    def test_nested_directory_creation(self, sample_vectors):
        """Test database creation in nested directory"""
        db_path = "test_dir/nested/database.db"
        
        try:
            db = VectorLiteDB(db_path, dimension=2)
            db.insert("test", sample_vectors["vec1"][:2])
            db.close()
            
            assert os.path.exists(db_path)
            assert os.path.isfile(db_path)
            
        finally:
            if os.path.exists(db_path):
                os.remove(db_path)
                os.rmdir("test_dir/nested")
                os.rmdir("test_dir")
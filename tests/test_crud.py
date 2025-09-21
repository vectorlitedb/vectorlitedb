"""
Test CRUD operations: Create, Read, Update, Delete
"""

import pytest
from vectorlitedb import VectorLiteDB


class TestInsertOperations:
    
    def test_insert_vector_with_metadata(self, empty_db, sample_vectors):
        """Test inserting vector with metadata"""
        metadata = {"label": "test", "category": "A", "score": 0.95}
        empty_db.insert("vec1", sample_vectors["vec1"], metadata)
        
        assert len(empty_db) == 1
        vector, returned_metadata = empty_db.get("vec1")
        assert vector == sample_vectors["vec1"]
        assert returned_metadata == metadata
    
    def test_insert_vector_without_metadata(self, empty_db, sample_vectors):
        """Test inserting vector without metadata"""
        empty_db.insert("vec1", sample_vectors["vec1"])
        
        assert len(empty_db) == 1
        vector, metadata = empty_db.get("vec1")
        assert vector == sample_vectors["vec1"]
        assert metadata is None
    
    def test_insert_multiple_vectors(self, empty_db, sample_vectors):
        """Test inserting multiple vectors"""
        for vec_id, vector in sample_vectors.items():
            metadata = {"id": vec_id, "index": len(empty_db)}
            empty_db.insert(vec_id, vector, metadata)
        
        assert len(empty_db) == len(sample_vectors)
        
        # Verify all vectors are retrievable
        for vec_id, expected_vector in sample_vectors.items():
            vector, metadata = empty_db.get(vec_id)
            assert vector == expected_vector
            assert metadata["id"] == vec_id
    
    def test_insert_duplicate_id_fails(self, empty_db, sample_vectors):
        """Test that inserting duplicate ID raises error"""
        empty_db.insert("duplicate", sample_vectors["vec1"])
        
        with pytest.raises(ValueError, match="ID already exists: duplicate"):
            empty_db.insert("duplicate", sample_vectors["vec2"])
    
    def test_insert_wrong_dimension_fails(self, empty_db):
        """Test that wrong dimension vector raises error"""
        # Database is 3D, try to insert 4D vector
        with pytest.raises(ValueError, match="Vector dimension mismatch: expected 3, got 4"):
            empty_db.insert("wrong_dim", [1, 2, 3, 4])
        
        # Try 2D vector
        with pytest.raises(ValueError, match="Vector dimension mismatch: expected 3, got 2"):
            empty_db.insert("wrong_dim", [1, 2])
    
    def test_insert_empty_vector(self, temp_db_path):
        """Test inserting empty vector in 0-dimensional space"""
        db = VectorLiteDB(temp_db_path, dimension=0)
        db.insert("empty", [], {"type": "empty"})
        
        vector, metadata = db.get("empty")
        assert vector == []
        assert metadata["type"] == "empty"
        
        db.close()
    
    def test_insert_large_vector(self, temp_db_path):
        """Test inserting large dimensional vector"""
        dimension = 1536  # OpenAI embedding size
        large_vector = [0.1] * dimension
        
        db = VectorLiteDB(temp_db_path, dimension=dimension)
        db.insert("large", large_vector, {"size": "large"})
        
        vector, metadata = db.get("large")
        assert len(vector) == dimension
        assert vector == large_vector
        
        db.close()


class TestReadOperations:
    
    def test_get_existing_vector(self, populated_db):
        """Test retrieving existing vector"""
        vector, metadata = populated_db.get("vec1")
        
        assert vector == [1.0, 0.0, 0.0]
        assert "category" in metadata
        assert "index" in metadata
    
    def test_get_nonexistent_vector_fails(self, empty_db):
        """Test that retrieving non-existent vector raises error"""
        with pytest.raises(KeyError, match="ID not found: nonexistent"):
            empty_db.get("nonexistent")
    
    def test_get_after_insert(self, empty_db, sample_vectors):
        """Test that get works immediately after insert"""
        metadata = {"immediate": True}
        empty_db.insert("immediate", sample_vectors["vec1"], metadata)
        
        vector, returned_metadata = empty_db.get("immediate")
        assert vector == sample_vectors["vec1"]
        assert returned_metadata == metadata


class TestDeleteOperations:
    
    def test_delete_existing_vector(self, populated_db):
        """Test deleting existing vector"""
        initial_count = len(populated_db)
        assert initial_count > 0
        
        populated_db.delete("vec1")
        
        assert len(populated_db) == initial_count - 1
        
        # Verify vector is gone
        with pytest.raises(KeyError):
            populated_db.get("vec1")
    
    def test_delete_nonexistent_vector_fails(self, empty_db):
        """Test that deleting non-existent vector raises error"""
        with pytest.raises(KeyError, match="ID not found: nonexistent"):
            empty_db.delete("nonexistent")
    
    def test_delete_all_vectors(self, populated_db, sample_vectors):
        """Test deleting all vectors one by one"""
        for vec_id in sample_vectors.keys():
            populated_db.delete(vec_id)
        
        assert len(populated_db) == 0
        
        # Verify all are gone
        for vec_id in sample_vectors.keys():
            with pytest.raises(KeyError):
                populated_db.get(vec_id)
    
    def test_delete_and_reinsert(self, empty_db, sample_vectors):
        """Test deleting vector and reinserting with same ID"""
        # Insert
        empty_db.insert("reinsert", sample_vectors["vec1"], {"version": 1})
        
        # Delete
        empty_db.delete("reinsert")
        
        # Reinsert with same ID but different data
        empty_db.insert("reinsert", sample_vectors["vec2"], {"version": 2})
        
        # Verify new data
        vector, metadata = empty_db.get("reinsert")
        assert vector == sample_vectors["vec2"]
        assert metadata["version"] == 2


class TestUpdateOperations:
    """
    Note: VectorLiteDB doesn't have native update - it's delete + insert
    These tests verify that update pattern works correctly
    """
    
    def test_update_vector_via_delete_insert(self, empty_db, sample_vectors):
        """Test updating vector by deleting and reinserting"""
        # Initial insert
        empty_db.insert("update_test", sample_vectors["vec1"], {"version": 1})
        
        # Update: delete old, insert new
        empty_db.delete("update_test")
        empty_db.insert("update_test", sample_vectors["vec2"], {"version": 2})
        
        # Verify update
        vector, metadata = empty_db.get("update_test")
        assert vector == sample_vectors["vec2"]
        assert metadata["version"] == 2
        
        # Verify length stayed same
        assert len(empty_db) == 1
    
    def test_update_metadata_only(self, empty_db, sample_vectors):
        """Test updating just metadata (keeping same vector)"""
        # Initial insert
        empty_db.insert("meta_update", sample_vectors["vec1"], {"status": "old"})
        
        # Update metadata
        empty_db.delete("meta_update")
        empty_db.insert("meta_update", sample_vectors["vec1"], {"status": "new", "updated": True})
        
        # Verify
        vector, metadata = empty_db.get("meta_update")
        assert vector == sample_vectors["vec1"]  # Same vector
        assert metadata["status"] == "new"
        assert metadata["updated"] is True


class TestLargeMetadata:
    
    def test_insert_large_metadata(self, empty_db, sample_vectors):
        """Test inserting vector with large metadata"""
        large_metadata = {
            "description": "x" * 1000,  # Large string
            "numbers": list(range(100)),  # Large list
            "nested": {"deep": {"very": {"nested": {"data": True}}}},
            "unicode": "🚀 Vector with emoji metadata 🧠"
        }
        
        empty_db.insert("large_meta", sample_vectors["vec1"], large_metadata)
        
        vector, metadata = empty_db.get("large_meta")
        assert vector == sample_vectors["vec1"]
        assert metadata == large_metadata
    
    def test_special_characters_in_metadata(self, empty_db, sample_vectors):
        """Test metadata with special characters"""
        special_metadata = {
            "quotes": 'Text with "quotes" and \'apostrophes\'',
            "newlines": "Line 1\nLine 2\nLine 3",
            "unicode": "Café, naïve, résumé, Москва, 北京",
            "symbols": "!@#$%^&*()[]{}|\\:;\"'<>?,./"
        }
        
        empty_db.insert("special", sample_vectors["vec1"], special_metadata)
        
        vector, metadata = empty_db.get("special")
        assert metadata == special_metadata


class TestVectorTypes:
    
    def test_integer_vectors(self, empty_db):
        """Test vectors with integer values"""
        int_vector = [1, 2, 3]
        empty_db.insert("integers", int_vector)
        
        vector, _ = empty_db.get("integers")
        assert vector == int_vector
    
    def test_float_vectors(self, empty_db):
        """Test vectors with float values"""
        float_vector = [1.5, 2.7, 3.14159]
        empty_db.insert("floats", float_vector)
        
        vector, _ = empty_db.get("floats")
        assert vector == float_vector
    
    def test_mixed_numeric_vectors(self, empty_db):
        """Test vectors with mixed int/float values"""
        mixed_vector = [1, 2.5, 3]
        empty_db.insert("mixed", mixed_vector)
        
        vector, _ = empty_db.get("mixed")
        assert vector == mixed_vector
    
    def test_zero_vectors(self, empty_db):
        """Test vectors with zero values"""
        zero_vector = [0.0, 0.0, 0.0]
        empty_db.insert("zeros", zero_vector)
        
        vector, _ = empty_db.get("zeros")
        assert vector == zero_vector
    
    def test_negative_vectors(self, empty_db):
        """Test vectors with negative values"""
        negative_vector = [-1.0, -2.5, -3.0]
        empty_db.insert("negative", negative_vector)
        
        vector, _ = empty_db.get("negative")
        assert vector == negative_vector
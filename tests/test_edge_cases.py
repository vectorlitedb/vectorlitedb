"""
Test edge cases, error handling, and boundary conditions
"""

import pytest
import math
import numpy as np
from vectorlitedb import VectorLiteDB


class TestInputValidation:
    
    def test_invalid_dimension_types(self, temp_db_path):
        """Test that invalid dimension types raise appropriate errors"""
        with pytest.raises((TypeError, ValueError)):
            VectorLiteDB(temp_db_path, dimension="invalid")
        
        with pytest.raises((TypeError, ValueError)):
            VectorLiteDB(temp_db_path, dimension=3.14)
        
        with pytest.raises((TypeError, ValueError)):
            VectorLiteDB(temp_db_path, dimension=None)
    
    def test_negative_dimension(self, temp_db_path):
        """Test that negative dimensions are handled"""
        with pytest.raises(ValueError):
            VectorLiteDB(temp_db_path, dimension=-1)
    
    def test_zero_dimension(self, temp_db_path):
        """Test zero-dimensional vectors"""
        # Zero dimension should work
        db = VectorLiteDB(temp_db_path, dimension=0)
        db.insert("empty", [], {"type": "zero_dim"})
        
        vector, metadata = db.get("empty")
        assert vector == []
        assert metadata["type"] == "zero_dim"
        
        # Search should work with empty vectors
        results = db.search([], top_k=1)
        assert len(results) == 1
        assert results[0]["id"] == "empty"
        
        db.close()
    
    def test_very_large_dimension(self, temp_db_path):
        """Test handling of very large dimensions"""
        # Test reasonable large dimension
        large_dim = 10000
        db = VectorLiteDB(temp_db_path, dimension=large_dim)
        
        # Should be able to create but might be slow
        large_vector = [0.1] * large_dim
        db.insert("large", large_vector)
        
        vector, _ = db.get("large")
        assert len(vector) == large_dim
        
        db.close()


class TestSpecialValues:
    
    def test_infinity_values(self, empty_db):
        """Test handling of infinity values in vectors"""
        inf_vector = [float('inf'), 1.0, float('-inf')]
        
        # Should handle infinity values
        empty_db.insert("infinity", inf_vector)
        vector, _ = empty_db.get("infinity")
        
        assert math.isinf(vector[0]) and vector[0] > 0
        assert vector[1] == 1.0
        assert math.isinf(vector[2]) and vector[2] < 0
        
        # Search with infinity should work
        results = empty_db.search([float('inf'), 0, 0], top_k=1)
        assert len(results) == 1
    
    def test_nan_values(self, empty_db):
        """Test handling of NaN values in vectors"""
        nan_vector = [float('nan'), 1.0, 2.0]
        
        # Should handle NaN values (behavior may vary)
        empty_db.insert("nan", nan_vector)
        vector, _ = empty_db.get("nan")
        
        assert math.isnan(vector[0])
        assert vector[1] == 1.0
        
        # Search behavior with NaN is undefined but shouldn't crash
        try:
            results = empty_db.search([float('nan'), 0, 0], top_k=1)
            # If it doesn't crash, results might be unpredictable
            assert isinstance(results, list)
        except (ValueError, TypeError):
            # NaN in search might raise error - that's acceptable
            pass
    
    def test_very_small_numbers(self, empty_db):
        """Test handling of very small floating point numbers"""
        tiny_vector = [1e-308, 2e-308, 3e-308]  # Near float64 limits
        
        empty_db.insert("tiny", tiny_vector)
        vector, _ = empty_db.get("tiny")
        
        # Should preserve small numbers
        assert vector == tiny_vector
        
        # Search should work
        results = empty_db.search(tiny_vector, top_k=1)
        assert len(results) == 1
        assert results[0]["id"] == "tiny"
    
    def test_very_large_numbers(self, empty_db):
        """Test handling of very large floating point numbers"""
        huge_vector = [1e100, 2e100, 3e100]
        
        empty_db.insert("huge", huge_vector)
        vector, _ = empty_db.get("huge")
        
        assert vector == huge_vector
        
        # Search should work without overflow
        results = empty_db.search(huge_vector, top_k=1)
        assert len(results) == 1
        assert results[0]["id"] == "huge"


class TestIDValidation:
    
    def test_empty_string_id(self, empty_db, sample_vectors):
        """Test handling of empty string as ID"""
        empty_db.insert("", sample_vectors["vec1"])
        
        vector, _ = empty_db.get("")
        assert vector == sample_vectors["vec1"]
    
    def test_very_long_id(self, empty_db, sample_vectors):
        """Test handling of very long IDs"""
        long_id = "x" * 10000  # 10KB ID
        
        empty_db.insert(long_id, sample_vectors["vec1"])
        vector, _ = empty_db.get(long_id)
        assert vector == sample_vectors["vec1"]
    
    def test_unicode_ids(self, empty_db, sample_vectors):
        """Test handling of Unicode characters in IDs"""
        unicode_ids = [
            "café",
            "北京",
            "🚀🧠",
            "نص عربي",
            "Ελληνικά"
        ]
        
        for unicode_id in unicode_ids:
            empty_db.insert(unicode_id, sample_vectors["vec1"])
            vector, _ = empty_db.get(unicode_id)
            assert vector == sample_vectors["vec1"]
    
    def test_special_character_ids(self, empty_db, sample_vectors):
        """Test IDs with special characters"""
        special_ids = [
            "id-with-hyphens",
            "id_with_underscores",
            "id.with.dots",
            "id@email.com",
            "id with spaces",
            "id/with/slashes",
            "id\\with\\backslashes",
            "id:with:colons",
            "id;with;semicolons",
            "id'with'quotes",
            'id"with"doublequotes'
        ]
        
        for special_id in special_ids:
            empty_db.insert(special_id, sample_vectors["vec1"])
            vector, _ = empty_db.get(special_id)
            assert vector == sample_vectors["vec1"]
    
    def test_numeric_string_ids(self, empty_db, sample_vectors):
        """Test IDs that are numeric strings"""
        numeric_ids = ["123", "0", "-456", "3.14159", "1e10"]
        
        for numeric_id in numeric_ids:
            empty_db.insert(numeric_id, sample_vectors["vec1"])
            vector, _ = empty_db.get(numeric_id)
            assert vector == sample_vectors["vec1"]


class TestMetadataEdgeCases:
    
    def test_none_metadata(self, empty_db, sample_vectors):
        """Test explicit None metadata"""
        empty_db.insert("none_meta", sample_vectors["vec1"], None)
        
        vector, metadata = empty_db.get("none_meta")
        assert vector == sample_vectors["vec1"]
        assert metadata is None
    
    def test_empty_dict_metadata(self, empty_db, sample_vectors):
        """Test empty dictionary metadata"""
        empty_db.insert("empty_meta", sample_vectors["vec1"], {})
        
        vector, metadata = empty_db.get("empty_meta")
        assert vector == sample_vectors["vec1"]
        assert metadata == {}
    
    def test_deeply_nested_metadata(self, empty_db, sample_vectors):
        """Test deeply nested metadata structures"""
        deep_metadata = {
            "level1": {
                "level2": {
                    "level3": {
                        "level4": {
                            "level5": {
                                "deep_value": "found"
                            }
                        }
                    }
                }
            }
        }
        
        empty_db.insert("deep", sample_vectors["vec1"], deep_metadata)
        
        vector, metadata = empty_db.get("deep")
        assert metadata["level1"]["level2"]["level3"]["level4"]["level5"]["deep_value"] == "found"
    
    def test_metadata_with_special_types(self, empty_db, sample_vectors):
        """Test metadata with various Python types"""
        complex_metadata = {
            "string": "text",
            "integer": 42,
            "float": 3.14159,
            "boolean": True,
            "list": [1, 2, "three", 4.0],
            "none_value": None,
            "nested_list": [[1, 2], [3, 4]],
            "mixed_types": [1, "two", 3.0, True, None]
        }
        
        empty_db.insert("complex", sample_vectors["vec1"], complex_metadata)
        
        vector, metadata = empty_db.get("complex")
        assert metadata == complex_metadata
    
    def test_metadata_with_unicode(self, empty_db, sample_vectors):
        """Test metadata with Unicode content"""
        unicode_metadata = {
            "english": "Hello, World!",
            "chinese": "你好世界",
            "arabic": "مرحبا بالعالم",
            "emoji": "🌍🚀🧠💡",
            "mixed": "Mix of English and 中文 and العربية"
        }
        
        empty_db.insert("unicode", sample_vectors["vec1"], unicode_metadata)
        
        vector, metadata = empty_db.get("unicode")
        assert metadata == unicode_metadata


class TestBoundaryConditions:
    
    def test_single_vector_database(self, empty_db, sample_vectors):
        """Test database with only one vector"""
        empty_db.insert("only", sample_vectors["vec1"], {"alone": True})
        
        # Basic operations
        assert len(empty_db) == 1
        vector, metadata = empty_db.get("only")
        assert vector == sample_vectors["vec1"]
        
        # Search should return the one vector
        results = empty_db.search([0, 0, 0], top_k=5)
        assert len(results) == 1
        assert results[0]["id"] == "only"
        
        # Filter that excludes the vector
        results = empty_db.search([0, 0, 0], top_k=5, 
                                filter=lambda meta: meta.get("alone") is False)
        assert len(results) == 0
    
    def test_maximum_reasonable_vectors(self, temp_db_path):
        """Test database with many vectors (stress test)"""
        db = VectorLiteDB(temp_db_path, dimension=10)
        
        # Insert many vectors (adjust count based on test time constraints)
        num_vectors = 5000
        for i in range(num_vectors):
            vector = [float(i % 10)] * 10  # Simple pattern
            metadata = {"index": i, "group": i // 100}
            db.insert(f"vec_{i:05d}", vector, metadata)
        
        assert len(db) == num_vectors
        
        # Spot check some vectors
        vector, metadata = db.get("vec_02500")
        assert metadata["index"] == 2500
        assert metadata["group"] == 25
        
        # Search should still work
        results = db.search([5.0] * 10, top_k=10)
        assert len(results) == 10
        
        db.close()
    
    def test_rapid_insert_delete_cycles(self, empty_db, sample_vectors):
        """Test rapid insertion and deletion cycles"""
        # Rapid insert/delete cycles
        for cycle in range(100):
            vec_id = f"cycle_{cycle}"
            empty_db.insert(vec_id, sample_vectors["vec1"])
            assert len(empty_db) == 1
            
            empty_db.delete(vec_id)
            assert len(empty_db) == 0
    
    def test_search_with_extreme_top_k(self, populated_db):
        """Test search with extreme top_k values"""
        total_vectors = len(populated_db)
        
        # top_k much larger than available
        results = populated_db.search([1, 0, 0], top_k=1000000)
        assert len(results) == total_vectors
        
        # top_k of 1
        results = populated_db.search([1, 0, 0], top_k=1)
        assert len(results) == 1


class TestErrorRecovery:
    
    def test_partial_operation_recovery(self, empty_db, sample_vectors):
        """Test that database remains consistent after failed operations"""
        # Insert valid data
        empty_db.insert("valid", sample_vectors["vec1"])
        initial_count = len(empty_db)
        
        # Try invalid operations
        try:
            empty_db.insert("valid", sample_vectors["vec2"])  # Duplicate ID
        except ValueError:
            pass
        
        try:
            empty_db.insert("invalid_dim", [1, 2, 3, 4])  # Wrong dimension
        except ValueError:
            pass
        
        # Database should be unchanged
        assert len(empty_db) == initial_count
        vector, _ = empty_db.get("valid")
        assert vector == sample_vectors["vec1"]
    
    def test_search_with_corrupted_data_simulation(self, empty_db):
        """Test search behavior with potentially problematic data"""
        # Insert vectors that might cause numerical issues
        problematic_vectors = [
            ("zeros", [0.0, 0.0, 0.0]),
            ("huge", [1e100, 1e100, 1e100]),
            ("tiny", [1e-100, 1e-100, 1e-100]),
            ("mixed", [1e100, 0.0, 1e-100])
        ]
        
        for vec_id, vector in problematic_vectors:
            empty_db.insert(vec_id, vector)
        
        # Search should handle these gracefully
        for query_name, query_vector in problematic_vectors:
            results = empty_db.search(query_vector, top_k=2)
            assert len(results) > 0
            
            # Similarities should be valid numbers (not NaN)
            for result in results:
                similarity = result["similarity"]
                assert isinstance(similarity, (int, float))
                assert not math.isnan(similarity)


class TestMemoryConstraints:
    
    def test_large_vector_handling(self, temp_db_path):
        """Test handling of individual large vectors"""
        # Test with reasonably large vector (careful not to exhaust memory)
        large_dim = 50000  # 50K dimensions
        
        db = VectorLiteDB(temp_db_path, dimension=large_dim)
        
        # Create large vector efficiently
        large_vector = [0.001] * large_dim
        large_vector[0] = 1.0  # Make it distinctive
        
        db.insert("large_individual", large_vector)
        
        # Verify it can be retrieved
        retrieved_vector, _ = db.get("large_individual")
        assert len(retrieved_vector) == large_dim
        assert retrieved_vector[0] == 1.0
        assert retrieved_vector[1] == 0.001
        
        # Search should work
        query = [1.0] + [0.001] * (large_dim - 1)
        results = db.search(query, top_k=1)
        assert len(results) == 1
        assert results[0]["id"] == "large_individual"
        
        db.close()
    
    def test_memory_efficient_operations(self, temp_db_path):
        """Test that operations don't unnecessarily consume memory"""
        # This test is more about not crashing than specific assertions
        db = VectorLiteDB(temp_db_path, dimension=1000)
        
        # Insert vectors in batches and verify memory doesn't grow unbounded
        for batch in range(10):
            for i in range(100):
                vec_id = f"batch_{batch}_vec_{i}"
                vector = [float(batch)] * 1000
                db.insert(vec_id, vector, {"batch": batch})
        
        # Basic operations should still work
        assert len(db) == 1000
        results = db.search([5.0] * 1000, top_k=5)
        assert len(results) == 5
        
        db.close()
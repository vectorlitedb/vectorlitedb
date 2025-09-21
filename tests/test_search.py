"""
Test search functionality and similarity algorithms
"""

import pytest
import numpy as np
from vectorlitedb import VectorLiteDB


class TestBasicSearch:
    
    def test_search_empty_database(self, empty_db):
        """Test searching empty database returns empty list"""
        results = empty_db.search([1.0, 0.0, 0.0], top_k=5)
        assert results == []
    
    def test_exact_match_search(self, populated_db):
        """Test that exact match returns highest similarity"""
        # Search for exact match of vec1
        results = populated_db.search([1.0, 0.0, 0.0], top_k=3)
        
        assert len(results) > 0
        assert results[0]["id"] == "vec1"  # Exact match should be first
        assert results[0]["similarity"] > 0.99  # Very high similarity
    
    def test_search_returns_requested_count(self, populated_db):
        """Test that search returns requested number of results"""
        total_vectors = len(populated_db)
        
        # Request fewer than available
        results = populated_db.search([1.0, 0.0, 0.0], top_k=2)
        assert len(results) == 2
        
        # Request more than available
        results = populated_db.search([1.0, 0.0, 0.0], top_k=total_vectors + 10)
        assert len(results) == total_vectors
    
    def test_search_result_structure(self, populated_db):
        """Test that search results have correct structure"""
        results = populated_db.search([1.0, 0.0, 0.0], top_k=1)
        
        assert len(results) == 1
        result = results[0]
        
        # Check required fields
        assert "id" in result
        assert "similarity" in result
        assert "metadata" in result
        
        # Check types
        assert isinstance(result["id"], str)
        assert isinstance(result["similarity"], (int, float))
        assert isinstance(result["metadata"], (dict, type(None)))
    
    def test_search_similarity_ordering(self, populated_db):
        """Test that results are ordered by similarity (highest first)"""
        results = populated_db.search([1.0, 0.0, 0.0], top_k=5)
        
        similarities = [r["similarity"] for r in results]
        
        # Should be in descending order
        for i in range(len(similarities) - 1):
            assert similarities[i] >= similarities[i + 1]


class TestDistanceMetrics:
    
    def test_cosine_similarity(self, temp_db_path):
        """Test cosine similarity metric"""
        db = VectorLiteDB(temp_db_path, dimension=3, distance_metric="cosine")
        
        # Insert orthogonal vectors
        db.insert("vec1", [1.0, 0.0, 0.0])
        db.insert("vec2", [0.0, 1.0, 0.0])
        db.insert("vec3", [2.0, 0.0, 0.0])  # Same direction as vec1, different magnitude
        
        # Search with vec1
        results = db.search([1.0, 0.0, 0.0], top_k=3)
        
        # vec1 should be first (exact match)
        assert results[0]["id"] == "vec1"
        
        # vec3 should be second (same direction) with high similarity
        vec3_result = next(r for r in results if r["id"] == "vec3")
        assert vec3_result["similarity"] > 0.99  # Cosine similarity ignores magnitude
        
        # vec2 should be last (orthogonal) with lower similarity
        vec2_result = next(r for r in results if r["id"] == "vec2")
        assert vec2_result["similarity"] < vec3_result["similarity"]
        
        db.close()
    
    def test_l2_distance(self, temp_db_path):
        """Test L2 (Euclidean) distance metric"""
        db = VectorLiteDB(temp_db_path, dimension=2, distance_metric="l2")
        
        # Insert vectors at different distances
        db.insert("origin", [0.0, 0.0])
        db.insert("close", [1.0, 0.0])
        db.insert("far", [10.0, 0.0])
        
        # Search from origin
        results = db.search([0.0, 0.0], top_k=3)
        
        # Origin should be first
        assert results[0]["id"] == "origin"
        
        # Close should be second
        assert results[1]["id"] == "close"
        
        # Far should be last
        assert results[2]["id"] == "far"
        
        # Similarities should decrease with distance
        assert results[0]["similarity"] > results[1]["similarity"] > results[2]["similarity"]
        
        db.close()
    
    def test_dot_product(self, temp_db_path):
        """Test dot product similarity metric"""
        db = VectorLiteDB(temp_db_path, dimension=2, distance_metric="dot")
        
        # Insert vectors with different dot products to [1, 1]
        db.insert("aligned", [1.0, 1.0])      # dot = 2
        db.insert("partial", [1.0, 0.0])     # dot = 1
        db.insert("opposite", [-1.0, -1.0])  # dot = -2
        
        # Search with [1, 1]
        results = db.search([1.0, 1.0], top_k=3)
        
        # Should be ordered by dot product (highest first)
        assert results[0]["id"] == "aligned"
        assert results[1]["id"] == "partial"
        assert results[2]["id"] == "opposite"
        
        db.close()


class TestSearchFiltering:
    
    def test_search_with_metadata_filter(self, populated_db):
        """Test search with metadata filtering"""
        # Filter for category A only
        results = populated_db.search(
            [1.0, 0.0, 0.0], 
            top_k=10,
            filter=lambda meta: meta.get("category") == "A"
        )
        
        # Should only return category A vectors
        for result in results:
            assert result["metadata"]["category"] == "A"
        
        # Should have fewer results than total
        assert len(results) < len(populated_db)
    
    def test_search_with_restrictive_filter(self, populated_db):
        """Test search with very restrictive filter"""
        # Filter that matches no vectors
        results = populated_db.search(
            [1.0, 0.0, 0.0],
            top_k=10,
            filter=lambda meta: meta.get("nonexistent_field") == "impossible"
        )
        
        assert results == []
    
    def test_search_with_complex_filter(self, populated_db):
        """Test search with complex metadata filter"""
        # Complex filter: category A AND index > 1
        results = populated_db.search(
            [1.0, 0.0, 0.0],
            top_k=10,
            filter=lambda meta: (meta.get("category") == "A" and 
                                meta.get("index", 0) > 1)
        )
        
        for result in results:
            metadata = result["metadata"]
            assert metadata["category"] == "A"
            assert metadata["index"] > 1
    
    def test_search_filter_with_none_metadata(self, empty_db):
        """Test filter handling when metadata is None"""
        # Insert vector without metadata
        empty_db.insert("no_meta", [1.0, 0.0, 0.0])
        
        # Filter should handle None metadata gracefully
        results = empty_db.search(
            [1.0, 0.0, 0.0],
            top_k=5,
            filter=lambda meta: meta is not None and meta.get("category") == "A"
        )
        
        assert results == []  # No results since metadata is None


class TestSearchEdgeCases:
    
    def test_search_with_zero_vector(self, populated_db):
        """Test searching with zero vector"""
        results = populated_db.search([0.0, 0.0, 0.0], top_k=3)
        
        # Should return results (behavior depends on distance metric)
        assert len(results) > 0
        assert all(isinstance(r["similarity"], (int, float)) for r in results)
    
    def test_search_with_very_large_values(self, populated_db):
        """Test searching with very large vector values"""
        large_query = [1e6, 1e6, 1e6]
        results = populated_db.search(large_query, top_k=3)
        
        assert len(results) > 0
        assert all(isinstance(r["similarity"], (int, float)) for r in results)
        assert all(not np.isnan(r["similarity"]) for r in results)
    
    def test_search_with_very_small_values(self, populated_db):
        """Test searching with very small vector values"""
        small_query = [1e-10, 1e-10, 1e-10]
        results = populated_db.search(small_query, top_k=3)
        
        assert len(results) > 0
        assert all(isinstance(r["similarity"], (int, float)) for r in results)
        assert all(not np.isnan(r["similarity"]) for r in results)
    
    def test_search_top_k_zero(self, populated_db):
        """Test searching with top_k=0"""
        results = populated_db.search([1.0, 0.0, 0.0], top_k=0)
        assert results == []
    
    def test_search_top_k_negative(self, populated_db):
        """Test searching with negative top_k"""
        # Should handle gracefully (return empty or all results)
        results = populated_db.search([1.0, 0.0, 0.0], top_k=-1)
        assert isinstance(results, list)


class TestSearchPerformance:
    
    def test_search_performance_baseline(self, large_db):
        """Test that search performance meets baseline expectations"""
        import time
        
        query_vector = [0.5] * 10
        
        # Time multiple searches
        times = []
        for _ in range(5):
            start = time.time()
            results = large_db.search(query_vector, top_k=10)
            end = time.time()
            times.append(end - start)
            
            # Verify results are correct
            assert len(results) == 10
            assert all("similarity" in r for r in results)
        
        avg_time = sum(times) / len(times)
        
        # Should complete search in reasonable time (adjust threshold as needed)
        assert avg_time < 0.1  # 100ms for 100 vectors
    
    def test_search_scales_with_top_k(self, large_db):
        """Test that search time doesn't dramatically increase with top_k"""
        import time
        
        query_vector = [0.5] * 10
        
        # Test different top_k values
        top_k_values = [1, 10, 50, 100]
        times = {}
        
        for k in top_k_values:
            start = time.time()
            results = large_db.search(query_vector, top_k=k)
            end = time.time()
            times[k] = end - start
            
            assert len(results) == min(k, len(large_db))
        
        # Times should be relatively similar (brute force search)
        time_ratio = times[100] / times[1]
        assert time_ratio < 3.0  # Shouldn't be more than 3x slower


class TestSearchConsistency:
    
    def test_search_deterministic(self, populated_db):
        """Test that search results are deterministic"""
        query = [0.7, 0.3, 0.1]
        
        # Run same search multiple times
        results1 = populated_db.search(query, top_k=3)
        results2 = populated_db.search(query, top_k=3)
        results3 = populated_db.search(query, top_k=3)
        
        # Results should be identical
        assert results1 == results2 == results3
    
    def test_search_after_modifications(self, empty_db, sample_vectors):
        """Test search consistency after database modifications"""
        # Insert some vectors
        for i, (vec_id, vector) in enumerate(list(sample_vectors.items())[:3]):
            empty_db.insert(vec_id, vector, {"index": i})
        
        query = [1.0, 0.0, 0.0]
        initial_results = empty_db.search(query, top_k=5)
        
        # Add more vectors
        for i, (vec_id, vector) in enumerate(list(sample_vectors.items())[3:]):
            empty_db.insert(vec_id, vector, {"index": i + 3})
        
        # Search again
        new_results = empty_db.search(query, top_k=5)
        
        # Should have more results
        assert len(new_results) > len(initial_results)
        
        # Original top result should still be similar (unless a better match was added)
        if len(initial_results) > 0:
            original_top = initial_results[0]
            # The original top result should still be in the new results
            original_ids = [r["id"] for r in initial_results]
            new_ids = [r["id"] for r in new_results]
            for original_id in original_ids:
                assert original_id in new_ids
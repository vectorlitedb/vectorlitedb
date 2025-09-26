"""
Shared pytest fixtures for VectorLiteDB tests
"""

import os
import tempfile
import pytest
import numpy as np
from vectorlitedb import VectorLiteDB


@pytest.fixture
def temp_db_path():
    """Create a temporary database file path"""
    # Generate a path without creating the file
    fd, db_path = tempfile.mkstemp(suffix='.db')
    os.close(fd)
    os.unlink(db_path)  # Remove the file but keep the path
    
    yield db_path
    
    # Cleanup
    if os.path.exists(db_path):
        os.remove(db_path)


@pytest.fixture
def empty_db(temp_db_path):
    """Create an empty VectorLiteDB instance"""
    db = VectorLiteDB(temp_db_path, dimension=3, distance_metric="cosine")
    yield db
    db.close()


@pytest.fixture
def sample_vectors():
    """Sample vectors for testing"""
    return {
        "vec1": [1.0, 0.0, 0.0],
        "vec2": [0.0, 1.0, 0.0], 
        "vec3": [0.0, 0.0, 1.0],
        "vec4": [0.7, 0.7, 0.0],  # Similar to vec1
        "vec5": [0.1, 0.9, 0.0]   # Similar to vec2
    }


@pytest.fixture
def populated_db(temp_db_path, sample_vectors):
    """Database pre-populated with sample vectors"""
    db = VectorLiteDB(temp_db_path, dimension=3, distance_metric="cosine")
    
    # Insert sample vectors with metadata
    for i, (vec_id, vector) in enumerate(sample_vectors.items()):
        metadata = {
            "category": "A" if i % 2 == 0 else "B",
            "index": i,
            "name": f"Vector {vec_id}"
        }
        db.insert(vec_id, vector, metadata)
    
    yield db
    db.close()


@pytest.fixture
def large_db(temp_db_path):
    """Database with many vectors for performance testing"""
    db = VectorLiteDB(temp_db_path, dimension=10, distance_metric="cosine")
    
    np.random.seed(42)  # Reproducible results
    for i in range(100):
        vector = np.random.rand(10).tolist()
        metadata = {"batch": i // 10, "index": i}
        db.insert(f"vec_{i:03d}", vector, metadata)
    
    yield db
    db.close()
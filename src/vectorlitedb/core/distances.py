import numpy as np
from typing import List, Union, Tuple

def validate_vectors(v1: List[float], v2: List[float]) -> Tuple[np.ndarray, np.ndarray]:
    """Validate and convert vector inputs.
    
    Args:
        v1: First vector
        v2: Second vector
        
    Returns:
        Tuple of numpy arrays
        
    Raises:
        ValueError: If vectors have different dimensions
    """
    v1_array = np.array(v1, dtype=np.float32)
    v2_array = np.array(v2, dtype=np.float32)
    
    if v1_array.shape != v2_array.shape:
        raise ValueError(f"Vector dimensions do not match: {v1_array.shape} vs {v2_array.shape}")
        
    return v1_array, v2_array

def l2_distance(v1: List[float], v2: List[float]) -> float:
    """Calculate Euclidean (L2) distance between two vectors.
    
    Args:
        v1: First vector
        v2: Second vector
        
    Returns:
        float: L2 distance
    """
    v1_array, v2_array = validate_vectors(v1, v2)
    return float(np.linalg.norm(v1_array - v2_array))

def cosine_similarity(v1: List[float], v2: List[float]) -> float:
    """Calculate cosine similarity between two vectors.
    
    Args:
        v1: First vector
        v2: Second vector
        
    Returns:
        float: Cosine similarity (1.0 for identical vectors, -1.0 for opposite vectors)
    """
    v1_array, v2_array = validate_vectors(v1, v2)
    
    # Prevent division by zero
    v1_norm = np.linalg.norm(v1_array)
    v2_norm = np.linalg.norm(v2_array)
    
    if v1_norm == 0 or v2_norm == 0:
        return 0.0
    
    return float(np.dot(v1_array, v2_array) / (v1_norm * v2_norm))

def dot_product(v1: List[float], v2: List[float]) -> float:
    """Calculate dot product between two vectors.
    
    Args:
        v1: First vector
        v2: Second vector
        
    Returns:
        float: Dot product
    """
    v1_array, v2_array = validate_vectors(v1, v2)
    return float(np.dot(v1_array, v2_array))

def distance_function_factory(distance_type: str):
    """Return the appropriate distance function based on the distance type.
    
    Args:
        distance_type: One of 'l2', 'cosine', or 'dot'
        
    Returns:
        function: Distance calculation function
        
    Raises:
        ValueError: If distance_type is not supported
    """
    if distance_type == "l2":
        return l2_distance
    elif distance_type == "cosine":
        return lambda v1, v2: 1.0 - cosine_similarity(v1, v2)
    elif distance_type == "dot":
        return lambda v1, v2: -dot_product(v1, v2)
    else:
        raise ValueError(f"Unsupported distance type: {distance_type}")
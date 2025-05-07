from typing import List, Dict, Any, Union, Optional
import numpy as np

def validate_vector(vector: List[float], dimension: Optional[int] = None) -> List[float]:
    """Validate a vector and return it as a list of floats.
    
    Args:
        vector: Input vector
        dimension: Expected dimension (if None, any dimension is accepted)
        
    Returns:
        list: Validated vector as list of floats
        
    Raises:
        ValueError: If vector is invalid or has incorrect dimension
    """
    if not isinstance(vector, (list, tuple, np.ndarray)):
        raise ValueError(f"Vector must be a list, tuple, or numpy array, got {type(vector)}")
    
    # Convert to list of floats
    try:
        vector_list = [float(x) for x in vector]
    except (ValueError, TypeError):
        raise ValueError("Vector must contain only numeric values")
    
    # Validate dimension if specified
    if dimension is not None and len(vector_list) != dimension:
        raise ValueError(f"Vector dimension mismatch: expected {dimension}, got {len(vector_list)}")
    
    return vector_list

def validate_id(id_value: str) -> str:
    """Validate an ID string.
    
    Args:
        id_value: ID string to validate
        
    Returns:
        str: Validated ID
        
    Raises:
        ValueError: If ID is invalid
    """
    if not isinstance(id_value, str):
        raise ValueError(f"ID must be a string, got {type(id_value)}")
    
    if not id_value:
        raise ValueError("ID cannot be empty")
    
    return id_value

def validate_metadata(metadata: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    """Validate metadata dictionary.
    
    Args:
        metadata: Metadata dictionary or None
        
    Returns:
        dict or None: Validated metadata
        
    Raises:
        ValueError: If metadata is invalid
    """
    if metadata is None:
        return None
    
    if not isinstance(metadata, dict):
        raise ValueError(f"Metadata must be a dictionary, got {type(metadata)}")
    
    # Make a copy to avoid modifying the original
    return metadata.copy()

def validate_top_k(top_k: int, max_value: int) -> int:
    """Validate top_k parameter.
    
    Args:
        top_k: Number of results to return
        max_value: Maximum allowed value
        
    Returns:
        int: Validated top_k
        
    Raises:
        ValueError: If top_k is invalid
    """
    if not isinstance(top_k, int):
        raise ValueError(f"top_k must be an integer, got {type(top_k)}")
    
    if top_k <= 0:
        raise ValueError(f"top_k must be positive, got {top_k}")
    
    return min(top_k, max_value)
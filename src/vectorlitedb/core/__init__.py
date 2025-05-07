# Core components for VectorLiteDB
from vectorlitedb.core.distances import (
    l2_distance, 
    cosine_similarity, 
    dot_product,
    distance_function_factory
)

__all__ = [
    "l2_distance", 
    "cosine_similarity", 
    "dot_product",
    "distance_function_factory"
]
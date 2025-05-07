import numpy as np
from typing import List, Dict, Any, Tuple, Optional, Callable

from vectorlitedb.core.distances import distance_function_factory
from vectorlitedb.utils.validation import validate_vector, validate_top_k

class FlatIndex:
    """Exact (brute-force) search index for vectors.
    
    This index performs exact search by comparing the query vector
    with every vector in the database.
    """
    
    def __init__(self, distance_type: str = 'cosine'):
        """Initialize flat index.
        
        Args:
            distance_type: Distance metric type ('l2', 'cosine', 'dot')
        """
        self.distance_type = distance_type
        self.distance_fn = distance_function_factory(distance_type)
    
    def search(self, query: List[float], vectors: Dict[str, List[float]], 
              top_k: int = 5, filter_fn: Optional[Callable[[Dict[str, Any]], bool]] = None,
              metadata: Dict[str, Optional[Dict[str, Any]]] = {}) -> List[Tuple[str, float]]:
        """Search for similar vectors.
        
        Args:
            query: Query vector
            vectors: Dictionary of vectors {id: vector}
            top_k: Number of results to return
            filter_fn: Optional function to filter results by metadata
            metadata: Dictionary of metadata {id: metadata}
            
        Returns:
            list: List of (id, distance) tuples sorted by similarity
        """
        if not vectors:
            return []
        
        # Validate inputs
        query = validate_vector(query)
        top_k = validate_top_k(top_k, len(vectors))
        
        # Calculate distances
        distances = []
        for vec_id, vector in vectors.items():
            # Apply metadata filter if provided
            if filter_fn is not None and vec_id in metadata:
                meta = metadata[vec_id]
                if meta is None or not filter_fn(meta):
                    continue
            
            distance = self.distance_fn(query, vector)
            distances.append((vec_id, distance))
        
        # Sort by distance (ascending)
        distances.sort(key=lambda x: x[1])
        
        # Return top k results
        return distances[:top_k]

class SearchResult:
    """Container for search results."""
    
    def __init__(self, id: str, similarity: float, metadata: Optional[Dict[str, Any]] = None):
        """Initialize search result.
        
        Args:
            id: Vector ID
            similarity: Similarity score (higher is better)
            metadata: Associated metadata
        """
        self.id = id
        self.similarity = similarity
        self.metadata = metadata or {}
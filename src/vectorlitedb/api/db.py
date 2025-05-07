# Main VectorLiteDB class implementation
from typing import List, Dict, Any, Tuple, Optional, Callable, Union

from vectorlitedb.storage.persistence import VectorStore
from vectorlitedb.core.index.flat import FlatIndex, SearchResult

class VectorLiteDB:
    def __init__(self, db_path: str, dimension: Optional[int] = None, distance_metric: str = "cosine"):
        """Initialize a new VectorLiteDB instance.
        
        Args:
            db_path: Path to the database file
            dimension: Vector dimension (required for new databases)
            distance_metric: Distance metric ("cosine", "l2", or "dot")
            
        Raises:
            ValueError: If dimension is not provided for a new database
        """
        self.db_path = db_path
        
        # Initialize storage
        self.store = VectorStore(db_path, dimension, distance_metric)
        
        # Initialize search index
        self.index = FlatIndex(distance_type=self.store.distance_type)
        
    def insert(self, id: str, vector: List[float], metadata: Optional[Dict[str, Any]] = None) -> None:
        """Insert a vector with optional metadata.
        
        Args:
            id: Unique identifier for the vector
            vector: Vector data as a list of floats
            metadata: Associated metadata
            
        Raises:
            ValueError: If vector has incorrect dimension or ID already exists
        """
        self.store.insert(id, vector, metadata)
        
    def search(self, query: List[float], top_k: int = 5, 
              filter: Optional[Callable[[Dict[str, Any]], bool]] = None) -> List[SearchResult]:
        """Search for similar vectors.
        
        Args:
            query: Query vector
            top_k: Number of results to return
            filter: Optional function to filter results by metadata
            
        Returns:
            list: Results containing IDs and similarity scores
        """
        # Use flat index for search
        results = self.index.search(
            query, 
            self.store.vectors, 
            top_k, 
            filter, 
            self.store.metadata
        )
        
        # Convert to SearchResult objects
        search_results = []
        for vec_id, distance in results:
            # Convert distance to similarity score (higher is better)
            if self.store.distance_type in ["l2", "cosine"]:
                # For distance metrics (lower is better), convert to similarity
                similarity = 1.0 / (1.0 + distance)
            else:
                # For dot product (higher is better, but negative due to factory)
                similarity = -distance
                
            metadata = self.store.metadata.get(vec_id)
            search_results.append(SearchResult(vec_id, similarity, metadata))
            
        return search_results
        
    def delete(self, id: str) -> None:
        """Delete a vector by ID.
        
        Args:
            id: ID of the vector to delete
            
        Raises:
            KeyError: If ID doesn't exist
        """
        self.store.delete(id)
        
    def close(self) -> None:
        """Close the database connection and save changes."""
        self.store.close()
        
    def __enter__(self):
        """Context manager entry."""
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()
"""
VectorLiteDB - A simple embedded vector database
"""

import os
import json
import struct
import numpy as np
from typing import List, Dict, Any, Tuple, Optional, Callable


# ============== Core Database Class ==============

class VectorLiteDB:
    """
    A simple file-based vector database for embeddings.
    
    Like SQLite, but for vectors. Stores everything in a single file.
    """
    
    def __init__(self, db_path: str, dimension: Optional[int] = None, distance_metric: str = "cosine"):
        """
        Initialize or open a vector database.
        
        Args:
            db_path: Path to the database file
            dimension: Vector dimension (required for new databases)
            distance_metric: One of "cosine", "l2", or "dot"
        """
        self.db_path = db_path
        self.distance_metric = distance_metric
        
        # In-memory storage
        self.vectors: Dict[str, List[float]] = {}
        self.metadata: Dict[str, Optional[Dict[str, Any]]] = {}
        
        # Load existing or create new
        if os.path.exists(db_path) and os.path.getsize(db_path) > 0:
            self._load()
        else:
            if dimension is None:
                raise ValueError("Dimension required for new database")
            self.dimension = dimension
            self._save()
    
    def insert(self, id: str, vector: List[float], metadata: Optional[Dict[str, Any]] = None) -> None:
        """
        Insert a vector with optional metadata.
        
        Args:
            id: Unique identifier
            vector: Embedding vector
            metadata: Optional metadata dictionary
        """
        # Validate
        if id in self.vectors:
            raise ValueError(f"ID already exists: {id}")
        
        if len(vector) != self.dimension:
            raise ValueError(f"Vector dimension mismatch: expected {self.dimension}, got {len(vector)}")
        
        # Store
        self.vectors[id] = vector
        self.metadata[id] = metadata
        
        # Persist
        self._save()
    
    def search(self, query: List[float], top_k: int = 5, 
              filter: Optional[Callable[[Dict[str, Any]], bool]] = None) -> List[Dict[str, Any]]:
        """
        Search for similar vectors using brute force.
        
        Args:
            query: Query vector
            top_k: Number of results to return
            filter: Optional function to filter by metadata
            
        Returns:
            List of results with id, similarity, and metadata
        """
        if not self.vectors:
            return []
        
        # Calculate distances to all vectors
        distances = []
        for vec_id, vector in self.vectors.items():
            # Apply filter if provided
            if filter and vec_id in self.metadata:
                meta = self.metadata[vec_id]
                if meta is None or not filter(meta):
                    continue
            
            # Calculate distance
            distance = self._calculate_distance(query, vector)
            distances.append((vec_id, distance))
        
        # Sort and get top k
        distances.sort(key=lambda x: x[1])
        results = []
        
        for vec_id, distance in distances[:top_k]:
            # Convert distance to similarity score
            if self.distance_metric in ["l2", "cosine"]:
                similarity = 1.0 / (1.0 + distance)
            else:  # dot product
                similarity = -distance
            
            results.append({
                "id": vec_id,
                "similarity": similarity,
                "metadata": self.metadata.get(vec_id, {})
            })
        
        return results
    
    def delete(self, id: str) -> None:
        """
        Delete a vector by ID.
        
        Args:
            id: Vector ID to delete
        """
        if id not in self.vectors:
            raise KeyError(f"ID not found: {id}")
        
        del self.vectors[id]
        del self.metadata[id]
        
        self._save()
    
    def get(self, id: str) -> Tuple[List[float], Optional[Dict[str, Any]]]:
        """
        Get a vector and its metadata by ID.
        
        Args:
            id: Vector ID
            
        Returns:
            Tuple of (vector, metadata)
        """
        if id not in self.vectors:
            raise KeyError(f"ID not found: {id}")
        
        return self.vectors[id], self.metadata[id]
    
    def close(self) -> None:
        """Save and close the database."""
        self._save()
    
    # ============== Internal Methods ==============
    
    def _calculate_distance(self, v1: List[float], v2: List[float]) -> float:
        """Calculate distance between two vectors."""
        v1_array = np.array(v1, dtype=np.float32)
        v2_array = np.array(v2, dtype=np.float32)
        
        if self.distance_metric == "l2":
            return float(np.linalg.norm(v1_array - v2_array))
        
        elif self.distance_metric == "cosine":
            # Cosine distance = 1 - cosine similarity
            norm1 = np.linalg.norm(v1_array)
            norm2 = np.linalg.norm(v2_array)
            if norm1 == 0 or norm2 == 0:
                return 1.0
            similarity = np.dot(v1_array, v2_array) / (norm1 * norm2)
            return float(1 - similarity)
        
        elif self.distance_metric == "dot":
            # Negative dot product (higher dot = more similar = lower distance)
            return float(-np.dot(v1_array, v2_array))
        
        else:
            raise ValueError(f"Unknown distance metric: {self.distance_metric}")
    
    def _save(self) -> None:
        """Save database to file."""
        os.makedirs(os.path.dirname(os.path.abspath(self.db_path)), exist_ok=True)
        
        with open(self.db_path, 'wb') as f:
            # Write header
            header = {
                "magic": "VLDB",
                "version": 1,
                "dimension": self.dimension,
                "distance_metric": self.distance_metric,
                "count": len(self.vectors)
            }
            header_json = json.dumps(header).encode('utf-8')
            f.write(struct.pack('I', len(header_json)))
            f.write(header_json)
            
            # Write vectors and metadata
            data = {
                "vectors": self.vectors,
                "metadata": self.metadata
            }
            data_json = json.dumps(data).encode('utf-8')
            f.write(data_json)
    
    def _load(self) -> None:
        """Load database from file."""
        with open(self.db_path, 'rb') as f:
            # Read header
            header_size = struct.unpack('I', f.read(4))[0]
            header_json = f.read(header_size)
            header = json.loads(header_json.decode('utf-8'))
            
            if header["magic"] != "VLDB":
                raise ValueError("Invalid file format")
            
            self.dimension = header["dimension"]
            self.distance_metric = header["distance_metric"]
            
            # Read data
            data_json = f.read()
            data = json.loads(data_json.decode('utf-8'))
            
            self.vectors = data["vectors"]
            self.metadata = data["metadata"]
    
    # ============== Context Manager ==============
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
    
    def __len__(self):
        return len(self.vectors)
    
    def __repr__(self):
        return f"VectorLiteDB(path='{self.db_path}', vectors={len(self.vectors)}, dim={self.dimension})"
import os
from typing import Dict, List, Any, Tuple, Optional

from vectorlitedb.storage.file_format import FileFormat
from vectorlitedb.utils.validation import validate_vector, validate_id, validate_metadata

class VectorStore:
    """Handles storage and retrieval of vectors and metadata."""
    
    def __init__(self, db_path: str, dimension: Optional[int] = None, distance_type: str = 'cosine'):
        """Initialize vector store.
        
        Args:
            db_path: Path to the database file
            dimension: Vector dimension (required for new databases)
            distance_type: Distance metric ('l2', 'cosine', 'dot')
            
        Raises:
            ValueError: If dimension is not provided for a new database
        """
        self.db_path = db_path
        self.file_format = FileFormat(db_path)
        
        # In-memory storage
        self.vectors: Dict[str, List[float]] = {}
        self.metadata: Dict[str, Optional[Dict[str, Any]]] = {}
        self.id_mapping: Dict[str, int] = {}
        self.next_pos = 0
        
        # Create new file or load existing
        try:
            if os.path.exists(db_path) and os.path.getsize(db_path) > 0:
                self._load_from_file()
            else:
                if dimension is None:
                    raise ValueError("Dimension must be specified for new database")
                self.dimension = dimension
                self.distance_type = distance_type
                self.file_format.create_new_file(dimension, distance_type)
        except (FileNotFoundError, ValueError) as e:
            # If file is corrupted or doesn't exist
            if dimension is None:
                raise ValueError("Dimension must be specified for new database") from e
            self.dimension = dimension
            self.distance_type = distance_type
            self.file_format.create_new_file(dimension, distance_type)
    
    def _load_from_file(self) -> None:
        """Load vectors and metadata from file."""
        try:
            # Read header
            self.dimension, self.distance_type, vector_count = self.file_format.read_header()
            
            # Read vectors and metadata
            self.id_mapping, self.vectors, self.metadata = self.file_format.read_vectors()
            
            # Set next position
            self.next_pos = vector_count
            
        except Exception as e:
            raise RuntimeError(f"Failed to load database: {str(e)}") from e
    
    def save(self) -> None:
        """Save vectors and metadata to file."""
        self.file_format.write_vectors(self.id_mapping, self.vectors, self.metadata)
    
    def insert(self, id_value: str, vector: List[float], metadata: Optional[Dict[str, Any]] = None) -> None:
        """Insert a vector with optional metadata.
        
        Args:
            id_value: Unique identifier for the vector
            vector: Vector data as a list of floats
            metadata: Associated metadata
            
        Raises:
            ValueError: If vector has incorrect dimension or ID already exists
        """
        # Validate inputs
        id_value = validate_id(id_value)
        vector = validate_vector(vector, self.dimension)
        metadata = validate_metadata(metadata)
        
        # Check if ID already exists
        if id_value in self.vectors:
            raise ValueError(f"ID already exists: {id_value}")
        
        # Store vector and metadata
        self.vectors[id_value] = vector
        self.metadata[id_value] = metadata
        self.id_mapping[id_value] = self.next_pos
        self.next_pos += 1
        
        # Save to file
        self.save()
    
    def get(self, id_value: str) -> Tuple[List[float], Optional[Dict[str, Any]]]:
        """Get a vector and its metadata by ID.
        
        Args:
            id_value: Vector ID
            
        Returns:
            tuple: (vector, metadata)
            
        Raises:
            KeyError: If ID doesn't exist
        """
        id_value = validate_id(id_value)
        
        if id_value not in self.vectors:
            raise KeyError(f"ID not found: {id_value}")
        
        return self.vectors[id_value], self.metadata[id_value]
    
    def delete(self, id_value: str) -> None:
        """Delete a vector by ID.
        
        Args:
            id_value: Vector ID
            
        Raises:
            KeyError: If ID doesn't exist
        """
        id_value = validate_id(id_value)
        
        if id_value not in self.vectors:
            raise KeyError(f"ID not found: {id_value}")
        
        # Remove vector and metadata
        del self.vectors[id_value]
        del self.metadata[id_value]
        del self.id_mapping[id_value]
        
        # Rebuild id_mapping to keep positions contiguous
        self.id_mapping = {id_val: i for i, id_val in enumerate(self.vectors.keys())}
        self.next_pos = len(self.vectors)
        
        # Save to file
        self.save()
    
    def get_all(self) -> Dict[str, Tuple[List[float], Optional[Dict[str, Any]]]]:
        """Get all vectors and metadata.
        
        Returns:
            dict: {id: (vector, metadata)}
        """
        return {id_val: (self.vectors[id_val], self.metadata[id_val]) for id_val in self.vectors}
    
    def close(self) -> None:
        """Close the vector store and save changes."""
        self.save()
import os
import struct
import numpy as np
from typing import Dict, List, Any, Tuple, Optional
import json

from vectorlitedb.utils.serialization import (
    MAGIC_BYTES, VERSION, HEADER_FORMAT,
    serialize_header, deserialize_header,
    serialize_vector, deserialize_vector,
    serialize_metadata, deserialize_metadata,
    serialize_id_mapping, deserialize_id_mapping
)

class FileFormat:
    """Handles reading and writing the VectorLiteDB file format."""
    
    def __init__(self, file_path: str):
        """Initialize the file format handler.
        
        Args:
            file_path: Path to the database file
        """
        self.file_path = file_path
        self.header_size = struct.calcsize(HEADER_FORMAT)
    
    def create_new_file(self, dimension: int, distance_type: str = 'cosine') -> None:
        """Create a new database file.
        
        Args:
            dimension: Vector dimension
            distance_type: Distance metric type ('l2', 'cosine', 'dot')
        """
        # Create parent directory if it doesn't exist
        os.makedirs(os.path.dirname(os.path.abspath(self.file_path)), exist_ok=True)
        
        # Create file
        with open(self.file_path, 'wb') as f:
            # Write header
            header = serialize_header(dimension, distance_type, 0)
            f.write(header)
            
            # Write empty ID mapping
            id_mapping = serialize_id_mapping({})
            id_mapping_size = len(id_mapping)
            f.write(struct.pack('I', id_mapping_size))
            f.write(id_mapping)
            
            # Write empty metadata index
            metadata_index = serialize_id_mapping({})
            f.write(struct.pack('I', len(metadata_index)))
            f.write(metadata_index)
    
    def read_header(self) -> Tuple[int, str, int]:
        """Read the file header.
        
        Returns:
            tuple: (dimension, distance_type, vector_count)
            
        Raises:
            FileNotFoundError: If file doesn't exist
            ValueError: If header is invalid
        """
        if not os.path.exists(self.file_path) or os.path.getsize(self.file_path) == 0:
            raise FileNotFoundError(f"File not found or empty: {self.file_path}")
            
        with open(self.file_path, 'rb') as f:
            header_data = f.read(self.header_size)
            if len(header_data) < self.header_size:
                raise ValueError(f"Invalid header size: expected {self.header_size}, got {len(header_data)}")
            return deserialize_header(header_data)
    
    def write_vectors(self, id_mapping: Dict[str, int], vectors: Dict[str, List[float]], 
                     metadata: Dict[str, Optional[Dict[str, Any]]]) -> None:
        """Write vectors, ID mapping, and metadata to file.
        
        Args:
            id_mapping: Dictionary mapping IDs to vector positions
            vectors: Dictionary mapping IDs to vectors
            metadata: Dictionary mapping IDs to metadata
        """
        # If file doesn't exist or is empty, get dimension from vectors
        if not os.path.exists(self.file_path) or os.path.getsize(self.file_path) == 0:
            if not vectors:
                dimension = 0
                distance_type = 'cosine'
            else:
                # Get dimension from first vector
                first_id = next(iter(vectors))
                dimension = len(vectors[first_id])
                distance_type = 'cosine'
        else:
            # Read existing header
            try:
                dimension, distance_type, _ = self.read_header()
            except (FileNotFoundError, ValueError):
                # If file is corrupted, get dimension from vectors
                if not vectors:
                    dimension = 0
                    distance_type = 'cosine'
                else:
                    first_id = next(iter(vectors))
                    dimension = len(vectors[first_id])
                    distance_type = 'cosine'
        
        vector_count = len(vectors)
        
        # Create parent directory if it doesn't exist
        os.makedirs(os.path.dirname(os.path.abspath(self.file_path)), exist_ok=True)
        
        with open(self.file_path, 'wb') as f:
            # Write header
            header = serialize_header(dimension, distance_type, vector_count)
            f.write(header)
            
            # Write ID mapping
            id_mapping_data = serialize_id_mapping(id_mapping)
            f.write(struct.pack('I', len(id_mapping_data)))
            f.write(id_mapping_data)
            
            # Prepare metadata index
            metadata_positions = {}
            metadata_blobs = []
            
            # Calculate positions for each metadata entry
            current_pos = 0
            for vector_id, meta in metadata.items():
                if meta:
                    metadata_positions[vector_id] = current_pos
                    meta_data = serialize_metadata(meta)
                    metadata_blobs.append(meta_data)
                    current_pos += len(meta_data) + 4  # Add 4 bytes for size
            
            # Write metadata index
            metadata_index = serialize_id_mapping(metadata_positions)
            f.write(struct.pack('I', len(metadata_index)))
            f.write(metadata_index)
            
            # Write vectors
            for vector_id in id_mapping:
                vector = vectors[vector_id]
                vector_data = serialize_vector(vector)
                f.write(vector_data)
            
            # Write metadata blobs
            for i, (vector_id, pos) in enumerate(metadata_positions.items()):
                meta = metadata[vector_id]
                meta_data = serialize_metadata(meta)
                f.write(struct.pack('I', len(meta_data)))
                f.write(meta_data)
    
    def read_vectors(self) -> Tuple[Dict[str, int], Dict[str, List[float]], Dict[str, Optional[Dict[str, Any]]]]:
        """Read vectors, ID mapping, and metadata from file.
        
        Returns:
            tuple: (id_mapping, vectors, metadata)
        """
        if not os.path.exists(self.file_path) or os.path.getsize(self.file_path) == 0:
            return {}, {}, {}
            
        with open(self.file_path, 'rb') as f:
            # Read header
            header_data = f.read(self.header_size)
            if len(header_data) < self.header_size:
                return {}, {}, {}
                
            dimension, distance_type, vector_count = deserialize_header(header_data)
            
            # If vector count is 0, return empty dicts
            if vector_count == 0:
                return {}, {}, {}
            
            # Read ID mapping
            try:
                id_mapping_size_data = f.read(4)
                if len(id_mapping_size_data) < 4:
                    return {}, {}, {}
                    
                id_mapping_size = struct.unpack('I', id_mapping_size_data)[0]
                id_mapping_data = f.read(id_mapping_size)
                if len(id_mapping_data) < id_mapping_size:
                    return {}, {}, {}
                    
                id_mapping = deserialize_id_mapping(id_mapping_data)
                
                # Read metadata index
                metadata_index_size_data = f.read(4)
                if len(metadata_index_size_data) < 4:
                    return {}, {}, {}
                    
                metadata_index_size = struct.unpack('I', metadata_index_size_data)[0]
                metadata_index_data = f.read(metadata_index_size)
                if len(metadata_index_data) < metadata_index_size:
                    return {}, {}, {}
                    
                metadata_positions = deserialize_id_mapping(metadata_index_data)
                
                # Read vectors
                vectors = {}
                for vector_id, pos in id_mapping.items():
                    # Each vector has dimension * 4 bytes (float32)
                    vector_data = f.read(dimension * 4)
                    if len(vector_data) < dimension * 4:
                        # If we can't read a complete vector, skip it
                        continue
                        
                    vector = deserialize_vector(vector_data, dimension)
                    vectors[vector_id] = vector
                
                # Read metadata
                metadata = {vector_id: None for vector_id in id_mapping}
                
                for vector_id, pos in metadata_positions.items():
                    if vector_id not in id_mapping:
                        # Skip metadata for vectors that don't exist
                        continue
                        
                    metadata_size_data = f.read(4)
                    if len(metadata_size_data) < 4:
                        break
                        
                    metadata_size = struct.unpack('I', metadata_size_data)[0]
                    metadata_data = f.read(metadata_size)
                    if len(metadata_data) < metadata_size:
                        break
                        
                    meta = deserialize_metadata(metadata_data)
                    metadata[vector_id] = meta
                
                return id_mapping, vectors, metadata
            except Exception as e:
                # If anything goes wrong during reading, return empty data
                return {}, {}, {}
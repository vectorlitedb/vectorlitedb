import json
import struct
import numpy as np
from typing import Dict, List, Any, Tuple, Union, Optional

# Constants for file format
MAGIC_BYTES = b'VLDB'
VERSION = 1
HEADER_FORMAT = '4sIBII'  # Magic bytes, Version, Distance metric, Dimension, Vector count

# Distance metric type mapping
DISTANCE_TYPES = {
    'l2': 0,
    'cosine': 1,
    'dot': 2
}

def serialize_vector(vector: List[float]) -> bytes:
    """Serialize a vector to bytes.
    
    Args:
        vector: Vector as list of floats
        
    Returns:
        bytes: Serialized vector
    """
    return struct.pack(f"{len(vector)}f", *vector)

def deserialize_vector(data: bytes, dimension: int) -> List[float]:
    """Deserialize a vector from bytes.
    
    Args:
        data: Serialized vector
        dimension: Vector dimension
        
    Returns:
        list: Deserialized vector
    """
    return list(struct.unpack(f"{dimension}f", data))

def serialize_metadata(metadata: Optional[Dict[str, Any]]) -> bytes:
    """Serialize metadata to bytes.
    
    Args:
        metadata: Metadata dictionary or None
        
    Returns:
        bytes: Serialized metadata
    """
    if metadata is None:
        return b''
    
    json_str = json.dumps(metadata)
    return json_str.encode('utf-8')

def deserialize_metadata(data: bytes) -> Optional[Dict[str, Any]]:
    """Deserialize metadata from bytes.
    
    Args:
        data: Serialized metadata
        
    Returns:
        dict or None: Deserialized metadata
    """
    if not data:
        return None
    
    return json.loads(data.decode('utf-8'))

def serialize_header(dimension: int, distance_type: str, vector_count: int) -> bytes:
    """Serialize file header.
    
    Args:
        dimension: Vector dimension
        distance_type: Distance metric type ('l2', 'cosine', 'dot')
        vector_count: Number of vectors
        
    Returns:
        bytes: Serialized header
    """
    distance_code = DISTANCE_TYPES.get(distance_type, 0)
    header = struct.pack(HEADER_FORMAT, MAGIC_BYTES, VERSION, distance_code, dimension, vector_count)
    return header

def deserialize_header(data: bytes) -> Tuple[int, str, int]:
    """Deserialize file header.
    
    Args:
        data: Serialized header
        
    Returns:
        tuple: (dimension, distance_type, vector_count)
        
    Raises:
        ValueError: If header is invalid
    """
    magic, version, distance_code, dimension, vector_count = struct.unpack(HEADER_FORMAT, data)
    
    if magic != MAGIC_BYTES:
        raise ValueError(f"Invalid file format: magic bytes mismatch")
    
    if version != VERSION:
        raise ValueError(f"Unsupported file version: {version}")
    
    distance_type = next((k for k, v in DISTANCE_TYPES.items() if v == distance_code), 'l2')
    
    return dimension, distance_type, vector_count

def serialize_id_mapping(id_mapping: Dict[str, int]) -> bytes:
    """Serialize ID to vector position mapping.
    
    Args:
        id_mapping: Dictionary mapping IDs to vector positions
        
    Returns:
        bytes: Serialized mapping
    """
    return json.dumps(id_mapping).encode('utf-8')

def deserialize_id_mapping(data: bytes) -> Dict[str, int]:
    """Deserialize ID to vector position mapping.
    
    Args:
        data: Serialized mapping
        
    Returns:
        dict: Deserialized mapping
    """
    return json.loads(data.decode('utf-8'))
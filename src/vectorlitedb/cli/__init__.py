"""Command-line interface for VectorLiteDB."""
import argparse
import json
import sys
from typing import List, Dict, Any, Optional

from vectorlitedb import VectorLiteDB

def parse_vector(vector_str: str) -> List[float]:
    """Parse a vector from string.
    
    Args:
        vector_str: Vector as string, e.g. "[1.0, 2.0, 3.0]"
        
    Returns:
        list: Vector as list of floats
    """
    try:
        # Try to parse as JSON
        vector = json.loads(vector_str)
        if not isinstance(vector, list):
            raise ValueError("Vector must be a list")
        return [float(x) for x in vector]
    except json.JSONDecodeError:
        # Try to parse as comma-separated values
        try:
            return [float(x.strip()) for x in vector_str.strip('[]').split(',')]
        except ValueError:
            raise ValueError(f"Could not parse vector: {vector_str}")

def parse_metadata(metadata_str: Optional[str]) -> Optional[Dict[str, Any]]:
    """Parse metadata from string.
    
    Args:
        metadata_str: Metadata as JSON string, or None
        
    Returns:
        dict or None: Parsed metadata
    """
    if not metadata_str:
        return None
    
    try:
        metadata = json.loads(metadata_str)
        if not isinstance(metadata, dict):
            raise ValueError("Metadata must be a JSON object")
        return metadata
    except json.JSONDecodeError:
        raise ValueError(f"Could not parse metadata: {metadata_str}")

def create_command(args):
    """Create a new database.
    
    Args:
        args: Command-line arguments
    """
    try:
        db = VectorLiteDB(args.db_path, dimension=args.dimension, distance_metric=args.distance)
        db.close()
        print(f"Created database at {args.db_path} with dimension {args.dimension}")
    except Exception as e:
        print(f"Error: {str(e)}", file=sys.stderr)
        sys.exit(1)

def insert_command(args):
    """Insert a vector into the database.
    
    Args:
        args: Command-line arguments
    """
    try:
        vector = parse_vector(args.vector)
        metadata = parse_metadata(args.metadata)
        
        db = VectorLiteDB(args.db_path)
        db.insert(id=args.id, vector=vector, metadata=metadata)
        db.close()
        
        print(f"Inserted vector with ID {args.id}")
    except Exception as e:
        print(f"Error: {str(e)}", file=sys.stderr)
        sys.exit(1)

def search_command(args):
    """Search for similar vectors.
    
    Args:
        args: Command-line arguments
    """
    try:
        query = parse_vector(args.query)
        
        db = VectorLiteDB(args.db_path)
        results = db.search(query=query, top_k=args.top_k)
        db.close()
        
        print(f"Found {len(results)} results:")
        for i, result in enumerate(results):
            print(f"  {i+1}. ID: {result.id}, Similarity: {result.similarity:.4f}")
            if args.show_metadata and result.metadata:
                print(f"     Metadata: {json.dumps(result.metadata)}")
    except Exception as e:
        print(f"Error: {str(e)}", file=sys.stderr)
        sys.exit(1)

def delete_command(args):
    """Delete a vector from the database.
    
    Args:
        args: Command-line arguments
    """
    try:
        db = VectorLiteDB(args.db_path)
        db.delete(id=args.id)
        db.close()
        
        print(f"Deleted vector with ID {args.id}")
    except Exception as e:
        print(f"Error: {str(e)}", file=sys.stderr)
        sys.exit(1)

def info_command(args):
    """Show information about the database.
    
    Args:
        args: Command-line arguments
    """
    try:
        db = VectorLiteDB(args.db_path)
        vector_count = len(db.store.vectors)
        dimension = db.store.dimension
        distance_type = db.store.distance_type
        db.close()
        
        print(f"Database: {args.db_path}")
        print(f"Vectors: {vector_count}")
        print(f"Dimension: {dimension}")
        print(f"Distance metric: {distance_type}")
    except Exception as e:
        print(f"Error: {str(e)}", file=sys.stderr)
        sys.exit(1)

def main():
    """Main entry point for the CLI."""
    parser = argparse.ArgumentParser(description="VectorLiteDB - SQLite for embeddings")
    subparsers = parser.add_subparsers(dest="command", help="Command to execute")
    
    # Create command
    create_parser = subparsers.add_parser("create", help="Create a new database")
    create_parser.add_argument("db_path", help="Path to the database file")
    create_parser.add_argument("--dimension", type=int, required=True, help="Vector dimension")
    create_parser.add_argument("--distance", choices=["l2", "cosine", "dot"], default="cosine", 
                               help="Distance metric (default: cosine)")
    create_parser.set_defaults(func=create_command)
    
    # Insert command
    insert_parser = subparsers.add_parser("insert", help="Insert a vector")
    insert_parser.add_argument("db_path", help="Path to the database file")
    insert_parser.add_argument("--id", required=True, help="Vector ID")
    insert_parser.add_argument("--vector", required=True, help="Vector as JSON array or comma-separated values")
    insert_parser.add_argument("--metadata", help="Metadata as JSON object")
    insert_parser.set_defaults(func=insert_command)
    
    # Search command
    search_parser = subparsers.add_parser("search", help="Search for similar vectors")
    search_parser.add_argument("db_path", help="Path to the database file")
    search_parser.add_argument("--query", required=True, help="Query vector as JSON array or comma-separated values")
    search_parser.add_argument("--top-k", type=int, default=5, help="Number of results to return (default: 5)")
    search_parser.add_argument("--show-metadata", action="store_true", help="Show metadata in results")
    search_parser.set_defaults(func=search_command)
    
    # Delete command
    delete_parser = subparsers.add_parser("delete", help="Delete a vector")
    delete_parser.add_argument("db_path", help="Path to the database file")
    delete_parser.add_argument("--id", required=True, help="Vector ID")
    delete_parser.set_defaults(func=delete_command)
    
    # Info command
    info_parser = subparsers.add_parser("info", help="Show database information")
    info_parser.add_argument("db_path", help="Path to the database file")
    info_parser.set_defaults(func=info_command)
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        sys.exit(1)
    
    args.func(args)

if __name__ == "__main__":
    main()

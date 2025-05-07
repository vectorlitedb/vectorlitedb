"""
Test script to verify the basic functionality of VectorLiteDB.
"""

import os
import sys
import tempfile

# Add the source directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

# Import VectorLiteDB
from vectorlitedb import VectorLiteDB

# Create a temporary file
temp_file = tempfile.NamedTemporaryFile(delete=False)
temp_file.close()

try:
    # Create a new database
    print("Creating database...")
    db = VectorLiteDB(temp_file.name, dimension=3)
    
    # Insert vectors
    print("Inserting vectors...")
    db.insert(id="vec1", vector=[1.0, 2.0, 3.0])
    db.insert(id="vec2", vector=[4.0, 5.0, 6.0])
    db.insert(id="vec3", vector=[7.0, 8.0, 9.0], metadata={"name": "Test Vector"})
    
    # Search for similar vectors
    print("Searching...")
    results = db.search(query=[1.0, 2.0, 3.0], top_k=3)
    
    # Display results
    print(f"Found {len(results)} results:")
    for i, result in enumerate(results):
        print(f"  {i+1}. ID: {result.id}, Similarity: {result.similarity:.4f}")
        if result.metadata:
            print(f"     Metadata: {result.metadata}")
    
    # Close the database
    db.close()
    
    # Test reopening the database
    print("\nReopening database...")
    db = VectorLiteDB(temp_file.name)
    
    # Verify vectors are still there
    results = db.search(query=[1.0, 2.0, 3.0], top_k=3)
    print(f"Found {len(results)} results after reopening:")
    for i, result in enumerate(results):
        print(f"  {i+1}. ID: {result.id}, Similarity: {result.similarity:.4f}")
    
    # Delete a vector
    print("\nDeleting vector vec1...")
    db.delete(id="vec1")
    
    # Verify deletion
    results = db.search(query=[1.0, 2.0, 3.0], top_k=3)
    print(f"Found {len(results)} results after deletion:")
    for i, result in enumerate(results):
        print(f"  {i+1}. ID: {result.id}, Similarity: {result.similarity:.4f}")
    
    db.close()
    print("\nAll tests passed!")
    
except Exception as e:
    print(f"Error: {str(e)}")
    raise

finally:
    # Clean up
    os.unlink(temp_file.name)
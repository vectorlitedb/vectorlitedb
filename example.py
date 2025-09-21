#!/usr/bin/env python3
"""
Simple VectorLiteDB Example
"""

import numpy as np
from vectorlitedb import VectorLiteDB

# Create or open a database
print("Creating database...")
db = VectorLiteDB("example.db", dimension=5)

# Insert some example vectors
print("Inserting vectors...")
db.insert(id="vec1", vector=[1.0, 0.0, 0.0, 0.0, 0.0], metadata={"label": "first"})
db.insert(id="vec2", vector=[0.0, 1.0, 0.0, 0.0, 0.0], metadata={"label": "second"})
db.insert(id="vec3", vector=[0.8, 0.2, 0.0, 0.0, 0.0], metadata={"label": "similar_to_first"})

print(f"Database now contains {len(db)} vectors")

# Search for similar vectors
print("\nSearching for vectors similar to [1.0, 0.1, 0.0, 0.0, 0.0]...")
results = db.search(query=[1.0, 0.1, 0.0, 0.0, 0.0], top_k=2)

print("Results:")
for result in results:
    print(f"  ID: {result['id']}")
    print(f"  Similarity: {result['similarity']:.3f}")
    print(f"  Metadata: {result['metadata']}")
    print()

# Close the database
db.close()
print("Database saved and closed!")
print(f"All data is stored in: example.db")
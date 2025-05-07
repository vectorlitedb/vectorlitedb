"""
Basic usage example for VectorLiteDB
"""

from vectorlitedb import VectorLiteDB

# Create a new database with vectors of dimension 3
db = VectorLiteDB("example.db", dimension=3)

# Insert vectors with IDs and optional metadata
db.insert(id="vec1", vector=[1.0, 2.0, 3.0])
db.insert(id="vec2", vector=[4.0, 5.0, 6.0])
db.insert(
    id="vec3",
    vector=[7.0, 8.0, 9.0],
    metadata={"name": "Sample", "tags": ["example", "test"]}
)

# Search for similar vectors
query = [1.0, 2.0, 3.1]  # Slightly different from vec1
results = db.search(query=query, top_k=2)

print("Search results:")
for result in results:
    print(f"  ID: {result.id}")
    print(f"  Similarity: {result.similarity:.4f}")
    print(f"  Metadata: {result.metadata}")
    print()

# Filter results by metadata
filtered_results = db.search(
    query=[5.0, 6.0, 7.0],
    top_k=3,
    filter=lambda meta: meta and "test" in meta.get("tags", [])
)

print("Filtered search results:")
for result in filtered_results:
    print(f"  ID: {result.id}")
    print(f"  Similarity: {result.similarity:.4f}")
    print(f"  Metadata: {result.metadata}")
    print()

# Delete a vector
db.delete(id="vec1")

# Verify it's deleted
results = db.search(query=[1.0, 2.0, 3.0], top_k=3)
print("Results after deletion:")
for result in results:
    print(f"  ID: {result.id}")
    print(f"  Similarity: {result.similarity:.4f}")
    print()

# Close the database (also happens automatically when the program ends)
db.close()

print("Database saved to example.db")
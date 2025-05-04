# Quick Start

## Installation

```bash
pip install vectorlitedb
```

## Basic Usage

```python
from vectorlitedb import VectorLiteDB

# Create or open a database
db = VectorLiteDB("my_vectors.db")

# Insert vectors with optional metadata
db.insert(id="doc1", vector=[0.1, 0.2, 0.3, 0.4, 0.5])
db.insert(id="doc2", 
          vector=[0.2, 0.3, 0.4, 0.5, 0.6], 
          metadata={"title": "Example", "tags": ["sample"]})

# Search for similar vectors
results = db.search(query=[0.15, 0.22, 0.31, 0.42, 0.51], top_k=5)

# Results contain IDs and similarity scores
for result in results:
    print(f"ID: {result.id}, Similarity: {result.similarity}")
    
# Filter search results by metadata
results = db.search(
    query=[0.2, 0.3, 0.4, 0.5, 0.6],
    filter=lambda meta: "sample" in meta["tags"],
    top_k=3
)

# Delete vectors
db.delete(id="doc1")

# Close the database (optional, happens automatically)
db.close()
```

# VectorLiteDB

[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
![Status](https://img.shields.io/badge/Status-Alpha-orange)
[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/downloads/)

> **SQLite for Embeddings** — A simple, embedded vector database that stores everything in a single file.

No server. No setup. Just vectors in a file.

## Why?

Every vector database is either a cloud service (Pinecone), needs a server (Chroma), or doesn't persist (FAISS). 

Sometimes you just want to store embeddings in a file and search them. Like SQLite does for relational data.

## Quick Start

```bash
pip install vectorlitedb
```

```python
from vectorlitedb import VectorLiteDB

# Create a database (just a file)
db = VectorLiteDB("my_vectors.db", dimension=384)

# Store vectors with metadata
db.insert(id="doc1", vector=embedding, metadata={"text": "Hello world"})
db.insert(id="doc2", vector=embedding2, metadata={"text": "Goodbye world"})

# Search (returns most similar vectors)
results = db.search(query=query_embedding, top_k=5)

# Results format
for result in results:
    print(f"ID: {result['id']}")
    print(f"Similarity: {result['similarity']}")
    print(f"Metadata: {result['metadata']}")
```

Your vectors are saved in `my_vectors.db`.

## What This Does

- **Stores** vectors (embeddings) with metadata in a single file
- **Searches** for similar vectors using cosine/L2/dot distance  
- **Persists** everything to disk automatically
- **Filters** results by metadata
- **Works offline** - no internet required

## What This Doesn't Do (Yet)

- ❌ Generate embeddings (use OpenAI, etc.)
- ❌ Scale beyond ~100K vectors (uses brute force search)
- ❌ Handle concurrent writes
- ❌ Optimize for speed

## Use Cases

- **Local RAG**: Store document embeddings for offline search
- **Personal AI**: Give your assistant persistent memory
- **Prototyping**: Test vector search ideas without infrastructure
- **Edge devices**: Semantic search on Raspberry Pi
- **Privacy**: Keep embeddings on your device

## API Reference

### Create/Open Database
```python
db = VectorLiteDB("path/to/file.db", dimension=384, distance_metric="cosine")
# distance_metric: "cosine" (default), "l2", or "dot"
```

### Insert Vectors
```python
db.insert(id="unique_id", vector=[0.1, 0.2, ...], metadata={"key": "value"})
```

### Search Vectors
```python
results = db.search(
    query=[0.1, 0.2, ...], 
    top_k=5,
    filter=lambda meta: meta.get("type") == "document"  # optional
)
```

### Get/Delete Vectors
```python
vector, metadata = db.get("unique_id")
db.delete("unique_id")
```

### Database Info
```python
print(len(db))  # number of vectors
print(repr(db))  # database summary
```

## Current Status

This is v0.1.0 - a working alpha focused on simplicity over performance.

**What works:**
- ✅ Basic CRUD operations  
- ✅ File persistence
- ✅ Metadata filtering
- ✅ ~100ms search for 10K vectors

**Planned:**
- 🚧 Better search algorithms (IVF, HNSW)
- 🚧 Memory-mapped files for large datasets
- 🚧 Language bindings (JavaScript, Rust, Go)

## Contributing

The codebase is intentionally small (~200 lines) and readable. 

Perfect for:

- First-time contributors
- Algorithm implementations  
- Performance optimizations
- Documentation improvements

See issues labeled [`good first issue`](https://github.com/vectorlitedb/vectorlitedb/labels/good%20first%20issue).

## Inspiration

- **SQLite**: Single-file, serverless, embedded
- **FAISS**: Fast similarity search

VectorLiteDB combines the best parts: embedded like SQLite, simple like a Python dict, persistent like a real database.

## License

VectorLiteDB is licensed under the [Apache 2.0 License](LICENSE).
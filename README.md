# VectorLiteDB: SQLite for Embeddings

[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
![Status](https://img.shields.io/badge/Status-Alpha-orange)
[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/downloads/)

Fast, private, embedded vector database for AI-first applications — no servers, no infra, just a single file.

## Overview

VectorLiteDB is an embedded, lightweight vector database designed for AI applications where cloud infrastructure is overkill. Think of it as **SQLite for embeddings** — developers can instantly embed fast vector search inside mobile, edge, or desktop apps with a single `.db` file.

Perfect for:
- 📱 Local RAG applications
- 🤖 Personal AI agents needing memory
- 📊 Offline semantic search
- 🔒 Privacy-first AI tools
- 🖥️ Edge and IoT applications

## Key Features

- 🚀 **Embedded Design**: No server to run, just import the library and open a .db file
- 💾 **Single-file Storage**: All vectors and metadata in one portable file
- 🔍 **Fast Similarity Search**: Optimized for 10K to 1M vectors
- 🔌 **Works Offline**: No cloud or internet dependency
- 🛠️ **Developer-friendly API**: Simple interface with minimal boilerplate
- 🔋 **Edge-optimized**: Runs efficiently on resource-constrained devices
- 🔒 **Privacy-first**: Data never leaves the device

## Quick Start

```bash
# Install via pip
pip install vectorlitedb
```

```python
# Basic usage
from vectorlitedb import VectorLiteDB

# Create or open a database
db = VectorLiteDB("my_knowledge.db")

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

## Why VectorLiteDB?

| Feature | VectorLiteDB | sqlite-vec | ChromaDB | FAISS | PGVector |
|---------|--------------|------------|----------|-------|----------|
| Deployment | Embedded | SQLite Extension | Python server | Library only | Postgres server |
| ANN | ✅ | ❌ | ❌ | ✅ | ❌ |
| Storage | ✅ Single file | ✅ SQLite | ✅ Server disk | ❌ Memory only | ✅ Postgres |
| CRUD Support | ✅ | Basic | Basic | ❌ | ✅ |
| Offline/Private | ✅ Native | ✅ | ⚠️ Hard | ❌ | ⚠️ Hard |
| DX/UX | SDK-first | SQL-first | OK | Researchy | SQL |

## Performance

- **Memory**: ~200MB RAM for 100K 1536-dimension vectors
- **Search Speed**: <100ms for top-5 results on 100K vectors
- **Storage**: Approximately 400 bytes per vector + metadata
- **Load Time**: < 1 second for 100K vectors
- **Edge Performance**: Usable on Raspberry Pi and similar devices

## Installation

```bash
pip install vectorlitedb
```

## Documentation

For comprehensive documentation, visit [docs.vectorlite.tech](https://docs.vectorlite.tech).

## Use Cases

### Local RAG Applications

```python
from vectorlitedb import VectorLiteDB

# Create database for document embeddings
db = VectorLiteDB("documents.db")

# Add document embeddings
for doc_id, embedding, content in my_documents:
    db.insert(
        id=doc_id,
        vector=embedding,
        metadata={"content": content, "source": "local"}
    )

# Query with user question embedding
results = db.search(query=question_embedding, top_k=5)

# Retrieve relevant contexts
contexts = [db.get_metadata(result.id)["content"] for result in results]

# Pass contexts to your LLM
answer = my_llm.generate(question, contexts)
```

### Edge AI Applications

```python
from vectorlitedb import VectorLiteDB

# Create database for image feature vectors
db = VectorLiteDB("image_features.db")

# Add image features
for image_path, features in process_images():
    db.insert(
        id=image_path,
        vector=features,
        metadata={"path": image_path}
    )
    
# Find similar images
results = db.search(query=query_image_features, top_k=10)
```

## Roadmap

- [x] Initial core architecture
- [x] Basic vector operations (insert, search)
- [x] Persistence layer with single file storage
- [ ] Inverted File Index (IVF) implementation
- [ ] HNSW graph-based indexing
- [ ] Memory optimizations for edge devices
- [ ] Advanced metadata filtering
- [ ] Additional programming language bindings

## Contributing

We welcome contributions of all kinds! See our [Contributing Guide](CONTRIBUTING.md) to get started.

## Community

- Ask questions on [GitHub Discussions](https://github.com/vectorlitedb/vectorlitedb/discussions)
- Follow us on [Twitter/X](https://twitter.com/vectorlitedb)

## License

VectorLiteDB is licensed under the [Apache License 2.0](LICENSE).

---

<p align="center">Made with ❤️ for the AI development community</p>
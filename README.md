# VectorLiteDB

[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
![Status](https://img.shields.io/badge/Status-Alpha-orange)
[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![Website](https://img.shields.io/badge/Website-vectorlite.tech-blue)](https://vectorlite.tech)
[![Docs](https://img.shields.io/badge/Docs-docs-green)](https://docs.vectorlite.tech)

Fast, private, embedded vector database for AI-first applications — no servers, no infra, just a single file.

## What is VectorLiteDB?

VectorLiteDB is an embedded, lightweight vector database designed for AI applications where cloud infrastructure is overkill. Think of it as **SQLite for embeddings** — developers can instantly embed fast vector search inside mobile, edge, or desktop apps with a single `.db` file.

Perfect for:
- Local AI applications with no server requirements
- Privacy-focused solutions where data stays on device
- Fast vector search optimized for 10K-1M vectors

## Installation

```bash
pip install vectorlitedb
```

## Quick Start

```python
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
```

## Documentation

For comprehensive documentation, tutorials, and benchmarks, visit [docs.vectorlite.tech](https://docs.vectorlite.tech).

## Roadmap

Check out our [Roadmap](https://github.com/orgs/vectorlitedb/projects/1) to see what features are planned for future releases.

## Want to Contribute?

VectorLiteDB is in early alpha stage and actively seeking contributors. Whether you're interested in improving performance, adding features, or enhancing documentation, your help is welcome. 

See our [Contributing Guide](CONTRIBUTING.md) for details on how to get started.

Join the discussion on [GitHub Discussions](https://github.com/vectorlitedb/vectorlitedb/discussions)

## License

VectorLiteDB is licensed under the [Apache License 2.0](LICENSE).

---

<p align="center">Made for the AI development community</p>
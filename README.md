# VectorLiteDB

[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
![Status](https://img.shields.io/badge/Status-Alpha-orange)
[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![Website](https://img.shields.io/badge/Website-vectorlite.tech-blue)](https://vectorlite.tech)
[![Docs](https://img.shields.io/badge/Docs-docs-green)](https://docs.vectorlite.tech)
[![Discord](https://img.shields.io/badge/Discord-Join-blue)](https://discord.gg/vectorlite)

> **SQLite for vector search** — A fast, embeddable vector database for on-device AI applications. No servers. No setup. Just a single `.db` file for intelligent memory.

## 🚀 Features

- **Embedded Design**: Import the library and open a `.db` file — no server setup required
- **Single-file Storage**: All vectors and metadata in one portable file
- **Fast Similarity Search**: Optimized for 10K to 1M vectors with multiple ANN algorithms
- **Works Offline**: No cloud or internet dependency
- **Edge-optimized**: Runs efficiently on resource-constrained devices
- **Privacy-first**: Data never leaves the device
- **Developer-friendly API**: Simple interface with minimal boilerplate

## 📦 Installation

```bash
pip install vectorlitedb
```

## 🎯 Use Cases

VectorLiteDB is perfect for:

- **Local RAG Systems**: Search PDFs and documents with local embedding + search
- **Personal AI Agents**: Vector memory that doesn't leak to cloud services
- **Semantic Cache**: Cache LLM answers with vector-based lookup
- **Embedded Copilots**: Add context-aware assistance to desktop/mobile apps
- **Edge AI Applications**: Industrial robots, retail kiosks, drones, field agents
- **Privacy-sensitive Products**: Healthcare, legal, and finance tools

## 🚀 Quick Start

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
```

## 📚 Documentation

Explore our comprehensive documentation:

- [Getting Started Guide](https://docs.vectorlite.tech/getting-started)
- [API Reference](https://docs.vectorlite.tech/api)
- [Performance Tuning](https://docs.vectorlite.tech/performance)
- [Integration Examples](https://docs.vectorlite.tech/examples)
- [Architecture Overview](https://docs.vectorlite.tech/architecture)

## 🔧 Technical Details

### Supported Algorithms

- **Flat Search**: Exact but slower, good for small collections
- **IVF (Inverted File Index)**: Balanced speed/accuracy for medium collections
- **HNSW (Hierarchical Navigable Small World)**: Fast approximate search for large collections

### Performance Characteristics

- Memory: ~200MB RAM for 100K 1536-dimension vectors
- Search Speed: <100ms for top-5 results on 100K vectors
- Storage: ~400 bytes per vector + metadata
- Load Time: <1 second for 100K vectors

## 🤝 Contributing

VectorLiteDB is in early alpha, and we welcome contributions! Here's how you can help:

1. Review our [Contributing Guide](CONTRIBUTING.md)
2. Check out the [Project Roadmap](https://github.com/orgs/vectorlitedb/projects/1)
3. Join our [Discord Community](https://discord.gg/vectorlite)
4. Participate in [GitHub Discussions](https://github.com/vectorlitedb/vectorlitedb/discussions)

### Development Setup

```bash
# Clone the repository
git clone https://github.com/vectorlitedb/vectorlitedb.git
cd vectorlitedb

# Create and activate virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install development dependencies
pip install -e ".[dev]"

# Run tests
pytest
```

## 📄 License

VectorLiteDB is licensed under the [Apache 2.0 License](LICENSE).

## 🌟 Why VectorLiteDB?

- **No Infrastructure Required**: Unlike cloud-first solutions (Pinecone, Weaviate)
- **Full CRUD Support**: Unlike search-only libraries (FAISS, Annoy)
- **Embedded Design**: Unlike server-first databases (Chroma, PGVector)
- **Privacy by Default**: Data stays on your device
- **Edge-optimized**: Runs efficiently on resource-constrained devices

## 📞 Support & Community

- [GitHub Issues](https://github.com/vectorlitedb/vectorlitedb/issues)
- [Discord Community](https://discord.gg/vectorlite)
- [GitHub Discussions](https://github.com/vectorlitedb/vectorlitedb/discussions)
- [Documentation](https://docs.vectorlite.tech)

---

<p align="center">
Made with ❤️ by the VectorLiteDB Team
</p>


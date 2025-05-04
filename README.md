# VectorLiteDB

[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
![Status](https://img.shields.io/badge/Status-Alpha-orange)
[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![Website](https://img.shields.io/badge/Website-vectorlite.tech-blue)](https://vectorlite.tech)
[![Docs](https://img.shields.io/badge/Docs-docs-green)](https://docs.vectorlite.tech)

**VectorLiteDB is SQLite for vector search — a fast, embeddable vector database for on-device AI.**  
No servers. No setup. Just a single `.db` file for intelligent memory.

Built for developers who need to:
- Run private, local vector search with zero infrastructure  
- Ship intelligent memory into desktop, mobile, or edge apps  
- Store and query 10K–1M vectors with fast similarity search  
- Power offline copilots, local RAG systems, and personal AI agents

---
## Install

```bash
pip install vectorlitedb
```

## Quick Example

```python
from vectorlitedb import VectorLiteDB

db = VectorLiteDB("my_vectors.db")

db.insert(id="note1", vector=[0.1, 0.2, 0.3])
db.insert(id="note2", vector=[0.2, 0.3, 0.4], metadata={"title": "Sample"})

results = db.search(query=[0.12, 0.22, 0.32], top_k=3)

for r in results:
    print(r.id, r.similarity)
```

## Documentation

See full API docs, examples, and benchmarks at:  
**https://docs.vectorlite.tech**

## Contributing

We’re in early alpha and open to contributions.  
To get started, read [CONTRIBUTING.md](CONTRIBUTING.md) or join our [GitHub Discussions](https://github.com/vectorlitedb/vectorlitedb/discussions).

## License

Apache License 2.0 — see [LICENSE](LICENSE)

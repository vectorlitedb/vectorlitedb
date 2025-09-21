"""
VectorLiteDB - SQLite for Embeddings

A simple, embedded vector database that stores everything in a single file.
No server required, no complex setup. Just import and use.
"""

__version__ = "0.1.0"

from .db import VectorLiteDB

__all__ = ["VectorLiteDB"]
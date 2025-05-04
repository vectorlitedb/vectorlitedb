# Example: Local RAG (Retrieval Augmented Generation) with VectorLiteDB

import os
import numpy as np
from vectorlitedb import VectorLiteDB

# This example shows how to implement a simple RAG system with VectorLiteDB
# In a real application, you would use a text embedding model and an LLM

# Mock embedding function (replace with real embedding model in production)
def get_embedding(text):
    # Mock function to generate random embeddings (for demonstration only)
    # In production, use a proper embedding model like SentenceTransformers
    np.random.seed(hash(text) % 2**32)
    return np.random.randn(384).astype(np.float32).tolist()

# Create a database
db_path = "documents.db"
db = VectorLiteDB(db_path)

# Example document corpus
documents = [
    {"id": "doc1", "title": "Vector Databases", "content": "Vector databases store and search vector embeddings efficiently."},
    {"id": "doc2", "title": "Embeddings", "content": "Embeddings represent text or images as high-dimensional vectors."},
    {"id": "doc3", "title": "Similarity Search", "content": "Similarity search finds vectors that are close to a query vector."},
    {"id": "doc4", "title": "Local Databases", "content": "Local databases store data on the same device as the application."}
]

# Insert documents into the database
for doc in documents:
    embedding = get_embedding(doc["content"])
    db.insert(
        id=doc["id"],
        vector=embedding,
        metadata={
            "title": doc["title"],
            "content": doc["content"]
        }
    )

# Example query
query = "How do vector databases work?"
query_embedding = get_embedding(query)

# Search for relevant documents
results = db.search(query=query_embedding, top_k=2)

# Display results
print(f"Query: {query}\n")
print("Relevant documents:")
for result in results:
    print(f"ID: {result.id}, Similarity: {result.similarity}")
    print(f"Title: {result.metadata['title']}")
    print(f"Content: {result.metadata['content']}")
    print()

# In a real RAG system, you would pass the retrieved documents to an LLM
# along with the original query to generate a response

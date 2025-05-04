# Example: Personal Assistant Memory with VectorLiteDB

import os
import numpy as np
from datetime import datetime
from vectorlitedb import VectorLiteDB

# This example shows how to implement a personal assistant's memory
# In a real application, you would use a text embedding model and an LLM

# Mock embedding function (replace with real embedding model in production)
def get_embedding(text):
    # Mock function to generate random embeddings (for demonstration only)
    # In production, use a proper embedding model like SentenceTransformers
    np.random.seed(hash(text) % 2**32)
    return np.random.randn(384).astype(np.float32).tolist()

# Create a memory database
db_path = "assistant_memory.db"
db = VectorLiteDB(db_path)

# Example memory entries
memories = [
    "User prefers vegetarian recipes",
    "User's birthday is May 15",
    "User is allergic to peanuts",
    "User's favorite color is blue",
    "User has a meeting every Monday at 10 AM",
    "User has three children named Alice, Bob, and Charlie"
]

# Store memories in the database
for i, memory in enumerate(memories):
    embedding = get_embedding(memory)
    db.insert(
        id=f"memory_{i}",
        vector=embedding,
        metadata={
            "text": memory,
            "timestamp": datetime.now().isoformat()
        }
    )

# Example query to the assistant
query = "What should I consider when planning dinner?"
query_embedding = get_embedding(query)

# Search for relevant memories
results = db.search(query=query_embedding, top_k=2)

# Display results
print(f"Query: {query}\n")
print("Relevant memories:")
for result in results:
    print(f"Memory: {result.metadata['text']}")
    print(f"Relevance: {result.similarity}")
    print()

# In a real assistant, you would pass these memories to an LLM
# along with the original query to generate a contextually informed response

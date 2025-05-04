# Benchmarks for search performance

import time
import numpy as np
from vectorlitedb import VectorLiteDB

def benchmark_search(vector_count=10000, dimension=1536, queries=100, database_path="benchmark.db"):
    """Benchmark search performance.
    
    Args:
        vector_count: Number of vectors to insert
        dimension: Vector dimension
        queries: Number of search queries to perform
        database_path: Path to database file
    
    Returns:
        dict: Results containing times and statistics
    """
    # Create test database
    db = VectorLiteDB(database_path)
    
    # Generate random vectors
    vectors = []
    for i in range(vector_count):
        vector = np.random.randn(dimension).astype(np.float32).tolist()
        vectors.append(vector)
        db.insert(id=f"vec_{i}", vector=vector)
    
    # Benchmark search
    search_times = []
    for i in range(queries):
        query = np.random.randn(dimension).astype(np.float32).tolist()
        
        start_time = time.time()
        results = db.search(query=query, top_k=10)
        end_time = time.time()
        
        search_times.append(end_time - start_time)
    
    # Calculate statistics
    avg_time = sum(search_times) / len(search_times)
    min_time = min(search_times)
    max_time = max(search_times)
    p95_time = sorted(search_times)[int(len(search_times) * 0.95)]
    
    # Return results
    return {
        "avg_search_time": avg_time,
        "min_search_time": min_time,
        "max_search_time": max_time,
        "p95_search_time": p95_time,
        "vector_count": vector_count,
        "dimension": dimension,
        "queries": queries
    }

def run_benchmarks():
    # Run benchmarks for different sizes
    small = benchmark_search(vector_count=1000, dimension=384, queries=50)
    medium = benchmark_search(vector_count=10000, dimension=768, queries=50)
    large = benchmark_search(vector_count=100000, dimension=1536, queries=20)
    
    # Print results
    print("\nSmall Dataset (1K vectors, 384-dim):")
    print(f"  Avg search time: {small['avg_search_time']*1000:.2f}ms")
    print(f"  P95 search time: {small['p95_search_time']*1000:.2f}ms")
    
    print("\nMedium Dataset (10K vectors, 768-dim):")
    print(f"  Avg search time: {medium['avg_search_time']*1000:.2f}ms")
    print(f"  P95 search time: {medium['p95_search_time']*1000:.2f}ms")
    
    print("\nLarge Dataset (100K vectors, 1536-dim):")
    print(f"  Avg search time: {large['avg_search_time']*1000:.2f}ms")
    print(f"  P95 search time: {large['p95_search_time']*1000:.2f}ms")

if __name__ == "__main__":
    run_benchmarks()

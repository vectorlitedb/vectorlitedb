# Benchmarks for insert performance

import time
import numpy as np
from vectorlitedb import VectorLiteDB

def benchmark_insert(vector_count=10000, dimension=1536, database_path="benchmark.db"):
    """Benchmark insert performance.
    
    Args:
        vector_count: Number of vectors to insert
        dimension: Vector dimension
        database_path: Path to database file
    
    Returns:
        dict: Results containing times and statistics
    """
    # Create test database
    db = VectorLiteDB(database_path)
    
    # Generate random vectors
    insert_times = []
    total_start = time.time()
    
    for i in range(vector_count):
        vector = np.random.randn(dimension).astype(np.float32).tolist()
        
        start_time = time.time()
        db.insert(id=f"vec_{i}", vector=vector)
        end_time = time.time()
        
        insert_times.append(end_time - start_time)
    
    total_end = time.time()
    total_time = total_end - total_start
    
    # Calculate statistics
    avg_time = sum(insert_times) / len(insert_times)
    min_time = min(insert_times)
    max_time = max(insert_times)
    p95_time = sorted(insert_times)[int(len(insert_times) * 0.95)]
    throughput = vector_count / total_time
    
    # Return results
    return {
        "avg_insert_time": avg_time,
        "min_insert_time": min_time,
        "max_insert_time": max_time,
        "p95_insert_time": p95_time,
        "total_time": total_time,
        "throughput": throughput,  # vectors per second
        "vector_count": vector_count,
        "dimension": dimension
    }

def run_benchmarks():
    # Run benchmarks for different sizes
    small = benchmark_insert(vector_count=1000, dimension=384)
    medium = benchmark_insert(vector_count=10000, dimension=768)
    
    # Print results
    print("\nSmall Dataset (1K vectors, 384-dim):")
    print(f"  Avg insert time: {small['avg_insert_time']*1000:.2f}ms")
    print(f"  Throughput: {small['throughput']:.2f} vectors/second")
    
    print("\nMedium Dataset (10K vectors, 768-dim):")
    print(f"  Avg insert time: {medium['avg_insert_time']*1000:.2f}ms")
    print(f"  Throughput: {medium['throughput']:.2f} vectors/second")

if __name__ == "__main__":
    run_benchmarks()

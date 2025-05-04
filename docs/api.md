# API Reference

## VectorLiteDB Class

The main interface for interacting with the database.

### Constructor

```python
VectorLiteDB(db_path, dimension=None, distance_metric="cosine")
```

- `db_path` (str): Path to the database file
- `dimension` (int, optional): Vector dimension (required for new databases)
- `distance_metric` (str, optional): Distance metric ("cosine", "l2", or "dot")

### Methods

#### insert

```python
insert(id, vector, metadata=None)
```

- `id` (str): Unique identifier for the vector
- `vector` (list): Vector data as a list of floats
- `metadata` (dict, optional): Associated metadata

#### search

```python
search(query, top_k=5, filter=None)
```

- `query` (list): Query vector
- `top_k` (int): Number of results to return
- `filter` (callable, optional): Function to filter results by metadata

Returns a list of results containing IDs and similarity scores.

#### delete

```python
delete(id)
```

- `id` (str): ID of the vector to delete

#### close

```python
close()
```

Close the database connection.

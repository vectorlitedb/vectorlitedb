"""
VectorLiteDB - A simple embedded vector database
"""

import json
import os
import struct
from json import JSONDecodeError
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

# ============== Core Database Class ==============


class VectorLiteDB:
    """
    A simple file-based vector database for embeddings.

    Like SQLite, but for vectors. Stores everything in a single file.
    """

    # Keep the WAL bounded without adding a public tuning surface yet.
    _CHECKPOINT_INTERVAL = 100

    def __init__(
        self,
        db_path: str,
        dimension: Optional[int] = None,
        distance_metric: str = "cosine",
    ):
        """
        Initialize or open a vector database.

        Args:
            db_path: Path to the database file
            dimension: Vector dimension (required for new databases)
            distance_metric: One of "cosine", "l2", or "dot"
        """
        self.db_path = db_path
        self._wal_path = self._get_wal_path()
        self._read_only = False
        self._dirty_ops = 0

        # Validate distance metric
        valid_metrics = {"cosine", "l2", "dot"}
        if distance_metric not in valid_metrics:
            raise ValueError(
                "Invalid distance_metric: "
                f"{distance_metric}. Must be one of {valid_metrics}"
            )
        self.distance_metric = distance_metric

        # In-memory storage
        self.vectors: Dict[str, List[float]] = {}
        self.metadata: Dict[str, Optional[Dict[str, Any]]] = {}

        # Load existing or create new
        if os.path.exists(db_path) and os.path.getsize(db_path) > 0:
            self._load_snapshot()
            if not os.access(db_path, os.W_OK):
                self._read_only = True
            self._replay_wal()
        else:
            if dimension is None:
                raise ValueError("Dimension required for new database")

            # Validate dimension type and value
            if not isinstance(dimension, int):
                raise TypeError(
                    f"Dimension must be an integer, got {type(dimension).__name__}"
                )
            if dimension < 0:
                raise ValueError(f"Dimension must be non-negative, got {dimension}")

            self.dimension = dimension
            self._write_snapshot()

    def insert(
        self, id: str, vector: List[float], metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Insert a vector with optional metadata.

        Args:
            id: Unique identifier
            vector: Embedding vector
            metadata: Optional metadata dictionary
        """
        if self._read_only:
            raise PermissionError("Cannot insert into read-only database")

        if id in self.vectors:
            raise ValueError(f"ID already exists: {id}")

        if len(vector) != self.dimension:
            raise ValueError(
                "Vector dimension mismatch: "
                f"expected {self.dimension}, got {len(vector)}"
            )

        self.vectors[id] = vector
        self.metadata[id] = metadata

        try:
            self._append_wal_record(
                {
                    "op": "insert",
                    "id": id,
                    "vector": vector,
                    "metadata": metadata,
                }
            )
        except BaseException:
            del self.vectors[id]
            del self.metadata[id]
            raise

        self._dirty_ops += 1
        self._maybe_checkpoint()

    def search(
        self,
        query: List[float],
        top_k: int = 5,
        filter: Optional[Callable[[Dict[str, Any]], bool]] = None,
    ) -> List[Dict[str, Any]]:
        """
        Search for similar vectors using brute force.

        Args:
            query: Query vector
            top_k: Number of results to return
            filter: Optional function to filter by metadata

        Returns:
            List of results with id, similarity, and metadata
        """
        if not self.vectors:
            return []

        distances = []
        for vec_id, vector in self.vectors.items():
            if filter and vec_id in self.metadata:
                meta = self.metadata[vec_id]
                if meta is None or not filter(meta):
                    continue

            distance = self._calculate_distance(query, vector)
            distances.append((vec_id, distance))

        distances.sort(key=lambda x: x[1])
        results = []

        for vec_id, distance in distances[:top_k]:
            if self.distance_metric in ["l2", "cosine"]:
                similarity = 1.0 / (1.0 + distance)
            else:
                similarity = -distance

            results.append(
                {
                    "id": vec_id,
                    "similarity": similarity,
                    "metadata": self.metadata.get(vec_id, {}),
                }
            )

        return results

    def delete(self, id: str) -> None:
        """
        Delete a vector by ID.

        Args:
            id: Vector ID to delete
        """
        if self._read_only:
            raise PermissionError("Cannot delete from read-only database")

        if id not in self.vectors:
            raise KeyError(f"ID not found: {id}")

        vector = self.vectors[id]
        metadata = self.metadata[id]

        del self.vectors[id]
        del self.metadata[id]

        try:
            self._append_wal_record({"op": "delete", "id": id})
        except BaseException:
            self.vectors[id] = vector
            self.metadata[id] = metadata
            raise

        self._dirty_ops += 1
        self._maybe_checkpoint()

    def get(self, id: str) -> Tuple[List[float], Optional[Dict[str, Any]]]:
        """
        Get a vector and its metadata by ID.

        Args:
            id: Vector ID

        Returns:
            Tuple of (vector, metadata)
        """
        if id not in self.vectors:
            raise KeyError(f"ID not found: {id}")

        return self.vectors[id], self.metadata[id]

    def close(self) -> None:
        """Checkpoint pending WAL changes and close the database."""
        if self._read_only:
            return

        if self._has_pending_wal():
            self._checkpoint()

    # ============== Internal Methods ==============

    def _calculate_distance(self, v1: List[float], v2: List[float]) -> float:
        """Calculate distance between two vectors."""
        v1_array = np.array(v1, dtype=np.float64)
        v2_array = np.array(v2, dtype=np.float64)

        if np.any(np.isnan(v1_array)) or np.any(np.isnan(v2_array)):
            return float("inf")

        if self.distance_metric == "l2":
            distance = float(np.linalg.norm(v1_array - v2_array))
            return distance if np.isfinite(distance) else float("inf")

        if self.distance_metric == "cosine":
            norm1 = np.linalg.norm(v1_array)
            norm2 = np.linalg.norm(v2_array)
            if norm1 == 0 or norm2 == 0:
                return 1.0

            if np.isinf(norm1) or np.isinf(norm2):
                return float("inf")

            similarity = np.dot(v1_array, v2_array) / (norm1 * norm2)
            similarity = np.clip(similarity, -1.0, 1.0)
            distance = float(1 - similarity)
            return distance if np.isfinite(distance) else float("inf")

        if self.distance_metric == "dot":
            dot_product = np.dot(v1_array, v2_array)
            return float(-dot_product) if np.isfinite(dot_product) else float("inf")

        raise ValueError(f"Unknown distance metric: {self.distance_metric}")

    def _get_wal_path(self) -> str:
        """Return the sidecar WAL path for this database."""
        return self.db_path + ".wal"

    def _ensure_db_directory(self) -> None:
        """Create the parent directory for the database files if needed."""
        os.makedirs(os.path.dirname(os.path.abspath(self.db_path)), exist_ok=True)

    def _append_wal_record(self, record: Dict[str, Any]) -> None:
        """Append a single durable WAL record as JSON Lines."""
        self._ensure_db_directory()

        with open(self._wal_path, "a", encoding="utf-8") as wal_file:
            wal_file.write(json.dumps(record))
            wal_file.write("\n")
            wal_file.flush()
            os.fsync(wal_file.fileno())

    def _apply_wal_record(self, record: Dict[str, Any]) -> None:
        """Apply a single WAL record to the in-memory state."""
        op = record.get("op")
        if op == "insert":
            self.vectors[record["id"]] = record["vector"]
            self.metadata[record["id"]] = record.get("metadata")
            return

        if op == "delete":
            self.vectors.pop(record["id"], None)
            self.metadata.pop(record["id"], None)
            return

        raise ValueError(f"Unknown WAL operation: {op}")

    def _replay_wal(self) -> None:
        """Replay the WAL into memory, ignoring only a truncated final record."""
        if not self._has_pending_wal():
            return

        with open(self._wal_path, "r", encoding="utf-8") as wal_file:
            lines = wal_file.readlines()

        for index, line in enumerate(lines):
            raw_line = line.strip()
            if not raw_line:
                continue

            try:
                record = json.loads(raw_line)
            except JSONDecodeError as exc:
                is_last_line = index == len(lines) - 1
                # A missing trailing newline likely means the process crashed mid-write.
                # Ignore only that incomplete tail so earlier committed records survive.
                if is_last_line and not line.endswith("\n"):
                    break
                raise ValueError("Invalid WAL format") from exc

            self._apply_wal_record(record)

    def _maybe_checkpoint(self) -> None:
        """Checkpoint after enough pending WAL operations have accumulated."""
        if self._dirty_ops >= self._CHECKPOINT_INTERVAL:
            self._checkpoint()

    def _has_pending_wal(self) -> bool:
        """Return True when a non-empty WAL file exists."""
        return os.path.exists(self._wal_path) and os.path.getsize(self._wal_path) > 0

    def _checkpoint(self) -> None:
        """Merge the in-memory state back into the main snapshot and clear the WAL."""
        if self._read_only:
            raise PermissionError("Cannot checkpoint read-only database")

        if not self._has_pending_wal():
            self._dirty_ops = 0
            return

        self._write_snapshot()
        # Clear the WAL only after the snapshot replace succeeds.
        os.remove(self._wal_path)
        self._dirty_ops = 0

    def _write_snapshot(self) -> None:
        """Save database to the main file atomically to prevent corruption."""
        self._ensure_db_directory()

        temp_path = self.db_path + ".tmp"
        try:
            with open(temp_path, "wb") as snapshot_file:
                header = {
                    "magic": "VLDB",
                    "version": 1,
                    "dimension": self.dimension,
                    "distance_metric": self.distance_metric,
                    "count": len(self.vectors),
                }
                header_json = json.dumps(header).encode("utf-8")
                snapshot_file.write(struct.pack("I", len(header_json)))
                snapshot_file.write(header_json)

                data = {"vectors": self.vectors, "metadata": self.metadata}
                data_json = json.dumps(data).encode("utf-8")
                snapshot_file.write(data_json)

                snapshot_file.flush()
                os.fsync(snapshot_file.fileno())

            os.replace(temp_path, self.db_path)
        except BaseException:
            if os.path.exists(temp_path):
                os.remove(temp_path)
            raise

    def _load_snapshot(self) -> None:
        """Load database from the main snapshot file."""
        with open(self.db_path, "rb") as snapshot_file:
            header_size = struct.unpack("I", snapshot_file.read(4))[0]
            header_json = snapshot_file.read(header_size)
            header = json.loads(header_json.decode("utf-8"))

            if header["magic"] != "VLDB":
                raise ValueError("Invalid file format")

            self.dimension = header["dimension"]
            self.distance_metric = header["distance_metric"]

            data_json = snapshot_file.read()
            data = json.loads(data_json.decode("utf-8"))

            self.vectors = data["vectors"]
            self.metadata = data["metadata"]

    # ============== Context Manager ==============

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    def __len__(self):
        return len(self.vectors)

    def __repr__(self):
        return (
            f"VectorLiteDB(path='{self.db_path}', vectors={len(self.vectors)}, "
            f"dim={self.dimension})"
        )

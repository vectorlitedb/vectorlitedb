# File Format Specification

The VectorLiteDB database file has a structured binary format:

## Header Block

- Magic bytes: "VLDB" (4 bytes)
- Version number (4 bytes)
- Distance metric type (1 byte)
- Vector dimension (4 bytes)
- Vector count (8 bytes)
- Timestamp (8 bytes)

## Index Block

- Index type (1 byte)
- Index parameters
- ID to vector position mapping

## Vector Data Block

- Contiguous array of fixed-size vectors
- Each vector stored as float32 values

## Metadata Block

- Serialized metadata objects
- References to vector IDs

# Tests for the main VectorLiteDB class

import unittest
import os
import tempfile
from vectorlitedb import VectorLiteDB

class TestVectorLiteDB(unittest.TestCase):
    def setUp(self):
        # Create a temporary file for the database
        self.temp_file = tempfile.NamedTemporaryFile(delete=False)
        self.temp_file.close()
        
        # Create a new database with dimension 3
        self.db = VectorLiteDB(self.temp_file.name, dimension=3)
        
    def test_insert(self):
        # Test vector insertion
        self.db.insert(id="test1", vector=[1.0, 2.0, 3.0])
        self.db.insert(id="test2", vector=[4.0, 5.0, 6.0])
        self.db.insert(id="test3", vector=[7.0, 8.0, 9.0], metadata={"name": "Test 3"})
        
        # Verify they're stored (indirectly through search)
        results = self.db.search(query=[1.0, 2.0, 3.0], top_k=3)
        self.assertEqual(len(results), 3)
        self.assertEqual(results[0].id, "test1")  # First result should be exact match
        
        # Test insert with wrong dimension
        with self.assertRaises(ValueError):
            self.db.insert(id="test4", vector=[1.0, 2.0, 3.0, 4.0])
        
        # Test insert with duplicate ID
        with self.assertRaises(ValueError):
            self.db.insert(id="test1", vector=[1.0, 2.0, 3.0])
        
    def test_search(self):
        # Insert test vectors
        self.db.insert(id="test1", vector=[1.0, 0.0, 0.0])
        self.db.insert(id="test2", vector=[0.0, 1.0, 0.0])
        self.db.insert(id="test3", vector=[0.0, 0.0, 1.0])
        
        # Test exact match
        results = self.db.search(query=[1.0, 0.0, 0.0], top_k=1)
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0].id, "test1")
        
        # Test with all vectors
        results = self.db.search(query=[1.0, 1.0, 1.0], top_k=3)
        self.assertEqual(len(results), 3)
        
        # Test with filter
        self.db.insert(id="test4", vector=[0.5, 0.5, 0.5], metadata={"category": "A"})
        self.db.insert(id="test5", vector=[0.6, 0.6, 0.6], metadata={"category": "B"})
        
        results = self.db.search(
            query=[0.5, 0.5, 0.5], 
            top_k=3, 
            filter=lambda meta: meta.get("category") == "A"
        )
        
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0].id, "test4")
        
    def test_delete(self):
        # Insert test vectors
        self.db.insert(id="test1", vector=[1.0, 0.0, 0.0])
        self.db.insert(id="test2", vector=[0.0, 1.0, 0.0])
        
        # Delete one vector
        self.db.delete(id="test1")
        
        # Verify it's deleted
        results = self.db.search(query=[1.0, 0.0, 0.0], top_k=2)
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0].id, "test2")
        
        # Test deleting non-existent vector
        with self.assertRaises(KeyError):
            self.db.delete(id="test999")
            
    def test_persistence(self):
        # Insert test vectors
        self.db.insert(id="test1", vector=[1.0, 0.0, 0.0])
        self.db.insert(id="test2", vector=[0.0, 1.0, 0.0])
        
        # Close the database
        self.db.close()
        
        # Reopen the database
        self.db = VectorLiteDB(self.temp_file.name)
        
        # Verify vectors are still there
        results = self.db.search(query=[1.0, 0.0, 0.0], top_k=2)
        self.assertEqual(len(results), 2)
        self.assertEqual(results[0].id, "test1")
        
    def tearDown(self):
        # Close the database
        self.db.close()
        
        # Remove the temporary file
        os.unlink(self.temp_file.name)
        
if __name__ == "__main__":
    unittest.main()

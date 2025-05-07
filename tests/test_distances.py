import unittest
import numpy as np
from vectorlitedb.core.distances import l2_distance, cosine_similarity, dot_product

class TestDistanceFunctions(unittest.TestCase):
    def test_l2_distance(self):
        v1 = [1.0, 0.0, 0.0]
        v2 = [0.0, 1.0, 0.0]
        self.assertAlmostEqual(l2_distance(v1, v2), np.sqrt(2), places=5)
        
        v3 = [1.0, 2.0, 3.0]
        v4 = [1.0, 2.0, 3.0]
        self.assertAlmostEqual(l2_distance(v3, v4), 0.0, places=5)
        
        v5 = [1.0, 2.0, 3.0]
        v6 = [4.0, 5.0, 6.0]
        self.assertAlmostEqual(l2_distance(v5, v6), np.sqrt(27), places=5)
    
    def test_cosine_similarity(self):
        v1 = [1.0, 0.0, 0.0]
        v2 = [0.0, 1.0, 0.0]
        self.assertAlmostEqual(cosine_similarity(v1, v2), 0.0, places=5)
        
        v3 = [1.0, 2.0, 3.0]
        v4 = [2.0, 4.0, 6.0]
        self.assertAlmostEqual(cosine_similarity(v3, v4), 1.0, places=5)
        
        v5 = [1.0, 2.0, 3.0]
        v6 = [-1.0, -2.0, -3.0]
        self.assertAlmostEqual(cosine_similarity(v5, v6), -1.0, places=5)
    
    def test_dot_product(self):
        v1 = [1.0, 0.0, 0.0]
        v2 = [0.0, 1.0, 0.0]
        self.assertAlmostEqual(dot_product(v1, v2), 0.0, places=5)
        
        v3 = [1.0, 2.0, 3.0]
        v4 = [4.0, 5.0, 6.0]
        self.assertAlmostEqual(dot_product(v3, v4), 4 + 10 + 18, places=5)

if __name__ == "__main__":
    unittest.main()
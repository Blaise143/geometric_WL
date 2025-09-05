import unittest
from src.utils import node_coloring

class TestNodeColoring(unittest.TestCase):
    def test_same_dict_different_order(self):
        obj1 = {"A": [1, 2], "B": [3, 4]}
        obj2 = {"B": [3, 4], "A": [1, 2]}
        self.assertEqual(node_coloring(obj1), node_coloring(obj2))

    def test_different_dicts(self):
        obj1 = {"A": [1, 2], "B": [3, 4]}
        obj2 = {"A": [1, 2], "B": [3, 5]}
        self.assertNotEqual(node_coloring(obj1), node_coloring(obj2))

    def test_consistency(self):
        obj1 = {"greeting": "π"}
        obj2 = {"greeting": "π"}
        self.assertEqual(node_coloring(obj1), node_coloring(obj2))

    def test_output_length(self):
        obj = {"A": 123}
        h = node_coloring(obj)
        self.assertEqual(len(h), 40)

if __name__ == "__main__":
    unittest.main()

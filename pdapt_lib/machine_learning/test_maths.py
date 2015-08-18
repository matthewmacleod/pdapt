import unittest
import maths

class TestSpectra(unittest.TestCase):

    def test_vector_add(self):
       self.assertEqual(maths.vector_add([1,2,3],[1,2,3]), [2,4,6])

    def test_vector_subtract(self):
       self.assertEqual(maths.vector_subtract([1,2,3],[1,2,3]), [0,0,0])

    def test_factorial(self):
       self.assertEqual(maths.factorial(5), 120)

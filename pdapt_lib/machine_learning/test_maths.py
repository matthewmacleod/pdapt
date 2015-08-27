import unittest
import pdapt_lib.machine_learning.maths as maths

class TestSpectra(unittest.TestCase):

    def test_vector_add(self):
       self.assertEqual(maths.vector_add([1,2,3],[1,2,3]), [2,4,6])

    def test_vector_subtract(self):
       self.assertEqual(maths.vector_subtract([1,2,3],[1,2,3]), [0,0,0])

    def test_factorial(self):
       self.assertEqual(maths.factorial(5), 120)

    def test_dihedral(self):
            self.assertEqual(maths.dihedral([-2.498019,2.157814,-1.513401],[-2.974569,3.029520,-1.062112],[-3.317570,2.819690,0.802274],[-3.629337,4.650860,1.235025]),164.23895763720364)

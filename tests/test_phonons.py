import unittest
import numpy as np
from pydmclab.hpc.phonons import AnalyzePhonons

from phonopy.structure.atoms import PhonopyAtoms
from phonopy.harmonic.dynamical_matrix import DynamicalMatrix

class UnitTestAnalyzePhonons(unittest.TestCase):
    
    def setUp(self):
        data_dir = "../pydmclab/data/test_data/phonons/phonons"
        self.ap = AnalyzePhonons(data_dir)

        self.data_dir = data_dir

    def test_init_(self):
        """
        Test the initialization of the AnalyzePhonons object with the supercell matrix and mesh arguments specified
        """
        supercell_matrix = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
        mesh=85
        
        ap_test = AnalyzePhonons(self.data_dir, supercell_matrix, mesh)
        if ap_test is not None:
            print("AnalyzePhonons object created")
        else:
            print("AnalyzePhonons object not created")
        
        self.assertIsInstance(ap_test.unitcell, PhonopyAtoms)
        self.assertIsInstance(ap_test.force_constants, np.ndarray)

        self.assertAlmostEqual(ap_test.force_constants[0][0][0][0], 1.8433662478, places=4)
        self.assertIsInstance(ap_test.dynamical_matrix, DynamicalMatrix)

        self.assertAlmostEqual(ap_test.thermal_properties()[30]['entropy'], 3479.485618173328, places=4)

    def test_mesh(self):
        mesh = self.ap.mesh
        self.assertIsInstance(mesh, dict)

    def test_force_constants(self):
        force_constants = self.ap.force_constants
        self.assertIsInstance(force_constants, np.ndarray)
        self.assertAlmostEqual(force_constants[0][0][0][0], 1.84336625, places=4)
        self.assertAlmostEqual(force_constants[0][5][2][2], -0.01228379, places=4)
    
    def test_thermal_properties(self):
        thermal_properties = self.ap.thermal_properties()
        self.assertAlmostEqual(thermal_properties[30]['entropy'], 3479.6892859770733, places=4)
        
        t_min, t_max, t_step = 0, 1000, 10
        tp = self.ap.thermal_properties(t_min=t_min, t_max=t_max, t_step=t_step, force_rerun=True)
        self.assertAlmostEqual(tp[30]['entropy'], 2395.48425510759, places=4)

    def test_total_dos(self):
        phonon_dos = self.ap.total_dos
        self.assertIsInstance(phonon_dos['frequency_points'], np.ndarray)
        self.assertAlmostEqual(phonon_dos['total_dos'][20], 0.02141757498414954, places=4)

if __name__ == '__main__':
    unittest.main(argv=['', '-v'], exit=False)
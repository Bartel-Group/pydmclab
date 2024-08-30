import unittest
import numpy as np
from pydmclab.hpc.analyze import VASPOutputs, AnalyzeVASP, AnalyzeBatch
from pydmclab.hpc.analyze import AnalyzePhonons


from phonopy.structure.atoms import PhonopyAtoms
from phonopy.harmonic.dynamical_matrix import DynamicalMatrix

class UnitTestVaspOutputs(unittest.TestCase):
    def test_vasp_outputs(self, calc_dir):
        vo = VASPOutputs(calc_dir)
        return
    
class UnitTestAnalyzePhonons(unittest.TestCase):
    def setUp(self):
        data_dir = "../pydmclab/data/test_data/analyze"
        self.ap = AnalyzePhonons(data_dir)

        self.data_dir = data_dir


    def test_init_(self, supercell_matrix=[[2, 0, 0], [0, 1, 0], [0, 0, 2]], mesh=50):
        ap_test = AnalyzePhonons(self.data_dir, supercell_matrix, mesh)
        if ap_test:
            print("AnalyzePhonons object created")
        else:
            print("AnalyzePhonons object not created")
        
        self.assertIsInstance(ap_test.unitcell, PhonopyAtoms)
        self.assertIsInstance(ap_test.force_constants, np.ndarray)

        self.assertEqual(ap_test.force_constants[0][0][0], np.array([ 1.84336625, -0.        , -0.        ]))
        self.assertIsInstance(ap_test.dynamical_matrix, DynamicalMatrix)

        self.assertEqual(ap_test.thermal_properties()[30]['entropy'], 5139.760194037135)


    def test_force_constants(self):
        force_constants = self.ap.force_constants()
        self.assertEqual(force_constants[0][0][0], np.array([ 1.84336625, -0.        , -0.        ]))
        self.assertEqual(force_constants[0][5][2], np.array([-0.        , -0.        , -0.01228379]))
    
    def test_thermal_properties(self):
        thermal_properties = self.ap.thermal_properties()
        self.assertEqual(thermal_properties[30]['entropy'], 2395.48425510759)


    def test_total_dos(self):
        phonon_dos = self.ap.total_dos()
        self.assertIsInstance(phonon_dos['frequency_points'], np.ndarray)
        self.assertEqual(phonon_dos['total_dos'][20], 0.02141757498414954)

        

class UnitTestAnalyzeVasp(unittest.TestCase):
    # Test AnalyzeVASP class

    def setUp(self):
        data_dir = "../pydmclab/data/test_data/analyze" #Need to add info to this folder to get stuff in AnalyzeVASP to work
        self.av = AnalyzeVASP(data_dir)

        self.data_dir = data_dir
    ap = AnalyzeVASP(calc_dir)


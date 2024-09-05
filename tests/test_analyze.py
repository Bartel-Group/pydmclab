import unittest
import numpy as np
from pydmclab.hpc.analyze import VASPOutputs, AnalyzeVASP, AnalyzeBatch


class UnitTestVaspOutputs(unittest.TestCase):
    def test_vasp_outputs(self, calc_dir):
        vo = VASPOutputs(calc_dir)
        return
        

class UnitTestAnalyzeVasp(unittest.TestCase):
    # Test AnalyzeVASP class

    def setUp(self):
        data_dir = "../pydmclab/data/test_data/analyze" #Need to add info to this folder to get stuff in AnalyzeVASP to work
        self.av = AnalyzeVASP(data_dir)

        self.data_dir = data_dir
    ap = AnalyzeVASP(calc_dir)


from pydmclab.hpc.analyze import VASPOutputs, AnalyzePhonons, AnalyzeVASP, AnalyzeBatch

class UnitTestAnalyze(unittest.TestCase):
    def test_vasp_outputs(self):
        vo = VASPOutputs()
        vasprun = vo.vasprun
    
    def test_analyze_vasp(self):
        # Test AnalyzeVASP class
        analyze_vasp = AnalyzeVASP()
        analyze_vasp.get_bandgap()
        analyze_vasp.get_optimized_structure()
        analyze_vasp.get_convergence()
        analyze_vasp.get_dos()
        analyze_vasp.get_bandstructure()
        analyze_vasp.get_phonon()
        analyze_vasp.get_ionic_steps()
        analyze_vasp.get_eigenvalues()
        analyze_vasp.get_force_constants()
        analyze_vasp.get_phonon_dos()
        analyze_vasp.get_phonon_bandstructure

import unittest

from pydmclab.hpc.passer import Passer
from pydmclab.core.struc import StrucTools
from pydmclab.utils.handy import read_json, write_json

from pydmclab.hpc.analyze import AnalyzeVASP

from pymatgen.io.vasp.inputs import Incar, Poscar, Kpoints

import os


class UnitTestPasser(unittest.TestCase):

    @classmethod
    def setUpClass(self): # Optional: This method runs once before all tests
        self.hse06_preggastatic_poscar_path = os.path.join("..", "pydmclab", "data", "test_data", "hpc", "passer", "calcs", "Co1Li1O2", "mp-22526", "fm", "hse06-preggastatic", "POSCAR")
        self.hse06_prelobster_incar_path = os.path.join("..", "pydmclab", "data", "test_data", "hpc", "passer", "calcs", "Co1Li1O2", "mp-22526", "fm", "hse06-prelobster", "INCAR")
        self.hse06_prelobster_ibzkpt_path = os.path.join("..", "pydmclab", "data", "test_data", "hpc", "passer", "calcs", "Co1Li1O2", "mp-22526", "fm", "hse06-prelobster", "IBZKPT")

    def setUp(self): # Initialization before each test method
        hse06_preggastatic_passer_dict_as_str = '{"xc_calc": "hse06-preggastatic", \
            "calc_list": ["gga-relax", "gga-static", "metagga-relax", "metagga-static", "metaggau-relax", "metaggau-static", \
            "hse06-preggastatic", "hse06-prelobster", "hse06-lobster"], \
            "calc_dir": "../pydmclab/data/test_data/hpc/passer/calcs/Co1Li1O2/mp-22526/fm/hse06-preggastatic", \
            "incar_mods": {}, \
            "launch_dir": "../pydmclab/data/test_data/hpc/passer/calcs/Co1Li1O2/mp-22526/fm", \
            "struc_src_for_hse": "metagga-relax"}'
        hse06_prelobster_passer_dict_as_str = '{"xc_calc": "hse06-prelobster", \
            "calc_list": ["gga-relax", "gga-static", "metagga-relax", "metagga-static", "metaggau-relax", "metaggau-static", \
            "hse06-preggastatic", "hse06-prelobster", "hse06-lobster"], \
            "calc_dir": "../pydmclab/data/test_data/hpc/passer/calcs/Co1Li1O2/mp-22526/fm/hse06-prelobster", \
            "incar_mods": {}, \
            "launch_dir": "../pydmclab/data/test_data/hpc/passer/calcs/Co1Li1O2/mp-22526/fm", \
            "struc_src_for_hse": "metagga-relax"}'
            
        self.passer_hse06_preggastatic = Passer(passer_dict_as_str=hse06_preggastatic_passer_dict_as_str)
        self.passer_hse06_prelobster = Passer(passer_dict_as_str=hse06_prelobster_passer_dict_as_str)
        ##### working progress #####
        
    @classmethod
    def tearDownClass(self): # Cleanup after all test methods have been run
        #pass # Clean up any resources here (e.g. close files, disconnect from databases, etc.)
        # Delete the POSCAR file created during the test_update_poscar method
        if os.path.exists(self.hse06_preggastatic_poscar_path):
            os.remove(self.hse06_preggastatic_poscar_path)
            print(f"\nDeleted {self.hse06_preggastatic_poscar_path}")
        
        # Delete the INCAR and IBZKPT files created during the test_setup_prelobster method
        if os.path.exists(self.hse06_prelobster_incar_path):
            os.remove(self.hse06_prelobster_incar_path)
            print(f"Deleted {self.hse06_prelobster_incar_path}")
            
        if os.path.exists(self.hse06_prelobster_ibzkpt_path):
            os.remove(self.hse06_prelobster_ibzkpt_path)
            print(f"Deleted {self.hse06_prelobster_ibzkpt_path}")
        ##### working progress #####
        
    def test_prev_xc_calc(self):
        """
        Test the prev_xc_calc property of the Passer class
        """
        self.assertEqual(self.passer_hse06_preggastatic.prev_xc_calc, "metagga-relax")
        self.assertEqual(self.passer_hse06_prelobster.prev_xc_calc, "hse06-preggastatic")
        ##### working progress #####
    
    def test_prev_xc(self):
        """
        Test the prev_xc property of the Passer class
        """
        self.assertEqual(self.passer_hse06_preggastatic.prev_xc, "metagga")

    def test_prev_calc(self):
        """
        Test the prev_calc property of the Passer class
        """
        self.assertEqual(self.passer_hse06_preggastatic.prev_calc, "relax")
    
    def test_curr_xc(self):
        """
        Test the curr_xc property of the Poscar class
        """
        self.assertEqual(self.passer_hse06_preggastatic.curr_xc, "hse06")

    def test_curr_calc(self):
        """
        Test the curr_calc property of the Poscar class
        """
        self.assertEqual(self.passer_hse06_preggastatic.curr_calc, "preggastatic")   
        
    def test_prev_calc_dir(self):
        """
        Test the prev_calc_dir property of the Poscar class
        """
        self.assertEqual(self.passer_hse06_preggastatic.prev_calc_dir, "../pydmclab/data/test_data/hpc/passer/calcs/Co1Li1O2/mp-22526/fm/metagga-relax")
        
    def test_prev_calc_convergence(self):
        """
        Test the prev_calc_convergence property of the Poscar class
        """
        self.assertEqual(self.passer_hse06_preggastatic.prev_calc_convergence, True)     
        
    def test_kill_job(self):
        """
        Test the kill_job method of the Passer class
        """
        self.assertEqual(self.passer_hse06_preggastatic.kill_job, False)
        
    def test_is_curr_calc_being_restarted(self):
        """
        Test the is_curr_calc_being_restarted method of the Passer class
        """
        self.assertEqual(self.passer_hse06_preggastatic.is_curr_calc_being_restarted, False)
    
    def test_update_poscar(self):
        """
        Test the update_poscar method of the Passer class
        """
        self.passer_hse06_preggastatic.update_poscar
        self.assertTrue(os.path.exists(self.hse06_preggastatic_poscar_path), f"{self.hse06_preggastatic_poscar_path} does not exist")
    
    def test_setup_prelobster(self):
        """
        Test the setup_prelobster method of the Passer class
        """
        self.assertEqual(self.passer_hse06_preggastatic.setup_prelobster, None)
        
        self.assertEqual(self.passer_hse06_prelobster.setup_prelobster, "incar_ibzkpt_copied")
        self.passer_hse06_prelobster.setup_prelobster
        self.assertTrue(os.path.exists(self.hse06_prelobster_incar_path), f"{self.hse06_prelobster_incar_path} does not exist")
        self.assertTrue(os.path.exists(self.hse06_prelobster_ibzkpt_path), f"{self.hse06_prelobster_ibzkpt_path} does not exist")
        

    

    # def test_copy_kpoints_for_prelobster(self):
    #     """
    #     Test the copy_kpoints_for_prelobster method of the Passer class
    #     """
    #     kpoints_path = os.path.join("..", "pydmclab", "data", "test_data", "hpc", "test_passer", "calcs", "Co1Li1O2", "mp-22526", "fm", "hse06-prelobster", "KPOINTS")
    #     self.assertEqual(self.passer_hse06_prelobster.copy_kpoints_for_prelobster, "copied kpoints")
    #     self.assertTrue(os.path.exists(kpoints_path), f"{kpoints_path} does not exist")
    
    # def test_copy_chgcar_for_parchg(self):
    #     """
    #     Test the copy_chgcar_for_parchg method of the Passer class
    #     """
    #     chgcar_path = os.path.join("..", "pydmclab", "data", "test_data", "hpc", "test_passer", "calcs", "Co1Li1O2", "mp-22526", "fm", "hse06-parchg", "CHGCAR")
    #     self.assertEqual(self.passer_hse06_parchg.copy_chgcar_for_parchg, "copied chgcar")
    #     self.assertTrue(os.path.exists(chgcar_path), f"{chgcar_path} does not exist")
    
    # def test_copy_kpoints_for_parchg(self):
    #     """
    #     Test the copy_kpoints_for_parchg method of the Passer class
    #     """
    #     kpoints_path = os.path.join("..", "pydmclab", "data", "test_data", "hpc", "test_passer", "calcs", "Co1Li1O2", "mp-22526", "fm", "hse06-parchg", "KPOINTS")
    #     self.assertEqual(self.passer_hse06_parchg.copy_kpoints_for_parchg, "copied kpoints")
    #     self.assertTrue(os.path.exists(kpoints_path), f"{kpoints_path} does not exist")
    
    # def test_copy_wavecar(self):
    #     """
    #     Test the copy_wavecar method of the Passer class
    #     """
    #     wavecar_path = os.path.join("..", "pydmclab", "data", "test_data", "hpc", "test_passer", "calcs", "Co1Li1O2", "mp-22526", "fm", "hse06-parchg", "WAVECAR")
    #     self.assertEqual(self.passer_hse06_parchg.copy_wavecar, "copied wavecar")
    #     self.assertTrue(os.path.exists(wavecar_path), f"{wavecar_path} does not exist")

    # def test_prev_gap(self):
    #     """
    #     Test the prev_gap property of the Passer class
    #     """
    #     self.assertEqual(self.passer_metagga_relax.prev_gap, 1.8732999999999995)
        
    # def test_bandgap_label(self):
    #     """
    #     Test the bandgap_label property of the Passer class
    #     """
    #     self.assertEqual(self.passer_metagga_relax.bandgap_label, "insulator")
        
    # def test_bandgap_based_incar_adjustments(self):
    #     """
    #     Test the bandgap_based_incar_adjustments property of the Passer class
    #     """
    #     self.assertEqual(self.passer_metagga_static.bandgap_based_incar_adjustments["ISMEAR"], -5)
    #     self.assertEqual(self.passer_metagga_static.bandgap_based_incar_adjustments["KSPACING"], 0.22)

    # def test_magmom_based_incar_adjustments(self):
    #     """
    #     Test the magmom_based_incar_adjustments property of the Passer class
    #     """
    #     self.assertEqual(self.passer_metagga_relax.magmom_based_incar_adjustments["MAGMOM"], 4*-0.0)
        
    # def test_nbands_based_incar_adjustments(self):
    #     """
    #     Test the nbands_based_incar_adjustments property of the Passer class
    #     """
    #     self.assertEqual(self.passer_hse06_lobster.nbands_based_incar_adjustments["NBANDS"], 24)
    
    # def test_prev_number_of_kpoints(self):
    #     """
    #     Test the prev_number_of_kpoints property of the Passer class
    #     """
    #     self.assertEqual(self.passer_hse06_prelobster.prev_number_of_kpoints, 868)
    
    # def test_kpoints_based_incar_adjustments(self):
    #     """
    #     Test the kpoints_based_incar_adjustments property of the Passer class
    #     """
    #     self.assertEqual(self.passer_hse06_lobster.kpoints_based_incar_adjustments(ncore=4)["KPAR"], 4)
        
    # def test_nelect_from_neutral_calc_dir(self):
    #     """
    #     Test the nelect_from_neutral_calc_dir method of the Passer class
    #     """
    #     self.assertEqual(self.passer_metagga_static.nelect_from_neutral_calc_dir, 24) 
        
    # #def test_charged_defects_based_incar_adjustments
        
    # def test_pass_kpoints_for_lobster(self):
    #     """
    #     Test the pass_kpoints_for_lobster method of the Passer class
    #     """
    #     kpoints_path = os.path.join("..", "pydmclab", "data", "test_data", "hpc", "test_passer", "calcs", "Co1Li1O2", "mp-22526", "fm", "hse06-lobster", "KPOINTS")
    #     self.assertEqual(self.passer_hse06_lobster.pass_kpoints_for_lobster, "copied IBZKPT from prev calc")
    #     self.assertTrue(os.path.exists(kpoints_path), f"{kpoints_path} does not exist")
        
    # def test_update_incar(self):
    #     """
    #     Test the update_incar method of the Passer class
    #     """
    #     incar_path = os.path.join("..", "pydmclab", "data", "test_data", "hpc", "test_passer", "calcs", "Co1Li1O2", "mp-22526", "fm", "hse06-lobster", "INCAR")
    #     self.assertEqual(self.passer_hse06_lobster.update_incar, "updated incar")
    #     self.assertTrue(os.path.exists(incar_path), f"{incar_path} does not exist")
    
    # def test_write_to_job_killer(self):
    #     """
    #     Test the write_to_job_killer method of the Passer class
    #     """
    #     job_killer_path = os.path.join("..", "pydmclab", "data", "test_data", "hpc", "test_passer", "job_killer.o")
    #     self.assertTrue(os.path.exists(job_killer_path), f"{job_killer_path} does not exist")
    
    # def test_complete_pass(self):
    #     """
    #     Test the complete_pass method of the Passer class
    #     """
    #     self.assertEqual(self.passer_metagga_static.complete_pass, "completed pass")        


if __name__ == "__main__":
    unittest.main()


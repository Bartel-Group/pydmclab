import unittest
import shutil
import os

from pydmclab.hpc.passer import Passer
from pydmclab.core.struc import StrucTools
from pydmclab.utils.handy import read_json, write_json
from pydmclab.hpc.analyze import AnalyzeVASP

from pymatgen.io.vasp.inputs import Incar, Poscar, Kpoints


class UnitTestPasser(unittest.TestCase):

    @classmethod
    def setUpClass(self): # Optional: This method runs once before all tests
        self.prev_xc_calc_dict = {"gga-loose":"gga-pre_loose", "gga-relax":"gga-loose", "gga-static":"gga-relax", \
            "metagga-relax":"gga-relax", "metagga-static": "metagga-relax", \
            "metaggau-relax":"gga-relax", "metaggau-static": "metaggau-relax", \
            "hse06-preggastatic":"metagga-relax", "hse06-prelobster":"hse06-preggastatic", "hse06-lobster":"hse06-prelobster", "hse06-parchg":"hse06-lobster"}
        
        self.metagga_static_incar_path = os.path.join("..", "pydmclab", "data", "test_data", "hpc", "passer", "calcs", "Co1Li1O2", "mp-22526", "fm", "metagga-static", "INCAR")
        self.metagga_static_incar_backup_path = os.path.join("..", "pydmclab", "data", "test_data", "hpc", "passer", "calcs", "Co1Li1O2", "mp-22526", "fm", "metagga-static", "INCAR_backup")
        self.metagga_static_wavecar_path = os.path.join("..", "pydmclab", "data", "test_data", "hpc", "passer", "calcs", "Co1Li1O2", "mp-22526", "fm", "metagga-static", "WAVECAR")
        self.metagga_static_chgcar_path = os.path.join("..", "pydmclab", "data", "test_data", "hpc", "passer", "calcs", "Co1Li1O2", "mp-22526", "fm", "metagga-static", "CHGCAR")
        self.metagga_static_errors_path = os.path.join("..", "pydmclab", "data", "test_data", "hpc", "passer", "calcs", "Co1Li1O2", "mp-22526", "fm", "metagga-static", "errors.o")

        self.hse06_preggastatic_poscar_path = os.path.join("..", "pydmclab", "data", "test_data", "hpc", "passer", "calcs", "Co1Li1O2", "mp-22526", "fm", "hse06-preggastatic", "POSCAR")
        self.hse06_preggastatic_wavecar_path = os.path.join("..", "pydmclab", "data", "test_data", "hpc", "passer", "calcs", "Co1Li1O2", "mp-22526", "fm", "hse06-preggastatic", "WAVECAR")
        
        self.hse06_prelobster_incar_path = os.path.join("..", "pydmclab", "data", "test_data", "hpc", "passer", "calcs", "Co1Li1O2", "mp-22526", "fm", "hse06-prelobster", "INCAR")
        self.hse06_prelobster_kpoints_path = os.path.join("..", "pydmclab", "data", "test_data", "hpc", "passer", "calcs", "Co1Li1O2", "mp-22526", "fm", "hse06-prelobster", "KPOINTS")
        
        self.hse06_lobster_kpoints_path = os.path.join("..", "pydmclab", "data", "test_data", "hpc", "passer", "calcs", "Co1Li1O2", "mp-22526", "fm", "hse06-lobster", "KPOINTS")
        self.hse06_lobster_incar_path = os.path.join("..", "pydmclab", "data", "test_data", "hpc", "passer", "calcs", "Co1Li1O2", "mp-22526", "fm", "hse06-lobster", "INCAR")
        self.hse06_lobster_incar_backup_path = os.path.join("..", "pydmclab", "data", "test_data", "hpc", "passer", "calcs", "Co1Li1O2", "mp-22526", "fm", "hse06-lobster", "INCAR_backup")

        self.hse06_parchg_chgcar_path = os.path.join("..", "pydmclab", "data", "test_data", "hpc", "passer", "calcs", "Co1Li1O2", "mp-22526", "fm", "hse06-parchg", "CHGCAR")
        self.hse06_parchg_kpoints_path = os.path.join("..", "pydmclab", "data", "test_data", "hpc", "passer", "calcs", "Co1Li1O2", "mp-22526", "fm", "hse06-parchg", "KPOINTS")
        
        self.job_killer_path = os.path.join("..", "pydmclab", "data", "test_data", "hpc", "passer", "calcs", "Co1Li1O2", "mp-22526", "fm", "job_killer.o")
        self.job_killer_backup_path = os.path.join("..", "pydmclab", "data", "test_data", "hpc", "passer", "calcs", "Co1Li1O2", "mp-22526", "fm", "job_killer_backup.o")

        # Make a backup of the original INCAR file
        shutil.copy(self.metagga_static_incar_path, self.metagga_static_incar_backup_path)
        shutil.copy(self.hse06_lobster_incar_path, self.hse06_lobster_incar_backup_path)
        shutil.copy(self.job_killer_path, self.job_killer_backup_path)

        ##### working progress #####

    def setUp(self): # Initialization before each test method
        metagga_static_passer_dict_as_str = '{"xc_calc": "metagga-static", \
            "calc_list": ["gga-relax", "gga-static", "metagga-relax", "metagga-static", "metaggau-relax", "metaggau-static", \
            "hse06-preggastatic", "hse06-prelobster", "hse06-lobster"], \
            "calc_dir": "../pydmclab/data/test_data/hpc/passer/calcs/Co1Li1O2/mp-22526/fm/metagga-static", \
            "incar_mods": {}, \
            "launch_dir": "../pydmclab/data/test_data/hpc/passer/calcs/Co1Li1O2/mp-22526/fm", \
            "struc_src_for_hse": "metagga-relax"}'     
        metagga_defect_neutral_passer_dict_as_str = '{"xc_calc": "metagga-defect_neutral", \
            "calc_list": ["gga-relax", "gga-static", "metagga-relax", "metagga-static", "metagga-defect_neutral", "metagga-defect_charged_p1", \
            "metaggau-relax", "metaggau-static", \
            "hse06-preggastatic", "hse06-prelobster", "hse06-lobster"], \
            "calc_dir": "../pydmclab/data/test_data/hpc/passer/calcs/Co1Li1O2/mp-22526/fm/metagga-defect_neutral", \
            "incar_mods": {}, \
            "launch_dir": "../pydmclab/data/test_data/hpc/passer/calcs/Co1Li1O2/mp-22526/fm", \
            "struc_src_for_hse": "metagga-relax"}'  
        metagga_defect_charged_p_1_passer_dict_as_str = '{"xc_calc": "metagga-defect_charged_p1", \
            "calc_list": ["gga-relax", "gga-static", "metagga-relax", "metagga-static", "metagga-defect_neutral", "metagga-defect_charged_p1", \
            "metaggau-relax", "metaggau-static", \
            "hse06-preggastatic", "hse06-prelobster", "hse06-lobster"], \
            "calc_dir": "../pydmclab/data/test_data/hpc/passer/calcs/Co1Li1O2/mp-22526/fm/metagga-defect_charged_p1", \
            "incar_mods": {}, \
            "launch_dir": "../pydmclab/data/test_data/hpc/passer/calcs/Co1Li1O2/mp-22526/fm", \
            "struc_src_for_hse": "metagga-relax"}'  
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
        hse06_lobster_passer_dict_as_str = '{"xc_calc": "hse06-lobster", \
            "calc_list": ["gga-relax", "gga-static", "metagga-relax", "metagga-static", "metaggau-relax", "metaggau-static", \
            "hse06-preggastatic", "hse06-prelobster", "hse06-lobster"], \
            "calc_dir": "../pydmclab/data/test_data/hpc/passer/calcs/Co1Li1O2/mp-22526/fm/hse06-lobster", \
            "incar_mods": {}, \
            "launch_dir": "../pydmclab/data/test_data/hpc/passer/calcs/Co1Li1O2/mp-22526/fm", \
            "struc_src_for_hse": "metagga-relax"}'
        hse06_parchg_passer_dict_as_str = '{"xc_calc": "hse06-parchg", \
            "calc_list": ["gga-relax", "gga-static", "metagga-relax", "metagga-static", "metaggau-relax", "metaggau-static", \
            "hse06-preggastatic", "hse06-prelobster", "hse06-lobster", "hse06-parchg"], \
            "calc_dir": "../pydmclab/data/test_data/hpc/passer/calcs/Co1Li1O2/mp-22526/fm/hse06-parchg", \
            "incar_mods": {}, \
            "launch_dir": "../pydmclab/data/test_data/hpc/passer/calcs/Co1Li1O2/mp-22526/fm", \
            "struc_src_for_hse": "metagga-relax"}'
            
        self.passer_metagga_static = Passer(passer_dict_as_str=metagga_static_passer_dict_as_str)
        self.passer_metagga_defect_neutral = Passer(passer_dict_as_str=metagga_defect_neutral_passer_dict_as_str)
        self.passer_metagga_defect_charged_p_1 = Passer(passer_dict_as_str=metagga_defect_charged_p_1_passer_dict_as_str)
        self.passer_hse06_preggastatic = Passer(passer_dict_as_str=hse06_preggastatic_passer_dict_as_str)
        self.passer_hse06_prelobster = Passer(passer_dict_as_str=hse06_prelobster_passer_dict_as_str)
        self.passer_hse06_lobster = Passer(passer_dict_as_str=hse06_lobster_passer_dict_as_str)
        self.passer_hse06_parchg = Passer(passer_dict_as_str=hse06_parchg_passer_dict_as_str)
        ##### working progress #####
        
    @classmethod
    def tearDownClass(self): # Cleanup after all test methods have been run
        #pass # Clean up any resources here (e.g. close files, disconnect from databases, etc.)
        # Delete the POSCAR file created during the test_update_poscar method
        if os.path.exists(self.metagga_static_wavecar_path):
            os.remove(self.metagga_static_wavecar_path)
            print(f"\nDeleted {self.metagga_static_wavecar_path}")
        
        if os.path.exists(self.metagga_static_chgcar_path):
            os.remove(self.metagga_static_chgcar_path)
            print(f"Deleted {self.metagga_static_chgcar_path}")
        
        if os.path.exists(self.metagga_static_errors_path):
            os.remove(self.metagga_static_errors_path)
            print(f"Deleted {self.metagga_static_errors_path}")
        
        if os.path.exists(self.hse06_preggastatic_poscar_path):
            os.remove(self.hse06_preggastatic_poscar_path)
            print(f"Deleted {self.hse06_preggastatic_poscar_path}")
            
        if os.path.exists(self.hse06_preggastatic_wavecar_path):
            os.remove(self.hse06_preggastatic_wavecar_path)
            print(f"Deleted {self.hse06_preggastatic_wavecar_path}")
        
        # Delete the INCAR and IBZKPT files created during the test_setup_prelobster method
        if os.path.exists(self.hse06_prelobster_incar_path):
            os.remove(self.hse06_prelobster_incar_path)
            print(f"Deleted {self.hse06_prelobster_incar_path}")
            
        if os.path.exists(self.hse06_prelobster_kpoints_path):
            os.remove(self.hse06_prelobster_kpoints_path)
            print(f"Deleted {self.hse06_prelobster_kpoints_path}")
        
        if os.path.exists(self.hse06_lobster_kpoints_path):
            os.remove(self.hse06_lobster_kpoints_path)
            print(f"Deleted {self.hse06_lobster_kpoints_path}")    
        
        if os.path.exists(self.hse06_parchg_chgcar_path):
            os.remove(self.hse06_parchg_chgcar_path)
            print(f"Deleted {self.hse06_parchg_chgcar_path}")
            
        if os.path.exists(self.hse06_parchg_kpoints_path):
            os.remove(self.hse06_parchg_kpoints_path)
            print(f"Deleted {self.hse06_parchg_kpoints_path}")
            
        # Restore the original INCAR file from the backup
        shutil.move(self.hse06_lobster_incar_backup_path, self.hse06_lobster_incar_path)
        shutil.move(self.metagga_static_incar_backup_path, self.metagga_static_incar_path)
        shutil.move(self.job_killer_backup_path, self.job_killer_path)
        
        ##### working progress #####
        
    def test_prev_xc_calc(self):
        """
        Test the prev_xc_calc property of the Passer class
        """
        self.assertEqual(self.passer_hse06_preggastatic.prev_xc_calc, self.prev_xc_calc_dict["hse06-preggastatic"])
        self.assertEqual(self.passer_hse06_prelobster.prev_xc_calc, self.prev_xc_calc_dict["hse06-prelobster"])
    
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
        
        self.assertEqual(self.passer_hse06_prelobster.setup_prelobster, "incar_ibzkpt copied")
        self.passer_hse06_prelobster.setup_prelobster
        self.assertTrue(os.path.exists(self.hse06_prelobster_incar_path), f"{self.hse06_prelobster_incar_path} does not exist")
        self.passer_hse06_prelobster.setup_prelobster
        self.assertTrue(os.path.exists(self.hse06_prelobster_kpoints_path), f"{self.hse06_prelobster_kpoints_path} does not exist")
        
    def test_setup_parchg(self):
        """
        Test the setup_parchg method of the Passer class
        """
        self.assertEqual(self.passer_hse06_preggastatic.setup_parchg, None)
        
        self.assertEqual(self.passer_hse06_parchg.setup_parchg, "chgcar_kpoints copied")
        self.passer_hse06_parchg.setup_prelobster
        self.assertTrue(os.path.exists(self.hse06_parchg_chgcar_path), f"{self.hse06_parchg_chgcar_path} does not exist")
        self.passer_hse06_parchg.setup_prelobster
        self.assertTrue(os.path.exists(self.hse06_parchg_kpoints_path), f"{self.hse06_parchg_kpoints_path} does not exist")
        
    def test_errors_encountered_in_curr_calc(self):
        """
        Test the errors_encountered_in_curr_calc property of the Passer class
        """
        self.assertEqual(self.passer_hse06_preggastatic.errors_encountered_in_curr_calc, ["error"])
   
    def test_copy_wavecar(self):
        """
        Test the copy_wavecar method of the Passer class
        """
        self.assertEqual(self.passer_hse06_preggastatic.copy_wavecar, "copied wavecar")
        self.passer_hse06_preggastatic.copy_wavecar
        self.assertTrue(os.path.exists(self.hse06_preggastatic_wavecar_path), f"{self.hse06_preggastatic_wavecar_path} does not exist")

    def test_copy_chgcar(self):
        """
        Test the copy_chgcar method of the Passer class
        """
        if os.path.exists(self.metagga_static_wavecar_path):
            os.remove(self.metagga_static_wavecar_path)
        self.assertEqual(self.passer_metagga_static.copy_chgcar, None)
        errors_path = os.path.join(self.passer_metagga_static.calc_dir, "errors.o")
        with open(errors_path, "w", encoding="utf-8") as f:
            f.write("eddrmm")
        self.assertEqual(self.passer_metagga_static.copy_chgcar, "copied chgcar")
        if os.path.exists(errors_path):
            os.remove(errors_path)
    
    def test_pass_kpoints_for_lobster(self):
        """
        Test the pass_kpoints_for_lobster method of the Passer class
        """
        self.assertEqual(self.passer_hse06_lobster.pass_kpoints_for_lobster, "copied IBZKPT from prev calc")
        self.passer_hse06_lobster.pass_kpoints_for_lobster
        self.assertTrue(os.path.exists(self.hse06_lobster_kpoints_path), f"{self.hse06_lobster_kpoints_path} does not exist")
        
    def test_prev_gap(self):
        """
        Test the prev_gap property of the Passer class
        """
        self.assertEqual(self.passer_metagga_static.prev_gap, 1.8732000000000006)
        
    def test_bandgap_label(self):
        """
        Test the bandgap_label property of the Passer class
        """
        self.assertEqual(self.passer_metagga_static.bandgap_label, "insulator")
        
    def test_bandgap_based_incar_adjustments(self):
        """
        Test the bandgap_based_incar_adjustments property of the Passer class
        """
        self.assertEqual(self.passer_metagga_static.bandgap_based_incar_adjustments["ISMEAR"], -5)
        self.assertEqual(self.passer_metagga_static.bandgap_based_incar_adjustments["KSPACING"], 0.22)

    def test_magmom_based_incar_adjustments(self):
        """
        Test the magmom_based_incar_adjustments property of the Passer class
        """
        self.assertEqual(self.passer_metagga_static.magmom_based_incar_adjustments["MAGMOM"], "0.0 0.0 0.0 0.0" or "-0.0 0.0 0.0 -0.0")
        
    def test_nbands_based_incar_adjustments(self):
        """
        Test the nbands_based_incar_adjustments property of the Passer class
        """
        self.assertEqual(self.passer_hse06_lobster.nbands_based_incar_adjustments["NBANDS"], 32)

    def test_prev_number_of_kpoints(self):
        """
        Test the prev_number_of_kpoints property of the Passer class
        """
        self.assertEqual(self.passer_hse06_lobster.prev_number_of_kpoints, 868)
    
    def test_kpoints_based_incar_adjustments(self):
        """
        Test the kpoints_based_incar_adjustments property of the Passer class
        """
        self.assertEqual(self.passer_hse06_lobster.kpoints_based_incar_adjustments(ncore=4)["KPAR"], 4)
        
    def test_nelect_from_neutral_calc_dir(self):
        """
        Test the nelect_from_neutral_calc_dir method of the Passer class
        """
        self.assertEqual(self.passer_metagga_defect_charged_p_1.nelect_from_neutral_calc_dir, 24) 
    
    def test_charged_defects_based_incar_adjustments(self):
        """
        Test the charged_defects_based_incar_adjustments method of the Passer class
        """
        self.assertEqual(self.passer_metagga_defect_charged_p_1.charged_defects_based_incar_adjustments["NELECT"], 23)
    #### working progress #####
    
    def test_poscar(self):
        """
        Test the poscar property of the Poscar class
        """
        self.assertEqual(self.passer_metagga_static.poscar.comment, Poscar.from_file(os.path.join(self.passer_metagga_static.calc_dir, "POSCAR")).comment)
        #self.assertEqual(self.passer_metagga_static.poscar, Poscar.from_file(os.path.join(self.passer_metagga_static.calc_dir, "POSCAR")))
    
    def test_update_incar(self):
        """
        Test the update_incar method of the Passer class
        """
        self.assertEqual(self.passer_hse06_lobster.update_incar(
            wavecar_out=self.passer_hse06_lobster.copy_wavecar,
            prelobster_out=self.passer_hse06_lobster.setup_prelobster,
            parchg_out=self.passer_hse06_lobster.setup_parchg,
            chgcar_out=self.passer_hse06_lobster.copy_chgcar
            ), "updated incar")
        self.passer_hse06_lobster.update_incar(
            wavecar_out=self.passer_hse06_lobster.copy_wavecar,
            prelobster_out=self.passer_hse06_lobster.setup_prelobster,
            parchg_out=self.passer_hse06_lobster.setup_parchg,
            chgcar_out=self.passer_hse06_lobster.copy_chgcar
            )
        self.assertEqual(Incar.from_file(os.path.join(self.passer_hse06_lobster.calc_dir, "INCAR"))["ISMEAR"], -5)
        self.assertEqual(Incar.from_file(os.path.join(self.passer_hse06_lobster.calc_dir, "INCAR"))["NBANDS"], 32)
        self.assertEqual(Incar.from_file(os.path.join(self.passer_hse06_lobster.calc_dir, "INCAR"))["ISTART"], 1)

    def test_write_to_job_killer(self):
        """
        Test the write_to_job_killer method of the Passer class
        """
        self.assertEqual(self.passer_metagga_static.write_to_job_killer, False)  
        self.passer_metagga_static.write_to_job_killer
        with open(self.job_killer_path, 'r', encoding='utf-8') as file:
            self.assertEqual(file.read().strip(), "good to pass")
            
    def test_complete_pass(self):
        """
        Test the complete_pass method of the Passer class
        """
        self.assertEqual(self.passer_metagga_static.complete_pass, "completed pass")  
        self.passer_metagga_static.complete_pass
        self.assertTrue(os.path.exists(self.metagga_static_wavecar_path), f"{self.metagga_static_wavecar_path} does not exist")
        self.assertEqual(Incar.from_file(os.path.join(self.passer_metagga_static.calc_dir, "INCAR"))["ISTART"], 1)



if __name__ == "__main__":
    unittest.main()


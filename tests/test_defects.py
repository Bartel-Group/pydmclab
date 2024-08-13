import unittest
import os
import numpy as np
from numpy.testing import assert_array_equal

from pymatgen.core.structure import Structure
from shakenbreak.input import Distortions
from doped.generation import DefectsGenerator

from pydmclab.utils.handy import read_json
from pydmclab.core.struc import StrucTools

from pydmclab.core.defects import (
    SupercellForDefects,
    DefectStructures,
    ShakeDefectiveStrucs,
    GenerateMostDefects,
)


class TestDefects_SupercellForDefects(unittest.TestCase):
    """
    Test the SupercellForDefects class in defects.py
    """

    def setUp(self):
        """
        Set up the test
        """

        self.data_dir = "../pydmclab/data/test_data/defects"
        strucs = read_json(os.path.join(self.data_dir, "AlN_struc.json"))
        self.AlN_struc_dict = strucs["Al1N1"]["mp-661"]

    def test_init(self):
        """
        Test the __init__ method of SupercellForDefects
        """

        AlN_supercell = SupercellForDefects(
            self.AlN_struc_dict,
            min_image_distance=15.2,
            min_atoms=93,
            force_cubic=False,
            force_diagnoal=False,
            ideal_threshold=0.2,
            savename="AlN_supercell",
        )

        self.assertIsInstance(
            AlN_supercell.sc_structure,
            Structure,
            "The structure is not a pymatgen Structure object!",
        )
        self.assertEqual(AlN_supercell.min_image_distance, 15.2)
        self.assertEqual(AlN_supercell.min_atoms, 93)
        self.assertEqual(AlN_supercell.force_cubic, False)
        self.assertEqual(AlN_supercell.force_diagonal, False)
        self.assertEqual(AlN_supercell.ideal_threshold, 0.2)
        self.assertEqual(AlN_supercell.pbar, None)
        self.assertEqual(AlN_supercell.savename, "AlN_supercell")
        self.assertEqual(AlN_supercell.data_dir, os.getcwd())

    def test_make_supercell(self):
        """
        Test the make_supercell method of SupercellForDefects
        """

        # check if the supercell has at least 120 atoms
        AlN_supercell = SupercellForDefects(
            self.AlN_struc_dict, min_image_distance=1.0, min_atoms=140
        ).make_supercell(verbose=False)

        total_atoms_in_supercell = len(
            StrucTools(AlN_supercell).structure_as_dict["sites"]
        )

        self.assertGreaterEqual(total_atoms_in_supercell, 120)

        # check if the minimum image distance is greater than 10.0
        AlN_supercell = SupercellForDefects(
            self.AlN_struc_dict, min_image_distance=10.0, min_atoms=1
        ).make_supercell(verbose=False)

        self.assertGreaterEqual(
            SupercellForDefects(AlN_supercell).curr_min_image_distance(), 10.0
        )

        # check if savename is working at intended
        AlN_supercell = SupercellForDefects(
            self.AlN_struc_dict,
            min_image_distance=10.0,
            min_atoms=50,
            savename="AlN_supercell_temp",
            data_dir=self.data_dir,
        ).make_supercell(verbose=False)

        savepath = os.path.join(self.data_dir, "AlN_supercell_temp.cif")

        self.assertTrue(
            os.path.exists(savepath), "The supercell was not saved as a cif!"
        )

        try:
            os.remove(savepath)
        except OSError as e:
            print(f"Error: {e.strerror}")

    def test_curr_min_image_distance(self):
        """
        Test the curr_min_image_distance method of SupercellForDefects
        """

        # find the minimum image distnace for the as loaded AlN structure
        # would use this if checking if current supercell meets your minimum image distance requirement
        AlN_current_cell_check = SupercellForDefects(
            self.AlN_struc_dict, min_image_distance=10, min_atoms=50
        )

        self.assertAlmostEqual(
            AlN_current_cell_check.curr_min_image_distance(),
            3.1286,
            3,
            "The minimum image distance is incorrect!",
        )

    def test_find_primitive_structure(self):
        """
        Test the find_primitive_structure method of SupercellForDefects
        """

        # find primitive structure of supercell and check expected amounts of atoms

        supercell_savepath = os.path.join(self.data_dir, "AlN_supercell.cif")

        AlN_supercell = StrucTools(supercell_savepath).structure_as_dict

        AlN_primitive = SupercellForDefects(AlN_supercell).find_primitive_structure()

        self.assertEqual(
            StrucTools(AlN_primitive).amts,
            {"Al": 2, "N": 2},
            "Primitive structure is incorrect!",
        )

    def test_find_primitive_structure_grid(self):
        """
        Test the find_primitive_structure_grid method of SupercellForDefects
        """

        supercell_savepath = os.path.join(self.data_dir, "AlN_supercell.cif")

        AlN_supercell = StrucTools(supercell_savepath).structure_as_dict

        AlN_primitive, AlN_primitive_grid = SupercellForDefects(
            AlN_supercell
        ).find_primitive_structure_grid()

        self.assertEqual(
            StrucTools(AlN_primitive).amts,
            {"Al": 2, "N": 2},
            "Primitive structure is incorrect!",
        )
        assert_array_equal(
            AlN_primitive_grid,
            np.array([[4.0, -2.0, 0.0], [-2.0, 4.0, 0.0], [2.0, 0.0, 2.0]]),
            "Grid is incorrect!",
        )


class TestDefects_DefectStructures(unittest.TestCase):
    """
    Test the DefectStructures class in defects.py
    """

    def setUp(self):
        """
        Set up the test
        """

        data_dir = "../pydmclab/data/test_data/defects"
        AlN_supercell_savepath = os.path.join(data_dir, "AlN_supercell.cif")
        self.AlN_supercell = StrucTools(AlN_supercell_savepath).structure_as_dict
        self.AlN_defect_struc = DefectStructures(
            self.AlN_supercell, ox_states={"Al": 3, "N": -3}, how_many=2, n_strucs=1
        )

    def test_init(self):
        """
        Test the __init__ method of DefectStructures
        """

        self.assertIsInstance(
            self.AlN_defect_struc.supercell,
            Structure,
            "The structure is not a pymatgen Structure object!",
        )
        self.assertEqual(self.AlN_defect_struc.ox_states, {"Al": 3, "N": -3})
        self.assertEqual(self.AlN_defect_struc.how_many, 2)
        self.assertEqual(self.AlN_defect_struc.n_strucs, 1)

    def test_vacancies(self):
        """
        Test the vacancies method of DefectStructures
        """

        AlN_vacancy_strucs = self.AlN_defect_struc.vacancies("N")

        self.assertEqual(
            StrucTools(AlN_vacancy_strucs[0]).formula,
            "Al48 N46",
            "Vacancy structure is incorrect!",
        )

    def test_subsitutions(self):
        """
        Test the substitutions method of DefectStructures
        """

        AlN_substitution_strucs = self.AlN_defect_struc.substitutions("Al_N")

        self.assertEqual(
            StrucTools(AlN_substitution_strucs[0]).formula,
            "Al50 N46",
            "Substitution structure is incorrect!",
        )

    def test_ordered_strucs(self):
        """
        Test getting various ordered structures
        """

        AlN_defect_struc = DefectStructures(
            self.AlN_supercell,
            ox_states={"Al": 3, "Ga": 3, "N": -3},
            how_many=1,
            n_strucs=1,
        )

        AlN_substiution_struc = AlN_defect_struc.substitutions("Ga_Al")

        AlN_defect_struc = DefectStructures(
            AlN_substiution_struc[0],
            ox_states={"Al": 3, "Ga": 3, "N": -3},
            how_many=1,
            n_strucs=1,
        )

        AlN_substiution_struc = AlN_defect_struc.substitutions("Ga_N")

        AlN_defect_struc = DefectStructures(
            AlN_substiution_struc[0],
            ox_states={"Al": 3, "Ga": 3, "N": -3},
            how_many=1,
            n_strucs=2,
        )

        AlN_vacancy_strucs = AlN_defect_struc.vacancies("Ga")

        self.assertEqual(len(AlN_vacancy_strucs), 2)


class TestDefects_ShakeDefectiveStrucs(unittest.TestCase):
    """
    Test the ShakeDefectiveStrucs class in defects.py
    """

    def setUp(self):
        """
        Set up the test
        """

        data_dir = "../pydmclab/data/test_data/defects"
        self.AlN_pristine_sc = read_json(
            os.path.join(data_dir, "AlN_pristine_supercell.json")
        )
        self.AlN_vacancy_defects = read_json(
            os.path.join(data_dir, "AlN_vacancy_defects.json")
        )
        self.AlN_shaking_vacancies = ShakeDefectiveStrucs(
            self.AlN_vacancy_defects, self.AlN_pristine_sc
        )

    def test_init(self):
        """
        Test the __init__ method of ShakeDefectiveStrucs
        """

        initial_defect_strucs = self.AlN_shaking_vacancies.initial_defect_strucs
        bulk_struc = self.AlN_shaking_vacancies.bulk_struc
        distortions = self.AlN_shaking_vacancies.distortions
        shaken_defects_data = self.AlN_shaking_vacancies.shaken_defects_data
        distortions_metadata = self.AlN_shaking_vacancies.distortions_metadata

        self.assertIsInstance(initial_defect_strucs, list)
        for struc in initial_defect_strucs:
            self.assertIsInstance(struc, Structure)
        self.assertIsInstance(bulk_struc, Structure)
        self.assertIsInstance(distortions, Distortions)
        self.assertIsInstance(shaken_defects_data, dict)
        self.assertEqual(
            shaken_defects_data["v_Al_C3v_N1.90"]["defect_type"], "vacancy"
        )
        self.assertIsInstance(
            shaken_defects_data["v_Al_C3v_N1.90"]["charges"][-3]["structures"][
                "Unperturbed"
            ],
            Structure,
        )
        self.assertIsInstance(
            shaken_defects_data["v_Al_C3v_N1.90"]["charges"][-3]["structures"][
                "distortions"
            ]["Rattled"],
            Structure,
        )
        self.assertIsInstance(distortions_metadata, dict)
        self.assertEqual(
            distortions_metadata["defects"]["v_Al_C3v_N1.90"]["unique_site"],
            [0.500238, 0.666785, 0.499644],
        )
        self.assertEqual(
            distortions_metadata["distortion_parameters"]["distortion_increment"], 0.1
        )

    def test_get_shaken_strucs_summary(self):
        """
        Test the get_shaken_strucs_summary method of ShakeDefectiveStrucs
        """

        shaken_strucs_summary = self.AlN_shaking_vacancies.get_shaken_strucs_summary

        self.assertIsInstance(shaken_strucs_summary, dict)
        self.assertEqual(
            list(shaken_strucs_summary.keys()), ["v_Al_C3v_N1.90", "v_N_C3v_Al1.90"]
        )
        self.assertEqual(len(shaken_strucs_summary["v_Al_C3v_N1.90"].keys()), 40)
        self.assertEqual(
            len(
                shaken_strucs_summary["v_Al_C3v_N1.90"][
                    "v_Al_C3v_N1.90__-1__Bond_Distortion_-10.0%"
                ]["sites"]
            ),
            95,
        )

    def test_get_shaken_strucs(self):
        """
        Test the get_shaken_strucs method of ShakeDefectiveStrucs
        """

        shaken_strucs = self.AlN_shaking_vacancies.get_shaken_strucs(
            -2, defects_of_interest=["v_Al_C3v_N1.90"]
        )

        self.assertEqual(list(shaken_strucs.keys()), ["Al47N48"])
        self.assertEqual(len(shaken_strucs["Al47N48"]), 13)
        self.assertEqual(
            len(
                shaken_strucs["Al47N48"]["v_Al_C3v_N1.90__-2__Bond_Distortion_-30.0%"][
                    "sites"
                ]
            ),
            95,
        )


class TestDefects_GenerateMostDefects(unittest.TestCase):
    """
    Test the GenerateMostDefects class in defects.py
    """

    def setUp(self):
        """
        Set up the test
        """

        data_dir = "../pydmclab/data/test_data/defects"
        strucs = read_json(os.path.join(data_dir, "AlN_struc.json"))
        self.AlN_struc_dict = strucs["Al1N1"]["mp-661"]
        self.AlN_defects = GenerateMostDefects(self.AlN_struc_dict)

    def test_init(self):
        """
        Test the __init__ method of GenerateMostDefects
        """

        self.assertIsInstance(
            self.AlN_defects.all_defects,
            DefectsGenerator,
            "The all_defects attribute is not a DefectsGenerator object!",
        )
        self.assertIsInstance(
            self.AlN_defects.pristine_struc,
            Structure,
            "The pristine_struc attribute is not a pymatgen Structure object!",
        )

    def test_summary_of_defects(self):
        """
        Test the summary_of_defects method of GenerateMostDefects
        """

        self.assertIsInstance(self.AlN_defects.summary_of_defects, DefectsGenerator)

    def test_to_dict(self):
        """
        Test the to_dict method of GenerateMostDefects
        """

        self.assertIsInstance(self.AlN_defects.to_dict, dict)

    def test_get_all_defective_supercells(self):
        """
        Test the get_all_defective_supercells method of GenerateMostDefects
        """

        all_defective_supercells = self.AlN_defects.get_all_defective_supercells

        self.assertIsInstance(all_defective_supercells, dict)
        self.assertEqual(len(all_defective_supercells), 59)
        self.assertEqual(len(all_defective_supercells["Al_N_+1"]["sites"]), 96)

    def test_add_charge_states(self):
        """
        Test the add_charge_states method of GenerateMostDefects
        """

        AlN_defects = GenerateMostDefects(self.AlN_struc_dict)
        AlN_defects.add_charge_states("v_Al", [2, 3])
        all_defective_supercells = AlN_defects.get_all_defective_supercells

        self.assertEqual(len(all_defective_supercells), 61)

    def test_remove_charge_states(self):
        """
        Test the remove_charge_states method of GenerateMostDefects
        """

        AlN_defects = GenerateMostDefects(self.AlN_struc_dict)
        AlN_defects.remove_charge_states("Al_N", [6, 5])
        all_defective_supercells = AlN_defects.get_all_defective_supercells

        self.assertEqual(len(all_defective_supercells), 57)


if __name__ == "__main__":
    unittest.main()

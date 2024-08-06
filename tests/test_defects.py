import unittest
import os
import numpy as np
from numpy.testing import assert_array_equal

from pymatgen.core.structure import Structure

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
        ).make_supercell

        total_atoms_in_supercell = len(
            StrucTools(AlN_supercell).structure_as_dict["sites"]
        )

        self.assertGreaterEqual(total_atoms_in_supercell, 120)

        # check if the minimum image distance is greater than 10.0
        AlN_supercell = SupercellForDefects(
            self.AlN_struc_dict, min_image_distance=10.0, min_atoms=1
        ).make_supercell

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
        ).make_supercell

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
        self.data_dir = "../pydmclab/data/test_data/defects"
        strucs = read_json(os.path.join(self.data_dir, "AlN_struc.json"))
        self.AlN_struc_dict = strucs["Al1N1"]["mp-661"]

    def test_init(self):
        """
        Test the __init__ method of DefectStructures
        """
        # placeholder

    def test_vacancies(self):
        """
        Test the vacancies method of DefectStructures
        """
        # placeholder

    def test_subsitutions(self):
        """
        Test the substitutions method of DefectStructures
        """
        # placeholder


# make another class to test

if __name__ == "__main__":
    unittest.main()

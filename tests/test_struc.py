import os
import unittest
import warnings

from pydmclab.core.struc import StrucTools
from pydmclab.utils.handy import read_json

from pymatgen.core.structure import Structure

TEST_DATA = os.path.join("..", "pydmclab", "data", "test_data", "struc")


class TestStrucTools(unittest.TestCase):
    """
    Test cases for the StrucTools class.
    """

    def setUp(self):
        cro_dict = read_json(os.path.join(TEST_DATA, "cro_structure.json"))
        crmo_dict = read_json(os.path.join(TEST_DATA, "crmo_structure.json"))
        cro_supercell_dict = read_json(os.path.join(TEST_DATA, "cro_supercell.json"))

        self.struc_cro = Structure.from_dict(cro_dict)
        self.struc_crmo = Structure.from_dict(crmo_dict)
        self.struc_supercell = Structure.from_dict(cro_supercell_dict)

        self.st_cro = StrucTools(self.struc_cro)
        self.st_crmo = StrucTools(self.struc_crmo)

    def test_init_with_empty_structure_path(self):
        with self.assertRaises(ValueError):
            StrucTools(structure="my_fake_path")

    def test_compact_formula(self):
        self.assertEqual(self.st_cro.compact_formula, "Cr2O3")

    def test_formula(self):
        self.assertEqual(self.st_cro.formula, "Cr4 O6")

    def test_els(self):
        self.assertEqual(self.st_cro.els, ["Cr", "O"])

    def test_amts(self):
        self.assertEqual(self.st_cro.amts, {"Cr": 4, "O": 6})

    def test_make_supercell(self):
        new_supercell = self.st_cro.make_supercell([1, 2, 3], verbose=False)
        self.assertEqual(len(new_supercell), 60)
        self.assertEqual(new_supercell, self.struc_supercell)

    def test_change_occ_for_site(self):
        new_struc = self.st_cro.change_occ_for_site(5, {"Cr": 0})
        self.assertEqual(len(new_struc), 9)

        new_struc = self.st_cro.change_occ_for_site(5, {"Li": 0.5})
        self.assertEqual(new_struc[5].species_string, "Li:0.5")

        new_supercell = self.st_cro.change_occ_for_site(
            55, {"Cr": 0}, structure=self.struc_supercell
        )
        self.assertEqual(len(new_supercell), 59)

        new_supercell = self.st_cro.change_occ_for_site(
            55, {"Li": 0.5}, structure=self.struc_supercell
        )
        self.assertEqual(new_supercell[55].species_string, "Li:0.5")

    def test_change_occ_for_el(self):
        new_struc = self.st_cro.change_occ_for_el("O", {"Li": 0.5})
        self.assertEqual(new_struc[5].species_string, "Li:0.5")

        new_supercell = self.st_cro.change_occ_for_el(
            "Cr", {"Li": 0.5}, structure=self.struc_supercell
        )
        self.assertEqual(new_supercell[5].species_string, "Li:0.5")

    def test_decorate_with_ox_states(self):
        self.st_cro.ox_states = {"Cr": 3, "O": -2}
        oxidized_struc = self.st_cro.decorate_with_ox_states
        self.assertEqual(oxidized_struc[0].species_string, "Cr3+")

    def test_get_ordered_structures(self):
        warnings.filterwarnings("ignore", category=DeprecationWarning)

        ordered_strucs = self.st_crmo.get_ordered_structures(n_strucs=2, verbose=False)
        self.assertEqual(len(ordered_strucs), 2)
        self.assertEqual(StrucTools(ordered_strucs[1]).compact_formula, "Cr1Mn1O3")

    def test_replace_species(self):
        strucs = self.st_cro.replace_species(
            species_mapping={"Cr": {"Cr": 0.5, "Mn": 0.5}},
            n_strucs=2,
            verbose=False,
        )
        self.assertEqual(StrucTools(strucs[1]).compact_formula, "Cr1Mn1O3")

    def test_get_spacegroup_info(self):
        self.assertEqual(self.st_cro.spacegroup_info["loose"]["number"], 167)
        self.assertEqual(self.st_cro.sg(), "R-3c")

    def test_scale_structure(self):
        initial_vol = self.st_cro.structure.volume
        scaled_vol = 1.2 * initial_vol
        scaled_struc = self.st_cro.scale_structure(1.2)
        self.assertAlmostEqual(scaled_struc.volume, scaled_vol, places=3)


if __name__ == "__main__":
    unittest.main()

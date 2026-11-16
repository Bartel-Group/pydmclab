import unittest
from pydmclab.core.hulls import (
    GetHullInputData,
    AnalyzeHull,
    ParallelHulls,
    MixingHull,
    UMAHull,
)
from pydmclab.data.thermochem import uma_s1p2p1_mp_formation_energies
from pydmclab.utils.handy import read_json


class UnitTestHulls(unittest.TestCase):
    def _test_get_hullin(self):
        compounds = ["CaTiO3", "TiO2", "Al2O3", "CaO"]
        Efs = [-0.1, -0.2, -0.3, -0.4]
        compound_to_energy = {}
        for i, c in enumerate(compounds):
            compound_to_energy[c] = {"Ef": Efs[i]}
        hullin = GetHullInputData(compound_to_energy, "Ef").hullin_data(remake=True)

        self.assertEqual(len(hullin), 2)

        self.assertEqual(hullin["Al_O"]["Al2O3"]["E"], -0.3)

        self.assertEqual(hullin["Ca_O_Ti"]["Ca1O3Ti1"]["amts"]["Ca"], 0.2)

    def _test_get_hullout(self):
        compounds = ["MnO", "MnO2"]
        Efs = [-2, -1.5]
        compound_to_energy = {}
        for i, c in enumerate(compounds):
            compound_to_energy[c] = {"Ef": Efs[i]}
        hullin = GetHullInputData(compound_to_energy, "Ef").hullin_data(remake=True)

        ah = AnalyzeHull(hullin, "Mn_O")
        hullout = ah.hull_output_data

        self.assertEqual(
            hullout["Mn1O1"]["Ed"], ah.cmpd_hull_output_data("Mn1O1")["Ed"]
        )

        Ed_MnO2 = (1 / 3) * ((-1.5 * 3) - (-2 * 2))

        self.assertAlmostEqual(hullout["Mn1O2"]["Ed"], Ed_MnO2, places=5)

    def _test_parallel_hulls(self):
        compounds = ["CaTiO3", "TiO2", "Al2O3", "CaO"]
        Efs = [-0.1, -0.2, -0.3, -0.4]
        compound_to_energy = {}
        for i, c in enumerate(compounds):
            compound_to_energy[c] = {"Ef": Efs[i]}
        serial_hullin = GetHullInputData(compound_to_energy, "Ef").hullin_data(
            remake=True
        )

        ph = ParallelHulls(compound_to_energy, "Ef", n_procs=2, fresh_restart=True)

        parallel_hullin = ph.parallel_hullin()

        self.assertEqual(
            serial_hullin["Ca_O_Ti"]["Ca1O1"]["E"],
            parallel_hullin["Ca_O"]["Ca1O1"]["E"],
        )

        compounds = ["MnO", "MnO2", "Fe2O3", "Fe3O4"]
        Efs = [-2, -1.5, -1, -1]
        compound_to_energy = {}
        for i, c in enumerate(compounds):
            compound_to_energy[c] = {"Ef": Efs[i]}
        hullin = GetHullInputData(compound_to_energy, "Ef").hullin_data(remake=True)

        ah = AnalyzeHull(hullin, "Mn_O")
        serial_hullout = ah.hull_output_data

        ph = ParallelHulls(compound_to_energy, "Ef", n_procs=2, fresh_restart=True)
        hullin = ph.parallel_hullin()
        print(ph.compounds)

        smallest_spaces = ph.smallest_spaces(hullin)
        parallel_hullout = ph.parallel_hullout(hullin, smallest_spaces, remake=True)

        self.assertEqual(serial_hullout["Mn1O2"]["Ed"], parallel_hullout["Mn1O2"]["Ed"])

    def test_mixing(self):

        # test 1 - input end members agree with clean basis and are not divided (actual basis sane as clean)

        input_energies_1 = {
            "Al2O3": {"E": -10},
            "Ga2O3": {"E": -30},
            "AlGaO3": {"E": -25},
            "Al3GaO6": {"E": -11},
        }

        end_members = ["Al2O3", "Ga2O3"]
        energy_key = "E"
        divide_left, divide_right = 1, 1

        mh = MixingHull(
            input_energies=input_energies_1,
            left_end_member=end_members[0],
            right_end_member=end_members[1],
            energy_key=energy_key,
            divide_left_by=divide_left,
            divide_right_by=divide_right,
        )

        results = mh.results

        self.assertEqual(results["Al2O3"]["E_mix_per_fu"], 0)
        self.assertEqual(results["Ga2O3"]["E_mix_per_fu"], 0)
        self.assertEqual(results["Al2O3"]["x"], 0)
        self.assertEqual(results["Ga2O3"]["x"], 1)
        self.assertEqual(results["Al1Ga1O3"]["x"], 0.5)
        self.assertEqual(results["Al3Ga1O6"]["stability"], False)
        self.assertEqual(results["Al1Ga1O3"]["stability"], True)

        E_mix_middle_hard = -25 * 5 - (0.5 * -30 * 5 + 0.5 * -10 * 5)

        self.assertEqual(results["Al1Ga1O3"]["E_mix_per_fu"], E_mix_middle_hard)

        self.assertEqual(results["Al1Ga1O3"]["num_atoms_in_basis"], 5)
        self.assertEqual(results["Al3Ga1O6"]["num_atoms_in_basis"], 5)

        # test 2 - input end members agree with clean basis and are divided (actual basis different from clean)

        input_energies_2 = {
            "LaCoO3": {"E": -10},
            "LaCoO2.5": {"E": -15},
            "LaCoO2.75": {"E": -30},
            "LaCoO2.875": {"E": -5},
        }

        end_members = ["LaCoO3", "La2Co2O5"]
        energy_key = "E"
        divide_left, divide_right = 1, 2

        mh = MixingHull(
            input_energies=input_energies_2,
            left_end_member=end_members[0],
            right_end_member=end_members[1],
            energy_key=energy_key,
            divide_left_by=divide_left,
            divide_right_by=divide_right,
        )

        results = mh.results

        self.assertEqual(results["Co1La1O3"]["E_mix_per_fu"], 0)
        self.assertEqual(results["Co2La2O5"]["E_mix_per_fu"], 0)
        self.assertEqual(results["Co1La1O3"]["x"], 0)
        self.assertEqual(results["Co2La2O5"]["x"], 1)
        self.assertEqual(results["Co4La4O11"]["x"], 0.5)
        self.assertEqual(results["Co8La8O23"]["stability"], False)
        self.assertEqual(results["Co4La4O11"]["stability"], True)

        E_mix_middle_hard = -30 * 4.75 - (0.5 * -15 * 4.5 + 0.5 * -10 * 5)

        self.assertEqual(results["Co4La4O11"]["E_mix_per_fu"], E_mix_middle_hard)

        self.assertEqual(results["Co2La2O5"]["num_atoms_in_basis"], 4.5)
        self.assertEqual(results["Co4La4O11"]["num_atoms_in_basis"], 4.75)

        # test 3 - input end members do not agree with clean basis and are not divided (actual basis different from clean)

        end_members = ["LaCoO3", "LaCoO2.5"]
        energy_key = "E"
        divide_left, divide_right = 1, 1

        mh = MixingHull(
            input_energies=input_energies_2,
            left_end_member=end_members[0],
            right_end_member=end_members[1],
            energy_key=energy_key,
            divide_left_by=divide_left,
            divide_right_by=divide_right,
        )

        results = mh.results

        self.assertEqual(results["Co1La1O3"]["E_mix_per_fu"], 0)
        self.assertEqual(results["Co2La2O5"]["E_mix_per_fu"], 0)
        self.assertEqual(results["Co1La1O3"]["x"], 0)
        self.assertEqual(results["Co2La2O5"]["x"], 1)
        self.assertEqual(results["Co4La4O11"]["x"], 0.5)
        self.assertEqual(results["Co8La8O23"]["stability"], False)
        self.assertEqual(results["Co4La4O11"]["stability"], True)

        E_mix_middle_hard = -30 * 4.75 - (0.5 * -15 * 4.5 + 0.5 * -10 * 5)

        self.assertEqual(results["Co4La4O11"]["E_mix_per_fu"], E_mix_middle_hard)

        self.assertEqual(results["Co2La2O5"]["num_atoms_in_basis"], 4.5)
        self.assertEqual(results["Co4La4O11"]["num_atoms_in_basis"], 4.75)

        # test 4 - input end members do not agree with clean basis and are divided (actual basis same as clean)

        # check that input energies don't have to be clean or reduced form
        input_energies_3 = {
            "O12Al8": {"E": -10},
            "Ga4O6": {"E": -30},
            "AlGaO3": {"E": -25},
            "Al3GaO6": {"E": -11},
        }

        end_members = ["Al2O3", "Ga4O6"]
        energy_key = "E"
        divide_left, divide_right = 1, 2

        mh = MixingHull(
            input_energies=input_energies_3,
            left_end_member=end_members[0],
            right_end_member=end_members[1],
            energy_key=energy_key,
            divide_left_by=divide_left,
            divide_right_by=divide_right,
        )

        results = mh.results

        self.assertEqual(results["Al2O3"]["E_mix_per_fu"], 0)
        self.assertEqual(results["Ga2O3"]["E_mix_per_fu"], 0)
        self.assertEqual(results["Al2O3"]["x"], 0)
        self.assertEqual(results["Ga2O3"]["x"], 1)
        self.assertEqual(results["Al1Ga1O3"]["x"], 0.5)
        self.assertEqual(results["Al3Ga1O6"]["stability"], False)
        self.assertEqual(results["Al1Ga1O3"]["stability"], True)

        E_mix_middle_hard = -25 * 5 - (0.5 * -30 * 5 + 0.5 * -10 * 5)

        self.assertEqual(results["Al1Ga1O3"]["E_mix_per_fu"], E_mix_middle_hard)

        self.assertEqual(results["Al1Ga1O3"]["num_atoms_in_basis"], 5)
        self.assertEqual(results["Al3Ga1O6"]["num_atoms_in_basis"], 5)

        # test 5 - input end members do not agree with clean basis and are divided (actual basis different from clean)

        end_members = ["Al8O12", "Ga4O6"]
        energy_key = "E"
        divide_left, divide_right = 2, 1

        mh = MixingHull(
            input_energies=input_energies_3,
            left_end_member=end_members[0],
            right_end_member=end_members[1],
            energy_key=energy_key,
            divide_left_by=divide_left,
            divide_right_by=divide_right,
        )

        results = mh.results

        self.assertEqual(results["Al2O3"]["E_mix_per_fu"], 0)
        self.assertEqual(results["Ga2O3"]["E_mix_per_fu"], 0)
        self.assertEqual(results["Al2O3"]["x"], 0)
        self.assertEqual(results["Ga2O3"]["x"], 1)
        self.assertEqual(results["Al1Ga1O3"]["x"], 0.5)
        self.assertEqual(results["Al3Ga1O6"]["stability"], False)
        self.assertEqual(results["Al1Ga1O3"]["stability"], True)

        E_mix_middle_hard = -25 * 10 - (0.5 * -30 * 10 + 0.5 * -10 * 10)

        self.assertEqual(results["Al1Ga1O3"]["E_mix_per_fu"], E_mix_middle_hard)

        self.assertEqual(results["Al1Ga1O3"]["num_atoms_in_basis"], 10)
        self.assertEqual(results["Al3Ga1O6"]["num_atoms_in_basis"], 10)

        # test full system

        input_energies = read_json("../demos/output/hulls/data/query_Li-Mn-Fe-O.json")
        energy_key = "Ef_mp"
        end_members = ["Li", "MnO2"]
        divide_left, divide_right = 1, 1
        mix = MixingHull(
            input_energies=input_energies,
            left_end_member=end_members[0],
            right_end_member=end_members[1],
            energy_key=energy_key,
            divide_left_by=divide_left,
            divide_right_by=divide_right,
        )
        out = mix.results
        # print(out["Li1Mn2O4"])
        self.assertEqual(out["Li1Mn2O4"]["stability"], True)
        self.assertAlmostEqual(out["Li1Mn2O4"]["E_mix_per_at"], -0.520, places=2)
        self.assertEqual(out["Li1Mn2O4"]["mixing_rxn"], "Li + 2 MnO2 -> LiMn2O4")
        self.assertEqual(out["Li1Mn2O4"]["x"], 2 / 3)
        self.assertEqual(
            out["Li1Mn2O4"]["E_mix_per_fu"] / out["Li1Mn2O4"]["E_mix_per_at"],
            2 / 3 * 3 + 1 / 3 * 1,
        )

        # test full system with difficult basis

        end_members = ["LiMnO2", "Li9Mn20O40"]
        divide_left, divide_right = 1, 20
        mix = MixingHull(
            input_energies=input_energies,
            left_end_member=end_members[0],
            right_end_member=end_members[1],
            energy_key=energy_key,
            divide_left_by=divide_left,
            divide_right_by=divide_right,
        )
        out = mix.results
        # print(out["Li1Mn2O4"])

        self.assertEqual(out["Li1Mn2O4"]["stability"], True)
        self.assertAlmostEqual(out["Li1Mn2O4"]["E_mix_per_at"], -0.011, places=2)
        self.assertAlmostEqual(out["Li1Mn2O4"]["x"], 10 / 11, places=7)
        self.assertAlmostEqual(
            out["Li1Mn2O4"]["E_mix_per_fu"] / out["Li1Mn2O4"]["E_mix_per_at"],
            (10 / 11) * (69 / 20) + (1 / 11) * 4,
            places=4,
        )


class UnitTestUMAHull(unittest.TestCase):
    def test_single_vs_list_input(self):
        refs = {"Mn1O1": -2.0}
        single = UMAHull(("MnO2", -1.5), refs).hull_output_data
        listed = UMAHull([("MnO2", -1.5)], refs).hull_output_data
        self.assertEqual(single, listed)
        self.assertEqual(len(single), 1)

    def test_formula_is_cleaned(self):
        out = UMAHull(("O2Mn", -1.5), {"Mn1O1": -2.0}).hull_output_data[0]
        self.assertEqual(out["formula"], "Mn1O2")

    def test_reference_compounds(self):
        refs = {"Mn1O1": -2.0, "Fe2O3": -1.7, "Li1Mn1O2": -2.1, "Mn1": -0.1}
        hull = UMAHull(("MnO2", -1.5), refs)
        # only references made of a subset of Mn-O, plus elements at Ef = 0
        self.assertEqual(hull.els, ["Mn", "O"])
        self.assertEqual(
            hull.reference_compounds, {"Mn1O1": -2.0, "Mn1": 0.0, "O1": 0.0}
        )

    def test_stable(self):
        # MnO2 -> MnO + 1/2 O2
        out = UMAHull(("MnO2", -1.5), {"Mn1O1": -2.0}).hull_output_data[0]
        Ed_hard = (1 / 3) * ((-1.5 * 3) - (-2 * 2))
        self.assertAlmostEqual(out["Ed"], Ed_hard, places=6)
        self.assertTrue(out["stability"])
        self.assertEqual(out["rxn"], "1.0_Mn1O1 + 1.0_O")
        self.assertIsNone(out["dE_polymorph"])

    def test_unstable(self):
        out = UMAHull(("MnO2", -1.0), {"Mn1O1": -2.0}).hull_output_data[0]
        Ed_hard = (1 / 3) * ((-1.0 * 3) - (-2 * 2))
        self.assertAlmostEqual(out["Ed"], Ed_hard, places=6)
        self.assertFalse(out["stability"])
        self.assertEqual(out["rxn"], "1.0_Mn1O1 + 1.0_O")

    def test_element(self):
        # an element at Ef = 0 ties w/ its own reference
        out = UMAHull(("Cu", 0.0), {}).hull_output_data[0]
        self.assertAlmostEqual(out["Ed"], 0, places=8)
        self.assertTrue(out["stability"])
        self.assertEqual(out["rxn"], "1.0_Cu (ref)")
        self.assertEqual(out["dE_polymorph"], 0)

    def test_polymorph(self):
        refs = {"Mn1O1": -2.0, "Mn1O2": -1.5}
        out = UMAHull(("MnO2", -1.4), refs).hull_output_data[0]
        self.assertAlmostEqual(out["Ed"], 0.1, places=6)
        self.assertAlmostEqual(out["dE_polymorph"], 0.1, places=6)
        self.assertFalse(out["stability"])
        self.assertEqual(out["rxn"], "1.0_Mn1O2 (ref)")

        out = UMAHull(("MnO2", -1.6), refs).hull_output_data[0]
        self.assertAlmostEqual(out["dE_polymorph"], -0.1, places=6)
        self.assertTrue(out["stability"])

    def test_new_compounds_compete(self):
        # MnO2 competes w/ the new MnO (no MnO in the reference data)
        out = UMAHull([("MnO", -2.0), ("MnO2", -1.0)], {}).hull_output_data
        self.assertEqual(out[1]["rxn"], "1.0_Mn1O1 (input 0) + 1.0_O")
        self.assertAlmostEqual(out[1]["Ed"], (1 / 3) * (-3 + 4), places=6)
        # and MnO competes w/ the new MnO2: MnO -> 1/2 Mn + 1/2 MnO2
        self.assertAlmostEqual(out[0]["Ed"], -2.0 - 0.5 * (-1.0 * 3) / 2, places=6)
        self.assertEqual(out[0]["rxn"], "0.5_Mn + 0.5_Mn1O2 (input 1)")

    def test_unrelated_spaces(self):
        refs = {"Mn1O1": -2.0, "Cl1Na1": -2.1}
        alone = UMAHull(("MnO2", -1.5), refs).hull_output_data[0]
        together = UMAHull([("MnO2", -1.5), ("NaCl", -2.0)], refs).hull_output_data
        self.assertEqual(alone, together[0])
        self.assertAlmostEqual(together[1]["Ed"], 0.1, places=6)

    def test_packaged_data(self):
        refs = uma_s1p2p1_mp_formation_energies()
        self.assertIn("Co1Li1O2", refs)

        # hypothetical LiCoO2 polymorph, 30 meV/atom below UMA's LiCoO2 (see test_energies.py)
        #   its UMA energy was set as -5.7276 - 0.030 eV/atom, but -5.7276 is a rounded
        #   ground-state energy (-5.727614457894556 eV/atom w/ the packaged data), so Ed and
        #   dE_polymorph come out as -0.030 + 0.0000144579 = -0.0299855 instead of -0.030
        out = UMAHull(("LiCoO2", -1.9063681179807537)).hull_output_data[0]
        self.assertEqual(out["formula"], "Co1Li1O2")
        self.assertTrue(out["stability"])
        self.assertEqual(out["rxn"], "1.0_Co1Li1O2 (ref)")
        self.assertAlmostEqual(out["Ed"], -0.029985542105444463, places=6)
        self.assertAlmostEqual(out["dE_polymorph"], -0.02998554210544402, places=6)
        self.assertAlmostEqual(
            out["Ef"] - out["dE_polymorph"], refs["Co1Li1O2"], places=8
        )


if __name__ == "__main__":
    unittest.main()

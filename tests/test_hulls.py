import unittest
from pydmclab.core.hulls import GetHullInputData, AnalyzeHull, ParallelHulls, MixingHull
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
        input_energies = {
            "Al2O3": {"E": -10},
            "Ga2O3": {"E": -30},
            "AlGaO3": {"E": -25},
            "Al3GaO6": {"E": -11},
        }

        end_members = ["Al2O3", "Ga2O3"]
        energy_key = "E"
        divide_left, divide_right = 1, 1

        mh = MixingHull(
            input_energies=input_energies,
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

        input_energies = read_json(
            "../pydmclab/demos/output/hulls/data/query_Li-Mn-Fe-O.json"
        )
        energy_key = "Ef_mp"
        end_members = ["Li", "MnO2"]
        mix = MixingHull(
            input_energies=input_energies,
            left_end_member=end_members[0],
            right_end_member=end_members[1],
            energy_key=energy_key,
            divide_left_by=divide_left,
            divide_right_by=divide_right,
        )
        out = mix.results
        print(out["Li1Mn2O4"])
        self.assertEqual(out["Li1Mn2O4"]["stability"], True)
        self.assertAlmostEqual(out["Li1Mn2O4"]["E_mix_per_at"], -0.520, places=2)
        self.assertEqual(out["Li1Mn2O4"]["mixing_rxn"], "Li + 2 MnO2 -> LiMn2O4")
        self.assertEqual(out["Li1Mn2O4"]["x"], 2 / 3)
        self.assertEqual(
            out["Li1Mn2O4"]["E_mix_per_fu"] / out["Li1Mn2O4"]["E_mix_per_at"],
            2 / 3 * 3 + 1 / 3 * 1,
        )

        end_members = ["LiMnO2", "Li9Mn20O40"]
        divide_right = 20
        mix = MixingHull(
            input_energies=input_energies,
            left_end_member=end_members[0],
            right_end_member=end_members[1],
            energy_key=energy_key,
            divide_left_by=divide_left,
            divide_right_by=divide_right,
        )
        out = mix.results
        print(out["Li1Mn2O4"])

        self.assertEqual(out["Li1Mn2O4"]["stability"], True)
        self.assertAlmostEqual(out["Li1Mn2O4"]["E_mix_per_at"], -0.011, places=2)
        self.assertEqual(out["Li1Mn2O4"]["x"], 1 / 3)
        self.assertAlmostEqual(
            out["Li1Mn2O4"]["E_mix_per_fu"] / out["Li1Mn2O4"]["E_mix_per_at"],
            (1 / 3) * 69 / 20 + (2 / 3) * 4,
            places=4,
        )


if __name__ == "__main__":
    unittest.main()

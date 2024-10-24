import os
import unittest
import numpy as np

from pymatgen.core import Structure

from ase.filters import FrechetCellFilter
from ase.io.trajectory import Trajectory

from pydmclab.utils.handy import read_json
from pydmclab.mlp.dynamics import CHGNetRelaxer, CHGNetCalculator, AnalyzeMD
from pydmclab.core.struc import StrucTools

from chgnet.model import CHGNet
from chgnet.graph import CrystalGraphConverter

TEST_DATA = os.path.join("..", "pydmclab", "data", "test_data", "mlp")


class TestCHGNetCalculator(unittest.TestCase):

    def test_model_init(self):
        calculator = CHGNetCalculator()
        self.assertIsInstance(calculator.model, CHGNet)

        calculator = CHGNetCalculator(model="0.2.0")
        self.assertIsInstance(calculator.model, CHGNet)
        self.assertEqual(calculator.model.version, "0.2.0")

    def test_use_device(self):
        # make sure that the device is set correctly and not overwritten by determine_device()
        calculator = CHGNetCalculator(use_device="cpu")
        self.assertEqual(calculator.device, "cpu")

        calculator = CHGNetCalculator(use_device="mps")
        self.assertEqual(calculator.device, "mps")


class TestCHGNetRelaxer(unittest.TestCase):

    def test_model_init(self):
        relaxer = CHGNetRelaxer()
        self.assertIsInstance(relaxer.model, CHGNet)

        relaxer = CHGNetRelaxer(model="0.2.0")
        self.assertIsInstance(relaxer.model, CHGNet)
        self.assertEqual(relaxer.version, "0.2.0")

    def test_use_device(self):
        relaxer = CHGNetRelaxer(use_device="cpu")
        self.assertEqual(relaxer.calculator.device, "cpu")

        relaxer = CHGNetRelaxer(use_device="mps")
        self.assertEqual(relaxer.calculator.device, "mps")

    def test_legacy_algo_relax(self):
        cro_structure = Structure.from_dict(
            read_json(os.path.join(TEST_DATA, "cro_structure.json"))
        )

        model = CHGNet.load(verbose=False)
        converter = CrystalGraphConverter(
            atom_graph_cutoff=6, bond_graph_cutoff=3, algorithm="legacy"
        )
        self.assertEqual(converter.algorithm, "legacy")

        model.graph_converter = converter

        relaxer = CHGNetRelaxer(model=model, use_device="cpu")
        results = relaxer.relax(
            cro_structure, verbose=True, ase_filter=FrechetCellFilter
        )
        self.assertEqual(
            results.keys(), {"final_structure", "final_energy", "trajectory"}
        )
        self.assertIsInstance(results["final_structure"], Structure)
        self.assertIsInstance(
            results["final_structure"].site_properties["magmom"], list
        )
        self.assertEqual(
            len(results["final_structure"].site_properties["magmom"]),
            len(results["final_structure"]),
        )
        self.assertTrue(
            all(
                isinstance(mm, float)
                for mm in results["final_structure"].site_properties["magmom"]
            )
        )
        self.assertEqual(
            {*results["trajectory"].__dict__},
            {
                "atoms",
                "energies",
                "forces",
                "stresses",
                "magmoms",
                "atomic_numbers",
                "atom_positions",
                "cells",
            },
        )
        self.assertEqual(len(results["trajectory"]), 2)
        self.assertLessEqual(
            results["trajectory"].energies[-1], results["trajectory"].energies[0]
        )
        self.assertAlmostEqual(results["trajectory"].energies[-1], -91.242733, places=4)

    def test_fast_algo_relax(self):
        cro_structure = Structure.from_dict(
            read_json(os.path.join(TEST_DATA, "cro_structure.json"))
        )

        model = CHGNet.load(verbose=False)
        converter = CrystalGraphConverter(
            atom_graph_cutoff=6, bond_graph_cutoff=3, algorithm="fast"
        )
        self.assertEqual(converter.algorithm, "fast")

        model.graph_converter = converter

        relaxer = CHGNetRelaxer(model=model, use_device="cpu")
        results = relaxer.relax(
            cro_structure, verbose=True, ase_filter=FrechetCellFilter
        )
        self.assertEqual(
            results.keys(), {"final_structure", "final_energy", "trajectory"}
        )
        self.assertIsInstance(results["final_structure"], Structure)
        self.assertIsInstance(
            results["final_structure"].site_properties["magmom"], list
        )
        self.assertEqual(
            len(results["final_structure"].site_properties["magmom"]),
            len(results["final_structure"]),
        )
        self.assertTrue(
            all(
                isinstance(mm, float)
                for mm in results["final_structure"].site_properties["magmom"]
            )
        )
        self.assertEqual(
            {*results["trajectory"].__dict__},
            {
                "atoms",
                "energies",
                "forces",
                "stresses",
                "magmoms",
                "atomic_numbers",
                "atom_positions",
                "cells",
            },
        )
        self.assertEqual(len(results["trajectory"]), 2)
        self.assertLessEqual(
            results["trajectory"].energies[-1], results["trajectory"].energies[0]
        )
        self.assertAlmostEqual(results["trajectory"].energies[-1], -91.242733, places=4)


class TestAnalyzeMD(unittest.TestCase):

    def setUp(self):
        self.data_dir = "../pydmclab/data/test_data/mlp"
        self.trajfile = os.path.join(self.data_dir, "chgnet_md.traj")
        self.logfile = os.path.join(self.data_dir, "chgnet_md.log")
        self.amd = AnalyzeMD(logfile=self.logfile, trajfile=self.trajfile)

    def test_log_summary(self):
        summary = self.amd.log_summary
        # test the summary is a list of dictionaries
        self.assertIsInstance(summary, list)
        for entry in summary:
            self.assertIsInstance(entry, dict)
        # test the correct summary length is obtained
        self.assertEqual(len(summary), 31)
        # test the final entry of the summary is correct (time is corrected)
        self.assertAlmostEqual(summary[-1]["t"], 0.3, places=5)
        self.assertAlmostEqual(summary[-1]["T"], 942.7, places=5)
        self.assertAlmostEqual(summary[-1]["Etot"], -517.6383, places=5)
        self.assertAlmostEqual(summary[-1]["Epot"], -527.3867, places=5)
        self.assertAlmostEqual(summary[-1]["Ekin"], 9.7484, places=5)
        self.assertEqual(summary[-1]["run"], 4)
        # test transition between runs is handled correctly
        self.assertAlmostEqual(summary[6]["t"], 0.06, places=5)
        self.assertAlmostEqual(summary[6]["T"], 269.8, places=5)
        self.assertAlmostEqual(summary[6]["Etot"], -531.0874, places=5)
        self.assertAlmostEqual(summary[6]["Epot"], -533.8775, places=5)
        self.assertAlmostEqual(summary[6]["Ekin"], 2.7901, places=5)
        self.assertEqual(summary[6]["run"], 2)

    def test_traj_summary(self):
        summary = self.amd.traj_summary
        # test the summary is a list of dictionaries
        self.assertIsInstance(summary, list)
        for entry in summary:
            self.assertIsInstance(entry, dict)
        # test that the entries are structure dictionaries
        self.assertIsInstance(StrucTools(summary[0]).structure, Structure)
        self.assertIsInstance(StrucTools(summary[11]).structure, Structure)
        self.assertIsInstance(StrucTools(summary[-1]).structure, Structure)
        # test the correct number of structures is obtained
        self.assertEqual(len(summary), 31)
        # check correct structures are being retained
        with Trajectory(self.trajfile) as trajs:
            # the 3rd (indice 2) trajectory in the trajectory file should be our 3nd entry
            trajs_2_positions = trajs[2].get_positions()
            summary_2_positions = StrucTools(summary[2]).structure.cart_coords
            self.assertTrue(
                np.allclose(trajs_2_positions, summary_2_positions, atol=1e-8)
            )
            # the 8th trajectory in the trajectory file should be our 7th entry
            traj_7_positions = trajs[7].get_positions()
            summary_6_positions = StrucTools(summary[6]).structure.cart_coords
            self.assertTrue(
                np.allclose(traj_7_positions, summary_6_positions, atol=1e-8)
            )
            # the 24 trajectory in the trajectory file should be our 22nd entry
            traj_23_positions = trajs[23].get_positions()
            summary_21_positions = StrucTools(summary[21]).structure.cart_coords
            self.assertTrue(
                np.allclose(traj_23_positions, summary_21_positions, atol=1e-8)
            )
            # check positions are changing sufficiently between steps for previous tests to be valid
            summary_7_positions = StrucTools(summary[7]).structure.cart_coords
            self.assertFalse(
                np.allclose(traj_7_positions, summary_7_positions, atol=1e-8)
            )


if __name__ == "__main__":
    unittest.main()

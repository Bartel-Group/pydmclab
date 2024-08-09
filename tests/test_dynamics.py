import os
import unittest

from pymatgen.core import Structure

from ase.filters import FrechetCellFilter

from pydmclab.utils.handy import read_json
from pydmclab.mlp.dynamics import CHGNetRelaxer, CHGNetCalculator

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
        self.assertEqual(results.keys(), {"final_structure", "trajectory"})
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
        self.assertEqual(results.keys(), {"final_structure", "trajectory"})
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


if __name__ == "__main__":
    unittest.main()

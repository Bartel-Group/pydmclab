import os
import unittest
import numpy as np
import shutil
import tempfile

from pymatgen.core import Structure
from pymatgen.io.ase import AseAtomsAdaptor

from ase import Atoms
from ase.filters import FrechetCellFilter
from ase.io.trajectory import Trajectory

from pydmclab.utils.handy import read_json
from pydmclab.mlp.chgnet.dynamics import (
    CHGNetRelaxer,
    CHGNetCalculator,
    CHGNetMD,
    AnalyzeMD,
)
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


class TestCHGNetMD(unittest.TestCase):

    @classmethod
    def setUpClass(cls):

        cls.temp_dir = tempfile.mkdtemp()

        cls.data_dir = "../pydmclab/data/test_data/mlp"

        # structure for fresh md run
        cls.struc = StrucTools(
            os.path.join(cls.data_dir, "cro_structure.json")
        ).structure

        # file path for fresh md run
        cls.traj_new = os.path.join(cls.temp_dir, "new_chgnet_md.traj")
        cls.log_new = os.path.join(cls.temp_dir, "new_chgnet_md.log")

        # partially completed reference md run
        cls.original_trajfile = os.path.join(cls.data_dir, "chgnet_md.traj")
        cls.original_logfile = os.path.join(cls.data_dir, "chgnet_md.log")

        # files for continuing md run (from reference)
        cls.trajfile = os.path.join(cls.temp_dir, "chgnet_md.traj")
        cls.logfile = os.path.join(cls.temp_dir, "chgnet_md.log")

        shutil.copy2(cls.original_trajfile, cls.trajfile)
        shutil.copy2(cls.original_logfile, cls.logfile)

    @classmethod
    def tearDownClass(cls):

        shutil.rmtree(cls.temp_dir)

    def test_run(self):

        # fresh md run
        md = CHGNetMD(
            structure=self.struc,
            model="0.3.0",
            relax_first=False,
            timestep=1.0,
            loginterval=1,
            trajfile=self.traj_new,
            logfile=self.log_new,
        )

        self.assertEqual(md.relaxer, None)
        self.assertIsInstance(md.calculator, CHGNetCalculator)
        self.assertIsInstance(md.structure, Structure)
        self.assertIsInstance(md.atoms, Atoms)

        md.run(steps=5)

        with open(self.log_new, "r", encoding="utf-8") as logf:

            lines = logf.readlines()

            self.assertEqual(len(lines), 7)
            self.assertIn("Time", lines[0])
            self.assertIn("0.0050", lines[-1])

        with Trajectory(self.traj_new) as trajs:

            self.assertEqual(len(trajs), 6)
            self.assertIsInstance(AseAtomsAdaptor.get_structure(trajs[-1]), Structure)

        # check the pre-md relax
        md = CHGNetMD(
            structure=self.struc,
            model="0.3.0",
            relax_first=True,
            timestep=1.0,
            loginterval=1,
            trajfile=self.traj_new,
            logfile=self.log_new,
        )

        self.assertIsInstance(md.relaxer, CHGNetRelaxer)
        self.assertIsInstance(md.structure, Structure)
        self.assertNotEqual(self.struc, md.structure)

    def test_continue_from_traj(self):

        # continuing from partially completed md run
        md = CHGNetMD.continue_from_traj(
            trajfile=self.trajfile,
            logfile=self.logfile,
            model="0.3.0",
            relax_first=False,
            timestep=1.0,
            loginterval=1,
        )

        self.assertEqual(md.relaxer, None)
        self.assertIsInstance(md.calculator, CHGNetCalculator)
        self.assertIsInstance(md.structure, Structure)
        self.assertIsInstance(md.atoms, Atoms)

        md.run(steps=5)

        with open(self.logfile, "r", encoding="utf-8") as logf:

            lines = logf.readlines()

            self.assertEqual(len(lines), 45)
            self.assertIn("Time", lines[38])
            self.assertIn("0.0050", lines[-1])

        with Trajectory(self.trajfile) as trajs:

            self.assertEqual(len(trajs), 40)
            self.assertIsInstance(AseAtomsAdaptor.get_structure(trajs[-1]), Structure)


class TestAnalyzeMD(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        # create a temporary directory
        cls.temp_dir = tempfile.mkdtemp()

        # store original file paths
        cls.data_dir = "../pydmclab/data/test_data/mlp"
        cls.original_trajfile = os.path.join(cls.data_dir, "chgnet_md.traj")
        cls.original_logfile = os.path.join(cls.data_dir, "chgnet_md.log")

        # create temporary file paths
        cls.trajfile = os.path.join(cls.temp_dir, "chgnet_md.traj")
        cls.logfile = os.path.join(cls.temp_dir, "chgnet_md.log")

        # copy original files to temporary location
        shutil.copy2(cls.original_trajfile, cls.trajfile)
        shutil.copy2(cls.original_logfile, cls.logfile)

        # initialize AnalyzeMD with temporary files
        cls.amd = AnalyzeMD(
            logfile=cls.logfile, trajfile=cls.trajfile, clean_files=True
        )

    @classmethod
    def tearDownClass(cls):
        # clean up temporary directory and its contents
        shutil.rmtree(cls.temp_dir)

    def test_log_summary(self):
        summary = self.amd.log_summary
        # test the summary is a list of dictionaries
        self.assertIsInstance(summary, list)
        for entry in summary:
            self.assertIsInstance(entry, dict)
        # test the correct summary length is obtained
        self.assertEqual(len(summary), 30)
        # test the final entry of the summary is correct (time is corrected)
        self.assertAlmostEqual(summary[-1]["t"], 0.3, places=5)
        self.assertAlmostEqual(summary[-1]["T"], 942.7, places=5)
        self.assertAlmostEqual(summary[-1]["Etot"], -517.6383, places=5)
        self.assertAlmostEqual(summary[-1]["Epot"], -527.3867, places=5)
        self.assertAlmostEqual(summary[-1]["Ekin"], 9.7484, places=5)
        self.assertEqual(summary[-1]["run"], 4)
        # test transition between runs is handled correctly
        self.assertAlmostEqual(summary[5]["t"], 0.06, places=5)
        self.assertAlmostEqual(summary[5]["T"], 269.8, places=5)
        self.assertAlmostEqual(summary[5]["Etot"], -531.0874, places=5)
        self.assertAlmostEqual(summary[5]["Epot"], -533.8775, places=5)
        self.assertAlmostEqual(summary[5]["Ekin"], 2.7901, places=5)
        self.assertEqual(summary[5]["run"], 2)

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
        self.assertEqual(len(summary), 30)
        # check correct structures are being retained
        with Trajectory(self.trajfile) as trajs:
            # the 3rd (indice 2) trajectory in the trajectory file should be our 3nd entry
            trajs_2_positions = trajs[2].get_positions()
            summary_2_positions = StrucTools(summary[2]).structure.cart_coords
            self.assertTrue(
                np.allclose(trajs_2_positions, summary_2_positions, atol=1e-8)
            )
            # the 8th trajectory in the trajectory file should be our 8th entry
            traj_7_positions = trajs[7].get_positions()
            summary_7_positions = StrucTools(summary[7]).structure.cart_coords
            self.assertTrue(
                np.allclose(traj_7_positions, summary_7_positions, atol=1e-8)
            )
            # check positions are changing sufficiently between steps for previous tests to be valid
            summary_8_positions = StrucTools(summary[8]).structure.cart_coords
            self.assertFalse(
                np.allclose(traj_7_positions, summary_8_positions, atol=1e-8)
            )

    def test_full_summary(self):
        full_summary = self.amd.full_summary
        self.assertIsInstance(full_summary, list)
        for entry in full_summary:
            self.assertIsInstance(entry, dict)
        self.assertEqual(len(full_summary), 30)
        self.assertEqual(
            {*full_summary[0].keys()},
            {"t", "T", "Etot", "Epot", "Ekin", "run", "structure"},
        )
        self.assertIsInstance(
            StrucTools(full_summary[3]["structure"]).structure, Structure
        )

    def test_get_E_T_t_data(self):
        t, E, T = self.amd.get_E_T_t_data
        self.assertIsInstance(E, list)
        self.assertIsInstance(T, list)
        self.assertIsInstance(t, list)
        self.assertEqual(len(E), 30)
        self.assertEqual(len(T), 30)
        self.assertEqual(len(t), 30)
        self.assertAlmostEqual(t[-1], 0.3, places=5)
        self.assertAlmostEqual(T[-1], 942.7, places=5)
        self.assertAlmostEqual(E[-1], -527.3867, places=5)
        self.assertAlmostEqual(E[5], -533.8775, places=5)
        self.assertAlmostEqual(T[5], 269.8, places=5)

    def test_get_T_distribution_data(self):
        T, T_in = self.amd.get_T_distribution_data(remove_outliers=0.95)
        self.assertIsInstance(T, np.ndarray)
        self.assertIsInstance(T_in, np.ndarray)
        self.assertEqual(T.shape, (30,))
        self.assertEqual(T_in.shape, (18,))
        self.assertAlmostEqual(T_in[-1], 722.3, places=5)
        self.assertAlmostEqual(T[-1], 942.7, places=5)
        self.assertAlmostEqual(T[5], 269.8, places=5)

    def test_get_xrd_data(self):
        xrd_data = self.amd.get_xrd_data(data_density=0.2)
        self.assertIsInstance(xrd_data, tuple)
        self.assertIsInstance(StrucTools(xrd_data[0]).structure, Structure)
        self.assertEqual(len(xrd_data), 6)


if __name__ == "__main__":
    unittest.main()

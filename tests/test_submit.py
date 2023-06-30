import unittest

from pydmclab.hpc.submit import SubmitTools

import os

from pymatgen.io.vasp.inputs import Kpoints

test_data_dir = "../pydmclab/data/test_data/vasp/test_submit"


class UnitTestSubmitTools(unittest.TestCase):
    def test_submission(self):
        mags = ["fm", "afm_0"]
        launch_dirs = {}
        for mag in mags:
            launch_dir = os.path.join(test_data_dir, "Mn1O1", "0", "dmc", mag)
            launch_dirs[launch_dir] = {
                "xcs": ["metagga"],
                "magmom": [5.0, -5.0] if mag == "afm_0" else None,
            }

        launch_dir = [l for l in launch_dirs if "afm" in l][0]

        user_configs = {
            "loose_incar": {"ENCUT": 1234},
            "files_to_inherit": ["WAVECAR", "CONTCAR", "CHGCAR"],
            "time": 97 * 60,
            "nodes": 4,
            "partition": "msidmc",
        }
        sub = SubmitTools(
            launch_dir=launch_dir,
            final_xcs=launch_dirs[launch_dir]["xcs"],
            magmom=launch_dirs[launch_dir]["magmom"],
            user_configs=user_configs,
        )

        self.assertEqual(
            sub.vasp_configs["loose_incar"]["ENCUT"],
            user_configs["loose_incar"]["ENCUT"],
        )
        self.assertEqual(
            sub.sub_configs["files_to_inherit"], user_configs["files_to_inherit"]
        )
        self.assertEqual(sub.slurm_configs["time"], user_configs["time"])

        self.assertEqual(sub.vasp_configs["magmom"], launch_dirs[launch_dir]["magmom"])
        self.assertEqual(sub.partitions["msidmc"]["sharing"], True)

        self.assertEqual(sub.slurm_options["nodes"], 1)
        self.assertEqual(sub.slurm_options["time"], 96 * 60)
        self.assertEqual(sub.slurm_options["partition"], user_configs["partition"])

        self.assertTrue(sub.sub_configs["mpi_command"] in sub.vasp_command)


if __name__ == "__main__":
    unittest.main()

import unittest

from pydmclab.hpc.launch import LaunchTools
from pydmclab.core.struc import StrucTools

from shutil import rmtree
import os

path_to_pos = "../pydmclab/data/test_data/vasp/test_launch/POSCAR"
calcs_dir = "../pydmclab/data/test_data/vasp/test_launch/calcs"

if os.path.exists(calcs_dir):
    rmtree(calcs_dir)

if not os.path.exists(calcs_dir):
    os.mkdir(calcs_dir)


class UnitTestLaunchTools(unittest.TestCase):
    def test_launch(self):
        user_configs = {"n_afm_configs": 1, "to_launch": {"dmc": ["metagga"]}}

        top_level = "Mn1O1"
        unique_ID = "0"
        magmoms = {0: [5.0, -5.0]}

        lt = LaunchTools(
            calcs_dir=calcs_dir,
            structure=StrucTools(path_to_pos).structure,
            top_level=top_level,
            unique_ID=unique_ID,
            magmoms=magmoms,
            ID_specific_vasp_configs={"Mn1O1_0": {"loose_incar": {"NELECT": 123}}},
            user_configs=user_configs,
        )

        configs = lt.configs
        self.assertEqual(configs["n_afm_configs"], user_configs["n_afm_configs"])

        launch_dirs = lt.launch_dirs(make_dirs=True)

        self.assertEqual(len(launch_dirs), 2)

        structure_path = os.path.join(calcs_dir, top_level, unique_ID)
        for standard in user_configs["to_launch"]:
            afm_configs = [
                "afm_%s" % str(i) for i in range(user_configs["n_afm_configs"])
            ]
            mags = afm_configs + ["fm"]
            for mag in mags:
                dir_to_launch = os.path.join(structure_path, standard, mag)
                self.assertTrue(dir_to_launch in launch_dirs)
                self.assertEqual(
                    launch_dirs[dir_to_launch]["ID_specific_vasp_configs"][
                        "loose_incar"
                    ]["NELECT"],
                    123,
                )
                if "afm" in mag:
                    self.assertTrue(
                        launch_dirs[dir_to_launch]["magmom"],
                        magmoms[int(mag.split("_")[1])],
                    )
                else:
                    self.assertTrue(launch_dirs[dir_to_launch]["magmom"] is None)
                self.assertTrue(os.path.exists(dir_to_launch))
                self.assertTrue(os.path.exists(os.path.join(dir_to_launch, "POSCAR")))


if __name__ == "__main__":
    unittest.main()

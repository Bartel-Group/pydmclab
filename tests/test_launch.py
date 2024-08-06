import unittest

from pydmclab.hpc.launch import LaunchTools
from pydmclab.core.struc import StrucTools

from shutil import rmtree
import os

HERE = os.path.dirname(os.path.abspath(__file__))


class UnitTestLaunchTools(unittest.TestCase):

    def setUp(self) -> None:

        # directory to store data
        self.test_data_dir = os.path.join(
            HERE, "..", "pydmclab", "data", "test_data", "launch"
        )

        # some structure that could be AFM
        self.MnO_structure = StrucTools(
            os.path.join(self.test_data_dir, "Mn1O1.vasp")
        ).structure

        # some structure that shouldn't be magnetic
        self.AlN_structure = StrucTools(
            os.path.join(self.test_data_dir, "Al1N1.vasp")
        ).structure

        # some calcs_dirs ('.../calcs') to launch from
        calcs_dirs = ["calcs_%s" % str(i) for i in range(5)]
        calcs_dirs = [
            os.path.join(self.test_data_dir, calc_dir_name)
            for calc_dir_name in calcs_dirs
        ]
        for calcs_dir in calcs_dirs:
            if os.path.exists(calcs_dir):
                rmtree(calcs_dir)
            if not os.path.exists(calcs_dir):
                os.mkdir(calcs_dir)
        self.calcs_dirs = calcs_dirs

        return

    def test_launch(self):

        # testing simple NM, no configs
        calcs_dir = self.calcs_dirs[0]
        structure = self.AlN_structure
        formula_indicator = StrucTools(structure).compact_formula
        struc_indicator = "my-struc"
        initial_magmoms = None
        user_configs = None

        lt = LaunchTools(
            calcs_dir=calcs_dir,
            structure=structure,
            formula_indicator=formula_indicator,
            struc_indicator=struc_indicator,
            initial_magmoms=initial_magmoms,
            user_configs=user_configs,
        )

        self.assertEqual(lt.valid_mags, ["nm"])

        launch_dirs = lt.launch_dirs(make_dirs=True)

        self.assertEqual(len(launch_dirs), 1)

        launch_dir = list(launch_dirs.keys())[0]
        self.assertEqual(
            launch_dir,
            os.path.join(calcs_dir, formula_indicator, struc_indicator, "nm"),
        )

        values = launch_dirs[launch_dir]

        self.assertEqual(values["magmom"], None)
        self.assertEqual(values["ID_specific_vasp_configs"], {})

        self.assertTrue(os.path.exists(launch_dir))

        self.assertTrue(os.path.exists(os.path.join(launch_dir, "POSCAR")))

        self.assertEqual(
            len(StrucTools(self.AlN_structure).structure),
            len(StrucTools(os.path.join(launch_dir, "POSCAR")).structure),
        )

        # testing w/ AFM and vasp-specific configs
        calcs_dir = self.calcs_dirs[1]
        structure = self.MnO_structure
        formula_indicator = StrucTools(structure).compact_formula
        struc_indicator = "some-struc"
        initial_magmoms = {0: [5.0, -5.0], 1: [-5.0, 5.0]}
        ID_specific_vasp_configs = {
            "_".join([formula_indicator, struc_indicator]): {
                "incar_mods": {"NEDOS": 4321}
            }
        }
        n_afm_configs = 2
        user_configs = {
            "ID_specific_vasp_configs": ID_specific_vasp_configs,
            "n_afm_configs": n_afm_configs,
        }

        lt = LaunchTools(
            calcs_dir=calcs_dir,
            structure=structure,
            formula_indicator=formula_indicator,
            struc_indicator=struc_indicator,
            initial_magmoms=initial_magmoms,
            user_configs=user_configs,
        )

        self.assertEqual(lt.valid_mags, ["fm", "afm_0", "afm_1"])

        launch_dirs = lt.launch_dirs(make_dirs=True)

        self.assertEqual(len(launch_dirs), 3)

        for launch_dir, values in launch_dirs.items():
            mag = launch_dir.split("/")[-1]
            self.assertEqual(
                launch_dir,
                os.path.join(calcs_dir, formula_indicator, struc_indicator, mag),
            )

            if mag == "afm_0":
                true_mag = initial_magmoms[0]
            elif mag == "afm_1":
                true_mag = initial_magmoms[1]
            else:
                true_mag = None
            self.assertEqual(values["magmom"], true_mag)
            self.assertEqual(
                values["ID_specific_vasp_configs"],
                ID_specific_vasp_configs[
                    "_".join([formula_indicator, struc_indicator])
                ],
            )

            self.assertTrue(os.path.exists(launch_dir))

            self.assertTrue(os.path.exists(os.path.join(launch_dir, "POSCAR")))

            self.assertEqual(
                len(StrucTools(self.MnO_structure).structure),
                len(StrucTools(os.path.join(launch_dir, "POSCAR")).structure),
            )


if __name__ == "__main__":
    unittest.main()

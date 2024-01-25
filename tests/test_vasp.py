import unittest

from pydmclab.hpc.vasp import VASPSetUp
from pydmclab.core.struc import StrucTools

import os

from pymatgen.io.vasp.inputs import Kpoints


test_data_dir = "../pydmclab/data/test_data/vasp/test_vasp"


class UnitTestVASPSetUp(unittest.TestCase):
    def test_user_configs(self):
        calc_dir = os.path.join(
            test_data_dir, "Al1N1", "0", "dmc", "nm", "metagga-static"
        )

        user_configs = {
            "loose_incar": {"NEDOS": 1234},
            "relax_incar": {"NEDOS": 4321},
            "static_incar": {"NEDOS": 1000, "KPAR": 12},
            "loose_kpoints": {"length": 100},
            "relax_kpoints": {"reciprocal_density": 1000},
            "static_kpoints": Kpoints().as_dict(),
            "static_potcar": {"N": "N_sv"},
        }

        vsu = VASPSetUp(calc_dir, user_configs=user_configs)
        configs = vsu.configs
        for key in user_configs:
            self.assertEqual(configs[key], user_configs[key])

        user_configs["standard"] = "dmc"
        user_configs["mag"] = "nm"
        user_configs["xc_to_run"] = "metagga"
        user_configs["calc_to_run"] = "relax"
        vsu = VASPSetUp(calc_dir, user_configs=user_configs)
        vasp_input = vsu.get_vasp_input

        self.assertEqual(vasp_input.user_incar_settings["METAGGA"], "R2SCAN")
        self.assertEqual(vasp_input.user_incar_settings["NCORE"], 4)
        self.assertEqual(vasp_input.user_incar_settings["NEDOS"], 4321)

        user_configs["standard"] = "dmc"
        user_configs["mag"] = "nm"
        user_configs["xc_to_run"] = "gga"
        user_configs["calc_to_run"] = "relax"
        vsu = VASPSetUp(calc_dir, user_configs=user_configs)
        vasp_input = vsu.get_vasp_input

        self.assertEqual(vasp_input.user_incar_settings["GGA"], "PE")
        self.assertEqual(vasp_input.user_incar_settings["NCORE"], 4)
        self.assertEqual(vasp_input.user_incar_settings["NEDOS"], 4321)

        user_configs["standard"] = "dmc"
        user_configs["mag"] = "nm"
        user_configs["xc_to_run"] = "metagga"
        user_configs["calc_to_run"] = "static"
        user_configs["lobster_static"] = True
        vsu = VASPSetUp(calc_dir, user_configs=user_configs)
        vasp_input = vsu.get_vasp_input

        self.assertEqual(vasp_input.user_incar_settings["NCORE"], 4)
        self.assertEqual(vasp_input.user_incar_settings["NEDOS"], 1000)
        self.assertEqual(vasp_input.user_incar_settings["ISYM"], -1)
        self.assertEqual(vasp_input.user_incar_settings["LAECHG"], True)

    def test_perturb_struc(self):
        calc_dir = os.path.join(
            test_data_dir, "Al1N1", "0", "dmc", "nm", "metagga-static"
        )
        s_initial = StrucTools(os.path.join(calc_dir, "orig_POSCAR")).structure
        user_configs = {}
        user_configs["perturb_struc"] = 0.1
        vsu = VASPSetUp(calc_dir, user_configs=user_configs)

        s_perturbed = vsu.structure

        self.assertNotEqual(s_initial[0].coords[0], s_perturbed[0].coords[0])
        self.assertLessEqual(s_initial[1].coords[1] - s_perturbed[1].coords[1], 0.2)

    def test_prepare_calc(self):
        calc_dir = os.path.join(
            test_data_dir, "Al1N1", "0", "dmc", "nm", "metagga-static"
        )

        # print(os.listdir(calc_dir))

        user_configs = {
            "loose_incar": {"NEDOS": 1234},
            "relax_incar": {"NEDOS": 4321},
            "static_incar": {"NEDOS": 1000, "KPAR": 12},
            "loose_kpoints": {"length": 100},
            "relax_kpoints": {"reciprocal_density": 1000},
            "static_kpoints": {"length": 25},
        }

        user_configs["standard"] = "dmc"
        user_configs["mag"] = "nm"
        user_configs["xc_to_run"] = "metagga"
        user_configs["calc_to_run"] = "static"
        user_configs["lobster_static"] = True
        vsu = VASPSetUp(calc_dir, user_configs=user_configs)
        vsu.prepare_calc

        self.assertTrue(os.path.exists(os.path.join(calc_dir, "INCAR")))
        self.assertTrue(os.path.exists(os.path.join(calc_dir, "KPOINTS")))
        self.assertTrue(os.path.exists(os.path.join(calc_dir, "POTCAR")))
        self.assertTrue(os.path.exists(os.path.join(calc_dir, "lobsterin")))

        with open(os.path.join(calc_dir, "POTCAR")) as f:
            yay = 0
            for line in f:
                if "PAW_PBE Al " in line:
                    yay += 1
                if "PAW_PBE N " in line:
                    yay += 1

        self.assertGreaterEqual(yay, 2)

        with open(os.path.join(calc_dir, "KPOINTS")) as f:
            yay = 0
            for line in f:
                print(line)
                if "Automatic" in line:
                    yay += 1
                if "25" in line:
                    yay += 1

        self.assertEqual(yay, 2)

        with open(os.path.join(calc_dir, "INCAR")) as f:
            yay = 0
            for line in f:
                if "ALGO = All" in line:
                    yay += 1
                if "KPAR = 12" in line:
                    yay += 1
                if "NSW = 0" in line:
                    yay += 1

        self.assertEqual(yay, 3)

        with open(os.path.join(calc_dir, "lobsterin")) as f:
            yay = 0
            for line in f:
                if "basisfunctions Al" in line:
                    yay += 1
                if "basisfunctions N" in line:
                    yay += 1
                if "COHPSteps" in line:
                    yay += 1

        self.assertEqual(yay, 3)

    def test_error_log(self):
        calc_dir = os.path.join(
            test_data_dir, "Al1N1", "0", "dmc", "nm", "metagga-static"
        )
        vsu = VASPSetUp(
            calc_dir,
            user_configs={"mag": "nm", "calc_to_run": "static", "xc_to_run": "metagga"},
        )

        errors = vsu.error_log

        self.assertEqual(len(errors), 2)
        self.assertTrue("ibzkpt" in errors)


if __name__ == "__main__":
    unittest.main()

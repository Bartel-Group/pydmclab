import os
from pydmclab.hpc.sets import GetSet
from pydmclab.core.struc import StrucTools
from pymatgen.io.vasp.sets import MPRelaxSet, MPScanRelaxSet, MPHSERelaxSet
from pydmclab.data.configs import load_base_configs
from pymatgen.io.vasp.sets import BadInputSetWarning

import unittest
import warnings

HERE = os.path.dirname(os.path.abspath(__file__))


class UnitTestGetSet(unittest.TestCase):

    def setUp(self) -> None:
        self.test_data_dir = os.path.join(
            HERE, "..", "pydmclab", "data", "test_data", "sets"
        )

        self.structure_AlN = StrucTools(
            os.path.join(self.test_data_dir, "Al1N1.vasp")
        ).structure

        self.structure_MnO = StrucTools(
            os.path.join(self.test_data_dir, "Mn1O1.vasp")
        ).structure

        self.configs = load_base_configs()

        warnings.filterwarnings("ignore", category=BadInputSetWarning)

        return

    def test_sets(self):

        potcar_functional, validate_magmom = None, False

        configs = self.configs

        structure = self.structure_AlN
        xc_to_run, calc_to_run = "gga", "relax"
        standard, mag = "dmc", "nm"
        functional = None

        modify_incar = {}
        modify_kpoints = {}
        modify_potcar = {}

        configs["xc_to_run"], configs["calc_to_run"] = xc_to_run, calc_to_run
        configs["standard"], configs["mag"] = standard, mag
        configs["functional"] = functional

        setter = GetSet(
            structure=structure,
            configs=configs,
            potcar_functional=potcar_functional,
            validate_magmom=validate_magmom,
            modify_incar=modify_incar,
            modify_kpoints=modify_kpoints,
            modify_potcar=modify_potcar,
        )

        vaspset = setter.vaspset
        self.assertEqual(vaspset.incar["GGA"], "Pe")

        configs["xc_to_run"] = "metagga"
        setter = GetSet(
            structure=structure,
            configs=configs,
            potcar_functional=potcar_functional,
            validate_magmom=validate_magmom,
            modify_incar=modify_incar,
            modify_kpoints=modify_kpoints,
            modify_potcar=modify_potcar,
        )

        vaspset = setter.vaspset
        self.assertEqual(vaspset.incar["METAGGA"], "R2scan")

        configs["xc_to_run"] = "hse06"
        setter = GetSet(
            structure=structure,
            configs=configs,
            potcar_functional=potcar_functional,
            validate_magmom=validate_magmom,
            modify_incar=modify_incar,
            modify_kpoints=modify_kpoints,
            modify_potcar=modify_potcar,
        )

        vaspset = setter.vaspset
        self.assertEqual(vaspset.incar["LHFCALC"], True)

        configs["xc_to_run"] = "gga"
        setter = GetSet(
            structure=structure,
            configs=configs,
            potcar_functional=potcar_functional,
            validate_magmom=validate_magmom,
            modify_incar=modify_incar,
            modify_kpoints=modify_kpoints,
            modify_potcar=modify_potcar,
        )

        vaspset = setter.vaspset

        self.assertGreater(vaspset.incar["NSW"], 0)
        self.assertEqual(vaspset.incar["ISIF"], 3)
        self.assertEqual(vaspset.incar["EDIFF"], 1e-6)
        self.assertEqual(vaspset.incar["EDIFFG"], -0.03)
        self.assertEqual(vaspset.incar["KSPACING"], 0.22)
        self.assertEqual(vaspset.incar["LREAL"], False)

        configs["calc_to_run"] = "static"
        setter = GetSet(
            structure=structure,
            configs=configs,
            potcar_functional=potcar_functional,
            validate_magmom=validate_magmom,
            modify_incar=modify_incar,
            modify_kpoints=modify_kpoints,
            modify_potcar=modify_potcar,
        )
        vaspset = setter.vaspset

        self.assertEqual(vaspset.incar["NSW"], 0)

        configs["calc_to_run"] = "lobster"
        setter = GetSet(
            structure=structure,
            configs=configs,
            potcar_functional=potcar_functional,
            validate_magmom=validate_magmom,
            modify_incar=modify_incar,
            modify_kpoints=modify_kpoints,
            modify_potcar=modify_potcar,
        )
        vaspset = setter.vaspset

        self.assertEqual(vaspset.incar["ISMEAR"], -5)

        configs["calc_to_run"] = "parchg"
        setter = GetSet(
            structure=structure,
            configs=configs,
            potcar_functional=potcar_functional,
            validate_magmom=validate_magmom,
            modify_incar=modify_incar,
            modify_kpoints=modify_kpoints,
            modify_potcar=modify_potcar,
        )
        vaspset = setter.vaspset

        self.assertEqual(vaspset.incar["LPARD"], True)

        configs["xc_to_run"], configs["calc_to_run"] = "gga", "relax"

        modify_incar["EDIFF"] = 1e-4
        modify_incar["EDIFFG"] = 1e-3
        modify_potcar["Al"] = "Al_pv"
        modify_kpoints["grid"] = [2, 2, 2]

        setter = GetSet(
            structure=structure,
            configs=configs,
            potcar_functional=potcar_functional,
            validate_magmom=validate_magmom,
            modify_incar=modify_incar,
            modify_kpoints=modify_kpoints,
            modify_potcar=modify_potcar,
        )
        vaspset = setter.vaspset

        self.assertEqual(vaspset.incar["EDIFF"], 1e-4)
        self.assertEqual(vaspset.incar["EDIFFG"], 1e-3)
        self.assertEqual(vaspset._config_dict["POTCAR"]["Al"], "Al_pv")
        self.assertEqual(vaspset.kpoints.as_dict()["kpoints"], [(2, 2, 2)])


if __name__ == "__main__":
    unittest.main()

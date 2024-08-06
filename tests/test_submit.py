import unittest

from pydmclab.hpc.submit import SubmitTools
from pydmclab.core.struc import StrucTools
from pydmclab.hpc.launch import LaunchTools

import os

from pymatgen.io.vasp.inputs import Kpoints

HERE = os.path.dirname(os.path.abspath(__file__))


class UnitTestSubmitTools(unittest.TestCase):

    def setUp(self) -> None:

        # directory to store data
        self.test_data_dir = os.path.join(
            HERE, "..", "pydmclab", "data", "test_data", "submit"
        )

        # some structure that shouldn't be magnetic
        self.AlN_structure = StrucTools(
            os.path.join(self.test_data_dir, "Al1N1.vasp")
        ).structure

        calcs_dir_AlN = os.path.join(self.test_data_dir, "calcs_AlN")
        if not os.path.exists(calcs_dir_AlN):
            os.mkdir(calcs_dir_AlN)

        calcs_dir = calcs_dir_AlN
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

        launch_dirs = lt.launch_dirs(make_dirs=True)

        self.launch_dir_AlN = list(launch_dirs.keys())[0]

        # some structure that could be AFM
        self.MnO_structure = StrucTools(
            os.path.join(self.test_data_dir, "Mn1O1.vasp")
        ).structure

        calcs_dir_MnO = os.path.join(self.test_data_dir, "calcs_MnO")
        if not os.path.exists(calcs_dir_MnO):
            os.mkdir(calcs_dir_MnO)

        # testing w/ AFM and vasp-specific configs
        calcs_dir = calcs_dir_MnO
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

        self.launch_dir_MnO_afm = [k for k in launch_dirs if "afm_1" in k]

        return

    def test_submit(self):

        launch_dir = self.launch_dir_AlN
        initial_magmom = None

        relaxation_xcs = ["gga"]
        static_addons = {}
        user_configs = {
            "relaxation_xcs": relaxation_xcs,
            "static_addons": static_addons,
        }
        st = SubmitTools(
            launch_dir=launch_dir,
            initial_magmom=initial_magmom,
            user_configs=user_configs,
        )

        self.assertEqual(st.calc_list, ["gga-relax", "gga-static"])

        relaxation_xcs = ["gga", "metagga"]
        static_addons = {}
        user_configs = {
            "relaxation_xcs": relaxation_xcs,
            "static_addons": static_addons,
        }
        st = SubmitTools(
            launch_dir=launch_dir,
            initial_magmom=initial_magmom,
            user_configs=user_configs,
        )

        self.assertEqual(
            st.calc_list, ["gga-relax", "gga-static", "metagga-relax", "metagga-static"]
        )

        relaxation_xcs = ["gga", "metagga"]
        static_addons = {"gga": ["lobster"], "metagga": ["bs"]}
        user_configs = {
            "relaxation_xcs": relaxation_xcs,
            "static_addons": static_addons,
            "run_static_addons_before_all_relaxes": False,
        }
        st = SubmitTools(
            launch_dir=launch_dir,
            initial_magmom=initial_magmom,
            user_configs=user_configs,
        )

        self.assertEqual(
            st.calc_list,
            [
                "gga-relax",
                "gga-static",
                "metagga-relax",
                "metagga-static",
                "gga-prelobster",
                "gga-lobster",
                "metagga-prelobster",
                "metagga-bs",
            ],
        )

        relaxation_xcs = ["gga", "metagga"]
        static_addons = {"gga": ["lobster"], "metagga": ["bs"]}
        user_configs = {
            "relaxation_xcs": relaxation_xcs,
            "static_addons": static_addons,
            "run_static_addons_before_all_relaxes": True,
        }
        st = SubmitTools(
            launch_dir=launch_dir,
            initial_magmom=initial_magmom,
            user_configs=user_configs,
        )

        self.assertEqual(
            st.calc_list,
            [
                "gga-relax",
                "gga-static",
                "gga-prelobster",
                "gga-lobster",
                "metagga-relax",
                "metagga-static",
                "metagga-prelobster",
                "metagga-bs",
            ],
        )

        relaxation_xcs = ["gga", "metagga"]
        static_addons = {"gga": ["lobster"], "metagga": ["bs"], "hse06": ["lobster"]}
        user_configs = {
            "relaxation_xcs": relaxation_xcs,
            "static_addons": static_addons,
            "run_static_addons_before_all_relaxes": False,
        }
        st = SubmitTools(
            launch_dir=launch_dir,
            initial_magmom=initial_magmom,
            user_configs=user_configs,
        )

        self.assertEqual(
            st.calc_list,
            [
                "gga-relax",
                "gga-static",
                "metagga-relax",
                "metagga-static",
                "gga-prelobster",
                "gga-lobster",
                "metagga-prelobster",
                "metagga-bs",
                "hse06-preggastatic",
                "hse06-prelobster",
                "hse06-lobster",
            ],
        )

        relaxation_xcs = ["gga"]
        static_addons = {}
        user_configs = {
            "relaxation_xcs": relaxation_xcs,
            "static_addons": static_addons,
            "vasp_version": 6,
        }
        st = SubmitTools(
            launch_dir=launch_dir,
            initial_magmom=initial_magmom,
            user_configs=user_configs,
        )

        self.assertIn("6.4.1", st.vasp_dir)

        relaxation_xcs = ["gga"]
        static_addons = {}
        user_configs = {
            "relaxation_xcs": relaxation_xcs,
            "static_addons": static_addons,
            "vasp_version": 5,
        }
        st = SubmitTools(
            launch_dir=launch_dir,
            initial_magmom=initial_magmom,
            user_configs=user_configs,
        )

        self.assertIn("5.4.4", st.vasp_dir)

        relaxation_xcs = ["gga"]
        static_addons = {}
        user_configs = {
            "relaxation_xcs": relaxation_xcs,
            "static_addons": static_addons,
            "mpi_command": "srun",
        }
        st = SubmitTools(
            launch_dir=launch_dir,
            initial_magmom=initial_magmom,
            user_configs=user_configs,
        )

        self.assertIn("--ntasks", st.vasp_command)

        relaxation_xcs = ["gga"]
        static_addons = {}
        user_configs = {
            "relaxation_xcs": relaxation_xcs,
            "static_addons": static_addons,
            "mpi_command": "mpirun",
        }
        st = SubmitTools(
            launch_dir=launch_dir,
            initial_magmom=initial_magmom,
            user_configs=user_configs,
        )

        self.assertIn("-np", st.vasp_command)

        self.assertEqual(st.job_name.count("."), 3)
        self.assertEqual(st.job_name.split(".")[-2], "nm")

        # TODO: everything requiring queue checking can only be run on HPC..

    def _old_test_submission(self):
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

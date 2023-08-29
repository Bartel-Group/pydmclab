import unittest

from pydmclab.hpc.helpers import (
    get_vasp_configs,
    get_slurm_configs,
    get_sub_configs,
    get_launch_configs,
)
import multiprocessing as multip


class UnitTestHelpers(unittest.TestCase):
    def test_get_vasp_configs(self):
        configs = get_vasp_configs(
            run_lobster=False,
            detailed_dos=False,
            modify_loose_incar=False,
            modify_relax_incar=False,
            modify_static_incar=False,
            modify_loose_kpoints=False,
            modify_relax_kpoints=False,
            modify_static_kpoints=False,
            modify_loose_potcar=False,
            modify_relax_potcar=False,
            modify_static_potcar=False,
        )

        self.assertEqual(configs["lobster_static"], False)

        configs = get_vasp_configs(
            run_lobster=True,
            detailed_dos=True,
            modify_loose_incar={"NEDOS": 1234, "LCHARG": True},
            modify_relax_incar=False,
            modify_static_incar=False,
            modify_loose_kpoints=False,
            modify_relax_kpoints={"reciprocal_density": 100},
            modify_static_kpoints=False,
            modify_loose_potcar=False,
            modify_relax_potcar=False,
            modify_static_potcar={"Al": "Al_pv"},
        )

        self.assertEqual(configs["lobster_static"], True)
        self.assertEqual(configs["COHPSteps"], 4000)
        self.assertEqual(configs["loose_incar"], {"NEDOS": 1234, "LCHARG": True})
        self.assertEqual(configs["relax_kpoints"], {"reciprocal_density": 100})
        self.assertEqual(configs["static_potcar"], {"Al": "Al_pv"})

    def test_get_slurm_configs(self):
        total_nodes = 2
        cores_per_node = 16
        walltime_in_hours = 23
        mem_per_core = "all"
        partition = "agsmall"
        error_file = "log.e"
        output_file = "log.o"
        account = "cbartel"

        slurm_configs = get_slurm_configs(
            total_nodes=total_nodes,
            cores_per_node=cores_per_node,
            walltime_in_hours=walltime_in_hours,
            mem_per_core=mem_per_core,
            partition=partition,
            error_file=error_file,
            output_file=output_file,
            account=account,
        ).copy()

        self.assertEqual(slurm_configs["nodes"], total_nodes)
        self.assertEqual(slurm_configs["ntasks"], cores_per_node * total_nodes)
        self.assertEqual(slurm_configs["time"], walltime_in_hours * 60)
        self.assertEqual(slurm_configs["mem-per-cpu"], "4000M")
        self.assertEqual(slurm_configs["partition"], "aglarge")

        partition = "agsmall,msidmc"
        total_nodes = 1
        slurm_configs = get_slurm_configs(
            total_nodes=total_nodes,
            cores_per_node=cores_per_node,
            walltime_in_hours=walltime_in_hours,
            mem_per_core=mem_per_core,
            partition=partition,
            error_file=error_file,
            output_file=output_file,
            account=account,
        ).copy()
        self.assertEqual(slurm_configs["mem-per-cpu"], "4000M")
        self.assertEqual(slurm_configs["partition"], "agsmall,msidmc")

        partition = "RM-shared"
        cores_per_node = 32
        slurm_configs = get_slurm_configs(
            total_nodes=total_nodes,
            cores_per_node=cores_per_node,
            walltime_in_hours=walltime_in_hours,
            mem_per_core=mem_per_core,
            partition=partition,
            error_file=error_file,
            output_file=output_file,
            account=account,
        ).copy()
        self.assertEqual(slurm_configs["mem-per-cpu"], "2000M")

    def test_get_sub_configs(self):
        machine = "msi"
        parallel = True
        start_over = False
        force_postprocess = False
        mpi_command = "mpirun"
        special_packing = False

        sub_configs = get_sub_configs(
            machine=machine,
            submit_calculations_in_parallel=parallel,
            delete_all_calculations_and_start_over=start_over,
            rerun_lobster=force_postprocess,
            mpi_command=mpi_command,
            special_packing=special_packing,
        )

        self.assertEqual(sub_configs["machine"], machine)
        self.assertEqual(sub_configs["n_procs"], multip.cpu_count() - 1)
        self.assertEqual(sub_configs["mpi_command"], mpi_command)

        parallel = 4
        sub_configs = get_sub_configs(
            machine=machine,
            submit_calculations_in_parallel=parallel,
            delete_all_calculations_and_start_over=start_over,
            rerun_lobster=force_postprocess,
            mpi_command=mpi_command,
            special_packing=special_packing,
        )

        self.assertEqual(sub_configs["n_procs"], parallel)

        parallel = False
        sub_configs = get_sub_configs(
            machine=machine,
            submit_calculations_in_parallel=parallel,
            delete_all_calculations_and_start_over=start_over,
            rerun_lobster=force_postprocess,
            mpi_command=mpi_command,
            special_packing=special_packing,
        )

        self.assertEqual(sub_configs["n_procs"], 1)

        special_packing = {
            "metagga": ["metagga-loose", "metagga-banana", "metagga-apple"]
        }
        sub_configs = get_sub_configs(
            machine=machine,
            submit_calculations_in_parallel=parallel,
            delete_all_calculations_and_start_over=start_over,
            rerun_lobster=force_postprocess,
            mpi_command=mpi_command,
            special_packing=special_packing,
        )

        self.assertEqual(sub_configs["packing"], special_packing)

        force_postprocess = True
        start_over = True
        sub_configs = get_sub_configs(
            machine=machine,
            submit_calculations_in_parallel=parallel,
            delete_all_calculations_and_start_over=start_over,
            rerun_lobster=force_postprocess,
            mpi_command=mpi_command,
            special_packing=special_packing,
        )

        self.assertEqual(sub_configs["force_postprocess"], force_postprocess)
        self.assertEqual(sub_configs["fresh_restart"], start_over)

    def test_get_launch_configs(self):
        standards = ["dmc"]
        xcs = ["metagga"]
        use_mp_thermo_data = False
        n_afm_configs = 0
        skip_xcs_for_standards = {"mp": ["gga", "metagga"]}

        launch_configs = get_launch_configs(
            standards=standards,
            xcs=xcs,
            use_mp_thermo_data=use_mp_thermo_data,
            n_afm_configs=n_afm_configs,
            skip_xcs_for_standards=skip_xcs_for_standards,
        )

        self.assertEqual(launch_configs["to_launch"], {"dmc": ["metagga"]})
        self.assertEqual(launch_configs["n_afm_configs"], n_afm_configs)

        use_mp_thermo_data = True
        launch_configs = get_launch_configs(
            standards=standards,
            xcs=xcs,
            use_mp_thermo_data=use_mp_thermo_data,
            n_afm_configs=n_afm_configs,
            skip_xcs_for_standards=skip_xcs_for_standards,
        )

        self.assertEqual(
            launch_configs["to_launch"], {"dmc": ["metagga"], "mp": ["ggau"]}
        )

        standards = ["dmc", "abc"]
        xcs = ["metagga", "hse"]
        use_mp_thermo_data = False
        skip_xcs_for_standards = {"abc": ["metagga"]}
        launch_configs = get_launch_configs(
            standards=standards,
            xcs=xcs,
            use_mp_thermo_data=use_mp_thermo_data,
            n_afm_configs=n_afm_configs,
            skip_xcs_for_standards=skip_xcs_for_standards,
        )

        self.assertEqual(
            launch_configs["to_launch"], {"dmc": ["metagga", "hse"], "abc": ["hse"]}
        )


if __name__ == "__main__":
    unittest.main()

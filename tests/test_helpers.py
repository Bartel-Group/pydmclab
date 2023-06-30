import unittest

from pydmclab.hpc.helpers import get_vasp_configs, get_slurm_configs


class UnitTestHelpers(unittest.TestCase):
    def test_vasp_configs(self):
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
        )

        self.assertEqual(slurm_configs["nodes"], total_nodes)
        self.assertEqual(slurm_configs["ntasks"], cores_per_node * total_nodes)
        self.assertEqual(slurm_configs["time"], walltime_in_hours * 60)
        self.assertEqual(slurm_configs["mem-per-cpu"], "4000M")
        self.assertEqual(slurm_configs["partition"], "aglarge")

        partition = "agsmall,msidmc"
        slurm_configs = get_slurm_configs(
            total_nodes=total_nodes,
            cores_per_node=cores_per_node,
            walltime_in_hours=walltime_in_hours,
            mem_per_core=mem_per_core,
            partition=partition,
            error_file=error_file,
            output_file=output_file,
            account=account,
        )
        self.assertEqual(slurm_configs["mem-per-cpu"], "4000M")

        partition = "RM-shared"
        total_nodes = 1
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
        )
        self.assertEqual(slurm_configs["mem-per-cpu"], "1900M")


if __name__ == "__main__":
    unittest.main()

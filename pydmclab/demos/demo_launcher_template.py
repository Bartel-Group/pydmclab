import os

from pydmclab.hpc.helpers import (
    get_query,
    check_query,
    get_strucs,
    check_strucs,
    get_magmoms,
    check_magmoms,
    get_launch_dirs,
    check_launch_dirs,
    submit_calcs,
    get_results,
    check_results,
    get_gs,
    check_gs,
    get_Efs,
    check_Efs,
    get_thermo_results,
    check_thermo_results,
    get_launch_configs,
    get_sub_configs,
    get_slurm_configs,
    get_vasp_configs,
    get_analysis_configs,
    make_sub_for_launcher,
)

"""
see [pydmclab docs](https://github.umn.edu/bartel-group/pydmclab/blob/main/docs.md) for help
"""

# where is this file
SCRIPTS_DIR = os.getcwd()

# where are my calculations going to live
CALCS_DIR = SCRIPTS_DIR.replace("scripts", "calcs")

# where is my data going to live
DATA_DIR = SCRIPTS_DIR.replace("scripts", "data")

for d in [CALCS_DIR, DATA_DIR]:
    if not os.path.exists(d):
        os.makedirs(d)

# if you need data from MP as a starting point (often the case), you need your API key
API_KEY = "__YOUR API KEY__"

# what to query MP for
## e.g., 'MnO2', ['MnO2', 'TiO2'], 'Ca-Ti-O, etc
COMPOSITIONS = None

# any configurations related to LaunchTools
LAUNCH_CONFIGS = get_launch_configs(
    standards=["dmc"],
    xcs=["metagga"],
    use_mp_thermo_data=False,
    n_afm_configs=0,
    skip_xcs_for_standards={"mp": ["gga", "metagga"]},
)

# any configurations related to SubmitTools
SUB_CONFIGS = get_sub_configs(
    submit_calculations_in_parallel=False,
    rerun_lobster=False,
    mpi_command="mpirun",
    special_packing=False,
    start_all_calculations_from_scratch=False,
)

# any configurations related to Slurm
SLURM_CONFIGS = get_slurm_configs(
    total_nodes=1,
    cores_per_node=8,
    walltime_in_hours=95,
    mem_per_core="all",
    partition="agsmall,msidmc",
    error_file="log.e",
    output_file="log.o",
    account="cbartel",
)

# any configurations related to VASPSetUp
VASP_CONFIGS = get_vasp_configs(
    run_lobster=False,
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

# any configurations related to AnalyzeBatch
ANALYSIS_CONFIGS = get_analysis_configs(
    analyze_calculations_in_parallel=False,
    analyze_structure=True,
    analyze_mag=False,
    analyze_charge=False,
    analyze_dos=False,
    analyze_bonding=False,
    exclude=[],
)

# whether or not you want to generate MAGMOMs
## True if you're running AFM, else False
GEN_MAGMOMS = True if LAUNCH_CONFIGS["n_afm_configs"] else False


def main():
    remake_sub_for_launcher = False

    remake_query = False
    print_query_check = True

    remake_strucs = False
    print_strucs_check = True

    remake_magmoms = False
    print_magmoms_check = True

    remake_launch_dirs = False
    print_launch_dirs_check = True

    remake_subs = True
    ready_to_launch = True

    remake_results = True
    print_results_check = True

    remake_gs = True
    print_gs_check = True

    remake_Efs = True
    print_Efs_check = True

    remake_thermo_results = True
    print_thermo_results_check = True

    if remake_sub_for_launcher:
        make_sub_for_launcher()

    comp = COMPOSITIONS
    query = get_query(comp=comp, data_dir=DATA_DIR, remake=remake_query)
    if print_query_check:
        check_query(query)

    strucs = get_strucs(query=query, data_dir=DATA_DIR, remake=remake_strucs)
    if print_strucs_check:
        check_strucs(strucs)

    if GEN_MAGMOMS:
        magmoms = get_magmoms(strucs=strucs, data_dir=DATA_DIR, remake=remake_magmoms)
        if print_magmoms_check:
            check_magmoms(strucs=strucs, magmoms=magmoms)
    else:
        magmoms = None

    launch_configs = LAUNCH_CONFIGS
    launch_dirs = get_launch_dirs(
        strucs=strucs,
        magmoms=magmoms,
        user_configs=launch_configs,
        data_dir=DATA_DIR,
        remake=remake_launch_dirs,
    )
    if print_launch_dirs_check:
        check_launch_dirs(launch_dirs)

    sub_configs = SUB_CONFIGS
    slurm_configs = SLURM_CONFIGS
    vasp_configs = VASP_CONFIGS
    user_sub_configs = {**sub_configs, **slurm_configs, **vasp_configs}
    if remake_subs:
        submit_calcs(
            launch_dirs=launch_dirs,
            user_configs=user_sub_configs,
            ready_to_launch=ready_to_launch,
            n_procs=sub_configs["n_procs"],
        )

    analysis_configs = ANALYSIS_CONFIGS
    results = get_results(
        launch_dirs=launch_dirs,
        user_configs=analysis_configs,
        data_dir=DATA_DIR,
        remake=remake_results,
    )
    if print_results_check:
        check_results(results)

    gs = get_gs(results=results, data_dir=DATA_DIR, remake=remake_gs)

    if print_gs_check:
        check_gs(gs)

    Efs = get_Efs(gs=gs, data_dir=DATA_DIR, remake=remake_Efs)

    if print_Efs_check:
        check_Efs(Efs)

    thermo = get_thermo_results(
        results=results, Efs=Efs, data_dir=DATA_DIR, remake=remake_thermo_results
    )

    if print_thermo_results_check:
        check_thermo_results(thermo)

    return


if __name__ == "__main__":
    main()

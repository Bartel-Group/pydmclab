import sys
import os

from pydmclab.utils.handy import read_json
from pydmclab.hpc.helpers import check_strucs, check_launch_dirs

HOME_PATH = os.environ["HOME"]
MD_HELPERS_DIR = "%s/bin/pydmclab/hpc_workflows/md_calcs" % HOME_PATH

if MD_HELPERS_DIR not in sys.path:
    sys.path.append(MD_HELPERS_DIR)

from md_helpers import (
    # get_chgnet_configs,
    get_fairchem_configs,
    get_slurm_configs,
    get_torch_configs,
    make_launch_dirs,
    make_md_scripts,
    make_submission_scripts,
    submit_jobs,
    collect_results,
    check_collected_results,
    lowest_energy_struc_results,
)

# where is this file
SCRIPTS_DIR = os.getcwd()

# where are my calculations going to live (maybe on scratch)
CALCS_DIR = SCRIPTS_DIR.replace("scripts", "calcs")

# where is my data going to live
DATA_DIR = SCRIPTS_DIR.replace("scripts", "data")

# make our calcs dir and data dir
for d in [CALCS_DIR, DATA_DIR]:
    if not os.path.exists(d):
        os.makedirs(d)

# set chgnet molecular dynamics configs
ARCHITECTURE_CONFIGS = get_fairchem_configs(
    name_or_path="uma-s-1",
    task_name="omat",
    ensembles=("nvt",),
    thermostats=("nh",),
    starting_temperature=None,
    taut=100,
    timestep=2.0,
    loginterval=100,
    nsteps=50000,
    temperatures=(1200.0,),
    pressure=1.01325e-4,
    addn_args={},
)

# set slurm submission script configs
SLURM_CONFIGS = get_slurm_configs(
    total_nodes=1,
    tasks_per_node=1,
    cores_per_task=8,
    walltime_in_hours=23,
    mem_per_core_in_MB=3900,
    partition="preempt,msismall,msidmc",
    error_file="log.e",
    output_file="log.o",
    account="cbartel",
)

# set torch configs
TORCH_CONFIGS = get_torch_configs(
    slurm_configs=SLURM_CONFIGS, num_intraop_threads=None, num_interop_threads=None
)

# collect all user configs
USER_CONFIGS = {**ARCHITECTURE_CONFIGS, **SLURM_CONFIGS, **TORCH_CONFIGS}

# location of md_template.py
MD_TEMPLATE = "%s/bin/pydmclab/hpc_workflows/md_calcs/md_template.py" % HOME_PATH


def main():

    # print strucs summary?
    print_strucs_check = True

    # remake launch directories? print launch_dirs summary?
    remake_launch_dirs = False
    print_launch_dirs_check = True

    # remake md scripts?
    remake_md_scripts = False

    # remake submission scripts?
    remake_subs = False

    # submit jobs?
    ready_to_launch = False

    # remake results? print results summary?
    remake_results = True
    print_results_check = True

    # prepare strucs.json for future use?
    prepare_future_strucs = True
    remake_prepare_future_strucs = True
    print_future_strucs_check = False

    # ----------------------------

    # retrieve structures
    strucs = read_json(os.path.join(DATA_DIR, "strucs.json"))
    if print_strucs_check:
        check_strucs(strucs)

    # get existing or make new launch directories
    launch_dirs = make_launch_dirs(
        strucs=strucs,
        user_configs=USER_CONFIGS,
        calcs_dir=CALCS_DIR,
        data_dir=DATA_DIR,
        remake=remake_launch_dirs,
    )
    if print_launch_dirs_check:
        check_launch_dirs(launch_dirs)

    # write md scripts in each launch directory
    make_md_scripts(
        launch_dirs=launch_dirs,
        user_configs=USER_CONFIGS,
        md_template=MD_TEMPLATE,
        remake=remake_md_scripts,
    )

    # write submission scripts in each launch directory
    make_submission_scripts(
        launch_dirs=launch_dirs, user_configs=USER_CONFIGS, remake=remake_subs
    )

    # submit jobs
    if ready_to_launch:
        submit_jobs(
            launch_dirs=launch_dirs,
            user_configs=USER_CONFIGS,
        )

    # collect results
    results = collect_results(
        launch_dirs=launch_dirs,
        user_configs=USER_CONFIGS,
        data_dir=DATA_DIR,
        remake=remake_results,
    )
    if print_results_check:
        check_collected_results(results=results, launch_dirs=launch_dirs)

    # prepare strucs.json for future use (such as for relaxation via CHGNet or DFT)
    # these will be the lowest energy structures sampled throughout the MD simulation
    if prepare_future_strucs:
        future_strucs = lowest_energy_struc_results(
            results=results,
            time_to_consider=0.5,
            num_strucs=10,
            data_dir=DATA_DIR,
            remake=remake_prepare_future_strucs,
        )
        if print_future_strucs_check:
            check_strucs(future_strucs)

    return


if __name__ == "__main__":
    main()

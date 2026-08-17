import sys
import os

from pydmclab.utils.handy import read_json
from pydmclab.hpc.helpers import check_strucs, check_launch_dirs


HOME_PATH = os.environ["HOME"]
RELAX_HELPERS_DIR = "%s/bin/pydmclab/hpc_workflows/relax_calcs" % HOME_PATH

if RELAX_HELPERS_DIR not in sys.path:
    sys.path.append(RELAX_HELPERS_DIR)

from relax_helpers import (
    # get_chgnet_configs,
    get_fairchem_configs,
    get_launch_configs,
    get_slurm_configs,
    get_torch_configs,
    setup_job,
    make_relax_scripts,
    make_submission_scripts,
    submit_jobs,
    collect_results,
    check_collected_results,
)

# where is this file
SCRIPTS_DIR = os.getcwd()

# where are my calculations going to live (same as scripts)
CALCS_DIR = SCRIPTS_DIR.replace("scripts", "calcs")

# alternatively can have calculations live on scratch
# _, _, _, USER_NAME = HOME_PATH.split("/")
# SCRATCH_PATH = os.path.join(os.environ["SCRATCH_GLOBAL"], USER_NAME)
# CALCS_DIR = SCRIPTS_DIR.replace("scripts", "calcs").replace(HOME_PATH, SCRATCH_PATH)

# where is my data going to live
DATA_DIR = SCRIPTS_DIR.replace("scripts", "data")

# make our calcs dir and data dir
for d in [CALCS_DIR, DATA_DIR]:
    if not os.path.exists(d):
        os.makedirs(d)

# set architecture configs using get_{architecture}_configs
#   these are model specific and can vary widely
ARCHITECTURE_CONFIGS = get_fairchem_configs(
    name_or_path="uma-s-1",
    task_name="omat",
    inference_settings="default",
    overrides=None,
    optimizer="FIRE",
    fmax=0.03,
    steps=500,
    relax_cell=True,
    ase_filter="FrechetCellFilter",
    params_asefilter=None,
    interval=1,
    verbose=False,
)

# set launch configs
LAUNCH_CONFIGS = get_launch_configs(batch_size=100, batch_id=0, save_interval=10)

# set slurm submission script configs
SLURM_CONFIGS = get_slurm_configs(
    total_nodes=1,
    tasks_per_node=1,
    cores_per_task=8,
    walltime_in_hours=12,
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
USER_CONFIGS = {
    **ARCHITECTURE_CONFIGS,
    **LAUNCH_CONFIGS,
    **SLURM_CONFIGS,
    **TORCH_CONFIGS,
}

# location of relax_template.py
RELAX_TEMPLATE = (
    "%s/bin/pydmclab/hpc_workflows/relax_calcs/relax_template.py" % HOME_PATH
)


def main():

    # print strucs summary?
    print_strucs_check = True

    # re-run setup? print setup summary?
    rerun_setup = False
    print_launch_dirs_check = True

    # remake relax scripts?
    remake_relax_scripts = False

    # remake submission scripts?
    remake_subs = False

    # submit jobs?
    ready_to_launch = False

    # remake results? print results summary?
    remake_results = True
    print_results_check = True

    # ----------------------------

    # retrieve structures
    strucs = read_json(os.path.join(DATA_DIR, "strucs.json"))
    if print_strucs_check:
        check_strucs(strucs)

    # run batching and directory setup
    # if rerun_setup = True, will completely redo batching
    #   and directory setup (all data in calcs will be lost)
    batching = setup_job(
        strucs=strucs,
        user_configs=USER_CONFIGS,
        calcs_dir=CALCS_DIR,
        data_dir=DATA_DIR,
        savename="batching.json",
        rerun=rerun_setup,
    )
    if print_launch_dirs_check:
        launch_dirs = [batching[batch_id]["launch_dir"] for batch_id in batching]
        check_launch_dirs(launch_dirs)

    # write relax scripts in each launch directory
    make_relax_scripts(
        batching=batching,
        user_configs=USER_CONFIGS,
        relax_template=RELAX_TEMPLATE,
        remake=remake_relax_scripts,
    )

    # write submission scripts in each launch directory
    make_submission_scripts(
        batching=batching, user_configs=USER_CONFIGS, remake=remake_subs
    )

    # submit jobs
    if ready_to_launch:
        submit_jobs(batching=batching, user_configs=USER_CONFIGS)

    # collect results
    results = collect_results(
        batching=batching,
        user_configs=USER_CONFIGS,
        data_dir=DATA_DIR,
        remake=remake_results,
    )
    if print_results_check:
        check_collected_results(results=results, batching=batching)

    return


if __name__ == "__main__":
    main()

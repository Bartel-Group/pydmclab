import sys
import os

from pydmclab.utils.handy import read_json
from pydmclab.hpc.helpers import check_strucs, check_launch_dirs

HOME_PATH = os.environ["HOME"]
PREDICTION_HELPERS_DIR = "%s/bin/pydmclab/hpc_workflows/prediction_calcs" % HOME_PATH

if PREDICTION_HELPERS_DIR not in sys.path:
    sys.path.append(PREDICTION_HELPERS_DIR)

from prediction_helpers import (
    get_chgnet_configs,
    get_mace_configs,
    get_launch_configs,
    get_slurm_configs,
    get_torch_configs,
    setup_job,
    make_prediction_scripts,
    make_submission_scripts,
    submit_jobs,
    collect_results,
    check_collected_results,
)

# set up some paths that will point to where your data/calculations will live
#  these are just defaults, you can change the `_DIR` variables to point to wherever you want
#
# The home directory path is used to point to your local copy of the pydmclab repo
#   pydmclab is assumed to be in /users/{number}/{username}/bin/pydmclab
#   and $HOME points to /users/{number}/{username}
_, _, _, USER_NAME = HOME_PATH.split("/")
SCRATCH_PATH = os.path.join(os.environ["SCRATCH_GLOBAL"], USER_NAME)

# where is this file
SCRIPTS_DIR = os.getcwd()

# where are my calculations going to live (defaults to scratch)
CALCS_DIR = SCRIPTS_DIR.replace("scripts", "calcs").replace(HOME_PATH, SCRATCH_PATH)

# where is my data going to live
DATA_DIR = SCRIPTS_DIR.replace("scripts", "data")

# make our calcs dir and data dir
for d in [CALCS_DIR, DATA_DIR]:
    if not os.path.exists(d):
        os.makedirs(d)

# set architecture configs using get_{architecture}_configs
#   these are model specific and can vary widely
ARCHITECTURE_CONFIGS = get_chgnet_configs(
    model="0.3.0",
    stress_weight=1 / 160.21766208,
    on_isolated_atoms="warn",
    task="efsm",
    return_site_energies=False,
    return_atom_feas=False,
    return_crystal_feas=False,
    batch_size=16,
    verbose=True,
)

# set launch configs
LAUNCH_CONFIGS = get_launch_configs(batch_size=300, save_interval=25)

# set slurm submission script configs
SLURM_CONFIGS = get_slurm_configs(
    total_nodes=1,
    cores_per_node=16,
    walltime_in_hours=12,
    mem_per_core_in_MB=3900,
    partition="msismall,msidmc",
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

# location of prediction_template.py
PREDICTION_TEMPLATE = (
    "%s/bin/pydmclab/hpc_workflows/prediction_calcs/prediction_template.py" % HOME_PATH
)


def main():

    # print strucs summary?
    print_strucs_check = True

    # re-run setup? print setup summary?
    rerun_setup = False
    print_launch_dirs_check = True

    # remake prediction scripts?
    remake_prediction_scripts = False

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

    # write prediction scripts in each launch directory
    make_prediction_scripts(
        batching=batching,
        user_configs=USER_CONFIGS,
        architecture_configs=ARCHITECTURE_CONFIGS,
        prediction_template=PREDICTION_TEMPLATE,
        remake=remake_prediction_scripts,
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
        architecture_configs=ARCHITECTURE_CONFIGS,
        data_dir=DATA_DIR,
        remake=remake_results,
    )
    if print_results_check:
        check_collected_results(results=results, batching=batching)

    return


if __name__ == "__main__":
    main()

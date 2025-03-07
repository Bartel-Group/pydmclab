from __future__ import annotations
from typing import TYPE_CHECKING, Literal
import os
import shutil
import subprocess
from pydmclab.mlp.dynamics import CHGNet, CHGNetCalculator
from pydmclab.utils.handy import read_json, write_json

if TYPE_CHECKING:
    from pydmclab.mlp import Versions
    from pydmclab.mlp.dynamics import CHGNet, CHGNetCalculator
    from ase.optimize.optimize import Optimizer as ASEOptimizer


def get_relax_configs(
    model: CHGNet | CHGNetCalculator | Versions | None = None,
    optimizer: ASEOptimizer | str = "FIRE",
    stress_weight: float | None = 1 / 160.21766208,
    on_isolated_atoms: Literal["ignore", "warn", "error"] = "warn",
    fmax: float | None = 0.1,
    steps: int | None = 500,
    relax_cell: bool | None = True,
    ase_filter: str | None = "FrechetCellFilter",
    params_asefilter: dict | None = None,
    interval: int | None = 1,
    verbose: bool = False,
):
    """
    Note: this assumes cpu only use on MSI

    Args:
        model: if None, uses model "0.3.0"
        optimizer: default is "FIRE", see pydmclab.mlp.dynamics for more options
        stress_weight: the conversion factor to convert GPa to eV/A^3
        on_isolated_atoms: what to do if isolated atoms are found
        fmax: the force convergence criterion
        steps: the maximum number of steps to try during relaxation
        relax_cell: whether to relax the cell (False is equivalent to ISIF = 2)
        ase_filter: the ASE filter to use
        params_asefilter: the parameters for the ASE filter
        interval: logging interval
        verbose: if True, prints relaxation information

    Returns:
        relax_configs (dict): dict of relaxation configurations
    """

    relax_configs = {}

    relax_configs["model"] = model
    relax_configs["optimizer"] = optimizer
    relax_configs["stress_weight"] = stress_weight
    relax_configs["on_isolated_atoms"] = on_isolated_atoms
    relax_configs["fmax"] = fmax
    relax_configs["steps"] = steps
    relax_configs["relax_cell"] = relax_cell
    relax_configs["ase_filter"] = ase_filter
    relax_configs["params_asefilter"] = params_asefilter
    relax_configs["relax_interval"] = interval
    relax_configs["verbose"] = verbose

    return relax_configs


def get_launch_configs(batch_size: int = 100, save_interval: int = 5):
    """
    Args:
        batch_size (int): the number of structures to relax per job
        save_interval (int): how often to save the relaxation results
            e.g. if save_interval = 5, then the relaxation results are saved every 5 structures

    Returns:
        launch_configs (dict): dict of launch configurations
    """

    if batch_size < 1 or save_interval < 1:
        raise ValueError(
            "batch_size and save_interval must be equal to or greater than 1"
        )
    if not isinstance(batch_size, int) or not isinstance(save_interval, int):
        raise TypeError("batch_size and save_interval must be integers")
    if save_interval > batch_size:
        raise ValueError("save_interval must be less than or equal to batch_size")

    launch_configs = {}

    launch_configs["batch_size"] = batch_size
    launch_configs["save_interval"] = save_interval

    return launch_configs


def get_slurm_configs(
    total_nodes: int = 1,
    cores_per_node: int = 8,
    walltime_in_hours: int = 12,
    mem_per_core_in_MB: int = 1900,
    partition: str = "agsmall, msismall, msidmc",
    error_file: str = "log.e",
    output_file: str = "log.o",
    account: str = "cbartel",
) -> dict:
    """
    Args:
        total_nodes (int): number of nodes
        cores_per_node (int): number of cores per node
        walltime_in_hours (int): walltime in hours
        mem_per_core_in_MB (int): memory per core in MB
        partition (str): partition
        error_file (str): error file
        output_file (str): output file
        account (str): account

    Returns:
        slurm_configs (dict): dict of SLURM configurations
    """

    if total_nodes > 1:
        raise NotImplementedError("more than one node not yet implemented")

    slurm_configs = {}

    slurm_configs["nodes"] = total_nodes
    slurm_configs["ntasks"] = int(total_nodes * cores_per_node)
    slurm_configs["time"] = int(walltime_in_hours * 60)
    slurm_configs["mem_per_core"] = str(int(mem_per_core_in_MB)) + "M"
    slurm_configs["partition"] = partition
    slurm_configs["error_file"] = error_file
    slurm_configs["output_file"] = output_file
    slurm_configs["account"] = account

    return slurm_configs


def get_torch_configs(
    slurm_configs: dict, num_intraop_threads: int = 4, num_interop_threads: int = 4
) -> dict:
    """
    Args:
        slurm_configs (dict): dict of SLURM configurations
        num_intraop_threads (int): number of intra-op threads
        num_interop_threads (int): number of inter-op threads

    Returns:
        torch_configs (dict): dict of torch configurations
    """

    if num_intraop_threads > slurm_configs["ntasks"]:
        raise ValueError("num_intraop_threads must be less than or equal to ntasks")
    if num_interop_threads > slurm_configs["ntasks"]:
        raise ValueError("num_interop_threads must be less than or equal to ntasks")

    torch_configs = {}
    torch_configs["num_intraop_threads"] = num_intraop_threads
    torch_configs["num_interop_threads"] = num_interop_threads

    return torch_configs


def batch_strucs(strucs: dict, batch_size: int) -> dict:
    """
    Args:
        strucs (dict): {formula: {struc_id: {Structure.as_dict()}}}
        batch_size (int): the number of structures to relax per job

    Returns:
        batched_strucs (dict): {batch_id: {formula_struc_id: {Structure.as_dict()}}}
    """

    batch_id = 0
    batched_strucs = {}
    current_batch = {}

    for formula in strucs:
        for struc_id, struc in strucs[formula].items():
            current_batch[f"{formula}_{struc_id}"] = struc
            if len(current_batch) == batch_size:
                batched_strucs[f"batch_{batch_id}"] = current_batch
                batch_id += 1
                current_batch = {}

    if current_batch:
        batched_strucs[f"batch_{batch_id}"] = current_batch

    return batched_strucs


def make_launch_dirs(batched_strucs: dict, calcs_dir: str) -> dict:
    """
    Makes directories for each batch

    Args:
        batched_strucs (dict): {batch_id: {formula_struc_id: {Structure.as_dict()}}}
        calcs_dir (str): path to calculations directory

    Returns:
        batching (dict): {batch_id: {"strucs": {formula_struc_id: {Structure.as_dict()}}, "launch_dir": str}}
    """

    batching = {}

    for batch_id, batch in batched_strucs.items():
        launch_dir = os.path.join(calcs_dir, batch_id)
        if os.path.exists(launch_dir):
            shutil.rmtree(launch_dir)
        os.makedirs(launch_dir)
        write_json(batch, os.path.join(launch_dir, "ini_strucs.json"))
        batching[batch_id] = {"strucs": batch, "launch_dir": launch_dir}

    return batching


def setup_job(
    strucs: dict,
    user_configs: dict,
    calcs_dir: str,
    data_dir: str,
    savename: str = "batching.json",
    rerun: bool = False,
) -> dict:
    """
    Args:
        strucs (dict): {formula: {struc_id: {Structure.as_dict()}}}
        user_configs (dict): user configs
        calcs_dir (str): path to calculations directory
        data_dir (str): path to data directory
        savename (str): name of json file to record batching
        rerun (bool): if True, will rebatch and re-setup directories

    Returns:
        batching (dict): {"batch_id": {"strucs": {formula_struc_id: {Structure.as_dict()}}, "launch_dir": str}}
    """

    fjson = os.path.join(data_dir, savename)
    if os.path.exists(fjson) and not rerun:
        return read_json(fjson)

    # batch the input structures
    batched_strucs = batch_strucs(strucs=strucs, batch_size=user_configs["batch_size"])

    # run directory setup
    batching = make_launch_dirs(batched_strucs=batched_strucs, calcs_dir=calcs_dir)

    write_json(batching, fjson)
    return read_json(fjson)


def make_relax_scripts(
    batching: dict, user_configs: dict, relax_template: str, remake: bool = False
) -> None:
    """
    Args:
        batching (dict): {"batch_id": {"launch_dir": str}}
        user_configs (dict): user configs
        relax_template (str): path to relax template
        remake (bool): if True, remake relax scripts

    Returns:
        None, writes chgnet relax script for each job (batch)
    """

    for batch_id in batching:

        launch_dir = batching[batch_id]["launch_dir"]

        relax_script = os.path.join(launch_dir, "chgnet_relax.py")

        if os.path.exists(relax_script) and not remake:
            continue

        with open(relax_template, "r", encoding="utf-8") as template_file:
            template_lines = template_file.readlines()

        relax_script_lines = template_lines.copy()

        for i, line in enumerate(relax_script_lines):
            if 'intra_op_threads = "placeholder"' in line:
                relax_script_lines[i] = (
                    f'    intra_op_threads = {user_configs["num_intraop_threads"]}\n'
                )
            elif 'inter_op_threads = "placeholder"' in line:
                relax_script_lines[i] = (
                    f'    inter_op_threads = {user_configs["num_interop_threads"]}\n'
                )
            elif 'model = "placeholder"' in line:
                relax_script_lines[i] = f'    model = "{user_configs["model"]}"\n'
            elif 'optimizer = "placeholder"' in line:
                relax_script_lines[i] = (
                    f'    optimizer = "{user_configs["optimizer"]}"\n'
                )
            elif 'stress_weight = "placeholder"' in line:
                relax_script_lines[i] = (
                    f'    stress_weight = {user_configs["stress_weight"]}\n'
                )
            elif 'on_isolated_atoms = "placeholder"' in line:
                relax_script_lines[i] = (
                    f'    on_isolated_atoms = "{user_configs["on_isolated_atoms"]}"\n'
                )
            elif 'fmax = "placeholder"' in line:
                relax_script_lines[i] = f'    fmax = {user_configs["fmax"]}\n'
            elif 'steps = "placeholder"' in line:
                relax_script_lines[i] = f'    steps = {user_configs["steps"]}\n'
            elif 'relax_cell = "placeholder"' in line:
                relax_script_lines[i] = (
                    f'    relax_cell = {user_configs["relax_cell"]}\n'
                )
            elif 'ase_filter = "placeholder"' in line:
                relax_script_lines[i] = (
                    f'    ase_filter = "{user_configs["ase_filter"]}"\n'
                )
            elif 'params_asefilter = "placeholder"' in line:
                relax_script_lines[i] = (
                    f'    params_asefilter = {user_configs["params_asefilter"]}\n'
                )
            elif 'relax_interval = "placeholder"' in line:
                relax_script_lines[i] = (
                    f'    relax_interval = {user_configs["relax_interval"]}\n'
                )
            elif 'verbose = "placeholder"' in line:
                relax_script_lines[i] = f'    verbose = {user_configs["verbose"]}\n'
            elif 'save_interval = "placeholder"' in line:
                relax_script_lines[i] = (
                    f'    save_interval = {user_configs["save_interval"]}\n'
                )

        with open(relax_script, "w", encoding="utf-8") as script_file:
            script_file.writelines(relax_script_lines)

        print(f"\nCreated new relax script for {launch_dir}")

    return


def make_submission_scripts(
    batching: dict, user_configs: dict, remake: bool = False
) -> None:
    """
    Args:
        batching (dict): {"batch_id": {"launch_dir": str}}
        user_configs (dict): user configs
        remake (bool): if true remake submission scripts

    Returns:
        job_names_by_dir (dict): dict of job names by launch directory
    """

    for batch_id in batching:

        launch_dir = batching[batch_id]["launch_dir"]

        relax_launcher = os.path.join(launch_dir, "sub.sh")

        if os.path.exists(relax_launcher) and not remake:
            continue

        job_name = f"chgnet_relax_{batch_id}"

        with open(relax_launcher, "w", encoding="utf-8") as f:
            f.write("#!/bin/bash -l\n")
            f.write(f"#SBATCH --nodes={user_configs['nodes']}\n")
            f.write(f"#SBATCH --ntasks={user_configs['ntasks']}\n")
            f.write(f"#SBATCH --time={user_configs['time']}\n")
            f.write(f"#SBATCH --mem-per-cpu={user_configs['mem_per_core']}\n")
            f.write(f"#SBATCH --error={user_configs['error_file']}\n")
            f.write(f"#SBATCH --output={user_configs['output_file']}\n")
            f.write(f"#SBATCH --account={user_configs['account']}\n")
            f.write(f"#SBATCH --job-name={job_name}\n")
            f.write(f"#SBATCH --partition={user_configs['partition']}\n")
            f.write("\n")
            f.write("python chgnet_relax.py\n")

        print(f"\nCreated new submission script for {launch_dir}")

    return


def check_job_submission_status(job_name: str) -> bool:
    """
    Note: this function is the same as the method in SubmitTools in pydmclab

    Args:
        job_name (str): name of job

    Returns:
        job_in_que_or_running (bool): True if job is in queue
    """

    # create a temporary file w/ jobs in queue with my username and this job_name
    scripts_dir = os.getcwd()
    fqueue = os.path.join(scripts_dir, "_".join(["q", job_name]) + ".o")
    with open(fqueue, "w", encoding="utf-8") as f:
        subprocess.call(
            [
                "squeue",
                f"--user={os.getlogin()}",
                "--noheader",
                f"--name={job_name}",
            ],
            stdout=f,
        )
        subprocess.call(
            [
                "squeue",
                f"--user={os.getlogin()}",
                "--partition=msidmc",
                "--noheader",
                f"--name={job_name}",
            ],
            stdout=f,
        )

    # get the job names I have in the queue
    names_in_q = []
    with open(fqueue, "r", encoding="utf-8") as f:
        for line in f:
            names_in_q.append([v for v in line.split(" ") if len(v) > 0][2])

    # delete the file I wrote w/ the queue output
    os.remove(fqueue)

    # if this job is in the queue, return True
    if len(names_in_q) > 0:
        # print("  %s already in queue, not messing with it\n" % job_name)
        return True

    # print("  %s not in queue, onward\n" % job_name)
    return False


def check_job_completion_status(launch_dir: str) -> bool:
    """
    Args:
        launch_dir (str): path to launch directory

    Returns:
        job_completed (bool): True if job has completed
    """

    num_ini_strucs = len(read_json(os.path.join(launch_dir, "ini_strucs.json")))
    batch_results = os.path.join(launch_dir, "chgnet_relax_results.json")
    if os.path.exists(batch_results):
        num_relaxed_strucs = len(read_json(batch_results))
    else:
        num_relaxed_strucs = 0

    if num_ini_strucs == num_relaxed_strucs:
        return True
    else:
        return False


def submit_jobs(batching: dict) -> None:
    """
    Args:
        batching (dict): {"batch_id": {"launch_dir": str}}
        user_configs (dict): user configs

    Returns:
        None, submits jobs if not already in queue or finished
    """

    scripts_dir = os.getcwd()

    for batch_id in batching:

        launch_dir = batching[batch_id]["launch_dir"]

        job_name = f"chgnet_relax_{batch_id}"

        # check if job is already in queue
        if check_job_submission_status(job_name):
            print(f"\n{job_name} is already in queue")
            continue

        # check if job has already finished
        if check_job_completion_status(launch_dir=launch_dir):
            print(f"\n{job_name} is finished")
            continue

        # submit job if not in queue or finished
        relax_launcher = os.path.join(launch_dir, "sub.sh")
        os.chdir(launch_dir)
        print(f"\nSubmitting {job_name}")
        subprocess.call(["sbatch", relax_launcher])
        os.chdir(scripts_dir)

    return


def collect_results(
    batching: dict, user_configs: dict, data_dir: str, remake: bool = True
) -> dict:
    """
    Args:
        batching (dict): {"batch_id": {"launch_dir": str}}
        user_configs (dict): user configs
        data_dir (str): path to data directory
        remake (bool): if True, remake results

    Returns:
        results (dict): dict of relaxation results and configs
    """

    fjson = os.path.join(data_dir, "relax_results.json")
    if os.path.exists(fjson) and not remake:
        return read_json(fjson)

    print("\nCollecting results")

    results = {"relax_results": {}, "relax_configs": {}}

    # record relax job settings
    results["relax_configs"]["model"] = user_configs["model"]
    results["relax_configs"]["optimizer"] = user_configs["optimizer"]
    results["relax_configs"]["stress_weight"] = user_configs["stress_weight"]
    results["relax_configs"]["on_isolated_atoms"] = user_configs["on_isolated_atoms"]
    results["relax_configs"]["fmax"] = user_configs["fmax"]
    results["relax_configs"]["steps"] = user_configs["steps"]
    results["relax_configs"]["relax_cell"] = user_configs["relax_cell"]
    results["relax_configs"]["ase_filter"] = user_configs["ase_filter"]
    results["relax_configs"]["params_asefilter"] = user_configs["params_asefilter"]
    results["relax_configs"]["interval"] = user_configs["relax_interval"]

    # collect results from each batch
    for batch_id in batching:

        launch_dir = batching[batch_id]["launch_dir"]

        # check if job is finished
        if not check_job_completion_status(launch_dir=launch_dir):
            continue

        batch_relax_results = read_json(
            os.path.join(launch_dir, "chgnet_relax_results.json")
        )

        for formula_struc_id, relax_result in batch_relax_results.items():
            formula, struc_id = formula_struc_id.split("_", 1)
            if formula not in results["relax_results"]:
                results["relax_results"][formula] = {}
            relax_result["batch_id"] = batch_id
            results["relax_results"][formula][struc_id] = relax_result

    write_json(results, fjson)
    return read_json(fjson)


def check_collected_results(results: dict, batching: dict) -> None:
    """
    Args:
        results (dict): dict of relaxation results and configs
        batching (dict): {"batch_id": {"launch_dir": str}}

    Returns:
        None, prints how many batches have been fully relaxed
    """

    results_possible = len(batching)

    unique_batch_ids = set()
    for formula in results["relax_results"]:
        for struc_id in results["relax_results"][formula]:
            unique_batch_ids.add(
                results["relax_results"][formula][struc_id]["batch_id"]
            )

    results_collected = len(unique_batch_ids)

    print(
        f"\nCollected results for {results_collected} / {results_possible} relax batches"
    )

    return

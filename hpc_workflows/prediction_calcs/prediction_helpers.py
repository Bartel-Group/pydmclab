from __future__ import annotations
from typing import TYPE_CHECKING, Literal

import os
import shutil
import subprocess

from tqdm import tqdm

from ase import units

from pydmclab.utils.handy import read_json, write_json

if TYPE_CHECKING:
    from pydmclab.mlp.chgnet import Versions, PredTask
    from pydmclab.mlp.chgnet.dynamics import CHGNet, CHGNetCalculator
    from pydmclab.mlp.mace.dynamics import MACECalculator
    from torch import nn


def get_chgnet_configs(
    model: CHGNet | CHGNetCalculator | Versions | None = None,
    stress_weight: float | None = 1 / 160.21766208,
    on_isolated_atoms: Literal["ignore", "warn", "error"] = "warn",
    task: PredTask = "efsm",
    return_site_energies: bool = False,
    return_atom_feas: bool = False,
    return_crystal_feas: bool = False,
    batch_size: int = 16,
):
    """
    Note: this assumes cpu only use on MSI

    Args:
        model: if None, uses model "0.3.0"
        stress_weight: the conversion factor to convert GPa to eV/A^3
        on_isolated_atoms: what to do if isolated atoms are found
        task: what to predict with the model
        return_site_energies: if True, returns site energies
        return_atom_feas: if True, returns atom features
        return_crystal_feas: if True, returns crystal features
        batch_size: the number of structures to predict per pass to the model
        verbose: if True, prints prediction information

    Returns:
        prediction_configs (dict): dict of model/prediction configurations
    """

    architecture_configs = {
        "architecture": "CHGNet",
        "relaxer_configs": {},
        "predict_structure_configs": {},
    }

    architecture_configs["relaxer_configs"]["model"] = model
    architecture_configs["relaxer_configs"]["stress_weight"] = stress_weight
    architecture_configs["relaxer_configs"]["on_isolated_atoms"] = on_isolated_atoms

    architecture_configs["predict_structure_configs"]["task"] = task
    architecture_configs["predict_structure_configs"][
        "return_site_energies"
    ] = return_site_energies
    architecture_configs["predict_structure_configs"][
        "return_atom_feas"
    ] = return_atom_feas
    architecture_configs["predict_structure_configs"][
        "return_crystal_feas"
    ] = return_crystal_feas
    architecture_configs["predict_structure_configs"]["batch_size"] = batch_size

    return architecture_configs


def get_mace_configs(
    models: MACECalculator | list[nn.Module] | nn.Module | list[str] | str,
    default_dtype: Literal["float32", "float64", "auto"] = "auto",
    model_type: Literal["MACE", "DipoleMACE", "EnergyDipoleMace"] = "MACE",
    energy_units_to_eV: float = 1.0,
    length_units_to_A: float = 1.0,
    charges_key: str = "Qs",
    compile_mode: (
        Literal[
            "default",
            "reduce-overhead",
            "max-autotune",
            "max-autotune-no-cudagraphs",
        ]
        | None
    ) = None,
    fullgraph: bool = True,
    enable_cueq: bool = False,
    include_dispersion: bool = False,
    damping_function: Literal["zero", "bj", "zerom", "bjm"] = "bj",
    dispersion_xc: str = "pbe",
    dispersion_cutoff: float = 40.0 * units.Bohr,
    remake_cache: bool = False,
):
    """
    Note: this assumes cpu only use on MSI

    Args:
        model: if None, uses model "0.3.0"
        stress_weight: the conversion factor to convert GPa to eV/A^3
        on_isolated_atoms: what to do if isolated atoms are found
        task: what to predict with the model
        return_site_energies: if True, returns site energies
        return_atom_feas: if True, returns atom features
        return_crystal_feas: if True, returns crystal features
        batch_size: the number of structures to predict per pass to the model
        verbose: if True, prints prediction information

    Returns:
        architecture_configs (dict): dict of prediction configurations
    """

    architecture_configs = {
        "architecture": "MACE",
        "relaxer_configs": {},
        "predict_structure_configs": {},
    }

    architecture_configs["relaxer_configs"]["models"] = models
    architecture_configs["relaxer_configs"]["default_dtype"] = default_dtype
    architecture_configs["relaxer_configs"]["model_type"] = model_type
    architecture_configs["relaxer_configs"]["energy_units_to_eV"] = energy_units_to_eV
    architecture_configs["relaxer_configs"]["length_units_to_A"] = length_units_to_A
    architecture_configs["relaxer_configs"]["charges_key"] = charges_key
    architecture_configs["relaxer_configs"]["compile_mode"] = compile_mode
    architecture_configs["relaxer_configs"]["fullgraph"] = fullgraph
    architecture_configs["relaxer_configs"]["enable_cueq"] = enable_cueq
    architecture_configs["relaxer_configs"]["include_dispersion"] = include_dispersion
    architecture_configs["relaxer_configs"]["damping_function"] = damping_function
    architecture_configs["relaxer_configs"]["dispersion_xc"] = dispersion_xc
    architecture_configs["relaxer_configs"]["dispersion_cutoff"] = dispersion_cutoff
    architecture_configs["relaxer_configs"]["remake_cache"] = remake_cache

    return architecture_configs


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
    mem_per_core_in_MB: int = 3900,
    partition: str = "msismall, msidmc",
    error_file: str = "log.e",
    output_file: str = "log.o",
    job_name: str | None = None,
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
    slurm_configs["job_name"] = job_name
    slurm_configs["account"] = account

    return slurm_configs


def get_torch_configs(
    slurm_configs: dict,
    num_intraop_threads: int | None = None,
    num_interop_threads: int | None = None,
) -> dict:
    """
    Args:
        slurm_configs (dict): dict of SLURM configurations
        num_intraop_threads (int): number of intra-op threads
        num_interop_threads (int): number of inter-op threads

    Returns:
        torch_configs (dict): dict of torch configurations
    """

    if isinstance(num_intraop_threads, int) and (
        num_intraop_threads > slurm_configs["ntasks"]
    ):
        raise ValueError("num_intraop_threads must be less than or equal to ntasks")
    if isinstance(num_interop_threads, int) and (
        num_interop_threads > slurm_configs["ntasks"]
    ):
        raise ValueError("num_interop_threads must be less than or equal to ntasks")

    torch_configs = {}
    torch_configs["num_intraop_threads"] = num_intraop_threads
    torch_configs["num_interop_threads"] = num_interop_threads

    return torch_configs


def batch_strucs(strucs: dict, batch_size: int) -> dict:
    """
    Args:
        strucs (dict): {formula: {struc_id: {Structure.as_dict()}}}
        batch_size (int): the number of structures to predict per job

    Returns:
        batched_strucs (dict): {batch_id: {formula_struc_id: {Structure.as_dict()}}}
    """

    batch_id = 0
    batched_strucs = {}
    current_batch = {}

    total_strucs = sum(len(strucs[formula]) for formula in strucs)

    with tqdm(total=total_strucs, desc="Batching structures") as pbar:
        for formula in strucs:
            for struc_id, struc in strucs[formula].items():
                current_batch[f"{formula}_{struc_id}"] = struc
                pbar.update(1)
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


def detect_indent(line: str) -> str:
    """Detect leading indentation (spaces or tabs) from a line. Need to indent lines within main() and other functions while writing to the script."""
    return line[: len(line) - len(line.lstrip())]


def make_prediction_scripts(
    batching: dict,
    user_configs: dict,
    architecture_configs: dict,
    prediction_template: str,
    remake: bool = False,
) -> None:
    """
    Args:
        batching (dict): {"batch_id": {"launch_dir": str}}
        user_configs (dict): user configs
        prediction_template (str): path to predict template
        remake (bool): if True, remake predict scripts

    Returns:
        None, writes predict script for each job (batch)
    """

    architecture = user_configs["architecture"]
    if architecture.lower() == "chgnet":
        model = user_configs["relaxer_configs"]["model"].replace(".", "")
    elif architecture.lower() == "mace":
        model = user_configs["relaxer_configs"]["models"]

    total_batches = len(batching)

    with tqdm(total=total_batches, desc="Making prediction scripts") as pbar:

        for batch_id in batching:

            launch_dir = batching[batch_id]["launch_dir"]

            prediction_script = os.path.join(
                launch_dir, f"{architecture.lower()}-{model}-prediction.py"
            )

            if os.path.exists(prediction_script) and not remake:
                continue

            with open(prediction_template, "r", encoding="utf-8") as template_file:
                template_lines = template_file.readlines()

            prediction_script_lines = template_lines.copy()

            for i, line in enumerate(prediction_script_lines):

                indent = detect_indent(line)

                if 'from pydmclab.mlp import "placeholder"' in line:
                    prediction_script_lines[i] = (
                        f"{indent}from pydmclab.mlp.{architecture.lower()}.dynamics import {architecture}Relaxer\n"
                    )

                elif 'intra_op_threads = "placeholder"' in line:
                    prediction_script_lines[i] = (
                        f'{indent}intra_op_threads = {user_configs["num_intraop_threads"]}\n'
                    )
                elif 'inter_op_threads = "placeholder"' in line:
                    prediction_script_lines[i] = (
                        f'{indent}inter_op_threads = {user_configs["num_interop_threads"]}\n'
                    )

                elif 'architecture = "placeholder"' in line:
                    prediction_script_lines[i] = (
                        f"{indent}architecture = '{architecture}'\n"
                    )

                elif 'relaxer_configs = "placeholder"' in line:
                    config_lines = [
                        f"{indent}{key} = {repr(value)}\n"
                        for key, value in architecture_configs[
                            "relaxer_configs"
                        ].items()
                    ]
                    prediction_script_lines[i : i + 1] = config_lines

                elif 'predict_structure_configs = "placeholder"' in line:
                    config_lines = [
                        f"{indent}{key} = {repr(value)}\n"
                        for key, value in architecture_configs[
                            "predict_structure_configs"
                        ].items()
                    ]
                    prediction_script_lines[i : i + 1] = config_lines

                elif 'save_interval = "placeholder"' in line:
                    prediction_script_lines[i] = (
                        f"{indent}save_interval = {user_configs['save_interval']}\n"
                    )

                elif 'results = os.path.join(curr_dir, "placeholder")' in line:
                    prediction_script_lines[i] = (
                        f"{indent}results = os.path.join(curr_dir, '{architecture.lower()}_{model}_prediction_results.json')\n"
                    )

                elif 'relaxer = "placeholder"' in line:

                    class_call_line = [f"{indent}relaxer = {architecture}Relaxer(\n"]
                    relaxer_config_lines = [
                        f"{indent}    {key} = {key},\n"
                        for key in architecture_configs["relaxer_configs"].keys()
                    ]
                    end_call_line = [f"{indent})\n"]

                    prediction_script_lines[i : i + 1] = (
                        class_call_line + relaxer_config_lines + end_call_line
                    )

                elif 'struc_results = "placeholder"' in line:
                    class_call_line = [
                        f"{indent}struc_results = relaxer.predict_structure(ini_struc, \n"
                    ]
                    predict_structure_config_lines = [
                        f"{indent}    {key} = {key},\n"
                        for key in architecture_configs[
                            "predict_structure_configs"
                        ].keys()
                    ]
                    end_call_line = [f"{indent})\n"]

                    prediction_script_lines[i : i + 1] = (
                        class_call_line + predict_structure_config_lines + end_call_line
                    )

            with open(prediction_script, "w", encoding="utf-8") as script_file:
                script_file.writelines(prediction_script_lines)

        pbar.update(1)

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

    architecture = user_configs["architecture"]
    if architecture.lower() == "chgnet":
        model = user_configs["relaxer_configs"]["model"].replace(".", "")
    elif architecture.lower() == "mace":
        model = user_configs["relaxer_configs"]["models"]

    for batch_id in batching:

        launch_dir = batching[batch_id]["launch_dir"]

        prediction_launcher = os.path.join(launch_dir, "sub.sh")

        if os.path.exists(prediction_launcher) and not remake:
            continue

        if user_configs["job_name"]:
            job_name = f"{user_configs['job_name']}_{batch_id}"
        else:
            job_name = f"{architecture.lower()}_{model}_prediction_{batch_id}"

        with open(prediction_launcher, "w", encoding="utf-8") as f:
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
            f.write(f"python {architecture.lower()}-{model}-prediction.py\n")

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
        return True

    return False


def check_job_completion_status(launch_dir: str, user_configs: dict) -> bool:
    """
    Args:
        launch_dir (str): path to launch directory

    Returns:
        job_completed (bool): True if job has completed
    """

    architecture = user_configs["architecture"]
    if architecture.lower() == "chgnet":
        model = user_configs["relaxer_configs"]["model"].replace(".", "")
    elif architecture.lower() == "mace":
        model = user_configs["relaxer_configs"]["models"]

    num_ini_strucs = len(read_json(os.path.join(launch_dir, "ini_strucs.json")))
    batch_results = os.path.join(
        launch_dir, f"{architecture.lower()}_{model}_prediction_results.json"
    )
    if os.path.exists(batch_results):
        num_predictioned_strucs = len(read_json(batch_results))
    else:
        num_predictioned_strucs = 0

    if num_ini_strucs == num_predictioned_strucs:
        return True
    else:
        return False


def submit_jobs(batching: dict, user_configs: dict) -> None:
    """
    Args:
        batching (dict): {"batch_id": {"launch_dir": str}}
        user_configs (dict): user configs

    Returns:
        None, submits jobs if not already in queue or finished
    """

    architecture = user_configs["architecture"]
    if architecture.lower() == "chgnet":
        model = user_configs["relaxer_configs"]["model"].replace(".", "")
    elif architecture.lower() == "mace":
        model = user_configs["relaxer_configs"]["models"]

    scripts_dir = os.getcwd()

    for batch_id in batching:

        launch_dir = batching[batch_id]["launch_dir"]

        if user_configs["job_name"]:
            job_name = f"{user_configs['job_name']}_{batch_id}"
        else:
            job_name = f"{architecture.lower()}_{model}_prediction_{batch_id}"

        # check if job is already in queue
        if check_job_submission_status(job_name):
            print(f"\n{job_name} is already in queue")
            continue

        # check if job has already finished
        if check_job_completion_status(
            launch_dir=launch_dir, user_configs=user_configs
        ):
            print(f"\n{job_name} is finished")
            continue

        # submit job if not in queue or finished
        prediction_launcher = os.path.join(launch_dir, "sub.sh")
        os.chdir(launch_dir)
        print(f"\nSubmitting {job_name}")
        subprocess.call(["sbatch", prediction_launcher])
        os.chdir(scripts_dir)

    return


def collect_results(
    batching: dict,
    user_configs: dict,
    architecture_configs: dict,
    data_dir: str,
    remake: bool = True,
) -> dict:
    """
    Args:
        batching (dict): {"batch_id": {"launch_dir": str}}
        user_configs (dict): user configs
        data_dir (str): path to data directory
        remake (bool): if True, remake results

    Returns:
        results (dict): dict of predictionation results and configs
    """
    architecture = user_configs["architecture"]
    if architecture.lower() == "chgnet":
        model = user_configs["relaxer_configs"]["model"].replace(".", "")
    elif architecture.lower() == "mace":
        model = user_configs["relaxer_configs"]["models"]

    fjson = os.path.join(
        data_dir, f"{architecture.lower()}_{model}_prediction_results.json"
    )
    if os.path.exists(fjson) and not remake:
        return read_json(fjson)

    results = {"prediction_results": {}, "architecture_configs": {}}

    results["architecture_configs"] = architecture_configs

    # collect results from each batch

    total_batches = len(batching)

    with tqdm(total=total_batches, desc="Collecting results") as pbar:
        for batch_id in batching:

            launch_dir = batching[batch_id]["launch_dir"]

            # check if job is finished
            if not check_job_completion_status(
                launch_dir=launch_dir, user_configs=user_configs
            ):
                pbar.update(1)
                continue

            batch_prediction_results = read_json(
                os.path.join(
                    launch_dir,
                    f"{architecture.lower()}_{model}_prediction_results.json",
                )
            )

            for formula_struc_id, prediction_result in batch_prediction_results.items():
                formula, mpid, the_word_clean, facet, the_word_step, step_number = formula_struc_id.split("_")
                if not results["prediction_results"].get(formula):
                    results["prediction_results"][formula] = {}
                if not results["prediction_results"][formula].get(facet):
                    results["prediction_results"][formula][facet] = {}
                prediction_result["batch_id"] = batch_id
                results["prediction_results"][formula][facet][step_number] = prediction_result
            pbar.update(1)

    write_json(results, fjson)
    return read_json(fjson)


def check_collected_results(results: dict, batching: dict) -> None:
    """
    Args:
        results (dict): dict of prediction results and configs
        batching (dict): {"batch_id": {"launch_dir": str}}

    Returns:
        None, prints how many batches have been fully predicted
    """

    results_possible = len(batching)

    unique_batch_ids = set()
    for formula in results["prediction_results"]:
        for struc_id in results["prediction_results"][formula]:
            for step_number in results["prediction_results"][formula][struc_id]:
                unique_batch_ids.add(
                    results["prediction_results"][formula][struc_id][step_number]["batch_id"]
                )

    results_collected = len(unique_batch_ids)

    print(f"\nCompleted {results_collected} / {results_possible} prediction batches")

    return

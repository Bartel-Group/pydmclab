from __future__ import annotations
from typing import TYPE_CHECKING, Literal

import os
import sys
import shutil
import subprocess

from tqdm import tqdm

from pydmclab.utils.handy import read_json, write_json

if TYPE_CHECKING:
    from pydmclab.mlp import Versions
    from ase.optimize.optimize import Optimizer as ASEOptimizer
    from fairchem.core.units.mlip_unit.api.inference import InferenceSettings


def get_chgnet_configs(
    model: Versions | None = None,
    optimizer: ASEOptimizer | str = "FIRE",
    stress_weight: float | None = 1 / 160.21766208,
    on_isolated_atoms: Literal["ignore", "warn", "error"] = "warn",
    fmax: float | None = 0.1,
    steps: int | None = 500,
    relax_cell: bool | None = True,
    ase_filter: str | None = "FrechetCellFilter",
    params_asefilter: dict | None = None,
    interval: int | None = 1,
    verbose: bool = True,
):
    """
    Note: this assumes cpu only use on MSI and only supports pretrained models

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
        interval: logging interval for relax obs
        verbose: if True, prints relaxation information

    Returns:
        relax_configs (dict): dict of architecture/ relaxer/ relax configurations
    """

    architecture_configs = {
        "architecture": "CHGNet",
        "relaxer_configs": {},
        "relax_configs": {},
    }

    architecture_configs["relaxer_configs"]["model"] = model
    architecture_configs["relaxer_configs"]["optimizer"] = optimizer
    architecture_configs["relaxer_configs"]["stress_weight"] = stress_weight
    architecture_configs["relaxer_configs"]["on_isolated_atoms"] = on_isolated_atoms

    architecture_configs["relax_configs"]["fmax"] = fmax
    architecture_configs["relax_configs"]["steps"] = steps
    architecture_configs["relax_configs"]["relax_cell"] = relax_cell
    architecture_configs["relax_configs"]["ase_filter"] = ase_filter
    architecture_configs["relax_configs"]["params_asefilter"] = params_asefilter
    architecture_configs["relax_configs"]["interval"] = interval
    architecture_configs["relax_configs"]["verbose"] = verbose

    return architecture_configs


def get_fairchem_configs(
    name_or_path: str,
    task_name: str,
    inference_settings: InferenceSettings | str = "default",
    overrides: dict | None = None,
    optimizer: ASEOptimizer | str = "FIRE",
    fmax: float | None = 0.1,
    steps: int | None = 500,
    relax_cell: bool | None = True,
    ase_filter: str | None = "FrechetCellFilter",
    params_asefilter: dict | None = None,
    interval: int | None = 1,
    verbose: bool = True,
):
    """
    Note: this assumes cpu only use on MSI and only supports pretrained models

    Args:
        name_or_path: the model name or a path to a checkpoint
        task_name: class of materials you are relaxing (e.g., "omat" for inorganic crystals)
        inference_settings: the inference settings to use ("default" is general purpose)
        overrides: overrides for the inference settings
        optimizer: default is "FIRE", see pydmclab.mlp.dynamics for more options
        fmax: the force convergence criterion
        steps: the maximum number of steps to try during relaxation
        relax_cell: whether to relax the cell (False is equivalent to ISIF = 2)
        ase_filter: the ASE filter to use
        params_asefilter: the parameters for the ASE filter
        interval: logging interval for relax obs
        verbose: if True, prints relaxation information

    Returns:
        relax_configs (dict): dict of architecture/ relaxer/ relax configurations
    """

    architecture_configs = {
        "architecture": "FAIRChem",
        "relaxer_configs": {},
        "relax_configs": {},
    }

    architecture_configs["relaxer_configs"]["name_or_path"] = name_or_path
    architecture_configs["relaxer_configs"]["task_name"] = task_name
    architecture_configs["relaxer_configs"]["inference_settings"] = inference_settings
    architecture_configs["relaxer_configs"]["overrides"] = overrides
    architecture_configs["relaxer_configs"]["optimizer"] = optimizer

    architecture_configs["relax_configs"]["fmax"] = fmax
    architecture_configs["relax_configs"]["steps"] = steps
    architecture_configs["relax_configs"]["relax_cell"] = relax_cell
    architecture_configs["relax_configs"]["ase_filter"] = ase_filter
    architecture_configs["relax_configs"]["params_asefilter"] = params_asefilter
    architecture_configs["relax_configs"]["interval"] = interval
    architecture_configs["relax_configs"]["verbose"] = verbose

    return architecture_configs

def get_nequix_configs(
        model_name = 'nequix-mp-1-pft',
        path = None,
        backend = 'jax',
        capacity_multiplier = 1.1,
        use_compile = True,
        use_kernel = False #for GPU only
):
    """
    See https://github.com/atomicarchitects/nequix for details on available models, many available models that are fine-tuned for phonons

    Args:
        model_name (str, default "nequix-mp-1"): Pretrained model alias to load or download.
        model_path (str | Path, optional): Path to local checkpoint; overrides model_name.
        backend ({"jax", "torch"}, default "jax"): Compute backend.
        capacity_multiplier (float, default 1.1): JAX-only; padding factor to limit recompiles.
        use_compile (bool, default True): Torch-only; on GPU, uses torch.compile().
        use_kernel (bool, default True): on GPU, use OpenEquivariance kernels.

    Returns:
        relax_configs (dict): dict of architecture/ relaxer/ relax configurations
    """

    architecture_configs = {
        "architecture": "Nequix",
        "relaxer_configs": {},
        "relax_configs": {},
    }

    architecture_configs["relaxer_configs"]["model_name"] = model_name
    architecture_configs["relaxer_configs"]["path"] = path
    architecture_configs["relaxer_configs"]["backend"] = backend
    architecture_configs["relaxer_configs"]["capacity_multiplier"] = capacity_multiplier
    architecture_configs["relaxer_configs"]["use_compile"] = use_compile
    architecture_configs["relaxer_configs"]["use_kernel"] = use_kernel

    #need to add relacer configs

    return architecture_configs

def get_phononcalc_configs(architecture,
                           relaxer_kwargs,
                           atom_disp= 0.015,
                           min_length = 20.0,
                           supercell_matrix = None,
                           t_step = 20,
                           t_max = 2000,
                           t_min = 0,
                           fmax = 1e-5,
                           max_steps = 5000,
                           relax_structure = True,
                           imaginary_freq_tol: float = -0.01,
                            on_imaginary_modes: Literal["error", "warn"] = "warn",
                            fix_imaginary_attempts: int = 0,
                            symprec: float = 1e-5,
                            write_force_constants: bool | str = False,
                            write_band_structure: bool | str = False,
                            write_total_dos: bool | str = False,
                            write_phonon: bool | str  = False,
                           relax_calc_kwargs = {'relax_cell' : True,
                                                'relax_atoms' :True,
                                                'cell_filter' : "FrechetCellFilter",
                                                'cell_filter_kwargs' : {},
                                                'interval' : 1,
                                                'perturb_distance': None,
                                                'fix_symmetry': False,
                                                'fix_atoms': False,},
):


    '''
    Configs for matcalc's PhononCalc calculator

    Args:
        architecture: what MLP architecture to load to the phonon calculator (FAIRCHem, Nequix, etc.)
        relaxer_kwargs: architecture-specific kwargs for loading the model. Note: these are only the kwargs needed to *load* the model. Relax kwargs are overridden by the other arguments

    '''
    architecture_configs = {
        "architecture": architecture,
        "is_phonon_calc": True,
        "relaxer_configs": {},
        "relax_configs": {},
        "phonon_configs": {}
    }

    if architecture == 'FAIRChem':
       architecture_configs['relaxer_configs'] = get_fairchem_configs(**relaxer_kwargs)["relaxer_configs"]
    elif architecture == 'CHGNet':
        architecture_configs['relaxer_configs']  = get_chgnet_configs(**relaxer_kwargs)["relaxer_configs"]
    elif architecture == 'Nequix':
        architecture_configs['relaxer_configs']  = get_nequix_configs(**relaxer_kwargs)["relaxer_configs"]

    architecture_configs["relax_configs"]["fmax"] = fmax
    architecture_configs["relax_configs"]["max_steps"] = max_steps
    architecture_configs["relax_configs"]['relax_structure'] = relax_structure
    relax_calc_kwargs = dict(relax_calc_kwargs)  # don't mutate the caller's dict
    if relax_calc_kwargs.get("cell_filter") == "FrechetCellFilter":
        relax_calc_kwargs.pop("cell_filter") #this way it will do the default which is FechetCellFilter class, not a string
    architecture_configs["relax_configs"]["relax_calc_kwargs"] = relax_calc_kwargs

    architecture_configs["phonon_configs"]["atom_disp"] = atom_disp
    architecture_configs["phonon_configs"]["min_length"] = min_length
    architecture_configs["phonon_configs"]["supercell_matrix"] = supercell_matrix
    architecture_configs["phonon_configs"]["t_step"] = t_step
    architecture_configs["phonon_configs"]["t_max"] = t_max
    architecture_configs["phonon_configs"]["t_min"] = t_min
    architecture_configs["phonon_configs"]["imaginary_freq_tol"] = imaginary_freq_tol
    architecture_configs["phonon_configs"]["on_imaginary_modes"] = on_imaginary_modes
    architecture_configs["phonon_configs"]["fix_imaginary_attempts"] = fix_imaginary_attempts
    architecture_configs["phonon_configs"]["symprec"] = symprec
    architecture_configs["phonon_configs"]["write_force_constants"] = write_force_constants
    architecture_configs["phonon_configs"]["write_band_structure"] = write_band_structure
    architecture_configs["phonon_configs"]["write_total_dos"] = write_total_dos
    architecture_configs["phonon_configs"]["write_phonon"] = write_phonon

    return architecture_configs



def get_launch_configs(
    batch_size: int = 100, batch_id: int = 0, save_interval: int = 5
):
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
    launch_configs["batch_id"] = batch_id
    launch_configs["save_interval"] = save_interval

    return launch_configs


def get_slurm_configs(
    total_nodes: int = 1,
    tasks_per_node: int = 1,
    cores_per_task: int = 8,
    walltime_in_hours: int = 12,
    mem_per_core_in_MB: int = 3900,
    partition: str = "preempt,msismall,msidmc",
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
    slurm_configs["ntasks"] = int(total_nodes * tasks_per_node)
    slurm_configs["cores_per_task"] = cores_per_task
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

    if (
        isinstance(num_intraop_threads, int)
        and num_intraop_threads
        > slurm_configs["ntasks"] * slurm_configs["cores_per_task"]
    ):
        raise ValueError("num_intraop_threads must be less than or equal to ntasks")
    if (
        isinstance(num_interop_threads, int)
        and num_interop_threads
        > slurm_configs["ntasks"] * slurm_configs["cores_per_task"]
    ):
        raise ValueError("num_interop_threads must be less than or equal to ntasks")

    torch_configs = {}
    torch_configs["num_intraop_threads"] = num_intraop_threads
    torch_configs["num_interop_threads"] = num_interop_threads

    return torch_configs


def batch_strucs(
    strucs: dict,
    batch_size: int,
    batch_id: int,
) -> dict:
    """
    Args:
        strucs (dict): {formula: {struc_id: {Structure.as_dict()}}}
        batch_size (int): the number of structures to relax per job

    Returns:
        batched_strucs (dict): {batch_id: {formula_struc_id: {Structure.as_dict()}}}
    """

    batch_id = batch_id
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
    batched_strucs = batch_strucs(
        strucs=strucs,
        batch_size=user_configs["batch_size"],
        batch_id=user_configs["batch_id"],
    )

    # run directory setup
    batching = make_launch_dirs(batched_strucs=batched_strucs, calcs_dir=calcs_dir)

    write_json(batching, fjson)
    return read_json(fjson)


def detect_indent(line: str) -> str:
    """
    Detect leading indentation (spaces or tabs) from a line.
    Need to indent lines within main() and other functions while writing to the script.
    """
    return line[: len(line) - len(line.lstrip())]


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
        None, writes relax/phonon script for each job (batch)
    """

    architecture = user_configs["architecture"]
    if architecture.lower() == "chgnet":
        model = user_configs["relaxer_configs"]["model"].replace(".", "")
    elif architecture.lower() == "fairchem":
        model_name = user_configs["relaxer_configs"]["name_or_path"]
        model_task = user_configs["relaxer_configs"]["task_name"]
        model = f"{model_name}-{model_task}"

    # is this a phonon workflow (has phonon_configs) or a plain relax workflow?
    is_phonon = "phonon_configs" in user_configs
    suffix = "phonons" if is_phonon else "relax"

    total_batches = len(batching)

    with tqdm(total=total_batches, desc="Making relaxation scripts") as pbar:

        for batch_id in batching:

            launch_dir = batching[batch_id]["launch_dir"]

            relax_script = os.path.join(
                launch_dir, f"{architecture.lower()}_{model}_{suffix}.py"
            )

            if os.path.exists(relax_script) and not remake:
                continue

            with open(relax_template, "r", encoding="utf-8") as template_file:
                template_lines = template_file.readlines()

            relax_script_lines = template_lines.copy()

            for i, line in enumerate(relax_script_lines):

                indent = detect_indent(line)

                if 'from pydmclab.mlp import "placeholder"' in line:
                    if is_phonon:
                        if "nequix" in architecture.lower():
                            # not yet implemented in pydmclab.mlp -- import straight from source
                            relax_script_lines[i] = (
                                f"{indent}from nequix.calculator import NequixCalculator\n"
                            )
                        else:
                            # need the raw ASE Calculator, not the Relaxer wrapper
                            relax_script_lines[i] = (
                                f"{indent}from pydmclab.mlp.{architecture.lower()}.dynamics import {architecture}Calculator\n"
                            )
                    else:
                        relax_script_lines[i] = (
                            f"{indent}from pydmclab.mlp.{architecture.lower()}.dynamics import {architecture}Relaxer\n"
                        )

                elif 'HELPERS_DIR = "placeholder"' in line:
                    if is_phonon:
                        # TODO: point this at wherever phonon_helpers.py actually lives
                        relax_script_lines[i] = (
                            f'{indent}HELPERS_DIR = "%s/bin/pydmclab/hpc_workflows/phonon_calcs/" % HOME_PATH\n'
                        )
                    else:
                        relax_script_lines[i] = ""

                elif 'intra_op_threads = "placeholder"' in line:
                    relax_script_lines[i] = (
                        f'{indent}intra_op_threads = {user_configs["num_intraop_threads"]}\n'
                    )

                elif 'inter_op_threads = "placeholder"' in line:
                    relax_script_lines[i] = (
                        f'{indent}inter_op_threads = {user_configs["num_interop_threads"]}\n'
                    )

                elif 'architecture = "placeholder"' in line:
                    relax_script_lines[i] = f"{indent}architecture = '{architecture}'\n"

                elif 'relaxer_configs = "placeholder"' in line:
                    config_lines = [
                        f"{indent}{key} = {repr(value)}\n"
                        for key, value in user_configs["relaxer_configs"].items()
                    ]
                    relax_script_lines[i : i + 1] = config_lines

                elif 'relax_configs = "placeholder"' in line:
                    config_lines = [
                        f"{indent}{key} = {repr(value)}\n"
                        for key, value in user_configs["relax_configs"].items()
                    ]
                    relax_script_lines[i : i + 1] = config_lines

                elif 'phonon_configs = "placeholder"' in line:
                    if is_phonon:
                        config_lines = [
                            f"{indent}{key} = {repr(value)}\n"
                            for key, value in user_configs["phonon_configs"].items()
                        ]
                        relax_script_lines[i : i + 1] = config_lines
                    else:
                        relax_script_lines[i] = ""

                elif 'save_interval = "placeholder"' in line:
                    relax_script_lines[i] = (
                        f"{indent}save_interval = {user_configs['save_interval']}\n"
                    )

                elif 'results = os.path.join(curr_dir, "placeholder")' in line:
                    relax_script_lines[i] = (
                        f"{indent}results = os.path.join(curr_dir, '{architecture.lower()}_{model}_{suffix}_results.json')\n"
                    )

                elif 'calculator = "placeholder"' in line and is_phonon:
                    # relaxer_configs includes 'optimizer', which the Calculator
                    # class doesn't accept -- everything else does
                    calc_keys = [
                        k for k in user_configs["relaxer_configs"].keys()
                        if k != "optimizer"
                    ]
                    class_call_line = [f"{indent}calculator = {architecture}Calculator(\n"]
                    calc_config_lines = [
                        f"{indent}    {key} = {key},\n" for key in calc_keys
                    ]
                    end_call_line = [f"{indent})\n"]
                    relax_script_lines[i : i + 1] = (
                        class_call_line + calc_config_lines + end_call_line
                    )

                elif 'phonon_calculator = "placeholder"' in line and is_phonon:
                    # PhononCalc takes calculator + everything in relax_configs
                    # and phonon_configs, plus 'optimizer' (pulled from relaxer_configs)
                    phonon_calc_keys = (
                        list(user_configs["relax_configs"].keys())
                        + list(user_configs["phonon_configs"].keys())
                    )
                    if "optimizer" in user_configs["relaxer_configs"]:
                        phonon_calc_keys.append("optimizer")

                    class_call_line = [
                        f"{indent}phonon_calculator = mtc.PhononCalc(\n",
                        f"{indent}    calculator,\n",
                    ]
                    phonon_calc_config_lines = [
                        f"{indent}    {key} = {key},\n" for key in phonon_calc_keys
                    ]
                    end_call_line = [f"{indent})\n"]
                    relax_script_lines[i : i + 1] = (
                        class_call_line + phonon_calc_config_lines + end_call_line
                    )

                elif 'relaxer = "placeholder"' in line and not is_phonon:

                    class_call_line = [f"{indent}relaxer = {architecture}Relaxer(\n"]
                    relaxer_config_lines = [
                        f"{indent}    {key} = {key},\n"
                        for key in user_configs["relaxer_configs"].keys()
                    ]
                    end_call_line = [f"{indent})\n"]
                    relax_script_lines[i : i + 1] = (
                        class_call_line + relaxer_config_lines + end_call_line
                    )

                elif 'struc_results = "placeholder"' in line and not is_phonon:
                    class_call_line = [
                        f"{indent}struc_results = relaxer.relax(ini_struc, \n"
                    ]
                    relax_structure_config_lines = [
                        f"{indent}    {key} = {key},\n"
                        for key in user_configs["relax_configs"].keys()
                    ]
                    end_call_line = [f"{indent})\n"]
                    relax_script_lines[i : i + 1] = (
                        class_call_line + relax_structure_config_lines + end_call_line
                    )

            with open(relax_script, "w", encoding="utf-8") as script_file:
                script_file.writelines(relax_script_lines)

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
    # is this a phonon workflow (has phonon_configs) or a plain relax workflow?
    is_phonon = "phonon_configs" in user_configs
    suffix = "phonons" if is_phonon else "relax"

    architecture = user_configs["architecture"]
    if architecture.lower() == "chgnet":
        model = user_configs["relaxer_configs"]["model"].replace(".", "")
    elif architecture.lower() == "fairchem":
        model_name = user_configs["relaxer_configs"]["name_or_path"]
        model_task = user_configs["relaxer_configs"]["task_name"]
        model = f"{model_name}-{model_task}"

    for batch_id in batching:

        launch_dir = batching[batch_id]["launch_dir"]

        relax_launcher = os.path.join(launch_dir, "sub.sh")

        if os.path.exists(relax_launcher) and not remake:
            continue

        job_name = f"{architecture.lower()}_{model}_{suffix}_{batch_id}"

        with open(relax_launcher, "w", encoding="utf-8") as f:
            f.write("#!/bin/bash -l\n")
            f.write(f"#SBATCH --nodes={user_configs['nodes']}\n")
            f.write(f"#SBATCH --ntasks={user_configs['ntasks']}\n")
            f.write(f"#SBATCH --cpus-per-task={user_configs['cores_per_task']}\n")
            f.write(f"#SBATCH --time={user_configs['time']}\n")
            f.write(f"#SBATCH --mem-per-cpu={user_configs['mem_per_core']}\n")
            f.write(f"#SBATCH --error={user_configs['error_file']}\n")
            f.write(f"#SBATCH --output={user_configs['output_file']}\n")
            f.write(f"#SBATCH --account={user_configs['account']}\n")
            f.write(f"#SBATCH --job-name={job_name}\n")
            f.write(f"#SBATCH --partition={user_configs['partition']}\n")
            f.write("\n")
            f.write(f"python {architecture.lower()}_{model}_{suffix}.py\n")

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


def check_job_completion_status(launch_dir: str, user_configs: dict) -> bool:
    """
    Args:
        launch_dir (str): path to launch directory

    Returns:
        job_completed (bool): True if job has completed
    """
    # is this a phonon workflow (has phonon_configs) or a plain relax workflow?
    is_phonon = "phonon_configs" in user_configs
    suffix = "phonons" if is_phonon else "relax"

    architecture = user_configs["architecture"]
    if architecture.lower() == "chgnet":
        model = user_configs["relaxer_configs"]["model"].replace(".", "")
    elif architecture.lower() == "fairchem":
        model_name = user_configs["relaxer_configs"]["name_or_path"]
        model_task = user_configs["relaxer_configs"]["task_name"]
        model = f"{model_name}-{model_task}"

    num_ini_strucs = len(read_json(os.path.join(launch_dir, "ini_strucs.json")))
    batch_results = os.path.join(
        launch_dir, f"{architecture.lower()}_{model}_{suffix}_results.json"
    )
    if os.path.exists(batch_results):
        num_relaxed_strucs = len(read_json(batch_results))
    else:
        num_relaxed_strucs = 0

    if num_ini_strucs == num_relaxed_strucs:
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
    # is this a phonon workflow (has phonon_configs) or a plain relax workflow?
    is_phonon = "phonon_configs" in user_configs
    suffix = "phonons" if is_phonon else "relax"

    architecture = user_configs["architecture"]
    if architecture.lower() == "chgnet":
        model = user_configs["relaxer_configs"]["model"].replace(".", "")
    elif architecture.lower() == "fairchem":
        model_name = user_configs["relaxer_configs"]["name_or_path"]
        model_task = user_configs["relaxer_configs"]["task_name"]
        model = f"{model_name}-{model_task}"

    scripts_dir = os.getcwd()

    for batch_id in batching:

        launch_dir = batching[batch_id]["launch_dir"]

        job_name = f"{architecture.lower()}_{model}_{suffix}_{batch_id}"

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
        relax_launcher = os.path.join(launch_dir, "sub.sh")
        os.chdir(launch_dir)
        print(f"\nSubmitting {job_name}")
        subprocess.call(["sbatch", relax_launcher])
        os.chdir(scripts_dir)

    return


def collect_results(
    batching: dict,
    user_configs: dict,
    data_dir: str,
    include_obs: bool = True,
    remake: bool = True,
) -> dict:
    """
    Args:
        batching (dict): {"batch_id": {"launch_dir": str}}
        user_configs (dict): user configs
        data_dir (str): path to data directory
        include_obs (bool): whether to include obs in collected resultsß
        remake (bool): if True, remake results

    Returns:
        results (dict): dict of relaxation results and configs
    """

    # is this a phonon workflow (has phonon_configs) or a plain relax workflow?
    is_phonon = "phonon_configs" in user_configs
    suffix = "phonons" if is_phonon else "relax"

    architecture = user_configs["architecture"]
    if architecture.lower() == "chgnet":
        model = user_configs["relaxer_configs"]["model"].replace(".", "")
    elif architecture.lower() == "fairchem":
        model_name = user_configs["relaxer_configs"]["name_or_path"]
        model_task = user_configs["relaxer_configs"]["task_name"]
        model = f"{model_name}-{model_task}"

    fjson = os.path.join(data_dir, f"{architecture.lower()}_{model}_{suffix}_results.json")
    if os.path.exists(fjson) and not remake:
        return read_json(fjson)

    print("\nCollecting results")

    results = {"relax_results": {}, "architecture_configs": {}}

    results["architecture_configs"]["architecture"] = user_configs["architecture"]
    results["architecture_configs"]["relaxer_configs"] = user_configs["relaxer_configs"]
    results["architecture_configs"]["relax_configs"] = user_configs["relax_configs"]

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

            batch_relax_results = read_json(
                os.path.join(
                    launch_dir,
                    f"{architecture.lower()}_{model}_{suffix}_results.json",
                )
            )

            for formula_struc_id, relax_result in batch_relax_results.items():
                formula, struc_id = formula_struc_id.split("_", 1)
                if formula not in results["relax_results"]:
                    results["relax_results"][formula] = {}
                relax_result["batch_id"] = batch_id
                if include_obs:
                    results["relax_results"][formula][struc_id] = relax_result
                else:
                    results["relax_results"][formula][struc_id] = {
                        k: v for k, v in relax_result.items() if k != "trajectory"
                    }
            pbar.update(1)

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

    print(f"\nCompleted {results_collected} / {results_possible} relax batches")

    return

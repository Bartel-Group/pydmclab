from __future__ import annotations
from typing import TYPE_CHECKING, Literal

import os
import subprocess

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
    relax_first: bool = True,
    ensembles: str | tuple[str] = "nvt",
    thermostats: str | tuple[str] = "bi",
    taut: float | None = None,
    timestep: float = 1.0,
    loginterval: int = 10,
    nsteps: int = 10000,
    temperatures: float | tuple[float] = 300.0,
    pressure: float = 1.01325e-4,
    addn_args: dict | None = None,
) -> dict:
    """
    Note: this assumes cpu only use on MSI and only supports pretrained models

    Args:
        model: if None, uses model "0.3.0"
        optimizer: default is "FIRE", see pydmclab.mlp.dynamics for more options
        stress_weight: the conversion factor to convert GPa to eV/A^3
        on_isolated_atoms: what to do if isolated atoms are found
        relax_first (bool): if True, relax structures before MD
        ensembles (str | tuple): 'nvt', 'npt', or 'nve'
        thermostats (str | tuple): 'nh' for Nose-Hoover, 'b' for Berendsen, or 'bi' for Berendsen_inhomogeneous
        taut (float): time constant for temperature coupling in fs
        timestep (float): timestep in fs
        loginterval (int): interval for logging in steps
        nsteps (int): number of steps
        temperatures (float | list[float]): temperature(s) in K
        pressure (float): pressure in GPa
        addn_args (dict): additional arguments (kwargs to pass to say the pre-relaxer)

    Returns:
        architecture_configs (dict): dict of architecture/ calculator/ md configurations
    """

    if isinstance(ensembles, str):
        ensembles = (ensembles,)
    if isinstance(thermostats, str):
        thermostats = (thermostats,)
    if isinstance(temperatures, float):
        temperatures = (temperatures,)

    valid_ensembles = ("nvt", "npt", "nve")
    if any(e not in valid_ensembles for e in ensembles):
        raise ValueError("valid ensembles are: 'nvt', 'npt', or 'nve'")

    valid_thermostats = ("b", "bi", "nh")
    if any(thermo not in valid_thermostats for thermo in thermostats):
        raise ValueError("valid thermostats are: 'b', 'bi', or 'nh'")

    thermostat_full_names = []
    if "b" in thermostats:
        thermostat_full_names.append("Berendsen")
    if "bi" in thermostats:
        thermostat_full_names.append("Berendsen_inhomogeneous")
    if "nh" in thermostats:
        thermostat_full_names.append("Nose-Hoover")

    if taut is None:
        taut = 100 * timestep

    if addn_args is None:
        addn_args = {}
    elif not isinstance(addn_args, dict):
        raise ValueError("addn_args must be a dictionary")

    architecture_configs = {
        "architecture": "CHGNet",
        "calculator_configs": {},
        "md_configs": {},
    }

    architecture_configs["calculator_configs"]["model"] = model
    architecture_configs["calculator_configs"]["optimizer"] = optimizer
    architecture_configs["calculator_configs"]["stress_weight"] = stress_weight
    architecture_configs["calculator_configs"]["on_isolated_atoms"] = on_isolated_atoms

    architecture_configs["md_configs"]["relax_first"] = relax_first
    architecture_configs["md_configs"]["ensembles"] = ensembles
    architecture_configs["md_configs"]["thermostats"] = tuple(thermostat_full_names)
    architecture_configs["md_configs"]["taut"] = taut
    architecture_configs["md_configs"]["timestep"] = timestep
    architecture_configs["md_configs"]["loginterval"] = loginterval
    architecture_configs["md_configs"]["steps"] = nsteps
    architecture_configs["md_configs"]["temperatures"] = temperatures
    architecture_configs["md_configs"]["pressure"] = pressure
    architecture_configs["md_configs"]["stress_weight"] = stress_weight
    architecture_configs["md_configs"]["addn_args"] = addn_args

    return architecture_configs


def get_fairchem_configs(
    name_or_path: str,
    task_name: str,
    inference_settings: InferenceSettings | str = "default",
    overrides: dict | None = None,
    ensembles: str | tuple[str] = "nvt",
    thermostats: str | tuple[str] = "bi",
    starting_temperature: float | None = 300.0,
    taut: float | None = None,
    timestep: float = 1.0,
    loginterval: int = 10,
    nsteps: int = 10000,
    temperatures: float | tuple[float] = 300.0,
    pressure: float = 1.01325e-4,
    logfile: str = "md.log",
    trajfile: str = "md.traj",
    addn_args: dict | None = None,
) -> dict:
    """
    Note: this assumes cpu only use on MSI and only supports pretrained models

    Args:
        name_or_path: the model name or a path to a checkpoint
        task_name: class of materials you are relaxing (e.g., "omat" for inorganic crystals)
        inference_settings: the inference settings to use ("default" is general purpose)
        overrides: overrides for the inference settings
        optimizer: default is "FIRE", see pydmclab.mlp.dynamics for more options
        ensembles (str | tuple): 'nvt', 'npt', or 'nve'
        thermostats (str | tuple): 'nh' for Nose-Hoover, 'b' for Berendsen, or 'bi' for Berendsen_inhomogeneous
        starting_temperature (float | None): starting temperature in K. if None, defaults to temperature of the simulation
        taut (float): time constant for temperature coupling in fs
        timestep (float): timestep in fs
        loginterval (int): interval for logging in steps
        nsteps (int): number of steps
        temperatures (float | tuple[float]): temperature(s) in K
        pressure (float): pressure in GPa
        logfile (str): .log file if it exists
        trajfile (str): .traj file if it exists
        addn_args (dict): additional arguments (kwargs to pass to say the pre-relaxer)

    Returns:
        architecture_configs (dict): dict of architecture/ calculator/ md configurations
    """

    if isinstance(ensembles, str):
        ensembles = (ensembles,)
    if isinstance(thermostats, str):
        thermostats = (thermostats,)
    if isinstance(temperatures, float):
        temperatures = (temperatures,)

    valid_ensembles = ("nvt", "npt", "nve")
    if any(e not in valid_ensembles for e in ensembles):
        raise ValueError("valid ensembles are: 'nvt', 'npt', or 'nve'")

    valid_thermostats = ("b", "bi", "nh")
    if any(thermo not in valid_thermostats for thermo in thermostats):
        raise ValueError("valid thermostats are: 'b', 'bi', or 'nh'")

    thermostat_full_names = []
    if "b" in thermostats:
        thermostat_full_names.append("Berendsen")
    if "bi" in thermostats:
        thermostat_full_names.append("Berendsen_inhomogeneous")
    if "nh" in thermostats:
        thermostat_full_names.append("Nose-Hoover")

    if taut is None:
        taut = 100 * timestep

    if addn_args is None:
        addn_args = {}
    elif not isinstance(addn_args, dict):
        raise ValueError("addn_args must be a dictionary")

    architecture_configs = {
        "architecture": "FAIRChem",
        "calculator_configs": {},
        "md_configs": {},
    }

    architecture_configs["calculator_configs"]["name_or_path"] = name_or_path
    architecture_configs["calculator_configs"]["task_name"] = task_name
    architecture_configs["calculator_configs"][
        "inference_settings"
    ] = inference_settings
    architecture_configs["calculator_configs"]["overrides"] = overrides

    architecture_configs["md_configs"]["ensembles"] = ensembles
    architecture_configs["md_configs"]["thermostats"] = tuple(thermostat_full_names)
    architecture_configs["md_configs"]["starting_temperature"] = starting_temperature
    architecture_configs["md_configs"]["taut"] = taut
    architecture_configs["md_configs"]["timestep"] = timestep
    architecture_configs["md_configs"]["loginterval"] = loginterval
    architecture_configs["md_configs"]["steps"] = nsteps
    architecture_configs["md_configs"]["temperatures"] = temperatures
    architecture_configs["md_configs"]["pressure"] = pressure
    architecture_configs["md_configs"]["logfile"] = logfile
    architecture_configs["md_configs"]["trajfile"] = trajfile
    architecture_configs["md_configs"]["addn_args"] = addn_args

    return architecture_configs


def get_slurm_configs(
    total_nodes: int = 1,
    cores_per_node: int = 16,
    walltime_in_hours: int = 24,
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


def make_launch_dirs(
    strucs: dict,
    user_configs: dict,
    calcs_dir: str,
    data_dir: str,
    savename: str = "launch_dirs.json",
    remake: bool = False,
) -> dict:
    """
    Args:
        strucs (dict): {formula: {struc_id: {Structure.as_dict()}}} from strucs.json
        user_configs (dict): user configs
        calcs_dir (str): path to calculations directory
        data_dir (str): path to data directory
        savename (str): name of json file to save launch directories
        remake_dirs (bool): if true add missing launch directories, else read existing from json
    Returns:
        launch_dirs (dict): dict of launch directories
    """

    fjson = os.path.join(data_dir, savename)
    if os.path.exists(fjson) and not remake:
        return read_json(fjson)

    ensembles = user_configs["md_configs"]["ensembles"]
    thermostats = user_configs["md_configs"]["thermostats"]
    temperatures = user_configs["md_configs"]["temperatures"]

    launch_dirs = {}
    for formula in strucs:
        for struc_id in strucs[formula]:
            for ensemble in ensembles:
                for thermostat in thermostats:
                    for temperature in temperatures:
                        launch_dir = os.path.join(
                            calcs_dir,
                            formula,
                            struc_id,
                            ensemble,
                            thermostat,
                            str(temperature),
                        )
                        # make launch directory if it doesn't exist
                        if not os.path.exists(launch_dir):
                            os.makedirs(launch_dir)
                        # save launch directory with its settings easy to access
                        launch_dirs[launch_dir] = {
                            "formula": formula,
                            "struc_id": struc_id,
                            "ensemble": ensemble,
                            "thermostat": thermostat,
                            "temperature": temperature,
                        }
                        # save initial structure as json in launch directory
                        struc = strucs[formula][struc_id]
                        write_json(struc, os.path.join(launch_dir, "ini_struc.json"))

    write_json(launch_dirs, fjson)
    return read_json(fjson)


def detect_indent(line: str) -> str:
    """
    Detect leading indentation (spaces or tabs) from a line.
    Need to indent lines within main() and other functions while writing to the script.
    """
    return line[: len(line) - len(line.lstrip())]


def get_model(user_configs: dict) -> tuple[str, str]:
    """
    Args:
        user_configs (dict): user configs

    Returns:
        A tuple containing the architecture and information about the model
        based on the architecture
    """
    architecture = user_configs["architecture"]
    if architecture.lower() == "chgnet":
        return (
            architecture,
            user_configs["calculator_configs"]["model"].replace(".", ""),
        )
    elif architecture.lower() == "fairchem":
        model_name = user_configs["calculator_configs"]["name_or_path"]
        model_task = user_configs["calculator_configs"]["task_name"]
        return (architecture, f"{model_name}-{model_task}")


def make_md_scripts(
    launch_dirs: dict, user_configs: dict, md_template: str, remake: bool = False
) -> None:
    """
    Args:
        launch_dirs (dict): dict of launch directories and their settings
        user_configs (dict): user configs
        md_template (str): path to md_template.py
        remake (bool): if True, remake md scripts

    Returns:
        None, writes chgnet md script for each job
    """

    architecture, model = get_model(user_configs)
    md_configs_leave_out = [
        "ensembles",
        "thermostats",
        "starting_temperature",
        "temperatures",
        "steps",
        "addn_args",
    ]
    md_configs_add_in = [
        "ensemble",
        "thermostat",
        "starting_temperature",
        "temperature",
    ]

    for launch_dir, settings in launch_dirs.items():

        md_script = os.path.join(launch_dir, f"{architecture.lower()}_{model}_md.py")

        if os.path.exists(md_script) and not remake:
            continue

        with open(md_template, "r", encoding="utf-8") as template_file:
            template_lines = template_file.readlines()

        md_script_lines = template_lines.copy()

        for i, line in enumerate(md_script_lines):

            indent = detect_indent(line)

            if 'from pydmclab.mlp import "placeholder"' in line:
                md_script_lines[i] = (
                    f"{indent}from pydmclab.mlp.{architecture.lower()}.dynamics import {architecture}MD\n"
                )

            if 'intra_op_threads = "placeholder"' in line:
                md_script_lines[i] = (
                    f'    intra_op_threads = {user_configs["num_intraop_threads"]}\n'
                )

            elif 'inter_op_threads = "placeholder"' in line:
                md_script_lines[i] = (
                    f'    inter_op_threads = {user_configs["num_interop_threads"]}\n'
                )

            elif 'architecture = "placeholder"' in line:
                md_script_lines[i] = f"{indent}architecture = '{architecture}'\n"

            elif 'calculator_configs = "placeholder"' in line:
                config_lines = [
                    f"{indent}{key} = {repr(value)}\n"
                    for key, value in user_configs["calculator_configs"].items()
                ]
                md_script_lines[i : i + 1] = config_lines

            elif 'md_configs = "placeholder"' in line:
                config_lines = [
                    f"{indent}{key} = {repr(value)}\n"
                    for key, value in user_configs["md_configs"].items()
                    if key not in md_configs_leave_out
                ]
                md_script_lines[i : i + 1] = config_lines

            elif 'ensemble = "placeholder"' in line:
                md_script_lines[i] = f'{indent}ensemble = "{settings["ensemble"]}"\n'

            elif 'thermostat = "placeholder"' in line:
                md_script_lines[i] = (
                    f'{indent}thermostat = "{settings["thermostat"]}"\n'
                )

            elif 'starting_temperature = "placeholder"' in line:
                if user_configs["md_configs"]["starting_temperature"] is None:
                    md_script_lines[i] = (
                        f'{indent}starting_temperature = {settings["temperature"]}\n'
                    )
                else:
                    md_script_lines[i] = (
                        f'{indent}starting_temperature = {user_configs["md_configs"]["starting_temperature"]}\n'
                    )

            elif 'temperature = "placeholder"' in line:
                md_script_lines[i] = (
                    f'{indent}temperature = {settings["temperature"]}\n'
                )

            elif 'steps = "placeholder"' in line:
                md_script_lines[i] = (
                    f'{indent}steps = {user_configs["md_configs"]["steps"]}\n'
                )

            elif 'md = "placeholder"' in line:
                class_call_line = [f"{indent}md = {architecture}MD(ini_struc,\n"]
                calculator_config_lines = [
                    f"{indent}    {key} = {key},\n"
                    for key in user_configs["calculator_configs"].keys()
                ]
                md_configs_keys = [
                    key
                    for key in user_configs["md_configs"].keys()
                    if key not in md_configs_leave_out
                ] + md_configs_add_in
                md_config_lines = [
                    f"{indent}    {key} = {key},\n" for key in md_configs_keys
                ]
                end_call_line = [f"{indent})\n"]
                md_script_lines[i : i + 1] = (
                    class_call_line
                    + calculator_config_lines
                    + md_config_lines
                    + end_call_line
                )

            elif 'md_continue = "placeholder"' in line:
                class_call_line = [
                    f"{indent}md_continue = {architecture}MD.continue_from_traj(\n"
                ]
                calculator_config_lines = [
                    f"{indent}    {key} = {key},\n"
                    for key in user_configs["calculator_configs"].keys()
                ]
                md_configs_keys = [
                    key
                    for key in user_configs["md_configs"].keys()
                    if key not in md_configs_leave_out
                ] + md_configs_add_in
                md_config_lines = [
                    f"{indent}    {key} = {key},\n" for key in md_configs_keys
                ]
                end_call_line = [f"{indent})\n"]
                md_script_lines[i : i + 1] = (
                    class_call_line
                    + calculator_config_lines
                    + md_config_lines
                    + end_call_line
                )

            elif 'full_summary, os.path.join(curr_dir, "placeholder")' in line:
                md_script_lines[i] = (
                    f"{indent}full_summary, os.path.join(curr_dir, '{architecture.lower()}_{model}_md_results.json')\n"
                )

        with open(md_script, "w", encoding="utf-8") as script_file:
            script_file.writelines(md_script_lines)

        print(f"\nCreated new MD script for {launch_dir}")

    return


def make_submission_scripts(
    launch_dirs: dict, user_configs: dict, remake: bool = False
) -> None:
    """
    Args:
        launch_dirs (dict): dict of launch directories and their settings
        user_configs (dict): user configs
        remake (bool): if true remake submission scripts

    Returns:
        job_names_by_dir (dict): dict of job names by launch directory
    """

    architecture, model = get_model(user_configs)

    for launch_dir, settings in launch_dirs.items():

        md_launcher = os.path.join(launch_dir, "sub.sh")

        if os.path.exists(md_launcher) and not remake:
            continue

        formula = launch_dir.split("/")[-4]
        struc_id = launch_dir.split("/")[-3]
        job_name = f'{architecture.lower()}_{model}_md_{formula}_{struc_id}_{settings["ensemble"]}_{settings["thermostat"]}_{settings["temperature"]}'

        with open(md_launcher, "w", encoding="utf-8") as f:
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
            f.write(f"python {architecture.lower()}_{model}_md.py\n")

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


def submit_jobs(launch_dirs: dict, user_configs: dict) -> None:
    """
    Args:
        launch_dirs (dict): dict of launch directories and their settings
        user_configs (dict): user configs

    Returns:
        None, submits jobs if not already in queue or finished
    """

    architecture, model = get_model(user_configs)

    scripts_dir = os.getcwd()

    for launch_dir, settings in launch_dirs.items():

        job_name = f'{architecture.lower()}_{model}_md_{settings["formula"]}_{settings["struc_id"]}_{settings["ensemble"]}_{settings["thermostat"]}_{settings["temperature"]}'

        # check if job is already in queue
        if check_job_submission_status(job_name):
            print(f"\n{job_name} is already in queue")
            continue

        # check if job has already finished
        results = os.path.join(
            launch_dir, f"{architecture.lower()}_{model}_md_results.json"
        )
        if os.path.exists(results):
            print(f"\n{job_name} is finished")
            continue

        # submit job if not in queue or finished
        md_launcher = os.path.join(launch_dir, "sub.sh")
        os.chdir(launch_dir)
        print(f"\nSubmitting {job_name}")
        subprocess.call(["sbatch", md_launcher])
        os.chdir(scripts_dir)

    return


def collect_results(
    launch_dirs: dict,
    user_configs: dict,
    data_dir: str,
    remake: bool = True,
) -> dict:
    """
    Args:
        launch_dirs (dict): dict of launch directories and their settings
        user_configs (dict): user configs
        data_dir (str): path to data directory
        remake (bool): if True, remake results

    Returns:
        results (dict): dict of MD results and configs
    """

    architecture, model = get_model(user_configs)

    fjson = os.path.join(data_dir, f"{architecture.lower()}_{model}_md_results.json")
    if os.path.exists(fjson) and not remake:
        return read_json(fjson)

    print("\nCollecting results:")

    results = {"md_results": {}, "architecture_configs": {}}

    # record architecture and MD job settings
    results["architecture_configs"]["architecture"] = user_configs["architecture"]
    results["architecture_configs"]["calculator_configs"] = user_configs[
        "calculator_configs"
    ]
    results["architecture_configs"]["md_configs"] = user_configs["md_configs"]

    # collect results
    for launch_dir, settings in launch_dirs.items():

        full_summary = os.path.join(
            launch_dir, f"{architecture.lower()}_{model}_md_results.json"
        )

        # check if job is finished
        if not os.path.exists(full_summary):
            continue

        # full summary of MD results
        full_summary = read_json(full_summary)

        # setup results dict
        formula = settings["formula"]
        if formula not in results["md_results"]:
            results["md_results"][formula] = {}
        struc_id = settings["struc_id"]
        if struc_id not in results["md_results"][formula]:
            results["md_results"][formula][struc_id] = {}
        ensemble = settings["ensemble"]
        if ensemble not in results["md_results"][formula][struc_id]:
            results["md_results"][formula][struc_id][ensemble] = {}
        thermostat = settings["thermostat"]
        if thermostat not in results["md_results"][formula][struc_id][ensemble]:
            results["md_results"][formula][struc_id][ensemble][thermostat] = {}
        temperature = settings["temperature"]

        # assign full summary to results dict
        results["md_results"][formula][struc_id][ensemble][thermostat][
            temperature
        ] = full_summary

    # save results
    write_json(results, fjson)
    return read_json(fjson)


def check_collected_results(results: dict, launch_dirs: dict) -> None:
    """
    Args:
        results (dict): dict of MD results and configs
        launch_dirs (dict): dict of launch directories and their settings

    Returns:
        None, prints how many MD simulation results were collected
    """

    results_possible = len(launch_dirs)
    results_collected = sum(
        len(results["md_results"][formula][struc_id][ensemble][thermostat])
        for formula in results["md_results"]
        for struc_id in results["md_results"][formula]
        for ensemble in results["md_results"][formula][struc_id]
        for thermostat in results["md_results"][formula][struc_id][thermostat]
        for temperature in results["md_results"][formula][struc_id][thermostat][
            temperature
        ]
    )
    print(
        f"\nCollected results for {results_collected} / {results_possible} MD simulations"
    )

    return


def lowest_energy_struc_results(
    results: dict,
    time_to_consider: float,
    num_strucs: int,
    data_dir: str,
    remake: bool = True,
) -> dict:
    """
    Args:
        results (dict): dict of MD results and architecture configs
        time_to_consider (float): percentage of results to consider for each MD simulation,
            e.g. 0.5 for the back half of the results
        num_strucs (int): number of lowest energy structures to collect for each MD simulation
        data_dir (str): path to data directory
        remake (bool): if True, remake lowest energy structures

    Returns:
        strucs (dict): dict of lowest energy structures in standard strucs.json form
    """

    num_of_results_considered = int(
        time_to_consider
        * results["architecture_configs"]["md_configs"]["steps"]
        // results["architecture_configs"]["md_configs"]["loginterval"]
    )
    if num_of_results_considered < num_strucs:
        raise ValueError(
            "more strucs requested than available in given time_to_consider"
        )
    if time_to_consider < 0 or time_to_consider > 1:
        raise ValueError("time_to_consider must be between 0 and 1")

    fjson = os.path.join(data_dir, "lowest_energy_strucs.json")
    if os.path.exists(fjson) and not remake:
        return read_json(fjson)

    print("\nCollecting lowest energy structures:")

    strucs = {}

    for formula in results["md_results"]:
        for struc_id in results["md_results"][formula]:
            for ensemble in results["md_results"][formula][struc_id]:
                for thermostat in results["md_results"][formula][struc_id][ensemble]:
                    for temperature in results["md_results"][formula][struc_id][
                        ensemble
                    ][thermostat]:

                        # full summary of MD results
                        full_summary = results["md_results"][formula][struc_id][
                            ensemble
                        ][thermostat][temperature]

                        # consider only the prescribed portion of the results
                        cutoff_index = int(time_to_consider * len(full_summary))
                        reduced_summary = full_summary[cutoff_index:]

                        # divide into num_strucs buckets
                        # get the lowest energy structure from each bucket
                        bucket_size = len(reduced_summary) // num_strucs
                        lowest_energy_strucs = {}
                        for i in range(num_strucs):
                            bucket = reduced_summary[
                                i * bucket_size : (i + 1) * bucket_size
                            ]
                            lowest_energy_struc = min(bucket, key=lambda x: x["Epot"])
                            new_struc_id = f"{struc_id}_{ensemble}_{thermostat}_{temperature}_md_struc_{i}"
                            if "structure" in lowest_energy_struc:
                                lowest_energy_strucs[new_struc_id] = (
                                    lowest_energy_struc["structure"]
                                )
                            else:
                                print(
                                    f"structure not found for {formula}_{struc_id}_{ensemble}_{thermostat}_{temperature}_bucket_{i}"
                                )

                        # setup strucs dict
                        if formula not in strucs:
                            strucs[formula] = {}

                        # assign lowest energy structures to strucs dict
                        strucs[formula].update(lowest_energy_strucs)

    # save lowest energy structures
    write_json(strucs, fjson)
    return read_json(fjson)

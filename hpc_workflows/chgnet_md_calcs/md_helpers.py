import os
import subprocess
from pydmclab.utils.handy import read_json, write_json


def get_md_configs(
    relax_first: bool = True,
    ensembles: str | tuple[str] = "nvt",
    thermostats: str | tuple[str] = "bi",
    taut: float | None = None,
    timestep: float = 1.0,
    loginterval: int = 10,
    nsteps: int = 10000,
    temperature: float = 300.0,
    pressure: float = 1.01325e-4,
    stress_weight: float | None = 1 / 160.21766208,
    addn_args: dict | None = None,
) -> dict:
    """
    Args:
        relax_first (bool): if True, relax structures before MD
        ensembles (str | tuple): 'nvt', 'npt', or 'nve'
        thermostats (str | tuple): 'nh' for Nose-Hoover, 'b' for Berendsen, or 'bi' for Berendsen_inhomogeneous
        taut (float): time constant for temperature coupling in fs
        timestep (float): timestep in fs
        loginterval (int): interval for logging in steps
        nsteps (int): number of steps
        temperature (float): temperature in K
        pressure (float): pressure in GPa
        stress_weight (float | None): stress weight
        addn_args (dict): additional arguments (kwargs to pass to say the pre-relaxer)

    Returns:
        md_configs (dict): dict of MD configurations
    """

    if isinstance(ensembles, str):
        ensembles = (ensembles,)
    if isinstance(thermostats, str):
        thermostats = (thermostats,)

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

    md_configs = {}

    md_configs["relax_first"] = relax_first
    md_configs["ensembles"] = ensembles
    md_configs["thermostats"] = tuple(thermostat_full_names)
    md_configs["taut"] = taut
    md_configs["timestep"] = timestep
    md_configs["loginterval"] = loginterval
    md_configs["nsteps"] = nsteps
    md_configs["temperature"] = temperature
    md_configs["pressure"] = pressure
    md_configs["stress_weight"] = stress_weight
    md_configs["addn_args"] = addn_args

    return md_configs


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

    ensembles = user_configs["ensembles"]
    thermostats = user_configs["thermostats"]

    launch_dirs = {}
    for formula in strucs:
        for struc_id in strucs[formula]:
            for ensemble in ensembles:
                for thermostat in thermostats:
                    launch_dir = os.path.join(
                        calcs_dir, formula, struc_id, ensemble, thermostat
                    )
                    # make launch directory if it doesn't exist
                    if not os.path.exists(launch_dir):
                        os.makedirs(launch_dir)
                    # save launch directory with its ensemble and thermostat easy to access
                    launch_dirs[launch_dir] = {
                        "formula": formula,
                        "struc_id": struc_id,
                        "ensemble": ensemble,
                        "thermostat": thermostat,
                    }
                    # save initial structure as json in launch directory
                    struc = strucs[formula][struc_id]
                    write_json(struc, os.path.join(launch_dir, "ini_struc.json"))

    write_json(launch_dirs, fjson)
    return read_json(fjson)


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

    for launch_dir, settings in launch_dirs.items():

        md_script = os.path.join(launch_dir, "chgnet_md.py")

        if os.path.exists(md_script) and not remake:
            continue

        with open(md_template, "r", encoding="utf-8") as template_file:
            template_lines = template_file.readlines()

        md_script_lines = template_lines.copy()

        for i, line in enumerate(md_script_lines):
            if 'intra_op_threads = "placeholder"' in line:
                md_script_lines[i] = (
                    f'    intra_op_threads = {user_configs["num_intraop_threads"]}\n'
                )
            elif 'inter_op_threads = "placeholder"' in line:
                md_script_lines[i] = (
                    f'    inter_op_threads = {user_configs["num_interop_threads"]}\n'
                )
            elif 'relax_first = "placeholder"' in line:
                md_script_lines[i] = (
                    f'    relax_first = {user_configs["relax_first"]}\n'
                )
            elif 'ensemble = "placeholder"' in line:
                md_script_lines[i] = f'    ensemble = "{settings["ensemble"]}"\n'
            elif 'thermostat = "placeholder"' in line:
                md_script_lines[i] = f'    thermostat = "{settings["thermostat"]}"\n'
            elif 'taut = "placeholder"' in line:
                md_script_lines[i] = f'    taut = {user_configs["taut"]}\n'
            elif 'timestep = "placeholder"' in line:
                md_script_lines[i] = f'    timestep = {user_configs["timestep"]}\n'
            elif 'loginterval = "placeholder"' in line:
                md_script_lines[i] = (
                    f'    loginterval = {user_configs["loginterval"]}\n'
                )
            elif 'nsteps = "placeholder"' in line:
                md_script_lines[i] = f'    nsteps = {user_configs["nsteps"]}\n'
            elif 'temperature = "placeholder"' in line:
                md_script_lines[i] = (
                    f'    temperature = {user_configs["temperature"]}\n'
                )
            elif 'pressure = "placeholder"' in line:
                md_script_lines[i] = f'    pressure = {user_configs["pressure"]}\n'
            elif 'addn_args = {"placeholder": "placeholder"}' in line:
                md_script_lines[i] = f'    addn_args = {user_configs["addn_args"]}\n'

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

    for launch_dir, settings in launch_dirs.items():

        md_launcher = os.path.join(launch_dir, "sub.sh")

        if os.path.exists(md_launcher) and not remake:
            continue

        formula = launch_dir.split("/")[-4]
        struc_id = launch_dir.split("/")[-3]
        job_name = f'chgnet_md_{formula}_{struc_id}_{settings["ensemble"]}_{settings["thermostat"]}'

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
            f.write("python chgnet_md.py\n")

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


def submit_jobs(launch_dirs: dict) -> None:
    """
    Args:
        launch_dirs (dict): dict of launch directories and their settings

    Returns:
        None, submits jobs if not already in queue or finished
    """

    scripts_dir = os.getcwd()

    for launch_dir, settings in launch_dirs.items():

        job_name = f'chgnet_md_{settings["formula"]}_{settings["struc_id"]}_{settings["ensemble"]}_{settings["thermostat"]}'

        # check if job is already in queue
        if check_job_submission_status(job_name):
            print(f"\n{job_name} is already in queue")
            continue

        # check if job has already finished
        results = os.path.join(launch_dir, "chgnet_md_results.json")
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

    fjson = os.path.join(data_dir, "md_results.json")
    if os.path.exists(fjson) and not remake:
        return read_json(fjson)

    print("\nCollecting results:")

    results = {"md_results": {}, "md_configs": {}}

    # record MD job settings
    results["md_configs"]["relax_first"] = user_configs["relax_first"]
    results["md_configs"]["ensembles"] = user_configs["ensembles"]
    results["md_configs"]["thermostats"] = user_configs["thermostats"]
    results["md_configs"]["taut"] = user_configs["taut"]
    results["md_configs"]["timestep"] = user_configs["timestep"]
    results["md_configs"]["loginterval"] = user_configs["loginterval"]
    results["md_configs"]["nsteps"] = user_configs["nsteps"]
    results["md_configs"]["temperature"] = user_configs["temperature"]
    results["md_configs"]["pressure"] = user_configs["pressure"]
    results["md_configs"]["stress_weight"] = user_configs["stress_weight"]
    results["md_configs"]["addn_args"] = user_configs["addn_args"]

    # collect results
    for launch_dir, settings in launch_dirs.items():

        full_summary = os.path.join(launch_dir, "chgnet_md_results.json")

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

        # assign full summary to results dict
        results["md_results"][formula][struc_id][ensemble][thermostat] = full_summary

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
        len(results["md_results"][formula][struc_id][ensemble])
        for formula in results["md_results"]
        for struc_id in results["md_results"][formula]
        for ensemble in results["md_results"][formula][struc_id]
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
        results (dict): dict of MD results and configs
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
        * results["md_configs"]["nsteps"]
        // results["md_configs"]["loginterval"]
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

                    # full summary of MD results
                    full_summary = results["md_results"][formula][struc_id][ensemble][
                        thermostat
                    ]

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
                        new_struc_id = (
                            f"{struc_id}_{ensemble}_{thermostat}_md_struc_{i}"
                        )
                        if "structure" in lowest_energy_struc:
                            lowest_energy_strucs[new_struc_id] = lowest_energy_struc[
                                "structure"
                            ]
                        else:
                            print(
                                f"structure not found for {formula}_{struc_id}_{ensemble}_{thermostat}_bucket_{i}"
                            )

                    # setup strucs dict
                    if formula not in strucs:
                        strucs[formula] = {}

                    # assign lowest energy structures to strucs dict
                    strucs[formula].update(lowest_energy_strucs)

    # save lowest energy structures
    write_json(strucs, fjson)
    return read_json(fjson)

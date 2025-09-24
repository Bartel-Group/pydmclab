from __future__ import annotations
from typing import TYPE_CHECKING

import os
from shutil import copyfile, rmtree
import subprocess
import json
from collections import OrderedDict

from pydmclab.core.struc import StrucTools
from pydmclab.hpc.vasp import VASPSetUp
from pydmclab.hpc.analyze import AnalyzeVASP
from pydmclab.hpc.collector import Collector
from pydmclab.data.configs import load_base_configs, load_partition_configs

# this is the directory where this file is located (for path purposes)
HERE = os.path.dirname(os.path.abspath(__file__))

if TYPE_CHECKING:
    pass


class SubmitTools(object):
    """
    This class is focused on figuring out how to prepare chains of calculations
        the idea being that the output from this class is some .sh file that you can
            "submit" to a queueing system
        this class will automatically crawl through the VASP output files and figure out
            how to edit that input and submission files to finish the desired calculations

    This class also makes use of pydmclab.hpc.vasp.VASPSetUp to help with the VASP input files

    """

    def __init__(
        self,
        launch_dir: str,
        initial_magmom: list | None,
        ID_specific_vasp_configs: dict | bool = None,
        user_configs: dict = {},
    ) -> None:
        """
        Args:
            launch_dir (str)
                directory to launch calculations from (to submit the submission file)
                    assumes initial structure is POSCAR in launch_dir
                        pydmclab.hpc.launch.LaunchTools will put it there
                within this directory, various VASP calculation directories (calc_dirs) will be created
                        gga-loose, gga-relax, gga-static, etc
                            VASP will be run in each of these, but we need to run some sequentially, so we pack them together in one submission script
                 eg if <blah blah>/SrZrS3/perovskite/fm is launch_dir
                    then these are calc_dirs <blah blah>/SrZrS3/perovskite/fm/gga-relax, ..../gga-static, ..../metagga-relax, etc

            initial_magmom (list)
                if running AFM, list of magnetic moments for each atom in structure
                    generated using MagTools
                    you should pass this here or let pydmclab.hpc.launch.LaunchTools put it there
                if not AFM, no need to pass anything. VASPSetUp handles nm and fm

            ID_specific_vasp_configs (dict)
                any configs that pertain to a particular structure for a particular formula
                    usually passed here from your launch_dirs.json file (just like magmom)
                Should be formatted as:
                    {(kpoint, incar, potcar)_mods: {param: value}}
                    xc-calc is assumed to be 'all-all' in every case and should not be specified (as it is now)
                    e.g., {'incar_mods': {'ISMEAR': 0, 'SIGMA': 0.05}} or {'kpoints_mods': {'grid': [6, 6, 6]}}

            user_configs (dict)
                any non-default parameters you want to pass
                    see pydmclab.data.data._hpc_configs.yaml for defaults
                        relevant configs are SUB_CONFIGS and SLURM_CONFIGS

        """

        # where the submission script will be written to

        self.launch_dir = launch_dir

        # just a reminder of how a launch directory looks (one calculation is defined as some formula in some structure with some initial magnetic config)
        # NOTE: should be made with LaunchTools
        formula_indicator, struc_indicator, mag = launch_dir.split("/")[-3:]

        # load default configs
        _base_configs = load_base_configs()

        # merge default configs with user configs
        configs = {**_base_configs, **user_configs}

        if ID_specific_vasp_configs:
            for input_file in ["incar", "kpoints", "potcar"]:
                key = f"{input_file}_mods"
                if key in ID_specific_vasp_configs:

                    if not isinstance(ID_specific_vasp_configs[key], dict):
                        raise ValueError(
                            f"{key} should be a dictionary of modifications, not {type(ID_specific_vasp_configs[key])}."
                        )

                    for sub_key, sub_value in ID_specific_vasp_configs[key].items():
                        if isinstance(sub_value, dict):
                            raise ValueError(
                                f"{key} is more than 2 layers deep. Avoid specifying 'xc-calc' as 'all-all' is assumed."
                            )

                    configs[key].update({"all-all": ID_specific_vasp_configs[key]})

        # set mag based on launch_dir
        configs["mag"] = mag

        # set magmom based on what's passed here
        configs["magmom"] = initial_magmom

        # create copy of configs
        self.configs = configs.copy()

        # need a POSCAR to initialize setup
        # LaunchTools should take care of this
        fpos = os.path.join(launch_dir, "POSCAR")
        if not os.path.exists(fpos):

            raise FileNotFoundError(
                "Need a POSCAR to initialize setup; POSCAR not found in {}".format(
                    self.launch_dir
                )
            )
        else:
            self.structure = StrucTools(fpos).structure

        # load partition configurations to help with slurm setup
        partitions = load_partition_configs()
        self.partitions = partitions

        # these are the xcs we want energies for --> each one of these should have a submission script
        #  i.e., they are the end of individual chains (e.g., ['gga', 'metagga'])
        self.relaxation_xcs = self.configs["relaxation_xcs"]

        # these are addons to each static calculation (e.g., {'gga': ['lobster', 'bs']})
        self.static_addons = self.configs["static_addons"]

        # True if you want to start all your calcs with a loose relaxation
        self.start_with_loose = self.configs["start_with_loose"]

        # list of calculations (eg ['gga-lobster']) you want to re-run even if they've converged
        self.fresh_restart = self.configs["fresh_restart"]

        # where are does your launcher.py file live (where is this code being executed)
        self.scripts_dir = os.getcwd()

        # this is like what "project" are you in (e.g., perovskites if your launcher is /home/my_stuff/perovskites/scripts/launcher.py)
        self.job_dir = self.scripts_dir.split("/")[-2]

    @property
    def calc_list(self) -> list:
        """
        Returns:
            [xc-calc in the order they should be executed]

            e.g., if relaxation_xcs = ['gga', 'metagga'] and static_addons = {'gga': ['lobster']}
                ['gga-relax', 'gga-static', 'metagga-relax', 'metagga-static', 'gga-lobster']

            if configs['run_static_addons_before_all_relaxes'] = True:
                would change to ['gga-relax', 'gga-static', 'gga-lobster', 'metagga-relax', 'metagga-static']


        """
        configs = self.configs

        # if user asks for a custom calc list, return that
        if ("custom_calc_list" in configs) and (
            configs["custom_calc_list"] is not None
        ):
            return configs["custom_calc_list"]

        relaxation_xcs = self.relaxation_xcs
        static_addons = self.static_addons

        calcs = []

        # figure out if we need to run gga before other functionals (in case not explicitly specifeid in relaxation_xcs)
        if (
            ("gga" in relaxation_xcs)
            or ("metagga" in relaxation_xcs)
            or ("metaggau" in relaxation_xcs)
            or ("hse06" in relaxation_xcs)
        ):
            first_xc = "gga"
        elif "ggau" in relaxation_xcs:
            first_xc = "ggau"
        elif len(relaxation_xcs) == 1:
            first_xc = relaxation_xcs[0]

        # if starting with a loose, make sure very first calc is loose
        if self.start_with_loose:
            first_xc_calcs = ["loose", "relax", "static"]
        else:
            first_xc_calcs = ["relax", "static"]

        calcs += ["-".join([first_xc, calc]) for calc in first_xc_calcs]

        # if we preappended some functional to get us started, make sure we don't duplicate
        if first_xc not in relaxation_xcs:
            xcs = [first_xc] + relaxation_xcs
        else:
            xcs = relaxation_xcs

        # add relaxations, statics, and static addons in the specified order
        for xc in xcs:
            if xc != first_xc:
                # we already added the first xc's calcs
                calcs += ["-".join([xc, calc]) for calc in ["relax", "static"]]
            if configs["run_static_addons_before_all_relaxes"]:
                # if we want to prioritize statics, add them to each functional as we go through
                calcs += ["-".join([xc, calc]) for calc in static_addons[xc]]
        if not configs["run_static_addons_before_all_relaxes"]:
            # if we want to prioritize relaxes, add the static addons to the end
            for xc in xcs:
                if xc in static_addons:
                    calcs += ["-".join([xc, calc]) for calc in static_addons[xc]]
        if ("hse06" not in relaxation_xcs) and ("hse06" in static_addons):
            # if we're running hse06 addons but not hse06 relaxations and statics
            calcs += ["-".join(["hse06", "preggastatic"])]
            calcs += ["hse06-" + addon for addon in static_addons["hse06"]]

        final_calcs = []
        for xc_calc in calcs:
            xc, calc = xc_calc.split("-")
            if calc in ["lobster", "bs", "parchg"]:
                final_calcs.append("-".join([xc, "prelobster"]))
                if calc == "parchg":
                    final_calcs.append("-".join([xc, "lobster"]))
            final_calcs.append(xc_calc)

        ordered_calcs = OrderedDict.fromkeys(final_calcs)
        final_minimal_calcs = list(ordered_calcs.keys())
        return final_minimal_calcs

    @property
    def queue_manager(self) -> str:
        """
        Returns queue manager (eg #SBATCH)
        """
        return self.configs["manager"]

    @property
    def slurm_options(self) -> dict:
        """
        Returns dictionary of slurm options {option (str) : value (str, float, int, bool}
            nodes, ntasks, walltime, etc

        To be written at the top of submission files
        """
        possible_options = [
            "nodes",
            "ntasks",
            "time",
            "error",
            "output",
            "account",
            "partition",
            "job-name",
            "mem-per-cpu",
            "mem-per-gpu",
            "constraint",
            "qos",
        ]
        configs = self.configs.copy()
        options = {
            option: configs[option] for option in possible_options if configs[option]
        }
        partitions = self.partitions.copy()
        if options["partition"] in partitions:
            partition_specs = partitions[options["partition"]]

            # make special amendments for GPU partitions
            if partition_specs["proc"] == "gpu":
                options["nodes"] = 1
                options["ntasks"] = 1
                options["gres"] = "gpu:%s:%s" % (
                    options["partition"].split("-")[0],
                    str(options["nodes"]),
                )

            # reduce walltime if needed
            max_time = partition_specs["max_wall"] / 60
            if options["time"] > max_time:
                options["time"] = max_time

            # reduce nodes if needed
            max_nodes = int(partition_specs["max_nodes"])
            if options["nodes"] > max_nodes:
                options["nodes"] = max_nodes

            # reduce ntasks if needed
            max_cores_per_node = int(partition_specs["cores_per_node"])
            if options["ntasks"] / options["nodes"] > max_cores_per_node:
                options["ntasks"] = max_cores_per_node * options["nodes"]

            # warn user if they are using non-sharing nodes but not requesting all cores
            if not partition_specs["sharing"]:
                if options["ntasks"] / options["nodes"] < max_cores_per_node:
                    print(
                        "WARNING: this node cant be shared but youre not using all of it ?"
                    )
        return options

    @property
    def bin_dir(self) -> str:
        """
        Returns bin directory (str) where things (eg LOBSTER) are located
        """
        configs = self.configs.copy()
        machine = configs["machine"]
        if machine == "msi":
            return "/home/cbartel/shared/bin"
        elif machine == "bridges2":
            return "/ocean/projects/mat230011p/shared/bin"
        elif machine == "expanse":
            return "/home/%s/bin/" % os.getlogin()
        else:
            raise NotImplementedError('dont have bin path for machine "%s"' % machine)

    @property
    def vasp_dir(self) -> str:
        """
        Returns directory (str) containing vasp executable

        MSI has VASP5 or VASP6 (set by configs['vasp_version'])
        Bridges has only VASP6
        """
        configs = self.configs.copy()
        machine = configs["machine"]
        version = configs["vasp_version"]
        if machine == "msi":
            preamble = "%s/vasp" % self.bin_dir
            if version == 5:
                return "%s/vasp.5.4.4.pl2" % preamble
            elif version == 6:
                return "%s/vasp.6.4.1" % preamble
            elif version == 7:
                return "%s/vasp.6.5.1" % preamble
            else:
                raise NotImplementedError(f"VASP version {version} not supported on MSI")
        elif machine == "bridges2":
            if version == 6:
                return "/opt/packages/VASP/VASP6/6.4.1/INTEL"
            else:
                raise NotImplementedError("VASP < 6 not on Bridges?")
        else:
            raise NotImplementedError('dont have VASP path for machine "%s"' % machine)

    @property
    def vasp_command(self) -> str:
        """
        Returns command used to execute vasp (str)
            e.g., 'srun -n 24 PATH_TO_VASP/vasp_std > vasp.o' (if mpi_command == 'srun')
            e.g., 'mpirun -np 24 PATH_TO_VASP/vasp_std > vasp.o' (if mpi_command == 'mpirun')
        """
        configs = self.configs.copy()
        vasp_exec = os.path.join(self.vasp_dir, configs["vasp"])

        if configs["mpi_command"] == "srun":
            return "\n%s --ntasks=%s --mpi=pmi2 %s > %s\n" % (
                configs["mpi_command"],
                str(configs["ntasks"]),
                vasp_exec,
                configs["fvaspout"],
            )
        elif configs["mpi_command"] == "mpirun":
            return "\n%s -np=%s %s > %s\n" % (
                configs["mpi_command"],
                str(configs["ntasks"]),
                vasp_exec,
                configs["fvaspout"],
            )

    @property
    def lobster_dir(self) -> str:
        """
        Returns directory (str) containing lobster executable

        MSI has LOBSTER4 or LOBSTER5 (set by configs['lobster_version'])
        Bridges has only LOBSTER4
        """
        configs = self.configs.copy()
        machine = configs["machine"]
        version = configs["lobster_version"]
        if machine == "msi":
            preamble = "%s/lobster" % self.bin_dir
            if version == 4:
                return "%s/lobster-4.1.0/lobster-4.1.0" % preamble
            elif version == 5:
                return "%s/lobster-5.1.0/lobster-5.1.0" % preamble
        elif machine == "bridges2":
            if version == 4:
                return "%s/lobster-4.1.0/lobster-4.1.0" % preamble
            else:
                raise NotImplementedError("LOBSTER > 4 not on Bridges?")
        else:
            raise NotImplementedError(
                'dont have LOBSTER path for machine "%s"' % machine
            )

    @property
    def lobster_command(self) -> str:
        """
        Returns command used to execute lobster (str)
        """
        lobster_path = self.lobster_dir
        # lobster_path = os.path.join(
        #     self.bin_dir, "lobster", "lobster-4.1.0", "lobster-4.1.0"
        # )
        return "\n%s\n" % lobster_path

    @property
    def bader_command(self) -> str:
        """
        Returns command used to execute bader (str)
        """
        chgsum = "%s/bader/chgsum.pl AECCAR0 AECCAR2" % self.bin_dir
        bader = "%s/bader/bader CHGCAR -ref CHGCAR_sum" % self.bin_dir
        return "\n%s\n%s\n" % (chgsum, bader)

    @property
    def job_name(self) -> str:
        """
        Returns job name based on launch_dir (str)
            eg if launch_dir = /home/my_stuff/perovskites/SrZrS3/perovskite/fm
                job_name = perovskites.SrZrS3.perovskite.fm
        """
        return ".".join(self.launch_dir.split("/")[-3:]) + "." + self.job_dir

    @property
    def is_job_in_queue(self) -> bool:
        """

        Returns:
            True if this job-name is already in the queue, else False

        will prevent you from messing with directories that have running/pending jobs
        """
        job_name = self.job_name

        # create a temporary file w/ jobs in queue with my username and this job_name
        # also double-checks if this job is in the msidmc partition
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

    @property
    def statuses(self) -> dict:
        """
        Returns dictionary of statuses for each calculation in the chain
            {xc-calc : status}

        statuses:
            new: this should be treated as a totally new calculation (never been executed)
            queued: this job is already in the queue
            continue: this job ran previously, but is not finished
            done: this job has already been finished
        """

        configs = self.configs.copy()

        fresh_restart = configs["fresh_restart"]
        if fresh_restart is None:
            fresh_restart = []

        launch_dir = self.launch_dir

        calc_list = self.calc_list

        job_in_q = self.is_job_in_queue
        if job_in_q:
            return {xc_calc: "queued" for xc_calc in calc_list}

        fpos_src = os.path.join(launch_dir, "POSCAR")

        # loop through all calculations and collect statuses
        statuses = {}

        # looping through each VASP calc in list of calculations
        for xc_calc in calc_list:

            # restart if this xc_calc is in your fresh_restart list
            restart_this_one = True if xc_calc in fresh_restart else False

            # (0) update vasp configs with the current xc and calc
            xc_to_run, calc_to_run = xc_calc.split("-")
            configs["xc_to_run"] = xc_to_run
            configs["calc_to_run"] = calc_to_run

            # (1) make calc_dir (or remove and remake if fresh_restart)
            calc_dir = os.path.join(launch_dir, xc_calc)

            if os.path.exists(calc_dir) and restart_this_one:
                rmtree(calc_dir)
            if not os.path.exists(calc_dir):
                os.mkdir(calc_dir)

            # if restart_this_one:
            #     # if restarting, status = new
            #     statuses[xc_calc] = "new"
            #     continue

            # (2) check convergence of current calc
            E_per_at = AnalyzeVASP(calc_dir).E_per_at
            convergence = True if E_per_at else False
            if convergence:
                # if calc is converged
                statuses[xc_calc] = "done"

            else:
                # now we're dealing with calcs that are not converged

                # (3) check for POSCAR
                fpos_dst = os.path.join(calc_dir, "POSCAR")
                if os.path.exists(fpos_dst):
                    # if there is a POSCAR, make sure its not empty
                    with open(fpos_dst, "r", encoding="utf-8") as f_tmp:
                        contents = f_tmp.readlines()
                    # if its empty, copy the initial structure to calc_dir
                    if len(contents) == 0:
                        copyfile(fpos_src, fpos_dst)

                # if theres no POSCAR, copy the initial structure to calc_dir
                if not os.path.exists(fpos_dst):
                    copyfile(fpos_src, fpos_dst)

                # (4) check for CONTCAR. if one exists, if its not empty, and if not fresh_restart, mark this job as one to "continue"
                # (ie later, we'll copy CONTCAR to POSCAR); otherwise, mark as new
                fcont_dst = os.path.join(calc_dir, "CONTCAR")
                if os.path.exists(fcont_dst):
                    with open(fcont_dst, "r", encoding="utf-8") as f_tmp:
                        contents = f_tmp.readlines()
                    if len(contents) > 0:
                        statuses[xc_calc] = "continue"
                        copyfile(fcont_dst, fpos_dst)
                    else:
                        statuses[xc_calc] = "new"
                else:
                    statuses[xc_calc] = "new"
            if (calc_to_run == "prelobster") and os.path.exists(
                os.path.join(calc_dir, "IBZKPT")
            ):
                statuses[xc_calc] = "done"
            if (calc_to_run == "parchg") and os.path.exists(
                os.path.join(calc_dir, "PARCHG")
            ):
                statuses[xc_calc] = "done"

        return statuses

    @property
    def prepare_directories(self) -> dict:
        """
        Given the statuses dictionary, prepare (update) each VASP calculation directory accordingly
        """
        statuses = self.statuses
        configs = self.configs.copy()
        launch_dir = self.launch_dir
        calc_list = self.calc_list

        for xc_calc in calc_list:
            status = statuses[xc_calc]
            if status in ["done"]:
                print("  %s is %s\n" % (xc_calc, status))
                # no work needs to be done for finished or queued calcs
                continue

            if status in ["queued"]:
                continue

            xc_to_run, calc_to_run = xc_calc.split("-")

            # get our configs before error handling
            configs_before_error_handling = configs.copy()
            configs_before_error_handling["xc_to_run"] = xc_to_run
            configs_before_error_handling["calc_to_run"] = calc_to_run

            calc_dir = os.path.join(launch_dir, xc_calc)

            # (5) initialize VASPSetUp with current VASP configs for this calculation
            vsu = VASPSetUp(
                calc_dir=calc_dir,
                user_configs=configs_before_error_handling,
            )
            updated_configs = vsu.configs.copy()

            # (6) check for errors in continuing and new jobs
            incar_changes = {}
            if status not in ["continue", "new"]:
                raise ValueError(
                    "something strange happened. %s status is not done, queued, continue, or new"
                    % xc_calc
                )
            is_calc_clean = vsu.is_clean
            if not is_calc_clean:
                # change INCAR based on errors and include in calc_configs
                incar_changes = vsu.incar_changes_from_errors

            # (7) if there are INCAR updates, add them to calc_configs as incar_mods
            if incar_changes:
                if xc_calc in updated_configs["incar_mods"]:
                    updated_configs["incar_mods"][xc_calc].update(incar_changes)
                else:
                    updated_configs["incar_mods"][xc_calc] = incar_changes.copy()

            # (8) write revised VASP input files to calc_dir
            vsu = VASPSetUp(calc_dir=calc_dir, user_configs=updated_configs)
            vsu.prepare_calc

            print("  %s is prepared\n" % xc_calc)
        return statuses

    def collection_status(self, xc_calc: str) -> bool:
        """
        Returns:
            True if no need to run collector
            False if need to run collector
        """
        xc, calc = xc_calc.split("-")
        if calc in ["prelobster"]:
            return True
        configs = self.configs.copy()
        launch_dir = self.launch_dir
        calc_dir = os.path.join(launch_dir, xc_calc)
        collector = Collector(calc_dir, configs)
        if collector.already_collected:
            return True
        else:
            return False

    @property
    def write_sub(self) -> bool:
        """
        A lot going on here. The objective is to write a submission script for each pack of VASP calculations
            each submission script will launch a chain of jobs
            this gets a bit tricky because a submission script is executed in bash
                it's essentially like moving to a compute node and typing each line of the submission script into the compute node's command line
                this means we can't really use python while the submission script is being executed

        1) check if job's in queue. if it is, just return (ie don't work on that job)

        2) write our slurm options at the top of sub file (#SBATCH ...)

        3) loop through all the calculations we want to do from this launch dir
            label them as "done", "continue", or "new"

        4) for "continue"
            copy CONTCAR to POSCAR to save progress

        5) for "new" and "continue"
            figure out what parent calculations to get data from
                e.g., gga-static for metagga-relax
            make sure that parent calculation finished without errors before passing data to next calc
                and before running next calc
                if a parent calc didnt finish, but we've moved onto the next job, kill the job, so we can (automatically) debug the parent calc

        6) write VASP commands

        7) if lobster_static and calc is static, write LOBSTER and BADER commands
        """
        # get launch_dir (where sub.sh gets written)
        launch_dir = self.launch_dir

        print("\n~~~~~~~~~~~~~~~~~~~~~~~\n\npreparing %s\n" % launch_dir)
        if self.is_job_in_queue:
            print("\nqueued as %s\n" % self.job_name)
            return

        # get configs dict
        configs = self.configs.copy()

        # get slurm-specific configs since these get handled differently
        slurm_options = self.slurm_options.copy()
        slurm_options["job-name"] = self.job_name

        queue_manager = self.queue_manager

        # get calc_list (what gets populated in sub.sh)
        calc_list = self.calc_list

        # get the statuses and prepare VASP calcs accordingly
        statuses = self.prepare_directories

        # start the submission script
        fsub = os.path.join(launch_dir, "sub.sh")

        # say where we want statuses to be echoed to
        fstatus = os.path.join(launch_dir, "status.o")

        with open(fsub, "w", encoding="utf-8") as f:
            # write the bin bash stuff at the top
            f.write("#!/bin/bash -l\n")

            # write the SLURM stuff (partition, nodes, time, etc) at the top
            for key in slurm_options:
                slurm_option = slurm_options[key]
                if slurm_option:
                    f.write("%s --%s=%s\n" % (queue_manager, key, str(slurm_option)))
            f.write("\n\n")

            # this is for running MPI jobs that may require large memory
            f.write("ulimit -s unlimited\n")

            # load certain modules if needed for MPI command
            if configs["mpi_command"] == "mpirun":
                if configs["machine"] == "msi":
                    if configs["vasp_version"] == 5:
                        f.write("module load impi/2018/release_multithread\n")
                    elif configs["vasp_version"] == 6:
                        unload = [
                            "mkl",
                            "intel/2018.release",
                            "intel/2018/release",
                            "impi/2018/release_singlethread",
                            "mkl/2018.release",
                            "impi/intel",
                        ]
                        load = ["mkl/2021/release", "intel/cluster/2021"]
                        for module in unload:
                            f.write("module unload %s\n" % module)
                        for module in load:
                            f.write("module load %s\n" % module)
                elif configs["machine"] == "bridges2":
                    f.write(
                        "module load intelmpi/2021.3.0-intel2021.3.0\nsource ~/.bashrc\nexport OMP_NUM_THREADS=1\n"
                    )

            for xc_calc in calc_list:
                status = statuses[xc_calc]
                xc_to_run, calc_to_run = xc_calc.split("-")

                # write status to status.o
                f.write('\necho "%s is %s" >> %s\n' % (xc_calc, status, fstatus))

                if status == "queued":
                    # don't do anything for queued calcs
                    continue

                # find our calc_dir (where VASP is executed for this xc_calc)
                calc_dir = os.path.join(launch_dir, xc_calc)

                if status == "done":
                    if calc_to_run in ["lobster", "bs"]:
                        if not os.path.exists(os.path.join(calc_dir, "lobsterout")):
                            f.write("\ncd %s\n" % calc_dir)
                            f.write(self.lobster_command)
                    if not self.collection_status(xc_calc):
                        # execute the collector (writes)
                        f.write("\ncd %s\n" % self.scripts_dir)
                        f.write(
                            "python collector.py %s %s \n"
                            % (calc_dir, os.path.join(self.scripts_dir, "configs.json"))
                        )
                    continue

                # retrieve the incar_mods that pertain to this particular calculation
                configs["xc_to_run"] = xc_to_run
                configs["calc_to_run"] = calc_to_run
                vsu = VASPSetUp(calc_dir=calc_dir, user_configs=configs)
                incar_mods = vsu.configs["modify_this_incar"]

                # get the info that must be read by the Passer between calcs
                passer_dict = {
                    "xc_calc": xc_calc,
                    "calc_list": calc_list,
                    "calc_dir": calc_dir,
                    "incar_mods": incar_mods,
                    "launch_dir": launch_dir,
                    "struc_src_for_hse": configs["struc_src_for_hse"],
                    "xc_calc_src_for_lobster": configs["xc_calc_src_for_lobster"],
                }
                passer_dict_as_str = json.dumps(passer_dict)

                # execute the passer
                f.write("\ncd %s\n" % self.scripts_dir)
                f.write("python passer.py '%s' \n" % passer_dict_as_str)

                # based on passer output, decide whether or not we need to cancel this job
                f.write(
                    "isInFile=$(cat %s | grep -c %s)\n"
                    % (os.path.join(launch_dir, "job_killer.o"), "kill")
                )
                f.write("if [ $isInFile -ge 1 ]; then\n")
                f.write("   scancel $SLURM_JOB_ID\n")
                f.write("fi\n")

                # (presuming we didn't cancel the job) go to calc_dir and run VASP
                f.write("cd %s\n" % calc_dir)
                f.write(self.vasp_command)

                # run lobster for certain static-addons
                if calc_to_run in ["lobster", "bs"]:
                    f.write(self.lobster_command)

                # run bader for all static jobs
                if calc_to_run in ["static", "lobster"]:
                    f.write(self.bader_command)

                # execute the collector
                f.write("\ncd %s\n" % self.scripts_dir)
                f.write(
                    "python collector.py %s %s \n"
                    % (calc_dir, os.path.join(self.scripts_dir, "configs.json"))
                )

            # nothing left to do, so cancel the job (sometimes done jobs will hang)
            f.write("\n\nscancel $SLURM_JOB_ID\n")
        print("wrote %s\n" % fsub)
        return True

    @property
    def launch_sub(self) -> None:
        """
        launch the submission script written in write_sub
            if job is not in queue already
            if there's something to launch
                (ie if all calcs are done, dont launch)
        """

        if self.is_job_in_queue:
            return

        configs = self.configs.copy()
        # shouldn't need to check this since it gets checked in statuses
        # if self.is_job_in_queue:
        #     return

        scripts_dir = self.scripts_dir
        launch_dir = self.launch_dir

        # determine what keywords to look for to see if job needs to be launched
        flags_that_need_to_be_executed = configs["execute_flags"]

        # see if there's anything to launch (eg don't launch if the submission script is just gonna echo done)
        fsub = os.path.join(launch_dir, "sub.sh")
        needs_to_launch = False
        with open(fsub, "r", encoding="utf-8") as f:
            contents = f.read()
            for flag in flags_that_need_to_be_executed:
                if flag in contents:
                    needs_to_launch = True

        if not needs_to_launch:
            print("nothing to launch in %s\n" % fsub)
            return

        # if we made it this far, launch it
        os.chdir(launch_dir)
        subprocess.call(["sbatch", "sub.sh"])
        os.chdir(scripts_dir)

        print("submitted %s\n" % fsub)


def main():
    """
    Execute code here for debugging purposes
    """
    return


if __name__ == "__main__":
    main()

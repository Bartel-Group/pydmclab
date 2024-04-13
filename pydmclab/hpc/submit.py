from pydmclab.core.struc import StrucTools
from pydmclab.hpc.vasp import VASPSetUp
from pydmclab.hpc.analyze import AnalyzeVASP
from pydmclab.utils.handy import read_yaml, write_yaml
from pydmclab.data.configs import (
    load_vasp_configs,
    load_slurm_configs,
    load_sub_configs,
    load_partition_configs,
)

from pymatgen.core.structure import Structure
from pymatgen.io.lobster import Lobsterin


import multiprocessing as multip
import os
from shutil import copyfile, rmtree
import subprocess
import warnings
import json

from subprocess import call

from pymatgen.io.vasp.sets import get_structure_from_prev_run

HERE = os.path.dirname(os.path.abspath(__file__))


class SubmitTools(object):
    """
    This class is focused on figuring out how to prepare chains of calculations
        the idea being that the output from this class is some file that you can
            "submit" to a queueing system
        this class will automatically crawl through the VASP output files and figure out
            how to edit that submission file to finish the desired calculations

    """

    def __init__(
        self,
        launch_dir,
        initial_magmom,
        user_configs={},
    ):
        """
        Args:
            launch_dir (str)
                directory to launch calculations from (to submit the submission file)
                    assumes initial structure is POSCAR in launch_dir
                        pydmclab.hpc.launch.LaunchTools will put it there
                within this directory, various VASP calculation directories (calc_dirs) will be created
                        gga-loose, gga-relax, gga-static, etc
                            VASP will be run in each of these, but we need to run some sequentially, so we pack them together in one submission script

            relaxation_xcs (list)
                list of xcs we want to do at least relax+static on

            stastic_addons (dict)
                dictionary of additional calculations to do after static

            magmom (list)
                list of magnetic moments for each atom in structure (or None if not AFM)
                    you should pass this here or let pydmclab.hpc.launch.LaunchTools put it there
                None if not AFM
                if AFM, pull it from a dictionary of magmoms you made with MagTools

            fresh_restart (list)
                list of calculations to start over

            user_configs (dict)
                any non-default parameters you want to pass
                these will override the defaults in the yaml files
                look at pydmc/data/data/*configs*.yaml for the defaults
                    note: _launch_configs.yaml will be modified using LaunchTools
                you should be passing stuff here!
                    VASPSetUp will expect xc_to_run, calc_to_run, standard, and mag
                        xc_to_run and calc_to_run will be passed based on final_xcs and sub_configs['packing']
                        standard and mag will get passed to it based on launch_dir
                you can also pass any settings in _vasp_configs.yaml, _slurm_configs.yaml, or _sub_configs.yaml here
                    _vasp_configs options:
                        how to manipulate the VASP inputs
                        see pydmclab.hpc.vasp or pydmclab.data.data._vasp_configs.yaml for options
                    _sub_configs options:
                        how to manipulate the executables in the submission script and packing of calculations
                        see pydmclab.hpc.submit or pydmclab.data.data._sub_configs.yaml for options
                    _slurm_configs options:
                        how to manipulate the slurm tags in the submission script
                        see pydmclab.data.data._slurm_configs.yaml for options

            vasp_configs_yaml (os.pathLike)
                path to yaml file containing baseline vasp configs
                    there's usually no reason to change this
                    this holds some default configs for VASP
                    can always be changed with user_configs

             slurm_configs_yaml (os.pathLike)
                path to yaml file containing baseline slurm configs
                    there's usually no reason to change this
                    this holds some default configs for slurm
                    can always be changed with user_configs

            sub_configs_yaml (os.pathLike)
                path to yaml file containing baseline submission file configs
                    there's usually no reason to change this
                    this holds some default configs for submission files
                    can always be changed with user_configs

            refresh_configs (bool)
                if True, will copy pydmclab baseline configs to your local directory
                    this is useful if you've made changes to the configs files in the directory you're working in and want to start over

        Returns:
            self.launch_dir (os.pathLike)
                directory to launch calculations from

            self.slurm_configs (dict)
                dictionary of slurm configs (in format similar to yaml)

            self.vasp_configs (dict)
                dictionary of vasp configs (in format similar to yaml)

            self.sub_configs (dict)
                dictionary of submission configs (in format similar to yaml)

            self.structure (Structure)
                pymatgen structure object from launch_dir/POSCAR

            self.partitions (dict)
                dictionary of info regarding partition configurations on MSI

            self.final_xcs (list)
                list of exchange correlation methods we want the final energy for
        """

        self.launch_dir = launch_dir

        # just a reminder of how a launch directory looks
        # NOTE: should be made with LaunchTools
        formula_indicator, struc_indicator, mag = launch_dir.split("/")[-3:]

        vasp_configs = load_vasp_configs()
        slurm_configs = load_slurm_configs()
        sub_configs = load_sub_configs()

        # we're going to modify vasp, slurm, and sub configs using one user_configs dict, so let's keep track of what's been applied
        user_configs_used = []

        for option in slurm_configs:
            if option in user_configs:
                if option not in user_configs_used:
                    new_value = user_configs[option]
                    slurm_configs[option] = new_value
                    user_configs_used.append(option)

        # create copy of slurm_configs to prevent unwanted updates
        self.slurm_configs = slurm_configs.copy()

        for option in sub_configs:
            if option in user_configs:
                if option not in user_configs_used:
                    new_value = user_configs[option]
                    sub_configs[option] = new_value
                    user_configs_used.append(option)

        # create copy of sub_configs to prevent unwanted updates
        self.sub_configs = sub_configs.copy()

        for option in vasp_configs:
            if option in user_configs:
                if option not in user_configs_used:
                    new_value = user_configs[option]
                    vasp_configs[option] = new_value
                    user_configs_used.append(option)

        # determine standard and mag from launch_dir
        vasp_configs["mag"] = mag

        # include magmom in vasp_configs
        vasp_configs["magmom"] = initial_magmom

        # create copy of vasp_configs to prevent unwanted updates
        self.vasp_configs = vasp_configs.copy()

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
            self.structure = Structure.from_file(fpos)

        # load partition configurations to help with slurm setup
        partitions = load_partition_configs()
        self.partitions = partitions

        # these are the xcs we want energies for --> each one of these should have a submission script
        # i.e., they are the end of individual chains
        self.relaxation_xcs = self.sub_configs["relaxation_xcs"]
        self.static_addons = self.sub_configs["static_addons"]
        self.start_with_loose = self.sub_configs["start_with_loose"]
        self.fresh_restart = self.sub_configs["fresh_restart"]
        self.scripts_dir = os.getcwd()
        self.job_dir = self.scripts_dir.split("/")[-2]

    @property
    def calc_list(self):
        """
        Returns:
            [xc-calc in the order they should be executed]
        """
        sub_configs = self.sub_configs
        if ("custom_calc_list" in sub_configs) and (
            sub_configs["custom_calc_list"] is not None
        ):
            return sub_configs["custom_calc_list"]

        relaxation_xcs = self.relaxation_xcs
        static_addons = self.static_addons

        calcs = []

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

        if self.start_with_loose:
            first_xc_calcs = ["loose", "relax", "static"]
        else:
            first_xc_calcs = ["relax", "static"]

        calcs += ["-".join([first_xc, calc]) for calc in first_xc_calcs]
        if first_xc in static_addons:
            calcs += ["-".join([first_xc, calc]) for calc in static_addons[first_xc]]

        for xc in relaxation_xcs:
            if xc == first_xc:
                continue
            calcs += ["-".join([xc, calc]) for calc in ["relax", "static"]]
            if xc in static_addons:
                calcs += ["-".join([xc, calc]) for calc in static_addons[xc]]

        return calcs

    @property
    def queue_manager(self):
        """
        Returns queue manager (eg #SBATCH)
        """
        return self.sub_configs["manager"]

    @property
    def slurm_options(self):
        """
        Returns dictionary of slurm options
            - nodes, ntasks, walltime, etc

        To be written at the top of submission files
        """
        slurm_configs = self.slurm_configs.copy()
        options = {
            option: slurm_configs[option]
            for option in slurm_configs
            if slurm_configs[option]
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
    def bin_dir(self):
        """
        Returns bin directory where things (eg LOBSTER) are located
        """
        sub_configs = self.sub_configs.copy()
        machine = sub_configs["machine"]
        if machine == "msi":
            return "/home/cbartel/shared/bin"
        elif machine == "bridges2":
            return "/ocean/projects/mat230011p/shared/bin"
        elif machine == "expanse":
            return "/home/%s/bin/" % os.getlogin()
        else:
            raise NotImplementedError('dont have bin path for machine "%s"' % machine)

    @property
    def vasp_dir(self):
        """
        Returns directory containing vasp executable
        """
        machine = self.sub_configs["machine"]
        version = self.sub_configs["vasp_version"]
        if machine == "msi":
            preamble = "%s/vasp" % self.bin_dir
            if version == 5:
                return "%s/vasp.5.4.4.pl2" % preamble
            elif version == 6:
                return "%s/vasp.6.4.1" % preamble
        elif machine == "bridges2":
            if version == 6:
                return "/opt/packages/VASP/VASP6/6.3+VTST"
            else:
                raise NotImplementedError("VASP < 6 not on Bridges?")
        else:
            raise NotImplementedError('dont have VASP path for machine "%s"' % machine)

    @property
    def vasp_command(self):
        """
        Returns command used to execute vasp
            e.g., 'srun -n 24 PATH_TO_VASP/vasp_std > vasp.o' (if mpi_command == 'srun')
            e.g., 'mpirun -np 24 PATH_TO_VASP/vasp_std > vasp.o' (if mpi_command == 'mpirun')
        """
        sub_configs = self.sub_configs.copy()
        vasp_configs = self.vasp_configs.copy()
        vasp_exec = os.path.join(self.vasp_dir, sub_configs["vasp"])
        slurm_options = self.slurm_options.copy()

        if sub_configs["mpi_command"] == "srun":
            return "\n%s --ntasks=%s --mpi=pmi2 %s > %s\n" % (
                sub_configs["mpi_command"],
                str(slurm_options["ntasks"]),
                vasp_exec,
                vasp_configs["fvaspout"],
            )
        elif sub_configs["mpi_command"] == "mpirun":
            return "\n%s -np=%s %s > %s\n" % (
                sub_configs["mpi_command"],
                str(slurm_options["ntasks"]),
                vasp_exec,
                vasp_configs["fvaspout"],
            )

    @property
    def lobster_command(self):
        """
        Returns command used to execute lobster
        """
        lobster_path = os.path.join(
            self.bin_dir, "lobster", "lobster-4.1.0", "lobster-4.1.0"
        )
        return "\n%s\n" % lobster_path

    @property
    def bader_command(self):
        """
        Returns command used to execute bader
        """
        chgsum = "%s/bader/chgsum.pl AECCAR0 AECCAR2" % self.bin_dir
        bader = "%s/bader/bader CHGCAR -ref CHGCAR_sum" % self.bin_dir
        return "\n%s\n%s\n" % (chgsum, bader)

    @property
    def job_name(self):
        """
        Returns job name based on launch_dir
        """
        return ".".join(self.launch_dir.split("/")[-3:]) + "." + self.job_dir

    @property
    def is_job_in_queue(self):
        """
        Args:
            job_name (str)
                name of job to check if in queue
                generated automatically by SubmitTools.prepare_directories using launch_dir
                    ".".join(launch_dir.split("/")[-4:] + [final_xc])

                beware that having two calculations with the same name will cause problems

        Returns:
            True if this job-name is already in the queue, else False

        will prevent you from messing with directories that have running/pending jobs
        """
        job_name = self.job_name
        # create a file w/ jobs in queue with my username and this job_name
        scripts_dir = os.getcwd()
        fqueue = os.path.join(scripts_dir, "_".join(["q", job_name]) + ".o")
        with open(fqueue, "w") as f:
            subprocess.call(
                ["squeue", "-u", "%s" % os.getlogin(), "--name=%s" % job_name], stdout=f
            )

        # get the job names I have in the queue
        names_in_q = []
        with open(fqueue) as f:
            for line in f:
                if "PARTITION" not in line:
                    names_in_q.append([v for v in line.split(" ") if len(v) > 0][2])

        # delete the file I wrote w/ the queue output
        os.remove(fqueue)

        # if this job is in the queue, return True
        if len(names_in_q) > 0:
            print(" !!! job already in queue, not messing with it")
            return True

        print(" not in queue, onward --> ")
        return False

    @property
    def statuses(self):
        """
        This gets called by SubmitTools.write_sub, so you should rarely call this on its own

        A lot going on here. The objective is to prepare a set of directories for all calculations of interest to a given submission script in a given launch_dir
            note: 1 submission script --> 1 chain (pack) of VASP calculations
            note: 1 launch_dir --> can have > 1 submission script (if working towards >1 "final_xc" like ['metagga', 'ggau'])

        1) For each xc-calc pair, create a directory (calc_dir)
            */launch_dir/xc-calc
                xc-calc could be gga-loose, metagga-relax, etc.

        2) Check if that calc_dir has a converged VASP job
            note: also checks "parents" (ie a static is labeled unconverged if its relax is unconverged)
                parents determined by sub_configs['packing']

            if calc and parents are converged:
                checks sub_configs['fresh_restart']
                    if fresh_restart = False --> label calc_dir as status='done' and move on
                    if fresh_restart = True --> label calc_dir as status='new' and start this calc over

        3) Put */launch_dir/POSCAR into */launch_dir/xc-calc/POSCAR if there's not a POSCAR there already

        4) Check if */calc_dir/CONTCAR exists and has data in it,
            if it does, copy */calc_dir/CONTCAR to */calc_dir/POSCAR and label status='continue' (ie continuing job)
            if it doesn't, label status='new' (ie new job)

        5) Initialize VASPSetUp for calc_dir
            modifies VASPSetUp(calc_dir).get_vasp_input with self.vasp_configs

        6) If status in ['continue', 'new'],
            check for errors using VASPSetUp(calc_dir).error_msgs
                may remove WAVECAR/CHGCAR
                will likely make edits to INCAR
        """

        # make copies of relevant configs dicts
        vasp_configs = self.vasp_configs.copy()
        sub_configs = self.sub_configs.copy()

        fresh_restart = sub_configs["fresh_restart"]
        launch_dir = self.launch_dir

        calc_list = self.calc_list

        print("\n\n~~~~~ starting to work on %s ~~~~~\n\n" % launch_dir)

        job_in_q = self.is_job_in_queue
        if job_in_q:
            return {xc_calc: "queued" for xc_calc in calc_list}

        fpos_src = os.path.join(launch_dir, "POSCAR")

        # loop through all calculations within each chain and collect statuses
        # statuses = {final_xc : {xc_calc : status}}
        statuses = {}

        # looping through each VASP calc in that chain
        for xc_calc in calc_list:

            restart_this_one = True if xc_calc in fresh_restart else False
            # initialize configs that are particular to this particular VASP calc in this chain
            calc_configs = {}

            # (0) update vasp configs with the current xc and calc
            xc_to_run, calc_to_run = xc_calc.split("-")
            calc_configs["xc_to_run"] = xc_to_run
            calc_configs["calc_to_run"] = calc_to_run

            # (1) make calc_dir (or remove and remake if fresh_restart)
            calc_dir = os.path.join(launch_dir, xc_calc)

            if os.path.exists(calc_dir) and restart_this_one:
                rmtree(calc_dir)
            if not os.path.exists(calc_dir):
                os.mkdir(calc_dir)

            if restart_this_one:
                statuses[xc_calc] = "new"
                continue

            # (2) check convergence of current calc
            E_per_at = AnalyzeVASP(calc_dir).E_per_at
            convergence = True if E_per_at else False
            if convergence:
                if calc_to_run == "relax":
                    statuses[xc_calc] = "done"
                    continue
                elif calc_to_run != "loose":
                    xc_calc_relax = "%s-relax" % xc_to_run
                    if xc_calc_relax in statuses:
                        if statuses[xc_calc_relax] == "done":
                            statuses[xc_calc] = "done"
                            continue
                        else:
                            statuses[xc_calc] = "new"
                            continue
                    else:
                        print(
                            "WARNING: %s not in statuses; did you mean to only run static?"
                            % xc_calc_relax
                        )
                        statuses[xc_calc] = "done"
                        continue
            else:
                # (4) check for POSCAR
                # flag to check whether POSCAR is newly copied (don't want to perturb already-perturbed structures)
                fpos_dst = os.path.join(calc_dir, "POSCAR")
                if os.path.exists(fpos_dst):
                    # if there is a POSCAR, make sure its not empty
                    with open(fpos_dst, "r") as f_tmp:
                        contents = f_tmp.readlines()
                    # if its empty, copy the initial structure to calc_dir
                    if len(contents) == 0:
                        copyfile(fpos_src, fpos_dst)

                # if theres no POSCAR, copy the initial structure to calc_dir
                if not os.path.exists(fpos_dst):
                    copyfile(fpos_src, fpos_dst)

                # (5) check for CONTCAR. if one exists, if its not empty, and if not fresh_restart, mark this job as one to "continue"
                # (ie later, we'll copy CONTCAR to POSCAR); otherwise, mark as NEWRUN
                fcont_dst = os.path.join(calc_dir, "CONTCAR")
                if os.path.exists(fcont_dst):
                    with open(fcont_dst, "r") as f_tmp:
                        contents = f_tmp.readlines()
                    if len(contents) > 0:
                        statuses[xc_calc] = "continue"
                    else:
                        statuses[xc_calc] = "new"
                else:
                    statuses[xc_calc] = "new"
        return statuses

    @property
    def prepare_directories(self):
        statuses = self.statuses
        vasp_configs = self.vasp_configs
        launch_dir = self.launch_dir
        calc_list = self.calc_list
        for xc_calc in calc_list:
            status = statuses[xc_calc]
            if status in ["done", "queued"]:
                continue
            if xc_calc == calc_list[0]:
                first_calc = True
            else:
                first_calc = False
            xc_to_run, calc_to_run = xc_calc.split("-")

            user_vasp_configs_before_error_handling = vasp_configs.copy()
            user_vasp_configs_before_error_handling["xc_to_run"] = xc_to_run
            user_vasp_configs_before_error_handling["calc_to_run"] = calc_to_run

            if first_calc and (status == "new"):
                user_vasp_configs_before_error_handling["perturb_struc"] = (
                    self.sub_configs["perturb_first_struc"]
                )

            calc_dir = os.path.join(launch_dir, xc_calc)

            # (6) initialize VASPSetUp with current VASP configs for this calculation
            vsu = VASPSetUp(
                calc_dir=calc_dir,
                user_configs=user_vasp_configs_before_error_handling,
            )

            user_vasp_configs = user_vasp_configs_before_error_handling.copy()

            # (7) check for errors in continuing jobs
            incar_changes = {}
            if status in ["continue", "new"]:
                is_calc_clean = vsu.is_clean
                if not is_calc_clean:
                    # change INCAR based on errors and include in calc_configs
                    incar_changes = vsu.incar_changes_from_errors

                # if there are INCAR updates, add them to calc_configs
                if incar_changes:
                    if xc_calc in user_vasp_configs["incar_mods"]:
                        user_vasp_configs["incar_mods"][xc_calc].update(incar_changes)
                    else:
                        user_vasp_configs["incar_mods"][xc_calc] = incar_changes

                # print("\n\n\n\n\n\n")
                # print(calc_dir)
                # print("THESE ARE MY USER_VASP_CONFIGS")
                # print(user_vasp_configs)
                # print("\n\n\n\n\n\n")
                # print(calc_dir)

                # update our vasp_configs with any modifications to the INCAR that we made to fix errors
                # user_vasp_configs = {**vasp_configs, **calc_configs}
                print("--------- may be some warnings (POTCAR ones OK) ----------")

                # (8) prepare calc_dir to launch
                vsu = VASPSetUp(calc_dir=calc_dir, user_configs=user_vasp_configs)

                vsu.prepare_calc

                print("-------------- warnings should be done ---------------")
                print("\n~~~~~ prepared %s ~~~~~\n" % calc_dir)
        return statuses

    @property
    def write_sub(self):
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

        # make copies of our starting configs
        vasp_configs = self.vasp_configs.copy()
        sub_configs = self.sub_configs.copy()

        launch_dir = self.launch_dir

        vasp_command = self.vasp_command
        slurm_options = self.slurm_options.copy()
        queue_manager = self.queue_manager

        calc_list = self.calc_list
        statuses = self.prepare_directories
        fsub = os.path.join(launch_dir, "sub.sh")
        fstatus = os.path.join(launch_dir, "status.o")
        slurm_options["job-name"] = self.job_name
        with open(fsub, "w", encoding="utf-8") as f:
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
            if sub_configs["mpi_command"] == "mpirun":
                if sub_configs["machine"] == "msi":
                    if sub_configs["vasp_version"] == 5:
                        f.write("module load impi/2018/release_multithread\n")
                    elif sub_configs["vasp_version"] == 6:
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
                elif sub_configs["machine"] == "bridges2":
                    f.write("module load intelmpi\nexport OMP_NUM_THREADS=1\n")

            for xc_calc in calc_list:
                status = statuses[xc_calc]
                f.write('\necho "%s is %s" >> %s\n' % (xc_calc, status, fstatus))

                if status in ["done", "queued"]:
                    continue
                # find our calc_dir (where VASP is executed for this xc_calc)
                xc_to_run, calc_to_run = xc_calc.split("-")
                calc_dir = os.path.join(launch_dir, xc_calc)

                incar_mods = vasp_configs["incar_mods"]
                if xc_calc in incar_mods:
                    incar_mods = incar_mods[xc_calc]
                else:
                    incar_mods = None

                passer_dict = {
                    "xc_calc": xc_calc,
                    "calc_list": calc_list,
                    "calc_dir": calc_dir,
                    "incar_mods": incar_mods,
                    "launch_dir": launch_dir,
                }

                passer_dict_as_str = json.dumps(passer_dict)

                f.write("cd %s\n" % self.scripts_dir)
                f.write("python passer.py '%s' \n" % passer_dict_as_str)

                # before passing data, make sure parent has finished without crashing (using bash..)
                f.write(
                    "isInFile=$(cat %s | grep -c %s)\n"
                    % (os.path.join(launch_dir, "job_killer.o"), "kill")
                )
                f.write("if [ $isInFile -ge 1 ]; then\n")
                f.write("   scancel $SLURM_JOB_ID\n")
                f.write("fi\n")

                f.write("cd %s\n" % calc_dir)
                f.write(self.vasp_command)

                if calc_to_run in ["lobster", "bs"]:
                    f.write(self.lobster_command)

                if calc_to_run == "static":
                    f.write(self.bader_command)
            f.write("\n\nscancel $SLURM_JOB_ID\n")
        return True

    @property
    def launch_sub(self):
        """
        launch the submission script written in write_sub
            if job is not in queue already
            if there's something to launch
                (ie if all calcs are done, dont launch)
        """
        if self.is_job_in_queue:
            return

        print("     now launching sub")
        scripts_dir = self.scripts_dir
        launch_dir = self.launch_dir

        # determine what keywords to look for to see if job needs to be launched
        flags_that_need_to_be_executed = self.sub_configs["execute_flags"]

        fsub = os.path.join(launch_dir, "sub.sh")

        needs_to_launch = False
        # see if there's anything to launch
        with open(fsub, "r") as f:
            contents = f.read()
            for flag in flags_that_need_to_be_executed:
                if flag in contents:
                    needs_to_launch = True

        if not needs_to_launch:
            print(" !!! nothing to launch here, not launching\n\n")
            return

        # if we made it this far, launch it
        os.chdir(launch_dir)
        subprocess.call(["sbatch", "sub.sh"])
        os.chdir(scripts_dir)


def setup_bandstructure(
    converged_static_dir, rerun=False, symprec=0.1, kpoints_line_density=20
):
    """

    function to create input files (INCAR, KPOINTS, POTCAR, POSCAR, lobsterin) for band structure calculations
        after static calculations

    ideally, this would be implemented as the end of a "packing" but slightly complicated because we need teh static calculation to be converged (or do we?)

    Args:
        converged_static_dir (str)
            path to converged static calculation

        rerun (bool)
            if True, rerun bandstructure calculation even if it's already converged

        symprec (float)
            symmetry precision for finding primitive cell and generating KPOINTS

        kpoints_line_density (int)
            how many kpoints between each high symmetry point

    Returns:
        directory to band structure calculation (str) or None if not ready to run this

    """
    # make sure static is converged
    av = AnalyzeVASP(converged_static_dir)
    if not av.is_converged:
        print("static calculation not converged; not setting up bandstructure yet")
        return None

    # get the paths to relevant input files from the static calculation
    files_from_static = ["POSCAR", "KPOINTS", "POTCAR", "INCAR"]
    fposcar_src, fkpoints_src, fpotcar_src, fincar_src = [
        os.path.join(converged_static_dir, f) for f in files_from_static
    ]

    # make a directory for the bandstructure calculation
    bs_dir = converged_static_dir.replace("-static", "-bs")
    if not os.path.exists(bs_dir):
        os.mkdir(bs_dir)

    # get the paths to relevant input files for the bandstructure calculation
    fposcar_dst, fkpoints_dst, fpotcar_dst, fincar_dst = [
        os.path.join(bs_dir, f) for f in files_from_static
    ]

    # make sure bandstructure calc didn't already run
    av = AnalyzeVASP(bs_dir)
    if av.is_converged and not rerun:
        print("bandstructure already converged")
        return None

    # initialize a Lobsterin object
    lobsterin = Lobsterin

    # create a primitive cell and write to bs POSCAR (needed so we don't have a bunch of overlapping bands)
    lobsterin.write_POSCAR_with_standard_primitive(
        POSCAR_input=fposcar_src, POSCAR_output=fposcar_dst, symprec=symprec
    )

    # create a line-mode KPOINTS file and write to bs KPOINTS
    try:
        lobsterin.write_KPOINTS(
            POSCAR_input=fposcar_dst,
            KPOINTS_output=fkpoints_dst,
            line_mode=True,
            symprec=symprec,
            kpoints_line_density=kpoints_line_density,
        )
    except ValueError:
        print("trying higher symprec")
        lobsterin.write_KPOINTS(
            POSCAR_input=fposcar_dst,
            KPOINTS_output=fkpoints_dst,
            line_mode=True,
            symprec=symprec * 2,
            kpoints_line_density=kpoints_line_density,
        )

    # copy our POTCAR from the static calc
    copyfile(fpotcar_src, fpotcar_dst)

    # write a lobsterin file, including fatband analysis
    lobsterin = Lobsterin.standard_calculations_from_vasp_files(
        POSCAR_input=fposcar_dst,
        INCAR_input=fincar_src,
        POTCAR_input=fpotcar_dst,
        option="standard_with_fatband",
    )
    flobsterin = os.path.join(bs_dir, "lobsterin")
    lobsterin.write_lobsterin(flobsterin)

    # write our bs INCAR
    lobsterin.write_INCAR(
        incar_input=fincar_src,
        incar_output=fincar_dst,
        poscar_input=fposcar_dst,
        further_settings={"ISMEAR": 0},
    )

    return bs_dir


def setup_parchg(converged_static_dir, rerun=False, eint=-1):
    """

    function to create input files (INCAR, KPOINTS, POTCAR, POSCAR, WAVECAR, CHGCAR) for partial charge calculation
        after static calculations

    Args:
        converged_static_dir (str)
            path to converged static calculation

        rerun (bool)
            if True, rerun bandstructure calculation even if it's already converged

        eint (flaot)
            the lower energy bound (relative to E_Fermi) to analyze partial charge density

    Returns:
        directory to band structure calculation (str) or None if not ready to run this

    """
    # make sure static is converged
    av = AnalyzeVASP(converged_static_dir)
    if not av.is_converged:
        print("static calculation not converged; not setting up parchg calc yet")
        return None

    # get the paths to relevant input files from the static calculation
    files_from_static = ["POSCAR", "KPOINTS", "POTCAR", "INCAR", "WAVECAR", "CHGCAR"]

    # make a directory for the bandstructure calculation
    parchg_dir = converged_static_dir.replace("-static", "-parchg")
    if not os.path.exists(parchg_dir):
        os.mkdir(parchg_dir)

    # make sure bandstructure calc didn't already run
    av = AnalyzeVASP(parchg_dir)
    if av.is_converged and not rerun:
        print("parchg already converged")
        return None

    new_incar_params = {
        "EINT": str(eint),
        "ISTART": "1",
        "LPARD": "True",
        "LSEPB": "False",
        "LSEPK": "False",
        "LWAVE": "False",
        "NBMOD": "-3",
    }

    for file_to_copy in files_from_static:
        f_src = os.path.join(converged_static_dir, file_to_copy)
        f_dst = os.path.join(parchg_dir, file_to_copy)
        copyfile(f_src, f_dst)

    with open(os.path.join(converged_static_dir, "INCAR")) as f_src:
        with open(os.path.join(parchg_dir, "INCAR"), "w") as f_dst:
            for key in new_incar_params:
                f_dst.write("%s = %s\n" % (key, new_incar_params[key]))
            for line in f_src:
                if line.split("=")[0].strip() in new_incar_params:
                    continue
                f_dst.write(line)

    return parchg_dir


def setup_dfpt(converged_static_dir, supercell_grid=[2, 2, 2], rerun=False):
    """
    function to create input files (INCAR, KPOINTS, POTCAR, POSCAR, WAVECAR, CHGCAR) for partial charge calculation
        after static calculations

    Args:
        converged_static_dir (str)
            path to converged static calculation

        supercell_grid (list)
            the supercell grid to create from original POSCAR

        rerun (bool)
            if True, rerun bandstructure calculation even if it's already converged

    Returns:
        directory to band structure calculation (str) or None if not ready to run this

    """
    # make sure static is converged
    av = AnalyzeVASP(converged_static_dir)
    if not av.is_converged:
        print("static calculation not converged; not setting up dfpt calc yet")
        return None

    # get the paths to relevant input files from the static calculation
    files_from_static = ["POSCAR", "KPOINTS", "POTCAR"]

    # make a directory for the bandstructure calculation
    dfpt_dir = converged_static_dir.replace("-static", "-dfpt")
    if not os.path.exists(dfpt_dir):
        os.mkdir(dfpt_dir)

    # make sure dfpt calc didn't already run
    av = AnalyzeVASP(dfpt_dir)
    if av.is_converged and not rerun:
        print("dfpt already converged")
        return None

    for file_to_copy in files_from_static:
        f_src = os.path.join(converged_static_dir, file_to_copy)
        f_dst = os.path.join(dfpt_dir, file_to_copy)
        copyfile(f_src, f_dst)

    copyfile(
        os.path.join(dfpt_dir, "POSCAR"), os.path.join(dfpt_dir, "POSCAR-unitcell")
    )
    st_unit = StrucTools(os.path.join(dfpt_dir, "POSCAR-unitcell"))
    supercell = st_unit.make_supercell(supercell_grid)
    supercell.to(fmt="POSCAR", filename=os.path.join(dfpt_dir, "POSCAR"))

    fstatic_incar = os.path.join(converged_static_dir, "INCAR")
    fdfpt_incar = os.path.join(dfpt_dir, "INCAR")

    new_incar_params = {
        "IBRION": 7,
        "NSW": 1,
        "IALGO": 38,
        "EDIFF": 1e-6,
        "ADDGRID": True,
        "ALGO": "Normal",
        "ISYM": 2,
    }

    incar_params_to_exclude = ["NPAR", "NCORE"]

    with open(fstatic_incar) as f_src:
        with open(fdfpt_incar, "w") as f_dst:
            for line in f_src:
                if line.split("=")[0].strip() in new_incar_params:
                    continue
                if line.split("=")[0].strip() in incar_params_to_exclude:
                    continue
                if "MAGMOM" in line:
                    magmom = line.split("=")[1].strip()
                    magmoms = magmom.split(" ")
                    new_magmoms = []
                    for mag in magmoms:
                        if mag:
                            old_number = int(mag.split("*")[0])
                            new_number = (
                                old_number
                                * supercell_grid[0]
                                * supercell_grid[1]
                                * supercell_grid[2]
                            )
                            old_mag = mag.split("*")[1]
                            new_mag = old_mag
                            new_magmoms.append("%s*%s" % (new_number, new_mag))
                    f_dst.write("MAGMOM = %s\n" % " ".join(new_magmoms))
                else:
                    f_dst.write(line)
            for key in new_incar_params:
                f_dst.write("%s = %s\n" % (key, new_incar_params[key]))

    fdfpt_kpoints = os.path.join(dfpt_dir, "KPOINTS")
    with open(fdfpt_kpoints, "w") as f_dst:
        f_dst.write("Regular\n")
        f_dst.write("0 0 0\n")
        f_dst.write("Gamma\n")
        f_dst.write("1 1 1\n")

    return dfpt_dir


def generate_finite_displacements(
    converged_static_dir, supercell_grid=[2, 2, 2], remake=False
):
    """
    Args:
        converged_static_dir (str)
            path to converged static calculation

        supercell_grid (list)
            the supercell grid to create from original POSCAR

    Returns:
        phonon_dir
            disp-*/
            POSCAR-*
    """
    # make sure static is converged
    av = AnalyzeVASP(converged_static_dir)
    if not av.is_converged:
        print("static calculation not converged; not setting up phonon calcs yet")
        return None

    # create a directory called phonons within the static directory
    phonon_dir = os.path.join(converged_static_dir, "phonons")
    if not os.path.exists(phonon_dir):
        os.mkdir(phonon_dir)

    # use my current directory as a reference point
    curr_dir = os.getcwd()

    # see if POSCARs are already created
    created_poscars = os.listdir(phonon_dir)
    created_poscars = [f for f in created_poscars if "POSCAR-" in f]

    # go into the phonon directory and generate the displacement POSCARs
    if remake or not created_poscars:
        copyfile(
            os.path.join(converged_static_dir, "CONTCAR"),
            os.path.join(phonon_dir, "POSCAR"),
        )
        os.chdir(phonon_dir)
        subprocess.call(
            [
                "phonopy",
                "-d",
                "--dim=%s %s %s"
                % (
                    str(supercell_grid[0]),
                    str(supercell_grid[1]),
                    str(supercell_grid[2]),
                ),
            ]
        )
        os.chdir(curr_dir)

    # get the paths to relevant input files from the static calculation
    files_from_static = ["KPOINTS", "POTCAR", "INCAR"]

    # some special INCAR settings for the static calculations of each displacement
    new_incar_params = {
        "IBRION": 2,
        "ISIF": 3,
        "ENCUT": 700,
        "EDIFF": 1e-7,
        "LAECHG": False,
        "LREAL": False,
        "ALGO": "Normal",
        "NSW": 0,
        "LCHARG": False,
    }

    # copy KPOINTS, INCAR, POTCAR from the static calculation into phonon_dir
    for file_to_copy in files_from_static:
        f_src = os.path.join(converged_static_dir, file_to_copy)
        f_dst = os.path.join(phonon_dir, file_to_copy)
        copyfile(f_src, f_dst)

    # modify the INCAR in the disp-* dir
    with open(os.path.join(converged_static_dir, "INCAR")) as f_src:
        with open(os.path.join(phonon_dir, "INCAR"), "w") as f_dst:
            for key in new_incar_params:
                f_dst.write("%s = %s\n" % (key, new_incar_params[key]))
            for line in f_src:
                if line.split("=")[0].strip() in new_incar_params:
                    continue
                if "MAGMOM" in line:
                    magmom = line.split("=")[1].strip()
                    magmoms = magmom.split(" ")
                    new_magmoms = []
                    for mag in magmoms:
                        if mag:
                            old_number = int(mag.split("*")[0])
                            new_number = (
                                old_number
                                * supercell_grid[0]
                                * supercell_grid[1]
                                * supercell_grid[2]
                            )
                            old_mag = mag.split("*")[1]
                            new_mag = old_mag
                            new_magmoms.append("%s*%s" % (new_number, new_mag))
                    f_dst.write("MAGMOM = %s\n" % " ".join(new_magmoms))
                    continue
                f_dst.write(line)

    return phonon_dir


def setup_finite_displacement_calcs(phonon_dir, remake=False, rerun=False):

    # grab the created POSCARs
    created_poscars = os.listdir(phonon_dir)
    created_poscars = [f for f in created_poscars if "POSCAR-" in f]

    # get the paths to relevant input files from the static calculation
    files_from_static = ["KPOINTS", "POTCAR", "INCAR"]

    # for each displacement,
    # create a disp-00* directory,
    # grab the POSCAR-00*,
    # copy the static calculation files,
    # modify the INCAR

    statuses = {}

    for poscar in created_poscars:
        number = poscar.split("-")[-1]

        # create disp-00* dir
        disp_dir = os.path.join(phonon_dir, "disp-%s" % number)
        if not os.path.exists(disp_dir):
            os.mkdir(disp_dir)

        statuses[number] = {"convergence": False, "calc_dir": disp_dir}

        # copy the displaced POSCAR-00* into the disp-00* dir
        poscar_src = os.path.join(phonon_dir, poscar)
        poscar_dst = os.path.join(disp_dir, "POSCAR")
        if not os.path.exists(poscar_dst) or remake:
            copyfile(poscar_src, poscar_dst)

        # check convergence
        av = AnalyzeVASP(disp_dir)
        if av.is_converged and not rerun:
            statuses[number]["convergence"] = True
            continue

        # copy KPOINTS, INCAR, POTCAR from the phonon dir into disp_dir
        for file_to_copy in files_from_static:
            f_src = os.path.join(phonon_dir, file_to_copy)
            f_dst = os.path.join(disp_dir, file_to_copy)
            if not os.path.exists(f_dst) or remake:
                copyfile(f_src, f_dst)

    return statuses


def setup_static_magtest(converged_static_dir, rerun=False):
    # make sure relax is converged
    av = AnalyzeVASP(converged_static_dir)
    if not av.is_converged:
        print("static calculation not converged; not setting up static mag test")
        return None

    relax_dir = converged_static_dir.replace("-static", "-relax")
    av_relax = AnalyzeVASP(relax_dir)
    if not av_relax.is_converged:
        print("relax calculation not converged; not setting up static mag test")
        return None

    # get the paths to relevant input files from the static calculation
    files_from_relax = ["POSCAR", "KPOINTS", "POTCAR", "INCAR", "WAVECAR", "CHGCAR"]

    # make a directory for the magtest calculation
    magtest_dir = converged_static_dir.replace("-static", "-static_magtest")
    if not os.path.exists(magtest_dir):
        os.mkdir(magtest_dir)

    # make sure bandstructure calc didn't already run
    av = AnalyzeVASP(magtest_dir)
    if av.is_converged and not rerun:
        print("magtest already converged")
        return None

    mag_decorated_relax_structure = get_structure_from_prev_run(
        av_relax.outputs.vasprun, av_relax.outputs.outcar
    )

    magmom = mag_decorated_relax_structure.site_properties["magmom"]
    magmom_string = " ".join([str(m) for m in magmom])

    new_incar_params = {"MAGMOM": magmom_string}

    for file_to_copy in files_from_relax:
        f_src = os.path.join(converged_static_dir, file_to_copy)
        f_dst = os.path.join(magtest_dir, file_to_copy)
        copyfile(f_src, f_dst)

    with open(os.path.join(converged_static_dir, "INCAR"), "r") as f_src:
        with open(os.path.join(magtest_dir, "INCAR"), "w") as f_dst:
            for key in new_incar_params:
                f_dst.write("%s = %s\n" % (key, new_incar_params[key]))
            for line in f_src:
                if line.split("=")[0].strip() in new_incar_params:
                    continue
                f_dst.write(line)

    return magtest_dir


def main():
    return


if __name__ == "__main__":
    main()

from pydmclab.core.struc import StrucTools
from pydmclab.hpc.vasp import VASPSetUp
from pydmclab.hpc.analyze import AnalyzeVASP
from pydmclab.utils.handy import read_yaml, write_yaml
from pydmclab.data.configs import load_base_configs, load_partition_configs

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

        _base_configs = load_base_configs()
        configs = _base_configs.copy()

        # we're going to modify vasp, slurm, and sub configs using one user_configs dict, so let's keep track of what's been applied
        user_configs_used = []

        for option in _base_configs:
            if option in user_configs:
                if option not in user_configs_used:
                    new_value = user_configs[option]
                    configs[option] = new_value
                    user_configs_used.append(option)

        # determine standard and mag from launch_dir
        configs["mag"] = mag

        # include magmom in vasp_configs
        configs["magmom"] = initial_magmom

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
            self.structure = Structure.from_file(fpos)

        # load partition configurations to help with slurm setup
        partitions = load_partition_configs()
        self.partitions = partitions

        # these are the xcs we want energies for --> each one of these should have a submission script
        # i.e., they are the end of individual chains
        self.relaxation_xcs = self.configs["relaxation_xcs"]
        self.static_addons = self.configs["static_addons"]
        self.start_with_loose = self.configs["start_with_loose"]
        self.fresh_restart = self.configs["fresh_restart"]
        self.scripts_dir = os.getcwd()
        self.job_dir = self.scripts_dir.split("/")[-2]

    @property
    def calc_list(self):
        """
        Returns:
            [xc-calc in the order they should be executed]
        """
        configs = self.configs
        if ("custom_calc_list" in configs) and (
            configs["custom_calc_list"] is not None
        ):
            return configs["custom_calc_list"]

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
        return self.configs["manager"]

    @property
    def slurm_options(self):
        """
        Returns dictionary of slurm options
            - nodes, ntasks, walltime, etc

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
    def bin_dir(self):
        """
        Returns bin directory where things (eg LOBSTER) are located
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
    def vasp_dir(self):
        """
        Returns directory containing vasp executable
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

        configs = self.configs.copy()

        fresh_restart = configs["fresh_restart"]
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

            if restart_this_one:
                statuses[xc_calc] = "new"
                continue

            # (2) check convergence of current calc
            E_per_at = AnalyzeVASP(calc_dir).E_per_at
            convergence = True if E_per_at else False
            if convergence:
                vsu = VASPSetUp(calc_dir, user_configs=configs)
                is_calc_clean = vsu.is_clean
                if not is_calc_clean:
                    statuses[xc_calc] = "continue"
                else:
                    if calc_to_run == "relax":
                        statuses[xc_calc] = "done"

                    elif calc_to_run != "loose":
                        xc_calc_relax = "%s-relax" % xc_to_run
                        if xc_calc_relax in statuses:
                            if statuses[xc_calc_relax] == "done":
                                statuses[xc_calc] = "done"

                            else:
                                statuses[xc_calc] = "new"

                        else:
                            print(
                                "WARNING: %s not in statuses; did you mean to only run static?"
                                % xc_calc_relax
                            )
                            statuses[xc_calc] = "done"

            else:
                # (4) check for POSCAR
                # flag to check whether POSCAR is newly copied
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
        configs = self.configs.copy()
        launch_dir = self.launch_dir
        calc_list = self.calc_list
        for xc_calc in calc_list:
            status = statuses[xc_calc]
            if status in ["done", "queued"]:
                continue

            xc_to_run, calc_to_run = xc_calc.split("-")

            user_vasp_configs_before_error_handling = configs.copy()
            user_vasp_configs_before_error_handling["xc_to_run"] = xc_to_run
            user_vasp_configs_before_error_handling["calc_to_run"] = calc_to_run

            calc_dir = os.path.join(launch_dir, xc_calc)

            print("\n~~~~~ working on %s ~~~~~\n" % calc_dir)

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

        if self.is_job_in_queue:
            return

        configs = self.configs.copy()
        slurm_options = self.slurm_options.copy()

        launch_dir = self.launch_dir

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
                    f.write("module load intelmpi\nexport OMP_NUM_THREADS=1\n")

            for xc_calc in calc_list:
                status = statuses[xc_calc]
                f.write('\necho "%s is %s" >> %s\n' % (xc_calc, status, fstatus))

                if status in ["done", "queued"]:
                    continue
                # find our calc_dir (where VASP is executed for this xc_calc)
                xc_to_run, calc_to_run = xc_calc.split("-")
                calc_dir = os.path.join(launch_dir, xc_calc)

                incar_mods = configs["incar_mods"]
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
        configs = self.configs.copy()
        if self.is_job_in_queue:
            return

        print("     now launching sub")
        scripts_dir = self.scripts_dir
        launch_dir = self.launch_dir

        # determine what keywords to look for to see if job needs to be launched
        flags_that_need_to_be_executed = configs["execute_flags"]

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


def main():
    return


if __name__ == "__main__":
    main()

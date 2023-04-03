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

import os
from shutil import copyfile, rmtree
import subprocess
import warnings

HERE = os.path.dirname(os.path.abspath(__file__))


class SubmitTools(object):
    """
    This class is focused on figuring out how to prepare chains of calculations
        - the idea being that the output from this class is some file that you can
            "submit" to a queueing system
        - this class will automatically crawl through the VASP output files and figure out
            how to edit that submission file to finish the desired calculations

    """

    def __init__(
        self,
        launch_dir,
        final_xcs,
        magmom,
        user_configs={},
        refresh_configs=["vasp", "sub", "slurm"],
        vasp_configs_yaml=os.path.join(os.getcwd(), "_vasp_configs.yaml"),
        slurm_configs_yaml=os.path.join(os.getcwd(), "_slurm_configs.yaml"),
        sub_configs_yaml=os.path.join(os.getcwd(), "_sub_configs.yaml"),
    ):

        """
        Args:
            launch_dir (str) - directory to launch calculations from (to submit the submission file)
                - assumes initial structure is POSCAR in launch_dir
                    - LaunchTools will put it there
                - within this directory, various VASP calculation directories (calc_dirs) will be created
                        - gga-loose, gga-relax, gga-static, etc
                            - VASP will be run in each of these, but we need to run some sequentially, so we pack them together in one submission script

            final_xcs (list) - list of exchange correlation methods we want the final energy for
                - you should pass this to SubmitTools
                    - e.g., ['metagga'] if you only want metagga or ['ggau', 'metagga'] if you want both of these methods

            magmom (list) - list of magnetic moments for each atom in structure (or None if not AFM)
                - you should pass this here
                    - None if not AFM
                    - otherwise pull it from a dictionary of magmoms you made with MagTools

            user_configs (dict) - any non-default parameters you want to pass
                - these will override the defaults in the yaml files
                - look at pydmc/data/data/*configs*.yaml for the defaults
                    - note: _launch_configs.yaml will be modified using LaunchTools
                - you should be passing stuff here!
                - VASPSetUp will expect xc_to_run, calc_to_run, standard, and mag
                    - xc_to_run and calc_to_run will be passed based on final_xcs and sub_configs['packing']
                    - standard and mag will get passed to it based on launch_dir
                - you can also pass any settings in _vasp_configs.yaml, _slurm_configs.yaml, or _sub_configs.yaml here
                - _vasp_configs options:
                    see pydmc/hpc/vasp.py or pydmc/data/data/_vasp_configs.yaml for options
                - _sub_configs options:
                    vasp: vasp_std # use vasp_gam for "loose" calcs (havent implemented yet)
                    vasp_dir: /home/cbartel/shared/bin/vasp/ # where vasp executable lives
                    mpi_command: srun # how to launch on multicore/multinode (may be mpirun depending on compilation)
                    manager: '#SBATCH' # how to manage interactions with the queue (some machines dont use slurm)

                    fqueue: q.o # this will be created in the folder where you execute python

                    fresh_restart: False # True if you want to re-run all calculations; False if you want to pick up where you left off
                    force_postprocess: False # if True, run LOBSTER and Bader again if vasp_configs.lobster_static = True even if output files exist

                    files_to_inherit: ['WAVECAR', 'CONTCAR'] # usually no need to change

                    execute_flags: ['srun', 'python', 'lobster', 'bader'] # what to look for in a submission script to see if it should be launched

                    # this defines what jobs should be chained together and in what order they should be executed
                    # if you only wanted to run a static calculation for metagga, you might pass packing['metagga'] = ['metagga-static']
                    packing:
                    gga:
                    - gga-loose
                    - gga-relax
                    - gga-static
                    ggau:
                    - ggau-loose
                    - ggau-relax
                    - ggau-static
                    metagga:
                    - gga-loose
                    - gga-relax
                    - gga-static
                    - metagga-relax
                    - metagga-static
                - _slurm_configs options:
                    nodes: 1 # how many nodes per launched sequence of calcs
                    ntasks: 16 # how many total cores
                    time: 1440 # how long in minutes before hitting walltime
                    error: log.e # where to write slurm errors to in launch_dir
                    output: log.o # where to write slurm output to in launch_dir
                    account: cbartel # account to charge
                    partition: msismall # partition to use
                    job-name: # unique job name; if none provided, will default to formula.id.standard.mag.xc
                    mem: # if you need more memory specify here (e.g., 128GB)
                    constraint: # may not need this ever on MSI
                    qos: # may not need this ever on MSI

            vasp_configs_yaml (os.pathLike) - path to yaml file containing baseline vasp configs
                - there's usually no reason to change this
                - this holds some default configs for VASP
                - can always be changed with user_configs

             slurm_configs_yaml (os.pathLike) - path to yaml file containing baseline slurm configs
                - there's usually no reason to change this
                - this holds some default configs for slurm
                - can always be changed with user_configs

            sub_configs_yaml (os.pathLike) - path to yaml file containing baseline submission file configs
                - there's usually no reason to change this
                - this holds some default configs for submission files
                - can always be changed with user_configs

            refresh_configs (bool) - if True, will copy pydmc baseline configs to your local directory
                - this is useful if you've made changes to the configs files in the directory you're working in and want to start over

        Returns:
            self.launch_dir (os.pathLike) - directory to launch calculations from
            self.valid_calcs (list) - list of calculations to run
            self.slurm_configs (dotdict) - dictionary of slurm configs (in format similar to yaml)
            self.vasp_configs (dotdict) - dictionary of vasp configs (in format similar to yaml)
            self.sub_configs (dotdict) - dictionary of submission configs (in format similar to yaml)
            self.files_to_inherit (list) - list of files to copy from calc to calc
            self.structure (Structure) - pymatgen structure object from launch_dir/POSCAR
            self.magmom (list) - list of magnetic moments for each atom in structure
            self.partitions (dotdict) - dictionary of info regarding partition configurations on MSI
        """

        self.launch_dir = launch_dir

        # just a reminder of how a launch director looks
        # NOTE: should be made with LaunchTools
        top_level, unique_ID, standard, mag = launch_dir.split("/")[-4:]

        # don't run gga twice
        if ("gga" in final_xcs) and ("metagga" in final_xcs):
            warnings.warn(
                "You have both gga and metagga in final_xcs. This is probably not what you want. I'm going to remove gga from final_xcs since you get it for free w/ metagga"
            )
            final_xcs.remove("gga")

        # copy baseline vasp configs to launch_dir if they don't exist
        if not os.path.exists(vasp_configs_yaml) or ("vasp" in refresh_configs):
            _vasp_configs = load_vasp_configs()
            write_yaml(_vasp_configs, vasp_configs_yaml)

        # copy baseline slurm configs to launch_dir if they don't exist
        if not os.path.exists(slurm_configs_yaml) or ("slurm" in refresh_configs):
            _slurm_configs = load_slurm_configs()
            write_yaml(_slurm_configs, slurm_configs_yaml)

        # copy baseline sub configs to launch_dir if they don't exist
        if not os.path.exists(sub_configs_yaml) or ("sub" in refresh_configs):
            _sub_configs = load_sub_configs()
            write_yaml(_sub_configs, sub_configs_yaml)

        # we're going to modify vasp, slurm, and sub configs using one user_configs dict, so let's keep track of what's been applied
        user_configs_used = []

        # update slurm_configs based on user_configs
        slurm_configs = read_yaml(slurm_configs_yaml)
        for option in slurm_configs:
            if option in user_configs:
                if option not in user_configs_used:
                    new_value = user_configs[option]
                    slurm_configs[option] = new_value
                    user_configs_used.append(option)

        # create copy of slurm_configs to prevent unwanted updates
        self.slurm_configs = slurm_configs.copy()

        # update sub_configs based on user_configs
        sub_configs = read_yaml(sub_configs_yaml)
        for option in sub_configs:
            if option in user_configs:
                if option not in user_configs_used:
                    new_value = user_configs[option]
                    sub_configs[option] = new_value
                    user_configs_used.append(option)

        # create copy of sub_configs to prevent unwanted updates
        self.sub_configs = sub_configs.copy()

        # update vasp_configs based on user_configs
        vasp_configs = read_yaml(vasp_configs_yaml)
        for option in vasp_configs:
            if option in user_configs:
                if option not in user_configs_used:
                    new_value = user_configs[option]
                    vasp_configs[option] = new_value
                    user_configs_used.append(option)

        # determine standard and mag from launch_dir
        vasp_configs["standard"] = standard
        vasp_configs["mag"] = mag

        # include magmom in vasp_configs
        vasp_configs["magmom"] = magmom

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
        self.final_xcs = final_xcs

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
        partition_specs = partitions[options["partition"]]

        # make sure slurm_options align with partition configurations
        if partition_specs["proc"] == "gpu":
            options["nodes"] = 1
            options["ntasks"] = 1
            options["gres"] = "gpu:%s:%s" % (
                options["partition"].split("-")[0],
                str(options["nodes"]),
            )
        if not partition_specs["sharing"]:
            options["ntasks"] = partition_specs["cores_per_node"]
        return options

    @property
    def vasp_command(self):
        """
        Returns command used to execute vasp
            e.g., 'srun -n 24 PATH_TO_VASP/vasp_std > vasp.o'
        """
        sub_configs = self.sub_configs.copy()
        vasp_configs = self.vasp_configs.copy()
        vasp_exec = os.path.join(sub_configs["vasp_dir"], sub_configs["vasp"])
        if sub_configs['mpi_command'] == 'srun':
            return "\n%s --ntasks=%s --mpi=pmi2 %s > %s\n" % (
                sub_configs["mpi_command"],
                str(self.slurm_options["ntasks"]),
                vasp_exec,
                vasp_configs["fvaspout"]
            )
        elif sub_configs['mpi_command'] == 'mpirun':
            return '\n%s -np=%s %s > %s\n' % (
                sub_configs['mpi_command'],
                str(self.slurm_options['ntasks']),
                vasp_exec,
                vasp_configs['fvaspout'])

    @property
    def lobster_command(self):
        """
        Returns command used to execute lobster
        """
        lobster = "/home/cbartel/shared/bin/lobster/lobster-4.1.0/lobster-4.1.0"
        return "\n%s\n" % lobster

    @property
    def bader_command(self):
        """
        Returns command used to execute bader
        """
        chgsum = "/home/cbartel/shared/bin/bader/chgsum.pl AECCAR0 AECCAR2"
        bader = "/home/cbartel/shared/bin/bader/bader CHGCAR -ref CHGCAR_sum"
        return "\n%s\n%s\n" % (chgsum, bader)

    def is_job_in_queue(self, job_name):
        """
        Returns:
            True if this job-name is already in the queue, else False

        Will prevent you from messing with directories that have running/pending jobs
        """
        scripts_dir = os.getcwd()
        fqueue = os.path.join(scripts_dir, "_".join(["q", job_name]) + ".o")
        with open(fqueue, "w") as f:
            subprocess.call(
                ["squeue", "-u", "%s" % os.getlogin(), "--name=%s" % job_name], stdout=f
            )
        names_in_q = []
        with open(fqueue) as f:
            for line in f:
                if "PARTITION" not in line:
                    names_in_q.append([v for v in line.split(" ") if len(v) > 0][2])
        os.remove(fqueue)
        if len(names_in_q) > 0:
            print(" !!! job already in queue, not messing with it")
            return True

        print(" not in queue, onward --> ")
        return False

    @property
    def prepare_directories(self):
        """
        This gets called by SubmitTools.write_sub, so you should rarely call this on its own

        A lot going on here. The objective is to prepare a set of directories for all calculations of interest to a given submission script in a given launch_dir
            - note: 1 submission script --> 1 chain of VASP calculations
            - note: 1 launch_dir --> can have > 1 submission script

        1) For each xc-calc pair, create a directory (calc_dir)
            - */launch_dir/xc-calc
            - xc-calc could be gga-loose, metagga-relax, etc.

        2) Check if that calc_dir has a converged VASP job
            - note: also checks "parents" (ie a static is labeled unconverged if its relax is unconverged)
                - parents determined by sub_configs['packing']
            - if calc and parents are converged:
                - checks sub_configs['fresh_restart']
                    - if fresh_restart = False --> label calc_dir as status='done' and move on
                    - if fresh_restart = True --> start this calc over

        3) Put */launch_dir/POSCAR into */launch_dir/xc-calc/POSCAR if there's not a POSCAR there already

        4) Check if */calc_dir/CONTCAR exists and has data in it,
            - if it does, copy */calc_dir/CONTCAR to */calc_dir/POSCAR and label status='continue' (ie continuing job)
            - if it doesn't, label status='new' (ie new job)

        5) Initialize VASPSetUp for calc_dir
            - modifies vasp_input_set with self.configs as requested in configs dictionaries (mainly vasp_configs which receives user_configs as well)

        6) If status in ['continue', 'new'],
            - check for errors using VASPSetUp
                - may remove WAVECAR/CHGCAR
                - will likely make edits to INCAR
        """

        # make copies of relevant configs dicts
        vasp_configs = self.vasp_configs.copy()
        sub_configs = self.sub_configs.copy()

        fresh_restart = sub_configs["fresh_restart"]
        launch_dir = self.launch_dir

        # determine the terminal ends of each chain of VASP calcs to be submitted
        final_xcs = self.final_xcs

        # determine how VASP jobs get chained together
        packing = sub_configs["packing"]

        print("\n\n~~~~~ starting to work on %s ~~~~~\n\n" % launch_dir)

        fpos_src = os.path.join(launch_dir, "POSCAR")

        # loop through all calculations within each chain and collect statuses
        # statuses = {final_xc : {xc_calc : status}}
        statuses = {}
        # looping through each chain
        for final_xc in final_xcs:
            job_name = ".".join(launch_dir.split("/")[-4:] + [final_xc])
            print("\nchecking if %s is in q" % job_name)
            if self.is_job_in_queue(job_name):
                continue
            statuses[final_xc] = {}

            # looping through each VASP calc in that chain
            counter = 0
            for xc_calc in packing[final_xc]:
                counter += 1

                # initialize configs that are particular to this particular VASP calc in this chain
                calc_configs = {}

                # (0) update vasp configs with the current xc and calc
                xc_to_run, calc_to_run = xc_calc.split("-")
                calc_configs["xc_to_run"] = xc_to_run
                calc_configs["calc_to_run"] = calc_to_run

                # (1) make calc_dir (or remove and remake if fresh_restart)
                calc_dir = os.path.join(launch_dir, xc_calc)
                if os.path.exists(calc_dir) and fresh_restart:
                    rmtree(calc_dir)
                if not os.path.exists(calc_dir):
                    os.mkdir(calc_dir)

                # (2) check convergence of current calc
                E_per_at = AnalyzeVASP(calc_dir).E_per_at
                convergence = True if E_per_at else False

                # (3) if converged, make sure parents have converged
                large_E_diff_between_relax_and_static = False
                if convergence:
                    # static calcs are currently the only ones that have parents that must be converged
                    if calc_to_run == "static":
                        static_energy = E_per_at
                        # that parent is a relax
                        parent_calc = "relax"
                        parent_xc_calc = "%s-%s" % (xc_to_run, parent_calc)
                        # make sure we're calculating relax and static (and not just)
                        if parent_xc_calc in packing[final_xc]:
                            parent_calc_dir = os.path.join(launch_dir, parent_xc_calc)
                            parent_energy = AnalyzeVASP(parent_calc_dir).E_per_at
                            parent_convergence = True if parent_energy else False
                            if not parent_energy:
                                print(
                                    "     %s (parent) not converged, need to continue this calc"
                                    % parent_xc_calc
                                )
                            else:
                                # if there is a large difference b/t the relax and static energy, something fishy happened, so let's start the static calc over
                                if abs(parent_energy - static_energy) > 0.2:
                                    print(
                                        "     %s (parent) and %s (child) energies differ by more than 0.2 eV/atom"
                                        % (parent_xc_calc, xc_calc)
                                    )
                                    large_E_diff_between_relax_and_static = True
                        else:
                            parent_convergence = True
                    else:
                        parent_convergence = True

                # if parents + current calc are converged, give it status = done
                if (
                    convergence
                    and parent_convergence
                    and not fresh_restart
                    and not large_E_diff_between_relax_and_static
                ):
                    print("     %s is already converged; skipping" % xc_calc)
                    status = "done"
                    statuses[final_xc][xc_calc] = status
                    continue

                # for jobs that are not DONE:

                # (4) check for POSCAR
                # flag to check whether POSCAR is newly copied (don't want to perturb already-perturbed structures)
                fpos_dst = os.path.join(calc_dir, "POSCAR")
                if os.path.exists(fpos_dst):
                    # if there is a POSCAR, make sure its not empty
                    contents = open(fpos_dst, "r").readlines()
                    # if its empty, copy the initial structure to calc_dir
                    if len(contents) == 0:
                        copyfile(fpos_src, fpos_dst)
                # if theres no POSCAR, copy the initial structure to calc_dir
                if not os.path.exists(fpos_dst):
                    copyfile(fpos_src, fpos_dst)

                # (5) check for CONTCAR. if one exists, if its not empty, and if not fresh_restart, mark this job as one to "continue" (ie later, we'll copy CONTCAR to POSCAR); otherwise, mark as NEWRUN
                fcont_dst = os.path.join(calc_dir, "CONTCAR")
                if os.path.exists(fcont_dst):
                    contents = open(fcont_dst, "r").readlines()
                    if (
                        (len(contents) > 0)
                        and not fresh_restart
                        and not large_E_diff_between_relax_and_static
                    ):
                        status = "continue"
                    else:
                        status = "new"
                else:
                    status = "new"

                statuses[final_xc][xc_calc] = status

                # set our user_configs based on our vasp_configs + our calc_configs
                # note: vasp_configs should hold the baseline vasp_configs + our user_configs
                # note: calc_configs should just hold xc_to_run and calc_to_run as of now
                user_vasp_configs_before_error_handling = {
                    **vasp_configs,
                    **calc_configs,
                }

                # (6) initialize VASPSetUp with current VASP configs for this calculation
                vsu = VASPSetUp(
                    calc_dir=calc_dir,
                    user_configs=user_vasp_configs_before_error_handling,
                )

                # (7) check for errors in continuing jobs
                incar_changes = {}
                if status in ["continue", "new"]:
                    calc_is_clean = vsu.is_clean
                    if not calc_is_clean:
                        # change INCAR based on errors and include in calc_configs
                        incar_changes = vsu.incar_changes_from_errors

                # if there are INCAR updates, add them to calc_configs
                if incar_changes:
                    incar_key = "%s_incar" % calc_to_run
                    if incar_key not in calc_configs:
                        calc_configs[incar_key] = {}
                    for setting in incar_changes:
                        calc_configs[incar_key][setting] = incar_changes[setting]

                # update our vasp_configs with any modifications to the INCAR that we made to fix errors
                user_vasp_configs = {**vasp_configs, **calc_configs}
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
        A lot going on here. The objective is to write a submission script for each calculation
            - each submission script will launch a chain of jobs
            - this gets a bit tricky because a submission script is executed in bash
                - it's essentially like moving to a compute node and typing each line of the submission script into the compute node's command line
                - this means we can't really use python while the submission script is being executed

        1) check if job's in queue. if it is, just return

        2) write our slurm options at the top of sub file

        3) loop through all the calculations we want to do from this launch dir
            - label them as "done", "continue", or "new"

        4) for "continue"
            - copy CONTCAR to POSCAR to save progress

        5) for "new" and "continue"
            - figure out what parent calculations to get data from
                - e.g., gga-static for metagga-relax
            - make sure that parent calculation finished without errors before passing data to next calc
                - and before running next calc
                - if a parent calc didnt finish, but we've moved onto the next job, kill the job, so we can (automatically) debug the parent calc

        6) write VASP commands

        7) if lobster_static and calc is static, write LOBSTER and BADER commands
        """

        final_xcs = self.final_xcs

        # make copies of our starting configs
        vasp_configs = self.vasp_configs.copy()
        sub_configs = self.sub_configs.copy()

        # determine which files will be passed from parent calcs to children
        files_to_inherit = sub_configs["files_to_inherit"]

        launch_dir = self.launch_dir

        vasp_command = self.vasp_command
        slurm_options = self.slurm_options.copy()
        queue_manager = self.queue_manager

        # determine how jobs will be chained together
        packing = sub_configs["packing"]

        # get our statuses from when we prepared VASP input files for each directory
        # statuses = {final_xc : {xc_calc : status}}
        # e.g., statuses['metagga']['gga-relax'] = 'done'
        statuses = self.prepare_directories
        for final_xc in statuses:
            # initialize a submission script for this chain
            fsub = os.path.join(launch_dir, "sub_%s.sh" % final_xc)
            # initialize a status log file for this chain
            fstatus = os.path.join(launch_dir, "status_%s.o" % final_xc)
            # create a job name based on the launch_dir
            # e.g., formula.ID.standard.mag.final_xc
            job_name = ".".join(launch_dir.split("/")[-4:] + [final_xc])
            print("\nchecking if %s is in q" % job_name)
            if self.is_job_in_queue(job_name):
                continue
            slurm_options["job-name"] = job_name
            # if job isnt in queue already, start writing a new submission script
            with open(fsub, "w") as f:
                f.write("#!/bin/bash -l\n")
                # write the SLURM stuff at the top
                for key in slurm_options:
                    slurm_option = slurm_options[key]
                    if slurm_option:
                        f.write(
                            "%s --%s=%s\n" % (queue_manager, key, str(slurm_option))
                        )
                f.write("\n\n")
                f.write("ulimit -s unlimited\n")
                if sub_configs['mpi_command'] == 'mpirun':
                    f.write('module load impi/2018/release_multithread\n')
                # now write what is needed for the chain of VASP calcs + postprocessing
                print("\n:::: writing sub now - %s ::::" % fsub)

                # use this counter to figure if there are parents for a given calc and who those parents are
                xc_calc_counter = -1
                for xc_calc in packing[final_xc]:
                    # loop through the calculations that should be chained together for a given final_xc
                    xc_calc_counter += 1
                    # grab the status
                    status = statuses[final_xc][xc_calc]
                    xc_to_run, calc_to_run = xc_calc.split("-")
                    calc_dir = os.path.join(launch_dir, xc_calc)
                    f.write(
                        "\necho working on %s _%s_ >> %s\n" % (xc_calc, status, fstatus)
                    )
                    if status == "done":
                        # write postprocessing commands if needed
                        if vasp_configs["lobster_static"]:
                            if calc_to_run == "static":
                                if sub_configs[
                                    "force_postprocess"
                                ] or not os.path.exists(
                                    os.path.join(calc_dir, "lobsterout")
                                ):
                                    f.write(self.lobster_command)
                                if sub_configs[
                                    "force_postprocess"
                                ] or not os.path.exists(
                                    os.path.join(calc_dir, "ACF.dat")
                                ):
                                    f.write(self.bader_command)
                        f.write("echo %s is done >> %s\n" % (xc_calc, fstatus))
                    else:
                        if status == "continue":
                            # copy the CONTCAR to the POSCAR
                            f.write(
                                "cp %s %s\n"
                                % (
                                    os.path.join(calc_dir, "CONTCAR"),
                                    os.path.join(calc_dir, "POSCAR"),
                                )
                            )

                        # figure out if we need to pass data from a parent calculation
                        pass_info = False if xc_calc_counter == 0 else True
                        if pass_info:
                            parent_xc_calc = packing[final_xc][xc_calc_counter - 1]
                            src_dir = os.path.join(launch_dir, parent_xc_calc)
                            # before passing data, make sure parent has finished without crashing
                            f.write(
                                "isInFile=$(cat %s | grep -c %s)\n"
                                % (os.path.join(src_dir, "OUTCAR"), "Elaps")
                            )
                            f.write("if [ $isInFile -eq 0 ]; then\n")
                            f.write(
                                '   echo "%s is not done yet so this job is being killed" >> %s\n'
                                % (parent_xc_calc, fstatus)
                            )
                            f.write("   scancel $SLURM_JOB_ID\n")
                            f.write("fi\n")

                            # pass files that need to be inherited
                            for file_to_inherit in files_to_inherit:
                                # don't pass WAVECARs from loose since KPOINTS will change
                                if ("loose" in parent_xc_calc) and (
                                    file_to_inherit == "WAVECAR"
                                ):
                                    continue
                                fsrc = os.path.join(src_dir, file_to_inherit)
                                if file_to_inherit == "CONTCAR":
                                    fdst = os.path.join(calc_dir, "POSCAR")
                                else:
                                    fdst = os.path.join(calc_dir, file_to_inherit)
                                if file_to_inherit == "CONTCAR":
                                    if os.path.exists(fsrc):
                                        contents = open(fsrc).readlines()
                                        if len(contents) < 0:
                                            continue
                                f.write("cp %s %s\n" % (fsrc, fdst))

                        # navigate to calculation directory and run vasp
                        f.write("cd %s\n" % calc_dir)
                        f.write("%s\n" % vasp_command)

                        # include postprocessing stuff as requested
                        if vasp_configs["lobster_static"]:
                            if calc_to_run == "static":
                                if (
                                    not os.path.exists(
                                        os.path.join(calc_dir, "lobsterout")
                                    )
                                    or sub_configs["force_postprocess"]
                                ):
                                    f.write(self.lobster_command)
                                if (
                                    not os.path.exists(
                                        os.path.join(calc_dir, "ACF.dat")
                                    )
                                    or sub_configs["force_postprocess"]
                                ):
                                    f.write(self.bader_command)
                        f.write(
                            "\necho launched %s-%s >> %s\n"
                            % (xc_to_run, calc_to_run, fstatus)
                        )
        return True

    @property
    def launch_sub(self):
        """
        launch the submission script written in write_sub
            - if job is not in queue already
            - if there's something to launch
                (ie if all calcs are done, dont launch)
        """
        final_xcs = self.final_xcs

        print("     now launching sub")
        scripts_dir = os.getcwd()
        launch_dir = self.launch_dir

        # determine what keywords to look for to see if job needs to be launched
        flags_that_need_to_be_executed = self.sub_configs["execute_flags"]

        for final_xc in final_xcs:

            # identify the submission script for this chain
            fsub = os.path.join(launch_dir, "sub_%s.sh" % final_xc)
            with open(fsub) as f:
                for line in f:
                    if "job-name" in line:
                        job_name = line[:-1].split("=")[-1]
            # see if jobs in queue
            if self.is_job_in_queue(job_name):
                continue
            needs_to_launch = False
            # see if there's anything to launch
            with open(fsub) as f:
                contents = f.read()
                for flag in flags_that_need_to_be_executed:
                    if flag in contents:
                        needs_to_launch = True
            if not needs_to_launch:
                print(" !!! nothing to launch here, not launching\n\n")
                return

            # if we made it this far, launch it
            os.chdir(launch_dir)
            subprocess.call(["sbatch", "sub_%s.sh" % final_xc])
            os.chdir(scripts_dir)


def main():

    return


if __name__ == "__main__":
    sub = main()

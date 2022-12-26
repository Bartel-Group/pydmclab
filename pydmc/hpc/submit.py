from pydmc.hpc.vasp import VASPSetUp
from pydmc.hpc.analyze import AnalyzeVASP
from pydmc.utils.handy import read_yaml, write_yaml, dotdict
from pydmc.data.configs import load_vasp_configs, load_slurm_configs, load_sub_configs, load_partition_configs

from pymatgen.core.structure import Structure

import os
from shutil import copyfile, rmtree
import subprocess

HERE = os.path.dirname(os.path.abspath(__file__))

class SubmitTools(object):
    """
    This class is focused on figuring out how to prepare chains of calculations
        - the idea being that the output from this class is some file that you can
            "submit" to a queueing system
        - this class will automatically crawl through the VASP output files and figure out
            how to edit that submission file to finish the desired calculations
    
    """
    def __init__(self,
                 launch_dir,
                 xcs,
                 magmom,
                 user_configs={},
                 refresh_configs=['vasp', 'sub', 'slurm'],
                 vasp_configs_yaml=os.path.join(os.getcwd(), '_vasp_configs.yaml'),
                 slurm_configs_yaml=os.path.join(os.getcwd(), '_slurm_configs.yaml'),
                 sub_configs_yaml=os.path.join(os.getcwd(), '_sub_configs.yaml')):
        
        """
        Args:
            launch_dir (str) - directory to launch calculations from (to submit the submission file)
                - assumes initial structure is POSCAR in launch_dir
                - within this directory, various VASP calculation directories (calc_dirs) will be created
                        - gga-loose, gga-relax, gga-static, etc
                            - VASP will be run in each of these, but we need to run these sequentially, so we pack them together in one submission script
            
            xcs (list) - list of exchange correlation methods to use in that launch_dir
            magmom (list) - list of magnetic moments for each atom in structure (or None if not AFM)
            user_configs (dict) - any non-default parameters you want to pass 
                - these will override the defaults in the yaml files
                - look at pydmc/data/data/*configs*.yaml for the defaults
                    - note: _launch_configs.yaml will be modified using LaunchTools
                - you should be passing stuff here!
                - some common ones might be:
                    {'mag' : 'afm_3',
                    'standard' : 'dmc',
                    'nodes' : 2,
                    'job-name' : SOMETHING_SPECIFIC,
                    'lobster_static' : False,
                    etc
                    etc}
                    - anything that's not in the default yaml files that you want to apply to calculations in this launch_dir

            magmom (list) - list of magnetic moments for each atom in structure
                - only specify if 'afm' in mag, otherwise magmom=None is fine
                - e.g., if afm_1, you might grab these from a magmoms dictionary you made with MagTools as:
                    magmom = magmoms[mpid]['1']
                
            vasp_configs_yaml (os.pathLike) - path to yaml file containing vasp configs
                - there's usually no reason to change this
                - this holds some default configs for VASP
                - can always be changed with user_configs
 
             slurm_configs_yaml (os.pathLike) - path to yaml file containing slurm configs
                - there's usually no reason to change this
                - this holds some default configs for slurm
                - can always be changed with user_configs  
                
            sub_configs_yaml (os.pathLike) - path to yaml file containing submission file configs
                - there's usually no reason to change this
                - this holds some default configs for submission files
                - can always be changed with user_configs             
                
            files_to_inherit (list) - list of files to copy from calc to calc
                - this shouldn't have to be changed
                
            refresh_configs (bool) - if True, will refresh the yaml files with the pydmc defaults
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
        
        top_level, unique_ID, standard, mag = launch_dir.split('/')[-4:] 


        if not os.path.exists(vasp_configs_yaml) or ('vasp' in refresh_configs):
            _vasp_configs = load_vasp_configs()
            write_yaml(_vasp_configs, vasp_configs_yaml)

        if not os.path.exists(slurm_configs_yaml) or ('slurm' in refresh_configs):
            _slurm_configs = load_slurm_configs()
            write_yaml(_slurm_configs, slurm_configs_yaml)
        
        if not os.path.exists(sub_configs_yaml) or ('sub' in refresh_configs):
            _sub_configs = load_sub_configs()
            write_yaml(_sub_configs, sub_configs_yaml)
            
        user_configs_used = []
            
        slurm_configs = read_yaml(slurm_configs_yaml)
        for option in slurm_configs:
            if option in user_configs:
                if option not in user_configs_used:
                    new_value = user_configs[option]
                    slurm_configs[option] = new_value
                    user_configs_used.append(option)
        
        #write_yaml(slurm_configs, slurm_configs_yaml)
        self.slurm_configs = dotdict(slurm_configs)
        
        sub_configs = read_yaml(sub_configs_yaml)
        for option in sub_configs:
            if option in user_configs:
                if option not in user_configs_used:
                    new_value = user_configs[option]
                    sub_configs[option] = new_value
                    user_configs_used.append(option)

        #write_yaml(sub_configs, sub_configs_yaml)
        self.sub_configs = dotdict(sub_configs)

        vasp_configs = read_yaml(vasp_configs_yaml)

        user_configs = {option : user_configs[option] for option in user_configs if option not in user_configs_used}
        vasp_configs = {**vasp_configs, **user_configs}
        
        vasp_configs['standard'] = standard
        vasp_configs['mag'] = mag
        vasp_configs['magmom'] = magmom
        
        #write_yaml(vasp_configs, vasp_configs_yaml)
        self.vasp_configs = dotdict(vasp_configs)
        
        fpos = os.path.join(launch_dir, 'POSCAR')
        if not os.path.exists(fpos):
            raise FileNotFoundError('Need a POSCAR to initialize setup; POSCAR not found in {}'.format(self.launch_dir))
        else:
            self.structure = Structure.from_file(fpos)
    
        partitions = load_partition_configs()
        self.partitions = dotdict(partitions)
        
        self.xcs = xcs
    
    @property
    def queue_manager(self):
        """
        Returns queue manager (eg #SBATCH)
        """
        return self.sub_configs.manager
        
    @property
    def slurm_options(self):
        """
        Returns dictionary of slurm options
            - nodes, ntasks, walltime, etc
        """
        slurm_configs = self.slurm_configs
        options = {option : slurm_configs[option] for option in slurm_configs if slurm_configs[option]}
        partitions = self.partitions
        partition_specs = partitions[options['partition']]
        if partition_specs['proc'] == 'gpu':
            options['nodes'] = 1
            options['ntasks'] = 1
            options['gres'] = 'gpu:%s:%s' % (options['partition'].split('-')[0], str(options['nodes']))
        if not partition_specs['sharing']:
            options['ntasks'] = partition_specs['cores_per_node']
        return options
    
    @property
    def vasp_command(self):
        """
        Returns command used to execute vasp
            e.g., 'srun -n 24 PATH_TO_VASP/vasp_std > vasp.o'
        """
        sub_configs = self.sub_configs
        vasp_configs = self.vasp_configs
        vasp_exec = os.path.join(sub_configs.vasp_dir, sub_configs.vasp)
        return '\n%s --ntasks=%s --mpi=pmi2 %s > %s\n' % (sub_configs.mpi_command, str(self.slurm_options['ntasks']), vasp_exec, vasp_configs.fvaspout)
    
    @property
    def lobster_command(self):
        """
        Returns command used to execute lobster
        """
        lobster = '/home/cbartel/shared/bin/lobster/lobster-4.1.0/lobster-4.1.0'
        return '\n%s\n' % lobster
    
    @property
    def bader_command(self):
        """
        Returns command used to execute bader
        """
        chgsum = '/home/cbartel/shared/bin/bader/chgsum.pl AECCAR0 AECCAR2'
        bader = '/home/cbartel/shared/bin/bader/bader CHGCAR -ref CHGCAR_sum'
        return '\n%s\n%s\n' % (chgsum, bader)
          
    @property
    def prepare_directories(self):
        """
        This gets called by SubmitTools.write_sub, so you should rarely call this on its own
        
        A lot going on here. The objective is to prepare a set of directories for all calculations of interest to a self.launch_dir (which holds a single submission script)
              
        1) For each xc-calc pair, create a directory (calc_dir)
            - */launch_dir/xc-calc
            
        2) Check if that calc_dir has a converged VASP job
            - note: also checks "parents" (ie a static is labeled unconverged if its relax is unconverged)
            - if calc and parents are converged:
                - checks sub_configs['fresh_restart']
                    - if fresh_restart = False --> label calc_dir as status='DONE' and move on
                    - if fresh_restart = True --> start this calc over
                    
        3) Put */launch_dir/POSCAR into */launch_dir/xc-calc/POSCAR if there's not a POSCAR there already
        
        4) Check if */calc_dir/CONTCAR exists and has data in it,
            - if it does, copy */calc_dir/CONTCAR to */calc_dir/POSCAR and label status='CONTINUE' (ie continuing job)
            - if it doesn't, label status='NEW' (ie new job)
            
        5) Initialize VASPSetUp for calc_dir
            - modifies vasp_input_set with self.configs as requested in configs dictionaries (mainly vasp_configs which receives user_configs as well)
            
        6) If status in ['CONTINUE', 'NEW'],
            - check for errors using VASPSetUp
                - may remove WAVECAR/CHGCAR
                - will likely make edits to INCAR
                      
        """

        vasp_configs = self.vasp_configs
        sub_configs = self.sub_configs
        fresh_restart = sub_configs.fresh_restart
        launch_dir = self.launch_dir
        xcs = self.xcs
        
        packing = sub_configs.packing

        print('\n\n~~~~~ starting to work on %s ~~~~~\n\n' % launch_dir)

        fpos_src = os.path.join(launch_dir, 'POSCAR')
        statuses = {}
        for xc in xcs:
            statuses[xc] = {}
            for xc_calc in packing[xc]:
                # start making vasp_configs just for this particular calculation
                calc_configs = {}
                curr_xc, curr_calc = xc_calc.split('-')
                
                # update vasp configs with the current xc and calc
                calc_configs['xc'] = curr_xc
                calc_configs['calc'] = curr_calc
                
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
                    if curr_calc == 'static':
                        static_energy = E_per_at
                        parent_calc = 'relax'
                        parent_xc_calc = '%s-%s' % (curr_xc, parent_calc)                            
                        parent_calc_dir = os.path.join(launch_dir, parent_xc_calc)
                        parent_energy = AnalyzeVASP(parent_calc_dir).E_per_at
                        parent_convergence = True if parent_energy else False
                        if not parent_energy:
                            print('     %s (parent) not converged, need to continue this calc' % parent_xc_calc)
                        else:
                            if abs(parent_energy - static_energy) > 0.2:
                                print('     %s (parent) and %s (child) energies differ by more than 0.2 eV/atom' % (parent_xc_calc, xc_calc))
                                large_E_diff_between_relax_and_static = True
                                # if there is a large difference, something fishy happened, so let's start the static calc over
                    else:
                        parent_convergence = True
                
                # if parents + current calc are converged, give it status = DONE
                if convergence and parent_convergence and not fresh_restart and not large_E_diff_between_relax_and_static:
                    print('     %s is already converged; skipping' % xc_calc)
                    status = 'done'
                    statuses[xc][xc_calc] = status
                    continue
                
                # for jobs that are not DONE:

                # (4) check for POSCAR
                fpos_dst = os.path.join(calc_dir, 'POSCAR')
                if os.path.exists(fpos_dst):
                    # if there is a POSCAR, make sure its not empty
                    contents = open(fpos_dst, 'r').readlines()
                    # if its empty, copy the initial structure to calc_dir
                    if len(contents) == 0:
                        copyfile(fpos_src, fpos_dst)
                # if theres no POSCAR, copy the initial structure to calc_dir
                if not os.path.exists(fpos_dst):
                    copyfile(fpos_src, fpos_dst)
                
                # (5) check for CONTCAR. if one exists, if its not empty, and if not fresh_restart, mark this job as one to "CONTINUE" (ie later, we'll copy CONTCAR to POSCAR); otherwise, mark as NEWRUN
                fcont_dst = os.path.join(calc_dir, 'CONTCAR')
                if os.path.exists(fcont_dst):
                    contents = open(fcont_dst, 'r').readlines()
                    if (len(contents) > 0) and not fresh_restart and not large_E_diff_between_relax_and_static:
                        status = 'continue'
                    else:
                        status = 'new'
                else:
                    status = 'new'
                
                statuses[xc][xc_calc] = status

                # (6) initialize VASPSetUp with current VASP configs for this calculation
                vsu = VASPSetUp(calc_dir=calc_dir, 
                                user_configs={**vasp_configs, **calc_configs})
                
                # (6) check for errors in continuing jobs
                incar_changes = {}
                if status in ['continue', 'new']:
                    calc_is_clean = vsu.is_clean
                    if not calc_is_clean:
                        # change INCAR based on errors and include in calc_configs
                        incar_changes = vsu.incar_changes_from_errors
                
                # if there are INCAR updates, add them to calc_configs
                if incar_changes:
                    incar_key = '%s_incar' % curr_calc
                    if incar_key not in calc_configs:
                        calc_configs[incar_key] = {}
                    for setting in incar_changes:
                        calc_configs[incar_key][setting] = incar_changes[setting]
                            

                print('--------- may be some warnings (POTCAR ones OK) ----------')
                # (7) prepare calc_dir to launch  
                VASPSetUp(calc_dir=calc_dir,
                        user_configs={**vasp_configs, **calc_configs}).prepare_calc()
                
                print('-------------- warnings should be done ---------------')
                print('\n~~~~~ prepared %s ~~~~~\n' % calc_dir)
        return statuses
   
   
    def is_job_in_queue(self, job_name):
        """
        Returns:
            True if this job-name is already in the queue, else False
            
        Will prevent you from messing with directories that have running/pending jobs
        """
        scripts_dir = os.getcwd()
        sub_configs = self.sub_configs
        fqueue = os.path.join(scripts_dir, sub_configs.fqueue)
        with open(fqueue, 'w') as f:
            subprocess.call(['squeue', '-u', '%s' % os.getlogin(), '--name=%s' % job_name], stdout=f)
        names_in_q = []
        with open(fqueue) as f:
            for line in f:
                if 'PARTITION' not in line:
                    names_in_q.append([v for v in line.split(' ') if len(v) > 0][2])
        if len(names_in_q) > 0:
            print(' !!! job already in queue, not messing with it')
            return True
        
        return False
        
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
            - label them as "DONE", "CONTINUE", or "NEWRUN"
            
        4) for "CONTINUE"
            - copy CONTCAR to POSCAR to save progress
            
        5) for "NEWRUN" and "CONTINUE"
            - figure out what parent calculations to get data from
                - e.g., gga-static for metagga-relax
            - make sure that parent calculation finished without errors before passing data to next calc
                - and before running next calc
                - if a parent calc didnt finish, but we've moved onto the next job, kill the job, so we can (automatically) debug the parent calc

        6) write VASP commands
        
        7) if lobster_static and calc is static, write LOBSTER and BADER commands
        """

        xcs = self.xcs
                
        vasp_configs = self.vasp_configs
        sub_configs = self.sub_configs
        
        files_to_inherit = sub_configs.files_to_inherit
        
        launch_dir = self.launch_dir
        
        vasp_command = self.vasp_command
        slurm_options = self.slurm_options
        queue_manager = self.queue_manager


        packing = sub_configs.packing

        statuses = self.prepare_directories
        for xc in xcs:
            fsub = os.path.join(launch_dir, 'sub_%s.sh' % xc)
            fstatus = os.path.join(launch_dir, 'status_%s.o' % xc)
            job_name = '.'.join(launch_dir.split('/')[-4:]+[xc])
            print('\nchecking if %s is in q' % launch_dir)
            if self.is_job_in_queue(job_name):
                return
            slurm_options['job-name'] = job_name
            with open(fsub, 'w') as f:
                f.write('#!/bin/bash -l\n')
                for key in slurm_options:
                    slurm_option = slurm_options[key]
                    if slurm_option:
                        f.write('%s --%s=%s\n' % (queue_manager, key, str(slurm_option)))
                f.write('\n\n')
                f.write('ulimit -s unlimited\n')
                print('\n:::: writing sub now - %s ::::' % fsub)
                xc_calc_counter = -1
                for xc_calc in packing[xc]:
                    xc_calc_counter += 1
                    status = statuses[xc][xc_calc]
                    xc_to_run, calc_to_run = xc_calc.split('-')                    
                    calc_dir = os.path.join(launch_dir, xc_calc)
                    f.write('\necho working on %s >> %s\n' % (xc_calc, fstatus))
                    if status == 'done':
                        if vasp_configs['lobster_static']:
                            if calc_to_run == 'static':
                                if sub_configs.force_postprocess or not os.path.exists(os.path.join(calc_dir, 'lobsterout')):
                                    f.write(self.lobster_command)
                                if sub_configs.force_postprocess or not os.path.exists(os.path.join(calc_dir, 'ACF.dat')):
                                    f.write(self.bader_command)
                        f.write('echo %s is done >> %s\n' % (xc_calc, fstatus))
                    else:
                        if status == 'continue':
                            f.write('cp %s %s\n' % (os.path.join(calc_dir, 'CONTCAR'), os.path.join(calc_dir, 'POSCAR')))
                        
                        pass_info = False if xc_calc_counter == 0 else True                    
                        if pass_info:
                            parent_xc_calc = packing[xc][xc_calc_counter - 1]
                            src_dir = os.path.join(launch_dir, parent_xc_calc)
                            f.write('isInFile=$(cat %s | grep -c %s)\n' % (os.path.join(src_dir, 'OUTCAR'), 'Elaps'))
                            f.write('if [ $isInFile -eq 0 ]; then\n')
                            f.write('   echo "%s is not done yet so this job is being killed" >> %s\n' % (parent_xc_calc, fstatus))
                            f.write('   scancel $SLURM_JOB_ID\n')
                            f.write('fi\n')

                            for file_to_inherit in files_to_inherit:
                                if ('loose' in parent_xc_calc) and (file_to_inherit == 'WAVECAR'):
                                    continue
                                fsrc = os.path.join(src_dir, file_to_inherit)
                                if file_to_inherit == 'CONTCAR':
                                    fdst = os.path.join(calc_dir, 'POSCAR')
                                else:
                                    fdst = os.path.join(calc_dir, file_to_inherit)
                                if file_to_inherit == 'CONTCAR':
                                    if os.path.exists(fsrc):
                                        contents = open(fsrc).readlines()
                                        if len(contents) < 0:
                                            continue
                                f.write('cp %s %s\n' % (fsrc, fdst))
                        
                        f.write('cd %s\n' % calc_dir)
                        f.write('%s\n' % vasp_command)
                        if vasp_configs['lobster_static']:
                            if calc_to_run == 'static':
                                if not os.path.exists(os.path.join(calc_dir, 'lobsterout')) or sub_configs.force_postprocess:
                                    f.write(self.lobster_command)
                                if not os.path.exists(os.path.join(calc_dir, 'ACF.dat')) or sub_configs.force_postprocess:
                                    f.write(self.bader_command)
                        f.write('\necho launched %s-%s >> %s\n' % (xc_to_run, calc_to_run, fstatus))
        return True
    
    @property
    def launch_sub(self):
        """
        launch the submission script written in write_sub
            - if job is not in queue already
            - if there's something to launch
                (ie if all calcs are done, dont launch)
        """
        xcs = self.xcs
        
        print('     now launching sub')
        scripts_dir = os.getcwd()
        launch_dir = self.launch_dir
        sub_configs = self.sub_configs   
        packing = sub_configs.packing
        flags_that_need_to_be_executed = ['srun', 'python', 'lobster', 'bader']

        for xc in xcs:

            fsub = os.path.join(launch_dir, 'sub_%s.sh' % xc)
            with open(fsub) as f:
                for line in f:
                    if 'job-name' in line:
                        job_name = line[:-1].split('=')[-1]
            if self.is_job_in_queue(job_name):
                continue
            needs_to_launch = False
            with open(fsub) as f:
                contents = f.read()
                for flag in flags_that_need_to_be_executed:
                    if flag in contents:
                        needs_to_launch = True
            if not needs_to_launch:
                print(' !!! nothing to launch here, not launching\n\n')
                return

            os.chdir(launch_dir)
            subprocess.call(['sbatch', 'sub_%s.sh' % xc])
            os.chdir(scripts_dir)

def main():
    
    
    return 
    
if __name__ == '__main__':
    sub = main()

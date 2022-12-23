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
                 valid_calcs,
                 user_configs={},
                 magmom=None,
                 vasp_configs_yaml=os.path.join(os.getcwd(), '_vasp_configs.yaml'),
                 slurm_configs_yaml=os.path.join(os.getcwd(), '_slurm_configs.yaml'),
                 sub_configs_yaml=os.path.join(os.getcwd(), '_sub_configs.yaml'),
                 files_to_inherit=['WAVECAR', 'CONTCAR'],
                 refresh_configs=[]):
        
        """
        Args:
            launch_dir (str) - directory to launch calculations from (to submit the submission file)
                - assumes initial structure is POSCAR in launch_dir
                - within this directory, various VASP calculation directories (calc_dirs) will be created
                        - gga-loose, gga-relax, gga-static, etc
                            - VASP will be run in each of these, but we need to run these sequentially, so we pack them together in one submission script
            
            valid_calcs (list) - list of calculations to run
                - this will get created automatically by LaunchTools
                    - i.e. it will figure out what are the minimal number of necessary calculations to run for a given launch_dir
                        (it will also figure out the minimal number of launch_dirs)
                - each item in the list is formatted as xc-calc (e.g., gga-loose)
                
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
        self.valid_calcs = valid_calcs

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
        
        if not slurm_configs['job-name']:
            slurm_configs['job-name'] = '.'.join(launch_dir.split('/')[-5:])
        
        self.slurm_configs = dotdict(slurm_configs)
        
        sub_configs = read_yaml(sub_configs_yaml)
        for option in sub_configs:
            if option in user_configs:
                if option not in user_configs_used:
                    new_value = user_configs[option]
                    sub_configs[option] = new_value
                    user_configs_used.append(option)

        self.sub_configs = dotdict(sub_configs)

        vasp_configs = read_yaml(vasp_configs_yaml)

        user_configs = {option : user_configs[option] for option in user_configs if option not in user_configs_used}
        vasp_configs = {**vasp_configs, **user_configs}
        self.vasp_configs = dotdict(vasp_configs)
        
        fpos = os.path.join(launch_dir, 'POSCAR')
        if not os.path.exists(fpos):
            raise FileNotFoundError('Need a POSCAR to initialize setup; POSCAR not found in {}'.format(self.launch_dir))
        else:
            self.structure = Structure.from_file(fpos)
    
        self.files_to_inherit = files_to_inherit
        partitions = load_partition_configs()
        self.partitions = dotdict(partitions)
        
        self.magmom = magmom

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
        configs = self.sub_configs
        vasp_exec = os.path.join(configs.vasp_dir, configs.vasp)
        return '\n%s --ntasks=%s --mpi=pmi2 %s > %s\n' % (configs.mpi_command, str(self.slurm_options['ntasks']), vasp_exec, configs.fvaspout)
    
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
    def calcs(self):
        """
        Returns list of calcs to run 
            - (eg ['loose', 'relax', 'static'])
        """
        vasp_configs = self.vasp_configs
        sub_configs = self.sub_configs
        calc = vasp_configs.calc
        calc_sequence = sub_configs.calc_sequence
        if calc_sequence and (calc == 'relax'):
            calcs = ['loose', 'relax', 'static']
        else:
            calcs = [calc]
        return calcs
    
    @property
    def xcs(self):
        """
        Returns list of exchange-correlation approaches to run
            - eg ['gga', 'metagga']
        """
        vasp_configs = self.vasp_configs
        sub_configs = self.sub_configs
        xc = vasp_configs.xc
        xc_sequence = sub_configs.xc_sequence
        if xc_sequence and (xc == 'metagga'):
            xcs = ['gga', xc]
        else:
            xcs = [xc]
        return xcs
          
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
        prepare_calc_options = ['potcar_functional',
                                'validate_magmom',
                                'mag',
                                'standard',
                                'fun']
        vasp_configs = self.vasp_configs
        sub_configs = self.sub_configs
        fresh_restart = sub_configs.fresh_restart
        calcs = self.calcs
        xcs = self.xcs
        valid_calcs = self.valid_calcs
        launch_dir = self.launch_dir

        print('\n\n~~~~~ starting to work on %s ~~~~~\n\n' % launch_dir)


        fpos_src = os.path.join(launch_dir, 'POSCAR')
        tags = []
        for xc in xcs:
            for calc in calcs:
                xc_calc = '%s-%s' % (xc, calc)

                # (0) make sure (xc, calc) combination is "valid" (determined with LaunchTools)
                if xc_calc not in valid_calcs:
                    print('     skipping %s b/c we probably dont need it' % xc_calc)
                    continue

                # (1) make calc_dir (or remove and remake if fresh_restart)
                calc_dir = os.path.join(launch_dir, xc_calc)
                if os.path.exists(calc_dir) and fresh_restart:
                    rmtree(calc_dir)
                if not os.path.exists(calc_dir):
                    os.mkdir(calc_dir)

                # (2) identify if given (xc, calc) has parents that need to finish first
                parents = []
                if calc == 'static':
                    for possible_parent_calc in ['relax']:
                        parent_xc_calc = '%s-%s' % (xc, possible_parent_calc)
                        if parent_xc_calc in valid_calcs:
                            parents.append(parent_xc_calc)
                            
                # (3) check convergence of current calc and parents
                convergence = AnalyzeVASP(calc_dir).is_converged
                all_parents_converged = True
                for parent_xc_calc in parents:
                    parent_calc_dir = os.path.join(launch_dir, parent_xc_calc)
                    parent_convergence = AnalyzeVASP(parent_calc_dir).is_converged
                    if not parent_convergence:
                        all_parents_converged = False
                        print('     %s (parent) not converged, need to continue this calc' % parent_xc_calc)

                # if parents + current calc are converged, give it status = DONE
                if convergence and all_parents_converged and not fresh_restart:
                    print('     %s is already converged; skipping' % xc_calc)
                    status = 'DONE'
                    tags.append('%s_%s' % (status, xc_calc))
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
                    if (len(contents) > 0) and not fresh_restart:
                        status = 'CONTINUE'
                    else:
                        status = 'NEWRUN'
                else:
                    status = 'NEWRUN'

                # (6) initialize VASPSetUp with configs
                vsu = VASPSetUp(calc_dir=calc_dir, 
                                magmom=self.magmom,
                                fvaspout=sub_configs.fvaspout,
                                fvasperrors=sub_configs.fvasperrors,
                                lobster_static=vasp_configs.lobster_static) 
                
                # pass loose/static/relax_INCAR/KPOINTS/POTCAR from vasp_configs to this calc
                calc_configs = {'modify_%s' % input_file.lower() : 
                    vasp_configs['%s_%s' % (calc, input_file)] for input_file in ['INCAR', 'KPOINTS', 'POTCAR']}
                # pass the other vasp_configs to this calc
                for key in prepare_calc_options:
                    calc_configs[key] = vasp_configs[key]
                
                # (6) check for errors in continuing jobs
                if status in ['CONTINUE', 'NEWRUN']:
                    calc_is_clean = vsu.is_clean
                    if not calc_is_clean:
                        # change INCAR based on errors and include in calc_configs
                        incar_changes = vsu.incar_changes_from_errors
                        calc_configs['modify_incar'] = {**calc_configs['modify_incar'], **incar_changes}

                print('--------- may be some warnings (POTCAR ones OK) ----------')
                # (7) prepare calc_dir to launch  
                vsu.prepare_calc(calc=calc,
                                xc=xc,
                                **calc_configs)
                
                print('-------------- warnings should be done ---------------')
                print('\n~~~~~ prepared %s ~~~~~\n' % calc_dir)
                tags.append('%s_%s' % (status, xc_calc))
        return tags
   
   
    @property
    def is_job_in_queue(self):
        """
        Returns:
            True if this job-name is already in the queue, else False
            
        Will prevent you from messing with directories that have running/pending jobs
        """
        scripts_dir = os.getcwd()
        sub_configs = self.sub_configs
        fqueue = os.path.join(scripts_dir, sub_configs.fqueue)
        job_name = self.slurm_configs['job-name']
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

                
        vasp_configs = self.vasp_configs
        sub_configs = self.sub_configs
        
        launch_dir = self.launch_dir
        print('\nchecking if %s is in q' % launch_dir)
        if self.is_job_in_queue:
            return
        
        vasp_command = self.vasp_command
        slurm_options = self.slurm_options
        queue_manager = self.queue_manager

        files_to_inherit = self.files_to_inherit

        fsub = os.path.join(launch_dir, sub_configs.fsub)
        
        fstatus = os.path.join(launch_dir, sub_configs.fstatus)

        with open(fsub, 'w') as f:
            f.write('#!/bin/bash -l\n')
            for key in slurm_options:
                slurm_option = slurm_options[key]
                if slurm_option:
                    f.write('%s --%s=%s\n' % (queue_manager, key, str(slurm_option)))
            f.write('\n\n')
            f.write('ulimit -s unlimited\n')
            tags = self.prepare_directories
            xc_calcs = [t.split('_')[1] for t in tags]
            curr_xcs = [xc_calc.split('-')[0] for xc_calc in xc_calcs]
            print('\n:::: writing sub now - %s ::::' % fsub)
            for tag in tags:
                status = tag.split('_')[0]
                xc, calc = tag.split('_')[1].split('-')
                curr_calcs = [xc_calc.split('-')[1] for xc_calc in xc_calcs if xc_calc.split('-')[0] == xc]
                curr_calcs = sorted(list(set(curr_calcs)))
                
                calc_dir = os.path.join(launch_dir, '-'.join([xc, calc]))
                if status == 'DONE':
                    f.write('\necho working on %s >> %s\n' % (tag, fstatus))
                    if vasp_configs['lobster_static']:
                        if sub_configs.force_postprocess or not os.path.exists(os.path.join(calc_dir, 'lobsterout')):
                            f.write(self.lobster_command)
                        if sub_configs.force_postprocess or not os.path.exists(os.path.join(calc_dir, 'ACF.dat')):
                            f.write(self.bader_command)
                    f.write('echo %s is done >> %s\n' % (tag.split('_')[1], fstatus))
                else:
                    if status == 'CONTINUE':
                        f.write('\necho working on %s >> %s\n' % (tag, fstatus))
                        f.write('cp %s %s\n' % (os.path.join(calc_dir, 'CONTCAR'), os.path.join(calc_dir, 'POSCAR')))
                    if status == 'NEWRUN':
                        f.write('\necho working on %s >> %s\n' % (tag, fstatus))
                    
                    if xc in ['gga', 'ggau']:
                        if calc == 'loose':
                            pass_info = False
                        elif calc == 'relax':
                            if 'loose' in curr_calcs:
                                pass_info = True
                                calc_prev = 'loose'
                            else:
                                pass_info = False
                        elif calc == 'static':
                            if 'relax' in curr_calcs:
                                pass_info = True
                                calc_prev = 'relax'
                            else:
                                pass_info = False
                        xc_prev = xc
                    elif xc in ['metagga']:
                        if calc == 'relax':
                            if 'gga' in curr_xcs:
                                xc_prev = 'gga'
                                if 'gga-static' in xc_calcs:
                                    calc_prev = 'static'
                                    pass_info = True
                                elif 'gga-relax' in xc_calcs:
                                    calc_prev = 'relax'
                                    pass_info = True
                                else:
                                    print('no gga calc to pass from for metagga...')
                                    pass_info = False
                            else:
                                pass_info = False
                        elif calc == 'static':
                            if 'relax' in curr_calcs:
                                pass_info = True
                                calc_prev = 'relax'
                                xc_prev = xc
                            else:
                                pass_info = False
                    
                    if pass_info:
                        src_dir = os.path.join(launch_dir, '-'.join([xc_prev, calc_prev]))
                        f.write('isInFile=$(cat %s | grep -c %s)\n' % (os.path.join(src_dir, 'OUTCAR'), 'Elaps'))
                        f.write('if [ $isInFile -eq 0 ]; then\n')
                        f.write('   echo "%s is not done yet so this job is being killed" >> %s\n' % (calc_prev, fstatus))
                        f.write('   scancel $SLURM_JOB_ID\n')
                        f.write('fi\n')

                        for file_to_inherit in files_to_inherit:
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
                    if calc == 'static':
                        if vasp_configs['lobster_static']:
                            if not os.path.exists(os.path.join(calc_dir, 'lobsterout')) or sub_configs.force_postprocess:
                                f.write(self.lobster_command)
                            if not os.path.exists(os.path.join(calc_dir, 'ACF.dat')) or sub_configs.force_postprocess:
                                f.write(self.bader_command)
                    f.write('\necho launched %s-%s >> %s\n' % (xc, calc, fstatus))
        return True
    
    @property
    def launch_sub(self):
        """
        launch the submission script written in write_sub
            - if job is not in queue already
            - if there's something to launch
                (ie if all calcs are done, dont launch)
        """
        if self.is_job_in_queue:
            return
        
        print('     now launching sub')
        scripts_dir = os.getcwd()
        launch_dir = self.launch_dir
        sub_configs = self.sub_configs   

        fsub = os.path.join(launch_dir, sub_configs.fsub)
        flags_that_need_to_be_executed = ['srun', 'python', 'lobster', 'bader']
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
        subprocess.call(['sbatch', 'sub.sh'])
        os.chdir(scripts_dir)

def main():
    
    
    return 
    
if __name__ == '__main__':
    sub = main()

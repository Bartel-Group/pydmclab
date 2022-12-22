from pydmc.VASPTools import VASPSetUp, VASPAnalysis
from pydmc.MagTools import MagTools
from pydmc.handy import read_yaml, dotdict, write_json, read_json
from pymatgen.core.structure import Structure

import os
from shutil import copyfile, rmtree
import subprocess

import warnings


HERE = os.path.dirname(os.path.abspath(__file__))

def is_calc_valid(structure,
                  standard,
                  xc,
                  calc,
                  mag,
                  magmom,
                  mag_override):
    """
    Returns:
        True if calculation should be launched; 
        False if some logic is violated
    """
    
    if standard == 'mp':
        if xc != 'ggau':
            return False
        
    if not mag_override:
        if mag == 'nm':
            if MagTools(structure).could_be_magnetic:
                return False
        elif 'afm' in mag:
            if not MagTools(structure).could_be_afm:
                return False
        elif mag == 'fm':
            if not MagTools(structure).could_be_magnetic:
                return False
        
    if 'afm' in mag:
        if not magmom:
            return False
        
    if xc == 'metagga':
        if calc == 'loose':
            return False
        
    return True

class SubmitTools(object):
    
    def __init__(self,
                 launch_dir,
                 valid_calcs,
                 vasp_configs_yaml=os.path.join(os.getcwd(), '_vasp_configs.yaml'),
                 slurm_configs_yaml=os.path.join(os.getcwd(), '_slurm_configs.yaml'),
                 sub_configs_yaml=os.path.join(os.getcwd(), '_sub_configs.yaml'),
                 user_configs={},
                 magmom=None,
                 files_to_inherit=['WAVECAR', 'CONTCAR'],
                 partitions_yaml=os.path.join(HERE, '_partitions.yaml'),
                 refresh_configs=[]):
        
        """
        Args:
            launch_dir (str) - directory to launch calculations from
                - assumes initial structure is POSCAR in launch_dir
                - this might be LiMn2O4_mp-123456
                    - then within this directory, you might generate a set of calc_dirs
                        - gga-loose, gga-relax, gga-static, etc
            config_yaml (str) - path to yaml file containing base configs
                if None:
                    - get them from pydmc
            user_configs (dict) - dictionary of user configs to override base configs
                common ones:
                    {'mag' : 'afm',
                    'standard' : 'mp',
                    'nodes' : 2,
                    'time' : '96:00:00',
                    'job-name' : SOMETHING_SPECIFIC}
                    
            magmom (list) - list of magnetic moments for each atom in structure
                - only specify if mag='afm'
                
            files_to_inherit (list) - list of files to copy from calc to calc
        
        Returns:
            self.launch_dir (os.pathLike) - directory to launch calculations from
            self.configs (dict) - dictionary of configs (in format similar to yaml)
            self.files_to_inherit (list) - list of files to copy from calc to calc
        """
        
        self.launch_dir = launch_dir
        self.valid_calcs = valid_calcs

        if not os.path.exists(vasp_configs_yaml) or ('vasp' in refresh_configs):
            pydmc_vasp_configs_yaml = os.path.join(HERE, '_vasp_configs.yaml')
            copyfile(pydmc_vasp_configs_yaml, vasp_configs_yaml)

        if not os.path.exists(slurm_configs_yaml) or ('slurm' in refresh_configs):
            pydmc_slurm_configs_yaml = os.path.join(HERE, '_slurm_configs.yaml')
            copyfile(pydmc_slurm_configs_yaml, slurm_configs_yaml)
        
        if not os.path.exists(sub_configs_yaml) or ('sub' in refresh_configs):
            pydmc_sub_configs_yaml = os.path.join(HERE, '_sub_configs.yaml')
            copyfile(pydmc_sub_configs_yaml, sub_configs_yaml)
            
        slurm_configs = read_yaml(slurm_configs_yaml)
        for option in slurm_configs:
            if option in user_configs:
                slurm_configs[option] = user_configs[option]
                del user_configs[option]
        
        if not slurm_configs['job-name']:
            slurm_configs['job-name'] = '.'.join(launch_dir.split('/')[-5:])
        
        self.slurm_configs = dotdict(slurm_configs)
        
        sub_configs = read_yaml(sub_configs_yaml)
        for option in sub_configs:
            if option in user_configs:
                sub_configs[option] = user_configs[option]
                del user_configs[option]

        self.sub_configs = dotdict(sub_configs)

        vasp_configs = read_yaml(vasp_configs_yaml)

        vasp_configs = {**vasp_configs, **user_configs}
        self.vasp_configs = dotdict(vasp_configs)
        
        fpos = os.path.join(launch_dir, 'POSCAR')
        if not os.path.exists(fpos):
            raise FileNotFoundError('Need a POSCAR to initialize setup; POSCAR not found in {}'.format(self.launch_dir))
        else:
            self.structure = Structure.from_file(fpos)
    
        self.files_to_inherit = files_to_inherit
        self.magmom = magmom
        
        self.partitions = dotdict(read_yaml(partitions_yaml))

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
        options = self.slurm_configs
        options = {k : v for k, v in options.items() if v is not None}
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
            e.g., 'mpirun -n 24 PATH_TO_VASP/vasp_std > vasp.o'
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
        A lot going on here. The objective is to prepare a set of directories for all calculations 
            - for a given initial crystal structure (material)
                - */launch_dir/POSCAR 
            - at a given magnetic ordering
                - user_configs['mag'] = 'nm', 'fm', OR 'afm'
                - if 'afm', must supply self.magmom
        
        When to call this:
            - when you are setting up a set of calculations for the first time
            - when you want to loop through a set of calculations that have partially finished
                - to run the ones that haven't started/finished yet (fresh_restart=False)
                - to re-run them all (fresh_restart=True)
        
        1) For each xc-calc pair, create a directory (calc_dir)
            - */launch_dir/xc-calc
        2) Check if that calc_dir has a converged VASP job
            - if it does
                - if fresh_restart = False --> label calc_dir as status='DONE' and move on
                - if fresh_restart = True --> remove prev calc in that dir
        3) Put */launch_dir/POSCAR into */launch_dir/xc-calc/POSCAR if there's not a POSCAR there already
        4) Check if */calc_dir/CONTCAR exists and has data in it,
            - if it does, copy */calc_dir/CONTCAR to */calc_dir/POSCAR and label status='CONTINUE' (ie continuing job)
            - if it doeqmsn't, label status='NEW' (ie new job)
        5) Initialize VASPSetUp for calc_dir
            - modifies vasp_input_set with self.configs as requested in **kwargs and configs.yaml
        6) If status='CONTINUE',
            - check for errors using vsu
                - may remove WAVECAR/CHGCAR
                - may make edits to INCAR
        7) prepare calc_dir to launch        
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
                convergence = VASPAnalysis(calc_dir).is_converged
                all_parents_converged = True
                for parent_xc_calc in parents:
                    parent_calc_dir = os.path.join(launch_dir, parent_xc_calc)
                    parent_convergence = VASPAnalysis(parent_calc_dir).is_converged
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
    def write_sub(self):
        """
        A lot going on here. The objective is to write a submission script for each calculation
            - so that I can iterate through a for loop and launch all calcs
            - each submission script will launch a chain of jobs
        
        """
        
        vasp_configs = self.vasp_configs
        sub_configs = self.sub_configs
        slurm_configs = self.slurm_configs
        
        launch_dir = self.launch_dir
        vasp_command = self.vasp_command
        slurm_options = self.slurm_options
        queue_manager = self.queue_manager

        xcs, calcs = self.xcs, self.calcs

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
                        if not os.path.exists(os.path.join(calc_dir, 'lobsterout')) or sub_configs.force_postprocess:
                            f.write(self.lobster_command)
                        if not os.path.exists(os.path.join(calc_dir, 'ACF.dat')) or sub_configs.force_postprocess:
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
        print('     now launching sub')
        scripts_dir = os.getcwd()
        launch_dir = self.launch_dir
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
            print(' !!! job already in queue, not launching')
            return

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
                
class LaunchTools(object):
    
    def __init__(self,
                 calcs_dir,
                 structure,
                 magmoms=None,
                 top_level='formula',
                 unique_ID='my-1234',
                 fyaml=os.path.join(os.getcwd(), '_launch_configs.yaml'),
                 user_configs={}):
        """
        Args:
        
            calcs_dir (os.path): directory where calculations will be launched
                - top_level will go at calcs_dir/top_level
                
            structure (Structure): pymatgen structure object
            
            top_level (str): top level directory
        - could be whatever you want
        - for instance, if I were looking at Li_{x}Co10O20 at varying x values between 0 and 10, I might make top_level = LiCoO2
        - if I was just running a geometry relaxation on a given chemical formula (let's say LiTiS2), I would call the top_level = LiTiS2 or even better, Li1S2Ti1 (the CompTools(formula).clean standard)
            
            unique_ID (str): level below top_level
                - could be a material ID in materials project
                - could be x in the example I described previously
                - up to you
        
            compare_to_mp (bool): whether calculations will be compared to mp
                - this will add standard = 'mp' to standards if it's not there already
                - this will add xc = 'ggau' to xcs if it's not there already
                
            n_afm_configs (int): number of AFM configurations to run
                - 0 means don't run AFM
                
            magmoms (dict): path to json that contains {index of configuration (int) : magmom (list) generated using MagTools}
                - if None and n_afm_configs > 0, will be generated here
                    - this is not advisable --> better to generate all afm magmoms for all materials, save as json, then pass that dict here

            relax (bool): whether structures will be relaxed
                - this should almost always be true
                
            override_mag (bool): whether to run nm for mag systems or fm for nm systems
                - this should almost always be false
        """
        
        if not os.path.exists(calcs_dir):
            os.mkdir(calcs_dir)
            
        if not os.path.exists(fyaml):
            pydmc_yaml = os.path.join(HERE, 'launch_configs.yaml')
            copyfile(pydmc_yaml, fyaml)
        
        launch_configs = read_yaml(fyaml)
        
        configs = {**launch_configs, **user_configs}

        configs['top_level'] = top_level
        configs['unique_ID'] = unique_ID
        configs['calcs_dir'] = calcs_dir
        
        if not isinstance(structure, dict):
            structure = structure.as_dict()

        configs['structure'] = structure

        if configs['n_afm_configs'] > 0:
            if MagTools(structure).could_be_afm:
                if not magmoms:
                    raise ValueError('You are running afm calculations but provided no magmoms, generate these first, then pass to LaunchTools')
        
        configs['magmoms'] = magmoms

        standards = configs['standards'].copy()
        if configs['compare_to_mp']:
            if 'mp' not in standards:
                standards.append('mp')

        configs['standards'] = standards

        self.configs = dotdict(configs)

    
    @property
    def valid_calcs(self):
        configs = self.configs
        if configs.relax_geometry:
            possible_calcs = ['loose', 'relax', 'static']
        else:
            possible_calcs = ['static']
        
        possible_mags = ['nm', 'fm']
        if configs.n_afm_configs > 0:
            possible_mags += ['afm_%s' % str(i) for i in range(configs.n_afm_configs)]

        xcs = configs.xcs.copy()
        if 'metagga' in xcs:
            if 'gga' not in xcs:
                xcs.append('gga')

        if configs.compare_to_mp:
            if 'ggau' not in xcs:
                xcs.append('ggau')

        standards = configs.standards                
        out = []
        for standard in standards:
            for xc in xcs:
                if (standard != 'mp') and (xc == 'ggau') and ('ggau' not in configs.xcs):
                    continue
                for mag in possible_mags:
                    for calc in possible_calcs:
                        if 'afm' in mag:
                            magmoms = configs.magmoms
                            idx = mag.split('_')[-1]
                            if (idx in magmoms):
                                magmom = magmoms[idx]
                            elif str(idx) in magmoms:
                                magmom = magmoms[str(idx)]
                            else:
                                magmom = None
                        else:
                            magmom = None
                        validity = is_calc_valid(configs.structure,
                                                    standard,
                                                    xc,
                                                    calc,
                                                    mag,
                                                    magmom,
                                                    configs.mag_override)
                        if validity:
                            out.append({'standard' : standard,
                                        'xc' : xc,
                                        'mag' : mag,
                                        'calc' : calc})
                        
        return out

    @property
    def launch_dirs(self):
        configs = self.configs
            
        valid_calcs = self.valid_calcs
        
        launch_dirs = []
    
        level0 = configs.calcs_dir
        level1 = configs.top_level
        level2 = configs.unique_ID
        
        unique_standards = sorted(list(set([d['standard'] for d in valid_calcs])))
        
        for standard in unique_standards:
            standard_dir = os.path.join(level0, level1, level2, standard)
            unique_xcs = [d['xc'] for d in valid_calcs if d['standard'] == standard]
            for xc in unique_xcs:
                if (xc == 'gga') and ('metagga' in unique_xcs):
                    continue
                xc_dir = os.path.join(standard_dir, xc)
                unique_mags = [d['mag'] for d in valid_calcs if d['standard'] == standard and d['xc'] == xc]
                for mag in unique_mags:
                    mag_dir = os.path.join(xc_dir, mag)
                    launch_dirs.append(mag_dir)
        
        return launch_dirs
    
    @property
    def launch_dirs_to_tags(self):
        launch_dirs = self.launch_dirs
        valid_calcs = self.valid_calcs
        
        d = {}
        for launch_dir in launch_dirs:
            d[launch_dir] = []
            standard, xc, mag = launch_dir.split('/')[-3:]
            for calc in valid_calcs:
                tag = '-'.join([calc['xc'], calc['calc']])
                calc_standard, calc_xc, calc_mag = calc['standard'], calc['xc'], calc['mag']
                if calc_standard == standard:
                    if calc_mag == mag:
                        if (calc_xc == xc) or ((calc_xc == 'gga') and (xc == 'metagga')):
                            d[launch_dir].append(tag)
                            
        return d
                
    @property
    def create_launch_dirs_and_make_POSCARs(self):
        
        launch_dirs = self.launch_dirs_to_tags
        for launch_dir in launch_dirs:
            if not os.path.exists(launch_dir):
                os.makedirs(launch_dir)
        
            fposcar = os.path.join(launch_dir, 'POSCAR')
            if not os.path.exists(fposcar):
                structure = Structure.from_dict(self.configs.structure)
                structure.to(fmt='poscar', filename=fposcar)


def make_sub_for_launcher():
    flauncher_sub = os.path.join(os.getcwd(), 'sub_launch.sh')
    launch_job_name = '-'.join([os.getcwd().split('/')[-2], 'launcher'])
    with open(flauncher_sub, 'w') as f:
        f.write('#!/bin/bash -l\n')
        f.write('#SBATCH --nodes=1\n')
        f.write('#SBATCH --ntasks=1\n')
        f.write('#SBATCH --time=4:00:00\n')
        f.write('#SBATCH --error=log_launch.e\n')
        f.write('#SBATCH --output=log_launch.o\n')
        f.write('#SBATCH --account=cbartel\n')
        f.write('#SBATCH --job-name=%s\n' % launch_job_name)
        f.write('#SBATCH --partition=msismall\n')
        f.write('\npython launch.py\n')

class BatchAnalysis(object):

    def __init__(self,
                 launch_dirs_to_tags):

        self.launch_dirs_to_tags = launch_dirs_to_tags

    def get_results(self,
                    top_level_key='formula',
                    magnetization=False,
                    relaxed_structure=False,
                    dos=None,
                    use_static=True,
                    check_relax=0.1):
        launch_dirs = self.launch_dirs_to_tags
        data = []
        for launch_dir in launch_dirs:
            print('\n~~~ analyzing %s ~~~' % launch_dir)
            top, ID, standard, xc, mag = launch_dir.split('/')[-5:]
            xc_calcs = launch_dirs[launch_dir]
            if use_static:
                xc_calcs = [c for c in xc_calcs if c.split('-')[-1] == 'static']
            for xc_calc in xc_calcs:
                print('     working on %s' % xc_calc)
                calc_data = {'info' : {},
                             'summary' : {},
                             'flags' : []}
                if magnetization:
                    calc_data['magnetization'] = {}
                if relaxed_structure:
                    calc_data['structure'] = {}
                if dos:
                    calc_data['dos'] = {}
                calc_dir = os.path.join(launch_dir, xc_calc)
                xc, calc = xc_calc.split('-')
                
                calc_data['info']['calc_dir'] = calc_dir
                calc_data['info']['mag'] = mag
                calc_data['info']['standard'] = standard
                calc_data['info'][top_level_key] = top
                calc_data['info']['ID'] = ID
                calc_data['info']['xc'] = xc
                calc_data['info']['calc'] = calc

                va = VASPAnalysis(calc_dir)
                convergence = va.is_converged
                E_per_at = va.E_per_at
                if convergence:
                    if (calc == 'static') and check_relax:
                        relax_calc_dir = calc_dir.replace('static', 'relax')
                        va_relax = VASPAnalysis(calc_dir)
                        convergence_relax = va_relax.is_converged
                        if not convergence_relax:
                            convergence = False
                            E_per_at = None
                        E_relax = va_relax.E_per_at
                        if E_per_at and E_relax:
                            E_diff = abs(E_per_at - E_relax)
                            if E_diff > check_relax:
                                data['flags'].append('large E diff b/t relax and static')
                
                calc_data['summary']['E'] = E_per_at
                calc_data['summary']['convergence'] = convergence
                if not convergence:
                    calc_data['flags'].append('not converged')
                    
                if relaxed_structure:
                    if convergence:
                        structure = va.contcar.as_dict()
                    else:
                        structure = None
                    calc_data['structure'] = structure
                
                if magnetization:
                    if convergence:
                        calc_data['magnetization'] = va.magnetization

                if dos:
                    if dos == 'tdos':
                        calc_data['dos'] = va.tdos()
                    elif dos == 'pdos':
                        calc_data['dos'] = va.pdos()
                    else:
                        raise NotImplementedError('only tdos and pdos are accepted args for dos')
                
                data.append(calc_data)

        return {'data' : data}

def main():
    
    from MPQuery import MPQuery
    from MagTools import MagTools
    mpq = MPQuery('***REMOVED***')
    mpid, cmpd = 'mp-22584', 'LiMn2O4'
    #mpid = 'mp-1301329' # LiMnTiO4
    #mpid = 'mp-770495' # Li5Ti2Mn3Fe3O16
    #mpid = 'mp-772660' # NbCrO4
    #mpid = 'mp-776873' # Cr2O3
    s = mpq.get_structure_by_material_id(mpid)
    #s.make_supercell([3,3,3])
    
    return sub
    
if __name__ == '__main__':
    sub = main()

from pydmc.VASPTools import VASPSetUp, VASPAnalysis
from pydmc.handy import read_yaml, dotdict
from pymatgen.core.structure import Structure

import os
from shutil import copyfile, rmtree


HERE = os.path.dirname(os.path.abspath(__file__))

class SubmitTools(object):
    
    def __init__(self,
                 launch_dir,
                 fyaml=os.path.join(os.getcwd(), 'base_configs.yaml'),
                 user_configs={},
                 magmom=None,
                 files_to_inherit=['WAVECAR', 'CONTCAR'],
                 fyaml_partitions=os.path.join(HERE, 'partitions.yaml')):
        
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
                    'walltime' : '96:00:00',
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
        
        if not os.path.exists(fyaml):
            pydmc_yaml = os.path.join(HERE, 'base_configs.yaml')
            copyfile(pydmc_yaml, fyaml)
        
        base_configs = read_yaml(fyaml)
        base_configs['SLURM']['time'] = int(base_configs['SLURM']['time'] / 60)
        
        slurm_options = ['nodes', 'time', 'job-name', 'partition',
                         'account', 'error', 'output', 'mem',
                         'time', 'ntasks', 'qos', 'constraint']
        
        for slurm_option in slurm_options:
            if slurm_option in user_configs:
                base_configs['SLURM'][slurm_option] = user_configs[slurm_option]
                del user_configs[slurm_option]

        configs = {**base_configs, **user_configs}
        self.configs = dotdict(configs)
        
        fpos = os.path.join(launch_dir, 'POSCAR')
        if not os.path.exists(fpos):
            raise FileNotFoundError('Need a POSCAR to initialize setup; POSCAR not found in {}'.format(self.launch_dir))
        else:
            self.structure = Structure.from_file(fpos)
    
        self.files_to_inherit = files_to_inherit
        self.magmom = magmom
        
        self.partitions = dotdict(read_yaml(fyaml_partitions))
           

    @property
    def slurm_manager(self):
        """
        Returns slurm manager (eg #SBATCH)
        """
        return self.configs.manager
        
    @property
    def slurm_options(self):
        """
        Returns dictionary of slurm options
            - nodes, ntasks, walltime, etc
        """
        options = self.configs.SLURM
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
        configs = self.configs
        vasp_exec = os.path.join(configs.vasp_dir, configs.vasp)
        return '\n%s --ntasks=%s --mpi=pmi2 %s > %s' % (configs.mpi_command, str(self.slurm_options['ntasks']), vasp_exec, configs.fvaspout)
    
    @property
    def calcs(self):
        """
        Returns list of calcs to run 
            - (eg ['loose', 'relax', 'static'])
        """
        configs = self.configs
        calc = configs.calc
        calc_sequence = configs.calc_sequence
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
        configs = self.configs
        xc = configs.xc
        xc_sequence = configs.xc_sequence
        if xc_sequence and (xc != 'gga'):
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
        configs = self.configs
        fresh_restart = configs.fresh_restart
        calcs = self.calcs
        xcs = self.xcs
        fpos_src = os.path.join(self.launch_dir, 'POSCAR')
        tags = []
        for xc in xcs:
            for calc in calcs:
                if (xc == 'metagga') and (calc == 'loose') and configs.xc_sequence:
                    print('not running loose metagga')
                    continue
                # (1) create calc_dir
                tag = '%s-%s' % (xc, calc)
                calc_dir = os.path.join(self.launch_dir, tag)

                if os.path.exists(calc_dir) and fresh_restart:
                    rmtree(calc_dir)
                if not os.path.exists(calc_dir):
                    os.mkdir(calc_dir)
                
                # (2) check convergence
                convergence = VASPAnalysis(calc_dir).is_converged
                if convergence and not fresh_restart:
                    print('%s is already converged; skipping' % tag)
                    status = 'DONE'
                    tags.append('%s_%s' % (status, tag))
                    continue
                
                # (3) check for POSCAR       
                fpos_dst = os.path.join(calc_dir, 'POSCAR')
                if not os.path.exists(fpos_dst):
                    copyfile(fpos_src, fpos_dst)
                
                # (4) check for CONTCAR
                fcont_dst = os.path.join(calc_dir, 'CONTCAR')
                if os.path.exists(fcont_dst):
                    contents = open(fcont_dst, 'r').readlines()
                    if (len(contents) > 0) and not fresh_restart:
                        status = 'CONTINUE'
                    else:
                        status = 'NEWRUN'
                else:
                    status = 'NEWRUN'

                # (5) initialize VASPSetUp with configs
                vsu = VASPSetUp(calc_dir=calc_dir, 
                                magmom=self.magmom,
                                fvaspout=self.configs.fvaspout,
                                fvasperrors=self.configs.fvasperrors) 
                
                calc_configs = {'modify_%s' % input_file.lower() : 
                    configs['%s_%s' % (calc, input_file)] for input_file in ['INCAR', 'KPOINTS', 'POTCAR']}
                for key in prepare_calc_options:
                    calc_configs[key] = configs[key]
                
                # (6) check for errors in continuing jobs
                if status in ['CONTINUE', 'NEWRUN']:
                    calc_is_clean = vsu.is_clean
                    if not calc_is_clean:
                        incar_changes = vsu.incar_changes_from_errors
                        calc_configs['modify_incar'] = {**calc_configs['modify_incar'], **incar_changes}
                      
                # (7) prepare calc_dir to launch  
                vsu.prepare_calc(calc=calc,
                                xc=xc,
                                **calc_configs)
                
                print('prepared %s' % calc_dir)
                tags.append('%s_%s' % (status, tag))
        return tags
   
    @property
    def write_sub(self):
        """
        A lot going on here. The objective is to write a submission script for each calculation
            - so that I can iterate through a for loop and launch all calcs
            - each submission script will launch a chain of jobs
        
        """
        configs = self.configs
        fsub = os.path.join(self.launch_dir, configs.fsub)
        fstatus = os.path.join(self.launch_dir, configs.fstatus)
        vasp_command = self.vasp_command
        options = self.slurm_options
        manager = self.slurm_manager
        xcs, calcs = self.xcs, self.calcs
        files_to_inherit = self.files_to_inherit
        with open(fsub, 'w') as f:
            f.write('#!/bin/bash -l\n')
            for key in options:
                option = options[key]
                if option:
                    f.write('%s --%s=%s\n' % (manager, key, str(option)))
            f.write('\n\n')
            f.write('ulimit -s unlimited\n')
            tags = self.prepare_directories
            for tag in tags:
                status = tag.split('_')[0]
                xc, calc = tag.split('_')[1].split('-')
                calc_dir = os.path.join(self.launch_dir, '-'.join([xc, calc]))
                if status == 'DONE':
                    f.write('\necho working on %s >> %s\n' % (tag, fstatus))
                    f.write('echo %s is done >> %s\n' % (tag.split('_')[1], fstatus))
                elif status == 'CONTINUE':
                    f.write('\necho working on %s >> %s\n' % (tag, fstatus))
                    f.write('cp %s %s\n' % (os.path.join(calc_dir, 'CONTCAR'), os.path.join(calc_dir, 'POSCAR')))
                elif status == 'NEWRUN':
                    f.write('\necho working on %s >> %s\n' % (tag, fstatus))
                    
                    if xcs.index(xc) != 0:
                        if calcs.index(calcs) == 0:
                            xc_prev = xcs[xcs.index(xc)-1]
                            calc_prev = calcs[-1]
                            pass_info = True
                        elif calcs.index(calc) != 0:
                            xc_prev = xc
                            calc_prev = calcs[calcs.index(calc)-1]
                            pass_info = True
                    elif xcs.index(xc) == 0:
                        if calcs.index(calc) == 0:
                            pass_info = False
                        elif calcs.index(calc) != 0:
                            xc_prev = xc
                            calc_prev = calcs[calcs.index(calc)-1]
                            pass_info = True
                    
                    if pass_info:
                        src_dir = os.path.join(self.launch_dir, '-'.join([xc_prev, calc_prev]))
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
                            f.write('cp %s %s\n' % (fsrc, fdst))
                    
                    f.write('cd %s\n' % calc_dir)
                    f.write('%s\n' % vasp_command)
                    f.write('\necho launched %s-%s >> %s\n' % (xc, calc, fstatus))
                
def main():
    return
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
    magtools = MagTools(s)
    
    #return st, st, st
    
    s_nm, s_fm = magtools.get_nonmagnetic_structure, magtools.get_ferromagnetic_structure
    
    afm_strucs = magtools.get_antiferromagnetic_structures
    
    s_afm = afm_strucs[0]
    afm_magmom = s_afm.site_properties['magmom']

    launch_dirs = [os.path.join('..', 'dev', cmpd, mpid, mag) for mag in ['nm', 'fm', 'afm']]
    strucs = [s_nm, s_fm, s_afm]
    strucs = dict(zip(launch_dirs, strucs))
    
    for mag in ['nm', 'fm', 'afm']:
        l = os.path.join('..', 'dev', cmpd, mpid, mag)
        for l in launch_dirs:
            if not os.path.exists(l):
                os.makedirs(l)
            s = strucs[l]
            s.to(fmt='poscar', filename=os.path.join(l, 'POSCAR'))
            if mag == 'afm':
                magmom = afm_magmom
            else:
                magmom = None
            sub = SubmitTools(launch_dir=l,
                            magmom=magmom,
                            user_configs={'fresh_restart': True,
                                          'mag' : mag})
            sub.write_sub
            print(sub.magmom)
    return sub
    
if __name__ == '__main__':
    sub = main()
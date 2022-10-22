import os
from shutil import copyfile
from VASPTools import VASPSetUp, VASPAnalysis
from handy import read_yaml, write_yaml, dotdict
from pymatgen.core.structure import Structure

class SubmitTools(object):
    
    def __init__(self,
                 launch_dir,
                 config_yaml=os.path.join(os.getcwd(), 'base_configs.yaml'),
                 user_configs={},
                 magmom=None,
                 files_to_inherit=['WAVECAR', 'CONTCAR']):
        
        """
        Args:
            launch_dir (str) - directory to launch calculations from
                - assumes initial structure is POSCAR in launch_dir
            config_yaml (str) - path to yaml file containing base configs
            user_configs (dict) - dictionary of user configs to override base configs
                common ones:
                    {'mag' : 'afm',
                    'standard' : 'mp',
                    'nodes' : 2,
                    'walltime' : '96:00:00'}
                    
            magmom (list) - list of magnetic moments for each atom in structure (only specify if mag='afm')
            files_to_inherit (list) - list of files to copy from calc to calc
        """
        
        self.launch_dir = launch_dir
        copyfile(os.path.join(config_yaml), os.path.join(self.launch_dir, 'base_configs.yaml'))
        fpos = os.path.join(launch_dir, 'POSCAR')
        if not os.path.exists(fpos):
            raise FileNotFoundError('Need a POSCAR to initialize setup; POSCAR not found in {}'.format(self.launch_dir))
        else:
            self.structure = Structure.from_file(fpos)
        
        base_configs = read_yaml(os.path.join(launch_dir, 'base_configs.yaml'))
        configs = {**base_configs, **user_configs}
        self.configs = dotdict(configs)
        
        self.files_to_inherit = files_to_inherit
        
        self.magmom = magmom
        
        print(self.configs)

    @property
    def slurm_manager(self):
        return self.configs.manager
    
    @property
    def partitions(self):
        
        partitions = {}
        partitions['agate'] = {}
        partitions['agate']['agsmall'] = {'cores_per_node' : 128,
                                            'sharing' : True,
                                            'max_walltime' : 96,
                                            'mem_per_core' : 4, # GB
                                            'max_nodes' : 1}
        
        partitions['agate']['aglarge']  = {'cores_per_node' : 128,
                                            'sharing' : False,
                                            'max_walltime' : 24,
                                            'mem_per_core' : 4, # GB
                                            'max_nodes' : 32}
        
        partitions['agate']['a100-4']  = {'cores_per_node' : 64,
                                            'sharing' : True,
                                            'max_walltime' : 24,
                                            'mem_per_core' : 4, # GB
                                            'max_nodes' : 4}
        
        partitions['agate']['a100-8']  = {'cores_per_node' : 128,
                                            'sharing' : True,
                                            'max_walltime' : 24,
                                            'mem_per_core' : 7.5, # GB
                                            'max_nodes' : 4}
        
        partitions['mesabi'] = {}
        partitions['mesabi']['amdsmall'] = {'cores_per_node' : 128,
                                            'sharing' : True,
                                            'max_walltime' : 24,
                                            'mem_per_core' : 7.5, # GB
                                            'max_nodes' : 4}
        
        # @CHRIS - keep working on this
        
    @property
    def slurm_options(self):
        return self.configs.SLURM
        
    @property
    def vasp_command(self):
        configs = self.configs
        vasp_exec = os.path.join(configs.vasp_dir, configs.vasp)
        return '\n%s -n %s %s > %s' % (configs.mpi_command, str(self.slurm_options['ntasks']), vasp_exec, configs.fvaspout)
    
    @property
    def calcs(self):
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
                
                tag = '%s-%s' % (xc, calc)
                calc_dir = os.path.join(self.launch_dir, tag)
                if not os.path.exists(calc_dir):
                    os.mkdir(calc_dir)
                    
                convergence = VASPAnalysis(calc_dir).is_converged(calc)
                if convergence and not fresh_restart:
                    print('%s is already converged; skipping' % tag)
                    status = 'DONE'
                    tags.append('%s_%s' % (status, tag))
                fpos_dst = os.path.join(calc_dir, 'POSCAR')
                if not os.path.exists(fpos_dst):
                    copyfile(fpos_src, fpos_dst)
                    
                vsu = VASPSetUp(calc_dir=calc_dir, 
                                magmom=self.magmom) 
                   
                calc_configs = {'modify_%s' % input_file.lower() : 
                    configs['%s_%s' % (calc, input_file)] for input_file in ['INCAR', 'KPOINTS', 'POTCAR']}
                for key in prepare_calc_options:
                    calc_configs[key] = configs[key]
                
                fcont_dst = os.path.join(calc_dir, 'CONTCAR')
                if os.path.exists(fcont_dst):
                    contents = open(fcont_dst, 'r').readlines()
                    if (len(contents) > 0) and not fresh_restart:
                        status = 'CONTINUE'
                    else:
                        status = 'NEWRUN'
                else:
                    status = 'NEWRUN'

                vsu.prepare_calc(calc=calc,
                                 xc=xc,
                                 **calc_configs)
                
                print('prepared %s' % calc_dir)
                tags.append('%s_%s' % (status, tag))
        return tags
   
    @property
    def write_sub(self):
        configs = self.configs
        fsub = os.path.join(self.launch_dir, configs.fsub)
        fstatus = os.path.join(self.launch_dir, configs.fstatus)
        vasp_command = self.vasp_command
        options = self.slurm_options
        manager = self.slurm_manager
        xcs, calcs = self.xcs, self.calcs
        files_to_inherit = self.files_to_inherit
        with open(fsub, 'w') as f:
            f.write('#!/bin/bash')
            for key in options:
                option = options[key]
                if option:
                    f.write('%s --%s=%s\n' % (manager, key, str(option)))
            f.write('\n\n')
            tags = self.prepare_directories
            lines_to_write_to_sub = []
            for tag in tags:
                status = tag.split('_')[0]
                xc, calc = tag.split('_')[1].split('-')
                calc_dir = os.path.join(self.launch_dir, '-'.join([xc, calc]))
                if status == 'DONE':
                    f.write('\nWorking on %s >> %s\n' % (tag, fstatus))
                    f.write('echo %s is done %s >> %s\n' % (tag.split('_')[1], fstatus))
                elif status == 'CONTINUE':
                    f.write('\nWorking on %s >> %s\n' % (tag, fstatus))
                    f.write('cp %s %s\n' % (os.path.join(calc_dir, 'CONTCAR'), os.path.join(calc_dir, 'POSCAR')))
                elif status == 'NEWRUN':
                    f.write('\nWorking on %s >> %s\n' % (tag, fstatus))
                    if xcs.index(xc) == 0:
                        xc_prev = None
                    else:
                        xc_prev = xcs[xcs.index(xc) - 1]
                    if calcs.index(calc) == 0:
                        calc_prev = None if not xc_prev else calcs[-1]
                    else:
                        calc_prev = calcs[calcs.index(calc) - 1]
                    if calc_prev:
                        if xc_prev:
                            src_dir = os.path.join(self.launch_dir, '-'.join([xc_prev, calc_prev]))
                        else:
                            src_dir = os.path.join(self.launch_dir, '-'.join([xc, calc_prev]))
                    else:
                        src_dir = None
                    if src_dir:
                        for file_to_inherit in files_to_inherit:
                            f.write('cp %s %s\n' % (os.path.join(src_dir, file_to_inherit), os.path.join(calc_dir, file_to_inherit)))
                    f.write('cd %s\n' % calc_dir)
                    f.write('%s\n' % vasp_command)
                    f.write('\necho launched %s-%s >> %s\n' % (xc, calc, fstatus))
                
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
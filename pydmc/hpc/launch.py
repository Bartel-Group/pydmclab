import os

from pymatgen.core.structure import Structure

from pydmc.utils.handy import read_yaml, write_yaml, dotdict, is_calc_valid
from pydmc.core.mag import MagTools
from pydmc.data.configs import load_launch_configs

HERE = os.path.dirname(os.path.abspath(__file__))

class LaunchTools(object):
    """
    This is a class to figure out:
        - what launch_dirs need to be created
            - i.e., which directories will house submission scripts
        - what calculations need to be run in each launch_dir
        - the general flow will be:
            - use LaunchTools to generate launch_dirs (and their corresponding "valid_calcs")
            - use SubmitTools to generate submission scripts for each launch_dir
                - note: SubmitTools is making heavy use of VASPSetUp to configure directories and set up each VASP calc
    """
    
    def __init__(self,
                 calcs_dir,
                 structure,
                 top_level,
                 unique_ID,
                 magmoms=None,
                 user_configs={},
                 launch_configs_yaml=os.path.join(os.getcwd(), '_launch_configs.yaml')):
        """
        Args:
        
            calcs_dir (os.path): directory where calculations will be stored
                - usually if I'm writing a "launch" script to configure and run a bunch of calcs from  a directory: os.getcwd() = */scripts:
                    - then calcs_dir will be os.getcwd().replace('scripts', 'calcs')
                    - I should also probably have a directory to store data called calcs_dir.replace('calcs', 'data')
                    - these are best practices but not strictly enforced in the code anywhere
                                    
            structure (Structure): pymatgen structure object
                - usually I want to run a series of calculations for some input structure
                    - this is the input structure
                    
            top_level (str): top level directory
                - could be whatever you want, but there are some guidelines
                    - usually this will be a chemical formula
                        - if I was just running a geometry relaxation on a given chemical formula 
                            - let's say LiTiS2)
                                - I would call the top_level = LiTiS2 or even better, Li1S2Ti1 (the CompTools(formula).clean standard)
                - for more complicated calcs,
                    - lets say I'm studying Li_{x}Co10O20 at varying x values between 0 and 10
                        - I might make top_level = LiCoO2
                
            unique_ID (str): level below top_level
                - could be a material ID in materials project (for standard geometry relaxations, this makes sense)
                - could be x in the LiCoO2 example I described previously
                - it's really up to you, but it must be unique within the top_level directory
                
            magmoms (dict): 
                - if you are running AFM calculations
                    - {index of configuration index (int) : magmom (list)} generated using MagTools
                    - best practice is to save this as a json in data_dir so you only call MagTools once
                - if you are not running AFM calculations
                    - can be None or {} 
                    
            user_configs (dict):
                - any setting you want to pass that's not default in pydmc/data/data/_launch_configs.yaml
                - some common ones to change might be:
                    {xcs : ['ggau'],
                     compare_to_mp : True,
                     n_afm_configs : 3,
                     etc.}
                     
            launch_configs_yaml (os.pathLike) - path to yaml file containing launch configs
                - there's usually no reason to change this
                - this holds some default configs for LaunchTools
                - can always be changed with user_configs  
                
        Returns:
            configs (dotdict): dictionary of all configs and arguments to LaunchTools
        """
        
        if not os.path.exists(calcs_dir):
            os.mkdir(calcs_dir)
            
        if not os.path.exists(launch_configs_yaml):
            _launch_configs = load_launch_configs()
            write_yaml(_launch_configs, launch_configs_yaml)
        
        _launch_configs = read_yaml(launch_configs_yaml)
        
        configs = {**_launch_configs, **user_configs}
            
        if not isinstance(structure, dict):
            structure = structure.as_dict()

        if configs['n_afm_configs'] > 0:
            if MagTools(structure).could_be_afm:
                if not magmoms:
                    raise ValueError('You are running afm calculations but provided no magmoms, generate these first, then pass to LaunchTools')
    
        standards = configs['standards'].copy()
        if configs['compare_to_mp']:
            if 'mp' not in standards:
                standards.append('mp')

        configs['standards'] = standards
        
        xcs_to_get_energies_for = configs['final_xcs'].copy()
        standard_to_xcs = {standard : 
                            {final_xc : [final_xc] for final_xc in xcs_to_get_energies_for} 
                                for standard in configs['standards']}
        
        if 'metagga' in xcs_to_get_energies_for:
            for standard in standard_to_xcs:
                standard_to_xcs[standard]['metagga'] = ['gga', 'metagga']
        
        if configs['compare_to_mp']:
            if 'mp' not in standard_to_xcs:
                standard_to_xcs['mp'] = {'ggau' : ['ggau']}
            
            else:
                standard_to_xcs['mp']['ggau'] = ['ggau']
                
        configs['standard_to_xcs'] = standard_to_xcs
        
        write_yaml(configs, launch_configs_yaml)

        configs['top_level'] = top_level
        configs['unique_ID'] = unique_ID
        configs['calcs_dir'] = calcs_dir
        
        self.magmoms = magmoms
        self.structure = structure

        self.configs = dotdict(configs)

    
    @property
    def valid_calcs(self):
        """
        Returns list of calculations that are "valid" given the configs
            - note: makes use of pydmc.utils.handy.is_calc_valid to check a given calc
        
        Calcs might be "invalid" if:
            - you are passing a structure from one functional to another, then there's no need to run a loose calc
            - you have a nonmagnetic material, there's no need to run magnetic calcs
            - you are using mp standards, there's no reason not to use MP functional (GGA+U)
            - you are passing mag = afm_9, but there are only 5 AFM configs
            - etc.
        """
        configs = self.configs
        if configs.relax_geometry:
            possible_calcs = ['loose', 'relax', 'static']
        else:
            possible_calcs = ['static']
        
        possible_mags = ['nm', 'fm']
        if configs.n_afm_configs > 0:
            possible_mags += ['afm_%s' % str(i) for i in range(configs.n_afm_configs)]

        standard_to_xcs = configs.standard_to_xcs
        out = []
        for standard in standard_to_xcs:
            for final_xc in standard_to_xcs[standard]:
                for xc_to_run in standard_to_xcs[standard][final_xc]:
                    for mag in possible_mags:
                        if 'afm' in mag:
                            magmoms = self.magmoms
                            idx = mag.split('_')[-1]
                            if (idx in magmoms):
                                magmom = magmoms[idx]
                            elif str(idx) in magmoms:
                                magmom = magmoms[str(idx)]
                            else:
                                magmom = None
                        else:
                            magmom = None
                        for calc in possible_calcs:
                            validity = is_calc_valid(self.structure,
                                                    standard,
                                                    xc_to_run,
                                                    calc,
                                                    mag,
                                                    magmom,
                                                    configs.mag_override)
                            if validity:
                                out.append({'standard' : standard,
                                            'final_xc' : final_xc,
                                            'xc_to_run' : xc_to_run,
                                            'mag' : mag,
                                            'calc' : calc})
                        
        return out

    @property
    def launch_dirs(self):
        """
        Returns the minimal list of directories that need a submission file to launch a chain of calcs
        
        These launch_dirs have a very prescribed structure:
            calcs_dir / top_level / unique_ID / standard / xc / mag / calc
            
            e.g.,
                this could be */calcs/Nd2O7Ru2/mp-19930/dmc/metagga/fm/relax
        """
        configs = self.configs
            
        valid_calcs = self.valid_calcs
        
        launch_dirs = []
    
        level0 = configs.calcs_dir
        level1 = configs.top_level
        level2 = configs.unique_ID
        
        unique_standards = sorted(list(set([d['standard'] for d in valid_calcs])))
        
        for standard in unique_standards:
            standard_dir = os.path.join(level0, level1, level2, standard)
            final_xcs = [d['final_xc'] for d in valid_calcs if d['standard'] == standard]
            for final_xc in final_xcs:
                final_xc_dir = os.path.join(standard_dir, final_xc)
                unique_mags = [d['mag'] for d in valid_calcs if d['standard'] == standard and d['final_xc'] == final_xc]
                for mag in unique_mags:
                    mag_dir = os.path.join(final_xc_dir, mag)
                    launch_dirs.append(mag_dir)
        
        return launch_dirs
    
    @property
    def launch_dirs_to_tags(self):
        """
        Returns:
            dictionary of {launch_dir : [tags]}
            
            for the minimal list of self.launch_dirs
            
            each "tag" corresponds with one instsance where vasp needs to be executed
                - tags have the form xc_to_run-calc
                    - e.g., 'gga-relax'
        
        """
        launch_dirs = self.launch_dirs
        valid_calcs = self.valid_calcs
        
        d = {}
        for launch_dir in launch_dirs:
            d[launch_dir] = []
            standard, final_xc, mag = launch_dir.split('/')[-3:]
            for calc in valid_calcs:
                tag = '-'.join([calc['xc_to_run'], calc['calc']])
                calc_standard, calc_final_xc, calc_mag = calc['standard'], calc['final_xc'], calc['mag']
                if (calc_standard == standard) and (calc_final_xc == final_xc) and (calc_mag == mag):
                    d[launch_dir].append(tag)
                            
        return d
                
    @property
    def create_launch_dirs_and_make_POSCARs(self):
        """
        Loops through my launch_dirs and puts a POSCAR in each one
        
        SubmitTools will take it from there to
            - create the other VASP inputs
            - make and submit a submission file
            - handle errors, etc
        """
        
        launch_dirs = self.launch_dirs_to_tags
        for launch_dir in launch_dirs:
            if not os.path.exists(launch_dir):
                os.makedirs(launch_dir)
        
            fposcar = os.path.join(launch_dir, 'POSCAR')
            if not os.path.exists(fposcar):
                structure = Structure.from_dict(self.structure)
                structure.to(fmt='poscar', filename=fposcar)
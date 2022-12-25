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
                 to_launch,
                 magmoms=None,
                 user_configs={},
                 refresh_configs=True,
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
                
            to_launch (dict) : 
                {standard (str) : [list of xcs (str)]}
                
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
            
        if not os.path.exists(launch_configs_yaml) or refresh_configs:
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
    
        if configs['compare_to_mp']:
            to_launch['mp'] = ['ggau']

        configs['top_level'] = top_level
        configs['unique_ID'] = unique_ID
        configs['calcs_dir'] = calcs_dir
        configs['to_launch'] = to_launch

        
        self.magmoms = magmoms
        self.structure = structure

        self.configs = dotdict(configs)

    @property
    def valid_mags(self):
        configs = self.configs
        if configs.override_mag:
            return configs.override_mag
        
        structure = self.structure
        if not MagTools(structure).could_be_magnetic:
            return ['nm']
        
        if not MagTools(structure).could_be_afm or not configs.n_afm_configs:
            return ['fm']
               
        max_desired_afm_idx = configs.n_afm_configs-1
        
        magmoms = self.magmoms 
               
        configs_in_magmoms = list(magmoms.keys())
        configs_in_magmoms = sorted([int(i) for i in configs_in_magmoms])
        max_available_afm_idx = max(configs_in_magmoms)
        
        max_afm_idx = min(max_desired_afm_idx, max_available_afm_idx)
        
        afm_indices = ['afm_%s' % str(i) for i in range(max_afm_idx+1)]
        
        return ['fm'] + afm_indices

    def launch_dirs(self,
                    make_dirs=True):
        """
        Returns the minimal list of directories that need a submission file to launch a chain of calcs
        
        These launch_dirs have a very prescribed structure:
            calcs_dir / top_level / unique_ID / standard / mag
            
            e.g.,
                this could be */calcs/Nd2O7Ru2/mp-19930/dmc/metagga/fm/relax
        """
        structure = self.structure
        configs = self.configs
        
        mags = self.valid_mags
        
        magmoms = self.magmoms
        
        to_launch = configs.to_launch
    
        level0 = configs.calcs_dir
        level1 = configs.top_level
        level2 = configs.unique_ID
        
        launch_dirs = {}
        for standard in to_launch:
            level3 = standard
            xcs = to_launch[standard]
            for mag in mags:
                magmom = None
                if 'afm' in mag:
                    idx = mag.split('_')[1]
                    if str(idx) in magmoms:
                        magmom = magmoms[str(idx)]
                    elif int(idx) in magmoms:
                        magmom = magmoms[int(idx)]
                        
                level4 = mag
                launch_dir = os.path.join(level0, level1, level2, level3, level4)
                launch_dirs[launch_dir] = {'xcs' : xcs,
                                           'magmom' : magmom}
                if make_dirs:
                    if not os.path.exists(launch_dir):
                        os.makedirs(launch_dir)
                    fposcar = os.path.join(launch_dir, 'POSCAR')
                    if not os.path.exists(fposcar):
                        structure = Structure.from_dict(structure)
                        structure.to(fmt='poscar', filename=fposcar)                    
        
        return launch_dirs
                
    @property
    def create_launch_dirs_and_make_POSCARs(self):
        """
        Loops through my launch_dirs and puts a POSCAR in each one
        
        SubmitTools will take it from there to
            - create the other VASP inputs
            - make and submit a submission file
            - handle errors, etc
        """
        
        launch_dirs = self.launch_dirs
        for launch_dir in launch_dirs:
            if not os.path.exists(launch_dir):
                os.makedirs(launch_dir)
        
            fposcar = os.path.join(launch_dir, 'POSCAR')
            if not os.path.exists(fposcar):
                structure = Structure.from_dict(self.structure)
                structure.to(fmt='poscar', filename=fposcar)

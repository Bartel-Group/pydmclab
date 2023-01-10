import os

from pymatgen.core.structure import Structure

from pydmc.utils.handy import read_yaml, write_yaml, is_calc_valid
from pydmc.core.mag import MagTools
from pydmc.data.configs import load_launch_configs
from pydmc.core.struc import StrucTools

HERE = os.path.dirname(os.path.abspath(__file__))

class LaunchTools(object):
    """
    This is a class to figure out:
        - what launch_dirs need to be created
            - i.e., which directories will house submission scripts
        - what calculation chains need to be run in each launch_dir
    The output is going to be:
        {launch_dir (str) : {'xcs' : [list of final xcs for each chain (str)],
                             'magmom' : [list of magmoms for the structure in that launch_dir (list)],}}
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
                - e.g., if I want to run gga+u and metagga with dmc standards, to_launch = {'dmc' : ['metagga', 'ggau']}
                
            magmoms (dict): 
                - if you are running AFM calculations
                    - {index of configuration index (int) : magmom (list)} generated using MagTools
                    - best practice is to save this as a json in data_dir so you only call MagTools once
                - if you are not running AFM calculations
                    - can be None or {} 
                    
            user_configs (dict):
                - any setting you want to pass that's not default in pydmc/data/data/_launch_configs.yaml
                - launch_configs:
                    compare_to_mp: False # if True, launch will get everything it needs to generate MP-consistent data
                    n_afm_configs: 0 # how many AFM configurations to run
                    override_mag: False # could be ['nm'] if you only want to run nonmagnetic, won't check for whether structure is mag or not mag, it will just do as you say


            refresh_configs (bool) - if True, will copy pydmc baseline configs to your local directory
                - this is useful if you've made changes to the configs files in the directory you're working in and want to start over
                                     
            launch_configs_yaml (os.pathLike) - path to yaml file containing launch configs
                - there's usually no reason to change this
                - this holds some default configs for LaunchTools
                - can always be changed with user_configs  
                
        Returns:
            configs (dotdict): dictionary of all configs and arguments to LaunchTools
        """
        
        # make our calcs_dir if it doesn't exist (this will hold all the launch_dirs)
        if not os.path.exists(calcs_dir):
            os.mkdir(calcs_dir)
        
        # make our local launch_configs file if it doesn't exist    
        if not os.path.exists(launch_configs_yaml) or refresh_configs:
            _launch_configs = load_launch_configs()
            write_yaml(_launch_configs, launch_configs_yaml)
        
        # initialize our baseline launch_configs
        _launch_configs = read_yaml(launch_configs_yaml)
        
        # update our baseline launch_configs with user_configs
        configs = {**_launch_configs, **user_configs}
        
        # make structure a dict() for easier handling
        if not isinstance(structure, dict):
            structure = structure.as_dict()

        # check to make sure we have magmoms if we're running AFM calcs
        if configs['n_afm_configs'] > 0:
            if MagTools(structure).could_be_afm:
                if not magmoms:
                    raise ValueError('You are running afm calculations but provided no magmoms, generate these first, then pass to LaunchTools')

        # include gga+u calcs w/ mp standards if we're comparing to MP
        if configs['compare_to_mp']:
            to_launch['mp'] = ['ggau']

        # add the required arguments to our configs file
        configs['top_level'] = top_level
        configs['unique_ID'] = unique_ID
        configs['calcs_dir'] = calcs_dir
        configs['to_launch'] = to_launch

        # store our magmoms and structure
        self.magmoms = magmoms
        self.structure = structure

        # make a copy of our configs to prevent unwanted changes
        self.configs = configs.copy()

    @property
    def valid_mags(self):
        """
        Returns:
            list of magnetic configuration names that make sense to run based on the inputs
            
        e.g., 
            - if we have a nonmagnetic system, this might be ['nm']
            - if we set n_afm_configs = 100, but our magmoms only has 3 configs, then this will just hold ['fm', 'afm_0', 'afm_1', 'afm_2']
            
        these are the set of "mags" that can be run given our inputs
        
        Note:
            - configs['override_mag'] will force that we use configs['override_mag'] as our mag        
        
        """
        # copy our configs
        configs = self.configs.copy()
        
        # return override_mag if we set it
        if configs['override_mag']:
            return configs['override_mag']
        
        structure = self.structure
        
        # if we're not magnetic, return nm
        if not MagTools(structure).could_be_magnetic:
            return ['nm']
        
        # if we can't be AFM or we didn't ask for AFM, but we are magnetic, return fm
        if not MagTools(structure).could_be_afm or not configs['n_afm_configs']:
            return ['fm']
        
        # figure out the max AFM index we can run based on what we asked for 
        
        max_desired_afm_idx = configs['n_afm_configs']-1
        
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
        Args:
            make_dirs (bool) - if True, make the launch_dir and populate it with a POSCAR
        Returns:
            a dictionary of:
                {launch_dir (str) : {'xcs': [list of final_xcs to submit],
                                     'magmom' : [list of magmoms for the structure in launch_dir]}}
        
        Returns the minimal list of directories that will house submission files (each of which launch a chain of calcs)
            - note a chain of calcs must have the same structure and magnetic information, otherwise, there's no reason to chain them
        
        These launch_dirs have a very prescribed structure:
            calcs_dir / top_level / unique_ID / standard / mag
            
            e.g.,
                - ../calcs/Nd2O7Ru2/mp-19930/dmc/fm
                - ../calcs/2/3/dmc/afm_4
                    - (if (2) was a unique compositional indicator and (3) was a unique structural indicator) 
        """
        structure = self.structure
        
        # make a copy of our configs to prevent unwanted changes
        configs = self.configs.copy()
        
        # the list of mags we can run
        mags = self.valid_mags
        
        magmoms = self.magmoms
        
        # final_xcs we want to run for each standard
        to_launch = configs['to_launch']
    
        # level0 houses all our launch_dirs
        level0 = configs['calcs_dir']
        
        # level1 describes the composition
        level1 = configs['top_level']
        
        # level2 describes the structure
        level2 = configs['unique_ID']
        
        launch_dirs = {}
        for standard in to_launch:
            # for each standard we asked for, use that as level3
            level3 = standard
            
            # we asked for certain xcs at each standard, hold them here
            xcs = to_launch[standard]
            
            for mag in mags:
                # for each mag we can run, use that as level4
                level4 = mag
                magmom = None
                if 'afm' in mag:
                    # grab the magmom if our calc is AFM
                    idx = mag.split('_')[1]
                    if str(idx) in magmoms:
                        magmom = magmoms[str(idx)]
                    elif int(idx) in magmoms:
                        magmom = magmoms[int(idx)]
                        
                # our launch_dir is now defined
                launch_dir = os.path.join(level0, level1, level2, level3, level4)
                
                # save the final_xcs we want to submit in this launch_dir
                    # SubmitTools will make 1 submission script for each final_xc
                # save the magmom as well. VASPSetUp will need that to set the INCARs for all calcs in this launch_dir
                launch_dirs[launch_dir] = {'xcs' : xcs,
                                           'magmom' : magmom}
                
                # if make_dirs, make the launch_dir and put a POSCAR in there
                if make_dirs:
                    if not os.path.exists(launch_dir):
                        os.makedirs(launch_dir)
                    fposcar = os.path.join(launch_dir, 'POSCAR')
                    if not os.path.exists(fposcar):
                        struc = Structure.from_dict(structure)
                        if configs['perturb_launch_poscar']:
                            initial_structure = struc.copy()
                            if isinstance(configs['perturb_launch_poscar'], bool):
                                perturbation = 0.05
                            else:
                                perturbation = configs['perturb_launch_poscar']
                            perturbed_structure = StrucTools(initial_structure).perturb(perturbation)
                            perturbed_structure.to(fmt='poscar', filename=fposcar)
                        else:
                            struc.to(fmt='poscar', filename=fposcar)                    
        
        return launch_dirs
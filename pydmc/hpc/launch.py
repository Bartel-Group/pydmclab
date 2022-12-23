import os
from shutil import copyfile

from pymatgen.core.structure import Structure

from pydmc.utils.handy import read_yaml, write_yaml, dotdict, is_calc_valid
from pydmc.core.mag import MagTools
from pydmc.data.configs import launch_configs




HERE = os.path.dirname(os.path.abspath(__file__))

class LaunchTools(object):
    
    def __init__(self,
                 calcs_dir,
                 structure,
                 magmoms=None,
                 top_level='formula',
                 unique_ID='my-1234',
                 launch_configs_yaml=os.path.join(os.getcwd(), '_launch_configs.yaml'),
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
            
        if not os.path.exists(launch_configs_yaml):
            _launch_configs = launch_configs()
            write_yaml(_launch_configs, launch_configs_yaml)
        
        _launch_configs = read_yaml(launch_configs_yaml)
        
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
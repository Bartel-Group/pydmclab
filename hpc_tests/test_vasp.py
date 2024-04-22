import unittest

from pydmclab.hpc.vasp import VASPSetUp
from pydmclab.core.struc import StrucTools
from pydmclab.utils.handy import read_json, write_json

from pydmclab.hpc.analyze import AnalyzeVASP

import os


CALC_DIR = '/home/cbartel/cbartel/projects/for_pydmclab_unit_tests/240422/default_launcher/calcs/Cl1Li1/mp-22905/nm/gga-relax'
class UnitTestVASPSetUp(unittest.TestCase):
    def test_mods(self):
        calc = 'gga-relax'
        user_configs = {'xc_to_run' : 'gga', 'calc_to_run' : 'relax', 'mag' : 'nm'}
        vsu = VASPSetUp(calc_dir=CALC_DIR, user_configs=user_configs)
        
        custom_configs = {'incar_mods' : {calc : {'POTIM' : 0.1}},
                        'kpoints_mods' : {calc : {'grid' : [2,2,2]}},
                        'potcar_mods' : {calc : {'Li' : 'Li_sv'}}}
        user_configs.update(custom_configs)
        vsu = VASPSetUp(calc_dir=CALC_DIR, user_configs=user_configs)
        vsu.prepare_calc

        av = AnalyzeVASP(CALC_DIR)
        self.assertEqual(av.outputs.incar['POTIM'], 0.1)
        self.assertEqual(av.outputs.potcar['Li']['pp'], 'Li_sv')
        found_it = False
        with open(os.path.join(CALC_DIR, 'KPOINTS')) as f:
            for line in f:
                if '2 2 2' in line:
                    found_it = True
        self.assertEqual(found_it, True)
        
        del user_configs
        user_configs = {'xc_to_run' : 'gga', 'calc_to_run' : 'relax', 'mag' : 'nm'}
        vsu = VASPSetUp(calc_dir=CALC_DIR, user_configs=user_configs)
        vsu.prepare_calc
        if os.path.exists(os.path.join(CALC_DIR, 'KPOINTS')):
            os.remove(os.path.join(CALC_DIR, 'KPOINTS'))

        calc = 'all-all'
        custom_configs = {'incar_mods' : {calc : {'POTIM' : 0.1}},
                        'kpoints_mods' : {calc : {'grid' : [2,2,2]}},
                        'potcar_mods' : {calc : {'Li' : 'Li_sv'}}}
        user_configs.update(custom_configs)
        vsu = VASPSetUp(calc_dir=CALC_DIR, user_configs=user_configs)
        vsu.prepare_calc

        av = AnalyzeVASP(CALC_DIR)
        self.assertEqual(av.outputs.incar['POTIM'], 0.1)
        self.assertEqual(av.outputs.potcar['Li']['pp'], 'Li_sv')
        found_it = False
        with open(os.path.join(CALC_DIR, 'KPOINTS')) as f:
            for line in f:
                if '2 2 2' in line:
                    found_it = True
                    self.assertEqual(found_it, True)
                    
        del user_configs
        user_configs = {'xc_to_run' : 'gga', 'calc_to_run' : 'relax', 'mag' : 'nm'}
        vsu = VASPSetUp(calc_dir=CALC_DIR, user_configs=user_configs)
        vsu.prepare_calc
        if os.path.exists(os.path.join(CALC_DIR, 'KPOINTS')):
            os.remove(os.path.join(CALC_DIR, 'KPOINTS'))

        calc = 'gga-all'
        custom_configs = {'incar_mods' : {calc : {'POTIM' : 0.1}},
                        'kpoints_mods' : {calc : {'grid' : [2,2,2]}},
                        'potcar_mods' : {calc : {'Li' : 'Li_sv'}}}
        user_configs.update(custom_configs)
        vsu = VASPSetUp(calc_dir=CALC_DIR, user_configs=user_configs)
        vsu.prepare_calc

        av = AnalyzeVASP(CALC_DIR)
        self.assertEqual(av.outputs.incar['POTIM'], 0.1)
        self.assertEqual(av.outputs.potcar['Li']['pp'], 'Li_sv')
        found_it = False
        with open(os.path.join(CALC_DIR, 'KPOINTS')) as f:
            for line in f:
                if '2 2 2' in line:
                    found_it = True
                    self.assertEqual(found_it, True)

        del user_configs
        user_configs = {'xc_to_run' : 'gga', 'calc_to_run' : 'relax', 'mag' : 'nm'}
        vsu = VASPSetUp(calc_dir=CALC_DIR, user_configs=user_configs)
        vsu.prepare_calc
        if os.path.exists(os.path.join(CALC_DIR, 'KPOINTS')):
            os.remove(os.path.join(CALC_DIR, 'KPOINTS'))

        calc = 'all-relax'
        custom_configs = {'incar_mods' : {calc : {'POTIM' : 0.1}},
                        'kpoints_mods' : {calc : {'grid' : [2,2,2]}},
                        'potcar_mods' : {calc : {'Li' : 'Li_sv'}}}
        user_configs.update(custom_configs)
        vsu = VASPSetUp(calc_dir=CALC_DIR, user_configs=user_configs)
        vsu.prepare_calc

        av = AnalyzeVASP(CALC_DIR)
        self.assertEqual(av.outputs.incar['POTIM'], 0.1)
        self.assertEqual(av.outputs.potcar['Li']['pp'], 'Li_sv')
        found_it = False
        with open(os.path.join(CALC_DIR, 'KPOINTS')) as f:
            for line in f:
                if '2 2 2' in line:
                    found_it = True
                    self.assertEqual(found_it, True)

        del user_configs
        user_configs = {'xc_to_run' : 'gga', 'calc_to_run' : 'relax', 'mag' : 'nm'}
        vsu = VASPSetUp(calc_dir=CALC_DIR, user_configs=user_configs)
        vsu.prepare_calc
        if os.path.exists(os.path.join(CALC_DIR, 'KPOINTS')):
            os.remove(os.path.join(CALC_DIR, 'KPOINTS'))
        incar = av.outputs.incar
        self.assertEqual(0, len([k for k in incar if k == 'POTIM']))

        user_configs = {'xc_to_run' : 'gga', 'calc_to_run' : 'relax', 'mag' : 'nm'}
        custom_configs = {'functional' : {'gga' : 'PS'}}
        user_configs.update(custom_configs)
        vsu = VASPSetUp(calc_dir=CALC_DIR, user_configs=user_configs)
        vsu.prepare_calc
        self.assertEqual(av.outputs.incar['GGA'], 'Ps')

class UnitTestSubmitCalcs(unittest.TestCase):
    def test_mods(self):
        return

if __name__ == "__main__":
    unittest.main()


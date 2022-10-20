from unittest.mock import MagicMixin
from CompTools import CompTools
import os
from pymatgen.io.vasp.sets import MPRelaxSet, MPStaticSet, MPScanRelaxSet, MPScanStaticSet, MPHSERelaxSet, MPHSEBSSet, MVLSlabSet, LobsterSet
from pymatgen.core.structure import Structure

from MPQuery import MPQuery

"""
Holey Moley, getting pymatgen to find your POTCARs is not trivial...
Here's the workflow I used:
    1) Download potpaw_LDA_PBE_52_54_orig.tar.gz from VASP
    2) Extract the tar into a directory, we'll call it ~/bin/pp
    3) $ pmg config -p ~/bin/pp pymatgen_pot
    4) $ pmg config -p PMG_VASP_PSP_DIR ~/bin/pymatgen_pot
    5) $ pmg config --add PMG_DEFAULT_FUNCTIONAL PBE_54

"""

class VASPSetUp(object):
    
    def __init__(self, calc_dir):
        """
        Args:
            calc_dir (os.PathLike) - directory where I want to execute VASP
            
        Returns:
            structure (pymatgen.Structure) - structure to be used for VASP calculation
        """
        
        self.calc_dir = calc_dir
        fpos = os.path.join(self.calc_dir, 'POSCAR')
        if not os.path.exists(fpos):
            print('No POSCAR found in {}'.format(self.calc_dir))
            self.structure = None
        else:
            self.structure = Structure.from_file(fpos)
        
    def prepare_calc(self,
                      xc='gga',
                      calc='relax',
                      fun='default',
                      standard='mp',
                      mag='fm',
                      modify_incar={}, 
                      modify_kpoints=None, 
                      modify_potcar=None, 
                      potcar_functional='PBE_54',
                      **kwargs):
        """
        Args:
            xc (str) - exchange-correlation functional
                - 'gga' (default) - Perdew-Burke-Ernzerhof (PBE)
                - 'metagga' (default) - r2SCAN
                
            calc (str) - type of calculation
                - 'relax' : geometry optimization
                - 'static' : static calculation
                - 'slab': slab calculation
                
            modify_incar (dict) - user-defined incar settings
                - e.g., {'NPAR' : 4}
                - notes: 
                    - {'MAGMOM' : [MAG_SITE1 MAG_SITE2 ...]}
                    - {LDAU' : [U_ION_1 U_ION_2 ...]}
                
            modify_kpoints (dict) - user-defined kpoint settings
                - Kpoints() for gamma only
                - {'reciprocal_density' : int} for automatic kpoint mesh
                - Kpoints(kpts=[[2,2,2]]) for a 2x2x2 grid
                
            modify_potcar (dict) - user-defined potcar settings
                - e.g., {'Gd' : 'Gd_3'}
                
            potcar_functional (str) - functional for POTCAR
                - note: not sure if chaning this will break code (POTCARs in pmg are a mess)
                - note: I'm not sure a good reason why we would need to change this
                
            **kwargs - additional arguments for VASPSet
        """
        
        if not self.structure:
            print('No structure found')
            return
        s = self.structure
        cmsa = CollinearMagneticStructureAnalyzer(s)
        if mag == 'nm':
            s = get_nonmagnetic_structure(s)
        elif mag == 'fm':
            s = get_ferromagnetic_structure(s)
        elif mag == 'afm':
            raise NotImplementedError('AFM not implemented yet')
        

        if xc == 'gga':
            if calc == 'relax':
                vaspset = MPRelaxSet
            elif calc == 'static':
                vaspset = MPStaticSet
                
            if fun != 'default':
                modify_incar['GGA'] = fun
                
        elif xc == 'metagga':
            if calc == 'relax':
                vaspset = MPScanRelaxSet
            elif calc == 'static':
                vaspset = MPScanStaticSet
                
            if fun != 'default':
                modify_incar['METAGGA'] = fun
                
        if calc == 'slab':
            vaspset = MVLSlabSet
            
        if standard == 'dmc':
            modify_incar['EDIFF'] = 1e-6
            modify_incar['EDIFFG'] = -0.01
            
            
        
        vasp_input = vaspset(self.structure, 
                             user_incar_settings=modify_incar, 
                             user_kpoints_settings=modify_kpoints, 
                             user_potcar_settings=modify_potcar, 
                             potcar_functional=potcar_functional,
                             **kwargs)
        
        vasp_input.write_input(self.calc_dir)
        return vasp_input
        
class VASPAnalysis(object):
    
    def __init__(self, calc_dir):
        
        self.calc_dir
        
        
def main():
    
    mpq = MPQuery('***REMOVED***')
    s = mpq.get_structure_by_material_id('mp-22584')
    s.to(filename='POSCAR')
    vsu = VASPSetUp(os.getcwd())
    vsu = vsu.prepare_calc()
    
    return vsu

if __name__ == '__main__':
    vsu = main()
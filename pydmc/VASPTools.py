from pydmc.CompTools import CompTools
from pydmc.handy import read_json, write_json
from pydmc.MagTools import MagTools

import os
import warnings

from pymatgen.io.vasp.sets import MPRelaxSet, MPStaticSet, MPScanRelaxSet, MPScanStaticSet, MPHSERelaxSet, MPHSEBSSet, MVLSlabSet, LobsterSet
from pymatgen.io.vasp.outputs import Vasprun
from pymatgen.core.structure import Structure
from pymatgen.io.vasp.inputs import Kpoints, Incar

"""
Holey Moley, getting pymatgen to find your POTCARs is not trivial...
Here's the workflow I used:
    1) Download potpaw_LDA_PBE_52_54_orig.tar.gz from VASP
    2) Extract the tar into a directory, we'll call it ~/bin/pp
    3) $ cd ~/bin
    4) $ pmg config -p ~/bin/pp pymatgen_pot
    5) $ pmg config -add PMG_VASP_PSP_DIR ~/bin/pymatgen_pot
    6) $ pmg config --add PMG_DEFAULT_FUNCTIONAL PBE_54

"""

class VASPSetUp(object):
    
    def __init__(self, 
                 calc_dir,
                 magmom=None):
        """
        Args:
            calc_dir (os.PathLike) - directory where I want to execute VASP
            magmom (dict) - {'magmom' : [list of magmoms (float)]}
                - or None
                - only needed for AFM calculations where orderings would have been determined using MagTools separately
        Returns:
            structure (pymatgen.Structure) - structure to be used for VASP calculation
        """
        
        self.calc_dir = calc_dir
        fpos = os.path.join(calc_dir, 'POSCAR')
        if not os.path.exists(fpos):
            raise FileNotFoundError('POSCAR not found in {}'.format(self.calc_dir))
        else:
            self.structure = Structure.from_file(fpos)
            
        self.magmom = magmom

        
    def get_vasp_input(self,
                      standard='mp',
                      xc='gga',
                      calc='relax',
                      fun='default',
                      mag='fm',
                      modify_incar={}, 
                      modify_kpoints=None, 
                      modify_potcar=None, 
                      potcar_functional='PBE_54',
                      validate_magmom=False,
                      **kwargs):
        """
        Args:
            standard (str) - for generating consistent data
                if not None:
                    options:
                        - 'mp' - Materials Project 
                        - 'dmc' - DMC 
                    note: this could override other args to ensure consistency
                - If None:
                    - specify whatever you'd like
                    
            xc (str) - rung of Jacob's ladder
                - modifies the default in "fun"
                    - 'gga' (default) - default=Perdew-Burke-Ernzerhof (PBE)
                    - 'ggau' - default=PBE+U with MP U's
                    - 'metagga' - default=r2SCAN
                
            calc (str) - type of calculation
                - 'loose' : loose geometry optimization
                - 'relax' : geometry optimization
                - 'static' : static calculation
                - 'slab' : slab geometry optmization
                
            fun (str) - specify functional of interest
                - 'default'
                    - PBE if xc == 'gga'
                    - r2SCAN if xc == 'metagga'
                    
            mag (str) - magnetic ordering type
                - 'nm' : nonmagnetic (ISPIN = 1)
                - 'fm' : ferromagnetic (ISPIN = 2)
                - 'afm' : antiferromagnetic (ISPIN = 2)
                    - note: structure passed to VASPSetUp must have magmoms defined for AFM calculation
                        - e.g., if many AFM orderings were generated using MagTools, tell VASPSetUp which one
                
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
            
            validate_magmom (bool) - VASPSet thing
                - note: setting to False because this was causing (as far as I could tell) non-useful errors    
            **kwargs - additional arguments for VASPSet
        """
        
        if standard and modify_incar:
            warnings.warn('You are attempting to generate consistent data, but modifying things in the INCAR')
        if standard and modify_kpoints:
            warnings.warn('You are attempting to generate consistent data, but modifying things in the KPOINTS')
        if standard and modify_potcar:
            warnings.warn('You are attempting to generate consistent data, but modifying things in the POTCAR')
        
        if MagTools(self.structure).could_be_magnetic and (mag == 'nm'):
            warnings.warn('Structure could be magnetic, but you are performing a nonmagnetic calculation')
        
        s = self.structure
        if mag == 'nm':
            s = MagTools(s).get_nonmagnetic_structure
        elif mag == 'fm':
            s = MagTools(s).get_ferromagnetic_structure
        elif 'afm' in mag:
            magmom = self.magmom
            if not magmom:
                raise ValueError('You must specify a magmom for an AFM calculation')
            if (min(magmom) >= 0) and (max(magmom) <= 0):
                raise ValueError('Structure is not AFM, but you are trying to run AFM calculation')
            s.add_site_property('magmom', magmom)

        if standard == 'mp':
            fun = 'default'
            
        if standard == 'dmc':
            fun = 'default'
            modify_incar['EDIFF'] = 1e-6
            modify_incar['EDIFFG'] = -0.03
            modify_incar['ISMEAR'] = 0
            modify_incar['ENCUT'] = 520
            modify_incar['ENAUG'] = 1040
            modify_incar['ISYM'] = 0
            modify_incar['SIGMA'] = 0.01
            modify_incar['LWAVE'] = False
            if calc != 'loose':
                modify_incar['KSPACING'] = 0.22
            if xc != 'ggau':
                modify_incar['LDAU'] = False
            modify_incar['ISPIN'] = 1 if mag == 'nm' else 2
            if calc != 'loose':
                modify_kpoints = {'reciprocal_density' : 500}
        
        if xc in ['gga', 'ggau']:
            vaspset = MPRelaxSet
            
            if fun != 'default':
                modify_incar['GGA'] = fun
                
        elif xc == 'metagga':
            vaspset = MPScanRelaxSet
                
            if fun != 'default':
                modify_incar['METAGGA'] = fun
                
        if calc == 'slab':
            vaspset = MVLSlabSet()
            
        if calc == 'loose':
            modify_kpoints = Kpoints()
            modify_incar['ENCUT'] = 400
            modify_incar['ENAUG'] = 800
            modify_incar['ISIF'] = 2
            modify_incar['EDIFF'] = 1e-5
        
        if calc == 'static':
            modify_incar['LCHARG'] = True
            modify_incar['LREAL'] = False
            modify_incar['NSW'] = 0
            modify_incar['LORBIT'] = 0
            modify_incar['LVHAR'] = True
            modify_incar['ICHARG'] = 11
            
        modify_incar['LWAVE'] = True
                
        vasp_input = vaspset(s, 
                             user_incar_settings=modify_incar, 
                             user_kpoints_settings=modify_kpoints, 
                             user_potcar_settings=modify_potcar, 
                             user_potcar_functional=potcar_functional,
                             validate_magmom=validate_magmom,
                             **kwargs)
        
        return vasp_input
    
    def prepare_calc(self, **kwargs):
        vasp_input = self.get_vasp_input(**kwargs)
        vasp_input.write_input(self.calc_dir)
        return vasp_input
    
    def error_msgs(self):
        return {
            "tet": ["Tetrahedron method fails for NKPT<4",
                    "Fatal error detecting k-mesh",
                    "Fatal error: unable to match k-point",
                    "Routine TETIRR needs special values",
                    "Tetrahedron method fails (number of k-points < 4)"],
            "inv_rot_mat": ["inverse of rotation matrix was not found (increase "
                            "SYMPREC)"],
            "brmix": ["BRMIX: very serious problems"],
            "subspacematrix": ["WARNING: Sub-Space-Matrix is not hermitian in "
                               "DAV"],
            "tetirr": ["Routine TETIRR needs special values"],
            "incorrect_shift": ["Could not get correct shifts"],
            "real_optlay": ["REAL_OPTLAY: internal error",
                            "REAL_OPT: internal ERROR"],
            "rspher": ["ERROR RSPHER"],
            "dentet": ["DENTET"],
            "too_few_bands": ["TOO FEW BANDS"],
            "triple_product": ["ERROR: the triple product of the basis vectors"],
            "rot_matrix": ["Found some non-integer element in rotation matrix"],
            "brions": ["BRIONS problems: POTIM should be increased"],
            "pricel": ["internal error in subroutine PRICEL"],
            "zpotrf": ["LAPACK: Routine ZPOTRF failed"],
            "amin": ["One of the lattice vectors is very long (>50 A), but AMIN"],
            "zbrent": ["ZBRENT: fatal internal in",
                       "ZBRENT: fatal error in bracketing"],
            "pssyevx": ["ERROR in subspace rotation PSSYEVX"],
            "eddrmm": ["WARNING in EDDRMM: call to ZHEGV failed"],
            "edddav": ["Error EDDDAV: Call to ZHEGV failed"],
            "grad_not_orth": [
                "EDWAV: internal error, the gradient is not orthogonal"],
            "nicht_konv": ["ERROR: SBESSELITER : nicht konvergent"],
            "zheev": ["ERROR EDDIAG: Call to routine ZHEEV failed!"],
            "elf_kpar": ["ELF: KPAR>1 not implemented"],
            "elf_ncl": ["WARNING: ELF not implemented for non collinear case"],
            "rhosyg": ["RHOSYG internal error"],
            "posmap": ["POSMAP internal error: symmetry equivalent atom not found"],
            "point_group": ["Error: point group operation missing"]
        }
        
    @property
    def error_log(self):
        error_msgs = self.error_msgs
        out_file = os.path.join(self.calc_dir, self.configs.fvaspout)
        errors = []
        with open(out_file) as f:
            contents = f.read()
        for e in error_msgs:
            for t in error_msgs[e]:
                if t in contents:
                    errors.append(e)
        return errors
    
    @property
    def is_clean(self):
        if VASPAnalysis(self.calc_dir).is_converged:
            return True
        if not os.path.exists(os.path.join(self.calc_dir, self.configs.fvaspout)):
            return True
        errors = self.error_log
        if len(errors) == 0:
            return True
        with open(os.path.join(self.calc_dir, self.configs.fvasperrors), 'w') as f:
            for e in errors:
                f.write(e+'\n')
        return False
    
    @property
    def incar_changes_from_errors(self):
        errors = self.error_log
        chgcar = os.path.join(self.calc_dir, 'CHGCAR')
        wavecar = os.path.join(self.calc_dir, 'WAVECAR')
        
        incar_changes = {}
        if 'grad_not_orth' in errors:
            incar_changes['SIGMA'] = 0.05
            if os.path.exists(wavecar):
                os.remove(wavecar)
            incar_changes['ALGO'] = 'Exact'
        if 'edddav' in errors:
            incar_changes['ALGO'] = 'All'
            if os.path.exists(chgcar):
                os.remove(chgcar)
        if 'eddrmm' in errors:
            if os.path.exists(wavecar):
                os.remove(wavecar)
        if 'subspacematrix' in errors:
            incar_changes['LREAL'] = 'FALSE'
            incar_changes['PREC'] = 'Accurate'
        if 'inv_rot_mat' in errors:
            incar_changes['SYMPREC'] = 1e-8
        if 'zheev' in errors:
            incar_changes['ALGO'] = 'Exact'
        if 'zpotrf' in errors:
            incar_changes['ISYM'] = -1
        if 'zbrent' in errors:
            incar_changes['IBRION'] = 1
        if 'brmix' in errors:
            incar_changes['IMIX'] = 1
        
        return incar_changes
            
class VASPAnalysis(object):
    
    def __init__(self, calc_dir):
        
        self.calc_dir = calc_dir
    
    def is_converged(self, calc):
        fvasprun = os.path.join(self.calc_dir, 'vasprun.xml')
        if not os.path.exists(fvasprun):
            return False
        
        vr = Vasprun(os.path.join(self.calc_dir, 'vasprun.xml'))
        if calc == 'static':
            return vr.converged_electronic
        else:
            return vr.converged
        
        
def main():
    from pydmc.MPQuery import MPQuery

    
    remake = False
    mpid = 'mp-770495'
    calc_dir = os.path.join(os.getcwd(), '..', 'dev', mpid)
    if not os.path.exists(calc_dir):
        os.mkdir(calc_dir)
    fpos = os.path.join(calc_dir, 'POSCAR')
    if not os.path.exists(fpos) or remake:
        mpq = MPQuery('***REMOVED***')
        s = mpq.get_structure_by_material_id(mpid)
        s.to(filename=os.path.join(calc_dir, 'POSCAR'))
        
    f_magmoms = os.path.join(calc_dir, 'magmoms.json')
    if not f_magmoms or remake:
        s = Structure.from_file(fpos)
        mt = MagTools(s)
        out = mt.get_antiferromagnetic_structures
        
        magmoms = {str(i) : out[i].site_properties['magmom'] for i in range(len(out))}
        
        magmoms = write_json(magmoms, f_magmoms)
    magmoms = read_json(f_magmoms)
    vsu = VASPSetUp(calc_dir, magmom=magmoms['10'])
    vsu = vsu.prepare_calc(standard='mp', 
                           mag='afm', 
                           xc='metagga', 
                           calc='static')
    
    print(calc_dir)
    
    return vsu

if __name__ == '__main__':
    vsu = main()
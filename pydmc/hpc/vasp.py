from pydmc.utils.handy import read_json, write_json
from pydmc.core.mag import MagTools
from pydmc.core.struc import StrucTools, SiteTools
from pydmc.core.comp import CompTools
from pydmc.hpc.analyze import AnalyzeVASP

import os
import warnings
from shutil import copyfile
import numpy as np

from pymatgen.io.vasp.sets import MPRelaxSet, MPScanRelaxSet
from pymatgen.core.structure import Structure
from pymatgen.io.vasp.inputs import Kpoints, Incar
from pymatgen.io.lobster.inputs import Lobsterin

"""
Holey Moley, getting pymatgen to find your POTCARs is not trivial...
Here's the workflow I used:
    1) Download potpaw_LDA_PBE_52_54_orig.tar.gz from VASP
    2) Extract the tar into a directory, we'll call it FULL_PATH/bin/pp
    3) Download potpaw_PBE.tgz from VASP
    4) Extract the tar INSIDE a directory: FULL_PATH/bin/pp/potpaw_PBE
    3) $ cd FULL_PATH/bin
    4) $ pmg config -p FULL_PATH/bin/pp pymatgen_pot
    5) $ pmg config -add PMG_VASP_PSP_DIR FULL_PATH/bin/pymatgen_pot
    6) $ pmg config --add PMG_DEFAULT_FUNCTIONAL PBE_54

"""

class VASPSetUp(object):
    """
    Use to write VASP inputs for a single initial structure
    
    Also changes inputs based on errors that are encountered
    """
    
    def __init__(self, 
                 calc_dir,
                 magmom=None,
                 fvaspout='vasp.o',
                 fvasperrors='errors.o',
                 lobster_static=True,
                 mag_override=False):
        """
        Args:
            calc_dir (os.PathLike) - directory where I want to execute VASP
            magmom (dict) - {'magmom' : [list of magmoms (float)]}
                - or None
                - only needed for AFM calculations where orderings would have been determined using MagTools separately
            fvaspout (str) - name of file to write VASP output to
            fvasperrors (str) - name of file to write VASP errors to
            lobster_static (bool) - if True, run LOBSTER on static calculations
            mag_override (bool) - allows user to run nonmagnetic calcs for magnetic systems and vice versa
            
        Returns:
            calc_dir (os.PathLike) - directory where I want to execute VASP
            structure (pymatgen.Structure) - structure to be used for VASP calculation
                - note: raises error if no POSCAR in calc_dir
            magmom (dict) - {'magmom' : [list of magmoms (float)]}
                - or None
                - only needed for AFM calculations where orderings would have been determined using MagTools separately        
            fvaspout (str) - name of file to write VASP output to
            fvasperrors (str) - name of file to write VASP errors to    
        """
        
        self.calc_dir = calc_dir
        fpos = os.path.join(calc_dir, 'POSCAR')
        if not os.path.exists(fpos):
            raise FileNotFoundError('POSCAR not found in {}'.format(self.calc_dir))
        else:
            self.structure = Structure.from_file(fpos)
            
        self.magmom = magmom
        self.fvaspout = fvaspout
        self.fvasperrors = fvasperrors
        self.lobster_static = lobster_static
        self.mag_override = mag_override

    def get_vasp_input(self,
                      standard='dmc',
                      xc='gga',
                      calc='relax',
                      fun='default',
                      mag='fm',
                      modify_incar={}, 
                      modify_kpoints=None, 
                      modify_potcar=None, 
                      potcar_functional='PBE_54',
                      validate_magmom=False,
                      verbose=True,
                      **kwargs):
        """
        Args:
            standard (str) - for generating consistent data
                if not None:
                    options:
                        - 'mp' - Materials Project
                            - this implies we want to compare energies to Materials Project, therefore strict adherence is required
                            - standard = 'mp' can only be combined w/ xc = 'ggau' as this is what's in MP (currently - 12/2022) 
                        - 'dmc' - DMC 
                        - specify whatever you'd like ('high_cutoff', 'lobster', 'custom', 'strict_ediff')
                            - if it's not in ['mp', 'dmc'], it won't do anything to calc, but might be useful flag                    
                    note: this could override other args to ensure consistency
                - If None:
                    - won't be used                    
                    
            xc (str) - rung of Jacob's ladder
                - modifies the default in "fun"
                    - 'gga' (default) - default=Perdew-Burke-Ernzerhof (PBE)
                    - 'ggau' - default=PBE+U with MP U's
                    - 'metagga' - default=r2SCAN
                
            calc (str) - type of calculation
                - 'loose' : loose geometry optimization (only 1 k point; easier convergence)
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
                        - e.g., if many AFM orderings were generated using MagTools, tell VASPSetUp which one to use
                            - a general approach might be to first create a dictionary of these magmoms = {IDX : [magmoms]},
                                then iterate through that dictionary to generate various 'afm_IDX' directories
                    - code will read anything containing "afm" as AFM (e.g., "afm_1")
                
            modify_incar (dict) - user-defined incar settings
                - e.g., {'NCORE' : 4}
                - notes: 
                    - {'MAGMOM' : [MAG_SITE1 MAG_SITE2 ...]}
                    - {LDAU' : [U_ION_1 U_ION_2 ...]}
                
            modify_kpoints (dict) - user-defined kpoint settings
                - Kpoints() for gamma only
                - {'reciprocal_density' : int} for automatic kpoint mesh
                - Kpoints(kpts=[[2,2,2]]) for a 2x2x2 grid
                - if not None:
                    - this will override KSPACING in INCAR (in addition to default kpoints settings)
                
            modify_potcar (dict) - user-defined potcar settings
                - e.g., {'Gd' : 'Gd_3'}
                
            potcar_functional (str) - functional for POTCAR
                - note: not sure if chaning this will break code (POTCARs in pmg are a mess)
                - note: I'm not sure a good reason why we would need to change this
            
            validate_magmom (bool) - VASPSet thing
                - note: setting to False because this was causing (as far as I could tell) non-useful errors
            
            verbose (bool) - print stuff
            
            **kwargs - additional arguments for VASPSet
        """
        
        if (standard == 'mp') and (xc != 'ggau'):
            warnings.warn('standard = mp, but xc != ggau; not setting up\n')
            return None
        
        if (calc == 'loose') and (xc == 'metagga'):
            warnings.warn('calc = loose; xc = metagga; not setting up\n')
            return None
        
        if (mag == 'nm') and (MagTools(self.structure).could_be_magnetic):
            if self.mag_override == False:
                warnings.warn('mag = nm, but structure could be magnetic\n')
                return None
            
        if (mag != 'nm') and (MagTools(self.structure).could_be_magnetic == False):
            if self.mag_override == False:
                warnings.warn('mag != nm, but structure could not be magnetic\n')
                return None
            
        if verbose:
            # tell user what they are modifying in case they are trying to match MP or other people's calculations
            if standard and modify_incar:
                warnings.warn('you are attempting to generate consistent data, but modifying things in the INCAR\n')
                #print('e.g., %s' % str(modify_incar))
                
            if standard and modify_kpoints:
                warnings.warn('you are attempting to generate consistent data, but modifying things in the KPOINTS\n')
                #print('e.g., %s' % str(modify_kpoints))

            if standard and modify_potcar:
                warnings.warn('you are attempting to generate consistent data, but modifying things in the POTCAR\n')
                #print('e.g., %s' % str(modify_potcar))
            
            # tell user they are doing a nonmagnetic calculation for a compound w/ magnetic elements
            if MagTools(self.structure).could_be_magnetic and (mag == 'nm'):
                warnings.warn('structure could be magnetic, but you are performing a nonmagnetic calculation\n')
        
        s = self.structure
        
        # get MAGMOM
        if mag == 'nm':
            s = MagTools(s).get_nonmagnetic_structure
        elif mag == 'fm':
            s = MagTools(s).get_ferromagnetic_structure
        elif 'afm' in mag:
            magmom = self.magmom
            if not magmom:
                raise ValueError('you must specify a magmom for an AFM calculation\n')
            if (min(magmom) >= 0) and (max(magmom) <= 0):
                raise ValueError('provided magmom that is not AFM, but you are trying to run an AFM calculation\n')
            s.add_site_property('magmom', magmom)

        # MP wants to set W_pv but we don't have that one in PBE54 (no biggie)
        if not modify_potcar:
            modify_potcar = {'W' : 'W'}
        elif isinstance(modify_potcar, dict):
            modify_potcar['W'] = 'W'
        # don't mess with much if trying to match Materials Project
        if standard == 'mp':
            fun = 'default'
            if not modify_kpoints:
                modify_kpoints = {'reciprocal_density' : 64}
            elif isinstance(modify_kpoints, dict):
                modify_kpoints['reciprocal_density'] = 64
                
        # setting DMC standards --> what to do on top of MPRelaxSet or MPScanRelaxSet
        if standard == 'dmc':
            fun = 'default' # same functional
            dmc_standard_configs = {'EDIFF' : 1e-6,
                                    'EDIFFG' : -0.03,
                                    'ISMEAR' : 0,
                                    'ENCUT' : 520,
                                    'ENAUG' : 1040,
                                    'ISYM' : 0,
                                    'SIGMA' : 0.01}
            for key in dmc_standard_configs:
                if key not in modify_incar:
                    modify_incar[key] = dmc_standard_configs[key]

            # use length = 25 means reciprocal space discretization of 25 K-points per Å−1
            if calc != 'loose':
                if not modify_kpoints:
                    modify_kpoints = {'length' : 25}
                
            # turn off +U unless we are specifying GGA+U    
            if xc != 'ggau':
                if 'LDAU' not in modify_incar:
                    modify_incar['LDAU'] = False
                
            # turn off ISPIN for nonmagnetic calculations
            if 'ISPIN' not in modify_incar:
                modify_incar['ISPIN'] = 1 if mag == 'nm' else 2
        
        # start from MPRelaxSet for GGA or GGA+U
        if xc in ['gga', 'ggau']:
            vaspset = MPRelaxSet
            
            # use custom functional (eg PBEsol) if you want
            if 'GGA' not in modify_incar:
                if fun != 'default':
                    modify_incar['GGA'] = fun.upper()
                else:
                    modify_incar['GGA'] = 'PE'

            # for strict comparison to Materials Project GGA calculations, we need to use the old POTCARs
            if standard == 'mp':
                potcar_functional = None
        # start from MPScanRelaxSet for meta-GGA
        elif xc == 'metagga':
            vaspset = MPScanRelaxSet
                
            # use custom functional (eg SCAN) if you want
            if 'METAGGA' not in modify_incar:
                if fun != 'default':
                    modify_incar['METAGGA'] = fun.upper()
                else:
                    modify_incar['METAGGA'] = 'R2SCAN'
        
        # default "loose" relax
        if calc == 'loose':
            modify_kpoints = Kpoints() # only use 1 kpoint
            loose_configs = {'ENCUT' : 400,
                             'ENAUG' : 800,
                             'ISIF' : 2,
                             'EDIFF' : 1e-5,
                             'NELM' : 40}
            for key in loose_configs:
                modify_incar[key] = loose_configs[key]
            # NOTE: this will override even user-specified settings in modify_incar
        
        # default "static" claculation
        if calc == 'static':
            static_configs = {'LCHARG' : True,
                              'LREAL' : False,
                              'NSW' : 0,
                              'LORBIT' : 0,
                              'LVHAR' : True,
                              'ICHARG' : 0,
                              'LAECHG' : True}
            for key in static_configs:
                modify_incar[key] = static_configs[key]
        
        # make sure WAVECAR is written unless told not to
        if 'LWAVE' not in modify_incar:
            modify_incar['LWAVE'] = True
        
        # use better parallelization
        if ('NCORE' not in modify_incar) and ('NPAR' not in modify_incar):
            modify_incar['NCORE'] = 4
        
        # add more ionic steps
        if 'NSW' not in modify_incar:
            modify_incar['NSW'] = 199
        
        # make sure spin is off for nm calculations
        if mag == 'nm':
            modify_incar['ISPIN'] = 1
        else:
            modify_incar['LORBIT'] = 11
        #print(modify_incar)    
        # initialize new VASPSet
        
        if self.lobster_static and (calc == 'static'):
            modify_incar['NEDOS'] = 4000
            modify_incar['ISTART'] = 0
            modify_incar['LAECHG'] = True
            if not modify_kpoints:
                # need KPOINTS file for LOBSTER
                modify_kpoints = {'reciprocal_density' : 500}
        
        if self.lobster_static:
            if xc == 'metagga':
                # gga-static will get ISYM = -1, so need to pass that to metagga
                modify_incar['ISYM'] = -1
            
        vasp_input = vaspset(s, 
                             user_incar_settings=modify_incar, 
                             user_kpoints_settings=modify_kpoints, 
                             user_potcar_settings=modify_potcar, 
                             user_potcar_functional=potcar_functional,
                             validate_magmom=validate_magmom,
                             **kwargs)
        
        return vasp_input
    
    def prepare_calc(self, **kwargs):
        """
        Write input files (INCAR, KPOINTS, POTCAR)
        """
        vasp_input = self.get_vasp_input(**kwargs)
        if not vasp_input:
            return None
#        print('\n\n\n')
#        print(vasp_input.incar)
#        print('\n\n\n')
        vasp_input.write_input(self.calc_dir)
        
        if self.lobster_static:
            analyzer = AnalyzeVASP(self.calc_dir)
            if analyzer.incar_parameters['NSW'] == 0:
                INCAR_input = os.path.join(self.calc_dir, 'INCAR_input')
                INCAR_output = os.path.join(self.calc_dir, 'INCAR')
                copyfile(INCAR_output, INCAR_input)
                POSCAR_input = os.path.join(self.calc_dir, 'POSCAR')
                POTCAR_input = os.path.join(self.calc_dir, 'POTCAR')
                lobsterin = Lobsterin.standard_calculations_from_vasp_files(POSCAR_input=POSCAR_input,
                                                                            INCAR_input=INCAR_input,
                                                                            POTCAR_input=POTCAR_input,
                                                                            option='standard')
                lobsterin.write_lobsterin(os.path.join(self.calc_dir, 'lobsterin'))
                lobsterin.write_INCAR(incar_input=INCAR_input,
                                      incar_output=INCAR_output,
                                      poscar_input=POSCAR_input)
                
            
        return vasp_input
    
    @property
    def error_msgs(self):
        """
        Dict of {group of errors (str) : [list of error messages (str) in group]}
        """
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
            "point_group": ["Error: point group operation missing"],
            "ibzkpt" : ["internal error in subroutine IBZKPT"],
            "bad_sym" : ["ERROR: while reading WAVECAR, plane wave coefficients changed"]
        }
        
    @property
    def unconverged_log(self):
        """
        checks to see if both ionic and electronic convergence have been reached
            if calculation had NELM # electronic steps, electronic convergence may not be met
            if calculation had NSW # ionic steps, ionic convergence may not be met
        
        returns a list, unconverged, that can have 0, 1, or 2 items
            if unconverged = []:
                the calculation either:
                    1) didn't finish (vasprun.xml not found or incomplete)
                    2) both ionic and electronic convergence were met
            if 'nelm_too_low' in unconverged:
                the calculation didn't reach electronic convergence
            if 'nsw_too_low' in unconverged:
                the calculation didn't reach ionic convergence
        """
        analyzer = AnalyzeVASP(self.calc_dir)
        unconverged = []
        if analyzer.is_converged:
            return unconverged
        
        vr = analyzer.vasprun
        if not vr:
            return unconverged
        
        electronic_convergence = vr.converged_electronic
        if 'relax' in self.calc_dir:
            ionic_convergence = vr.converged_ionic
        else:
            ionic_convergence = True
            
        if not electronic_convergence:
            unconverged.append('nelm_too_low')
        if not ionic_convergence:
            unconverged.append('nsw_too_low')
            
        return unconverged
        
        
    @property
    def error_log(self):
        """
        Parse fvaspout for error messages
        
        Returns list of errors (str)        
        """
        error_msgs = self.error_msgs
        out_file = os.path.join(self.calc_dir, self.fvaspout)
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
        """
        True if no errors found, else False
        """
        clean = False
        if AnalyzeVASP(self.calc_dir).is_converged:
            clean = True
        if not os.path.exists(os.path.join(self.calc_dir, self.fvaspout)):
            clean = True
        if clean == True:
            with open(os.path.join(self.calc_dir, self.fvasperrors), 'w') as f:
                f.write('')
            return clean       
        errors = self.error_log + self.unconverged_log
        if len(errors) == 0:
            return True
        with open(os.path.join(self.calc_dir, self.fvasperrors), 'w') as f:
            for e in errors:
                f.write(e+'\n')
        return clean
    
    @property
    def incar_changes_from_errors(self):
        """
        Automatic INCAR changes based on errors
            - NOTE: also may remove WAVECAR and/or CHGCAR as needed
        Returns {INCAR key (str) : INCAR value (str)}
        """
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
            incar_changes['ALGO'] = 'Normal'
        if 'subspacematrix' in errors:
            incar_changes['LREAL'] = False
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
        if 'ibzkpt' in errors:
            incar_changes['SYMPREC'] = 1e-10
            incar_changes['ISMEAR'] = 0
            incar_changes['ISYM'] = -1
        if 'posmap' in errors:
            incar_changes['SYMPREC'] = 1e-5
            incar_changes['ISMEAR'] = 0
            incar_changes['ISYM'] = -1
        if 'nelm_too_low' in self.unconverged_log:
            incar_changes['NELM'] = 399
            incar_changes['ALGO'] = 'All'
        if 'nsw_too_low' in self.unconverged_log:
            incar_changes['NSW'] = 399
        if 'real_optlay' in errors:
            incar_changes['LREAL'] = False
        if 'bad_sym' in errors:
            incar_changes['ISYM'] = -1
        return incar_changes
            

        
def main():
    calc_dir = 'examples/vasp_dmc/data/gga-static-LiF-lobster'
    va = AnalyzeVASP(calc_dir)
    
    pdos = va.pdos(remake=False)
    tdos = va.tdos(remake=True, pdos=read_json(os.path.join(calc_dir, 'pdos.json')))
    #tdos = None
    return pdos, tdos

if __name__ == '__main__':
    pdos, tdos = main()

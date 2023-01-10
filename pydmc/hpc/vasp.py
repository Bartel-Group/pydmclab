from pydmc.core.mag import MagTools
from pydmc.core.struc import StrucTools
from pydmc.hpc.analyze import AnalyzeVASP, VASPOutputs
from pydmc.data.configs import load_vasp_configs
from pydmc.utils.handy import read_yaml, write_yaml, dotdict

import os
import warnings
from shutil import copyfile

from pymatgen.io.vasp.sets import MPRelaxSet, MPScanRelaxSet
from pymatgen.core.structure import Structure
from pymatgen.io.vasp.inputs import Kpoints
from pymatgen.io.lobster.inputs import Lobsterin

"""
Holey Moley, getting pymatgen to find your POTCARs is not trivial...
Here's the workflow I used:
    1) Download potpaw_LDA_PBE_52_54_orig.tar.gz from VASP
    2) Extract the tar into a directory, we'll call it FULL_PATH/bin/pp
    3) Download potpaw_PBE.tgz from VASP
    4) Extract the tar INSIDE a directory: FULL_PATH/bin/pp/potpaw_PBE
    5) $ cd FULL_PATH/bin
    6) $ pmg config -p FULL_PATH/bin/pp pymatgen_pot
    7) $ pmg config --add PMG_VASP_PSP_DIR FULL_PATH/bin/pymatgen_pot
    8) $ pmg config --add PMG_DEFAULT_FUNCTIONAL PBE_54
    
Now that this has been done, new users must just do:
    1) $ pmg config --add PMG_VASP_PSP_DIR FULL_PATH/bin/pymatgen_pot
    2) $ pmg config --add PMG_DEFAULT_FUNCTIONAL PBE_54

"""

class VASPSetUp(object):
    """
    Use to write VASP inputs for a single VASP calculation
        - a calculation here might be defined as the same:
            - initial structure
            - initial magnetic configurations
            - input settings (INCAR, KPOINTS, POTCAR)
            - etc.
    
    Also changes inputs based on errors that are encountered
    
    Note that we rarely need to call this class directly
        - instead we'll manage things through pydmc.hpc.submit.SubmitTools and pydmc.hpc.launch.LaunchTools
    """
    
    def __init__(self, 
                 calc_dir,
                 user_configs={},
                 vasp_configs_yaml=os.path.join(os.getcwd(), '_vasp_configs.yaml',),
                 refresh_configs=True):
        """
        Args:
            calc_dir (os.PathLike) - directory where I want to execute VASP
                - there must be a POSCAR in calc_dir
                - the other input files will be added automatically within this code
            user_configs (dict) - user-defined configs
                - options in vasp_configs_yaml
            vasp_configs_yaml (os.PathLike) - path to yaml file with possible user_configs and default values
            refresh_configs (bool) - if True, will copy pydmc baseline configs to your local directory
                - this is useful if you've made changes to the configs files in the directory you're working in and want to start over           
                
        Returns:
            calc_dir (os.PathLike) - directory where I want to execute VASP
            structure (pymatgen.Structure) - structure to be used for VASP calculation
                - note: raises error if no POSCAR in calc_dir     
            configs (dict)
                # VASPSetUp (general)
                # these should get passed through LaunchTools
                standard: # make mp if comparing to MP data
                mag:  # **change** as desired
                magmom: # you will need to pass this for AFM calcs {'magmom' : [list of magmoms (float)]}

                # these should get passed through SubmitTools
                xc_to_run:  # gga, ggau, or metagga
                calc_to_run: # usually this is what we want unless you want an independent "static" or "loose" calc

                fun: # usually this is fine
                lobster_static: True # can make False if you don't want to run LOBSTER

                # some output files
                fvaspout: vasp.o # where vasp output goes in calc_dir for each vasp run
                fvasperrors: errors.o # where vasp errors go in calc_dir for each vasp run

                # may be desirable to pass various configs for each type of calculation
                # INCARs should be {INCAR_FLAG : value}
                # KPOINTs should be Kpoints object
                # POTCARs should be {el (str) : potcar flag (str)}

                # loose calculations
                loose_incar: {} # e.g., {'ENCUT' : 555}
                loose_kpoints: {} # e.g., {'length' : 25}
                loose_potcar: {} # e.g., {'W' : 'W_pv'}

                # relax calculations
                relax_incar: {}
                relax_kpoints: {}
                relax_potcar: {}

                # static calculations
                static_incar: {}
                static_kpoints: {}
                static_potcar: {}

                # lobster calculations
                lobster_incar: {}
                lobster_kpoints: {}
                lobster_potcar: {}

                # rarely need to change these:
                mag_override: False # if True, allows user to run nm calcs for mag systems and vice versa
                potcar_functional: PBE_54 # probably don't change
                validate_magmom: False # probably don't change
        """
        
        # this is where we will execute VASP
        self.calc_dir = calc_dir
        
        # we should have a POSCAR in calc_dir already 
            # e.g., LaunchTools will set this up for you
        fpos = os.path.join(calc_dir, 'POSCAR')
        if not os.path.exists(fpos):
            raise FileNotFoundError('POSCAR not found in {}'.format(calc_dir))
        else:
            self.structure = Structure.from_file(fpos)
            
        # write a local yaml with vasp configs
            # if you don't have one or
            # if you want to "refresh" them
        if not os.path.exists(vasp_configs_yaml) or refresh_configs:
            _vasp_configs = load_vasp_configs()
            write_yaml(_vasp_configs, vasp_configs_yaml)
        
        # read local yaml to get baseline vasp_configs
        _vasp_configs = read_yaml(vasp_configs_yaml)
        
        # augment baseline vasp_configs with user_configs
            # NOTE: user_configs will overwrite any keys shared with vasp_configs
        configs = {**_vasp_configs, **user_configs}
        
        essential_configs = ['xc_to_run', 'calc_to_run', 'standard', 'mag']
        for essential in essential_configs:
            if essential not in configs.keys():
                raise KeyError('{} must be specified in user_configs'.format(essential))
        
        # copy configs to prevent further changes                 
        self.configs = configs.copy()
        
        perturbation = self.configs['perturb_struc']
        if perturbation:
            initial_structure = self.structure.copy()
            perturbed_structure = StrucTools(initial_structure).perturb(perturbation)
            self.structure = perturbed_structure

    @property
    def get_vasp_input(self):
        """
        Returns:
            vasp_input (pymatgen.io.vasp.sets.VaspInputSet
            
        Uses configs to modify pymatgen's VaspInputSets as required
        """
        
        # copy configs to prevent unwanted updates
        configs = self.configs.copy()
        
        # initialize how we're going to modify each vasp input file with configs specs
        modify_incar = configs['%s_incar' % configs['calc_to_run']].copy()
        modify_kpoints = configs['%s_kpoints' % configs['calc_to_run']].copy()
        modify_potcar = configs['%s_potcar' % configs['calc_to_run']].copy()
        
        # initialize potcar functional
        potcar_functional = configs['potcar_functional']
        
        # this should be kept off in general, gives unuseful warnings (I think)
        validate_magmom = configs['validate_magmom']
        
        # tell user what they are modifying in case they are trying to match MP or other people's calculations
        if configs['standard'] and modify_incar:
            warnings.warn('you are attempting to generate consistent data, but modifying things in the INCAR\n')
            #print('e.g., %s' % str(modify_incar))
            
        if configs['standard'] and modify_kpoints:
            warnings.warn('you are attempting to generate consistent data, but modifying things in the KPOINTS\n')
            #print('e.g., %s' % str(modify_kpoints))

        if configs['standard'] and modify_potcar:
            warnings.warn('you are attempting to generate consistent data, but modifying things in the POTCAR\n')
            #print('e.g., %s' % str(modify_potcar))
        
        # tell user they are doing a nonmagnetic calculation for a compound w/ magnetic elements
        if MagTools(self.structure).could_be_magnetic and (configs['mag'] == 'nm'):
            warnings.warn('structure could be magnetic, but you are performing a nonmagnetic calculation\n')
        
        structure = self.structure
        
        # add MAGMOM to structure
        if configs['mag'] == 'nm':
            # if non-magnetic, MagTools takes care of this
            structure = MagTools(structure).get_nonmagnetic_structure
        elif configs['mag'] == 'fm':
            # if ferromagnetic, MagTools takes care of this
            structure = MagTools(structure).get_ferromagnetic_structure
        elif 'afm' in configs['mag']:
            # if antiferromagnetic, we need to aprovide a MAGMOM
            magmom = configs['magmom']
            if not magmom:
                raise ValueError('you must specify a magmom for an AFM calculation\n')
            if (min(magmom) >= 0) and (max(magmom) <= 0):
                raise ValueError('provided magmom that is not AFM, but you are trying to run an AFM calculation\n')
            structure.add_site_property('magmom', magmom)
        
        # MP wants to set W_pv but we don't have that one in PBE54 (no biggie)
        if configs['standard'] != 'mp':
            modify_potcar['W'] = 'W'
            
        # don't mess with much if trying to match Materials Project
        if configs['standard'] == 'mp':
            # use our default functional
            fun = None
            # set KPOINTS to be MP-consistent
            if not isinstance(modify_kpoints, dict):
                modify_kpoints = {}
                modify_kpoints['reciprocal_density'] = 64
                
        # setting DMC standards --> what to do on top of MPRelaxSet or MPScanRelaxSet (pymatgen defaults)
        if configs['standard'] == 'dmc':
            # use our default functional
            fun = None
            
            # tweak a few INCAR settings
                # converge using forces (EDIFFG < 0)
                # stricter EDIFF
                # ISMEAR = 0 (less convergence errors)
                # fix ENCUT, ENAUG to be reasonable
                # turn off symmetry (mostly, ISYM = 0)
            dmc_standard_settings = {'EDIFF' : 1e-6,
                                    'EDIFFG' : -0.03,
                                    'ISMEAR' : 0,
                                    'ENCUT' : 520,
                                    'ENAUG' : 1040,
                                    'ISYM' : 0,
                                    'SIGMA' : 0.01}
            for key in dmc_standard_settings:
                if key not in modify_incar:
                    modify_incar[key] = dmc_standard_settings[key]

            # use length = 25 means reciprocal space discretization of 25 K-points per Å−1
            if configs['calc_to_run'] != 'loose':
                if not modify_kpoints:
                    modify_kpoints = {'length' : 25}
                
            # turn off +U unless we are specifying GGA+U    
            if configs['xc_to_run'] != 'ggau':
                if 'LDAU' not in modify_incar:
                    modify_incar['LDAU'] = False
                
            # turn off ISPIN for nonmagnetic calculations
            if 'ISPIN' not in modify_incar:
                modify_incar['ISPIN'] = 1 if configs['mag'] == 'nm' else 2
        
        # start from MPRelaxSet for GGA or GGA+U
        if configs['xc_to_run'] in ['gga', 'ggau']:
            vaspset = MPRelaxSet
            
            # use custom functional (eg PBEsol) if you want
                # needs to be specified in user_configs['fun']
                # otherwise use PBE for gga, gga+u
            if 'GGA' not in modify_incar:
                if fun:
                    modify_incar['GGA'] = fun.upper()
                else:
                    modify_incar['GGA'] = 'PE'

            # for strict comparison to Materials Project GGA(+U) calculations, we need to use the old POTCARs
            if configs['standard'] == 'mp':
                potcar_functional = None
                
        # start from MPScanRelaxSet for meta-GGA
        elif configs['xc_to_run'] == 'metagga':         
            
            vaspset = MPScanRelaxSet
                
            # use custom functional (eg SCAN) if you want
                # needs to be specified in user_configs['fun']
                # otherwise use r2SCAN for metagga           
            if 'METAGGA' not in modify_incar:
                if fun:
                    modify_incar['METAGGA'] = fun.upper()
                else:
                    modify_incar['METAGGA'] = 'R2SCAN'
        
        # default "loose" relax
        if configs['calc_to_run'] == 'loose':
            # use only 1 kpoint
            modify_kpoints = Kpoints() 
            # make the settings a little looser
            loose_settings = {'ENCUT' : 400,
                             'ENAUG' : 800,
                             'ISIF' : 3,
                             'EDIFF' : 1e-5,
                             'NELM' : 40}
            for key in loose_settings:
                if key not in modify_incar:
                    modify_incar[key] = loose_settings[key]
        
        # default "static" claculation
        if configs['calc_to_run'] == 'static':
            # don't optimize the geometry
            # do save things like charge density
            static_settings= {'LCHARG' : True,
                              'LREAL' : False,
                              'NSW' : 0,
                              'LORBIT' : 0,
                              'LVHAR' : True,
                              'ICHARG' : 0,
                              'LAECHG' : True}
            for key in static_settings:
                if key not in static_settings:
                    modify_incar[key] = static_settings[key]
        
        # make sure WAVECAR is written unless told user specified not to
        if 'LWAVE' not in modify_incar:
            modify_incar['LWAVE'] = True
        
        # use better parallelization
        if ('NCORE' not in modify_incar) and ('NPAR' not in modify_incar):
            modify_incar['NCORE'] = 4
        
        # add more ionic steps
        if 'NSW' not in modify_incar:
            modify_incar['NSW'] = 199
        
        # make sure spin is off for nm calculations
        if configs['mag'] == 'nm':
            modify_incar['ISPIN'] = 1
        else:
            # make sure magnetization is written to OUTCAR for magnetic calcs
            modify_incar['LORBIT'] = 11
        
        # if we are doing LOBSTER, need special parameters
        # note: some of this gets handled later for us
        if configs['lobster_static'] and (configs['calc_to_run'] == 'static'):
            if configs['standard'] != 'mp':
                # want more DOS points
                # want to write charge densities
                # new NBANDS so don't want to start from WAVECAR
                lobster_incar_settings = {'NEDOS' : 4000,
                                          'ISTART' : 0,
                                          'LAECHG' : True}
                for key in lobster_incar_settings:
                    if key not in configs['lobster_incar']:
                        modify_incar[key] = lobster_incar_settings[key]
                
                for key in configs['lobster_incar']:
                    modify_incar[key] = configs['lobster_incar'][key]

                if not configs['lobster_kpoints']:
                    # need KPOINTS file for LOBSTER
                    modify_kpoints = {'length' : 25}
                else:
                    modify_kpoints = configs['lobster_kpoints']
        
        if configs['lobster_static']:
            if configs['xc_to_run'] == 'metagga':
                # gga-static will get ISYM = -1, so need to pass that to metagga relax otherwise WAVECAR from GGA doesnt help metagga
                modify_incar['ISYM'] = -1


        print('modify_incar = %s' % modify_incar)
        # initialize new VASPSet with all our settings   
        vasp_input = vaspset(structure, 
                             user_incar_settings=modify_incar, 
                             user_kpoints_settings=modify_kpoints, 
                             user_potcar_settings=modify_potcar, 
                             user_potcar_functional=potcar_functional,
                             validate_magmom=validate_magmom)
        
        return vasp_input
    
    @property
    def prepare_calc(self):
        """
        Write input files (INCAR, KPOINTS, POTCAR)
        """
        
        configs = self.configs.copy()
        calc_dir = self.calc_dir
        
        vasp_input = self.get_vasp_input
        if not vasp_input:
            return None

        # write input files
        vasp_input.write_input(calc_dir)
        
        # for LOBSTER, use Janine George's Lobsterin approach (mainly to get NBANDS)
        if (configs['lobster_static']) and (configs['calc_to_run'] == 'static'):
            INCAR_input = os.path.join(calc_dir, 'INCAR_input')
            INCAR_output = os.path.join(calc_dir, 'INCAR')
            copyfile(INCAR_output, INCAR_input)
            POSCAR_input = os.path.join(calc_dir, 'POSCAR')
            POTCAR_input = os.path.join(calc_dir, 'POTCAR')
            lobsterin = Lobsterin.standard_calculations_from_vasp_files(POSCAR_input=POSCAR_input,
                                                                        INCAR_input=INCAR_input,
                                                                        POTCAR_input=POTCAR_input,
                                                                        option='standard')
            lobsterin.write_lobsterin(os.path.join(calc_dir, 'lobsterin'))
            lobsterin.write_INCAR(incar_input=INCAR_input,
                                    incar_output=INCAR_output,
                                    poscar_input=POSCAR_input)
                
            
        return vasp_input
    
    @property
    def error_msgs(self):
        """
        Dict of {group of errors (str) : [list of error messages (str) in group]}
            - the error messages are things that VASP will write to fvaspout
            - we'll crawl fvaspout and assemble what errors made VASP fail,
                then we'll make edits to VASP calculation to clean them up for re-launch
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
        calc_dir = self.calc_dir
        configs = self.configs.copy()
        analyzer = AnalyzeVASP(calc_dir)
        outputs = VASPOutputs(calc_dir)
        unconverged = []
        
        # if calc is fully converged, return empty list (calc is done)
        if analyzer.is_converged:
            return unconverged
        
        # if vasprun doesnt exist, return empty list (calc errored out or didnt start yet)
        vr = outputs.vasprun
        if not vr:
            return unconverged
        
        # make sure last electronic loop converged in calc
        electronic_convergence = vr.converged_electronic
        
        # if we're relaxing the geometry, make sure last ionic loop converged
        if configs['calc_to_run'] == 'relax':
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
        out_file = os.path.join(self.calc_dir, self.configs['fvaspout'])
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
        True if no errors found and calc is fully converged, else False
        """
        configs = self.configs.copy()
        calc_dir = self.calc_dir
        clean = False
        if AnalyzeVASP(calc_dir).is_converged:
            clean = True
        if not os.path.exists(os.path.join(calc_dir, configs['fvaspout'])):
            clean = True
        if clean == True:
            with open(os.path.join(calc_dir, configs['fvasperrors']), 'w') as f:
                f.write('')
            return clean       
        errors = self.error_log + self.unconverged_log
        if len(errors) == 0:
            return True
        with open(os.path.join(calc_dir, configs['fvasperrors']), 'w') as f:
            for e in errors:
                f.write(e+'\n')
        return clean
    
    @property
    def incar_changes_from_errors(self):
        """
        Automatic INCAR changes based on errors
            - note: also may remove WAVECAR and/or CHGCAR as needed
            
        Returns {INCAR key (str) : INCAR value (str)}
        
        This will get passed to VASPSetUp the next time we launch (using SubmitTools)
        
        These error fixes are mostly taken from custodian (https://github.com/materialsproject/custodian/blob/809d8047845ee95cbf0c9ba45f65c3a94840f168/custodian/vasp/handlers.py)
            + a few of my own fixes I've added over the years
        """
        calc_dir = self.calc_dir
        errors = self.error_log
        unconverged_log = self.unconverged_log
        chgcar = os.path.join(calc_dir, 'CHGCAR')
        wavecar = os.path.join(calc_dir, 'WAVECAR')
        
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
        if 'nelm_too_low' in unconverged_log:
            incar_changes['NELM'] = 399
            incar_changes['ALGO'] = 'All'
        if 'nsw_too_low' in unconverged_log:
            incar_changes['NSW'] = 399
        if 'real_optlay' in errors:
            incar_changes['LREAL'] = False
        if 'bad_sym' in errors:
            incar_changes['ISYM'] = -1
        return incar_changes
        
def main():

    return 

if __name__ == '__main__':
    main()

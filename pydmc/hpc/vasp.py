from pydmc.core.mag import MagTools
from pydmc.hpc.analyze import AnalyzeVASP, VASPOutputs

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
    7) $ pmg config -add PMG_VASP_PSP_DIR FULL_PATH/bin/pymatgen_pot
    8) $ pmg config --add PMG_DEFAULT_FUNCTIONAL PBE_54

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
                 magmom=None,
                 fvaspout='vasp.o',
                 fvasperrors='errors.o',
                 lobster_static=True,
                 mag_override=False):
        """
        Args:
            calc_dir (os.PathLike) - directory where I want to execute VASP
                - there must be a POSCAR in calc_dir
                - the other input files will be added automatically within this code
            magmom (dict) - {'magmom' : [list of magmoms (float)]}
                - or None
                - only needed for AFM calculations where orderings would have been determined separately using MagTools
                
            The following will usually get overwritten by SubmitTools or LaunchTools using "config" files or user-defined configs:
                fvaspout (str) - name of file to write VASP output to
                    - generally no need to change
                    - in pydmc/data/data/_sub_configs.yaml
                fvasperrors (str) - name of file to write VASP errors to
                    - generally no need to change
                    - in pydmc/data/data/_sub_configs.yaml
                lobster_static (bool) - if True, run LOBSTER on static calculations
                    - note 1: this changes the way the static calculation is performed as LOBSTER needs certain settings
                    - note 2: this als runs Bader charge analysis
                    - in pydmc/data/data/_vasp_configs.yaml
                mag_override (bool) - allows user to run nonmagnetic calcs for magnetic systems and vice versa
                    - generally unadvisable to change
                    - in pydmc/data/data/_launch_configs.yaml
        Returns:
            calc_dir (os.PathLike) - directory where I want to execute VASP
            structure (pymatgen.Structure) - structure to be used for VASP calculation
                - note: raises error if no POSCAR in calc_dir
            magmom (dict) - {'magmom' : [list of magmoms (float)]}
                - or None
                - only needed for AFM calculations where orderings would have been determined using MagTools separately        
            fvaspout (str) - name of file to write VASP output to
            fvasperrors (str) - name of file to write VASP errors to
            lobster_static (bool) - if True, run LOBSTER on static calculations    
            mag_override (bool) - allows user to run nonmagnetic calcs for magnetic systems and vice versa
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
                      verbose=False,
                      **kwargs):
        """
        Args:
            standard (str) - defines the group of input settings (INCAR, KPOINTS, POTCAR) we'll use
                - unless comparing to Materials Project, 'dmc' will be our typical group standard
                - if comparing to Materials Project, 'mp' will be needed
                    - note: this takes care of itself in LaunchTools using the config "compare_to_mp" (in pydmc/data/_launch_configs)
                - when experimenting with other settings, specify whatever you'd like ('high_cutoff', 'slab', 'custom', 'strict_ediff')
                    - if it's not in ['mp', 'dmc'], it won't do anything to calc, but might be useful flag                    
                        - SubmitTools and LaunchTools will make note of standard and use that to define calculation (and calculation directory trees)                  
                    
            xc (str) - rung of Jacob's ladder
                - modifies the default in "fun"
                    - 'gga' (default) - default=Perdew-Burke-Ernzerhof (PBE)
                    - 'ggau' - default=PBE+U with MP U values
                    - 'metagga' - default=r2SCAN
                
                - note: if you set xc='metagga', SubmitTools will also run xc='gga'
                    - metagga's converge more quickly when preconditioned with gga, so you get the gga results for free
                
            calc (str) - type of calculation
                - currently implemented:
                    - 'loose' : loose geometry optimization (only 1 k point; easier convergence)
                    - 'relax' : geometry optimization (to find the low-energy crystal structure near the inputted structure)
                    - 'static' : static calculation (to get a more accurate energy/electronic structure at the equilibrium geometry)
                        - this is needed because the k-point grid is made based on the input structure, but 'relax' may change the structure
                            - performing a static calculation at the relaxed structure produces more sensible k-point grid

                - not implemented yet, but should be in the near term
                    - 'slab' : slab geometry optimization (ie for surfaces)
                    - 'neb' : NEB calculation (to find the minimum energy path (activation energy) between two structures)

                - not implemented yet, but would be nice to have eventually
                    - 'phonon' : phonon calculation (to get phonon dispersion curves)
                    - 'twod' : for 2D materials (this might fall under the umbrella of slab)
                    - 'interface' : for interfaces (this might be too complex to fall under slab umbrella)
                    - 'cluster' : for clusters (like nanoparticles)
                
            fun (str) - specify functional of interest
                - 'default'
                    - PBE if xc == 'gga' or 'ggau'
                    - r2SCAN if xc == 'metagga'
                    
            mag (str) - magnetic ordering type
                - 'nm' : nonmagnetic (ISPIN = 1)
                - 'fm' : ferromagnetic (ISPIN = 2)
                - 'afm_*' : antiferromagnetic (ISPIN = 2)
                    - note: magmom must be passed to VASPSetUp only for AFM calculations
                        - e.g., if many AFM orderings were generated using MagTools, tell VASPSetUp which one to use
                            - a general approach might be to first create a dictionary of these using MagTools.get_afm_magmoms (magmoms = {IDX : [magmoms]})
                    - code will read anything containing "afm" as AFM (e.g., "afm_1")
                        - most natural to use afm_0, afm_1, ... to specify different AFM orderings
                        
            modify_incar (dict) - user-defined incar settings
                - e.g., {'NCORE' : 4}
                - most settings should be specified as int, str, bool, or float, except (at least):
                    - {'MAGMOM' : [MAG_SITE1 MAG_SITE2 ...]}
                    - {LDAU' : [U_ION_1 U_ION_2 ...]}
                - see https://www.vasp.at/wiki/index.php/Category:INCAR_tag for all settings
                
            modify_kpoints (dict) - user-defined kpoint settings
                - a lot of (slightly confusing) options here
                    - Kpoints() means only use a single Kpoint
                    - {'length' : 25} is a pretty sensible default (currently for dmc)
                        - this is what VASP people tend to use, it seems
                    - Kpoints(kpts=[[a,b,c]]) for a axbxc grid
                    - {'reciprocal_density' : int} for automatic kpoint mesh
                        - note: this is not the clearest approach
                    - read more at https://www.vasp.at/wiki/index.php/KPOINTS

                - if not None:
                    - this will override KSPACING in INCAR (in addition to default kpoints settings)
                        - KSPACING is a new thing in VASP where you can set the KPOINTS in the INCAR instead
                            - I'm not too familiar/comfortable with this yet
                
            modify_potcar (dict) - user-defined potcar settings
                - e.g., {'Gd' : 'Gd_3'}
                    - different "flags" mean different e- treated as valence
                
            potcar_functional (str) - functional for POTCAR
                - note 1: not sure if changing this will break code (POTCARs in pmg are a mess)
                - note 2: I'm not sure a good reason why we would need to change this
                - note 3: I think I'm changing this now when standard = 'mp' to use VERY old POTCARs
                
            validate_magmom (bool) - VASPSet thing
                - note: setting to False because this was causing (as far as I could tell) non-useful errors
                - don't believe this is necessary for us
            
            verbose (bool) - print stuff
            
            **kwargs - additional arguments for VASPSet (see https://pymatgen.org/pymatgen.io.vasp.sets.html)
        """
        
        # most of these warnings should be unnecessary as these things are handled in LaunchTools/SubmitTools
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
        if standard != 'mp':
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
                
        # setting DMC standards --> what to do on top of MPRelaxSet or MPScanRelaxSet (pymatgen defaults)
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

            # for strict comparison to Materials Project GGA(+U) calculations, we need to use the old POTCARs
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
        if mag == 'nm':
            modify_incar['ISPIN'] = 1
        else:
            # make sure magnetization is written to OUTCAR for magnetic calcs
            modify_incar['LORBIT'] = 11
        
        # if we are doing LOBSTER, need special parameters
        # note: some of this gets handled later for us
        if self.lobster_static and (calc == 'static'):
            if standard != 'mp':
                modify_incar['NEDOS'] = 4000
                modify_incar['ISTART'] = 0
                modify_incar['LAECHG'] = True
                if not modify_kpoints:
                    # need KPOINTS file for LOBSTER
                    modify_kpoints = {'length' : 25}
        
        if self.lobster_static:
            if xc == 'metagga':
                # gga-static will get ISYM = -1, so need to pass that to metagga relax otherwise WAVECAR from GGA doesnt help metagga
                modify_incar['ISYM'] = -1

        # initialize new VASPSet with all our settings   
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

        # write input files
        vasp_input.write_input(self.calc_dir)
        
        # for LOBSTER, use Janine George's Lobsterin approach (mainly to get NBANDS)
        if self.lobster_static:
            outputs = VASPOutputs(self.calc_dir)
            if outputs.incar.as_dict()['NSW'] == 0:
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
        analyzer = AnalyzeVASP(self.calc_dir)
        outputs = VASPOutputs(self.calc_dir)
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
        True if no errors found and calc is fully converged, else False
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
            - note: also may remove WAVECAR and/or CHGCAR as needed
        Returns {INCAR key (str) : INCAR value (str)}
        
        This will get passed to VASPSetUp the next time we launch (using SubmitTools/LaunchTools)
        
        These error fixes are mostly taken from custodian (https://github.com/materialsproject/custodian/blob/809d8047845ee95cbf0c9ba45f65c3a94840f168/custodian/vasp/handlers.py)
            + a few of my own fixes I've added over the years
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

    return 

if __name__ == '__main__':
    main()

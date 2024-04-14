from pydmclab.core.mag import MagTools
from pydmclab.core.struc import StrucTools
from pydmclab.hpc.analyze import AnalyzeVASP, VASPOutputs
from pydmclab.data.configs import load_vasp_configs
from pydmclab.utils.handy import read_yaml, write_yaml, dotdict
from pydmclab.hpc.sets import GetSet

import os
import warnings
from shutil import copyfile

from pymatgen.io.vasp.sets import (
    MPRelaxSet,
    MPScanRelaxSet,
    MPHSERelaxSet,
    BadInputSetWarning,
)
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

    def __init__(self, calc_dir, prev_calc=None, user_configs={}):

        # this is where we will execute VASP
        self.calc_dir = calc_dir
        self.prev_calc = prev_calc

        # we should have a POSCAR in calc_dir already
        # e.g., LaunchTools will set this up for you
        fpos = os.path.join(calc_dir, "POSCAR")
        if not os.path.exists(fpos):
            raise FileNotFoundError("POSCAR not found in {}".format(calc_dir))
        else:
            structure = Structure.from_file(fpos)
            perturbation = user_configs["perturb_struc"]
            if perturbation:
                initial_structure = structure.copy()
                structure = StrucTools(initial_structure).perturb(perturbation)
            self.structure = structure

        self.default_configs = load_vasp_configs()
        self.user_configs = user_configs

    @property
    def configs(self):
        user_configs = self.user_configs
        xc_to_run = user_configs["xc_to_run"]
        calc_to_run = user_configs["calc_to_run"]

        relevant_mod_keys = [
            "-".join([xc, calc])
            for xc in [xc_to_run, "all"]
            for calc in [calc_to_run, "all"]
        ]

        for input_file in ["incar", "kpoints", "potcar"]:
            user_configs["modify_this_%s" % input_file] = {}
            for xc_calc in relevant_mod_keys:
                if xc_calc in user_configs["%s_mods" % input_file]:
                    user_configs["modify_this_%s" % input_file].update(
                        user_configs["%s_mods" % input_file][xc_calc]
                    )

        configs = {**self.default_configs, **user_configs}
        return configs.copy()

    @property
    def get_vaspset(self):
        """
        Returns:
            vasp_input (pymatgen.io.vasp.sets.VaspInputSet)

        Starting from a pymatgen
        Uses configs to modify pymatgen's VaspInputSets as required
        """

        # copy configs to prevent unwanted updates
        configs = self.configs.copy()

        # initialize how we're going to modify each vasp input file with configs specs

        modify_incar = configs["modify_this_incar"]
        modify_kpoints = configs["modify_this_kpoints"]
        modify_potcar = configs["modify_this_potcar"]

        # initialize potcar functional
        potcar_functional = configs["potcar_functional"]

        # this should be kept off in general, gives unuseful warnings (I think)
        validate_magmom = configs["validate_magmom"]

        structure = self.structure

        xc_calc = "-".join([configs["xc_to_run"], configs["calc_to_run"]])

        # add MAGMOM to structure
        if configs["mag"] == "nm":
            # if non-magnetic, MagTools takes care of this
            structure = MagTools(structure).get_nonmagnetic_structure
        elif configs["mag"] == "fm":
            # if ferromagnetic, MagTools takes care of this
            structure = MagTools(structure).get_ferromagnetic_structure
        elif "afm" in configs["mag"]:
            # if antiferromagnetic, we need to aprovide a MAGMOM
            magmom = configs["magmom"]
            if not magmom:
                raise ValueError("you must specify a magmom for an AFM calculation\n")
            if (min(magmom) >= 0) and (max(magmom) <= 0):
                raise ValueError(
                    "provided magmom that is not AFM, but you are trying to run an AFM calculation\n"
                )
            structure.add_site_property("magmom", magmom)

        vaspset = GetSet(
            structure=structure,
            configs=configs,
            potcar_functional=potcar_functional,
            validate_magmom=validate_magmom,
            modify_incar=modify_incar,
            modify_kpoints=modify_kpoints,
            modify_potcar=modify_potcar,
        ).vaspset

        return vaspset

    @property
    def prepare_calc(self):
        """
        Write input files (INCAR, KPOINTS, POTCAR)
        """

        warnings.filterwarnings("ignore", category=BadInputSetWarning)

        configs = self.configs.copy()
        calc_dir = self.calc_dir

        vaspset = self.get_vaspset
        if not vaspset:
            return None

        # write input files
        vaspset.write_input(calc_dir)

        # for LOBSTER, use Janine George's Lobsterin approach (mainly to get NBANDS)
        if configs["calc_to_run"] in ["lobster", "bs"]:
            INCAR_input = os.path.join(calc_dir, "INCAR_input")
            INCAR_output = os.path.join(calc_dir, "INCAR")
            copyfile(INCAR_output, INCAR_input)
            POSCAR_input = os.path.join(calc_dir, "POSCAR_input")
            POSCAR_output = os.path.join(calc_dir, "POSCAR")
            copyfile(POSCAR_output, POSCAR_input)
            KPOINTS_output = os.path.join(calc_dir, "KPOINTS")
            POTCAR_input = os.path.join(calc_dir, "POTCAR_input")
            POTCAR_output = os.path.join(calc_dir, "POTCAR")
            copyfile(POTCAR_output, POTCAR_input)

            if configs["calc_to_run"] == "lobster":

                lobsterin = Lobsterin.standard_calculations_from_vasp_files(
                    POSCAR_input=POSCAR_input,
                    INCAR_input=INCAR_input,
                    POTCAR_input=POTCAR_input,
                    option="standard",
                )

                lobsterin_dict = lobsterin.as_dict()

                lobsterin_dict["COHPSteps"] = configs["COHPSteps"]
                lobsterin = Lobsterin.from_dict(lobsterin_dict)

            elif configs["calc_to_run"] == "bs":
                lobsterin = Lobsterin
                lobsterin.write_POSCAR_with_standard_primitive(
                    POSCAR_input=POSCAR_input,
                    POSCAR_output=POSCAR_output,
                    symprec=configs["bs_symprec"],
                )
                try:
                    lobsterin.write_KPOINTS(
                        POSCAR_input=POSCAR_output,
                        KPOINTS_output=KPOINTS_output,
                        line_mode=True,
                        symprec=configs["bs_symprec"],
                        kpoints_line_density=configs["bs_line_density"],
                    )
                except ValueError:
                    print("trying higher symprec")
                    lobsterin.write_KPOINTS(
                        POSCAR_input=POSCAR_output,
                        KPOINTS_output=KPOINTS_output,
                        line_mode=True,
                        symprec=configs["bs_symprec"] * 2,
                        kpoints_line_density=configs["bs_line_density"],
                    )

                lobsterin = Lobsterin.standard_calculations_from_vasp_files(
                    POSCAR_input=POSCAR_output,
                    INCAR_input=INCAR_input,
                    POTCAR_input=POTCAR_output,
                    option="standard_with_fatband",
                )
            flobsterin = os.path.join(calc_dir, "lobsterin")
            lobsterin.write_lobsterin(flobsterin)

        return vaspset

    @property
    def error_msgs(self):
        """
        Dict of {group of errors (str) : [list of error messages (str) in group]}
            - the error messages are things that VASP will write to fvaspout
            - we'll crawl fvaspout and assemble what errors made VASP fail,
                then we'll make edits to VASP calculation to clean them up for re-launch
        """
        return {
            "tet": [
                "Tetrahedron method fails for NKPT<4",
                "Fatal error detecting k-mesh",
                "Fatal error: unable to match k-point",
                "Routine TETIRR needs special values",
                "Tetrahedron method fails (number of k-points < 4)",
            ],
            "inv_rot_mat": [
                "inverse of rotation matrix was not found (increase " "SYMPREC)"
            ],
            "brmix": ["BRMIX: very serious problems"],
            "subspacematrix": ["WARNING: Sub-Space-Matrix is not hermitian in " "DAV"],
            "tetirr": ["Routine TETIRR needs special values"],
            "incorrect_shift": ["Could not get correct shifts"],
            "real_optlay": ["REAL_OPTLAY: internal error", "REAL_OPT: internal ERROR"],
            "rspher": ["ERROR RSPHER"],
            "dentet": ["DENTET"],
            "too_few_bands": ["TOO FEW BANDS"],
            "triple_product": ["ERROR: the triple product of the basis vectors"],
            "rot_matrix": ["Found some non-integer element in rotation matrix"],
            "brions": ["BRIONS problems: POTIM should be increased"],
            "pricel": ["internal error in subroutine PRICEL"],
            "zpotrf": ["LAPACK: Routine ZPOTRF failed"],
            "amin": ["One of the lattice vectors is very long (>50 A), but AMIN"],
            "zbrent": [
                "ZBRENT: fatal internal in",
                "ZBRENT: fatal error in bracketing",
            ],
            "pssyevx": ["ERROR in subspace rotation PSSYEVX"],
            "eddrmm": ["WARNING in EDDRMM: call to ZHEGV failed"],
            "edddav": ["Error EDDDAV: Call to ZHEGV failed"],
            "grad_not_orth": ["EDWAV: internal error, the gradient is not orthogonal"],
            "nicht_konv": ["ERROR: SBESSELITER : nicht konvergent"],
            "zheev": ["ERROR EDDIAG: Call to routine ZHEEV failed!"],
            "elf_kpar": ["ELF: KPAR>1 not implemented"],
            "elf_ncl": ["WARNING: ELF not implemented for non collinear case"],
            "rhosyg": ["RHOSYG internal error"],
            "posmap": ["POSMAP internal error: symmetry equivalent atom not found"],
            "point_group": ["Error: point group operation missing"],
            "ibzkpt": ["internal error in subroutine IBZKPT"],
            "bad_sym": [
                "ERROR: while reading WAVECAR, plane wave coefficients changed"
            ],
            "num_prob": ["num prob"],
            "sym_too_tight": ["try changing SYMPREC"],
            "coef": ["while reading plane", "while reading WAVECAR"],
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

        # if vasprun doesnt exist, return empty list (calc errored out or didnt start yet)
        vr = outputs.vasprun
        if not vr:
            return unconverged

        Etot = analyzer.E_per_at
        if Etot and (Etot > 0):
            unconverged.append("Etot_positive")

        if ("static" in calc_dir) and os.path.exists(
            calc_dir.replace("static", "relax")
        ):
            relax_dir = calc_dir.replace("static", "relax")
            E_relax = AnalyzeVASP(relax_dir).E_per_at
            if E_relax:
                if Etot:
                    if abs(E_relax - Etot) > 0.1:
                        unconverged.append("static_energy_changed_alot")

        # if calc is fully converged, return empty list (calc is done)
        if analyzer.is_converged:
            return unconverged

        # make sure last electronic loop converged in calc
        electronic_convergence = vr.converged_electronic

        # if we're relaxing the geometry, make sure last ionic loop converged
        if configs["calc_to_run"] == "relax":
            ionic_convergence = vr.converged_ionic
        else:
            ionic_convergence = True

        if not electronic_convergence:
            unconverged.append("nelm_too_low")
        if not ionic_convergence:
            unconverged.append("nsw_too_low")

        return unconverged

    @property
    def error_log(self):
        """
        Parse fvaspout for error messages

        Returns list of errors (str)
        """
        error_msgs = self.error_msgs
        out_file = os.path.join(self.calc_dir, self.configs["fvaspout"])
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
        if not os.path.exists(os.path.join(calc_dir, configs["fvaspout"])):
            clean = True
        if clean == True:
            with open(os.path.join(calc_dir, configs["fvasperrors"]), "w") as f:
                f.write("")
            return clean
        errors = self.error_log + self.unconverged_log
        if len(errors) == 0:
            return True
        with open(os.path.join(calc_dir, configs["fvasperrors"]), "w") as f:
            for e in errors:
                f.write(e + "\n")
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
        chgcar = os.path.join(calc_dir, "CHGCAR")
        wavecar = os.path.join(calc_dir, "WAVECAR")

        curr_incar = Incar.from_file(os.path.join(calc_dir, "INCAR")).as_dict()

        incar_changes = {}
        if "Etot_positive" in unconverged_log:
            incar_changes["ALGO"] = "All"
        if "static_energy_changed_alot" in unconverged_log:
            incar_changes["ALGO"] = "All"
        if "grad_not_orth" in errors:
            incar_changes["SIGMA"] = 0.05
            if os.path.exists(wavecar):
                os.remove(wavecar)
            incar_changes["ALGO"] = "Exact"
        if "edddav" in errors:
            incar_changes["ALGO"] = "All"
            if os.path.exists(chgcar):
                os.remove(chgcar)
        if "eddrmm" in errors:
            if os.path.exists(wavecar):
                os.remove(wavecar)
            incar_changes["ALGO"] = "Normal"
        if "subspacematrix" in errors:
            incar_changes["LREAL"] = False
            incar_changes["PREC"] = "Accurate"
        if "inv_rot_mat" in errors:
            incar_changes["SYMPREC"] = 1e-8
        if "zheev" in errors:
            incar_changes["ALGO"] = "Exact"
        if "pssyevx" in errors:
            incar_changes["ALGO"] = "Normal"
        if "zpotrf" in errors:
            incar_changes["ISYM"] = -1
        if "zbrent" in errors:
            incar_changes["IBRION"] = 1
        if "brmix" in errors:
            incar_changes["IMIX"] = 1
        if "ibzkpt" in errors:
            incar_changes["SYMPREC"] = 1e-10
            incar_changes["ISMEAR"] = 0
            incar_changes["ISYM"] = -1
        if "posmap" in errors:
            incar_changes["SYMPREC"] = 1e-5
            incar_changes["ISMEAR"] = 0
            incar_changes["ISYM"] = -1
        if "nelm_too_low" in unconverged_log:
            if "NELM" in curr_incar:
                prev_nelm = curr_incar["NELM"]
            else:
                prev_nelm = 100
            incar_changes["NELM"] = prev_nelm + 100
            incar_changes["ALGO"] = "All"
        if "nsw_too_low" in unconverged_log:
            if "NSW" in curr_incar:
                prev_nsw = curr_incar["NSW"]
            else:
                prev_nsw = 199
            incar_changes["NSW"] = prev_nsw + 100
        if "real_optlay" in errors:
            incar_changes["LREAL"] = False
        if "bad_sym" in errors:
            incar_changes["ISYM"] = -1
        if "amin" in errors:
            incar_changes["AMIN"] = 0.01
        if "pricel" in errors:
            incar_changes["SYMPREC"] = 1e-8
            incar_changes["ISYM"] = 0
        if "num_prob" in errors:
            incar_changes["ISMEAR"] = -1
        if "sym_too_tight" in errors:
            incar_changes["ISYM"] = -1
            if "SYMPREC" in curr_incar:
                prev_symprec = curr_incar["SYMPREC"]
            else:
                prev_symprec = 1e-6
            new_symprec = prev_symprec / 10
            incar_changes["SYMPREC"] = new_symprec
            # incar_changes["SYMPREC"] = 1e-3
        if "coef" in errors:
            if os.path.exists(wavecar):
                os.remove(wavecar)
        return incar_changes


def main():
    return


if __name__ == "__main__":
    main()

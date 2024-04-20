import os
import warnings
from shutil import copyfile

from pymatgen.core.structure import Structure
from pymatgen.io.vasp.inputs import Incar
from pymatgen.io.vasp.sets import BadInputSetWarning
from pymatgen.io.lobster.inputs import Lobsterin

from pydmclab.core.mag import MagTools
from pydmclab.hpc.analyze import AnalyzeVASP, VASPOutputs
from pydmclab.data.configs import load_base_configs
from pydmclab.hpc.sets import GetSet


class VASPSetUp(object):
    """
    Use to write VASP inputs for a single VASP calculation
        a calculation here might be defined as the same:
            initial structure
            initial magnetic configurations
            input settings (INCAR, KPOINTS, POTCAR)
            etc.

    Also changes inputs based on errors that are encountered

    Note that we rarely need to call this class directly
        instead we'll manage things through pydmclab.hpc.submit.SubmitTools and pydmclab.hpc.launch.LaunchTools

    Using this will require configuring your POTCARs to work with pymatgen.
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

    def __init__(self, calc_dir, user_configs=None):
        """
        Args:
            calc_dir (str):
                path to directory where VASP calculation will be run

            user_configs (bool or dict):
                if None, will use default configs
                if dict, will use user_configs
                    these will override any default configs
                    see pydmclab.data.data._hpc_configs for defaults that can be changed
                        relevant configs to this class are those that affect "VASP_CONFIGS"
        """
        # proper handling for default arg that could be dict
        if user_configs is None:
            user_configs = {}

        # this is where we will execute VASP
        self.calc_dir = calc_dir

        # we should have a POSCAR in calc_dir already
        # e.g., LaunchTools will set this up for you
        fpos = os.path.join(calc_dir, "POSCAR")
        if not os.path.exists(fpos):
            raise FileNotFoundError("POSCAR not found in {}".format(calc_dir))
        else:
            structure = Structure.from_file(fpos)
            self.structure = structure

        # load base configs from pydmclab.data.data._hpc_configs
        self.default_configs = load_base_configs()

        # load user configs
        self.user_configs = user_configs

    @property
    def configs(self):
        """
        Blends default configs with user_configs
            does some work with incar_mods, kpoints_mods, and potcar_mods
                basically, you will have passed these dicts for many calculation types
                but your are only setting up VASP for one of those. this will retrieve the mods for this particular calculation
        """
        user_configs = self.user_configs

        # what kind of xc (gga, metagga, etc.)
        xc_to_run = user_configs["xc_to_run"]

        # what kind of calc (static, relax, dfpt, etc.)
        calc_to_run = user_configs["calc_to_run"]

        # relevant xc-calcs are this particula xc-calc or if all is specified for xc and/or calc
        relevant_mod_keys = [
            "-".join([xc, calc])
            for xc in [xc_to_run, "all"]
            for calc in [calc_to_run, "all"]
        ]

        # loop through incar, kpoints, potcar, and retrieve any relevant mods that should apply on top of defaults
        for input_file in ["incar", "kpoints", "potcar"]:
            if user_configs["%s_mods" % input_file] is None:
                user_configs["%s_mods" % input_file] = {}
            user_configs["modify_this_%s" % input_file] = {}
            for xc_calc in relevant_mod_keys:
                if xc_calc in user_configs["%s_mods" % input_file]:
                    user_configs["modify_this_%s" % input_file].update(
                        user_configs["%s_mods" % input_file][xc_calc]
                    )

        # blend default configs with user_configs, giving user_configs priority
        configs = {**self.default_configs, **user_configs}
        return configs.copy()

    @property
    def get_vaspset(self):
        """
        Returns:
            vasp_input (pymatgen.io.vasp.sets.VaspInputSet)

        Start from pydmclab.hpc.sets.GetSet and apply user_configs on top
        """

        # copy configs to prevent unwanted updates
        configs = self.configs.copy()

        # initialize how we're going to modify each vasp input file with configs specs
        modify_incar = configs["modify_this_incar"]
        modify_kpoints = configs["modify_this_kpoints"]
        modify_potcar = configs["modify_this_potcar"]

        # initialize potcar functional
        potcar_functional = configs["potcar_functional"]

        # this should be kept off in general, gives unhelpful warnings (I think)
        validate_magmom = configs["validate_magmom"]

        # need the structure to process magnetism
        structure = self.structure

        # add MAGMOM to structure
        if configs["mag"] == "nm":
            # if non-magnetic, MagTools takes care of this (no MAGMOM)
            structure = MagTools(structure).get_nonmagnetic_structure
        elif configs["mag"] == "fm":
            # if ferromagnetic, MagTools takes care of this (mag elements get MAGMOM = 5, nonmag get MAGMOM = 0.6 by default)
            structure = MagTools(structure).get_ferromagnetic_structure
        elif "afm" in configs["mag"]:
            # if antiferromagnetic, we need to aprovide a MAGMOM (presumably you ran MagTools to generate AFM configurations already)
            magmom = configs["magmom"]
            if not magmom:
                raise ValueError("you must specify a magmom for an AFM calculation\n")
            if (min(magmom) >= 0) and (max(magmom) <= 0):
                raise ValueError(
                    "provided magmom that is not AFM, but you are trying to run an AFM calculation\n"
                )
            # add this AFM magmom to our Structure
            structure.add_site_property("magmom", magmom)

        # get the VaspInputSet
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
        # ignore pymatgen warnings about bad input sets. we should be handling the ones that matter to us already
        warnings.filterwarnings("ignore", category=BadInputSetWarning)

        # get our configs dict
        configs = self.configs.copy()

        # where is VASP going to be run
        calc_dir = self.calc_dir

        # get our Vasp input set
        vaspset = self.get_vaspset

        # if something went awry and we don't have a VASP input set, don't set up the calculation
        if not vaspset:
            return None

        # write input files
        vaspset.write_input(calc_dir)

        # for LOBSTER, use pymatnge io lobseter to get lobsterin
        if configs["calc_to_run"] in ["lobster", "bs"]:

            poscar = os.path.join(calc_dir, "POSCAR")
            incar = os.path.join(calc_dir, "INCAR")
            potcar = os.path.join(calc_dir, "POTCAR")
            kpoints = os.path.join(calc_dir, "KPOINTS")

            # if doing regular LOBSTER (ie no bandstructure), just need a Lobsterin file. all other settings get taken care of for us (through Sets or Passer)
            if configs["calc_to_run"] == "lobster":

                lobsterin = Lobsterin.standard_calculations_from_vasp_files(
                    POSCAR_input=poscar,
                    INCAR_input=incar,
                    POTCAR_input=potcar,
                    option="standard",
                )

                lobsterin_dict = lobsterin.as_dict()

                # adjust COHPSteps based on how fine of a COHP/DOS user wants
                lobsterin_dict["COHPSteps"] = configs["COHPSteps"]
                lobsterin = Lobsterin.from_dict(lobsterin_dict)

            # if getting bandstructure, need Lobsterin to do more work for us
            elif configs["calc_to_run"] == "bs":
                lobsterin = Lobsterin

                # get a primitive cell but save the original cell as POSCAR_input
                poscar_input = os.path.join(calc_dir, "POSCAR_input")
                copyfile(poscar, poscar_input)
                lobsterin.write_POSCAR_with_standard_primitive(
                    POSCAR_input=poscar_input,
                    POSCAR_output=poscar,
                    symprec=configs["bs_symprec"],
                )

                # get a KPOINTS path but save original KPOINTS as KPOINTS_input
                kpoints_input = os.path.join(calc_dir, "KPOINTS_input")
                copyfile(kpoints, kpoints_input)
                try:
                    lobsterin.write_KPOINTS(
                        POSCAR_input=poscar,
                        KPOINTS_output=kpoints,
                        line_mode=True,
                        symprec=configs["bs_symprec"],
                        kpoints_line_density=configs["bs_line_density"],
                    )
                except ValueError:
                    # NOTE: this is not a great fix..
                    # sometimes we can't find a path with the given symprec
                    print("trying higher symprec")
                    lobsterin.write_KPOINTS(
                        POSCAR_input=poscar,
                        KPOINTS_output=kpoints,
                        line_mode=True,
                        symprec=configs["bs_symprec"] * 2,
                        kpoints_line_density=configs["bs_line_density"],
                    )

                # get our lobsterin, specifying that we want a fatband calculation
                lobsterin = Lobsterin.standard_calculations_from_vasp_files(
                    POSCAR_input=poscar,
                    INCAR_input=incar,
                    POTCAR_input=potcar,
                    option="standard_with_fatband",
                )

            # write lobsterin file to calc_dir so LOBSTER can be executed
            flobsterin = os.path.join(calc_dir, "lobsterin")
            lobsterin.write_lobsterin(flobsterin)

        return vaspset

    @property
    def error_msgs(self):
        """
        Returns:
            dict of {group of errors (str) : [list of error messages (str) in group]}

        the error messages are things that VASP will write to fvaspout
        we'll crawl fvaspout and assemble what errors made VASP fail,
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
        Returns:
            list of str messages

        checks to see if both ionic and electronic convergence have been reached
            if calculation had NELM # electronic steps, electronic convergence may not be met
            if calculation had NSW # ionic steps, ionic convergence may not be met
            if the final energy is positive, then calculation is definitely not converged
            if the final energy changes a lot between static and relax, then static calculation is deemed not converged

        NOTE: if calculation hasn't started or errored out, this will return an empty list
            the purpose of this method is to detect calculations that *look* converged (Elaps in OUTCAR) but aren't actually

        if 'nelm_too_low' in unconverged:
            the calculation didn't reach electronic convergence
        if 'nsw_too_low' in unconverged:
            the calculation didn't reach ionic convergence
        if 'Etot_positive' in unconverged:
            the final energy is positive
        if 'static_energy_changed_alot' in unconverged:
            the final energy changed a lot between static and relax (will force static to be recalculated)
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

        # get final energy. if it exists (calc looks converged) and is positive, add to unconverged
        Etot = analyzer.E_per_at
        if Etot and (Etot > 0):
            unconverged.append("Etot_positive")

        # for static calculations,
        # if relax calculation exists, compare energies
        # if they differ by more than specified tolerance (default = 0.1 eV/atom), add to unconverged
        # if we specified a tolerance for this
        if (
            ("static" in calc_dir)
            and os.path.exists(calc_dir.replace("static", "relax"))
            and configs["relax_static_energy_diff_tol"]
        ):
            relax_dir = calc_dir.replace("static", "relax")
            E_relax = AnalyzeVASP(relax_dir).E_per_at

            # make sure relax has an energy
            if E_relax:
                # make sure static has an energy
                if Etot:
                    # compare the two; if too high, call static unconverged
                    if abs(E_relax - Etot) > configs["relax_static_energy_diff_tol"]:
                        unconverged.append("static_energy_changed_alot")

        # if calc is fully converged (ionically and electronically), return empty list (calc is done)
        if analyzer.is_converged:
            return unconverged

        # make sure last electronic loop is converged
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
        Returns
            list of errors (str)

        Parse fvaspout (calc_dir/vasp.o) for error messages

        """
        error_msgs = self.error_msgs
        out_file = os.path.join(self.calc_dir, self.configs["fvaspout"])

        # if no vasp.o file, return empty list
        if not os.path.exists(out_file):
            return []

        # accumulate list of errors found in vasp.o
        errors = []
        with open(out_file, "r", encoding="utf-8") as f:
            contents = f.read()
        for error_tag, error_text_list in error_msgs.items():
            for error_text in error_text_list:
                if error_text in contents:
                    errors.append(error_tag)
        return errors

    @property
    def all_errors(self):
        """
        Returns:
            list of str messages

        combines error_log and unconverged_log
        """
        return self.error_log + self.unconverged_log

    @property
    def is_clean(self):
        """
        Returns:
            True if no errors found and calc is fully converged, else False
        """
        configs = self.configs.copy()
        calc_dir = self.calc_dir

        # start by assuming calc is not clean
        clean = False

        # combine error lists (from vasp.o and false convergence)
        errors = self.all_errors

        # if calc is converged and no errors, then it's clean
        if AnalyzeVASP(calc_dir).is_converged and (len(errors) == 0):
            clean = True

        # if calc hasn't been executed yet, it's clean
        elif not os.path.exists(os.path.join(calc_dir, configs["fvaspout"])):
            clean = True

        # otherwise, it's not clean
        else:
            clean = False

        if clean is True:
            # write empty errors.o file
            with open(
                os.path.join(calc_dir, configs["fvasperrors"]), "w", encoding="utf-8"
            ) as f:
                f.write("")
            return clean
        elif clean is False:
            # populate errors.o file with errors so they can be corrected
            with open(
                os.path.join(calc_dir, configs["fvasperrors"]), "w", encoding="utf-8"
            ) as f:
                for e in errors:
                    f.write(e + "\n")
            return clean
        return clean

    @property
    def incar_changes_from_errors(self):
        """
        Returns
            {INCAR key (str) : INCAR value (str)}

        Automatic INCAR changes based on errors
            note: also may remove WAVECAR and/or CHGCAR as needed

        This will get passed to VASPSetUp the next time we launch (using SubmitTools)

        These error fixes are mostly taken from custodian (https://github.com/materialsproject/custodian/blob/809d8047845ee95cbf0c9ba45f65c3a94840f168/custodian/vasp/handlers.py)
            + a few of my own fixes I've added over the years

        NOTE: not clear if the order I apply these has the wrong effect in some circumstances
        """

        configs = self.configs.copy()
        calc_dir = self.calc_dir
        errors = self.all_errors

        chgcar = os.path.join(calc_dir, "CHGCAR")
        wavecar = os.path.join(calc_dir, "WAVECAR")

        # get current settings so that fix is not redundant
        curr_incar = Incar.from_file(os.path.join(calc_dir, "INCAR")).as_dict()

        incar_changes = {}
        if "Etot_positive" in errors:
            incar_changes["ALGO"] = "All"
        if "static_energy_changed_alot" in errors:
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
            if "ALGO" not in incar_changes:
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
        if "nelm_too_low" in errors:
            # default behavior is to add 100 to NELM
            # you can specify flexible convergence criteria, in which case we'll change EDIFF
            if "NELM" in curr_incar:
                prev_nelm = curr_incar["NELM"]
            else:
                prev_nelm = 100
            if (prev_nelm > 300) and configs["flexible_convergence_criteria"]:
                print("electronic relaxation troubles. changing EDIFF")
                prev_ediff = curr_incar["EDIFF"]
                incar_changes["EDIFF"] = prev_ediff * 10
            incar_changes["NELM"] = prev_nelm + 100
            incar_changes["ALGO"] = "All"
        if "nsw_too_low" in errors:
            # default behavior is to add 100 to NSW
            # you can specify flexible convergence criteria, in which case we'll change EDIFFG
            if "NSW" in curr_incar:
                prev_nsw = curr_incar["NSW"]
            else:
                prev_nsw = 199
            if (prev_nsw > 299) and configs["flexible_convergence_criteria"]:
                print("ionic relaxation troubles. changing EDIFF")
                prev_ediffg = curr_incar["EDIFFG"]
                if prev_ediffg > 0:
                    incar_changes["EDIFFG"] = prev_ediffg * 10
                elif prev_ediffg < 0:
                    incar_changes["EDIFFG"] = prev_ediffg * 2
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
        if "coef" in errors:
            if os.path.exists(wavecar):
                os.remove(wavecar)
        return incar_changes


def main():
    """
    Execute code here if debugging this file
    """
    return


if __name__ == "__main__":
    main()

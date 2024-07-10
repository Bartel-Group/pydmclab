import sys
import os
import json
from shutil import copyfile
import numpy as np
import traceback

from pymatgen.io.vasp.inputs import Incar, Poscar, Kpoints
from pymatgen.io.vasp.sets import get_structure_from_prev_run

from pydmclab.hpc.analyze import AnalyzeVASP
from pydmclab.core.struc import StrucTools


class Passer(object):
    """
    This class is made to be executed on compute nodes (ie it gets called between VASP calls for a series of jobs that are packed together)

    The idea is to run this to figure out how to pass stuff from freshly converged calcs to subsequent calcs

    The main things we care about are:
        changing smearing (ISMEAR, SIGMA) based on band gap
        changing kpoints (KSPACING) based on band gap
        passing CONTCAR --> POSCAR
        passing WAVECAR
        passing optimized magnetic moments as initial guesses (MAGMOM)
        passing NBANDS for lobster

    Can be customized to do whatever you'd like between calculations that are chained together in your calc_list
    """

    def __init__(self, passer_dict_as_str):
        """
        Args:
            passer_dict_as_str (str):
                a json string that contains the following keys
            xc_calc (str):
                the current xc-calculation type (eg "gga-static")
            calc_list (list):
                the list of all xc-calculation types that have been run (eg ["gga-static", "gga-relax", "gga-lobster"])
            calc_dir (str):
                the directory of the current calculation
            incar_mods (dict):
                a dictionary of user-defined INCAR modifications that apply to the recipient of the passing
            launch_dir (str):
                the directory from which the job was launched
            struc_src_for_hse (str):
                the source of the structure for hse calculations (eg "metagga-relax")
        """
        passer_dict = json.loads(passer_dict_as_str)

        self.xc_calc = passer_dict["xc_calc"]
        self.calc_list = passer_dict["calc_list"]
        self.calc_dir = passer_dict["calc_dir"]
        self.incar_mods = passer_dict["incar_mods"]
        self.launch_dir = passer_dict["launch_dir"]
        self.struc_src_for_hse = passer_dict["struc_src_for_hse"]

    @property
    def poscar(self):
        """
        Returns:
            the structure of the current calculation
        """
        return Poscar.from_file(os.path.join(self.calc_dir, "POSCAR"))
    
    @property
    def errors_encountered_in_curr_calc(self):
        """
        Returns:
            get all errors present in errors.o file so can augment passer behavior as needed
        """

        errors_o = os.path.join(self.calc_dir, "errors.o")

        if not os.path.exists(errors_o):
            return None

        with open(errors_o, "r", encoding="utf-8") as f:
            return [line.strip() for line in f]

    @property
    def prev_xc_calc(self):
        """
        Returns:
            the parent xc_calc (eg 'gga-relax') that should pass stuff to the present xc_calc (eg 'gga-static)
        """

        calc_list = self.calc_list

        curr_xc_calc = self.xc_calc
        struc_src_for_hse = self.struc_src_for_hse
        curr_xc, curr_calc = curr_xc_calc.split("-")
        if curr_calc == "loose":
            # just setting some dummy thing b/c nothing should come before loose
            prev_xc_calc = curr_xc_calc.replace(curr_calc, "pre_loose")
            return prev_xc_calc

        if curr_calc == "relax":
            # for gga/gga+u, inherit from loose if it exists, otherwise don't inherit
            if curr_xc in ["gga", "ggau"]:
                prev_xc_calc = curr_xc_calc.replace(curr_calc, "loose")
            # for metagga/hse, inherit from gga
            else:
                prev_xc_calc = curr_xc_calc.replace(curr_xc, "gga")
            return prev_xc_calc

        if curr_calc == "static":
            # static calcs inherit from relax

            if curr_xc == "hse06":
                # for hse06-static, inherit from relax if it exists, otherwise inherit from preggastatic
                if "hse06-relax" not in calc_list:
                    return curr_xc_calc.replace(curr_calc, "preggastatic")
            prev_xc_calc = curr_xc_calc.replace(curr_calc, "relax")
            return prev_xc_calc

        if curr_xc == "hse06":
            if curr_calc == "preggastatic":
                # for hse06-preggastatic, inherit from the source structure selected by the user
                prev_xc_calc = struc_src_for_hse
                return prev_xc_calc
            if curr_calc != "parchg":
                # for hse06-parchg, inherit from hse06-static; for other addons in hse06, inherit from preggastatic
                prev_xc_calc = curr_xc_calc.replace(curr_calc, "preggastatic")
                return prev_xc_calc
        
        if curr_calc == "parchg":
            # for parchg, inherit from lobster
            return curr_xc_calc.replace(curr_calc, "lobster")

        # everything else inherits from static
        return curr_xc_calc.replace(curr_calc, "static")
    
    @property
    def prev_xc(self):
        """
        Returns:
            xc (str) for parent calculation
                take the parent xc_calc and split it to get the exchange functional type
        """
        prev_xc_calc = self.prev_xc_calc
        return prev_xc_calc.split('-')[0]
    
    @property
    def prev_calc(self):
        """
        Returns:
            xc (str) for parent calculation
                take the parent xc_calc and split it to get the calculation type
        """
        prev_xc_calc = self.prev_xc_calc
        return prev_xc_calc.split('-')[1]
    
    @property
    def curr_xc(self):
        """
        Returns:
            xc (str) for current calculation
                take the current xc_calc and split it to get the exchange functional type
        """
        curr_xc_calc = self.xc_calc
        return curr_xc_calc.split('-')[0]
    
    @property
    def curr_calc(self):
        """
        Returns:
            calc (str) for current calculation
                take the current xc_calc and split it to get the calculation type
        """
        curr_xc_calc = self.xc_calc
        return curr_xc_calc.split('-')[1]
    
    @property
    def prev_calc_dir(self):
        """
        Returns:
            calc_dir (str) for parent calculation
                take the current calc_dir and replace the current xc_calc with the previous xc_calc
        """
        calc_dir = self.calc_dir
        curr_xc_calc = self.xc_calc
        prev_xc_calc = self.prev_xc_calc
        return calc_dir.replace(curr_xc_calc, prev_xc_calc)

    @property
    def prev_calc_convergence(self):
        """
        Returns:
            True if parent is converged else False
        """
        prev_calc_dir = self.prev_calc_dir
        if not os.path.exists(prev_calc_dir):
            return False
        return AnalyzeVASP(prev_calc_dir).is_converged

    @property
    def kill_job(self):
        """
        Returns:
            True if child should not be launched
                if parent is not converged
            False if child should be launched
                parent doesn't exist (ie nothing to inherit)
                parent is converged
        """
        calc_list = self.calc_list
        prev_xc_calc = self.prev_xc_calc
        if prev_xc_calc not in calc_list:
            return False
        prev_calc_convergence = self.prev_calc_convergence
        if not prev_calc_convergence:
            return True
        return False
    
    @property
    def is_curr_calc_being_restarted(self):

        calc_dir = self.calc_dir
        curr_contcar = os.path.join(calc_dir, "CONTCAR")
        if not os.path.exists(curr_contcar):
            return False
        with open(curr_contcar, "r", encoding="utf-8") as f:
            contents = f.read()
            if len(contents) > 0:
                return True
            else:
                return False
    
    @property
    def copy_contcar_to_poscar(self):
        """
        Copies CONTCAR from parent to POSCAR of child
        """
        kill_job = self.kill_job
        if kill_job:
            return None

        curr_contcar = os.path.join(self.calc_dir, "CONTCAR")
        prev_contcar = os.path.join(self.prev_calc_dir, "CONTCAR")
        curr_poscar = os.path.join(self.calc_dir, "POSCAR")
        if self.is_curr_calc_being_restarted:
            copyfile(curr_contcar, curr_poscar)
            return "copied contcar from current calc"

        if os.path.exists(prev_contcar):
            with open(prev_contcar, "r", encoding="utf-8") as f:
                contents = f.read()
                if len(contents) > 0:
                    copyfile(prev_contcar, curr_poscar)
                    return "copied contcar from prev calc"
                else:
                    return None
        return None
    
    @property
    def clean_poscar(self):
        """
        Returns:
            For any near zero values in lattice (e.g. 0.00000#####...), round to 0
                to avoid potential symprec related errors
        """
        poscar = self.poscar

        struc = poscar.structure
        lattice = np.copy(struc.lattice.matrix)
        lattice[np.abs(lattice) < 1e-5] = 0.0
        struc.lattice = lattice

        struc.to(filename=os.path.join(self.calc_dir, "POSCAR"), fmt="poscar")

    @property
    def copy_incar_for_prelobster(self):
        """
        Copies INCAR from parent to child
            only pass if current calculation is prelobster to keep the same INCAR settings
            prelobster calcs are applied to generate a KPOINTS with the tetrahedra information (see ISMEAR = -5)
        """
        kill_job = self.kill_job
        if kill_job:
            return None

        curr_calc = self.curr_calc

        if "prelobster" not in curr_calc:
            return None

        src_dir = self.prev_calc_dir
        dst_dir = self.calc_dir

        fsrc = os.path.join(src_dir, "INCAR")
        if os.path.exists(fsrc):
            copyfile(fsrc, os.path.join(dst_dir, "INCAR"))
            return "copied incar"
        return None
    
    @property
    def copy_kpoints_for_prelobster(self):
        """
        Copies KPOINTS from parent to child
            only pass if current calculation is prelobster to keep the same KPOINTS settings
        """
        kill_job = self.kill_job
        if kill_job:
            return None

        curr_calc = self.curr_calc

        if "prelobster" not in curr_calc:
            return None

        src_dir = self.prev_calc_dir
        dst_dir = self.calc_dir

        fsrc = os.path.join(src_dir, "KPOINTS")
        if os.path.exists(fsrc):
            copyfile(fsrc, os.path.join(dst_dir, "KPOINTS"))
            return "copied kpoints"
        return None
    
    @property
    def copy_chgcar_for_parchg(self):
        """
        Copies CHGCAR from parent to child
            only pass if current calculation is parchg
        """
        kill_job = self.kill_job
        if kill_job:
            return None

        curr_calc = self.curr_calc

        if "parchg" not in curr_calc:
            return None

        src_dir = self.prev_calc_dir
        dst_dir = self.calc_dir

        fsrc = os.path.join(src_dir, "CHGCAR")
        if os.path.exists(fsrc):
            copyfile(fsrc, os.path.join(dst_dir, "CHGCAR"))
            return "copied chgcar"
        return None

    @property
    def copy_kpoints_for_parchg(self):
        """
        Copies KPOINTS from parent to child
            only pass if current calculation is parchg
        """
        kill_job = self.kill_job
        if kill_job:
            return None

        curr_calc = self.curr_calc

        if "parchg" not in curr_calc:
            return None

        src_dir = self.prev_calc_dir
        dst_dir = self.calc_dir

        fsrc = os.path.join(src_dir, "IBZKPT")
        if os.path.exists(fsrc):
            copyfile(fsrc, os.path.join(dst_dir, "KPOINTS"))
            return "copied kpoints"
        return None

    @property
    def copy_wavecar(self):
        """
        Copies WAVECAR from parent to child
            doesn't pass if current calculation is relax or lobster
                (because KPOINTS will be different)
        """
        kill_job = self.kill_job
        if kill_job:
            return None
        
        if self.is_curr_calc_being_restarted:
            return None

        errors_to_avoid_wavecar_passing = [
            "grad_not_orth",
            "eddrmm",
            "sym_too_tight",
            "bad_sym",
            "coef",
        ]

        errors_in_curr_calc = self.errors_encountered_in_curr_calc

        if set(errors_to_avoid_wavecar_passing) and errors_in_curr_calc:
            return None

        prev_wavecar = os.path.join(self.prev_calc_dir, "WAVECAR")
        curr_wavecar = os.path.join(self.calc_dir, "WAVECAR")

        if os.path.exists(prev_wavecar):
            copyfile(prev_wavecar, curr_wavecar)
            return "copied wavecar"
        return None

    @property
    def prev_gap(self):
        """
        Returns:
            parent's band gap (float) if parent is ready to pass else None
        """
        kill_job = self.kill_job
        if kill_job:
            return None

        # try to get bandgap
        gap_props = AnalyzeVASP(self.prev_calc_dir).gap_properties
        if gap_props and ("bandgap" in gap_props):
            return gap_props["bandgap"]
        return None

    @property
    def bandgap_label(self):
        """
        Returns:
            'metal'
                if parent band gap is < 0.01 eV
            'semiconductor'
                if parent band gap is < 0.5 eV or the structure is very large
            'insulator'
                if parent band gap is > 0.5 eV and structure is small

        need to worry about size of structure b/c ISMEAR = -5 is no good for large strucs
        """
        prev_gap = self.prev_gap
        if prev_gap is None:
            return None

        if prev_gap < 1e-2:
            return "metal"
        else:
            if len(StrucTools(os.path.join(self.calc_dir, "POSCAR")).structure) > 64:
                return "semiconductor"
            else:
                if prev_gap < 0.5:
                    return "semiconductor"
                else:
                    return "insulator"

    @property
    def bandgap_based_incar_adjustments(self):
        """
        Returns:
            a dictionary of INCAR adjustments based on band gap
                KSPACING, ISMEAR, SIGMA
        """
        curr_calc = self.curr_calc

        # if no parent bandgap can't be found, just stick to defaults
        bandgap_label = self.bandgap_label
        if not bandgap_label:
            return {}

        # do not change the ISMEAR, SIGMA, KSPACING for lobster and prelobster calcs
        if curr_calc in ["lobster", "prelobster"]:
            return {}

        adjustments = {}

        # more or less stolen from pymatgen
        if bandgap_label == "metal":
            adjustments["ISMEAR"] = 2
            adjustments["SIGMA"] = 0.2
            rmin = max(1.5, 25.22 - 2.87 * self.prev_gap)  # Eq. 25
            kspacing = 2 * np.pi * 1.0265 / (rmin - 1.0183)  # Eq. 29
            adjustments["KSPACING"] = min(kspacing, 0.44)
        elif bandgap_label == "semiconductor":
            adjustments["ISMEAR"] = 0
            adjustments["SIGMA"] = 0.05
            adjustments["KSPACING"] = 0.22
        elif bandgap_label == "insulator":
            adjustments["ISMEAR"] = -5
            adjustments["KSPACING"] = 0.22

        return adjustments

    @property
    def magmom_based_incar_adjustments(self):
        """
        Returns:
            a dictionary of INCAR adjustments based on magnetic moments
                MAGMOM drawn from previous calculation's optimized magnetic moments
        return no adjustments if
            parent doesn't exist
            parent calc is nonmagnetic
            parent calc is not converged
        """

        prev_calc_dir = self.prev_calc_dir
        if not os.path.exists(prev_calc_dir):
            return {}
        prev_incar = Incar.from_file(os.path.join(prev_calc_dir, "INCAR")).as_dict()
        if "ISPIN" not in prev_incar:
            return {}
        if prev_incar["ISPIN"] == 1:
            return {}

        av_prev = AnalyzeVASP(prev_calc_dir)
        vr_prev = av_prev.outputs.vasprun
        if not vr_prev:
            return {}

        if not av_prev.is_converged:
            return {}

        # if parent exists, is magnetic, has a vasprun, is converged, get its optimized magnetic moments as child's initial MAGMOM
        prev_structure = get_structure_from_prev_run(
            av_prev.outputs.vasprun, av_prev.outputs.outcar
        )

        magmom = prev_structure.site_properties["magmom"]
        magmom_string = " ".join([str(m) for m in magmom])

        return {"MAGMOM": magmom_string}

    @property
    def nbands_based_incar_adjustments(self):
        """
        Returns:
            a dictionary of INCAR adjustments based on NBANDS
                NBANDS = 1.5 * NBANDS of previous calculation for LOBSTER
        """
        prev_calc_dir = self.prev_calc_dir
        if not os.path.exists(prev_calc_dir):
            return {}

        # grab NBANDS from parent's OUTCAR
        av_prev = AnalyzeVASP(prev_calc_dir)
        prev_settings = av_prev.outputs.all_input_settings
        # if no OUTCAR, don't change NBANDS
        if not prev_settings:
            return {}

        old_nbands = prev_settings["NBANDS"]
        # based on CJB heuristic; note pymatgen io lobster seems to set too few bands by default
        new_nbands = {"NBANDS": int(1.5 * old_nbands)}
        return new_nbands

    @property
    def prev_number_of_kpoints(self):
        """
        Returns:
            parent's number_of_kpoints (float) if parent is ready to pass else None
        """
        kill_job = self.kill_job
        if kill_job:
            return None

        prev_calc_dir = self.prev_calc_dir
        prev_ibz = os.path.join(prev_calc_dir, "IBZKPT")

        try:
            kpoints = Kpoints.from_file(prev_ibz)
        except:
            return None

        kpts = kpoints.kpts
        if not kpts:
            return None

        num_kpoints = len(kpoints.kpts)

        if num_kpoints:
            return num_kpoints
        return None

    def kpoints_based_incar_adjustments(self, ncore, min_total_tasks=32):
        """
        Returns:
            a dictionary of INCAR adjustments based on kpoints
                KPAR
        """
        curr_xc = self.curr_xc
        curr_calc = self.curr_calc

        # only change KPAR for hse06 calcs
        if curr_xc != "hse06":
            return {}

        # do not set KPAR for preggastatic calcs
        if "preggastatic" in curr_calc:
            return {}

        prev_number_of_kpoints = self.prev_number_of_kpoints
        if not prev_number_of_kpoints:
            return {}

        adjustments = {}

        cores_for_kpoints = min_total_tasks / ncore

        kpar = max(
            [
                v
                for v in range(1, 9)
                # total number of kpoints and cores for kpoints must be divisible by KPAR
                if (prev_number_of_kpoints % v == 0) and (cores_for_kpoints % v == 0)
            ]
        )

        adjustments["KPAR"] = kpar
        return adjustments

    @property
    def pass_kpoints_for_lobster(self):
        """
        Passes prelobster's IBZKPT to lobster's KPOINTS
        """
        kill_job = self.kill_job
        if kill_job:
            return None
        
        curr_calc = self.curr_calc
        curr_calc_dir = self.calc_dir

        if curr_calc != "lobster":
            return None

        prev_calc_dir = curr_calc_dir.replace(curr_calc, 'prelobster')
        if not os.path.exists(prev_calc_dir):
            return None

        prev_ibz = os.path.join(prev_calc_dir, "IBZKPT")
        prev_kpt = os.path.join(prev_calc_dir, "KPOINTS")
        curr_kpt = os.path.join(curr_calc_dir, "KPOINTS")

        if os.path.exists(prev_ibz):
            copyfile(prev_ibz, curr_kpt)
            return "copied IBZKPT from prev calc"
        elif os.path.exists(prev_kpt):
            copyfile(prev_kpt, curr_kpt)
            return "copied KPOINTS from prev calc"
        return None

    @property
    def update_incar(self):
        """
        Returns: Nothing
            Updates INCAR based on band gap, magnetic moments, and NBANDS

            Writes new INCAR to file
        """

        # get bandgap related adjustments if relevant (ISMEAR, SIGMA, KSPACING)
        bandgap_based_incar_adjustments = self.bandgap_based_incar_adjustments

        # get new magmom if relevant (MAGMOM)
        magmom_based_incar_adjustments = self.magmom_based_incar_adjustments

        # get kpoints related adjustments if relevant
        # kpoints_based_incar_adjustments = self.kpoints_based_incar_adjustments

        # merge bandgap and magmom
        incar_adjustments = magmom_based_incar_adjustments.copy()
        incar_adjustments.update(bandgap_based_incar_adjustments)
        # incar_adjustments.update(kpoints_based_incar_adjustments)

        curr_calc = self.curr_calc
        if curr_calc in ["lobster"]:
            # for lobster calcs, we can't use KSPACING
            if "KSPACING" in incar_adjustments:
                del incar_adjustments["KSPACING"]

            # lobster calcs should have ISMEAR = 0 and SIGMA = 0.05 (I think there are issues with other ISMEAR values)
            incar_adjustments["ISMEAR"] = -5
            # incar_adjustments["SIGMA"] = 0.05

            # update NBANDS if doing lobster
            nbands_based_incar_adjustments = self.nbands_based_incar_adjustments
            incar_adjustments.update(nbands_based_incar_adjustments)

        was_wavecar_copied = self.copy_wavecar
        if was_wavecar_copied:
            incar_adjustments["ISTART"] = 1
            
        was_incar_copied_prelobster = self.copy_incar_for_prelobster
        if was_incar_copied_prelobster:
            # since prelobster calcs are used to generate KPOINTS with tetrahedra information (ISMEAR = -5), we don't want to run any vasp steps
            incar_adjustments["NELM"] = 0
            incar_adjustments["NSW"] = 0
            incar_adjustments["ISMEAR"] = -5
            
        was_kpoints_copied_prelobster = self.copy_kpoints_for_prelobster
        if was_kpoints_copied_prelobster:
            if "KSPACING" in incar_adjustments:
                del incar_adjustments["KSPACING"]
        
        was_chgcar_copied_parchg = self.copy_chgcar_for_parchg
        if was_chgcar_copied_parchg:
            incar_adjustments["ICHARG"] = 1

        was_kpoints_copied_parchg = self.copy_kpoints_for_parchg
        if was_kpoints_copied_parchg:
            if "KSPACING" in incar_adjustments:
                del incar_adjustments["KSPACING"]

        # make sure we don't override user-defined INCAR modifications
        user_incar_mods = self.incar_mods
        if user_incar_mods is None:
            user_incar_mods = {}
        if incar_adjustments is None:
            incar_adjustments = {}
        incar = Incar.from_file(os.path.join(self.calc_dir, "INCAR"))
        ncore = incar["NCORE"] if "NCORE" in incar else 1
        self.kpoints_based_incar_adjustments(ncore)

        # loop through adjustments and apply them
        for key, value in incar_adjustments.items():
            if user_incar_mods:
                if (key not in user_incar_mods) or (key == "MAGMOM"):
                    incar[key] = value
            else:
                incar[key] = value

        # apply our user-defined mods last to give them precedence
        hubbard_keys = ["LDAUU", "LDAUJ", "LDAUL"]
        for key, value in user_incar_mods.items():
            if key in hubbard_keys:
                continue
            incar[key] = value

        poscar = self.poscar
        for key in hubbard_keys:
            if key in user_incar_mods:
                el_to_value = user_incar_mods[key]
                if not el_to_value:
                    continue

                incar[key] = [
                    (
                        el_to_value.get(sym, 0)
                        if isinstance(el_to_value.get(sym, 0), (float, int))
                        else 0
                    )
                    for sym in poscar.site_symbols
                ]

        # write to INCAR
        incar.write_file(os.path.join(self.calc_dir, "INCAR"))
        return "updated incar"

    @property
    def write_to_job_killer(self):
        """
        Writes to a file in launch_dir called job_killer.o that will trigger the job to be canceled
            b/c of the try/except block in main, this will also write the error message to the file if passer fails for some reason
        """
        kill_job = self.kill_job
        fready_to_pass = os.path.join(self.launch_dir, "job_killer.o")
        with open(fready_to_pass, "w", encoding="utf-8") as f:
            if kill_job:
                f.write("kill this job")
            else:
                f.write("good to pass")

    @property
    def complete_pass(self):
        """
        copy files + update INCAR
        """
        self.copy_contcar_to_poscar
        self.update_incar  # this also copies wavecar
        self.pass_kpoints_for_lobster

        return "completed pass"


def debug():
    """
    Execute this to avoid the try/except and really figure out what's causing this script to fail
    """
    # get info that pertains to the present calculation
    passer_dict_as_str = sys.argv[1]

    # initialize the Passer for this claculation
    passer = Passer(passer_dict_as_str=passer_dict_as_str)

    passer.write_to_job_killer
    passer.complete_pass


def main():
    """
    This gets executed from your scripts_dir
    """
    # get info that pertains to the present calculation
    passer_dict_as_str = sys.argv[1]

    # initialize the Passer for this claculation
    passer = Passer(passer_dict_as_str=passer_dict_as_str)

    # try to write to job_killer and complete pass (copy CONTCAR, WAVECAR and update INCAR)
    try:
        passer.write_to_job_killer
        passer.complete_pass

    # if this fails for some reason, kill the job and populate job_killer.o with python error message that caused failure
    except Exception as e:
        fready_to_pass = os.path.join(passer.launch_dir, "job_killer.o")
        with open(fready_to_pass, "w", encoding="utf-8") as f:
            f.write("kill this job\n\n\n")
            f.write(str(e))
        f.write(traceback.print_exc())


if __name__ == "__main__":
    main()

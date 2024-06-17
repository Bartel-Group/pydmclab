import sys
import os
import json
from shutil import copyfile
import numpy as np

from pymatgen.io.vasp.inputs import Incar, Poscar
from pymatgen.io.vasp.sets import get_structure_from_prev_run

from pydmclab.hpc.analyze import AnalyzeVASP, VASPOutputs
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
        """
        passer_dict = json.loads(passer_dict_as_str)

        self.xc_calc = passer_dict["xc_calc"]
        self.calc_list = passer_dict["calc_list"]
        self.calc_dir = passer_dict["calc_dir"]
        self.incar_mods = passer_dict["incar_mods"]
        self.launch_dir = passer_dict["launch_dir"]

    @property
    def poscar(self):
        """
        Returns:
            the structure of the current calculation
        """
        return Poscar.from_file(os.path.join(self.calc_dir, "POSCAR"))

    @property
    def prev_xc_calc(self):
        """
        Returns:
            the parent xc_calc (eg 'gga-relax') that should pass stuff to the present xc_calc (eg 'gga-static)
        """
        curr_xc_calc = self.xc_calc
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
            prev_xc_calc = curr_xc_calc.replace(curr_calc, "relax")
            return prev_xc_calc

        if "defect_neutral" in curr_calc:
            if curr_xc in ["gga", "ggau"]:
                # setting dummy reference b/c nothing should come before gga-defect_neutral
                prev_xc_calc = curr_xc_calc.replace(curr_calc, "pre_defect_neutral")
            else:
                # for metagga or otherwise, inherit from neutral gga
                prev_xc_calc = curr_xc_calc.replace(curr_xc, "gga")
            return prev_xc_calc

        if "defect_charged" in curr_calc:
            if curr_xc in ["gga", "ggau"]:
                if "1kpt" in curr_calc:
                    # for gga/gga_u, inherit from 1kpt gga/gga+u
                    prev_xc_calc = curr_xc_calc.replace(
                        curr_calc, "defect_neutral_1kpt"
                    )
                else:
                    # for gga/gga+u, inherit from neutral gga/gga+u
                    prev_xc_calc = curr_xc_calc.replace(curr_calc, "defect_neutral")
            else:
                # for metagga or otherwise, inherit from charged gga calculation
                prev_xc_calc = curr_xc_calc.replace(curr_xc, "gga")
            return prev_xc_calc

        # everything else inherits from static
        return curr_xc_calc.replace(curr_calc, "static")

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
    def copy_contcar_to_poscar(self):
        """
        Copies CONTCAR from parent to POSCAR of child
        """
        kill_job = self.kill_job
        if kill_job:
            src_dir = self.calc_dir
            dst_dir = src_dir
            fpos = os.path.join(src_dir, "POSCAR")
            fcont = os.path.join(src_dir, "CONTCAR")
            if not os.path.exists(fcont):
                return None
            with open(fcont, "r", encoding="utf-8") as f:
                contents = f.read()
                if len(contents) > 0:
                    copyfile(fcont, fpos)
                    return "copied contcar from current calc"
                else:
                    return None
        src_dir = self.prev_calc_dir
        dst_dir = self.calc_dir
        fsrc = os.path.join(src_dir, "CONTCAR")
        if os.path.exists(fsrc):
            copyfile(fsrc, os.path.join(dst_dir, "POSCAR"))
        return "copied contcar from prev calc"

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

        curr_xc_calc = self.xc_calc
        curr_calc = curr_xc_calc.split("-")[1]

        # don't pass WAVECAR for these calcs
        if curr_calc in ["relax", "lobster"]:
            return None

        src_dir = self.prev_calc_dir
        dst_dir = self.calc_dir

        fsrc = os.path.join(src_dir, "WAVECAR")
        if os.path.exists(fsrc):
            copyfile(fsrc, os.path.join(dst_dir, "WAVECAR"))
        return "copied wavecar"

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
        # if no parent bandgap can't be found, just stick to defaults
        bandgap_label = self.bandgap_label
        if not bandgap_label:
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
    def nelect_from_neutral_calc_dir(self):
        """
        Returns:
            number of electrons in neutral defect structure
        """

        calc_dir = self.calc_dir
        curr_xc_calc = self.xc_calc
        calc_list = self.calc_list

        # get list of neutral defect directories
        neutral_xc_calcs = [c for c in calc_list if "defect_neutral" in c.split("-")[1]]

        # if there is at least one neutral defect directory, use the first one
        # otherwise, raise an error
        if neutral_xc_calcs:
            neutral_xc_calc = neutral_xc_calcs[0]
            neutral_dir = calc_dir.replace(curr_xc_calc, neutral_xc_calc)
        else:
            raise ValueError("No defect_neutral directory found in calc_list")

        # check if the neutral defect directory exists
        if not os.path.exists(neutral_dir):
            raise ValueError(
                "Referenced neutral defect calculation directory not found"
            )

        # get all input settings from neutral defect directory
        all_input_settings = VASPOutputs(neutral_dir).all_input_settings

        # if NELECT is not found in the input settings, raise an error
        if "NELECT" not in all_input_settings:
            raise ValueError("NELECT not found in neutral defect directory")

        # return the number of electrons in the neutral defect structure
        return all_input_settings["NELECT"]

    @property
    def charged_defects_based_incar_adjustments(self):
        """
        Method will be called when the calc name include "defect_charged"

        A calc name of the form "xc-calculation_modifiers" is expected where
        the charge state of the defect is the first modifier

        For example, "gga-defect_charged_p1_1kpt_loose" is acceptable

        Returns:
            a dictionary of INCAR adjustments based on relative charge state of defect
                NELECT = NELECT of neutral defect structure - relative charge state
        """

        curr_xc_calc = self.xc_calc

        if "defect_charged" not in curr_xc_calc:
            return {}

        # get the charge state from xc-calculation name
        charge_state = curr_xc_calc.split("-")[1].split("_")[2]
        sign, value = charge_state[0], charge_state[1]

        # adjustment nelect based on charge state (p and m reference relative charge)
        if sign == "p":
            nelect_adj = -1 * int(value)
        elif sign == "m":
            nelect_adj = int(value)
        else:
            raise ValueError("Charge state must be designated by p or m")

        # find number of electrons in parent neutral defect structure
        neutral_nelect = self.nelect_from_neutral_calc_dir

        # adjust number of electorn to create charged defect structure
        charged_nelect = neutral_nelect + nelect_adj

        return {"NELECT": int(charged_nelect)}

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

        # merge bandgap and magmom
        incar_adjustments = magmom_based_incar_adjustments.copy()
        incar_adjustments.update(bandgap_based_incar_adjustments)

        curr_xc_calc = self.xc_calc
        if curr_xc_calc.split("-")[1] == "lobster":
            # for lobster calcs, we can't use KSPACING
            if "KSPACING" in incar_adjustments:
                del incar_adjustments["KSPACING"]

            # lobster calcs should have ISMEAR = 0 and SIGMA = 0.05 (I think there are issues with other ISMEAR values)
            incar_adjustments["ISMEAR"] = 0
            incar_adjustments["SIGMA"] = 0.05

            # update NBANDS if doing lobster
            nbands_based_incar_adjustments = self.nbands_based_incar_adjustments
            incar_adjustments.update(nbands_based_incar_adjustments)

        if "defect_charged" in curr_xc_calc:
            # update NELECT based on relative charge of defect
            incar_adjustments.update(self.charged_defects_based_incar_adjustments)

        # make sure we don't override user-defined INCAR modifications
        user_incar_mods = self.incar_mods
        if user_incar_mods is None:
            user_incar_mods = {}
        if incar_adjustments is None:
            incar_adjustments = {}
        incar = Incar.from_file(os.path.join(self.calc_dir, "INCAR"))

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
        self.copy_wavecar
        self.update_incar
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


if __name__ == "__main__":
    main()

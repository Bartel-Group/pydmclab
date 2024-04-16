import sys
import os
import json
from shutil import copyfile
import numpy as np

from pymatgen.io.vasp.inputs import Incar
from pymatgen.io.vasp.sets import get_structure_from_prev_run

from pydmclab.hpc.analyze import AnalyzeVASP
from pydmclab.core.struc import StrucTools


class Passer(object):
    """
    This class is made to be executed on compute nodes (ie it gets called between VASP calls for a series of jobs that are packed together)

    The idea is to run this to figure out how to pass stuff from freshly converged calcs to subsequent calcs

    The main things we care about are:
        - changing smearing (ISMEAR, SIGMA) based on band gap
        - changing kpoints (KSPACING) based on band gap
        - passing CONTCAR --> POSCAR
        - passing WAVECAR
        - passing optimized magnetic moments as initial guesses (MAGMOM)
        - passing NBANDS for lobster
    """

    def __init__(self, passer_dict_as_str):
        """
        Args:
            passer_dict_as_str (str): a json string that contains the following keys
                - xc_calc (str): the current xc-calculation type (eg "gga-static")
                - calc_list (list): the list of all xc-calculation types that have been run (eg ["gga-static", "gga-relax", "gga-lobster"])
                - calc_dir (str): the directory of the current calculation
                - incar_mods (dict): a dictionary of user-defined INCAR modifications
                - launch_dir (str): the directory from which the job was launched
        """
        passer_dict = json.loads(passer_dict_as_str)

        self.xc_calc = passer_dict["xc_calc"]
        self.calc_list = passer_dict["calc_list"]
        self.calc_dir = passer_dict["calc_dir"]
        self.incar_mods = passer_dict["incar_mods"]
        self.launch_dir = passer_dict["launch_dir"]

    @property
    def prev_xc_calc(self):
        """
        Returns:
            the parent xc_calc that should pass stuff to the present xc_calc
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

        # everything else inherits from static
        return curr_xc_calc.replace(curr_calc, "static")

    @property
    def prev_calc_dir(self):
        """
        Returns:
            calc_dir (str) for parent calculation
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
                - if parent is not converged
            False if child should be launched
                - parent doesn't exist (ie nothing to inherit)
                - parent is converged
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
    def prev_ready_to_pass(self):
        """
        Returns:
            True if parent is converged
            False if parent doesn't exist or is not ready to pass
        """
        kill_job = self.kill_job
        if kill_job:
            return False
        calc_list = self.calc_list
        prev_xc_calc = self.prev_xc_calc
        if prev_xc_calc not in calc_list:
            return False
        return True

    @property
    def copy_contcar_to_poscar(self):
        """
        Copies CONTCAR from parent to POSCAR of child
        """
        prev_ready = self.prev_ready_to_pass
        if not prev_ready:
            return None
        src_dir = self.prev_calc_dir
        dst_dir = self.calc_dir
        fsrc = os.path.join(src_dir, "CONTCAR")
        if os.path.exists(fsrc):
            copyfile(fsrc, os.path.join(dst_dir, "POSCAR"))
        return "copied contcar"

    @property
    def copy_wavecar(self):
        """
        Copies WAVECAR from parent to child
            doesn't pass if current calculation is relax or lobster
                (because KPOINTS will be different)
        """
        prev_ready = self.prev_ready_to_pass
        if not prev_ready:
            return None
        curr_xc_calc = self.xc_calc
        curr_calc = curr_xc_calc.split("-")[1]

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
        prev_ready = self.prev_ready_to_pass
        if prev_ready:
            return AnalyzeVASP(self.prev_calc_dir).gap_properties["bandgap"]
        else:
            return None

    @property
    def bandgap_label(self):
        """
        Returns:
            'metal'
                - if parent band gap is < 0.01 eV
            'semiconductor'
                - if parent band gap is < 0.5 eV or the structure is very large
            'insulator'
                - if parent band gap is > 0.5 eV and structure is small
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
            return None
        av = AnalyzeVASP(prev_calc_dir)
        prev_settings = av.outputs.all_input_settings
        new_nbands = {}
        if prev_settings:
            old_nbands = prev_settings["NBANDS"]
            # based on CJB heuristic; note pymatgen io lobster seems to set too few bands by default
            new_nbands = {"NBANDS": int(1.5 * old_nbands)}
        return new_nbands

    @property
    def update_incar(self):
        """
        Returns: Nothing
            Updates INCAR based on band gap, magnetic moments, and NBANDS

            Writes new INCAR to file
        """
        bandgap_based_incar_adjustments = self.bandgap_based_incar_adjustments
        magmom_based_incar_adjustments = self.magmom_based_incar_adjustments

        incar_adjustments = magmom_based_incar_adjustments.copy()
        incar_adjustments.update(bandgap_based_incar_adjustments)

        curr_xc_calc = self.xc_calc
        if curr_xc_calc.split("-")[1] == "lobster":
            if "KSPACING" in incar_adjustments:
                del incar_adjustments["KSPACING"]
            incar_adjustments["ISMEAR"] = 0
            incar_adjustments["SIGMA"] = 0.05
            nbands_based_incar_adjustments = self.nbands_based_incar_adjustments
            incar_adjustments.update(nbands_based_incar_adjustments)
        user_incar_mods = self.incar_mods
        incar = Incar.from_file(os.path.join(self.calc_dir, "INCAR"))
        for key, value in incar_adjustments.items():
            if user_incar_mods:
                if (key not in user_incar_mods) or (key == "MAGMOM"):
                    incar[key] = value
            else:
                incar[key] = value
        for key, value in user_incar_mods:
            incar[key] = value
        incar.write_file(os.path.join(self.calc_dir, "INCAR"))
        return "updated incar"

    @property
    def write_to_job_killer(self):
        """
        Writes to a file in launch_dir called job_killer.o that will trigger the job to be canceled
            if a required parent is not converged (ie errored out) before the child job is launched
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


def main():
    """
    This gets executed from your scripts_dir
    """
    passer_dict_as_str = sys.argv[1]
    passer = Passer(passer_dict_as_str=passer_dict_as_str)
    try:
        passer.write_to_job_killer
        passer.complete_pass
    except Exception as e:
        fready_to_pass = os.path.join(passer.launch_dir, "job_killer.o")
        with open(fready_to_pass, "w", encoding="utf-8") as f:
            f.write("kill this job\n\n\n")
            f.write(str(e))


if __name__ == "__main__":
    main()

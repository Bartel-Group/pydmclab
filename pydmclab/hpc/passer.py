import sys
import os
from shutil import copyfile
import numpy as np
from pydmclab.hpc.analyze import AnalyzeVASP
from pydmclab.core.struc import StrucTools
from pymatgen.io.vasp.inputs import Incar
from pymatgen.io.vasp.sets import get_structure_from_prev_run

import json


class Passer(object):

    def __init__(self, passer_dict_as_str):
        passer_dict = json.loads(passer_dict_as_str)

        self.xc_calc = passer_dict["xc_calc"]
        self.calc_list = passer_dict["calc_list"]
        self.calc_dir = passer_dict["calc_dir"]
        self.incar_mods = passer_dict["incar_mods"]
        self.launch_dir = passer_dict["launch_dir"]

    @property
    def prev_xc_calc(self):
        curr_xc_calc = self.xc_calc
        curr_xc, curr_calc = curr_xc_calc.split("-")
        calc_list = self.calc_list
        if curr_calc == "loose":
            prev_xc_calc = curr_xc_calc.replace(curr_calc, "pre_loose")
            return prev_xc_calc
        if curr_calc == "relax":
            if curr_xc in ["gga", "ggau"]:
                prev_xc_calc = curr_xc_calc.replace(curr_calc, "loose")
            else:
                prev_xc_calc = curr_xc_calc.replace(curr_xc, "gga")
            return prev_xc_calc
        if curr_calc == "static":
            prev_xc_calc = curr_xc_calc.replace(curr_calc, "relax")
            return prev_xc_calc
        return curr_xc_calc.replace(curr_calc, "static")

    @property
    def prev_calc_dir(self):
        calc_dir = self.calc_dir
        curr_xc_calc = self.xc_calc
        prev_xc_calc = self.prev_xc_calc
        return calc_dir.replace(curr_xc_calc, prev_xc_calc)

    @property
    def prev_calc_convergence(self):
        prev_calc_dir = self.prev_calc_dir
        if not os.path.exists(prev_calc_dir):
            return False
        return AnalyzeVASP(prev_calc_dir).is_converged

    @property
    def kill_job(self):
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
        prev_ready = self.prev_ready_to_pass
        if not prev_ready:
            return None
        src_dir = self.prev_calc_dir
        dst_dir = self.calc_dir
        fsrc = os.path.join(src_dir, "WAVECAR")
        if os.path.exists(fsrc):
            copyfile(fsrc, os.path.join(dst_dir, "POSCAR"))
        return "copied contcar"

    @property
    def copy_wavecar(self):
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
        prev_ready = self.prev_ready_to_pass
        if prev_ready:
            return AnalyzeVASP(self.prev_calc_dir).gap_properties["bandgap"]
        else:
            return None

    @property
    def structure(self):
        return StrucTools(os.path.join(self.calc_dir, "POSCAR")).structure

    @property
    def bandgap_label(self):
        prev_gap = self.prev_gap
        if prev_gap is None:
            return None

        if prev_gap < 1e-2:
            return "metal"
        else:
            if len(self.structure) > 64:
                return "semiconductor"
            else:
                if prev_gap < 0.5:
                    return "semiconductor"
                else:
                    return "insulator"

    @property
    def bandgap_based_incar_adjustments(self):
        bandgap_label = self.bandgap_label
        if not bandgap_label:
            return {}

        adjustments = {}

        if bandgap_label == "metal":
            adjustments["ISMEAR"] = 0
            adjustments["SIGMA"] = 0.2
            rmin = max(1.5, 25.22 - 2.87 * self.prev_gap)  # Eq. 25
            kspacing = 2 * np.pi * 1.0265 / (rmin - 1.0183)  # Eq. 29
            return min(kspacing, 0.44)
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
    def update_incar(self):
        bandgap_based_incar_adjustments = self.bandgap_based_incar_adjustments
        magmom_based_incar_adjustments = self.magmom_based_incar_adjustments

        incar_adjustments = magmom_based_incar_adjustments.copy()
        incar_adjustments.update(bandgap_based_incar_adjustments)

        curr_xc_calc = self.xc_calc
        if curr_xc_calc.split("-")[1] == "lobster":
            incar_adjustments["KSPACING"] = None
            incar_adjustments["ISMEAR"] = 0
            incar_adjustments["SIGMA"] = 0.05
        user_incar_mods = self.incar_mods
        incar = Incar.from_file(os.path.join(self.calc_dir, "INCAR"))
        for key, value in incar_adjustments.items():
            if user_incar_mods:
                if (key not in user_incar_mods) or (key == "MAGMOM"):
                    incar[key] = value
            else:
                incar[key] = value

        incar.write_file(os.path.join(self.calc_dir, "INCAR"))
        return "updated incar"

    @property
    def write_to_job_killer(self):
        kill_job = self.kill_job
        fready_to_pass = os.path.join(self.launch_dir, "job_killer.o")
        with open(fready_to_pass, "w", encoding="utf-8") as f:
            if kill_job:
                f.write("kill this job")
            else:
                f.write("good to pass")

    @property
    def complete_pass(self):
        self.copy_contcar_to_poscar
        self.copy_wavecar
        self.update_incar
        return "completed pass"


def main():
    passer_dict_as_str = sys.argv[1]
    passer = Passer(passer_dict_as_str=passer_dict_as_str)
    passer.write_to_job_killer
    passer.complete_pass


if __name__ == "__main__":
    main()

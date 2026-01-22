import sys
import os

import matgl

from pymatgen.io.vasp import Poscar

from pydmclab.hpc.fp import FPRelaxer
from pydmclab.core.struc import StrucTools
from pydmclab.utils.handy import read_json


def main():

    # where this calculation, its inputs, and outputs live
    calc_dir = os.path.dirname(os.path.abspath(__file__))

    # setup output and error file streams
    sys.stdout = open(os.path.join(calc_dir, "relax.o"), "w", buffering=1)
    sys.stderr = open(os.path.join(calc_dir, "relax.e"), "w", buffering=1)

    if not os.path.exists(os.path.join(calc_dir, "fp_settings.sjon")):
        raise FileNotFoundError("fp_settings.json file is missing")

    # load in foundation potential (fp) settings
    print("Loading in fp settings...\n")
    fp_settings = read_json("fp_settings.json")

    # directory where fp files live
    fp_model_dir = fp_settings["fp_model_dir"]

    # relaxation settings
    optimizer = fp_settings["optimizer"]
    relax_cell = fp_settings["relax_cell"]
    stress_weight = fp_settings["stress_weight"]
    fmax = fp_settings["fmax"]
    steps = fp_settings["steps"]
    interval = fp_settings["interval"]
    cell_filter = fp_settings["cell_filter"]
    params_cell_filter = fp_settings["params_cell_filter"]
    fp_kwargs = fp_settings["fp_kwargs"]

    # load fp
    print("Loading FP model...\n")
    pot = matgl.load_model(fp_model_dir)

    # initialize relaxer
    print("Initializing the relaxer...\n")
    relaxer = FPRelaxer(
        potential=pot,
        calc_dir=calc_dir,
        optimizer=optimizer,
        relax_cell=relax_cell,
        stress_weight=stress_weight,
    )

    # load structure from POSCAR
    print("Reading the POSCAR...\n")
    fpos = os.path.join(calc_dir, "POSCAR")
    struc = StrucTools(fpos).structure

    # relax structure
    print("Starting relaxation...\n")
    relax_results = relaxer.relax(
        atoms=struc,
        fmax=fmax,
        steps=steps,
        interval=interval,
        traj_file=None,
        ase_cellfilter=cell_filter,
        params_asecellfilter=params_cell_filter,
        **(fp_kwargs or {}),
    )

    # ensure CONTCAR is the final structure
    final_structure = relax_results["final_structure"]
    Poscar(final_structure).write_file(os.path.join(calc_dir, "CONTCAR"))

    # check for convergence
    print("Relaxation done...\n")
    obs = relax_results["trajectory"]
    final_fmax = obs["fmaxs"][-1]
    if final_fmax > fmax:
        print(f"Final fmax = {final_fmax:.5f} eV/Å > target fmax = {fmax:.5f} eV/Å")
        print("Further relaxation needed")
        print("\t- Check that the relaxation is making progress towards convergence.")
        print("\t- Consider increasing the number of relaxation steps.")
    else:
        print(f"{len(obs["energies"])} trajectory steps recorded")
        print(
            f"{obs["energies"][-1] - obs["energies"][0]:.5f} eV overall change in energy"
        )
        print("FP relaxation converged!!!")

    return


if __name__ == "__main__":
    main()

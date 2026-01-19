import os

import matgl
from matgl.ext.ase import Relaxer

from pymatgen.io.vasp import Poscar

from pydmclab.core.struc import StrucTools
from pydmclab.utils.handy import read_json, write_json

# thought for later
#   could write in such a way that after every interval (e.g., each step if interval is 1)
#   the structure is written to a CONTCAR and can be copied back to POSCAR if fmax was not reached;
#   the convergence criteria would need to be checked

# another thought for later
#   also consider implementing the option to define convergence as reaching fmax
#   or reaching the max number of steps (in case the FP gets stuck)
#   along with this, convergence should be reading a message in the output file


def main():

    # where this calculation, its inputs, and outputs live
    calc_dir = os.path.dirname(os.path.abspath(__file__))

    # NEED TO ADD IN SOME ERROR MESSAGING IF SETTINGS JSON MISSING OR EMPTY

    # load in foundation potential (fp) settings
    fp_settings = read_json("fp_settings.json")

    # NEED TO ADD IN SOME ERROR MESSAGING IF SETTINGS ARE MESSED UP

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
    pot = matgl.load_model(fp_model_dir)

    # initialize relaxer
    relaxer = Relaxer(
        potential=pot,
        optimizer=optimizer,
        relax_cell=relax_cell,
        stress_weight=stress_weight,
    )

    # load structure from POSCAR
    fpos = os.path.join(calc_dir, "POSCAR")
    struc = StrucTools(fpos).structure

    # relax structure
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

    # save trajectory
    obs = relax_results["trajectory"]
    obs_data = {
        "energies": obs.energies,
        "forces": obs.forces,
        "stresses": obs.stresses,
        "atom_positions": obs.atom_positions,
        "cell": obs.cells,
        "atomic_numbers": obs.atoms.get_atomic_numbers(),
    }
    obs_data = write_json(obs_data, "traj.json")

    # write CONTCAR
    final_struc = relax_results["final_structure"]
    Poscar(final_struc).write_file("CONTCAR")

    return


if __name__ == "__main__":
    main()

import os
import torch
from pydmclab.core.struc import StrucTools
from pydmclab.mlp.dynamics import CHGNetRelaxer
from pydmclab.utils.handy import read_json, write_json


def main():

    # torch settings
    intra_op_threads = "placeholder"
    inter_op_threads = "placeholder"

    # set torch setings
    torch.set_num_threads(intra_op_threads)
    torch.set_num_interop_threads(inter_op_threads)

    # relax settings
    model = "placeholder"
    optimizer = "placeholder"
    stress_weight = "placeholder"
    on_isolated_atoms = "placeholder"
    fmax = "placeholder"
    steps = "placeholder"
    relax_cell = "placeholder"
    ase_filter = "placeholder"
    params_asefilter = "placeholder"
    relax_interval = "placeholder"
    verbose = "placeholder"

    # save settings
    save_interval = "placeholder"

    # current directory
    curr_dir = os.getcwd()

    # check what structures have been relaxed
    results = os.path.join(curr_dir, "chgnet_relax_results.json")
    if os.path.exists(results):
        relax_results = read_json(results)
    else:
        relax_results = {}

    # get initial strucs
    all_ini_strucs = read_json(os.path.join(curr_dir, "ini_strucs.json"))
    ini_strucs = {
        name: struc
        for name, struc in all_ini_strucs.items()
        if name not in relax_results
    }

    # initialize the relaxer
    relaxer = CHGNetRelaxer(
        model=model,
        optimizer=optimizer,
        stress_weight=stress_weight,
        on_isolated_atoms=on_isolated_atoms,
    )

    # relax structures
    current_relax_results = {}
    for name, ini_struc in ini_strucs.items():

        ini_struc = StrucTools(ini_struc).structure

        struc_results = relaxer.relax(
            atoms=ini_struc,
            fmax=fmax,
            steps=steps,
            relax_cell=relax_cell,
            ase_filter=ase_filter,
            params_asefilter=params_asefilter,
            interval=relax_interval,
            verbose=verbose,
        )

        struc_results["final_structure"] = StrucTools(
            struc_results["final_structure"]
        ).structure_as_dict
        struc_results["trajectory"] = struc_results["trajectory"].as_dict()

        current_relax_results.update({name: struc_results})

        # save results on the given save interval
        if len(current_relax_results) == save_interval:
            relax_results.update(current_relax_results)
            write_json(relax_results, results)
            current_relax_results = {}

    # save the last set of results
    if current_relax_results:
        relax_results.update(current_relax_results)
        write_json(relax_results, results)

    return


if __name__ == "__main__":
    main()

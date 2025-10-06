import os
import torch

from tqdm import tqdm

from pydmclab.core.struc import StrucTools
from pydmclab.mlp import "placeholder"
from pydmclab.utils.handy import read_json, write_json, convert_numpy_to_native


def main():

    # torch settings
    intra_op_threads = "placeholder"
    inter_op_threads = "placeholder"

    # set torch setings
    if intra_op_threads is not None:
        torch.set_num_threads(intra_op_threads)
    if inter_op_threads is not None:
        torch.set_num_interop_threads(inter_op_threads)

    # architecture type
    architecture = "placeholder"

    # model settings (these are model specific and can vary widely)
    #   see the associated model relaxer class for args
    relaxer_configs = "placeholder"

    # prediction settings
    #  see the associated relax method for your chosen model
    relax_configs = "placeholder"

    # save settings
    save_interval = "placeholder"

    # current directory
    curr_dir = os.getcwd()

    # check what structures have been relaxed
    results = os.path.join(curr_dir, "placeholder")
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
    relaxer = "placeholder"

    # relax structures
    total_strucs = len(ini_strucs)
    current_relax_results = {}
    
    with tqdm(total=total_strucs, desc="Relaxing Structures") as pbar:
        for name, ini_struc in ini_strucs.items():

            ini_struc = StrucTools(ini_struc).structure

            struc_results = "placeholder"

            struc_results["final_structure"] = struc_results["final_structure"].as_dict()
            struc_results["trajectory"] = struc_results["trajectory"].as_dict()
            struc_results = convert_numpy_to_native(struc_results)
            current_relax_results.update({name: struc_results})
            
            pbar.update(1)

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

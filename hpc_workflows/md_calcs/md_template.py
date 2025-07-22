import os
import torch
from pydmclab.core.struc import StrucTools
from pydmclab.mlp.analyze import AnalyzeMD
from pydmclab.utils.handy import write_json
from pydmclab.mlp import "placeholder"


def find_remaining_steps(logfile, trajfile, nsteps, timestep):
    amd = AnalyzeMD(logfile=logfile, trajfile=trajfile)
    summary = amd.log_summary
    last_time = summary[-1]["t"] * 1000  # in fs
    remaining_steps = round((nsteps * timestep - last_time) / timestep)
    return remaining_steps


def main():

    # torch settings
    intra_op_threads = "placeholder"
    inter_op_threads = "placeholder"

    # set torch setings
    if intra_op_threads is not None:
        torch.set_num_threads(intra_op_threads)
    if inter_op_threads is not None:
        torch.set_num_threads(inter_op_threads)

    # architecture type
    architecture = "placeholder"

    # calculator settings (these are calculator specific and can vary widely)
    #   see the associated calculator md class for args
    calculator_configs = "placeholder"

    # md settings
    #  see the associated md method for your chosen model
    md_configs = "placeholder"
    ensemble = "placeholder"
    thermostat = "placeholder"
    starting_temperature = "placeholder"
    temperature = "placeholder"
    steps = "placeholder"

    # current directory
    curr_dir = os.getcwd()

    # initial struc
    ini_struc = StrucTools(os.path.join(curr_dir, "ini_struc.json")).structure

    # running MD simulation
    if not os.path.exists(trajfile):
        # Initialize MD object
        md = "placeholder"

        # run MD simulation
        md.run(steps=steps)
    else:
        # find steps remaining
        remaining_steps = find_remaining_steps(logfile, trajfile, steps, timestep)
        if remaining_steps > 0:
            # continue from existing trajectory
            md_continue = "placeholder"

            # continue MD simulation
            md_continue.run(steps=remaining_steps)

    # collect results locally
    amd = AnalyzeMD(logfile=logfile, trajfile=trajfile)
    log_summary = amd.log_summary
    traj_summary = amd.traj_summary
    if len(log_summary) != len(traj_summary):
        print(
            "Mismatch between log and traj summaries, deleting log and traj files, need to rerun"
        )
        os.remove(logfile)
        os.remove(trajfile)
    else:
        full_summary = amd.full_summary
        write_json(
            full_summary, os.path.join(curr_dir, "placeholder")
        )

    return


if __name__ == "__main__":
    main()

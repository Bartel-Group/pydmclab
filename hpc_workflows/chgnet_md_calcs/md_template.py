import os
import torch
from pydmclab.core.struc import StrucTools
from pydmclab.mlp.analyze import AnalyzeMD
from pydmclab.mlp.chgnet.dynamics import CHGNetMD
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
    torch.set_num_threads(intra_op_threads)
    torch.set_num_interop_threads(inter_op_threads)

    # architecture type
    architecture = "placeholder"

    # calculator settings (these are calculator specific and can vary widely)
    #   see the associated calculator md class for args
    calculator_configs = "placeholder"

    # md settings
    #  see the associated md method for your chosen model
    md_configs = "placeholder"

    # MD simulation settings
    relax_first = "placeholder"
    ensemble = "placeholder"
    thermostat = "placeholder"
    taut = "placeholder"
    timestep = "placeholder"
    loginterval = "placeholder"
    nsteps = "placeholder"
    temperature = "placeholder"
    pressure = "placeholder"
    addn_args = {"placeholder": "placeholder"}

    # current directory
    curr_dir = os.getcwd()

    # initial struc
    ini_struc = StrucTools(os.path.join(curr_dir, "ini_struc.json")).structure

    # trajectory and log files
    save_traj = os.path.join(curr_dir, "placeholder")
    save_log = os.path.join(curr_dir, "placeholder")

    # running MD simulation
    if not os.path.exists(save_traj):
        # Initialize MD object
        md = "placeholder"

        # run MD simulation
        md.run(steps=nsteps)
    else:
        # find steps remaining
        remaining_steps = find_remaining_steps(save_log, save_traj, nsteps, timestep)
        if remaining_steps > 0:
            # continue from existing trajectory
            continue_md = CHGNetMD.continue_from_traj(
                model="0.3.0",
                temperature=temperature,
                pressure=pressure,
                ensemble=ensemble,
                thermostat=thermostat,
                timestep=timestep,
                taut=taut,
                trajfile=save_traj,
                logfile=save_log,
                loginterval=loginterval,
                **addn_args,
            )
            # continue MD simulation
            continue_md.run(steps=remaining_steps)

    # collect results locally
    amd = AnalyzeMD(logfile=save_log, trajfile=save_traj)
    log_summary = amd.log_summary
    traj_summary = amd.traj_summary
    if len(log_summary) != len(traj_summary):
        print(
            "Mismatch between log and traj summaries, deleting log and traj files, need to rerun"
        )
        os.remove(save_log)
        os.remove(save_traj)
    else:
        full_summary = amd.full_summary
        write_json(
            full_summary, os.path.join(curr_dir, "placeholder")
        )

    return


if __name__ == "__main__":
    main()

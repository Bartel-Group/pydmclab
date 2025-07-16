import os
import shutil
from pydmclab.core.struc import StrucTools
from pydmclab.mlp.chgnet.dynamics import CHGNetMD
from pydmclab.mlp.analyze import AnalyzeMD

DATA_DIR = os.path.join("output", "mlp-continue-md")
if not os.path.exists(DATA_DIR):
    os.makedirs(DATA_DIR)

# incomplete trajectory and log files
INCOMPLETE_TRAJ = os.path.join(DATA_DIR, "incomplete_chgnet_md.traj")
INCOMPLETE_LOG = os.path.join(DATA_DIR, "incomplete_chgnet_md.log")

INI_STRUC = os.path.join("cifs", "mp-505766-SrCoO3-supercell.cif")


# this function is used to find the remaining steps (can change as needed)
def find_remaining_steps(logfile, nsteps, timestep):
    amd = AnalyzeMD(logfile=logfile)
    summary = amd.log_summary
    last_time = summary[-1]["t"] * 1000  # in fs
    remaining_steps = round((nsteps * timestep - last_time) / timestep)
    return remaining_steps


def main():

    # name of trajectory and log files we will add to
    save_traj = os.path.join(DATA_DIR, "chgnet_md.traj")
    save_log = os.path.join(DATA_DIR, "chgnet_md.log")

    # remove these files if they exist for fresh demo
    if os.path.exists(save_traj):
        os.remove(save_traj)
    if os.path.exists(save_log):
        os.remove(save_log)

    # make copies of the incomplete files for the demo
    # (not necessary in practice, can use/ append to the existing files)
    shutil.copy(INCOMPLETE_TRAJ, save_traj)
    shutil.copy(INCOMPLETE_LOG, save_log)

    # setup initial structure
    initial_struc = StrucTools(INI_STRUC).structure
    perturbed_struc = StrucTools(initial_struc).perturb(0.1)

    # MD simulation settings
    T, timestep, nsteps, loginterval = 1200, 1, 300, 10

    # determine if we are continuing from an existing trajectory
    if not os.path.exists(save_traj):
        # setup new MD simulation
        md = CHGNetMD(
            structure=perturbed_struc,
            model="0.3.0",
            use_device="cpu",
            relax_first=True,
            temperature=T,
            timestep=timestep,
            trajfile=save_traj,
            logfile=save_log,
            loginterval=loginterval,
        )
        # run MD simulation
        md.run(steps=nsteps)
    else:
        # continue from existing trajectory
        md = CHGNetMD.continue_from_traj(
            model="0.3.0",
            use_device="cpu",
            temperature=T,
            timestep=timestep,
            trajfile=save_traj,
            logfile=save_log,
            loginterval=loginterval,
        )
        # find steps remaining
        remaining_steps = find_remaining_steps(save_log, nsteps, timestep)
        # continue MD simulation
        md.run(steps=remaining_steps)

    # analyze the MD trajectory files and log files
    amd = AnalyzeMD(logfile=save_log, trajfile=save_traj)

    # print time of final step (should be 0.3 ps)
    print(f"Final time of MD simulation: {amd.log_summary[-1]['t']} ps")

    # print number of trajectory structures (should be 31 structures)
    print(f"Number of structures in trajectory: {len(amd.traj_summary)}")

    return


if __name__ == "__main__":
    main()

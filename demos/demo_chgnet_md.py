import os
from pydmclab.core.struc import StrucTools
from pydmclab.mlp.dynamics import CHGNetRelaxer, CHGNetMD, AnalyzeMD


DATA_DIR = os.path.join("output", "mlp-md")
if not os.path.exists(DATA_DIR):
    os.makedirs(DATA_DIR)

TEST_STRUC = os.path.join("cifs", "mp-18767-LiMnO2.cif")
SAVE_TRAJ = os.path.join("output", "mlp-md", "chgnet_md.traj")
SAVE_LOG = os.path.join("output", "mlp-md", "chgnet_md.log")


def main():
    # Load structure
    initial_structure = StrucTools(TEST_STRUC).structure

    # Perturb structure
    perturbed_structure = initial_structure.perturb(0.3)

    # Initialize our simulation temperature (in K), number of steps, and log interval
    # Setting low number of steps for demo purposes
    T, nsteps, loginterval = 1800, 100, 10

    # Initialize MD object with the perturbed structure
    md = CHGNetMD(
        structure=perturbed_structure,
        model="0.3.0",
        temperature=T,
        relax_first=True,
        trajfile=SAVE_TRAJ,
        logfile=SAVE_LOG,
        loginterval=loginterval,
        # **kwargs can be used to pass additional arguments to the relaxer object
        # see pydmclab.mlp.dynamics.CHGNetRelaxer for more details
    )

    # Set the number of steps for the MD simulation and run (takes a bit of time)
    md.run(steps=nsteps)

    # Analyze the MD trajectory files and log files
    amd = AnalyzeMD(SAVE_LOG, SAVE_TRAJ)

    # amd contains the results of the MD simulation and can be accessed through the following methods:
    # amd.log_summary
    # amd.traj_summary
    # amd.full_summary
    # amd.plot_E_T_t

    return amd


if __name__ == "__main__":
    amd = main()

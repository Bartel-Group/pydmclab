import os
from pydmclab.core.struc import StrucTools
from pydmclab.mlp.fairchem.dynamics import FAIRChemMD
from pydmclab.mlp.analyze import AnalyzeMD


DATA_DIR = os.path.join("output", "mlp-md")
if not os.path.exists(DATA_DIR):
    os.makedirs(DATA_DIR)

TEST_STRUC = os.path.join("cifs", "mp-18767-LiMnO2.cif")
SAVE_TRAJ = os.path.join("output", "mlp-md", "fairchem_md.traj")
SAVE_LOG = os.path.join("output", "mlp-md", "fairchem_md.log")


def main():

    # Remove trajectory and log files if they exist otherwise they will be appended
    if os.path.exists(SAVE_TRAJ):
        os.remove(SAVE_TRAJ)
    if os.path.exists(SAVE_LOG):
        os.remove(SAVE_LOG)

    # Load structure
    initial_structure = StrucTools(TEST_STRUC).structure

    # Perturb structure
    perturbed_structure = initial_structure.perturb(0.3)

    # Initialize our simulation temperature (in K), number of steps, and log interval
    # Setting low number of steps for demo purposes
    T, nsteps, loginterval = 1800, 100, 10

    # Initialize MD object with the perturbed structure
    md = FAIRChemMD(
        atoms=perturbed_structure,
        name_or_path="uma-s-1",
        task_name="omat",
        ensemble="nvt",
        thermostat="Berendsen_inhomogeneous",
        temperature=T,
        trajfile=SAVE_TRAJ,
        logfile=SAVE_LOG,
        loginterval=loginterval,
    )

    # Set the number of steps for the MD simulation and run (takes a bit of time)
    md.run(steps=nsteps)

    # Analyze the MD trajectory files and log files
    amd = AnalyzeMD(SAVE_LOG, SAVE_TRAJ)

    # amd contains the results of the MD simulation and can be accessed through the following methods:
    # amd.log_summary
    # amd.traj_summary
    # amd.full_summary
    amd.plot_E_T_t()

    return amd


if __name__ == "__main__":
    amd = main()

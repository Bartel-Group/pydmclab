import os
from pydmclab.core.struc import StrucTools
from pydmclab.mlp.fairchem.dynamics import FAIRChemRelaxer


DATA_DIR = os.path.join("output", "mlp-relaxation")
if not os.path.exists(DATA_DIR):
    os.makedirs(DATA_DIR)

TEST_STRUC = os.path.join("cifs", "mp-18767-LiMnO2.cif")
SAVE_TRAJ = os.path.join("output", "mlp-relaxation", "fairchem_relaxation.traj")


def main():
    # Load structure
    initial_structure = StrucTools(TEST_STRUC).structure

    # Perturb structure
    perturbed_structure = initial_structure.perturb(0.3)

    # Initialize relaxer
    relaxer = FAIRChemRelaxer(name_or_path="uma-s-1", task_name="omat")

    # Relax structure
    results = relaxer.relax(
        perturbed_structure,
        fmax=0.1,
        steps=50,
        traj_path=SAVE_TRAJ,
        verbose=True,
    )

    # Get results from the observer
    observer = results["trajectory"]
    final_energy = results["final_energy"]
    print("UMA took {} steps to converge.".format(len(observer)))

    # Get initial and final energies
    print("Initial energy: {:.3f} eV".format(observer.energies[0]))
    print("Final energy: {:.3f} eV".format(final_energy))

    return observer


if __name__ == "__main__":
    obs = main()

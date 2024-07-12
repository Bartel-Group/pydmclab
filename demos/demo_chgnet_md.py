import os
from pydmclab.core.struc import StrucTools
from pydmclab.mlp.chgnet_md import CHGNetRelaxer


DATA_DIR = os.path.join("output", "mlp-relaxation")
if not os.path.exists(DATA_DIR):
    os.makedirs(DATA_DIR)

TEST_STRUC = os.path.join("cifs", "mp-18767-LiMnO2.cif")
SAVE_TRAJ = os.path.join("output", "mlp-relaxation", "chgnet_traj.traj")


def main():
    # Load structure
    initial_structure = StrucTools(TEST_STRUC).structure

    # Perturb structure
    perturbed_structure = initial_structure.perturb(0.3)

    # Initialize relaxer
    relaxer = CHGNetRelaxer(model="0.3.0", use_device="mps")

    # Relax structure
    results = relaxer.relax(
        perturbed_structure,
        fmax=0.1,
        steps=50,
        traj_path=SAVE_TRAJ,
        verbose=False,
    )

    print(relaxer.predict_structure(initial_structure))

    # Get results from the observer
    observer = results["trajectory"]
    print("CHGNet took {} steps to converge.".format(len(observer)))

    # Get initial and final energies
    print("Initial energy: {:.3f} eV".format(observer.energies[0]))
    print("Final energy: {:.3f} eV".format(observer.energies[-1]))

    return None


if __name__ == "__main__":
    main()

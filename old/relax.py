from pydmclab.core.struc import StrucTools
from pymatgen.core.structure import Structure
from chgnet.model.dynamics import TrajectoryObserver
from chgnet.model import StructOptimizer, CHGNet, CHGNetCalculator
from pymatgen.io.ase import AseAtomsAdaptor


class Relaxer(object):
    """for relaxing structures using ML potentials"""

    def __init__(self, initial_structure, model="chgnet") -> None:
        """
        Args:
            initial_structure (pymatgen.Structure):
                initial structure to optimize (relax)

            model (str):
                which MLP to use
        """

        self.initial_structure = StrucTools(initial_structure).structure
        self.st = StrucTools(self.initial_structure)
        self.model = model

        if self.model == "chgnet":
            self.chgnet = CHGNet.load()
            self.relaxer = StructOptimizer(model=self.chgnet)
            self.result = self.relaxer.relax(self.initial_structure, verbose=True)

    @property
    def relaxed_structure(self):
        """
        Returns:
            optimized structure (pymatgen.Structure)
        """
        return self.result["final_structure"]

    @property
    def trajectory(self):
        """
        Returns:
            chgnet TrajectoryObserver object
        """
        return self.result["trajectory"]

    @property
    def predictions(self):
        """

        Returns:
            dict
            {'initial' : CHGNet predictions for initial structure,
            'final' : CHGNet predictions for optimized structure}
                predictions =
                    {'e' : energy per atom (eV/atom),
                     'm' : magnetic moment (mu_B),
                     'f' : forces (eV/Angstrom),
                     's' : stress (GPa),}
        """
        s_initial = self.initial_structure
        s_final = self.relaxed_structure
        initialized_model = self.chgnet

        if self.model == "chgnet":
            predictions = {
                "initial": initialized_model.predict_structure(structure=s_initial),
                "final": initialized_model.predict_structure(structure=s_final),
            }
        else:
            raise NotImplementedError

        return predictions

    @property
    def E_per_at(self):
        """
        Returns:
            energy per atom (eV/atom) predicted by specified model
        """
        return self.predictions["final"]["e"]

    @property
    def trajectory_attributes(self):
        """
        Returns:
            list of tuples summarizing the trajectory : (step, energy, structure (pymatgen.Structure))
        """
        trajectory_energies = self.get_trajectory_energies()
        trajectory_structures = self.get_trajectory_strucs()

        trajectory_attributes = [
            (i, energy, structure)
            for i, (energy, structure) in enumerate(
                zip(trajectory_energies, trajectory_structures)
            )
        ]

        return trajectory_attributes

    def get_trajectory_strucs(self):
        """
        Returns:
            List of pymatgen.Structure objects corresponding to the structure at each step of the trajectory
            Starting with the initial structure
        """

        trajectory = self.trajectory
        initial_structure = self.initial_structure

        strucs = [initial_structure]
        species = [atom for atom in initial_structure.species]

        if self.model == "chgnet":
            # Trajectory calls may be different for different models?

            atom_positions = trajectory.atom_positions
            cells = trajectory.cells
            magmoms = trajectory.magmoms

        for i, (coords, lattice, magmom) in enumerate(
            zip(atom_positions, cells, magmoms)
        ):
            site_properties = {"magmom": magmom}
            struc = Structure(
                lattice=lattice,
                species=species,
                coords=coords,
                site_properties=site_properties,
            )
            strucs.append(struc)
        return strucs

    def get_trajectory_energies(self):
        """
        Returns:
            List of energies that follows the trajectory
            Starting with the initial energy
        """
        trajectory = self.trajectory
        initial_energy = self.predictions["initial"]["e"]

        if self.model == "chgnet":
            energies = [initial_energy, *trajectory.energies]

        return energies


def main():
    fposcar = "../data/test_data/vasp/AlN/POSCAR"
    test_relaxer = Relaxer(fposcar, model="chgnet")

    relaxation_results = {
        "relaxer": test_relaxer,
        "initial_structure": test_relaxer.initial_structure,
        "optimized_structure": test_relaxer.relaxed_structure,
        "optimized_energy": test_relaxer.E_per_at,
        "trajectory_observer": test_relaxer.trajectory,
        "trajectory_attributes": test_relaxer.trajectory_attributes,
    }

    print(relaxation_results)
    return test_relaxer, relaxation_results


if __name__ == "__main__":
    relaxer, relaxation_results = main()

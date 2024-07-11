from chgnet.model import StructOptimizer, CHGNet
from pydmclab.mlp.md import Relaxer, Observer
from pymatgen.core.structure import Structure


class CHGNetRelaxer(Relaxer):
    """for relaxing structures using CHGNet"""

    def __init__(
        self,
        initial_structure: Structure,
    ) -> None:
        """
        Args:
            initial_structure (pymatgen.Structure):
                initial structure to optimize (relax)
        """
        super().__init__(initial_structure)
        self._model = CHGNet.load()
        self._relaxer = StructOptimizer(model=self._model)
        self.results = self._relaxer.relax(self.initial_structure, verbose=False)
        self.predictions = {
            "initial": self._model.predict_structure(structure=self.initial_structure),
            "final": self._model.predict_structure(structure=self.final_structure),
        }


class CHGNetObserver(Observer):
    """for manipulating CHGNet trajectories"""

    def __init__(
        self,
        relaxer: CHGNetRelaxer,
    ) -> None:
        """
        Args:
            relaxer (CHGNetRelaxer): CHGNetRelaxer object
        """
        super().__init__(relaxer)
        self.structures = self._set_structures()
        self.energies = self._set_energies()

    def _set_structures(self):
        """
        Returns:
            list of structures (pymatgen.Structure)
        """
        strucs = []
        species = [atom for atom in self.initial_structure.species]

        atom_positions = self.trajectory.atom_positions
        cells = self.trajectory.cells
        magmoms = self.trajectory.magmoms

        for i, (coord, lattice, magmom) in enumerate(
            zip(atom_positions, cells, magmoms)
        ):
            strucs.append(
                Structure(
                    lattice=lattice,
                    species=species,
                    coords=coord,
                    site_properties={"magmom": magmom},
                )
            )

        return strucs

    def _set_energies(self):
        """
        Returns:
            list of energies (eV/atom) (float)
        """
        energies_per_atom = [
            energy / len(self.initial_structure) for energy in self.trajectory.energies
        ]
        return energies_per_atom


def main():
    fposcar = "../data/test_data/vasp/AlN/POSCAR"
    chgnet_relaxer = CHGNetRelaxer(fposcar)
    chgnet_trajectory = CHGNetObserver(chgnet_relaxer)

    print(chgnet_trajectory.energies)

    return chgnet_relaxer, chgnet_trajectory


if __name__ == "__main__":
    rel, traj = main()

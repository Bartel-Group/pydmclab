from pydmclab.core.struc import StrucTools
from pymatgen.core.structure import Structure


class Relaxer(object):
    """Parent class for relaxing structures using arbitrary ML potentials"""

    def __init__(
        self,
        initial_structure: Structure,
    ) -> None:
        """
        Args:
            initial_structure (pymatgen.Structure):
                initial structure to optimize (relax)
        """
        self.initial_structure = StrucTools(initial_structure).structure
        self._results = {"final_structure": Structure, "trajectory": None}
        self._predictions = {"initial": {"e": float}, "final": {"e": float}}

    @property
    def results(self):
        """
        Returns:
            dict
            {'final_structure' : relaxed structure (pymatgen.Structure),
                'trajectory' : TrajectoryObserver object}
        """
        return self._results

    @results.setter
    def results(self, values):
        assert isinstance(
            values, dict
        ), "Results must be a dict with keys 'final_structure' and 'trajectory'."
        assert isinstance(
            values["final_structure"], Structure
        ), "Results['final_structure'] must be a pymatgen.Structure object."
        assert (
            values["trajectory"] is not None
        ), "Results['trajectory'] must be a TrajectoryObserver object."
        self._results = values

    @property
    def predictions(self):
        """
        Returns:
            dict
            {'initial' : predictions for initial structure,
            'final' : predictions for optimized structure}
                predictions =
                    {'e' : energy per atom (eV/atom)...}
        """
        return self._predictions

    @predictions.setter
    def predictions(self, values):
        assert isinstance(
            values, dict
        ), "Predictions must be a dict with keys 'initial' and 'final'."
        assert isinstance(values["initial"], dict) and isinstance(
            values["final"], dict
        ), "Predictions['initial'] and Predictions['final'] must be dicts."
        assert all(key in values["initial"].keys() for key in ["e"]) and all(
            key in values["final"].keys() for key in ["e"]
        ), "Predictions['initial'] and Predictions['final'] must contain the key 'e': energy per atom (eV/atom)."
        self._predictions = values

    @property
    def final_structure(self):
        """
        Returns:
            optimized structure (pymatgen.Structure)
        """
        return self.results["final_structure"]

    @property
    def trajectory(self):
        """
        Returns:
            TrajectoryObserver object
        """
        return self.results["trajectory"]

    @property
    def E_per_at(self):
        """
        Returns:
            energy per atom (eV/atom) predicted by specified model
        """
        return self.predictions["final"]["e"]


class Observer(object):
    """Parent class for manipulating mlp trajectory objects"""

    def __init__(self, relaxer: Relaxer) -> None:
        """
        Args:
            Relaxer (Relaxer): Relaxer object
        """
        assert isinstance(
            relaxer, Relaxer
        ), "Must be initialized with a Relaxer object."
        self._relaxer = relaxer
        self._predictions = self._relaxer.predictions
        self._structures = [Structure]
        self._energies = [float]

        self.trajectory = self._relaxer.trajectory
        self.initial_structure = self._relaxer.initial_structure
        self.final_structure = self._relaxer.final_structure
        self.E_per_at = self._relaxer.E_per_at

    @property
    def structures(self):
        """
        Returns:
            list of structures (pymatgen.Structure)
        """
        return self._structures

    @structures.setter
    def structures(self, values):
        assert isinstance(
            values, list
        ), "Structures must be a list of pymatgen.Structure objects."
        assert all(
            isinstance(structure, Structure) for structure in values
        ), "Structures must be a list of pymatgen.Structure objects."
        self._structures = values

    @property
    def energies(self):
        """
        Returns:
            list of energies (eV/atom)
        """
        return self._energies

    @energies.setter
    def energies(self, values):
        assert isinstance(values, list), "Energies must be a list of floats."
        assert all(
            isinstance(energy, float) for energy in values
        ), "Energies must be a list of floats."
        self._energies = values


def main():
    fposcar = "../data/test_data/vasp/AlN/POSCAR"
    test_relaxer = Relaxer(fposcar)
    test_trajectory = Observer(test_relaxer)

    print(test_relaxer.initial_structure == test_trajectory.initial_structure)

    return None


if __name__ == "__main__":
    main()

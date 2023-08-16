from pydmclab.core.struc import StrucTools
from chgnet.model import StructOptimizer, CHGNet, CHGNetCalculator
from pymatgen.io.ase import AseAtomsAdaptor


class Relaxer(object):
    """for relaxing structures using ML potentials"""

    def __init__(self, initial_structure, model="chgnet"):
        """
        Args:
            initial_structure (pymatgen.Structure):
                initial structure to optimize (relax)

            model (str):
                which MLP to use
        """
        self.initial_structure = StrucTools(initial_structure).structure
        # self.st = StrucTools(self.initial_structure)
        self.model = model

    def relaxer(self, verbose=True):
        """
        Args:
            verbose (bool):
                if True, print the output steps

        Returns:
            a converged "relaxer" object
            for model = 'chgnet'
                {'final_structure' : optimized structure (pymatgen.Structure),
                 'trajectory' : chgnet.model.dynamics.TrajectoryObserver object}
        """
        if self.model == "chgnet":
            relaxer = StructOptimizer()
            return relaxer.relax(self.initial_structure, verbose=verbose)
        else:
            raise NotImplementedError

    @property
    def relaxed_structure(self):
        """
        Returns:
            optimized structure (pymatgen.Structure)
        """
        return self.relaxer(False)["final_structure"]

    @property
    def trajectory(self):
        """
        Returns:
            Not sure how to use this ...
        """
        return self.relaxer(False)["trajectory"]

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

        return {
            "initial": CHGNet().predict_structure(structure=s_initial),
            "final": CHGNet().predict_structure(structure=s_final),
        }

    @property
    def E_per_at(self):
        """
        Returns:
            final energy per atom predicted by CHGNet for CHGNet-optimized structure
        """
        return self.predictions["final"]["e"]


def main():
    fposcar = "../data/test_data/vasp/AlN/POSCAR"

    relaxer = Relaxer(fposcar, model="chgnet")
    return relaxer


if __name__ == "__main__":
    relaxer = main()

from pydmclab.core.comp import CompTools
from pydmclab.utils.handy import eVat_to_kJmol, kJmol_to_eVat

from pymatgen.analysis.reaction_calculator import Reaction
from pymatgen.core.composition import Composition


def main():
    formation_energies = {"Li": 0, "NbO2": -2, "LiNbO2": -3}
    reactants = ["Li", "NbO2"]
    products = ["LiNbO2"]

    re = ReactionEnergy(formation_energies, reactants, products)
    return re


if __name__ == "__main__":
    re = main()

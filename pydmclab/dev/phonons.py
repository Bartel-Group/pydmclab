from ase import Atoms
from mace.calculators import MACECalculator

from pymatgen.io.ase import AseAtomsAdaptor

from pydmclab.core.struc import StrucTools

import numpy as np
import os
import matplotlib.pyplot as plt


def main():
    fcif = "/Users/cbartel/Downloads/Cr2O3.cif"
    st = StrucTools(fcif)
    s = st.structure
    atoms = AseAtomsAdaptor.get_atoms(s)
    return atoms


if __name__ == "__main__":
    out = main()

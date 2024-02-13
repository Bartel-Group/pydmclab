# note: requires pymatgen-analysis-defects
""" 
pip install -e .[defects]

"""

from pydmclab.core.struc import StrucTools
from pymatgen.analysis.defects.supercells import get_sc_fromstruct


class SupercellForDefects(object):
    def __init__(self, unitcell, min_atoms=60, max_atoms=299, savename="supercell.cif"):
        """
        unitcell (Structure or structure file or Structure.as_dict): unit cell
        min_atoms (int): minimum number of atoms in supercell
        max_atoms (int): maximum number of atoms in supercell
        savename (str): name of file to save supercell

        """
        self.unitcell = StrucTools(unitcell).structure
        self.min_atoms = min_atoms
        self.max_atoms = max_atoms
        self.savename = savename

    @property
    def supercell(self):
        """
        Returns:
            supercell (pymatgen Structure object)
            saves supercell as .cif
        """
        unitcell = self.unitcell
        supercell_grid = get_sc_fromstruct(unitcell, self.min_atoms, self.max_atoms)
        supercell = unitcell * supercell_grid

        if self.savename:
            supercell.to(fmt="cif", filename=self.savename)

        return supercell


class DefectStructures(object):
    def __init__(
        self,
        supercell,
        ox_states=None,
        how_many=1,
        n_strucs=1,
    ):
        """
        Args:
            supercell (Structure or structure file or Structure.as_dict): bulk structure
            ox_states (dict): oxidation states
                {element (str) : oxidation state (int or float)}
            how_many (int): number of defects (relative to bulk cell)
            n_strucs (int): number of structures to generate

        Returns:
            transforms bulk to pymatgen Structure object
        """
        self.supercell = StrucTools(supercell).structure
        self.ox_states = ox_states
        self.how_many = how_many
        self.n_strucs = n_strucs

    def vacancies(self, el_to_remove):
        """
        Args:
            el_to_remove (str): element to remove

        Returns:
            dictionary of structures with vacancies
                {index (int) : Structure.as_dict}
        """

        bulk = self.supercell
        st = StrucTools(bulk)

        n_el_bulk = st.amts[el_to_remove]
        x_el_defect = (n_el_bulk - self.how_many) / n_el_bulk

        vacancy = st.change_occ_for_el(el_to_remove, {el_to_remove: x_el_defect})
        st = StrucTools(vacancy, ox_states=self.ox_states)

        strucs = st.get_ordered_structures(n_strucs=self.n_strucs)

        return strucs

    def substitutions(self, substitution):
        """
        Args:
            substitution (str): element to put in and element to remove
                e.g. "Ti_Cr" Ti on Cr site is the substitution

        Returns:
            dictionary of structures with substitutions
                {index (int) : Structure.as_dict}
        """
        bulk = self.supercell
        st = StrucTools(bulk)

        el_to_put_in, el_to_remove = substitution.split("_")

        n_old_el = st.amts[el_to_remove]
        x_new_el = self.how_many / n_old_el
        x_old_el = 1 - x_new_el

        sub = st.change_occ_for_el(
            el_to_remove, {el_to_remove: x_old_el, el_to_put_in: x_new_el}
        )
        st = StrucTools(sub, ox_states=self.ox_states)

        strucs = st.get_ordered_structures(n_strucs=self.n_strucs)

        return strucs


def main():
    import os

    unitcell = "/Users/cbartel/Downloads/Cr2O3.cif"
    supercell = unitcell.replace(".cif", "_super.cif")
    if not os.path.exists(supercell):
        supercell = SupercellForDefects(unitcell, savename=supercell).supercell
    dg = DefectStructures(
        supercell, ox_states={"Cr": 3, "O": -2, "Ti": 3}, how_many=1, n_strucs=5
    )
    # dg.supercell, None, None
    s = dg.supercell
    vacancies = dg.vacancies("O")
    print(StrucTools(vacancies[0]).formula)
    subs = dg.substitutions("Ti_Cr")
    print(StrucTools(subs[0]).formula)

    return s, vacancies, subs


if __name__ == "__main__":
    s, vacancies, subs = main()

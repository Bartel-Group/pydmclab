# note: requires pymatgen-analysis-defects
""" 
pip install -e .[defects]

"""
import os
from pydmclab.core.struc import StrucTools
from pymatgen.analysis.defects.supercells import get_sc_fromstruct
from doped.generation import DefectsGenerator, get_ideal_supercell_matrix


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


class NewSupercellForDefects(object):
    """
    Purpose: find ideal supercell for defect calculations (uses get_ideal_supercell_matrix from doped)
    """

    def __init__(
        self,
        unitcell,
        min_image_distance=10.0,
        min_atoms=50,
        force_cubic=False,
        force_diagnoal=False,
        ideal_threshold=0.1,
        pbar=None,
        savename=None,
        data_dir=os.getcwd(),
    ):
        """
        Allows for determination of "ideal" supercell for defect calculations

        In general, a cubic and/ or diagonalized structure will be favored and a minimum image
        distance (distance that would exist between repeating point defects) tries to be achieved

        Args:
            unitcell (Structure)
                pymatgen Structure object or structure file or Structure.asdict()
            min_image_distance (float)
                minimum image distance (in Å) between periodic atoms/ sites in supercell
            min_atoms (int)
                minimum number of atoms to include in supercell
            force_cubic (bool)
                if True, uses CubicSupercellTransformation from pymatgen-analysis-defects to generate supercell
            force_diagonal (bool)
                if True, forces the return of a diagonal transformation matrix
            ideal_threshold (float)
                even if min_image_distance and min_atom are already satisified, will test supercells up to the
                threshold of (1 + ideal_threshold) * min-size-supercell to see if any are found that yield a
                diagonalized expansion of the unitcell (this being the "ideal" transformation matrix)
            pbar (tqdm)
                related to progress bar
            savename (None or str)
                if None, generated supercell won't be saved as .cif
                if str, generated supercell will be saved as savename.cif
            data_dir (str)
                directory where generated supercell .cif will be located

            For more information, see doped.generation.DefectsGenerator (https://github.com/SMTG-Bham/doped/tree/main)

        Attributes:
            same as a input arguments
        """

        self.unitcell = StrucTools(unitcell).structure
        self.min_image_distance = min_image_distance
        self.min_atoms = min_atoms
        self.force_cubic = force_cubic
        self.force_diagonal = force_diagnoal
        self.ideal_threshold = ideal_threshold
        self.pbar = pbar
        self.savename = savename
        self.data_dir = data_dir

    @property
    def make_supercell(self):
        """
        Returns:
            supercell (pymargen Structure object)
            saves supercell as .cif if savename given (defaults to saving in current directory)
        """

        unitcell = self.unitcell
        ideal_supercell_grid = get_ideal_supercell_matrix(
            unitcell,
            min_image_distance=self.min_image_distance,
            min_atoms=self.min_atoms,
            force_cubic=self.force_cubic,
            force_diagonal=self.force_diagonal,
            ideal_threshold=self.ideal_threshold,
            pbar=self.pbar,
        )
        supercell = unitcell * ideal_supercell_grid

        if self.savename:
            StrucTools(supercell).structure_to_cif(
                filename=self.savename, data_dir=self.data_dir
            )

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

    def vacancies(self, el_to_remove, algo_to_use=0):
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

        strucs = st.get_ordered_structures(algo=algo_to_use, n_strucs=self.n_strucs)

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


class GenerateMostDefects(object):
    """
    Purpose: wrapper for doped.generation.DefectsGenerator
    """

    def __init__(
        self,
        pristine_struc,
        extrinsic=None,
        interstitial_coords=None,
        generate_supercell=True,
        charge_state_gen_kwargs=None,
        supercell_gen_kwargs=None,
        interstitial_gen_kwargs=None,
        target_frac_coords=None,
        processes=None,
    ):
        """
        Allows for automatic generation of most defects for a given pristine structure

        Addition or subtraction of additional charge states can be performed after
        the initial generation step

        Once all desired defects are generated, the user can output a dictionary of
        the form: {defect_name: defective_structure_super_cell}

        Args:
            pristine_struc (Structure)
                pymatgen Structure object or structure file or Structure.asdict()
                can be either unit cell or supercell (see generate_supercell)
            extrinsic (str, list, or dict)
                list elements to be used as extrinsic dopants
                if dict, keys are host elements and values the extrinsic dopants
            interstitial_coords (list)
                list of fractional coordinates to use as interstitial defect sites
            generate_supercell (bool)
                whether to generate a supercell for the defect calculations
                if False, then input structure is used directly
            charge_state_gen_kwargs (dict)
                can use to control completeness or sparseness of guessed charge states
                control by specifying:
                    "probability_threshold" (default=0.0075)
                    "padding" (default=1)
            supercell_gen_kwargs (dict)
                can use to control supercell generation constraints
                control by specifying:
                    "min_image_distance" (default=10)
                    "min_atoms" (default=50)
                    "ideal_threshold" (default=0.1)
                    "force_cubic" (default=False)
                    "force_diagonal" (default=False)
            interstitial_gen_kwargs (dict, bool)
                can use to control interstitial defect generation constraints
                if set to False, then no interstitial defects are generated
            target_frac_coords (list)
                the defect is placed as close as possible to these fractional coordinates
            processes (int)
                for multiprocessing (in doped), don't need to worry about this

            For more information, see doped.generation.DefectsGenerator (https://github.com/SMTG-Bham/doped/tree/main)

        Attributes:
            all_defects (DefectsGenerator)
                contains all the initially generated defects
        """

        self.pristine_struc = StrucTools(pristine_struc).structure

        all_defects = DefectsGenerator(
            self.pristine_struc,
            extrinsic=extrinsic,
            interstitial_coords=interstitial_coords,
            generate_supercell=generate_supercell,
            charge_state_gen_kwargs=charge_state_gen_kwargs,
            supercell_gen_kwargs=supercell_gen_kwargs,
            interstitial_gen_kwargs=interstitial_gen_kwargs,
            target_frac_coords=target_frac_coords,
            processes=processes,
        )

        self.all_defects = all_defects

    @property
    def summary_of_defects(self):
        """
        Returns:
            all_defects (DefectsGenerator)
                prints out summary of these defects
                object containing all the initially generated defects
        """

        return self.all_defects

    @property
    def to_dict(self):
        """
        Returns:
            all_defects (dict)
                converts DefectsGenerator object containing defects to a dictionary
        """

        return self.all_defects.as_dict()

    @property
    def get_all_defective_supercells(self):
        """
        Returns:
            all_defective_strucs (dict)
                dictionary of all defective supercells generated (pymatgen Structure objects)
                {defect_name: defective_structure_super_cell}
        """

        all_defects_as_dict = self.to_dict

        all_defective_strucs = {}
        for defect_name, defect_struc in all_defects_as_dict["defect_entries"].items():
            all_defective_strucs[defect_name] = StrucTools(
                defect_struc.defect_supercell
            ).structure_as_dict

        return all_defective_strucs

    def add_charge_states(self, defect_entry_name, charge_states):
        """
        Adds additional charge states to a defect in the all_defects attribute

        Args:
            defect_entry_name (str)
                name of defect entry
            charge_states (list)
                charge states to add to defect entry

        Returns:
            None
        """

        self.all_defects.add_charge_states(defect_entry_name, charge_states)

    def remove_charge_states(self, defect_entry_name, charge_states):
        """
        Removes charge states from a defect in the all_defects attribute

        Args:
            defect_entry_name (str)
                name of defect entry
            charge_states (list)
                charge states to remove from defect entry

        Returns:
            None
        """

        self.all_defects.remove_charge_states(defect_entry_name, charge_states)


def main():
    import os

    unitcell = "/Users/cbartel/Downloads/Cr2O3.cif"
    supercell = NewSupercellForDefects(unitcell).make_supercell
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

""" 
pip install doped
"""

import os
import numpy as np

# from pymatgen.analysis.defects.supercells import get_sc_fromstruct
from doped.generation import DefectsGenerator, get_ideal_supercell_matrix
from doped.utils.supercells import get_min_image_distance
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
from doped.utils.symmetry import (
    get_primitive_structure,
    _get_supercell_matrix_and_possibly_rotate_prim,
    get_clean_structure,
)
from pydmclab.core.struc import StrucTools
from shakenbreak.input import Distortions


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
            ).structure

        return all_defective_strucs

    @property
    def get_bulk_supercell(self):
        """
        Returns:
            bulk_supercell (pymatgen Structure object)
                supercell of the pristine structure
        """

        all_defects_as_dict = self.to_dict

        bulk_supercell = StrucTools(all_defects_as_dict["bulk_supercell"]).structure

        return bulk_supercell

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


class SupercellForDefects(object):
    """
    Purpose: find ideal supercell and/ or understand suitability of manually created supercell for defect calculations
    """

    def __init__(
        self,
        sc_structure,
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
            sc_structure (Structure)
                pymatgen Structure object or structure file or Structure.asdict()
                unitcell input if you want to make into a supercell for defect calcs (gets updated it
                    supercell structure when run method "make_supercell")
                supercell input if you want to understand the applicability of the supercell for defect calcs
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

        self.sc_structure = StrucTools(sc_structure).structure
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
            supercell (pymatgen Structure object)
            saves supercell as .cif if savename given (defaults to saving in current directory)
        """

        unitcell = self.sc_structure
        ideal_supercell_grid = get_ideal_supercell_matrix(
            unitcell,
            min_image_distance=self.min_image_distance,
            min_atoms=self.min_atoms,
            force_cubic=self.force_cubic,
            force_diagonal=self.force_diagonal,
            ideal_threshold=self.ideal_threshold,
            pbar=self.pbar,
        )
        new_sc_structure = unitcell * ideal_supercell_grid

        print(
            "The minimum image distance of the generated supercell structure is: %.2f Å \n"
            % (self.curr_min_image_distance(new_sc_structure))
        )

        print("The supercell grid in terms of the input structure is:")
        print(ideal_supercell_grid)

        print(
            "\nThe supercell grid generated in terms of the deterministic primitive structure is:"
        )
        rot_prim_struc, prim_supercell_grid = self.find_primitive_structure_grid(
            new_sc_structure
        )
        print(prim_supercell_grid)

        if self.savename:
            StrucTools(new_sc_structure).structure_to_cif(
                filename=self.savename, data_dir=self.data_dir
            )

        return new_sc_structure

    def curr_min_image_distance(self, sc_structure=None):
        """
        Returns:
            minimum image distance for the current structure (float)
                the current structure could be the initially input structure
                or if make_supercell has been run, it will be the supercell structure
        """

        if not sc_structure:
            sc_structure = self.sc_structure

        return get_min_image_distance(sc_structure)

    def find_primitive_structure(self, sc_structure=None):
        """
        Returns:
            primitive structure associated with input and/ or supercell (pymatgen Structure object)
                this structure is deterministic in that there is some logic to how the structure is
                chosen when there are multiple primitive structures possible (see https://github.com/SMTG-Bham/doped/blob/main/doped/utils/symmetry.py)
        """

        if not sc_structure:
            sc_structure = self.sc_structure

        # initializes SpacegroupAnalyzer object
        struc_as_sga = SpacegroupAnalyzer(sc_structure, symprec=0.01)

        # find deterministic primitive structure
        try:
            primitive_struc = get_primitive_structure(struc_as_sga)
        except ValueError:
            print(
                "Unable to find appropriate SpacegroupAnalyzer symmetry dataset for the input structure"
            )

        return primitive_struc

    def find_primitive_structure_grid(self, sc_structure=None):
        """
        Returns:
            rotated primitive structure (pymatgen Structure object)
                note: it may or may not rotate the initial input structure to to find the transformation
            find supercell transformation in terms of the primitive structure (np array)
        """

        if not sc_structure:
            sc_structure = self.sc_structure

        # the following directly follows from doped (see init for DefectsGenerator)

        rotated_primitive_structure, primitive_supercell_matrix = (
            _get_supercell_matrix_and_possibly_rotate_prim(
                self.find_primitive_structure(sc_structure), sc_structure
            )
        )

        # T maps orig prim struct to new prim struct; T*orig = new -> orig = T^-1*new
        rotated_primitive_structure, T_for_math = get_clean_structure(
            rotated_primitive_structure, return_T=True
        )

        # supercell matrix P was: P*orig = super -> P*T^-1*new = super -> P' = P*T^-1
        primitive_supercell_matrix = np.matmul(
            primitive_supercell_matrix, np.linalg.inv(T_for_math)
        )

        return rotated_primitive_structure, primitive_supercell_matrix


class ShakeStrucs(object):
    """
    Purpose: wrapper for ShakeNBreak NEED TO ADD MORE HERE
    """

    def __init__(
        self,
        defect_strucs,
        bulk_struc,
        oxidation_states=None,
        padding=0,
        num_of_electrons=None,
        distortion_increment=0.1,
        bond_distortions=None,
        local_rattle=False,
        distorted_elements=None,
        distorted_atoms=None,
        mc_rattle_kwargs=None,
    ):
        """
        need to add
        """
        if mc_rattle_kwargs is None:
            mc_rattle_kwargs = {}

        distortions = Distortions.from_structures(
            defect_strucs,
            bulk_struc,
            oxidation_states=oxidation_states,
            padding=padding,
            dict_number_electrons_user=num_of_electrons,
            distortion_increment=distortion_increment,
            bond_distortions=bond_distortions,
            local_rattle=local_rattle,
            distorted_elements=distorted_elements,
            distorted_atoms=distorted_atoms,
            **mc_rattle_kwargs,
        )

        self.distortions = distortions

        self.defects_dict = {}
        self.distortion_metadata = {}

    @property
    def shake_n_break(self):
        """
        need to add
        """

        defects_dict, distortion_metadata = self.distortions.apply_distortions()

        return defects_dict, distortion_metadata

    @property
    def get_shaken_strucs_info_as_dict(self):
        """
        need to add
        """

        shake_n_break_results = self.shake_n_break

        return shake_n_break_results[0]

    @property
    def get_shaken_struc_metadata_as_dict(self):
        """
        need to add
        """

        shake_n_break_results = self.shake_n_break

        return shake_n_break_results[1]

    @property
    def get_shaken_strucs(self):
        """
        need to add
        """

        shaken_strucs_as_dict = self.get_shaken_strucs_info_as_dict

        shaken_strucs_summary = {}
        for defect_struc_name, all_shaken_struc_info in shaken_strucs_as_dict.items():
            for defect_struc_charge, charged_struc in all_shaken_struc_info[
                "charges"
            ].items():
                for perturbed_struc_name, perturbed_struc in charged_struc[
                    "structures"
                ]["distortions"].items():
                    shaken_strucs_summary[
                        defect_struc_name
                        + "_"
                        + str(defect_struc_charge)
                        + "_"
                        + perturbed_struc_name
                    ] = perturbed_struc

        return shaken_strucs_summary


def main():

    from pydmclab.utils.handy import read_json

    # testing GenerateMostDefects --> need to check still
    # data_dir = "/Users/lanne056/Documents/AJ-Research/local-scripts/MOx-redox-local/data/AlN_testing"
    # strucs = read_json(os.path.join(data_dir, "struc.json"))
    # AlN_struc_dict = strucs["Al1N1"]["mp-661"]
    # AlN_defects = GenerateMostDefects(AlN_struc_dict)
    # AlN_defects.add_charge_states("v_Al", [2, 3])
    # AlN_defects.remove_charge_states("Al_N", [6, 5])
    # AlN_defect_strucs = AlN_defects.get_all_defective_supercells

    # testing SupercellForDefects --> need to edit slightly
    # data_dir = "/Users/lanne056/Documents/AJ-Research/local-scripts/MOx-redox-local/data/AlN_testing"
    # strucs = read_json(os.path.join(data_dir, "struc.json"))
    # AlN_struc_dict = strucs["Al1N1"]["mp-661"]
    # AlN_supercell = SupercellForDefects(AlN_struc_dict)
    # print(AlN_supercell.curr_min_image_distance())
    # AlN_supercell_struc = AlN_supercell.make_supercell
    # print(AlN_supercell.curr_min_image_distance())
    # StrucTools(AlN_supercell_struc).amts
    # AlN_primitive = AlN_supercell.find_primitive_structure
    # StrucTools(AlN_primitive).amts
    # rot_prim, prim_grid = SupercellForDefects(AlN_primitive).find_primitive_structure_grid
    # print(prim_grid)

    # testing ShakeStrucs --> code below should execute without issue
    # data_dir = "/Users/lanne056/Documents/AJ-Research/local-scripts/MOx-redox-local/data/AlN_testing"
    # strucs = read_json(os.path.join(data_dir, "struc.json"))
    # AlN_struc_dict = strucs["Al1N1"]["mp-661"]
    # AlN_defects = GenerateMostDefects(AlN_struc_dict)
    # AlN_defect_strucs = AlN_defects.get_all_defective_supercells
    # AlN_pristine = AlN_defects.get_bulk_supercell
    # AlN_defect_v_Al_m3 = AlN_defect_strucs["v_Al_-3"]
    # AlN_shaking_v_Al = ShakeStrucs(AlN_defect_v_Al_m3,AlN_pristine)
    # AlN_shaken_strucs_v_Al = AlN_shaking_v_Al.get_shaken_strucs
    # AlN_defect_v_N_p3 = AlN_defect_strucs["v_N_+3"]
    # AlN_shaking_v_N = ShakeStrucs(AlN_defect_v_N_p3,AlN_pristine)
    # AlN_shaken_strucs_v_N = AlN_shaking_v_N.get_shaken_strucs

    return


if __name__ == "__main__":
    main()

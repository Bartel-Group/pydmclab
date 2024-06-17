# note: requires doped
""" 
pip install doped
"""

import os
import numpy as np

from pydmclab.core.struc import StrucTools
from pydmclab.core.comp import CompTools

from doped.generation import DefectsGenerator, get_ideal_supercell_matrix
from doped.utils.supercells import get_min_image_distance
from doped.utils.symmetry import (
    get_primitive_structure,
    _get_supercell_matrix_and_possibly_rotate_prim,
    get_clean_structure,
)
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
from shakenbreak.input import Distortions


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
                pymatgen Structure object or Structure file or Structure as dict
                unitcell input if you want to make into a supercell for defect calcs
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
            minimum image distance for the input structure (float)
        """

        if not sc_structure:
            sc_structure = self.sc_structure

        return get_min_image_distance(sc_structure)

    def find_primitive_structure(self, sc_structure=None):
        """
        Returns:
            primitive structure associated with input structure (pymatgen Structure object)
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
                note: it may or may not rotate the initial input structure to find the transformation
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


class ShakeDefectiveStrucs(object):
    """
    Purpose: wrapper for shakebreak.input.Distortions
    """

    def __init__(
        self,
        initial_defect_strucs,
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
        Takes in a list of defective structures and a bulk structure and
        applies distortions to the defective structures

        The goal of generating these distorted structures is to identify
        potential minima in the energy landscape that might otherwise be
        missed when running DFT calculations

        If you want to shake a bulk (pristine) structure, see StrucTools.perturb()

        Args:
            initial_defect_strucs (list of Structures)
                pymatgen Structure objects or Structure files or Structures as dicts
                generally supercells with a point defect
            bulk_struc (Structure)
                pymatgen Structure object or Structure file or Structure as dict
                corresponding supercell of the pristine structure
            oxidation_states (dict)
                dictionary of oxidation states for each element in the structure
                e.g. {"Al": 3, "N": -3} for AlN
                if None, oxidation states are guessed
                oxidation states are used to determine number of defect neighbors to distort
            padding (int)
                considered defect charges states range from 0 to the defect oxidation state
                padding adds additional charge states on both sides of this range
            num_of_electrons (dict)
                enforce the number of missing or extra electrons in the neutral defect state
                dict has the form {defect_name: negative of the electron count change}
                e.g., removing a neutral Al from AlN would result in a loss of 3 electrons
                from the system, so "negative of electron count change" = -(-3) = 3
            distortion_increment (float)
                bond distortions will range from 0 to +/- 0.6 in steps of this value
            bond_distortions (list)
                list of bond distortions to apply to nearest neighbors in place of default set
            local_rattle (bool)
                if True, will apply random displacements that tail off moving away from defect site
                if False, will apply same amplitude rattle to each supercell site
                shakenbreak suggests False will generally have better performance
            distorted_elements (dict)
                specify the neighboring elements to distort for each defect
                e.g., {"defect_name": ["element1", "element2", ...]}
                if None, the closest neighbors to defect are chosen
            distorted_atoms (dict)
                specify the neighboring atoms to distort for each defect
                e.g., {"defect_name": [atom1, atom2, ...]}
                if None, the closest neighbors to defect are chosen
            mc_rattle_kwargs (dict)
                additional keyword arguments to pass to the rattle function

            For more information, see doped.generation.DefectsGenerator (https://github.com/SMTG-Bham/ShakeNBreak/blob/main/shakenbreak/input.py)

        Attributes:
            initial_defect_strucs (list of Structures)
                list of input defective structures (generally supercells with a point defect)
            bulk_struc (Structure)
                pristine structure (correspond supercell of bulk structure)
            distortions (Distortions)
                created shakenbreak Distortion object
            shaken_defects_data (dict)
                multi-level dict containing shaken defective structures and defect site information
            distortions_metadata (dict)
                metadata associated with the creation of the shaken defective structures

        """
        if not isinstance(initial_defect_strucs, list):
            initial_defect_strucs = [initial_defect_strucs]

        if mc_rattle_kwargs is None:
            mc_rattle_kwargs = {}

        initial_defect_strucs = [
            StrucTools(struc).structure for struc in initial_defect_strucs
        ]
        bulk_struc = StrucTools(bulk_struc).structure

        distortions = Distortions.from_structures(
            initial_defect_strucs,
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

        shaken_defects_data, distortion_metadata = distortions.apply_distortions()

        for defect in shaken_defects_data:
            print(
                "\n\033[1m%s defect type: %s\033[0m"
                % (defect, shaken_defects_data[defect]["defect_type"])
            )
        print()

        self.initial_defect_strucs = initial_defect_strucs
        self.bulk_struc = bulk_struc
        self.distortions = distortions
        self.shaken_defects_data = shaken_defects_data
        self.distortions_metadata = distortion_metadata

    @property
    def get_shaken_strucs_summary(self):
        """
        Returns:
            shaken_strucs_summary (dict)
                dictionary of collected shaken structures for each intial defective structure (Structure as dict)
                {defect_name: {defect_name__defect_charge__perturbed_struc: perturbed_structure}}
        """

        shaken_strucs_info = self.shaken_defects_data

        shaken_strucs_summary = {}
        for defect_name, all_shaken_struc_info in shaken_strucs_info.items():
            shaken_strucs_summary[defect_name] = {}
            for defect_charge, charged_strucs in all_shaken_struc_info[
                "charges"
            ].items():
                for perturbed_struc_name, perturbed_struc in charged_strucs[
                    "structures"
                ]["distortions"].items():
                    shaken_strucs_summary[defect_name][
                        defect_name
                        + "__"
                        + str(defect_charge)
                        + "__"
                        + perturbed_struc_name
                    ] = StrucTools(perturbed_struc).structure_as_dict

        return shaken_strucs_summary

    def get_shaken_strucs(self, relative_chg_of_interest, defects_of_interest=None):
        """
        When running DFT calcs for charged defects using pydmclab launcher script,
        may only be able to run calculations for a single relative charge state per script

        The output of this method should be compatible with generation of strucs.json

        Args:
            relative_chg_of_interest (int)
                relative charge state of defect of interest
            defect_types_of_interest (list)
                list of defect names of interest

        Returns:
            shaken_strucs (dict)
                dictionary of shaken defective structures of interest (Structure as dict)
                {formula_of_defective_structure: {defect_name__defect_charge__perturbed_struc: perturbed_structure}}
        """

        if defects_of_interest is None:
            defects_of_interest = list(self.shaken_defects_data.keys())

        shaken_strucs_summary = self.get_shaken_strucs_summary

        # filter shaken_strucs_summary for structures matching defect of interest and relative charge of interest
        shaken_strucs_of_interest = {}
        for defect_name in defects_of_interest:
            shaken_strucs_of_interest[defect_name] = {}
            for shaken_struc_name, shaken_struc in shaken_strucs_summary[
                defect_name
            ].items():
                struc_rel_chg = shaken_struc_name.split("__")[1]
                if struc_rel_chg == str(relative_chg_of_interest):
                    shaken_strucs_of_interest[defect_name][
                        shaken_struc_name
                    ] = shaken_struc

        # setup output dictionary to be compatible with generation of strucs.json
        shaken_strucs = {}
        for (
            defect_name,
            set_of_shaken_strucs_of_interest,
        ) in shaken_strucs_of_interest.items():
            shaken_strucs_as_list = list(set_of_shaken_strucs_of_interest.values())
            if shaken_strucs_as_list:
                # clean structure (needed for some doped generated strucs) and then get clean formula
                formula_of_defective_structure = CompTools(
                    StrucTools(shaken_strucs_as_list[0]).structure.formula
                ).clean
                shaken_strucs[formula_of_defective_structure] = (
                    set_of_shaken_strucs_of_interest
                )

        return shaken_strucs


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
                pymatgen Structure object or Structure file or Structure as dict
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

        pristine_struc = StrucTools(pristine_struc).structure

        all_defects = DefectsGenerator(
            pristine_struc,
            extrinsic=extrinsic,
            interstitial_coords=interstitial_coords,
            generate_supercell=generate_supercell,
            charge_state_gen_kwargs=charge_state_gen_kwargs,
            supercell_gen_kwargs=supercell_gen_kwargs,
            interstitial_gen_kwargs=interstitial_gen_kwargs,
            target_frac_coords=target_frac_coords,
            processes=processes,
        )

        self.pristine_struc = pristine_struc
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
                dictionary of all generated defective supercells (Structure as dict)
                {defect_name: defective_structure_super_cell}
        """

        all_defects_as_dict = self.to_dict

        all_defective_strucs = {}
        for defect_name, defect_struc in all_defects_as_dict["defect_entries"].items():
            all_defective_strucs[defect_name] = StrucTools(
                defect_struc.defect_supercell
            ).structure_as_dict

        return all_defective_strucs

    @property
    def get_bulk_supercell(self):
        """
        Returns:
            bulk_supercell (Structure as dict)
                supercell of the pristine structure
        """

        all_defects_as_dict = self.to_dict

        bulk_supercell = StrucTools(
            all_defects_as_dict["bulk_supercell"]
        ).structure_as_dict

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


def main():

    import os
    from pydmclab.utils.handy import read_json

    data_dir = "/Users/lanne056/Documents/AJ-Research/local-scripts/MOx-redox-local/data/AlN_testing"
    strucs = read_json(os.path.join(data_dir, "struc.json"))
    AlN_struc_dict = strucs["Al1N1"]["mp-661"]

    # testing DefectStructures
    unitcell = AlN_struc_dict
    supercell = SupercellForDefects(unitcell).make_supercell
    dg = DefectStructures(
        supercell, ox_states={"Al": 3, "N": -3}, how_many=1, n_strucs=5
    )
    # dg.supercell, None, None
    s = dg.supercell
    vacancies = dg.vacancies("N")
    print(StrucTools(vacancies[0]).formula)
    subs = dg.substitutions("Al_N")
    print(StrucTools(subs[0]).formula)

    # testing SupercellForDefects
    AlN_supercell_check = SupercellForDefects(AlN_struc_dict)
    print(AlN_supercell_check.curr_min_image_distance())
    AlN_supercell_struc = AlN_supercell_check.make_supercell
    AlN_actual_supercell = SupercellForDefects(AlN_supercell_struc)
    print(AlN_actual_supercell.curr_min_image_distance())
    print(StrucTools(AlN_supercell_struc).amts)
    AlN_primitive = AlN_actual_supercell.find_primitive_structure()
    print(StrucTools(AlN_primitive).amts)
    rot_prim, prim_grid = SupercellForDefects(
        AlN_primitive
    ).find_primitive_structure_grid()
    print(rot_prim)
    print(prim_grid)

    # testing GenerateMostDefects and ShakeDefectiveStrucs
    AlN_defects = GenerateMostDefects(AlN_struc_dict)
    AlN_defects.add_charge_states("v_Al", [2, 3])
    AlN_defects.remove_charge_states("Al_N", [6, 5])
    AlN_defect_strucs = AlN_defects.get_all_defective_supercells
    AlN_pristine = AlN_defects.get_bulk_supercell
    AlN_defect_vacancies = [AlN_defect_strucs["v_Al_-3"], AlN_defect_strucs["v_N_+3"]]
    AlN_shaking_vacancies = ShakeDefectiveStrucs(AlN_defect_vacancies, AlN_pristine)
    AlN_shaken_strucs_vacancies = AlN_shaking_vacancies.get_shaken_strucs_summary
    AlN_shaken_strucs_vacancies_0 = AlN_shaking_vacancies.get_shaken_strucs(0)
    AlN_shaken_strucs_vacancies_m1 = AlN_shaking_vacancies.get_shaken_strucs(-1)
    print(AlN_shaken_strucs_vacancies.keys())
    print(AlN_shaken_strucs_vacancies_0.keys())
    print(AlN_shaken_strucs_vacancies_m1.keys())
    print()
    AlN_defect_Al_N_4 = AlN_defect_strucs["Al_N_+4"]
    AlN_shaking_Al_N = ShakeDefectiveStrucs(AlN_defect_Al_N_4, AlN_pristine)
    AlN_shaken_strucs_Al_N = AlN_shaking_Al_N.get_shaken_strucs_summary
    AlN_shaken_strucs_Al_N_0 = AlN_shaking_Al_N.get_shaken_strucs(0)
    AlN_shaken_strucs_Al_N_p4 = AlN_shaking_Al_N.get_shaken_strucs(4)
    print(AlN_shaken_strucs_Al_N.keys())
    print(AlN_shaken_strucs_Al_N_0["Al49N47"].keys())
    print(AlN_shaken_strucs_Al_N_p4["Al49N47"].keys())

    return


if __name__ == "__main__":
    main()

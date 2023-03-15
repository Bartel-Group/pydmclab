from pydmc.core.comp import CompTools

from pymatgen.core.structure import Structure
from pymatgen.transformations.standard_transformations import (
    OrderDisorderedStructureTransformation,
    AutoOxiStateDecorationTransformation,
    OxidationStateDecorationTransformation,
)
from pymatgen.analysis.structure_matcher import StructureMatcher
from pymatgen.core.composition import Element, Composition
from pymatgen.core.ion import Ion
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer

import os


class StrucTools(object):
    """
    Purpose: to manipulate crystal structures for DFT calculations
    """

    def __init__(self, structure, ox_states=None):
        """
        Args:
            structure (Structure): pymatgen Structure object
                - if dict, assumes it is Structure.as_dict(); converts to Structure object
                - if str, assumes it is a path to a structure file, converts to Structure object
            ox_states (dict): dictionary of oxidation states {el (str) : oxidation state (int)}
                - or None

        """
        if isinstance(structure, dict):
            structure = Structure.from_dict(structure)
        if isinstance(structure, str):
            if os.path.exists(structure):
                structure = Structure.from_file(structure)
            else:
                raise ValueError(
                    "you passed a string to StrucTools > this means a path to a structure > but the path is empty ..."
                )
        self.structure = structure
        self.ox_states = ox_states

    @property
    def structure_as_dict(self):
        """

        Returns:
            dict: pymatgen Structure.as_dict()
        """
        return self.structure.as_dict()

    @property
    def compact_formula(self):
        """
        "clean" (reduced, systematic) formula (str) for structure
        """
        return CompTools(self.structure.formula).clean

    @property
    def formula(self):
        """
        pretty (unreduced formula) for structure
        """
        return self.structure.formula

    @property
    def els(self):
        """
        list of unique elements (str) in structure
        """
        return CompTools(self.compact_formula).els

    def make_supercell(self, grid):
        """
        Args:
            grid (list) - [nx, ny, nz]

        Returns:
            Structure repeated nx, ny, nz

            so to make a 1x2x3 supercell of the initial structure, use:
                supercell = StrucTools(structure).make_supercell([1, 2, 3])
        """
        structure = self.structure
        print("making supercell with grid %s\n" % str(grid))
        structure.make_supercell(grid)
        return structure

    def perturb(self, perturbation=0.1):
        """
        Args:
            perturbation (float) - distance in Angstrom to randomly perturb each atom

        Returns:
            Structure w/ perturbations
        """
        structure = self.structure
        structure.perturb(perturbation)
        return structure

    @property
    def decorate_with_ox_states(self):
        """
        Returns oxidation state decorated structure
            - uses Auto algorithm if no ox_states are provided
            - otherwise, applies ox_states
        """
        print("decorating with oxidation states\n")
        structure = self.structure
        ox_states = self.ox_states
        if not ox_states:
            print("     automatically\n")
            transformer = AutoOxiStateDecorationTransformation()
        else:
            transformer = OxidationStateDecorationTransformation(
                oxidation_states=ox_states
            )
            print("     using %s" % str(ox_states))
        return transformer.apply_transformation(structure)

    def get_ordered_structures(self, algo=0, decorate=True, n_strucs=1):
        """
        Args:
            algo (int) - 0 = fast, 1 = complete, 2 = best first
                - see pymatgen.transformations.standard_transformations.OrderDisorderedStructureTransformation
                - 0 usually OK
            decorate (bool) - whether to decorate with oxidation states
                - if False, self.structure must already have them
            n_strucs (int) - number of ordered structures to return

        Returns:
            dict of ordered structures {index : structure (Structure.as_dict())}
                - index = 0 has lowest Ewald energy
        """
        transformer = OrderDisorderedStructureTransformation(algo=algo)
        if decorate:
            structure = self.decorate_with_ox_states
        else:
            structure = self.structure
        return_ranked_list = n_strucs if n_strucs > 1 else False

        print("ordering disordered structures\n")
        out = transformer.apply_transformation(
            structure, return_ranked_list=return_ranked_list
        )
        out = [i["structure"] for i in out]
        # print(out[0])
        if isinstance(out, list):
            print("getting unique structures\n")
            matcher = StructureMatcher()
            groups = matcher.group_structures(out)
            out = [groups[i][0] for i in range(len(groups))]
            return {i: out[i].as_dict() for i in range(len(out))}
        else:
            return {0: out.as_dict()}

    def replace_species(self, species_mapping, n_strucs=1):
        """
        Args:
            species_mapping (dict) - {Element(el) :
                                        {Element(el1) : fraction el1,
                                                        fraction el2}}
            n_strucs (int) - number of ordered structures to return if disordered

        Returns:
            dict of ordered structures {index : structure (Structure.as_dict())}
                - index = 0 has lowest Ewald energy
        """
        structure = self.structure
        print("replacing species with %s\n" % str(species_mapping))

        disappearing_els = []
        for el_to_replace in species_mapping:
            if (len(species_mapping[el_to_replace]) == 1) and (
                list(species_mapping[el_to_replace].values())[0] == 0
            ):
                structure.remove_species(species=[el_to_replace])
                disappearing_els.append(el_to_replace)

        if disappearing_els:
            for el in disappearing_els:
                del species_mapping[el]

        if species_mapping:
            structure.replace_species(species_mapping)
        if structure.is_ordered:
            return {0: structure.as_dict()}
        else:
            structools = StrucTools(structure, self.ox_states)
            return structools.get_ordered_structures(n_strucs=n_strucs)

    def BROKEN_get_structures_with_dilute_vacancy(
        self, el_to_replace, n_strucs=1, structure=None
    ):
        """
        @TODO: revisit this


        Args:
            el_to_replace (str) - element to replace with vacancy
            n_strucs (int) - number of ordered structures to return if disordered
            structure (Structure) - structure to create vacancy in
                - if None, use self.structure

        Returns:
            dict of ordered structures {index : structure (Structure.as_dict())}
                - each structure will be missing 1 el_to_replace
        """
        if not structure:
            s = self.structure
        else:
            s = structure
        species_mapping = {
            Element(el_to_replace): {Element(el_to_replace): 1 - 1 / len(s)}
        }
        if not structure:
            return self.replace_species(species_mapping, n_strucs=n_strucs)
        else:
            return StrucTools(structure).replace_species(
                species_mapping, n_strucs=n_strucs
            )

    @property
    def spacegroup_info(self):
        """
        Returns:
            dict of spacegroup info with 'tight' or 'loose' symmetry tolerance
            e.g.,
                data['tight']['number'] returns spacegroup number with tight tolerance
                data['loose']['symbol'] returns spacegroup symbol with loose tolerance

        """
        data = {
            "tight": {"symprec": 0.01, "number": None, "symbol": None},
            "loose": {"symprec": 0.1, "number": None, "symbol": None},
        }
        for symprec in [0.01, 0.1]:
            sga = SpacegroupAnalyzer(self.structure, symprec=symprec)
            number = sga.get_space_group_number()
            symbol = sga.get_space_group_symbol()

            if symprec == 0.01:
                key = "tight"
            elif symprec == 0.1:
                key = "loose"

            data[key]["number"] = number
            data[key]["symbol"] = symbol

        return data

    def sg(self, number_or_symbol="symbol", loose_or_tight="loose"):
        """

        returns spacegroup number of symbol with loose or tight tolerance

        Args:
            number_or_symbol (str, optional): _description_. Defaults to 'symbol'.
            loose_or_tight (str, optional): _description_. Defaults to 'loose'.

        Returns:
            spacegroup number or symbol with loose or tight tolerance
        """
        sg_info = self.spacegroup_info
        return sg_info[loose_or_tight][number_or_symbol]


class SiteTools(object):
    """
    make it a little easier to get site info from structures

    """

    def __init__(self, structure, index):
        """
        Args:
            structure (Structure) - pymatgen structure
            index (int) - index of site in structure

        Returns:
            pymatgen Site object
        """
        if isinstance(structure, dict):
            structure = Structure.from_dict(structure)
        self.site = structure[index]

    @property
    def site_dict(self):
        """
        Returns:
            dict of site info (from Pymatgen)
        """
        return self.site.as_dict()

    @property
    def coords(self):
        """
        Returns:
            array of fractional coordinates for site ([x, y, z])
        """
        return self.site.frac_coords

    @property
    def magmom(self):
        """
        Returns:
            magnetic moment for site (float) or None
        """
        props = self.site.properties
        if props:
            if "magmom" in props:
                return props["magmom"]
        return None

    @property
    def is_fully_occ(self):
        """
        Returns:
            True if site is fully occupied else False
        """
        return self.site.is_ordered

    @property
    def ion(self):
        """
        Returns:
            whatever is occupying site (str)
                - could be multiple ions, multiple elements, one element, one ion, etc
        """
        return self.site.species_string

    @property
    def el(self):
        """
        Returns:
            just the element occupying the site (even if it has an oxidation state)
        """
        return CompTools(Composition(self.ion).formula).els[0]

    @property
    def ox_state(self):
        """
        Returns:
            oxidation state (float) of site
        """
        if self.is_fully_occ:
            return self.site_dict["species"][0]["oxidation_state"]
        else:
            print("cant determine ox state for partially occ site")
            return None


def main():

    return


if __name__ == "__main__":
    main()

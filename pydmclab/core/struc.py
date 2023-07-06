from pydmclab.core.comp import CompTools

from pymatgen.core.structure import Structure
from pymatgen.transformations.standard_transformations import (
    OrderDisorderedStructureTransformation,
    AutoOxiStateDecorationTransformation,
    OxidationStateDecorationTransformation,
)
from pymatgen.analysis.structure_matcher import StructureMatcher
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
from pymatgen.core.surface import SlabGenerator

import os
import numpy as np


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
        # convert Structure.as_dict() to Structure
        if isinstance(structure, dict):
            structure = Structure.from_dict(structure)

        # convert file into Structure
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

    @property
    def amts(self):
        """
        Returns:
            {el (str): number of el in struc (int)}
        """
        els = self.els
        amts = {el: 0 for el in els}
        structure = self.structure
        for i, site in enumerate(structure):
            el = SiteTools(structure, i).el
            if el:
                amts[el] += 1
        return amts

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

    def change_occ_for_site(self, site_idx, new_occ, structure=None):
        """

        return a structure with a new occupation for some site

        Args:
            site_idx (int):
                index of site in structure to change

            new_occ (dict):
                dictionary telling me the new occupation on that site
                    e.g., {'Li' : 0.5, 'Fe' : 0.5}

            structure (None or pymatgen Structure object):
                if None, start from self.structure
                else, start from structure
                    (in case you don't want to start from the structure that initialized StrucTools)

        Returns:
            pymatgen Structure object with new occupation
        """

        if not structure:
            structure = self.structure

        s = structure.copy()

        if np.sum(list(new_occ.values())) == 0:
            # if new occupation is 0, remove that site
            s.remove_sites([site_idx])
        else:
            # otherwise, update the occupation
            s[site_idx].species = new_occ
        return s

    def change_occ_for_el(self, el, new_occ, structure=None):
        """
        Args:
            el (str)
                element to change occupation for

            new_occ (dict)
                {el : new occupation (float)}

            structure (None or pymatgen Structure object)
                if None, start from self.structure
        """
        if not structure:
            structure = self.structure

        # for all sites having that element, change the occupation
        for i, site in enumerate(structure):
            if SiteTools(structure, i).el == el:
                structure = self.change_occ_for_site(i, new_occ, structure=structure)

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
        els = self.els
        if not ox_states:
            ox_states = {els[0]: 0}

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
            algo (int):
                method for enumeration
                    0 = fast, 1 = complete, 2 = best first
                        see pymatgen.transformations.standard_transformations.OrderDisorderedStructureTransformation
                        0 usually OK

            decorate (bool)
                whether to decorate with oxidation states
                    if False, self.structure must already have them

            n_strucs (int)
                number of ordered structures to return

        Returns:
            dict of ordered structures
            {index : structure (Structure.as_dict())}
                - index = 0 has lowest Ewald energy
        """

        # initialize ordering engine
        transformer = OrderDisorderedStructureTransformation(algo=algo)

        # decorat with oxidation states or not
        if decorate:
            structure = self.decorate_with_ox_states
        else:
            structure = self.structure

        # only return one structure if n_strucs = 1
        return_ranked_list = n_strucs if n_strucs > 1 else False

        # generate ordered structure
        print("ordering disordered structures\n")
        out = transformer.apply_transformation(
            structure, return_ranked_list=return_ranked_list
        )

        if isinstance(out, list):
            # more than 1 structure, so check for duplicates (symmetrically equivalent structures) and remove them
            print("getting unique structures\n")
            matcher = StructureMatcher()
            out = [i["structure"] for i in out]
            # find unique groups of structures
            groups = matcher.group_structures(out)
            out = [groups[i][0] for i in range(len(groups))]
            return {i: out[i].as_dict() for i in range(len(out))}
        else:
            # if only one structure is made, return in same formation (dict)
            return {0: out.as_dict()}

    def replace_species(
        self,
        species_mapping,
        n_strucs=1,
        use_ox_states_in_mapping=False,
        use_occ_in_mapping=True,
    ):
        """
        Args:
            species_mapping (dict)
                {Element(el) :
                    {Element(el1) : fraction el1,
                     Element(el2) : fraction el2,
                     ...},
                 ...}

            n_strucs (int)
                number of ordered structures to return if disordered

            use_ox_states_in_mapping (bool)
                if False, will remove oxidation states before doing replacements

            use_occ_in_mapping (bool)
                if False, will set all occupancies to 1.0 before doing replacements

        Returns:
            dict of ordered structures
                {index : structure (Structure.as_dict())}
                    index = 0 has lowest Ewald energy
        """
        structure = self.structure
        print("replacing species with %s\n" % str(species_mapping))

        # purge oxidation states if you'd like
        if not use_ox_states_in_mapping:
            structure.remove_oxidation_states()

        # ignore the original occupancies if you'd like (sometimes convenient)
        if not use_occ_in_mapping:
            els = self.els
            for el in els:
                structure = self.change_occ_for_el(el, {el: 1.0}, structure=structure)

        # figure out which elements have occupancy becoming 0
        disappearing_els = []
        for el_to_replace in species_mapping:
            if (len(species_mapping[el_to_replace]) == 1) and (
                list(species_mapping[el_to_replace].values())[0] == 0
            ):
                structure.remove_species(species=[el_to_replace])
                disappearing_els.append(el_to_replace)

        # remove these no longer existing elements
        if disappearing_els:
            for el in disappearing_els:
                del species_mapping[el]

        # replace species according to mapping
        if species_mapping:
            structure.replace_species(species_mapping)

        if structure.is_ordered:
            # if the replacement leads to an ordered structure, return it (in a dict)
            return {0: structure.as_dict()}
        else:
            # otherwise, need to order this partially occupied structure
            structools = StrucTools(structure, self.ox_states)
            return structools.get_ordered_structures(n_strucs=n_strucs)

    @property
    def spacegroup_info(self):
        """
        Returns:
            dict of spacegroup info with 'tight' or 'loose' symmetry tolerance
                tight means symprec = 0.01
                loose means symprec = 0.1
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
            number_or_symbol (str):
                whether to return the number or the symbol

            loose_or_tight (str):
                whether to use the loose or tight tolerance

        Returns:
            spacegroup number (int) or symbol (str) with loose or tight tolerance
        """
        sg_info = self.spacegroup_info
        return sg_info[loose_or_tight][number_or_symbol]

    def scale_structure(self, scale_factor, structure=None):
        """

        Isotropically scale a structure

        Args:
            scale_factor (float): fractional scaling of the structure volume
                - e.g., 1.2 will increase each lattice vector by 20%
                - e.g., 0.8 will make eaech lattice vector 80% of the initial length
                - e.g., 1.0 will do nothing

            structure (Structure, optional): structure to scale. Defaults to None.
                - if None will use self.structure fed to initialization

        Returns:
            pymatgen structure object
        """
        if not structure:
            structure = self.structure.copy()
        else:
            structure = structure.copy()
        orig_vol = structure.volume
        new_vol = orig_vol * scale_factor

        # scaling occurs only on the lattice (i.e., the top of POSCAR)
        structure.scale_lattice(new_vol)

        return structure

    def get_slabs(
        self,
        miller=(1, 0, 0),
        min_slab_size=10,
        min_vacuum_size=10,
        center_slab=True,
        in_unit_planes=True,
        reorient_lattice=True,
    ):
        """
        Args:
            TODO

        Returns:
            TODO

        Use case:

            data = {}
            bulks = get_strucs()
            millers = [(1, 0, 0), (0, 1, 0), (0, 0, 1)]
            for b in bulks:
                st = StrucTools(b)
                formula = st.formula
                data[formula] = {}
                for m in millers:
                    slabs = st.get_slabs(miller=m)
                    data[formula].update(slabs)
        """
        bulk = self.decorate_with_ox_states

        slabgen = SlabGenerator(
            bulk,
            miller_index=miller,
            min_slab_size=min_slab_size,
            min_vacuum_size=min_vacuum_size,
            center_slab=center_slab,
            in_unit_planes=in_unit_planes,
            reorient_lattice=reorient_lattice,
        )

        slabs = slabgen.get_slabs()

        out = {}
        for i, slab in enumerate(slabs):
            key = "".join([str(v) for v in miller]) + "_" + str(i)
            out[key] = slab.as_dict()

        return out


class SiteTools(object):
    """
    Purpose: make it a little easier to get site info from structures

    """

    def __init__(self, structure, index):
        """
        Args:
            structure (Structure)
                pymatgen Structure

            index (int)
                index of site in structure

        Returns:
            pymatgen Site object
        """
        structure = StrucTools(structure).structure
        self.site = structure[index]

    @property
    def site_dict(self):
        """
        Returns:
            dict of site info (from Pymatgen)
                {'species' : [{'element' : element, 'occu' : occupation, ...}, {...}}],
                 'abc' : fractional coordinates ([a, b, c])
                 'lattice' : Lattice object,
                 'properties' : dict (e.g., {'magmom' : 3})}
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
    def site_string(self):
        """
        unique string to represent a complex site

        Returns:
            occupation_element_oxstate__occupation_element_oxstate__... for each ion occupying a site
        """
        d = self.site_dict
        species = d["species"]
        ions = []
        for entry in species:
            el = entry["element"]
            if "oxidation_state" in entry:
                ox = float(entry["oxidation_state"])
            else:
                ox = None
            occ = float(entry["occu"])
            if ox:
                if ox < 0:
                    ox = str(abs(ox)) + "-"
                else:
                    ox = str(ox) + "+"
            else:
                ox = "0.0+"
            occ = str(occ)
            name_to_join = []
            for thing in [occ, el, ox]:
                if thing:
                    name_to_join.append(thing)
            name = "_".join(name_to_join)
            ions.append(name)
        if len(ions) == 1:
            return ions[0]
        else:
            return "__".join(ions)

    @property
    def ion(self):
        """
        Returns:
            the ion (element + oxidation state) occupying the site (str)
                None if > 1 ion
        """
        site_string = self.site_string
        if "__" in site_string:
            print("Multiple ions in site, returning None")
            return None

        return "".join([site_string.split("_")[1], site_string.split("_")[2]])

    @property
    def el(self):
        """
        Returns:
            just the element occupying the site (str)
                even if it has an oxidation state)

            None if more than one element occupies a site
        """
        site_string = self.site_string
        if "__" in site_string:
            print("Multiple ions in site, returning None")
            return None

        return site_string.split("_")[1]

    @property
    def ox_state(self):
        """
        Returns:
            oxidation state (float) of site
                averaged over all ions occupying the site
        """
        d = self.site_dict
        ox = 0
        species = d["species"]
        for entry in species:
            if entry["oxidation_state"]:
                ox += entry["oxidation_state"] * entry["occu"]
        return ox


def main():
    return


if __name__ == "__main__":
    main()

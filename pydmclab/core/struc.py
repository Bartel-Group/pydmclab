import os
import subprocess
import yaml
import random
from collections import Counter, defaultdict
from shutil import copyfile, rmtree
from typing import List, Tuple, Dict, Literal

import numpy as np

from pymatgen.core import Structure, PeriodicSite
from pymatgen.core.surface import SlabGenerator
from pymatgen.transformations.standard_transformations import (
    OrderDisorderedStructureTransformation,
    AutoOxiStateDecorationTransformation,
    OxidationStateDecorationTransformation,
)
from pymatgen.analysis import structure_matcher
from pymatgen.analysis.structure_matcher import StructureMatcher
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer

from pydmclab.core import struc as pydmc_struc
from pydmclab.core.comp import CompTools


class StrucTools(object):
    """
    Purpose: to manipulate crystal structures for DFT calculations
    """

    def __init__(
        self, structure: Structure | dict | str, ox_states: dict | None = None
    ) -> None:
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
    def structure_as_dict(self) -> dict:
        """

        Returns:
            dict: pymatgen Structure.as_dict()
        """
        return self.structure.as_dict()

    @property
    def compact_formula(self) -> str:
        """
        "clean" (reduced, systematic) formula (str) for structure
        """
        return CompTools(self.structure.formula).clean

    @property
    def formula(self) -> str:
        """
        pretty (unreduced formula) for structure
        """
        return self.structure.formula

    @property
    def els(self) -> list[str]:
        """
        list of unique elements (str) in structure
        """
        return CompTools(self.compact_formula).els

    @property
    def amts(self) -> dict[str, int]:
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

    def make_supercell(self, grid: list[int], verbose: bool = True) -> Structure:
        """
        Args:
            grid (list) - [nx, ny, nz]
            verbose (bool) - whether to print out grid

        Returns:
            Structure repeated nx, ny, nz

            so to make a 1x2x3 supercell of the initial structure, use:
                supercell = StrucTools(structure).make_supercell([1, 2, 3])
        """
        structure = self.structure
        if verbose:
            print("making supercell with grid %s\n" % str(grid))
        structure.make_supercell(grid)
        return structure

    def perturb(self, perturbation: float = 0.1) -> Structure:
        """
        Args:
            perturbation (float) - distance in Angstrom to randomly perturb each atom

        Returns:
            Structure w/ perturbations
        """
        structure = self.structure
        structure.perturb(perturbation)
        return structure

    def change_occ_for_site(
        self,
        site_idx: int,
        new_occ: dict[str, float],
        structure: Structure | None = None,
    ) -> Structure:
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

    def change_occ_for_el(
        self, el: str, new_occ: dict[str, float], structure: Structure | None = None
    ) -> Structure:
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
    def decorate_with_ox_states(self) -> Structure:
        """
        Returns oxidation state decorated structure
            - uses Auto algorithm if no ox_states are provided
            - otherwise, applies ox_states
        """
        print("decorating with oxidation states\n")
        structure = self.structure
        ox_states = self.ox_states

        els = self.els
        if (len(els) == 1) and not ox_states:
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

    def get_ordered_structures(
        self,
        algo: Literal[0, 1, 2] = 0,
        decorate: bool = True,
        n_strucs: int = 1,
        verbose: bool = True,
    ) -> dict[int, dict]:
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
        return_ranked_list = n_strucs * 1000 if n_strucs > 1 else False

        # generate ordered structure
        if verbose:
            print("ordering disordered structures\n")
        out = transformer.apply_transformation(
            structure, return_ranked_list=return_ranked_list
        )

        if isinstance(out, list):
            # more than 1 structure, so check for duplicates (symmetrically equivalent structures) and remove them
            if verbose:
                print("getting unique structures\n")
            matcher = StructureMatcher()
            out = [i["structure"] for i in out]
            # find unique groups of structures
            groups = matcher.group_structures(out)
            out = [groups[i][0] for i in range(len(groups))]
            strucs = {i: out[i].as_dict() for i in range(len(out))}
            strucs = {i: strucs[i] for i in range(n_strucs) if i in strucs}
        else:
            # if only one structure is made, return in same formation (dict)
            return {0: out.as_dict()}

    def replace_species(
        self,
        species_mapping: dict[str, dict[str, float]],
        n_strucs: int = 1,
        use_ox_states_in_mapping: bool = False,
        use_occ_in_mapping: bool = True,
        verbose: bool = True,
    ) -> dict[int, dict]:
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
        if verbose:
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
            return structools.get_ordered_structures(n_strucs=n_strucs, verbose=verbose)

    @property
    def spacegroup_info(self) -> dict[str, dict[str, int | str]]:
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

    def sg(
        self,
        number_or_symbol: Literal["number", "symbol"] = "symbol",
        loose_or_tight: Literal["loose", "tight"] = "loose",
    ) -> int | str:
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

    def scale_structure(
        self, scale_factor: float, structure: Structure | None = None
    ) -> Structure:
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
        miller: tuple[int, int, int] = (1, 0, 0),
        *,
        min_slab_size: int = 10,
        min_vacuum_size: int = 10,
        center_slab: bool = True,
        in_unit_planes: bool = True,
        reorient_lattice: bool = True,
        symmetrize: bool = True,
        max_normal_search: int | None = None,
        tolerance: float = 0.1,
        ftolerance: float = 0.1,
    ) -> dict[str, dict]:
        """
        Args:
            miller (tuple): miller index of slab
            min_slab_size (int): minimum slab size in Angstrom
            min_vacuum_size (int): minimum vacuum size in Angstrom
            center_slab (bool): whether to center the slab
            in_unit_planes (bool): whether to use unit planes
            reorient_lattice (bool): whether to reorient the lattice
            symmetrize (bool): whether to symmetrize the slab
            tolerance (float): tolerance for symmetrization
            ftolerance (float): tolerance for finding the fractional positions of the terminations

        Returns:
            dict of slabs
                {miller_min_vacuum_size_i : slab.as_dict()}

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
            max_normal_search=max_normal_search,
        )

        slabs = slabgen.get_slabs(symmetrize=symmetrize, tol=tolerance, ftol=ftolerance)

        miller_str = "".join([str(i) for i in miller])

        out = {miller_str: {}, "bulk_structure": bulk.as_dict()}
        for i, slab in enumerate(slabs):
            out[miller_str][i] = {}
            out[miller_str][i]["slab"] = slab.as_dict()
            out[miller_str][i]["vacuum_size"] = min_vacuum_size
            out[miller_str][i]["slab_size"] = min_slab_size
            out[miller_str][i]["center_slab"] = center_slab
            out[miller_str][i]["in_unit_planes"] = in_unit_planes
            out[miller_str][i]["reorient_lattice"] = reorient_lattice

        return out

    def structure_to_cif(self, filename: str, data_dir: str = None) -> None:
        """
        Coverts a structure to a cif file and saves it to a directory, useful for VESTA viewing

        Args:
            filename (str): name of cif file
            data_dir (str): path to directory to save cif file

        Returns:
            None
        """
        if not data_dir:
            data_dir = os.getcwd()

        if not os.path.exists(data_dir):
            os.makedirs(data_dir)

        self.structure.to(filename=os.path.join(data_dir, f"{filename}.cif"))

        return None


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


class SolidSolutionGenerator:
    """
    Generate quasi-random solid solutions (SQS) between two crystal structures.
    This is accomplished using the sqsgenerator package.

    Attributes:
        endmembers (List[Structure]): List of two endmember structures.
        num_solns (int): Number of solutions to generate.
        supercell_dim (List[int]): Dimensions of the supercell.
        element_A (str): Element that differs in the first endmember.
        element_B (str): Element that differs in the second endmember.
        disordered_solns (List[Structure]): List of disordered solid solution structures.
        ordered_solns (List[Structure]): List of ordered solid solution structures.
        sqs_solns (List[Structure]): List of special quasirandom structures (SQS).
        dirs (Dict[str, str]): Dictionary of directory paths used in the process.

    Example usage:
        struc_A = Structure.from_file('Endmembers/SrRuO3.cif')
        struc_B = Structure.from_file('Endmembers/SrZrO3.cif')
        generator = SolidSolutionGenerator(endmembers=[struc_A, struc_B], num_solns=15, supercell_dim=[2, 1, 2])
        disordered, ordered, sqs = generator.run(sqsgen_path='/path/to/sqsgen')

    Note:
        '/path/to/sqsgen' should be replaced with the location of your sqsgen installation.
    """

    def __init__(
        self,
        endmembers: List[Structure],
        num_solns: int = 15,
        supercell_dim: List[int] = [2, 2, 2],
    ):
        """
        Initialize the SolidSolutionGenerator.

        Args:
            endmembers (List[Structure]): List of two endmember structures.
            num_solns (int, optional): Number of solutions to generate. Defaults to 15.
            supercell_dim (List[int], optional): Dimensions of the supercell. Defaults to [2, 2, 2].
        """
        self.endmembers = endmembers
        self.num_solns = num_solns
        self.supercell_dim = supercell_dim
        self.element_A: str = None
        self.element_B: str = None
        self.disordered_solns: List[Structure] = None
        self.ordered_solns: List[Structure] = None
        self.sqs_solns: List[Structure] = None

        # Create necessary directories
        self.dirs: Dict[str, str] = {
            "output": "Ordered_Solutions",
            "yaml": "yaml_files",
            "sqs": "SQS",
            "temp": "working_dir",
        }
        for dir_name in self.dirs.values():
            os.makedirs(dir_name, exist_ok=True)

    def generate_solid_solutions(self) -> List[Structure]:
        """
        Generate disordered solid solutions between the two endmember structures.

        Returns:
            List[Structure]: List of disordered solid solution structures.

        Raises:
            ValueError: If the endmember structures are incompatible or if more than one element differs between them.
        """
        struc_A, struc_B = self.endmembers
        assert (
            len(self.endmembers) == 2
        ), f"There should be 2 endmembers, but you provided {len(self.endmembers)}."

        # Ensure the two endmember structures are compatible
        matcher = structure_matcher.StructureMatcher(
            scale=True,
            attempt_supercell=True,
            primitive_cell=False,
            comparator=structure_matcher.FrameworkComparator(),
        )
        try:
            struc_B = matcher.get_s2_like_s1(struc_A, struc_B)
        except ValueError:
            struc_A = matcher.get_s2_like_s1(struc_B, struc_A)

        assert (
            struc_A is not None and struc_B is not None
        ), "The endmember structures you provided cannot be matched!"

        struc_A.remove_oxidation_states()
        struc_B.remove_oxidation_states()

        # Determine which elements differ between the two endmembers
        elements_A = set([str(el) for el in struc_A.composition.elements])
        elements_B = set([str(el) for el in struc_B.composition.elements])
        differing_elements = elements_A.symmetric_difference(elements_B)

        if len(differing_elements) != 2:
            raise ValueError(
                "The code currently supports systems where only one element differs between the endmembers."
            )

        # Map differing elements to variables
        self.element_A = differing_elements.intersection(elements_A).pop()
        self.element_B = differing_elements.intersection(elements_B).pop()

        # Create dummy structures while saving original species and occupancies
        A_species, B_species = [], []

        for index, site in enumerate(struc_A):
            site_dict = site.as_dict()
            A_species.append(site_dict["species"][0]["element"])
            site_dict["species"] = [
                {"element": "Li", "oxidation_state": 0.0, "occu": 1.0}
            ]
            struc_A[index] = PeriodicSite.from_dict(site_dict)

        for index, site in enumerate(struc_B):
            site_dict = site.as_dict()
            B_species.append(site_dict["species"][0]["element"])
            site_dict["species"] = [
                {"element": "Li", "oxidation_state": 0.0, "occu": 1.0}
            ]
            struc_B[index] = PeriodicSite.from_dict(site_dict)

        # Interpolate the two structures and ignore the first entry (an end-member)
        interp_structs = struc_A.interpolate(
            struc_B, nimages=self.num_solns, interpolate_lattices=True
        )[1:]

        # Compute fractional occupancies
        soln_interval = 1.0 / (self.num_solns + 1)
        soln_fractions = [(i + 1) * soln_interval for i in range(self.num_solns)]

        # Place fractional occupancies on each site
        for index, (A, B) in enumerate(zip(A_species, B_species)):
            for i in range(self.num_solns):
                site_dict = interp_structs[i][index].as_dict()
                site_dict["species"] = []

                if A == B:
                    site_dict["species"].append(
                        {"element": A, "oxidation_state": 0.0, "occu": 1.0}
                    )
                else:
                    c1 = 1 - soln_fractions[i]
                    c2 = soln_fractions[i]
                    site_dict["species"].append(
                        {"element": A, "oxidation_state": 0.0, "occu": c1}
                    )
                    site_dict["species"].append(
                        {"element": B, "oxidation_state": 0.0, "occu": c2}
                    )

                interp_structs[i][index] = PeriodicSite.from_dict(site_dict)

        self.disordered_solns = interp_structs
        return interp_structs

    def order_disordered_structure(self, disordered_structure: Structure) -> Structure:
        """
        Convert a disordered pymatgen Structure object to an ordered one
        by assigning species to sites according to their fractional occupancies.

        For this class, the precise ordering does not matter. But the sqsgen
        package requires the structure to be ordered. All that matters is the
        relative number of each species on the sites to be disordered, which is
        automatically determined by the fractional occupancies in disordered_structure.

        Args:
            disordered_structure (Structure): The input disordered structure.

        Returns:
            Structure: An ordered version of the input structure.
        """
        # Initialize an empty structure with the same lattice
        ordered_structure: Structure = Structure(disordered_structure.lattice, [], [])

        disordered_sites: List[PeriodicSite] = []
        total_occupancy: Dict[str, float] = defaultdict(float)

        # Separate ordered and disordered sites
        for site in disordered_structure:
            if site.is_ordered:
                # Ordered site: add directly to the new structure
                ordered_structure.append(site.species, site.frac_coords)
            else:
                # Disordered site: collect for processing
                disordered_sites.append(site)
                for specie, occupancy in site.species.items():
                    total_occupancy[str(specie)] += occupancy

        # Total number of disordered sites
        num_disordered_sites: int = len(disordered_sites)

        # Calculate the number of each species to assign
        n_specie: Dict[str, int] = {}
        fractional_part: Dict[str, float] = {}
        for specie, total_occ in total_occupancy.items():
            n_specie[specie] = int(total_occ)
            fractional_part[specie] = total_occ - n_specie[specie]

        total_assigned: int = sum(n_specie.values())
        diff: int = num_disordered_sites - total_assigned

        # Adjust the counts to match the total number of disordered sites
        if diff > 0:
            # Need to add species
            for _ in range(diff):
                # Find specie with the largest fractional part
                specie: str = max(fractional_part, key=fractional_part.get)
                n_specie[specie] += 1
                fractional_part[specie] = 0  # Avoid selecting again
        elif diff < 0:
            # Need to remove species
            for _ in range(-diff):
                # Find specie with the smallest fractional part and positive count
                specie_candidates: List[str] = [
                    s for s in fractional_part if n_specie[s] > 0
                ]
                specie: str = min(specie_candidates, key=lambda s: fractional_part[s])
                n_specie[specie] -= 1
                fractional_part[specie] = 1  # Avoid selecting again

        # Create a list of species according to the counts
        species_list: List[str] = []
        for specie, count in n_specie.items():
            species_list.extend([specie] * count)

        # Shuffle the list to distribute species randomly
        random.shuffle(species_list)

        # Assign species to the disordered sites
        for site, specie in zip(disordered_sites, species_list):
            ordered_structure.append({specie: 1.0}, site.frac_coords)

        return ordered_structure

    def generate_ordered_solutions(self) -> List[Structure]:
        """
        Generate ordered solid solutions from the disordered solutions.

        Returns:
            List[Structure]: List of ordered solid solution structures.
        """
        if self.disordered_solns is None:
            self.generate_solid_solutions()

        ordered_solns = []
        for i, interp_struc in enumerate(self.disordered_solns):
            interp_struc.make_supercell(self.supercell_dim)
            inerp_struc = interp_struc.remove_oxidation_states()
            ordered_struc = self.order_disordered_structure(interp_struc)
            ordered_solns.append(ordered_struc)
            ordered_struc.to(
                filename=os.path.join(self.dirs["output"], f"{i}.vasp"), fmt="poscar"
            )

        self.ordered_solns = ordered_solns
        return ordered_solns

    def _parse_structure(
        self, file_path: str
    ) -> Tuple[List[List[float]], List[List[float]], List[str], Structure]:
        """
        Parse a structure file and extract lattice, coordinates, species, and structure object.

        Args:
            file_path (str): Path to the structure file.

        Returns:
            Tuple[List[List[float]], List[List[float]], List[str], Structure]:
                Lattice matrix, fractional coordinates, species list, and Structure object.
        """
        structure = Structure.from_file(file_path)
        structure.remove_oxidation_states()
        lattice = structure.lattice.matrix.tolist()
        species = [str(site.specie) for site in structure]
        coords = [site.frac_coords.tolist() for site in structure]
        return lattice, coords, species, structure

    def _write_output_file(
        self,
        output_path: str,
        lattice: List[List[float]],
        coords: List[List[float]],
        species: List[str],
    ) -> None:
        """
        Write the structure information to a YAML file for SQS generation.

        Args:
            output_path (str): Path to write the output YAML file.
            lattice (List[List[float]]): Lattice matrix.
            coords (List[List[float]]): Fractional coordinates of sites.
            species (List[str]): List of species on each site.
        """
        composition_dict = dict(Counter(species))
        filtered_composition = {
            self.element_A: composition_dict.get(self.element_A, 0),
            self.element_B: composition_dict.get(self.element_B, 0),
        }
        species = [
            self.element_A if specie == self.element_B else specie for specie in species
        ]
        data = {
            "structure": {
                "supercell": [1, 1, 1],
                "lattice": lattice,
                "coords": coords,
                "species": species,
            },
            "iterations": 1e6,
            "shell_weights": {1: 1.0},
            "which": self.element_A,
            "composition": filtered_composition,
        }
        with open(output_path, "w") as file:
            yaml.dump(data, file, default_flow_style=False)

    def generate_sqs(self, sqsgen_path: str = "path/to/sqsgen") -> List[Structure]:
        """
        Generate special quasirandom structures (SQS) from the ordered solutions.

        Args:
            sqsgen_path (str, optional): Path to the sqsgen executable. Defaults to "path/to/sqsgen".

        Returns:
            List[Structure]: List of SQS structures.
        """
        if self.ordered_solns is None:
            self.generate_ordered_solutions()

        # Convert ordered structures to YAML format
        for i, fname in enumerate(os.listdir(self.dirs["output"])):
            input_file = os.path.join(self.dirs["output"], fname)
            output_file = os.path.join(
                self.dirs["yaml"], f"{os.path.splitext(fname)[0]}.yaml"
            )
            lattice, coords, species, structure = self._parse_structure(input_file)
            self._write_output_file(output_file, lattice, coords, species)

        # Generate an SQS from each YAML file
        calc_sqs = [
            os.path.join(sqsgen_path, "sqsgen"),
            "run",
            "iteration",
            "sqs_input.yaml",
        ]
        export_sqs = [
            os.path.join(sqsgen_path, "sqsgen"),
            "export",
            "sqs_input.result.yaml",
        ]
        sqs_solns = []

        for fname in os.listdir(self.dirs["yaml"]):
            # Copy YAML input to temp_dir
            copyfile(
                os.path.join(self.dirs["yaml"], fname),
                os.path.join(self.dirs["temp"], "sqs_input.yaml"),
            )
            os.chdir(self.dirs["temp"])

            # Run sqsgen and export CIF(s)
            subprocess.run(
                calc_sqs,
                check=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )
            subprocess.run(
                export_sqs,
                check=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )

            # Copy one SQS to the sqs_dir
            cif_filenames = [
                filename for filename in os.listdir(".") if filename.endswith(".cif")
            ]
            struc = Structure.from_file(cif_filenames[0])
            num_struc = fname.split(".")[0]
            struc.to(filename=f'../{self.dirs["sqs"]}/{num_struc}.vasp', fmt="poscar")
            sqs_solns.append(struc)

            # Clean up by removing all files in temp_dir
            for existing_fname in os.listdir("."):
                os.remove(existing_fname)

            os.chdir("..")

        self.sqs_solns = sqs_solns
        return sqs_solns

    def cleanup(self) -> None:
        """
        Remove all directories created during the process.
        """
        for dir_name in self.dirs.values():
            rmtree(dir_name)

    def run(
        self, sqsgen_path: str = "path/to/sqsgen", cleanup: bool = True
    ) -> Tuple[List[Structure], List[Structure], List[Structure]]:
        """
        Run the entire solid solution generation process.

        Args:
            sqsgen_path (str, optional): Path to the sqsgen executable. Defaults to "path/to/sqsgen".
            cleanup (bool, optional): Whether to remove temporary directories after completion. Defaults to True.

        Returns:
            Tuple[List[Structure], List[Structure], List[Structure]]:
                Lists of disordered, ordered, and SQS structures.
        """
        self.generate_solid_solutions()
        self.generate_ordered_solutions()
        self.generate_sqs(sqsgen_path)

        if cleanup:
            self.cleanup()

        return self.disordered_solns, self.ordered_solns, self.sqs_solns

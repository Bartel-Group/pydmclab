from __future__ import annotations

import os
import subprocess
import yaml
import json
import random
import warnings

from typing import TYPE_CHECKING
from collections import Counter, defaultdict
from shutil import copyfile, rmtree
from typing import Any, List, Tuple, Dict, Literal, Optional, Union

import numpy as np

from math import lcm
from pathlib import Path

from pymatgen.core import Structure, PeriodicSite, Composition
from pymatgen.core.surface import SlabGenerator
from pymatgen.transformations.standard_transformations import (
    OrderDisorderedStructureTransformation,
    AutoOxiStateDecorationTransformation,
    OxidationStateDecorationTransformation,
)
from pymatgen.analysis import structure_matcher
from pymatgen.analysis.structure_matcher import StructureMatcher
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
from pymatgen.analysis.diffraction.xrd import XRDCalculator

from sqsgenerator import load_result_pack


from pydmclab.core import struc as pydmc_struc
from pydmclab.core.comp import CompTools

if TYPE_CHECKING:
    from pydmclab.core.energies import ChemPots


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
        scaling_factor: int = 1000,
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

            scaling_factor (int)
                (n_strucs x scaling_factor) structures are initially generated
                to ensure sufficient sampling of ordered strucs

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
        return_ranked_list = n_strucs * scaling_factor if n_strucs > 1 else False

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
            return {i: strucs[i] for i in range(n_strucs) if i in strucs}
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

    def get_xrd_pattern(
        self,
        wavelength: str | float = "CuKa",
        symprec: float = 0,
        debye_waller_factors: dict | None = None,
        scaled: bool = True,
        two_theta_range: Tuple[float, float] | None = (0, 90),
    ) -> dict[str, list]:
        """
        Args:
            wavelength: wavelength of X-ray radiation
                (see pymatgen.analysis.diffraction.xrd.XRDCalculator for str options)
            symprec: symmetry precision for structure refinement (no refinement if 0)
            debye_waller_factors: {element symbol: float} for specifying Debye-Waller factors
            scaled: whether to scale the intensities to a max of 100 (True) or use absolute values (False)
            two_theta_range: range of two-theta values to calculate

        Returns:
            dict of XRD pattern
                {'two_thetas' : numpy array of two-theta values in degrees (numpy.float64),
                 'intensities' : numpy array of intensities (numpy.float64),
                 'hkl_and_multiplicity' : list of {'hkl': miller indices, 'multiplicity': multiplicity},
                 'd_spacing' : list of d-spacings (numpy.float64)}
        """
        # setup the calculator
        xrd_calculator = XRDCalculator(
            wavelength=wavelength,
            symprec=symprec,
            debye_waller_factors=debye_waller_factors,
        )
        # find the pattern
        xrd_pattern_of_struc = xrd_calculator.get_pattern(
            self.structure, scaled=scaled, two_theta_range=two_theta_range
        )
        # process the data
        xrd_hkls = [hkl[0] for hkl in xrd_pattern_of_struc.hkls]
        return {
            "two_thetas": xrd_pattern_of_struc.x,
            "intensities": xrd_pattern_of_struc.y,
            "hkl_and_multiplicity": xrd_hkls,
            "d_spacing": xrd_pattern_of_struc.d_hkls,
        }

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

        out = {miller_str: {}, "bulk_template": bulk.as_dict()}
        for i, slab in enumerate(slabs):
            out[miller_str][i] = {}
            out[miller_str][i]["slab"] = slab.as_dict()
            out[miller_str][i]["vacuum_size"] = min_vacuum_size
            out[miller_str][i]["slab_size"] = min_slab_size
            out[miller_str][i]["center_slab"] = center_slab
            out[miller_str][i]["in_unit_planes"] = in_unit_planes
            out[miller_str][i]["reorient_lattice"] = reorient_lattice
            out[miller_str][i]["symmetrize"] = symmetrize
            out[miller_str][i]["tolerance"] = tolerance
            out[miller_str][i]["ftolerance"] = ftolerance
            out[miller_str][i]["max_normal_search"] = max_normal_search

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
            if "oxidation_state" in entry and entry["oxidation_state"] is not None:
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
    """Generate quasi-random solid solutions (SQS) between two crystal structures.

    This class uses the sqsgenerator package to create special quasi-random
    structures that mimic random solid solutions while maintaining specific
    short-range order parameters.

    Attributes:
        endmembers: List of two endmember structures.
        supercell_dim: Dimensions of the supercell.
        element_a: Element that differs in the first endmember.
        element_b: Element that differs in the second endmember.
        disordered_solns: List of disordered solid solution structures.
        ordered_solns: List of ordered solid solution structures.
        sqs_solns: List of special quasirandom structures (SQS).
        sqs_data: List of dictionaries containing SQS results data.
        dirs: Dictionary of directory paths used in the process.

    Example:
        >>> from pymatgen.core import Structure
        >>> struc_a = Structure.from_file('endmember_a.cif')
        >>> struc_b = Structure.from_file('endmember_b.cif')
        >>> generator = SolidSolutionGenerator(
        ...     endmembers=[struc_a, struc_b],
        ...     supercell_dim=[2, 1, 2]
        ... )
        >>> disordered, ordered, sqs, data = generator.run()
    """

    def __init__(
        self,
        endmembers: List[Structure],
        supercell_dim: Optional[List[int]] = None,
    ) -> None:
        """Initialize the SolidSolutionGenerator.

        Args:
            endmembers: List of exactly two endmember structures.
            supercell_dim: Dimensions of the supercell. Defaults to [2, 2, 2].

        Raises:
            ValueError: If endmembers list doesn't contain exactly 2 structures.
        """
        if len(endmembers) != 2:
            raise ValueError(
                f"Expected exactly 2 endmembers, got {len(endmembers)}"
            )

        self.endmembers = endmembers
        self.supercell_dim = supercell_dim or [2, 2, 2]

        # Initialize attributes that will be set during processing
        self.element_a: Optional[str] = None
        self.element_b: Optional[str] = None
        self.disordered_solns: Optional[List[Structure]] = None
        self.ordered_solns: Optional[List[Structure]] = None
        self.sqs_solns: Optional[List[Structure]] = None
        self.sqs_data: Optional[List[Dict[str, Any]]] = None
        self.num_solns: Optional[int] = None

        # Create necessary directories
        self.dirs: Dict[str, str] = {
            "output": "Ordered_Solutions",
            "json": "json_files",
            "sqs": "SQS",
            "temp": "working_dir",
        }
        self._create_directories()

    def _create_directories(self) -> None:
        """Create necessary working directories."""
        for dir_path in self.dirs.values():
            Path(dir_path).mkdir(exist_ok=True)

    def generate_solid_solutions(self) -> List[Structure]:
        """
        Generate disordered solid solutions between the two endmember structures.
        The number of solutions is automatically determined based on the number of
        sites that differ between the matched endmembers in the supercell.

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
        self.element_a = differing_elements.intersection(elements_A).pop()
        self.element_b = differing_elements.intersection(elements_B).pop()

        # Count the number of differing sites in the supercell
        struc_A_super = struc_A.copy()
        struc_B_super = struc_B.copy()
        struc_A_super.make_supercell(self.supercell_dim)
        struc_B_super.make_supercell(self.supercell_dim)
        
        num_differing_sites = 0
        for site_A, site_B in zip(struc_A_super, struc_B_super):
            if str(site_A.specie) != str(site_B.specie):
                num_differing_sites += 1

        # Determine number of solutions automatically
        self.num_solns = num_differing_sites
        print(f"Automatically determined {self.num_solns} intermediate compositions based on differing sites in supercell {self.supercell_dim}.")

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
        """Convert a disordered Structure to an ordered one.

        Assigns species to sites according to their fractional occupancies.
        The precise ordering is randomized, but the total composition is preserved.
        This is required by the sqsgen package which operates on ordered structures.

        Args:
            disordered_structure: The input disordered structure.

        Returns:
            An ordered version of the input structure with the same composition.
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
        """Generate ordered solid solutions from the disordered solutions.

        Creates supercells and converts disordered structures to ordered ones
        by randomly assigning species according to fractional occupancies.

        Returns:
            List of ordered solid solution structures.
        """
        if self.disordered_solns is None:
            self.generate_solid_solutions()

        ordered_solns = []
        for i, interp_struc in enumerate(self.disordered_solns):
            interp_struc.make_supercell(self.supercell_dim)
            interp_struc = interp_struc.remove_oxidation_states()
            ordered_struc = self.order_disordered_structure(interp_struc)
            ordered_solns.append(ordered_struc)
            ordered_struc.to(
                filename=os.path.join(self.dirs["output"], f"{i}.vasp"), fmt="poscar"
            )

        self.ordered_solns = ordered_solns
        return ordered_solns

    def _parse_structure(
        self, file_path: Union[str, Path]
    ) -> Tuple[List[List[float]], List[List[float]], List[str], Structure]:
        """Parse a structure file and extract components.

        Args:
            file_path: Path to the structure file.

        Returns:
            Tuple containing:
                - Lattice matrix as nested list
                - Fractional coordinates as nested list
                - Species list
                - Structure object
        """
        try:
            structure = Structure.from_file(file_path)
        except Exception as e:
            raise ValueError(f"Failed to read structure from {file_path}: {e}") from e

        structure.remove_oxidation_states()
        lattice = structure.lattice.matrix.tolist()
        species = [str(site.specie) for site in structure]
        coords = [site.frac_coords.tolist() for site in structure]
        return lattice, coords, species, structure

    def _write_json_file(
        self,
        output_path: Union[str, Path],
        lattice: List[List[float]],
        coords: List[List[float]],
        species: List[str],
    ) -> None:
        """Write structure information to a JSON file for SQS generation.

        Args:
            output_path: Path to write the output JSON file.
            lattice: Lattice matrix as nested list.
            coords: Fractional coordinates of sites as nested list.
            species: List of species on each site.
        """
        composition_dict = dict(Counter(species))

        # Get the differing elements that should be optimized in SQS
        if self.element_a is None or self.element_b is None:
            raise ValueError(
                "Differing elements not determined. Call generate_solid_solutions() first."
            )

        element_a_count = composition_dict.get(self.element_a, 0)
        element_b_count = composition_dict.get(self.element_b, 0)

        if element_a_count == 0 or element_b_count == 0:
            raise ValueError(
                f"Structure does not contain both differing elements "
                f"({self.element_a}, {self.element_b})"
            )

        # Create the JSON structure with sublattice mode
        data = {
            "structure": {
                "lattice": lattice,
                "coords": coords,
                "species": species,
                "supercell": [1, 1, 1]
            },
            "iterations": 1000000,
            "sublattice_mode": "split",
            "shell_weights": {
                "1": 1.0,
                "2": 0.5
            },
            "composition": [{
                "sites": [self.element_a, self.element_b],
                self.element_a: element_a_count,
                self.element_b: element_b_count
            }],
            "max_results_per_objective": 5
        }
        
        with open(output_path, "w", encoding="utf-8") as file:
            json.dump(data, file, indent=4)


    def generate_sqs(self) -> Tuple[List[Structure], List[Dict[str, Any]]]:
        """Generate special quasirandom structures (SQS) from ordered solutions.

        Uses the sqsgenerator package to optimize structures for minimal
        short-range order parameters while maintaining the target composition.

        Returns:
            Tuple containing:
                - List of best SQS structures
                - List of dictionaries with SQS optimization data
        """
        if self.ordered_solns is None:
            self.generate_ordered_solutions()

        # Convert ordered structures to JSON format
        for i, fname in enumerate(os.listdir(self.dirs["output"])):
            input_file = os.path.join(self.dirs["output"], fname)
            output_file = os.path.join(
                self.dirs["json"], f"{os.path.splitext(fname)[0]}.json"
            )
            lattice, coords, species, structure = self._parse_structure(input_file)
            self._write_json_file(output_file, lattice, coords, species)

        # Generate an SQS from each JSON file
        sqs_solns = []
        sqs_data = []

        # Filter to only process .json files
        json_files = sorted([f for f in os.listdir(self.dirs["json"]) if f.endswith(".json")])

        for fname in json_files:
            json_path = os.path.join(self.dirs["json"], fname)
            num_struc = fname.split(".")[0]
            
            # Use absolute paths for clarity
            abs_json_path = os.path.abspath(json_path)
            abs_temp_dir = os.path.abspath(self.dirs["temp"])
            
            # Change to temp directory for sqsgen execution
            original_dir = os.getcwd()
            os.chdir(abs_temp_dir)

            try:
                # Run sqsgen
                print(f"Running SQS optimization for composition {num_struc}...")
                subprocess.run(
                    ["sqsgen", "run", "-i", abs_json_path],
                    check=True,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                )

                # Load the result pack for data extraction
                mpack_file = abs_json_path.replace(".json", ".mpack")
                with open(mpack_file, "rb") as f:
                    pack = load_result_pack(f.read())

                # Get the best solution
                best_solution = pack.best()

                # Extract SRO parameters (handle different result types)
                try:
                    sro_full = best_solution.sro()  # Full array
                    sro_pair = best_solution.sro(self.element_a, self.element_b)  # List for each shell
                except AttributeError:
                    # For sublattice mode, SRO parameters may not be available or have different interface
                    sro_full = None
                    sro_pair = None
                
                # Get objective function values
                objectives = []
                for obj, solutions in pack:
                    objectives.append({
                        "objective": float(obj),
                        "num_solutions": len(solutions)
                    })
                
                # Store data
                data_dict = {
                    "composition_index": int(num_struc),
                    "best_objective": float(pack[0][0]),
                    "all_objectives": objectives
                }

                # Add SRO parameters if available
                if sro_full is not None and sro_pair is not None:
                    data_dict["sro_parameters"] = {
                        "full_array": sro_full.tolist(),
                        f"{self.element_a}_{self.element_b}": [float(x) for x in sro_pair]
                    }
                else:
                    data_dict["sro_parameters"] = None
                sqs_data.append(data_dict)

                # Export the best structure using sqsgen output structure
                subprocess.run(
                    ["sqsgen", "output", "-o", mpack_file, "structure", "-f", "cif"],
                    check=True,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                )

                # sqsgen always creates sqs-0-0.cif, so we need to rename it immediately
                default_cif_filename = "sqs-0-0.cif"
                target_cif_filename = f"sqs-{num_struc}-0.cif"

                if os.path.exists(default_cif_filename):
                    os.rename(default_cif_filename, target_cif_filename)
                else:
                    raise FileNotFoundError(f"Expected CIF file {default_cif_filename} was not created by sqsgen")

                # Read with pymatgen and save as VASP
                struc = Structure.from_file(target_cif_filename)
                output_path = os.path.join(original_dir, self.dirs["sqs"], f"{num_struc}.vasp")
                struc.to(filename=output_path, fmt="poscar")
                sqs_solns.append(struc)

                # Clean up CIF file
                os.remove(target_cif_filename)

                print(f"Completed SQS for composition {num_struc}: objective = {data_dict['best_objective']:.6f}")

            except subprocess.CalledProcessError as e:
                print(f"Error running sqsgen for {fname}: {e.stderr}")
            except Exception as e:
                print(f"Error processing {fname}: {e}")
                
            finally:
                # Return to original directory
                os.chdir(original_dir)

        self.sqs_solns = sqs_solns
        self.sqs_data = sqs_data
        
        # Save summary data
        summary_path = os.path.join(self.dirs["sqs"], "sqs_summary.json")
        with open(output_path, "w", encoding="utf-8") as file:
            json.dump(data, file, indent=4)
        
        print(f"\nSQS generation complete! Results saved to {self.dirs['sqs']}/")
        
        return sqs_solns, sqs_data

    def cleanup(self) -> None:
        """
        Remove all directories created during the process.
        """
        for dir_name in self.dirs.values():
            if os.path.exists(dir_name):
                rmtree(dir_name)

    def run(
        self, cleanup: bool = True
    ) -> Tuple[List[Structure], List[Structure], List[Structure], List[Dict[str, Any]]]:
        """Run the entire solid solution generation process.

        Executes the complete workflow: generates disordered solutions,
        converts to ordered structures, optimizes SQS, and optionally cleans up.

        Args:
            cleanup: Whether to remove temporary directories after completion.

        Returns:
            Tuple containing:
                - List of disordered solution structures
                - List of ordered solution structures
                - List of SQS structures
                - List of SQS optimization data dictionaries
        """
        self.generate_solid_solutions()
        self.generate_ordered_solutions()
        self.generate_sqs()

        if cleanup:
            self.cleanup()

        return self.disordered_solns, self.ordered_solns, self.sqs_solns, self.sqs_data

class SlabTools(object):
    """
    A class for manipulating slabs and computing their properties.
    """

    def __init__(
        self,
        slab_structure: Structure | dict | str,
        slab_e_per_at: float,
        *,
        unreduced_bulk_composition: Composition | dict | str = None,
        bulk_e_per_at: float = None,
    ) -> None:
        """
        Initialize the SlabTools object.

        Args:
            slab_structure (Structure | dict | str): pymatgen Structure object, Structure.as_dict(), or path to structure file.
            slab_e_per_at (float): Energy per atom of the slab.
            unreduced_bulk_composition (Composition | dict | str, optional): Unreduced bulk composition. This is the full composition of the bulk from which the slab was cleaved.
                (e.g., if the slab was cleaved from SrTiO3 and is 10 layers thick, the unreduced bulk composition would be Sr10Ti10O30).
            bulk_e_per_at (float, optional): Energy per atom of the bulk. Defaults to None.
        """

        slab_structure = StrucTools(slab_structure).structure

        self.slab_structure = slab_structure
        self.slab_e_per_at = slab_e_per_at

        if isinstance(unreduced_bulk_composition, (dict, str)):
            unreduced_bulk_composition = Composition(unreduced_bulk_composition)

        self.unreduced_bulk_composition = unreduced_bulk_composition
        self.bulk_e_per_at = bulk_e_per_at

    @property
    def is_stoich(self) -> bool:
        """
        Check if the slab is stoichiometric with respect to the bulk composition.
        """
        if not self.unreduced_bulk_composition:
            raise ValueError(
                "Unreduced bulk composition must be provided to check stoichiometry."
            )

        return self.slab_structure.composition == self.unreduced_bulk_composition

    @property
    def off_stoichiometry(self) -> dict[str, float]:
        """
        Calculate the off-stoichiometry of the slab with respect to the bulk composition.
        """
        if not self.unreduced_bulk_composition:
            raise ValueError(
                "Unreduced bulk composition must be provided to calculate off-stoichiometry."
            )

        unreduced_slab_composition = self.slab_structure.composition
        unreduced_slab_composition.allow_negative = True

        return (unreduced_slab_composition - self.unreduced_bulk_composition).as_dict()

    def surface_area(
        self, vacuum_axis: Literal["a", "b", "c", "auto"] = "auto", verbose: bool = True
    ) -> float:
        """
        Returns the surface area of the slab.
        """
        lattice_mattrix = self.slab_structure.lattice.matrix

        if not isinstance(vacuum_axis, str) or vacuum_axis not in [
            "a",
            "b",
            "c",
            "auto",
        ]:
            raise ValueError("vacuum_axis must be one of 'a', 'b', 'c', or 'auto'.")

        axis_lengths = {
            "a": np.linalg.norm(lattice_mattrix[0]),
            "b": np.linalg.norm(lattice_mattrix[1]),
            "c": np.linalg.norm(lattice_mattrix[2]),
        }

        if vacuum_axis != "auto" and axis_lengths[vacuum_axis] != max(
            axis_lengths.values()
        ):
            warnings.warn(
                f"The specified vacuum axis '{vacuum_axis}' is not the longest axis in the lattice.",
                category=UserWarning,
            )

        if vacuum_axis == "a":
            lattice_mattrix = np.delete(lattice_mattrix, 0, axis=0)
        elif vacuum_axis == "b":
            lattice_mattrix = np.delete(lattice_mattrix, 1, axis=0)
        elif vacuum_axis == "c":
            lattice_mattrix = np.delete(lattice_mattrix, 2, axis=0)
        elif vacuum_axis == "auto":
            if verbose:
                print("Auto selected - Vacuum axis will be set to the largest axis.")
            index_of_largest_axis = np.argmax(
                [np.linalg.norm(axis) for axis in lattice_mattrix]
            )
            lattice_mattrix = np.delete(lattice_mattrix, index_of_largest_axis, axis=0)

        surface_area = np.linalg.norm(np.cross(lattice_mattrix[0], lattice_mattrix[1]))

        return surface_area

    def surface_energy(
        self,
        *,
        vacuum_axis: Literal["a", "b", "c", "auto"] = "auto",
        ref_potentials: ChemPots | dict = None,
        verbose: bool = True,
        **kwargs,
    ) -> float:

        if not (self.unreduced_bulk_composition and self.bulk_e_per_at):
            raise ValueError(
                "Unreduced bulk composition and bulk energy per atom must be provided to calculate surface energy."
            )

        slab_e_tot = self.slab_e_per_at * len(self.slab_structure)
        bulk_e_tot = self.bulk_e_per_at * self.unreduced_bulk_composition.num_atoms

        surface_area = self.surface_area(vacuum_axis=vacuum_axis, verbose=verbose)

        if not self.is_stoich:
            from pydmclab.core.energies import ChemPots

            excess_or_deficient_amts = self.off_stoichiometry

            if not ref_potentials:
                temperature = kwargs.pop("temperature", None)
                if not temperature:
                    mus = ChemPots(**kwargs)
                    ref_potentials = mus.chempots
                    ref_potentials = {
                        k: v
                        for k, v in ref_potentials.items()
                        if k in excess_or_deficient_amts
                    }
                else:
                    mus_at_0K = ChemPots(temperature=0, **kwargs)
                    ref_pots_at_0K = mus_at_0K.chempots
                    mus_at_temp = ChemPots(temperature=temperature, **kwargs)
                    ref_pots_at_temp = mus_at_temp.chempots
                    ref_potentials = {
                        el: ref_pots_at_0K[el] + ref_pots_at_temp[el]
                        for el in excess_or_deficient_amts
                    }

            elif isinstance(ref_potentials, ChemPots):
                ref_potentials = ref_potentials.chempots

            missing_refs = set(excess_or_deficient_amts) - set(ref_potentials)
            if missing_refs:
                raise ValueError(
                    f"Reference potentials are missing for the following elements: {missing_refs}."
                )

            delta_mu_sum = sum(
                excess_or_deficient_amts[el] * ref_potentials[el]
                for el in excess_or_deficient_amts
            )

            return (slab_e_tot - bulk_e_tot - delta_mu_sum) / (2 * surface_area)

        return (slab_e_tot - bulk_e_tot) / (2 * surface_area)

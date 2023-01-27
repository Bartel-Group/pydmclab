from pydmc.core.struc import StrucTools, SiteTools
import itertools
import random
from pymatgen.core.structure import Structure
from pymatgen.transformations.site_transformations import (
    ReplaceSiteSpeciesTransformation,
)
from pymatgen.analysis.structure_matcher import StructureMatcher


"""
Using enumlib:
    - install enumlib from https://github.com/msg-byu/enumlib
    - for MAC
        - install xcode ($ xcode-select --install)
        - install gfortran (https://github.com/fxcoudert/gfortran-for-macOS/releases)
    - follow enumlib instructions
    - mkdir */enumlib.bin
    - move */enumlib/src/enum.x and */enumlib/aux_src/makeStr.py to */enumlib/bin
    - add #! /usr/bin/env python to top of */enumlib/bin/makeStr.py
    - add */enumlib/bin to PATH (PATH=$PATH:*/enumlib/bin)
"""


class MagTools(object):
    def __init__(
        self,
        structure,
        max_afm_combos=100,
        afm_spins=(-5, 5),
        fm_spins=(0.6, 5),
        randomize_afm=True,
        treat_as_nm=[],
    ):
        """
        Args:
            structure (Structure): pymatgen Structure object
            max_afm_combos (int): maximum number of AFM spin configurations to generate
            afm_spins (tuple): low and high spin for AFM initialization
            fm_spins (tuple): zero and non-zero spin for FM initialization
            randomize_afm (bool): randomize AFM spin configurations
                - randomization occurs in two different steps
                    - when AFM combinations are enumerated,
                        if the number of combinations is greater than max_afm_combos,
                        then we randomly select max_afm_combos combinations
                    - when the unique AFM structures are generated,
                        we randomly order them to avoid having the same "kinds" of AFM orderings always appearing first
                    - random seeds are being used so both operations should be deterministic
            treat_as_nm (list): list of elements to treat as non-magnetic
                - e.g., if you want to only explore various initial configurations for other element(s)
        """
        if isinstance(structure, dict):
            structure = Structure.from_dict(structure)
        structure.remove_oxidation_states()
        self.structure = structure
        self.max_afm_combos = max_afm_combos
        self.fm_spins = fm_spins
        self.afm_spins = afm_spins
        self.randomize_afm = randomize_afm
        self.treat_as_nm = treat_as_nm

    @property
    def magnetic_ions(self):
        """
        Aggregated from MP + matminer (**propably a better list to use somewhere**)

        from_MP = https://github.com/materialsproject/pymatgen/blob/master/pymatgen/analysis/magnetism/default_magmoms.yaml
        from_matminer = https://github.com/hackingmaterials/matminer/blob/main/matminer/featurizers/composition/element.py
        """
        from_matminer = [
            "Ti",
            "V",
            "Cr",
            "Mn",
            "Fe",
            "Co",
            "Ni",
            "Cu",
            "Nb",
            "Mo",
            "Tc",
            "Ru",
            "Rh",
            "Pd",
            "Ag",
            "Ta",
            "W",
            "Re",
            "Os",
            "Ir",
            "Pt",
        ]
        from_MP = [
            "Co",
            "Cr",
            "Fe",
            "Mn",
            "Mo",
            "Ni",
            "V",
            "W",
            "Ce",
            "Eu",
            "Ti",
            "V",
            "Cr",
            "Mn",
            "Fe",
            "Co",
            "Ni",
            "Cu",
            "Pr",
            "Nd",
            "Pm",
            "Sm",
            "Gd",
            "Tb",
            "Dy",
            "Ho",
            "er",
            "Tm",
            "Yb",
            "Np",
            "Ru",
            "Os",
            "Ir",
            "U",
        ]
        magnetic = list(set(from_matminer + from_MP))
        treat_as_nm = self.treat_as_nm
        return sorted([el for el in magnetic if el not in treat_as_nm])

    @property
    def magnetic_ions_in_struc(self):
        """
        list of elements (str) in structure that are magnetic
        """
        els = StrucTools(self.structure).els
        magnetic_ions = self.magnetic_ions
        return sorted(list(set([el for el in els if el in magnetic_ions])))

    @property
    def could_be_magnetic(self):
        """
        True if any magnetic ions are in structure else False
        """
        return True if len(self.magnetic_ions_in_struc) > 0 else False

    @property
    def could_be_afm(self):
        """
        True if there are at least two magnetic sites in structure
        """
        if not self.could_be_magnetic:
            return False

        magnetic_ions = self.magnetic_ions_in_struc
        magnetic_sites = 0
        structure = self.structure
        for idx in range(len(structure)):
            el = SiteTools(structure, idx).el
            if el in magnetic_ions:
                magnetic_sites += 1
            if magnetic_sites > 1:
                if magnetic_sites % 2 == 0:
                    return True

    @property
    def get_nonmagnetic_structure(self):
        """
        Returns nonmagnetic Structure with magmom of zeros
        """
        s = self.structure
        magmom = [0 for i in range(len(s))]
        s_tmp = s.copy()
        s_tmp.add_site_property("magmom", magmom)
        return s_tmp

    @property
    def get_ferromagnetic_structure(self):
        """
        Returns Structure with all magnetic ions ordered ferromagnetically
            - nonmagnetic ions are given spin = spins[0] (default: 0.6)
            - magnetic ions are given spin = spins[1] (default: 5)
        """
        spins = self.fm_spins
        magnetic_ions_in_struc = self.magnetic_ions_in_struc
        if len(magnetic_ions_in_struc) == 0:
            return None
        s = self.structure
        # magnetic_sites = [i for i in range(len(s)) if s[i].species_string in magnetic_ions_in_struc]
        magmom = [
            spins[0] if s[i].species_string not in magnetic_ions_in_struc else spins[1]
            for i in range(len(s))
        ]
        s_tmp = s.copy()
        s_tmp.add_site_property("magmom", magmom)
        return s_tmp

    @property
    def get_antiferromagnetic_structures(self):
        """
        This is a chaotic way to get antiferromagnetic configurations
            - but it doesn't require enumlib interaction with pymatgen
            - it seems reasonably efficient, might break down for large/complex structures
            - note 1: it has no idea which configurations are "most likely" to be low energy
            - note 2: it may require execution on MSI compute nodes

        Basic workflow:
            - start from the NM structure
            - for all sites containing ions in magnetic_ions
                - generate all possible combinations of 0 (spin down) or 1 (spin up) for each site
                    - if I had four sites w/ mag ions this might be: [(0,0,0,1), (0,0,1,1), ...]
                - retain only the combinations that have average = 0.5 (ie half spin down, half spin up)
            - now apply all these combinations to the structure
                - generate a new structure for each combination that puts max(spin) on sites with 1 and min(spin) on sites with 0
            - now figure out which newly generated structures are symmetrically distinct
                - change the identities of sites that are spin up/down using oxidation state surrogate
                    - these ox states aren't physically meaningful, just a placeholder
                    - spin up: 8+, spin down: 8-
            - now use StructureMatcher to find unique structures to return

        Returns:
            list of unique Structure objects with antiferromagnetic ordering
                - exhaustive if len(combos) <= max_combos
                - no idea which are most likely to be low energy
                - reasonable to randomly sample if a very large list

            in general, probably unnecessary to calculate more than ~2-10 of these per structure
        """

        # parameters that could be args...
        spins = self.afm_spins  # magnitudes of high/low spin
        max_combos = (
            self.max_afm_combos
        )  # max number of combinations to try (for cases with very many possible combos)

        # which ions in structure are magnetic:
        magnetic_ions_in_struc = self.magnetic_ions_in_struc
        if len(magnetic_ions_in_struc) == 0:
            return None

        strucs_with_magmoms = []
        s = self.get_nonmagnetic_structure

        # get sites w/ magnetic ions
        magnetic_sites = [
            i for i in range(len(s)) if s[i].species_string in magnetic_ions_in_struc
        ]

        print("%i magnetic sites" % len(magnetic_sites))
        # enumerate all possible ways to yield afm ordering
        combos = itertools.product(range(len(spins)), repeat=len(magnetic_sites))
        combos = list(combos)
        print("%i combos" % len(combos))
        combos = [c for c in combos if sum(c) == 0.5 * len(magnetic_sites)]
        print("%i afm combos" % len(combos))

        # randomly reduce list if too big for practical usage
        if len(combos) > max_combos:
            combos = random.Random(0).sample(combos, max_combos)
            print("%i afm combos" % len(combos))

        # decorate structures w/ magmoms for all afm orderings
        for j in range(len(combos)):
            c = combos[j]
            magnetic_moments = [spins[c[i]] for i in range(len(c))]
            site_idxs_to_magmom = dict(zip(magnetic_sites, magnetic_moments))
            for i in range(len(s)):
                if i not in magnetic_sites:
                    site_idxs_to_magmom[i] = 0
            magmom = [site_idxs_to_magmom[i] for i in range(len(s))]
            s_tmp = s.copy()
            s_tmp.add_site_property("magmom", magmom)
            # print(s_tmp.site_properties)
            strucs_with_magmoms.append(s_tmp)
        print("made strucs")

        # replace spin-up and spin-down sites with new species for symmetry matching
        fake_strucs = []
        for struc in strucs_with_magmoms:
            magmom = struc.site_properties["magmom"]
            spin_up = [i for i in magnetic_sites if magmom[i] == spins[1]]
            spin_down = [i for i in magnetic_sites if magmom[i] == spins[0]]
            indices_species_map = {
                idx: struc[idx].species_string + "8+"
                if idx in spin_up
                else struc[idx].species_string + "8-"
                for idx in spin_up + spin_down
            }
            rsst = ReplaceSiteSpeciesTransformation(indices_species_map)
            s_tmp = rsst.apply_transformation(struc)
            fake_strucs.append(s_tmp)
        fake_strucs_dict = dict(zip(list(range(len(fake_strucs))), fake_strucs))

        matcher = StructureMatcher()
        groups = matcher.group_structures(
            [fake_strucs_dict[i] for i in fake_strucs_dict]
        )
        unique_strucs = []
        for g in groups:
            s = g[0]
            s.remove_oxidation_states()
            unique_strucs.append(s)

        #        print('%i unique afm structures' % len(unique_strucs))

        if self.randomize_afm:
            random.Random(1).shuffle(unique_strucs)

        print("%i unique afm structures" % len(unique_strucs))
        return unique_strucs

    @property
    def get_afm_magmoms(self):
        """
        Returns:
            dict of magmoms for each AFM ordering for a given structure
                {idx of configuration (int) : magmoms (list)}
        """
        afm_strucs = self.get_antiferromagnetic_structures
        magmoms = {}
        if afm_strucs:
            for i in range(len(afm_strucs)):
                magmoms[i] = afm_strucs[i].site_properties["magmom"]

        return magmoms


def main():

    return


if __name__ == "__main__":
    out = main()

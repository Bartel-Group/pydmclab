import numpy as np
import math
from itertools import combinations

from pydmc.data.thermochem import (
    mp2020_compatibility_dmus,
    mus_at_0K,
    mus_at_T,
    mus_from_mp_no_corrections,
)
from pydmc.data.features import atomic_masses
from pydmc.core.comp import CompTools
from pydmc.utils.handy import eVat_to_kJmol, kJmol_to_eVat


class ChemPots(object):
    """
    return dictionary of chemical potentials {el : chemical potential (eV/at)} based on user inputs
    """

    def __init__(
        self,
        temperature=0,
        xc="gga",
        functional="pbe",
        standard="dmc",
        partial_pressures={},  # atm
        diatomics=["H", "N", "O", "F", "Cl"],
        oxide_type="oxide",
        R=8.6173303e-5,  # eV/K
        user_chempots={},
        user_dmus={},
    ):
        """
        Args:
            temperature (int) - temperature in Kelvin
            xc (str) - xc for DFT calculations
            functional (str) - explicit functional for DFT claculations (don't include +U in name)
            standard (str) - standard for DFT calculations
            partial_pressures (dict) - {el (str) : partial pressure (atm)}
                - adjusts chemical potential of gaseous species based on RTln(p/p0)
            diatomics (list) - list of diatomic elements
                - if el is in diatomics, will use 0.5 * partial pressure effecton mu
            oxide_type (str) - type of oxide
                - this only affects MP Formation energies
                - they use different corrections for oxides, peroxides, and superoxides
            user_chempots (dict) - {el (str) : chemical potential (eV/at)}
                - specifies the chemical potential you want to use for el
                - will override everything
            user_dmus (dict) - {el (str) : delta_mu (eV/at)}
                - specifies the change in chemical potential you want to use for el
                - will override everything except user_chempots
        """
        self.temperature = temperature
        self.xc = xc
        self.functional = functional
        self.standard = standard
        self.partial_pressures = partial_pressures
        self.diatomics = diatomics
        self.oxide_type = oxide_type
        self.R = R
        if standard == "mp":
            mp_dmus = mp2020_compatibility_dmus()
            for el in mp_dmus["anions"]:
                user_dmus[el] = -mp_dmus["anions"][el]
            if xc == "ggau":
                for el in mp_dmus["U"]:
                    user_dmus[el] = -mp_dmus["U"][el]
            if self.oxide_type == "peroxide":
                user_dmus[el] = --mp_dmus["peroxide"]["O"]
            elif self.oxide_type == "superoxide":
                user_dmus[el] = --mp_dmus["superoxide"]["O"]

        self.user_dmus = user_dmus
        self.user_chempots = user_chempots

    @property
    def chempots(self):
        """
        Returns:
            dictionary of chemical potentials {el : chemical potential (eV/at)} based on user inputs
        """

        if self.temperature == 0:
            if (self.standard == "dmc") or ("meta" in self.xc):
                all_mus = mus_at_0K()
                els = sorted(list(all_mus[self.standard][self.functional].keys()))
                mus = {
                    el: all_mus[self.standard][self.functional][el]["mu"] for el in els
                }
            elif (self.standard == "mp") and ("meta" not in self.xc):
                mus = mus_from_mp_no_corrections()
        else:
            allowed_Ts = list(range(300, 2100, 100))
            if self.temperature not in allowed_Ts:
                raise ValueError("Temperature must be one of %s" % allowed_Ts)
            all_mus = mus_at_T()
            mus = all_mus[str(self.temperature)]

        if self.partial_pressures:
            for el in self.partial_pressures:
                if el in self.diatomics:
                    factor = 1 / 2
                else:
                    factor = 1
                mus[el] += (
                    -self.R
                    * self.temperature
                    * factor
                    * np.log(self.partial_pressures[el])
                )
        if self.user_dmus:
            for el in self.user_dmus:
                mus[el] += self.user_dmus[el]
        if self.user_chempots:
            for el in self.user_chempots:
                mus[el] = self.user_chempots[el]

        return mus


class FormationEnergy(object):
    """
    TO DO:
        - write tests/demo
    """

    def __init__(
        self,
        formula,
        E_DFT,  # eV/at
        chempots,  # from ThermoTools.ChemPots.chempots
        structure=False,
        atomic_volume=False,
        override_Ef_0K=False,
    ):

        """
        Args:
            formula (str) - chemical formula
            E_DFT (float) - DFT energy (eV/at)
            chempots (dict) - {el (str) : chemical potential (eV/at)}
                - probably generated using ChemPots.chempots

            Only required for getting temperature-dependent formation energies:
                structure (Structure) - pymatgen structure object
                atomic_volume (float) - atomic volume (A^3/atom)
                override_Ef_0K (float) - formation energy at 0 K (eV/at)
                    - if False, compute Ef_0K using FormationEnergy.Ef_0K
        """
        self.formula = CompTools(formula).clean
        self.E_DFT = E_DFT
        self.chempots = chempots
        self.structure = structure
        self.atomic_volume = atomic_volume
        self.override_Ef_0K = override_Ef_0K

    @property
    def weighted_elemental_energies(self):
        """
        Returns:
            weighted elemental energies (eV per formula unit)
        """
        mus = self.chempots
        els_to_amts = CompTools(self.formula).amts
        return np.sum([mus[el] * els_to_amts[el] for el in els_to_amts])

    @property
    def Ef_0K(self):
        """
        Returns:
            formation energy at 0 K (eV/at)
        """
        if self.override_Ef_0K:
            return self.override_Ef_0K
        formula = self.formula
        n_atoms = CompTools(formula).n_atoms
        weighted_elemental_energies = self.weighted_elemental_energies
        E_per_fu = self.E_DFT * n_atoms
        return (1 / n_atoms) * (E_per_fu - weighted_elemental_energies)

    @property
    def reduced_mass(self):
        """
        Returns weighted reduced mass of composition
            - only needed for G(T) see Chris B Nature Comms 2019
        """
        names = CompTools(self.formula).els
        els_to_amts = CompTools(self.formula).amts
        nums = [els_to_amts[el] for el in names]
        mass_d = atomic_masses()
        num_els = len(names)
        num_atoms = np.sum(nums)
        denom = (num_els - 1) * num_atoms
        if denom <= 0:
            print("descriptor should not be applied to unary compounds (elements)")
            return np.nan
        masses = [mass_d[el] for el in names]
        good_masses = [m for m in masses if not math.isnan(m)]
        if len(good_masses) != len(masses):
            for el in names:
                if math.isnan(mass_d[el]):
                    print("I dont have a mass for %s..." % el)
                    return np.nan
        else:
            pairs = list(combinations(names, 2))
            pair_red_lst = []
            for i in range(len(pairs)):
                first_elem = names.index(pairs[i][0])
                second_elem = names.index(pairs[i][1])
                pair_coeff = nums[first_elem] + nums[second_elem]
                pair_prod = masses[first_elem] * masses[second_elem]
                pair_sum = masses[first_elem] + masses[second_elem]
                pair_red = pair_coeff * pair_prod / pair_sum
                pair_red_lst.append(pair_red)
            return np.sum(pair_red_lst) / denom

    def dGf(self, temperature=0):
        """
        Args:
            temperature (int) - temperature (K)
        Returns:
            formation energy at temperature (eV/at)
                - see Chris B Nature Comms 2019
        """
        T = temperature
        Ef_0K = self.Ef_0K
        if T == 0:
            return Ef_0K
        else:
            Ef_0K = self.Ef_0K
            m = self.reduced_mass
            if self.atomic_volume:
                V = self.atomic_volume
            elif self.structure:
                V = self.structure.volume / len(self.structure)
            else:
                raise ValueError("Need atomic volume or structure to compute G(T)")

            Gd_sisso = (
                (-2.48e-4 * np.log(V) - 8.94e-5 * m / V) * T + 0.181 * np.log(T) - 0.882
            )
            weighted_elemental_energies = self.weighted_elemental_energies
            G = Ef_0K + Gd_sisso
            n_atoms = CompTools(self.formula).n_atoms

            return (1 / n_atoms) * (G * n_atoms - weighted_elemental_energies)

class ReactionEnergy(object):
    
    def __init__(self,
                 formation_energies,
                 reactants,
                 products,
                 open_to=[],
                 norm='atom',
                 allowed_filler=['O2', 'N2']):
        """        

        Args:
            formation_energies (dict): {formula (str): formation energy (eV/at)}
                - formation energies should account for chemical potentials (e.g., due to partial pressures)
            reactants (dict): {formula (str) : stoichiometry (int)}
            products (dict): {formula (str) : stoichiometry (int)}
            open_to (list): list of elements to be considered "open" in the reaction. Defaults to None.
            norm (str, dict): if 'atom', then calculate reaction energy per atom of products formed
                - otherwise, specify a basis like: {'O' : 3} to normalize per three moles of O in the products formed
        """
        
        self.formation_energies = formation_energies
        self.reactants = reactants
        self.products = products
        self.open_to = open_to
        self.norm = norm
        
    @property
    def species(self):
        """
        puts the reactants and products in the same dictionary

        Returns:
            {formula (str) : {'side' : 'left' for reactants, 'right' for products},
                              'amt' : stoichiometry (float) in reaction}}
        """
        species = {}
        reactants, products = self.reactants, self.products
        energies = self.formation_energies
        for r in reactants:
            species[CompTools(r).clean] = {'side' : 'left',
                                            'amt' : reactants[r],
                                            'Ef' : energies[r]}
        for p in products:
            species[CompTools(p).clean] = {'side' : 'right',
                                             'amt' : products[p],
                                            'Ef' : energies[p]}
        return species
    
    def check_species_balance(self, species):
        """
        Args:
            species (dict): {formula (str) : {'side' : 'left' for reactants, 'right' for products},
                              'amt' : stoichiometry (float) in reaction}}
        Returns:
            {element (str) : 0 if balanced, else < 0 if more on left, > 0 if more on right}
        """
        
        involved_elements = [CompTools(formula).els for formula in species]
        involved_elements = sorted(list(set([item for sublist in involved_elements for item in sublist])))
        balance = {}
        for el in involved_elements:
            left, right = 0, 0
            for formula in species:
                if el in CompTools(formula).els:
                    if species[formula]['side'] == 'left':
                        left += CompTools(formula).els[el] * species[formula]['amt']
                    elif species[formula]['side'] == 'right':
                        right += CompTools(formula).els[el] * species[formula]['amt']
            balance[el] = left + right

        return left + ' --> ' + right
        
    @property
    def E_rxn(self):
        species = self.species
        dE_rxn = 0
        for formula in species:
            if CompTools(formula).n_els == 1:
                continue

            if species[formula]['side'] == 'left':
                sign = -1
            elif species[formula]['side'] == 'right':
                sign = 1
            else:
                raise ValueError
            coef = species[formula]['amt']
            Ef = species[formula]['Ef']
            Ef = eVat_to_kJmol(Ef, formula)
            dE_rxn += sign*coef*Ef

        return dE_rxn
    
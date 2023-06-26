import numpy as np
import math
from itertools import combinations

from pydmclab.data.thermochem import (
    mp2020_compatibility_dmus,
    mus_at_0K,
    mus_at_T,
    mus_from_mp_no_corrections,
    mus_from_bartel2019_npj,
)
from pydmclab.data.features import atomic_masses
from pydmclab.core.comp import CompTools
from pydmclab.core.struc import StrucTools

from pymatgen.analysis.reaction_calculator import Reaction
from pymatgen.core.composition import Composition


class ChemPots(object):
    """
    return dictionary of chemical potentials {el : chemical potential (eV/at)} based on user inputs
    """

    def __init__(
        self,
        temperature=0,
        functional="pbe",
        standard="dmc",
        partial_pressures={},  # atm
        diatomics=["H", "N", "O", "F", "Cl"],
        oxide_type="oxide",
        user_chempots={},
        user_dmus={},
    ):
        """
        Args:
            temperature (int)
                temperature in Kelvin
                    if T > 0, will use experimental data from Barin's thermochemical data of pure substances

            functional (str)
                explicit functional for DFT claculations (don't include + in name)
                    currently supports "pbe" (for GGA), "pbeu" (for GGA+U), "r2scan" (for meta-GGA)

            standard (str)
                standard for DFT calculations
                    currently supports "dmc" (for DMC) and "mp" (for Materials Project)

            partial_pressures (dict)
                {el (str) : partial pressure (atm)}
                    adjusts chemical potential of gaseous species based on RTln(p/p0)
                        where p0 = 1 atm

            diatomics (list)
                list of diatomic elements
                    if el is in diatomics, will use 0.5 * partial pressure effect on mu

            oxide_type (str)
                type of oxide
                    this only affects MP formation energies
                    they use different corrections for oxides, peroxides, and superoxides

            user_chempots (dict)
                {el (str) : chemical potential (eV/at)}
                    specifies the chemical potential you want to use for el
                    will override everything

            user_dmus (dict)
                {el (str) : delta_mu (eV/at)}
                    specifies the change in chemical potential you want to use for el
                    will be added on top of everything except user_chempots
        """
        self.temperature = temperature
        self.functional = functional
        self.standard = standard
        self.partial_pressures = partial_pressures
        self.diatomics = diatomics
        self.oxide_type = oxide_type
        self.user_dmus = user_dmus
        self.user_chempots = user_chempots

    @property
    def apply_mp_corrections(self):
        """
        Returns:
            updates user_dmus to include MP corrections
        """
        user_dmus = self.user_dmus.copy()

        # load the data extracted in ~2022 (from 2020 compatibility scheme)
        mp_dmus = mp2020_compatibility_dmus()

        # shift the anions MP wants to shift
        for el in mp_dmus["anions"]:
            user_dmus[el] = -mp_dmus["anions"][el]

        # apply U corrections
        if self.functional == "pbeu":
            for el in mp_dmus["U"]:
                user_dmus[el] = -mp_dmus["U"][el]

        # apply different kinds of oxide corrections
        if self.oxide_type == "peroxide":
            user_dmus[el] = -mp_dmus["peroxide"]["O"]
        elif self.oxide_type == "superoxide":
            user_dmus[el] = -mp_dmus["superoxide"]["O"]

        self.user_dmus = user_dmus.copy()

    @property
    def chempots(self):
        """
        Returns:
            dictionary of chemical potentials
                {el : chemical potential (eV/at)} based on user inputs
        """
        T = self.temperature
        standard, functional = self.standard, self.functional
        if T == 0:
            # use DFT data at 0 K
            if (standard == "dmc") or (functional in ["scan", "r2scan"]):
                # use DMC data
                all_mus = mus_at_0K()
                els = sorted(list(all_mus[functional].keys()))
                mus = {el: all_mus[functional][el]["mu"] for el in els}
            else:
                # use MP data
                mus = mus_from_mp_no_corrections()
        else:
            # use experimental data at T > 0 K
            allowed_Ts = list(range(300, 2100, 100))
            if T not in allowed_Ts:
                raise ValueError("Temperature must be one of %s" % allowed_Ts)
            all_mus = mus_at_T()
            mus = all_mus[str(T)].copy()

        # apply partial pressure correction for activity of gaseous elements
        partial_pressures = self.partial_pressures
        diatomics = self.diatomics
        R = 8.6173303e-5  # eV/K

        if partial_pressures:
            for el in partial_pressures:
                if el in diatomics:
                    # correct diatomics b/c they exist as O2 yet their mu is stored as O
                    factor = 1 / 2
                else:
                    factor = 1
                mus[el] += R * T * factor * np.log(partial_pressures[el])

        if (standard == "mp") and (T == 0):
            # apply MP corrections if needed
            self.apply_mp_corrections

        user_dmus = self.user_dmus
        if user_dmus:
            # apply any dmus if needed
            for el in user_dmus:
                mus[el] += user_dmus[el]

        user_chempots = self.user_chempots
        if user_chempots:
            # specify any mus directly
            for el in user_chempots:
                mus[el] = user_chempots[el]

        return mus.copy()


class FormationEnthalpy(object):
    """
    For computing formation energies (~equivalently enthalpies) at 0 K
    """

    def __init__(
        self,
        formula,
        E_DFT,
        chempots,
    ):
        """
        Args:
            formula (str)
                chemical formula

            E_DFT (float)
                DFT energy (eV/atom)
                    per atom should mean per CompTools(formula).n_atoms

            chempots (dict)
                {el (str) : chemical potential (eV/at)}
                    probably generated using ChemPots(...).chempots

        """
        self.formula = CompTools(formula).clean
        self.E_DFT = E_DFT
        self.chempots = chempots

    @property
    def weighted_elemental_energies(self):
        """
        Returns:
            weighted elemental energies (eV per formula unit)

                for Al2O3, this would be 2 * mu_Al + 3 * mu_O
        """
        mus = self.chempots
        els_to_amts = CompTools(self.formula).amts
        for el in els_to_amts:
            if (el not in mus) or not mus[el]:
                raise ValueError('No chemical potential for "%s"' % el)
        return np.sum([mus[el] * els_to_amts[el] for el in els_to_amts])

    @property
    def Ef(self):
        """
        Returns:
            formation energy at 0 K (eV/atom)
                the formation reaction for AxBy is xA + yB --> AxBy
                this reaction energy is computed on a per formula unit (molar) basis
                then divided by the number of atoms (x + y) to get eV/atom formation energies
        """
        formula = self.formula
        n_atoms = CompTools(formula).n_atoms
        weighted_elemental_energies = self.weighted_elemental_energies
        E_per_fu = self.E_DFT * n_atoms
        return (1 / n_atoms) * (E_per_fu - weighted_elemental_energies)


class FormationEnergy(object):
    """
    This class is for computing formation energies at T > 0 K

    By default, uses the Bartel2018 model for vibrational entropy: https://doi.org/10.1038/s41467-018-06682-4

    """

    def __init__(
        self,
        formula,
        Ef,
        chempots,
        structure=False,
        atomic_volume=False,
        x_config=None,
        n_config=1,
        include_Svib=True,
        include_Sconfig=False,
    ):
        """
        Args:
            formula (str)
                chemical formula

            Ef (float)
                DFT formation enthalpy at 0 K (eV/at)
                    or any formation enthalpy at T <= 298 K

            chempots (dict)
                {el (str) : chemical potential (eV/at)}
                    probably generated using ChemPots.chempots

            structure (Structure)
                pymatgen structure object (or Structure.as_dict() or path to structure file)
                    either structure or atomic_volume needed for vibrational entropy calculation

            atomic_volume (float)
                atomic volume (A^3/atom)
                    either structure or atomic_volume needed for vibrational entropy calculation

            x_config (float)
                partial occupancy parameter to compute configurational entropy
                    needed to compute configurational entropy [x in xlnx + (1-x)ln(1-x))]

            n_config (int)
                number of inequivalent sites exhibiting ideal solution behavior
                    this would be one if I have one site that is partially occupied by two ions
                    this would be two if I have two sites that are each partially occupied by two ions

            include_Svib (bool)
                whether to include vibrational entropy (Bartel model)

            include_Sconfig (bool)
                whether to include configurational entropy (ideal mixing model)
        """
        self.formula = CompTools(formula).clean
        self.Ef = Ef
        self.chempots = chempots
        if structure:
            structure = StrucTools(structure).structure
        self.structure = structure
        self.atomic_volume = atomic_volume
        self.include_Svib = include_Svib
        self.include_Sconfig = include_Sconfig
        self.x_config = x_config
        self.n_config = n_config

        if include_Svib:
            if not structure and not atomic_volume:
                raise ValueError(
                    "Must provide structure or atomic volume to compute Svib"
                )

        if include_Sconfig:
            if not (x_config and n_config):
                if x_config != 0:
                    raise ValueError(
                        "Must provide x_config and n_config to compute Sconfig"
                    )

    @property
    def weighted_elemental_energies(self):
        """
        Returns:
            weighted elemental energies (eV per formula unit)
        """
        mus = self.chempots
        els_to_amts = CompTools(self.formula).amts
        for el in els_to_amts:
            if (el not in mus) or not mus[el]:
                raise ValueError('No chemical potential for "%s"' % el)
        return np.sum([mus[el] * els_to_amts[el] for el in els_to_amts])

    @property
    def reduced_mass(self):
        """
        Returns weighted reduced mass of composition
            needed if include_Svib = True (Chris B Nature Comms 2019)
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

    @property
    def S_config(self):
        """
        configurational entropy from ideal mixing model (float, eV/atom/K)
            no short range order
            completely random occupation



        -kB * n_config * (x_config * ln(x_config) + (1-x_config) * ln(1-x_config))
        """
        x, n = self.x_config, self.n_config
        if x in [0, 1]:
            return 0
        kB = 8.617e-5  # eV/K
        S_config = (-kB * n * (x * np.log(x) + (1 - x) * np.log(1 - x))) / CompTools(
            self.formula
        ).n_atoms
        return S_config

    def dGf(self, temperature):
        """
        Args:
            temperature (int)
                temperature (K)

        Returns:
            formation energy at temperature (eV/at)
                see Chris B Nature Comms 2019
        """
        T = temperature
        Ef_0K = self.Ef
        if T == 0:
            # use 0 K formation energy
            return Ef_0K
        if self.include_Svib:
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

            dGf = (1 / n_atoms) * (G * n_atoms - weighted_elemental_energies)

        if self.include_Sconfig:
            if not self.include_Svib:
                # start from 0 K formation energy
                dGf = Ef_0K
            dGf += -T * self.S_config

        return dGf


class DefectFormationEnergy(object):
    def __init__(
        self,
        E_pristine,
        formula_pristine,
        Eg_pristine,
        E_defect,
        formula_defect,
        charge_defect,
        shared_el_for_basis,
        chempots,
        charge_correction,
        gap_discretization=0.1,
    ):
        """
        Args:
            E_pristine (float)
                DFT total energy (or formation energy) of pristine compound (eV/atom)

            formula_pristine (str)
                formula of pristine compound

            Eg_pristine (float)
                band gap of pristine compound (eV)

            E_defect (float)
                DFT total energy (or formation energy) of defect-containing compound (eV/atom)

            formula_defect (str)
                formula of defect-containing compound

            charge_defect (int)
                charge of defect-containing compound

            shared_el_for_basis (str)
                element shared between pristine and defect-containing compounds
                    - for normalizing formula units
                    - e.g., if pristine = Al2O3 and defect = Al4O5S1
                        - shared_el_for_basis = "Al"
                        - so that I know to use Al4O6 for pristine for calculations..

            chempots (dict)
                chemical potentials (eV/atom)
                    {el (str) :
                        chempot (float)
                    for any el exchanged between defect and pristine}

            charge_correction (float)
                charge correction (eV/cell)
                    this gets added before dividing by number of defects

            gap_discretization (float)
                how fine of a grid do you want to calculate the formation energy over
                smaller numbers = more grid points

        Returns:
            shared_el_for_basis
            chempots
            charge_correction
            gap_discretization
            pristine (dict)
                {'E' : total energy per atom in clean formula (eV/atom),
                 'formula' : the clean formula,
                 'Eg' : band gap (eV),
                 'multiplier' : how many times the clean formula is multiplied to get the defect-containing formula.
                 'basis_formula' : the formula of the basis (e.g., Al4O6 for Al2O3 if the defect if Al4O5S1)}
            defect (dict)
                {'E' : total energy per atom in defect-containing formula (eV/atom),
                 'formula' : the clean defect-containing formula,
                 'q' : charge of defect-containing formula}

        """
        # clean formulas
        formula_pristine = CompTools(formula_pristine).clean
        formula_defect = CompTools(formula_defect).clean

        # make dicts for pristine and defect
        pristine = {"E": E_pristine, "formula": formula_pristine, "Eg": Eg_pristine}
        defect = {"E": E_defect, "formula": formula_defect, "q": charge_defect}

        self.shared_el_for_basis = shared_el_for_basis
        self.chempots = chempots
        self.charge_correction = charge_correction

        # map pristine to defect basis
        pristine_multiplier = CompTools(formula_defect).stoich(
            shared_el_for_basis
        ) / CompTools(formula_pristine).stoich(shared_el_for_basis)

        pristine["multiplier"] = pristine_multiplier

        pristine_els = CompTools(pristine["formula"]).els
        pristine_amounts = [
            CompTools(pristine["formula"]).stoich(el) * pristine_multiplier
            for el in pristine_els
        ]
        pristine_basis_formula = ""
        for i, el in enumerate(pristine_els):
            pristine_basis_formula += el + str(int(pristine_amounts[i]))
        pristine["basis_formula"] = pristine_basis_formula

        self.pristine = pristine
        self.defect = defect
        self.gap_discretization = gap_discretization

    @property
    def dE(self):
        """
        Returns:
            the energy difference between the pristine structure and the defective structure
                E[X^q] - E[pristine]
            the basis is the defect-containing formula unit

            e.g.,
                if pristine = Al2O3
                    defect = Al4O5S1
                   E[X^q] = E_Al4O5S1
                   E[pristine] = 2*E_Al2O3 (to make Al4O6)
                   dE = E_Al4O5S1*10 atoms - 2*E_Al2O3*5 atoms

        """
        pristine = self.pristine
        defect = self.defect

        return (
            defect["E"] * CompTools(defect["formula"]).n_atoms
            - pristine["E"]
            * CompTools(pristine["formula"]).n_atoms
            * pristine["multiplier"]
        )

    @property
    def els_exchanged(self):
        """
        Returns:
            dictionary of elements exchanged between pristine and defect-containing compounds
            {el (str) :
                dn (float)}
                dn > 0 --> more of that el in defect than pristine
                dn < 0 --> less of that el in defect than pristine
        """
        pristine = self.pristine
        defect = self.defect
        els = sorted(
            list(
                set(
                    CompTools(pristine["formula"]).els
                    + CompTools(defect["formula"]).els
                )
            )
        )

        dn = {}
        for el in els:
            n_pristine = (
                CompTools(pristine["formula"]).stoich(el) * pristine["multiplier"]
            )
            n_defect = CompTools(defect["formula"]).stoich(el)
            dn[el] = n_defect - n_pristine

        return dn

    @property
    def defect_type(self):
        """
        Returns:
            (str) what type of defect is it?
                - vacancy
                - interstitial
                - substitional

            NOTE: what about antisite defects?
        """
        dn = self.els_exchanged
        removed_from_pristine = []
        added_to_pristine = []
        for el in dn:
            if dn[el] < 0:
                removed_from_pristine.append(el)
            elif dn[el] > 0:
                added_to_pristine.append(el)
        if (len(removed_from_pristine) > 0) and (len(added_to_pristine) > 0):
            defect_type = "substitional"
        elif len(removed_from_pristine) > 0:
            defect_type = "vacancy"
        elif len(added_to_pristine) > 0:
            defect_type = "interstitial"
        else:
            raise ValueError("defect_type not found")

        return defect_type

    @property
    def n_defects(self):
        """
        Returns:
            how many defects (int) (referenced to the defect-containing basis formula)
        """
        dn = self.els_exchanged
        el = [el for el in dn if dn[el] != 0][0]
        return abs(dn[el])

    @property
    def sum_mus(self):
        """
        Returns:
            sum(dn_i * mu_i) for all elements that are exchanged

            e.g., if pristine = Al2O3 and defect = Al4O5S1
                - so basis = Al4O6
                - dn_Al = 4 - 4 = 0
                - dn_O = 5 - 6 = -1
                - dn_S = 1 - 0 = 1
                - so sum_mus = -1 * mu_O + 1 * mu_S

        """
        chempots = self.chempots
        dn = self.els_exchanged
        sum_mus = 0.0
        for el in dn:
            if dn[el] == 0:
                continue
            sum_mus += dn[el] * chempots[el]

        return sum_mus

    @property
    def Efs(self):
        """

        Returns:
            {E_Fermi (float, eV relative to VBM) : defect formation energy (float, eV/defect)}

            Ef = Ef[X^q] - Ef[pristine] + sum(dn_i*mu_i) + q * E_Fermi + charge_correction

            note:
                - the charge correction gets divided by the number of defects
        """
        pristine = self.pristine
        Eg = pristine["Eg"]
        multiplier = pristine["multiplier"]
        gap_discretization = self.gap_discretization
        if Eg > 0:
            E_Fermis = np.arange(0, Eg + gap_discretization, gap_discretization)
        else:
            E_Fermis = np.array([0.0])
        dE = self.dE
        sum_mus = self.sum_mus
        q = self.defect["q"]
        charge_correction = self.charge_correction
        n_defects = self.n_defects
        energies = [
            ((dE - sum_mus) / multiplier + q * E_Fermi + charge_correction) / n_defects
            for E_Fermi in E_Fermis
        ]
        return dict(zip(E_Fermis, energies))


class ReactionEnergy(object):

    """
    *** This is a work in progress ***

    @TODO:
        - write tests/demo
        - incorporate filler
        - incorporate normalization
        - incorporate open systems

    """

    def __init__(
        self, input_energies, reactants, products, energy_key="Ef", norm="rxn"
    ):
        """

        Args:
            input_energies (dict)
                {formula (str): {< energy key> : formation energy (eV/at)}
                    formation energies should account for chemical potentials (e.g., due to partial pressures)

            reactants (list):
                list of reactant compositions (str)

            products (list):
                list of product compositions (str)

            energy_key (str):
                how to find the formation energies in formation_energies

            norm (str):
                how to normalize the reaction energy
                    'rxn' : the molar reaction energy for the reaction in ReactionEnergy.rxn_string
                    'atom' : the molar reaction energy per atom in the products side of the reaction in ReactionEnergy.rxn_string
        """

        self.input_energies = {
            CompTools(c).clean: {"E": input_energies[c][energy_key]}
            for c in input_energies
        }
        self.reactants = [Composition(c) for c in reactants]
        self.products = [Composition(c) for c in products]

    @property
    def rxn(self):
        """
        Returns:
            Pymatgen Reaction object
        """
        rxn = Reaction(self.reactants, self.products)
        return rxn

    @property
    def rxn_string(self):
        """
        Returns:
            string representation of the reaction
        """
        return self.rxn.__str__()

    @property
    def coefs(self):
        """
        Returns:
            {formula (str) : stoichiometry (float) in reaction}
        """
        rxn = self.rxn
        coefs = rxn._coeffs
        all_comp = rxn._all_comp
        all_comp = [CompTools(c.formula).clean for c in all_comp]
        unique_comp = list(set(all_comp))
        out = {c: 0 for c in unique_comp}
        for i, coef in enumerate(coefs):
            comp = all_comp[i]
            out[comp] += np.round(coef, 8)
        return out

    @property
    def species(self):
        """

        Returns:
            {formula (str) : {'coef' : stoichiometry (float) in reaction},
                              'E' : energy (float, eV/atom) of that formula}}
        """
        species = {}
        coefs = self.coefs
        energies = self.input_energies
        for c in coefs:
            if c != 0:
                species[c] = {
                    "coef": coefs[c],
                    "E": energies[c]["E"] if CompTools(c).n_els > 1 else 0,
                }

        return species

    @property
    def dE_rxn(self):
        """
        Returns:
            reaction energy (float)
                eV/atom if self.norm == 'atom'
                eV/rxn if self.norm == 'rxn'

        """
        species = self.species
        norm = self.norm
        dE_rxn = 0
        norm = 1
        for formula in species:
            coef = species[formula]["coef"]
            if (norm == "atom") and (coef > 0):
                norm += coef * CompTools(formula).n_atoms
            Ef = species[formula]["E"]
            dE_rxn += coef * Ef * CompTools(formula).n_atoms

        return dE_rxn / norm


def main():
    return


if __name__ == "__main__":
    out = main()

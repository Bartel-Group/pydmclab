import numpy as np
import math
from itertools import combinations

from pydmc.data.thermochem import (
    mp2020_compatibility_dmus,
    mus_at_0K,
    mus_at_T,
    mus_from_mp_no_corrections,
    mus_from_bartel2019_npj,
)
from pydmc.data.features import atomic_masses
from pydmc.core.comp import CompTools


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
        R=8.6173303e-5,  # eV/K
        user_chempots={},
        user_dmus={},
    ):
        """
        Args:
            temperature (int) - temperature in Kelvin
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
            if functional == "pbeu":
                for el in mp_dmus["U"]:
                    user_dmus[el] = -mp_dmus["U"][el]
            if self.oxide_type == "peroxide":
                user_dmus[el] = -mp_dmus["peroxide"]["O"]
            elif self.oxide_type == "superoxide":
                user_dmus[el] = -mp_dmus["superoxide"]["O"]

        self.user_dmus = user_dmus
        self.user_chempots = user_chempots

    @property
    def chempots(self):
        """
        Returns:
            dictionary of chemical potentials {el : chemical potential (eV/at)} based on user inputs
        """

        if self.temperature == 0:
            if (self.standard == "dmc") or (self.functional in ["scan", "r2scan"]):
                all_mus = mus_at_0K()
                els = sorted(list(all_mus[self.functional].keys()))
                mus = {el: all_mus[self.functional][el]["mu"] for el in els}
            else:
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
                    self.R
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


class FormationEnthalpy(object):
    """
    For computing formation energies (~equivalently enthalpies) at 0 K

    TO DO:
        - write tests/demo
    """

    def __init__(
        self,
        formula,
        E_DFT,  # eV/at
        chempots,  # from ThermoTools.ChemPots.chempots
    ):

        """
        Args:
            formula (str) - chemical formula
            E_DFT (float) - DFT energy (eV/at)
            chempots (dict) - {el (str) : chemical potential (eV/at)}
                - probably generated using ChemPots.chempots

        """
        self.formula = CompTools(formula).clean
        self.E_DFT = E_DFT
        self.chempots = chempots

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
    def Ef(self):
        """
        Returns:
            formation energy at 0 K (eV/at)
        """
        formula = self.formula
        n_atoms = CompTools(formula).n_atoms
        weighted_elemental_energies = self.weighted_elemental_energies
        E_per_fu = self.E_DFT * n_atoms
        return (1 / n_atoms) * (E_per_fu - weighted_elemental_energies)


class FormationEnergy(object):
    """
    This class is for computing formation energies at T > 0 K

    Automatically uses the Bartel2018 model: https://doi.org/10.1038/s41467-018-06682-4

    TO DO:
        - write tests/demo
        - test ideal mixing entropy?
    """

    def __init__(
        self,
        formula,
        Ef,  # eV/at
        chempots,  # from ThermoTools.ChemPots.chempots
        structure=False,
        atomic_volume=False,
        x_config=None,
        n_config=1,
        include_Svib=True,
        include_Sconfig=False,
    ):

        """
        Args:
            formula (str) - chemical formula
            Ef (float) - DFT formation enthalpy at 0 K (eV/at)
                - or any formation enthalpy at T <= 298 K
            chempots (dict) - {el (str) : chemical potential (eV/at)}
                - probably generated using ChemPots.chempots
            structure (Structure) - pymatgen structure object
                - either structure or atomic_volume needed for vibrational entropy calculation
            atomic_volume (float) - atomic volume (A^3/atom)
                - either structure or atomic_volume needed for vibrational entropy calculation
            override_Ef_0K (float) - formation energy at 0 K (eV/at)
                - if False, compute Ef_0K using FormationEnergy.Ef_0K
            x_config (float) - partial occupancy parameter to compute configurational entropy
                - needed to compute configurational entropy
            n_config (int) - number of systems exhibiting ideal solution behavior
                - this would be one if I have one site that is partially occupied by two ions
                - this would be two if I have two sites that are each partially occupied by two ions
            include_Svib (bool) - whether to include vibrational entropy
            include_Sconfig (bool) - whether to include configurational entropy
        """
        self.formula = CompTools(formula).clean
        self.Ef = Ef
        self.chempots = chempots
        self.structure = structure
        self.atomic_volume = atomic_volume
        self.include_Svib = include_Svib
        self.include_Sconfig = include_Sconfig
        self.x_config = x_config
        self.n_config = n_config

        if include_Svib:
            if not structure and not atomic_volume:
                raise ValueError(
                    "Must provide structure and atomic volume to compute Svib"
                )

        if include_Sconfig:
            if not (x_config and n_config):
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
        return np.sum([mus[el] * els_to_amts[el] for el in els_to_amts])

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
        Ef_0K = self.Ef
        if T == 0:
            return Ef_0K
        else:
            if self.include_Svib:
                m = self.reduced_mass
                if self.atomic_volume:
                    V = self.atomic_volume
                elif self.structure:
                    V = self.structure.volume / len(self.structure)
                else:
                    raise ValueError("Need atomic volume or structure to compute G(T)")

                Gd_sisso = (
                    (-2.48e-4 * np.log(V) - 8.94e-5 * m / V) * T
                    + 0.181 * np.log(T)
                    - 0.882
                )
                weighted_elemental_energies = self.weighted_elemental_energies
                G = Ef_0K + Gd_sisso
                n_atoms = CompTools(self.formula).n_atoms

                dGf = (1 / n_atoms) * (G * n_atoms - weighted_elemental_energies)
            else:
                dGf = Ef_0K
                if self.include_Sconfig:
                    x, n = self.x_config, self.n_config
                    kB = 8.617e-5  # eV/K
                    S_config = (
                        -kB * n * (x * np.log(x) + (1 - x) * np.log(1 - x))
                    ) / CompTools(self.formula).n_atoms
                    dGf += -T * S_config
            return dGf


def main():

    mus = ChemPots(functional="r2scan", standard="dmc")

    Ef = FormationEnthalpy(
        formula="IrO2", E_DFT=-0.10729517e04 / 48, chempots=mus.chempots
    ).Ef

    temperature = 2000
    mus = ChemPots(temperature=temperature)

    # return mus, mus
    dGf = FormationEnergy(
        formula="IrO2", Ef=Ef, chempots=mus.chempots, atomic_volume=10.76
    ).dGf(temperature)
    print(Ef)
    print(dGf)

    return mus, fe


if __name__ == "__main__":
    mus, fe = main()

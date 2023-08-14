from pydmclab.core.struc import StrucTools, SiteTools

# from pymatgen.analysis.defects import Substitution, Interstitial, Vacancy
from pymatgen.core import PeriodicSite, Species, Structure

from pymatgen.analysis.phase_diagram import PDEntry, PhaseDiagram
from pymatgen.core import Composition, Element
from pydmclab.core.comp import CompTools

from pymatgen.analysis.chempot_diagram import ChemicalPotentialDiagram

import numpy as np

import sympy
from sympy.solvers.inequalities import (
    solve_rational_inequalities,
    solve_poly_inequalities,
)


class DefectTools(object):
    def __init__(self, pristine_structure):
        pass


class ChemPotBounds(object):
    def __init__(
        self, input_energies, reference_compound, energy_key="E", fixed_els=[]
    ):
        """
        Args:
            input_energies (dict):
                {formula (str) :
                    {energy_key (str) :
                        formation energy in eV/atom (float)}}}
                these should be ground-state energies for all formulas that can compete for stability with "reference_compound"
                    e.g., all phases in the Ba-Zr-S chemical space for BaZrS3

            reference_compound (str):
                compound you are trying to assess chemical potential bounds for

            energy_key (str): Defaults to "E".
                the key inside of each formula in input_energies where the energy per atom lives
                e.g., input_energies['Li2FeP2S6']['E'] should return the DFT total energy per atom if energy_key = 'E'

            fixed_els (list): Defaults to [].
                this analysis is really only built for 2 or 3d chemical potential triangles
                    if you have a compound with 4+ elements, you need to "fix" the chemical potential of N-3 elements (where N is the number of elements in your compound)

        """
        # determine chemical space we're working in
        els = CompTools(reference_compound).els

        # sanitize input energies. make formulas "clean" and remove irrelevant formulas (those outside of the reference compound's chemical space)
        input_energies = {
            CompTools(k).clean: {"E": input_energies[k][energy_key]}
            for k in input_energies
            if set(CompTools(k).els).issubset(set(els))
        }

        # sanitize reference compound
        reference_compound = CompTools(reference_compound).clean

        self.reference_compound = reference_compound

        # add formation energy of elements to input_energies
        # **NOTE**: this assumes we have formation energies (i.e., formation energy of elements = 0)
        for el in els:
            input_energies[el] = {"E": 0}

        self.input_energies = input_energies
        self.els = els

        self.fixed_els = fixed_els

        self.variable_els = [el for el in els if el not in fixed_els]

    @property
    def entries(self):
        """
        Returns:
            list of PDEntry objects for each compound in input_energies
                this sill be needed for the ChemicalPotentialDiagram object
                note: energies passed to pymatgen PhaseDiagrams are per formula unit
        """
        input_energies = self.input_energies

        return [
            PDEntry(Composition(k), input_energies[k]["E"] * CompTools(k).n_atoms)
            for k in input_energies
        ]

    @property
    def fixed_mus(self):
        fixed_els = self.fixed_els
        if not fixed_els:
            return {}
        entries = self.entries
        pd = PhaseDiagram(entries)
        ref_cmpd = Composition(self.reference_compound)
        mus_from_pd = pd.get_composition_chempots(ref_cmpd)
        return {el: mus_from_pd[el] for el in fixed_els}

    @property
    def cpd(self):
        """
        Returns:
            ChemicalPotentialDiagram object from the PDEntry objects we made
        """
        return ChemicalPotentialDiagram(self.entries)

    @property
    def stable_domain(self):
        """
        Returns:
            list of dictionaries that indicate the chemical potential domains where our reference compound is thermodynamically stable
                each dictionary looks like:
                    {element (str) : change in chemical potential compared to the reference (float)}
                each dictionary can be thought of as an extreme point in chemical potential space where the reference compound is no longer stable
                    for binary compounds, there should be two points (two dictionaries)
                        meaning the reference compound is stable along the line formed by these two points
                    for ternary and higher, not sure how many points this should be?
                        seems like 6 for MgCr2S4... hmm...
        """
        els = self.els
        cpd = self.cpd
        els_for_domains = cpd.elements
        domains = cpd._get_domains()

        reference_compound = self.reference_compound

        out = {reference_compound: []}
        for formula in domains:
            if CompTools(formula).clean == reference_compound:
                out[reference_compound] = []
                for point in domains[formula]:
                    point_dict = {
                        els_for_domains[i].name: np.round(point[i], decimals=6)
                        for i in range(len(els))
                    }

                    out[reference_compound].append(point_dict)
        return out[reference_compound]

    @property
    def mu_limits(self):
        fixed_mus = self.fixed_mus
        stable_domain = self.stable_domain
        els = self.els
        limits = {el: None for el in els}
        for el in els:
            if el in fixed_mus:
                limits[el] = (fixed_mus[el], fixed_mus[el])
            else:
                mus = [point[el] for point in stable_domain]
                limits[el] = (min(mus), max(mus))
        return limits

    @property
    def constrain_based_on_formation_energy(self):
        ref_cmpd = self.reference_compound
        input_energies = self.input_energies
        Ef_per_fu = input_energies[ref_cmpd]["E"] * CompTools(ref_cmpd).n_atoms

        fixed_mus = self.fixed_mus
        variable_els = self.variable_els

        if len(variable_els) == 3:
            x, y, z = sympy.symbols("x, y, z")

            eq = (
                CompTools(ref_cmpd).stoich(variable_els[0]) * x
                + CompTools(ref_cmpd).stoich(variable_els[1]) * y
                + CompTools(ref_cmpd).stoich(variable_els[2]) * z
                - Ef_per_fu
            )

        elif len(variable_els) == 2:
            x, y = sympy.symbols("x, y")

            eq = (
                CompTools(ref_cmpd).stoich(variable_els[0]) * x
                + CompTools(ref_cmpd).stoich(variable_els[1]) * y
                + np.sum([CompTools(ref_cmpd).stoich(el) * fixed_mus[el] for el in fixed_mus]))
                - Ef_per_fu
            )
        else:
            raise ValueError("only works for 2 or 3d chemical potential triangles")

        return sympy.Poly(eq)

    def mus_for_formation_energy_calculation(self, extrinsic_els=[]):
        els = self.els
        mu_limits = self.mu_limits
        mus = {}
        for variable_el in els:
            rich_label = "%s-rich" % variable_el
            poor_label = "%s-poor" % variable_el
            mus[rich_label] = {}
            mus[poor_label] = {}
            other_els = [el for el in els if el != variable_el]
            mus[rich_label][variable_el] = mu_limits[variable_el][1]
            mus[poor_label][variable_el] = mu_limits[variable_el][0]

            def el_sorter(el):
                return -abs(
                    Element(el).electron_affinity
                    - Element(variable_el).electron_affinity
                )

            other_els = sorted(other_els, key=el_sorter) + sorted(
                extrinsic_els, key=el_sorter
            )

            for other_el in other_els:
                return

    def are_mus_valid_for_reference_compound(self, mus, thresh=1e-2):
        mu_limits = self.mu_limits
        for el in mus:
            mu_min, mu_max = mu_limits[el]
            if mus[el] < mu_min:
                print("mu too negative for %s" % el)
                return False
            if mus[el] > mu_max:
                print("mu too positive for %s" % el)
                return False

        ref_cmpd = self.reference_compound
        reference_formation_energy_per_fu = (
            self.input_energies[ref_cmpd]["E"] * CompTools(ref_cmpd).n_atoms
        )

        summed_mus = 0
        for el in mus:
            summed_mus += mus[el] * CompTools(ref_cmpd).stoich(el)

        if abs(summed_mus - reference_formation_energy_per_fu) > thresh:
            print("summed_mus != reference_formation_energy_per_fu")
            return False

        return True

    @property
    def plot_mus(self):
        return self.cpd.get_plot()


def main():
    from pydmclab.utils.handy import read_json

    fjson = "../data/test_data/ternarypd/Efs.json"
    input_energies = read_json(fjson)

    # input_energies = {"MnO": {"E": -1.977}, "MnO2": {"E": -1.81}, "Mn2O3": {"E": -2.01}}
    # reference_compound = "MnO"

    reference_compound = "MgCr2S4"
    obj = ChemPotBounds(input_energies, reference_compound, energy_key="Ef")
    return obj
    return


if __name__ == "__main__":
    obj = main()

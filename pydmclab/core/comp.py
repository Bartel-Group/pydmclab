from pymatgen.core.composition import Composition
import numpy as np


class CompTools(object):
    def __init__(self, formula: str):
        """
        Args:
            formula (str) - chemical formula

        Returns:
            formula
        """
        self.formula = formula

    @property
    def clean(self):
        """
        Returns:
            formula (str) that has been:
                - sorted by elements
                - parentheses removed
                - fractions --> integers

        In general, we want to use this for maximum consistency
        """
        # reduce
        formula = Composition(self.formula).reduced_formula

        # alphabetize
        formula = Composition(formula).alphabetical_formula

        # expand if it has decimals
        if "." in formula:
            formula = Composition(formula).get_integer_formula_and_factor()[0]

        # alphabetize
        formula = Composition(formula).alphabetical_formula

        # remove parentheses
        formula = Composition(formula).to_pretty_string()

        # remove "2" for diatomic elements
        if (len(formula) <= 3) and (formula[-1] == "2"):
            formula = formula.replace("2", "1")
        return formula

    @property
    def pretty(self):
        """
        Returns:
            formula (str) that is visually pleasing (chemically meaningful)
                - note: reduces the formula
        """
        return Composition(self.formula).reduced_formula

    @property
    def amts(self):
        """
        Returns:
            dictionary of elements (str) and their amounts (float)
                - note: starts with "clean" formula
                {el (str) : amt (float)}

            CompTools('Al4O6').amts = {'Al' : 2, 'O' : 3}
        """

        return Composition(self.clean).get_el_amt_dict()

    def mol_frac(self, el:str):
        """
        Returns:
            the molar fraction (float) of an element (str)
                - note: starts with "clean" formula

            CompTools('Al4O6').mol_frac('Al') = 0.4
        """
        return Composition(self.clean).get_atomic_fraction(el)

    def stoich(self, el:str):
        """
        Returns:
            the stoichiometry of an element (int)
                - note: starts with "clean" formula
            e.g., if CompTools(c).clean == 'Al2Mg1O4', then CompTools(c).stoich('O') = 4
        """
        stoich = self.mol_frac(el) * self.n_atoms
        stoich = np.round(stoich, 0)
        return int(stoich)

    @property
    def chemsys(self):
        """
        Returns:
            chemical system (str) of the formula
                - sorted
                - elements (str) joined by "-"

            CompTools('Al4O6').chemsys = 'Al-O'
        """
        return Composition(self.clean).chemical_system

    @property
    def els(self):
        """
        Returns:
            list of elements (str) in the formula
                - sorted

            CompTools('Al4O6').els = ['Al', 'O']
        """
        return list(sorted(self.chemsys.split("-")))

    @property
    def n_els(self):
        """
        Returns:
            number of elements (int) in the formula

            CompTools('Al4O6').n_els = 2
        """
        return len(self.els)

    @property
    def n_atoms(self):
        """
        Returns:
            number of atoms (int) in the formula
                - note: starts with "clean" formula

            CompTools('Al4O6').n_atoms = 5
        """
        return np.sum(list(self.amts.values()))

    def find_scaling(self, formula_to_compare:str):
        """
        Args:
            formula_to_compare (str)
                formula to compare to self.formula

        Returns:
            scaling factor (float)
                the factor by which self.formula must be scaled to match formula_to_compare
                e.g., self.formula = "La2Co2O5" and formula_to_compare = "LaCoO2.5", then the output will be 0.5
        """

        # get a cleaned formula and scaling factor of current formula
        clean_self, current_formula_scale = Composition(
            self.formula
        ).get_integer_formula_and_factor()

        # get a cleaned formula and scaling factor of comparison formula
        clean_compare, comparison_formula_scale = Composition(
            formula_to_compare
        ).get_integer_formula_and_factor()

        # check that the formulas represent the same compound
        if clean_self != clean_compare:
            raise ValueError(
                "Formulas must represent the same compound to be compared: %s != %s"
                % (clean_self, clean_compare)
            )

        # calculate and return scaling factor
        return comparison_formula_scale / current_formula_scale
# el_order=None is default
    def label_for_plot(self, el_order:str):
        """
        @NOTE:
            this is redundant w/ pydmclab.plotting.utils.get_label ?

        @FEATURE:
            - would be great if this worked for decimal formulas

        Args:
            el_order (list) - order of elements (str) for plotting

        Returns:
            label (str) for plotting (includes $ for subscripts)
        """
        # formula = self.clean if reduce else self.formula
        if not el_order:
            el_order = self.els
        amts = self.amts
        label = r"$"
        for el in el_order:
            if el in amts:
                n_el = amts[el]
                if n_el == 1:
                    label += el
                elif n_el > 1:
                    if int(n_el) - n_el == 0:
                        label += el + "_{%s}" % str(int(n_el))
                    else:
                        label += el + "_{%.1f}" % float(n_el)
        label += "$"
        return label


def main():
    return


if __name__ == "__main__":
    main()

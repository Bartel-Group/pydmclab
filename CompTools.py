from pymatgen.core.composition import Composition
import numpy as np


class CompTools(object):
    
    def __init__(self, formula):
        """
        Args: 
            formula (str) - chemical formula
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
        """
        formula = Composition(self.formula).reduced_formula
        formula = Composition(formula).alphabetical_formula
        if '.' in formula:
            formula = Composition(formula).get_integer_formula_and_factor()[0]
        return Composition(formula).to_pretty_string()
    
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
        """
        return Composition(CompTools(self.formula).clean).to_reduced_dict
    
    def mol_frac(self, el):
        """
        Returns:
            the molar fraction of an element (float)
                - note: starts with "clean" formula
        """
        return Composition(CompTools(self.formula).clean).get_atomic_fraction(el)
    
    @property
    def chemsys(self):
        """
        Returns:
            chemical system (str) of the formula
                - sorted
                - elements separated by "-"
        """
        return Composition(self.formula).chemical_system
    
    @property
    def els(self):
        """
        Returns:
            list of elements (str) in the formula
                - sorted
        """
        return list(sorted(CompTools(self.formula).chemsys.split('-')))
    
    @property
    def n_els(self):
        """
        Returns:
            number of elements (int) in the formula
        """
        return len(CompTools(self.formula).els)
    
    @property
    def n_atoms(self):
        """
        Returns:
            number of atoms (int) in the formula
                - note: starts with "clean" formula
        """
        return np.sum(list(CompTools(self.formula).amts.values()))
    
    def label_for_plot(self, el_order=None, reduce=True):
        """
        Returns:
            label (str) for plotting (includes $ for subscripts)
        """
        formula = CompTools(self.formula).clean if reduce else self.formula
        if not el_order:
            el_order = CompTools(formula).els
        amts = CompTools(formula).amts
        label = r'$'
        for el in el_order:
            if el in amts:
                n_el = amts[el]
                if n_el == 1:
                    label += el
                elif n_el > 1:
                    if int(n_el) - n_el == 0:
                        label += el + '_{%s}' % str(int(n_el))
                    else:
                        label += el + '_{%.1f}' % float(n_el)
        label += '$'
        return label
        
    
    
    
def main():
    formula = 'NaV2.5(PO4)3'
    return CompTools(formula)
    

if __name__ == '__main__':
    o = main()
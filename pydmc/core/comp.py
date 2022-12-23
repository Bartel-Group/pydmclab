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
                
        In general, we want to use this for maximum consistency
        """
        formula = Composition(self.formula).reduced_formula
        formula = Composition(formula).alphabetical_formula
        if '.' in formula:
            formula = Composition(formula).get_integer_formula_and_factor()[0]
        formula = Composition(formula).alphabetical_formula
        formula = Composition(formula).to_pretty_string()

        if (len(formula) <= 3) and (formula[-1] == '2'):
            formula = formula.replace('2', '1')
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
        """
        
        return Composition(self.clean).get_el_amt_dict()
    
    def mol_frac(self, el):
        """
        Returns:
            the molar fraction (float) of an element (str)
                - note: starts with "clean" formula
        """
        return Composition(self.clean).get_atomic_fraction(el)
    
    def stoich(self, el):
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
        """
        return Composition(self.clean).chemical_system
    
    @property
    def els(self):
        """
        Returns:
            list of elements (str) in the formula
                - sorted
        """
        return list(sorted(self.chemsys.split('-')))
    
    @property
    def n_els(self):
        """
        Returns:
            number of elements (int) in the formula
        """
        return len(self.els)
    
    @property
    def n_atoms(self):
        """
        Returns:
            number of atoms (int) in the formula
                - note: starts with "clean" formula
        """
        return np.sum(list(self.amts.values()))
    
    def label_for_plot(self, el_order=None, reduce=True):
        """
        Returns:
            label (str) for plotting (includes $ for subscripts)
        """
        #formula = self.clean if reduce else self.formula
        if not el_order:
            el_order = self.els
        amts = self.amts
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
    return 
    

if __name__ == '__main__':
    o = main()
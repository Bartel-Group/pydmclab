from pydmclab.core.comp import CompTools

""" 
Purpose: 
  CompTools is used to perform chemical formula manipulations

In pydmclab, formulas are expected to be "clean" (CompTools(formula).clean)
  - this ensures that you can always find a chemical formula
  - for instance, you wouldn't want CaTiO3 to sometimes appear as CaTiO3, Ca1Ti1O3, CaTi1O3, etc.
    - CompTools(< any of these formula representations >).clean will always yield Ca1O3Ti1
        - sorts els alphabetically
        - reduces formula
        - removes spaces
        - removes parentheses
        - includes stoichiometry (even if 1)
  - Note: there are some currently strange exceptions (due to [pymatgen special formulas](https://github.com/materialsproject/pymatgen/blob/v2023.5.10/pymatgen/core/composition.py#L75))
        - Li2O2, Na2O2, K2O2, H2O2, Cs2O2, Rb2O2
        - these formulas won't reduce
        
Unit tests:
  - see pydmclab/tests/test_comp.py
"""


def print_initial_final(initial, final):
    """
    Args:
        initial (str) - initial formula
        final (str) - final formula

    Returns:
        prints the initial and final formulas
    """

    print("\ninitial: %s --> %s (final)" % (initial, final))


def demo_basic_formula_manipulations():
    """
    prints initial formulas and CompTools(initial).clean formulas
        - these should be:
            - alphabetized
            - strings with no spaces
            - integer amounts for each element
            - reduced (i.e., cannot be simplified to smaller integers)

    """
    print("\n")
    initial_formulas = ["TiO2", "NaV2(PO4)3", "Ga20As40", "AlO1.5"]

    for initial in initial_formulas:
        print_initial_final(initial, CompTools(initial).clean)

    return


def demo_amts():
    """
    prints the extracted integer amount of each element in the formula
    """
    print("\n")
    initial = "NaV2.5(PO4)3"

    ct = CompTools(initial)
    clean_formula = ct.clean
    print("\nclean formula: %s\n" % clean_formula)

    amts_dict = ct.amts
    for el in amts_dict:
        print("%i %s in %s" % (amts_dict[el], el, clean_formula))


def demo_mol_frac_and_stoich():
    """
    prints the extracted integer amount and fractional amount of each element in the formula
    """
    print("\n")
    initial = "Cs2AgInCl6"

    ct = CompTools(initial)
    clean_formula = ct.clean
    print("\nclean formula: %s\n" % clean_formula)

    els = ct.els
    for el in els:
        print(
            "%i %s in %s; fractional amount = %.2f"
            % (ct.stoich(el), el, clean_formula, ct.mol_frac(el))
        )


def demo_diatomics():
    """
    ensures that diatomic elements get simplifed to O1, N1, etc
        - this is desired behavior for ThermoTools
        - might need to modify CompTools.label_for_plot to make these diatomic
    """
    print("\n")
    initial_formulas = ["O2", "H2", "Cl2", "Br2", "N2"]
    for initial in initial_formulas:
        print(CompTools(initial).clean)


def demo_plot_labels():
    """
    converts formula strings into strings with subscripts for plotting purposes
    """
    initial_formulas = ["LiMn2O4", "MgCr2S4", "Si1.6C1.6", "MgAlO3.5"]

    print("\n")
    el_order = ("Li", "Si", "Mg", "Al", "Cr", "Mn", "C", "O")
    for initial in initial_formulas:
        ct = CompTools(initial)
        clean_formula = ct.clean
        print("%s = %s" % (clean_formula, ct.label_for_plot(el_order=el_order)))


def demo_simple_comp():
    formula = "NaV2(PO4)3"
    ct_obj = CompTools(formula)
    return ct_obj


def main():
    # return demo_simple_comp()
    demo_basic_formula_manipulations()
    demo_amts()
    demo_mol_frac_and_stoich()
    demo_diatomics()
    demo_plot_labels()
    return


if __name__ == "__main__":
    a = main()

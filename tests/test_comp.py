import unittest
from pydmclab.core.comp import CompTools


class UnitTestCompTools(unittest.TestCase):
    def test_clean(self):
        formula = "V4Na2(P2O8)6.5"
        self.assertEqual(CompTools(formula).clean, "Na2O52P13V4")
        formula = "Ti2O4"
        self.assertEqual(CompTools(formula).clean, "O2Ti1")

    def test_pretty(self):
        formula = "V4Na2(P2O8)6.5"
        self.assertEqual(CompTools(formula).pretty, "Na2V4(PO4)13")

    def test_els_to_amts(self):
        formula = "V4Na2(P2O8)6.5"
        self.assertEqual(CompTools(formula).amts, {"Na": 2, "V": 4, "P": 13, "O": 52})

    def test_mol_frac(self):
        formula = "V4Na2(P2O8)6.5"
        self.assertEqual(CompTools(formula).mol_frac("Na"), 2 / 71)
        self.assertEqual(CompTools(formula).mol_frac("Ti"), 0.0)

    def test_stoich(self):
        formula = "V4Na2(P2O8)6.5"
        self.assertEqual(CompTools(formula).stoich("Na"), 2)

    def test_chemsys(self):
        formula = "V4Na2(P2O8)6.5"
        self.assertEqual(CompTools(formula).chemsys, "Na-O-P-V")

    def test_els(self):
        formula = "V4Na2(P2O8)6.5"
        self.assertEqual(CompTools(formula).els, ["Na", "O", "P", "V"])

    def test_n_els(self):
        formula = "V4Na2(P2O8)6.5"
        self.assertEqual(CompTools(formula).n_els, 4)

    def test_n_atoms(self):
        formula = "V4Na2(P2O8)6.5"
        self.assertEqual(CompTools(formula).n_atoms, 71)

    def test_find_scaling(self):
        formula = "V4Na2(P2O8)6.5"
        formula_to_compare = "V2Na1(P1O4)3.25"
        self.assertEqual(CompTools(formula).find_scaling(formula_to_compare), 0.5)

    def test_label_for_plot(self):
        formula = "V4Na2(P2O8)6.5"
        self.assertEqual(
            CompTools(formula).label_for_plot(), r"$Na_{2}O_{52}P_{13}V_{4}$"
        )


if __name__ == "__main__":
    unittest.main(argv=[''], exit=False)

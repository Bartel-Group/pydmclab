import unittest
from pydmclab.core.query import MPLegacyQuery


class UnitTestQuery(unittest.TestCase):
    def test_data_for_comp3(self):
        API_KEY = "***REMOVED***"
        mpq = MPLegacyQuery(API_KEY)
        q = mpq.get_data_for_comp(comp="Yb-F", only_gs=True)
        unique_formulas = ["Yb", "F", "F2Yb1", "F3Yb1"]
        formulas_from_q = [q[k]["cmpd"] for k in q]
        print(formulas_from_q)
        self.assertEqual(len(unique_formulas), len(formulas_from_q))








"""
class UnitTestQuery2(unittest.TestCase):
    def test_data_for_comp1(self):
        API_KEY = "***REMOVED***"
        mpq = MPQuery(API_KEY)
        q = mpq.get_data_for_comp(comp="MnO")
        self.assertEqual(len(q), 1)

    def test_data_for_comp2(self):
        API_KEY = "***REMOVED***"
        mpq = MPQuery(API_KEY)
        q = mpq.get_data_for_comp(comp="MnO", only_gs=False)
        self.assertGreater(len(q), 1)
"""

if __name__ == "__main__":
    unittest.main()
    #unittest.main(argv=[''], exit=False)

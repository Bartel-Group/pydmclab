import unittest
from pydmclab.core.query import MPQuery


class UnitTestQuery(unittest.TestCase):
    def test_data_for_comp(self):
        API_KEY = "***REMOVED***"

        mpq = MPQuery(API_KEY)

        q = mpq.get_data_for_comp(comp="MnO")
        self.assertEqual(len(q), 1)

        return


if __name__ == "__main__":
    unittest.main()

import unittest
from pydmclab.core.mag import MagTools
import numpy as np


def get_generic_query(remake=False):
    from pydmclab.core.query import MPQuery
    from pydmclab.utils.handy import read_json, write_json
    import os

    fjson = "query.json"

    if not remake and os.path.exists(fjson):
        return read_json(fjson)
    comps = ["MgO", "MnO", "Cr2O3"]
    mpq = MPQuery(api_key="***REMOVED***")
    q = mpq.get_data_for_comp(comps, only_gs=True)
    q = {q[k]["cmpd"]: q[k] for k in q}
    write_json(q, fjson)
    return read_json(fjson)


class UnitTestMag(unittest.TestCase):
    def test_could_be_magnetic(self):
        q = get_generic_query()
        MgO = q["Mg1O1"]["structure"]
        MnO = q["Mn1O1"]["structure"]
        Cr2O3 = q["Cr2O3"]["structure"]

        mt = MagTools(MgO)
        self.assertFalse(mt.could_be_magnetic)
        mt = MagTools(MnO)
        self.assertTrue(mt.could_be_magnetic)
        mt = MagTools(Cr2O3)
        self.assertTrue(mt.could_be_magnetic)
        self.assertTrue(mt.could_be_afm)
        return

    def test_get_nm_fm(self):
        q = get_generic_query()
        MgO = q["Mg1O1"]["structure"]
        MnO = q["Mn1O1"]["structure"]

        mt = MagTools(MgO)
        s = mt.get_nonmagnetic_structure
        magmoms = []
        for site in s:
            magmoms.append(site.magmom)
        self.assertEqual(sum(magmoms), 0)

        mt = MagTools(MnO)
        s = mt.get_ferromagnetic_structure
        magmoms = []
        for site in s:
            if site.specie.symbol == "Mn":
                magmoms.append(site.magmom)
        self.assertEqual(np.mean(magmoms), 5)

    def test_get_afm(self):
        q = get_generic_query()
        Cr2O3 = q["Cr2O3"]["structure"]

        mt = MagTools(Cr2O3)

        strucs = mt.get_antiferromagnetic_structures

        self.assertEqual(len(strucs), 3)

    def test_treat_as_nm(self):
        q = get_generic_query()
        MnO = q["Mn1O1"]["structure"]

        mt = MagTools(MnO, treat_as_nm=["Mn"])

        self.assertFalse(mt.could_be_magnetic)

        self.assertIsNone(mt.get_ferromagnetic_structure)


if __name__ == "__main__":
    unittest.main()

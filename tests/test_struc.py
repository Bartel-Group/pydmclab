import unittest
from pydmclab.core.struc import StrucTools, SiteTools
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


class UnitTestStruc(unittest.TestCase):
    def test_struc(self):
        q = get_generic_query()
        Cr2O3 = q["Cr2O3"]["structure"]

        st = StrucTools(Cr2O3)

        self.assertEqual("Cr2O3", st.compact_formula)
        self.assertEqual("Cr4 O6", st.formula)
        self.assertEqual(["Cr", "O"], st.els)

        supercell = st.make_supercell([1, 2, 3])

        self.assertEqual(len(supercell), 60)

        st = StrucTools(Cr2O3)
        new_struc = st.change_occ(5, {"Li": 0.5})

        self.assertEqual(new_struc[5].species_string, "Li:0.500")

        new_struc = st.change_occ(55, {"Li": 0.5}, structure=supercell)
        self.assertEqual(new_struc[55].species_string, "Li:0.500")

        new_struc = st.change_occ(5, {"Cr": 0})
        self.assertEqual(len(new_struc), 9)

        st = StrucTools(Cr2O3, ox_states={"Cr": 3, "O": -2})

        s = st.decorate_with_ox_states
        self.assertEqual(s[0].species_string, "Cr3+")

        s = StrucTools(Cr2O3.copy()).structure
        for i in range(4):
            s = st.change_occ(i, {"Cr": 0.5, "Mn": 0.5}, structure=s)

        st = StrucTools(s)
        strucs = st.get_ordered_structures(n_strucs=2)

        self.assertEqual(StrucTools(strucs[1]).compact_formula, "Cr1Mn1O3")

        st = StrucTools(Cr2O3)

        strucs = st.replace_species(
            species_mapping={"Cr": {"Cr": 0.5, "Mn": 0.5}}, n_strucs=2
        )

        self.assertEqual(StrucTools(strucs[1]).compact_formula, "Cr1Mn1O3")

        st = StrucTools(Cr2O3)

        self.assertEqual(st.spacegroup_info["loose"]["number"], 167)

        self.assertEqual(st.sg(), "R-3c")

        initial_vol = st.structure.volume

        scaled_vol = 1.2 * initial_vol

        scaled_struc = st.scale_structure(1.2)
        self.assertAlmostEqual(scaled_struc.volume, scaled_vol, places=3)

    def test_site(self):
        q = get_generic_query()
        Cr2O3 = q["Cr2O3"]["structure"]
        st = StrucTools(Cr2O3, ox_states={"Cr": 3, "O": -2})
        s = st.decorate_with_ox_states
        print(s[0])
        s = s.as_dict()
        index = 3

        sitet = SiteTools(s, index)
        self.assertEqual(sitet.ion, "Cr3+")
        self.assertEqual(sitet.el, "Cr")
        self.assertEqual(sitet.ox_state, 3.0)


if __name__ == "__main__":
    unittest.main()

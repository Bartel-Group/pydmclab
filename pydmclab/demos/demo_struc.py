from pydmclab.core.struc import StrucTools, SiteTools
from pydmclab.core.query import MPQuery

from pymatgen.core.structure import Structure


""" 
Purpose: 
  StrucTools is used to perform manipulations to crystal structures
  SiteTools helps you do this by extracting information for individual sites in a structure
  
Unit tests:
  - see pydmclab/tests/test_struc.py

"""

MPID = "mp-19417"
API_KEY = "***REMOVED***"


def demo_basic_structure_manipulations(mpid=MPID):
    """
    - get the structure formula
    - reduce that formula (get the clean version)
    - get the elements in the structure
    - count the number of sites
    - make a supercell
    """

    print("\n ~~ demoing basic manipulations ~~ \n")

    s = MPQuery(API_KEY).get_structure_by_material_id(mpid)
    st = StrucTools(s)

    print("\nformula = %s" % st.formula)
    print("compact formula = %s" % st.compact_formula)
    print("structure has %s" % st.els)
    print("downloaded structure has %i sites" % len(st.structure))
    st.make_supercell([1, 2, 3])
    print("supercell has %i sites" % len(st.structure))


def demo_ox_state_decoration(mpid=MPID):
    """
    - decorate w/ oxidation states using pymatgen's automatic algorithm
    - decorate w/ user-specified oxidation states
    - inspect the oxidation state using SiteTools.ion
    """
    print("\n ~~ demoing oxidation state decoration ~~ \n")
    s = MPQuery(API_KEY).get_structure_by_material_id(mpid)
    st = StrucTools(s)

    site = SiteTools(s, 0)
    print("from MP, site 0 is %s" % site.ion)

    s = st.decorate_with_ox_states

    print("after auto oxi state decoration, site 0 is %s" % SiteTools(s, 0).ion)

    st = StrucTools(s, ox_states={"Fe": 3, "Ti": 3, "O": -2})

    s = st.decorate_with_ox_states

    print(
        "after forcing Fe and Ti to be ox state = 3, site 0 is %s" % SiteTools(s, 0).ion
    )


def demo_replace_species_and_order(mpid=MPID):
    """
    - make a supercell
    - replace a fullly occupied site with a mixed occupancy site
    - generate ordered versions of this mixed occupancy structure
    - inspect some sites in one of the ordered structures
    """
    print("\n ~~ demoing replace species ~~ \n")
    s = MPQuery(API_KEY).get_structure_by_material_id(mpid)
    st = StrucTools(s)
    st.make_supercell([1, 2, 3])

    species_map = {"Fe": {"Fe": 0.875, "Cr": 0.125}}

    ordered_strucs = st.replace_species(species_map, n_strucs=10)

    print("generated %i structures" % len(ordered_strucs))

    struc_idx = 3
    struc = ordered_strucs[struc_idx]
    print("\nstructure %i has formula %s" % (struc_idx, StrucTools(struc).formula))
    site_idx = 12
    site = SiteTools(struc, site_idx)
    print(
        "site %i has el = %s, coords = %s with ox state = %s"
        % (site_idx, site.el, site.coords, site.ox_state)
    )


def demo_dilute_vacancy(mpid=MPID):
    """
    - change the occupancy of an element from 1 (fully occupied) to some value that leads to 1 vacancy

    """
    print("\n ~~ demoing dilute vacancy ~~ \n")
    s = MPQuery(API_KEY).get_structure_by_material_id(mpid)
    st = StrucTools(s)

    print("initial formula = ", st.formula)

    el_to_remove = "O"
    n_el_in_struc = st.amts["O"]
    n_el_to_remove = 1
    new_occ = (n_el_in_struc - n_el_to_remove) / n_el_in_struc

    s_with_vac = st.change_occ_for_el(
        el_to_remove, {el_to_remove: new_occ}, structure=None
    )

    ordered_strucs = StrucTools(
        s_with_vac, ox_states={"O": -2, "Fe": 3, "Ti": 3}
    ).get_ordered_structures(n_strucs=10)

    print("final formula = ", StrucTools(ordered_strucs[0]).formula)


def main():
    demo_basic_structure_manipulations()
    demo_ox_state_decoration(mpid=MPID)
    demo_replace_species_and_order(mpid=MPID)
    demo_dilute_vacancy(mpid=MPID)
    return


if __name__ == "__main__":
    main()

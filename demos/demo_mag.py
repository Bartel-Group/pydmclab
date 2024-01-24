from pydmc.core.mag import MagTools

from pydmclab.core.query import MPQuery

""" 
Purpose: 
  MagTools is used to generate magnetic orderings for structures
  
Unit tests:
  - see pydmclab/tests/test_mag.py

"""


def demo_get_afm_structures():
    """
    gets AFM orderings for Cr2O3
    """

    comps = ["Cr2O3"]
    mpq = MPQuery(api_key="***REMOVED***")
    q = mpq.get_data_for_comp(comps, only_gs=True)
    q = {q[k]["cmpd"]: q[k] for k in q}

    Cr2O3 = q["Cr2O3"]["structure"]

    mt = MagTools(Cr2O3)

    strucs = mt.get_antiferromagnetic_structures

    print(strucs[1])


def main():
    demo_get_afm_structures()
    return


if __name__ == "__main__":
    main()

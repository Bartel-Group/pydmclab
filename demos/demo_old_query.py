from pydmclab.core.query import MPLegacyQuery as MPQuery
from pydmclab.core.struc import StrucTools
from pydmclab.utils.handy import read_json, write_json
import os

""" 
Purpose:
    get data from the Materials Project database
    
Note:
    uses "old" API
"""

API_KEY = "***REMOVED***"
DATA_DIR = os.path.join("output", "query")
CHEMSYS = "Li-Mn-F-S"


def demo_get_groundstate_hull_data_for_chemsys(
    chemsys=CHEMSYS,
    remake=False,
    data_dir=DATA_DIR,
    api_key=API_KEY,
):
    """
    Args:
        chemsys (str)
            chemical system to query ('-'.join([elements]))

        only_gs (bool)
            if True, remove non-ground state polymorphs from MP query
                good practice to do this before doing hull analysis b/c non-gs polymorphs trivially have Ehull=Ehull_gs + dE_polymorph-gs

        dict_key (str)
            key to use to orient dictionary of MPQuery results
                'cmpd' is the default behavior in MPQuery, meaning we get a dictionary that looks like {CHEMICAL_FORMULA : {DATA}}

        remake (bool)
            if True, re-query MP

    Returns:
        gs (dict): dictionary of ground state data from MPQuery for chemsys
    """
    fjson = os.path.join(data_dir, "query_gs_all_" + chemsys + ".json")
    if not remake and os.path.exists(fjson):
        return read_json(fjson)

    mpq = MPQuery(api_key)
    out = mpq.get_data_for_comp(comp=chemsys)

    return write_json(out, fjson)


def reorient_gs_data_from_ID_keys_to_cmpd_keys(d_gs):
    """
    Args:
        d_gs (dict): dictionary of ground state data from MPQuery for chemsys
            {MP_ID : {DATA}}

    Returns:
        d_gs_cmpd (dict):
            reoriented so that keys are chemical formula instead of MP ID
    """
    d_gs_cmpd = {}
    for k in d_gs:
        d_gs_cmpd[d_gs[k]["cmpd"]] = d_gs[k]
    return d_gs_cmpd


def demo_entry_retrieval(mpid="mp-530748"):
    """
    Grab a calculated entry from MP
    """
    mpq = MPQuery(API_KEY)
    properties = list(mpq.typical_properties) + ["band_gap"]
    entry = mpq.get_entry_by_material_id(
        material_id=mpid,
        properties=properties,
        incl_structure=True,
        conventional=False,
        compatible_only=True,
    )
    print(entry)
    return entry


def demo_structure_retrieval(mpid="mp-1094961"):
    mpq = MPQuery(API_KEY)
    s = mpq.get_structure_by_material_id(material_id=mpid)

    print("\nThe formula for this structure is %s" % StrucTools(s).formula)
    print("The first site in the structure is %s" % s[0])

    return s


def demo_vaspinput_retrieval(mpid="mp-1938"):
    inputs = MPQuery(API_KEY).get_vasp_inputs(material_id=mpid)
    print("\n The %s for this calculation was %s" % ("ALGO", inputs["incar"]["ALGO"]))
    return inputs


def main():
    d_gs = demo_get_groundstate_hull_data_for_chemsys(remake=True)
    d_gs_cmpd = reorient_gs_data_from_ID_keys_to_cmpd_keys(d_gs)
    entry = demo_entry_retrieval()
    structure = demo_structure_retrieval()
    vasp_inputs = demo_vaspinput_retrieval()
    d_gs, d_gs_cmpd = None, None
    return d_gs, d_gs_cmpd, entry, structure, vasp_inputs


if __name__ == "__main__":
    out = main()

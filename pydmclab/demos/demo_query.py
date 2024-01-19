from pydmclab.core.query import MPQuery
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
CHEMSYS = "Sr-Zr-S"


def demo_get_a_structure(mpid):
    """
    Args:
        mpid (str): Materials Project ID

    Returns:
        structure (Structure)
    """
    mpq = MPQuery(api_key=API_KEY)
    data = mpq.get_structures_by_material_id([mpid])
    s = data[mpid]
    print("got structure for %s. it has %i sites" % (mpid, len(s)))
    return s


def demo_get_structures(mpids):
    """
    Args:
        mpids (list): list of Materials Project IDs (str)

    Returns:
        {mpid (str) : structure (Structure))}
    """
    mpq = MPQuery(api_key=API_KEY)
    data = mpq.get_structures_by_material_id(mpids)
    print("got %i structures. one each for: %s" % (len(data), ", ".join(data.keys())))
    return data


def demo_get_data_for_formula(formula):
    """
    Args:
        formula (str): chemical formula

    Returns:
        data (dict): dictionary of data from MPQuery for formula
    """
    mpq = MPQuery(api_key=API_KEY)
    data = mpq.get_data(search_for=formula)
    print("found %i mpids for %s" % (len(data), formula))
    return data


def demo_get_ground_state_data_for_formulas(formulas):
    """
    Args:
        formulas (list): list of chemical formulas

    Returns:
        data (dict): dictionary of data from MPQuery for formulas
    """
    mpq = MPQuery(api_key=API_KEY)
    data = mpq.get_data(search_for=formulas, only_gs=True)
    print(
        "queried gs data for %i formulas so got %i mpids" % (len(formulas), len(data))
    )
    return data


def demo_get_small_structures_in_chemsys(chemsys):
    """
    Args:
        chemsys (str): chemical system to query ('-'.join([elements]))

    Returns:
        data (dict): dictionary of data from MPQuery for chemsys
    """
    mpq = MPQuery(api_key=API_KEY)
    data = mpq.get_data(chemsys, max_sites_per_structure=30, only_gs=False)
    print(
        "queried small structures for %s. got %i mpids. most sites in a structure = %i"
        % (chemsys, len(data), max([data[k]["nsites"] for k in data]))
    )
    return data


def demo_get_hull_data(chemsys):
    """
    Args:
        chemsys (str): chemical system to query ('-'.join([elements]))

    Returns:
        data (dict): dictionary of data from MPQuery for chemsys (including subspaces)

    """
    mpq = MPQuery(api_key=API_KEY)
    data = mpq.get_data(chemsys, only_gs=True, include_sub_phase_diagrams=True)
    print("%i compounds in %s hull" % (len(data), chemsys))


def main():
    mpid = "mp-390"
    mpids = ["mp-390", "mp-1938", "mp-1094961"]
    formula = ["SrZrS3"]
    formulas = ["BaZrS3", "SrZrS3"]
    chemsys = "Ca-Al-O"
    hull_chemsys = "Ba-Zr-S"

    s = demo_get_a_structure(mpid)
    strucs = demo_get_structures(mpids)
    data_for_SrZrS3 = demo_get_data_for_formula(formula)
    gs_data_for_formulas = demo_get_ground_state_data_for_formulas(formulas)
    data_for_LiMgCl = demo_get_small_structures_in_chemsys(chemsys)
    data_for_BaZrS_hull = demo_get_hull_data(hull_chemsys)
    return s, strucs


if __name__ == "__main__":
    out = main()

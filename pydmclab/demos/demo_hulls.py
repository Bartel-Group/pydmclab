from pydmclab.core.comp import CompTools
from pydmclab.core.query import MPQuery
from pydmclab.core.hulls import GetHullInputData, AnalyzeHull, ParallelHulls, MixingHull
from pydmclab.utils.handy import read_json, write_json
from pydmclab.plotting.utils import set_rc_params, get_colors
import matplotlib.pyplot as plt

import os
import numpy as np

set_rc_params()

""" 
Purpose: 
    The convex hull analysis is central to understanding the stability of materials
        Convex hulls are made from formation energy vs composition analyses for a given chemical space
          Standard hulls using formation energies, which are referenced to elemental phases
            These formation energies can be DFT dEf(0 K), dGf(T, p), dphi_f(T, p, mu) etc
          Hulls can also be made from an arbitrary reference (e.g., energies relative to chosen end-member compounds)
            In pydmclab, these are called "Mixing Hulls"
    
    Typical hulls (elemental end-members, n-dimensional for n elements in a chemical space)
        Serial mode:
          - GetHullInputData prepares the input
          - AnalyzeHull processes it
        Parallel mode:
          - ParallelHulls does it all
    
    Mixing hulls (compound end-members, 2D supported, n-d coming later)
        MixingHulls
  
Unit tests:
  - see pydmclab/tests/test_hulls.py

"""
# Chris B's API key for MP query
API_KEY = "***REMOVED***"

# chemical system to test on
CHEMSYS = "Li-Nb-O-F"

# where to save data
DATA_DIR = os.path.join("output", "hulls", "data")
if not os.path.exists(DATA_DIR):
    os.mkdir(DATA_DIR)

# where to save figures
FIG_DIR = os.path.join("output", "hulls", "figures")
if not os.path.exists(FIG_DIR):
    os.mkdir(FIG_DIR)


def get_mp_data_for_chemsys(
    comp=CHEMSYS,
    only_gs=True,
    data_dir=DATA_DIR,
    remake=False,
):
    """
    Query MP to get some data for a chemical system to paly around with

    Args:
        chemsys (str)
            chemical system to query ('-'.join([elements]))

        only_gs (bool):
            if True, remove non-ground state polymorphs from MP query
                good practice to do this before doing hull analysis b/c non-gs polymorphs trivially have Ehull=Ehull_gs + dE_polymorph-gs

        data_dir (str):
            directory to save data

        remake (bool):
            if True, re-query MP

    Returns:
        gs (dict)
            dictionary of ground state data from MPQuery for chemsys
                {formula (str) : {'Ef_mp' : formation energy (eV/atom)}
    """
    fjson = os.path.join(data_dir, "query_" + comp + ".json")
    if not remake and os.path.exists(fjson):
        return read_json(fjson)

    mpq = MPQuery(API_KEY)
    d = mpq.get_data_for_comp(comp=comp, only_gs=only_gs)

    # data is MPID-keyed, let's make it formula-keyed
    out = {}
    formulas = list(set([d[k]["cmpd"] for k in d]))
    for formula in formulas:
        # NOTE: this only works because I know my query only has 1 structure per formula (only_gs = True)
        gs_key = [k for k in d if d[k]["cmpd"] == formula][0]
        out[formula] = d[gs_key]
    return write_json(out, fjson)


def serial_get_hull_input_data(
    gs, formation_energy_key="Ef_mp", remake=False, data_dir=DATA_DIR, chemsys=CHEMSYS
):
    """
    Let's prepare the hull input file for a "typical" (element-bounded) nD hull

    Args:
        gs (dict)
            dictionary of ground state data from MPQuery for chemsys
                {formula (str) : {'Ef_mp' : formation energy (eV/atom)}

        formation_energy_key (str):
            key to use for formation energy in gs
                'Ef_mp' is default behavior in MPQuery

        remake (bool)
            if True, re-calculate hull input data

        data_dir (str)
            directory to save data

        chemsys (str)
            chemical system to query ('-'.join([elements]))
                just used to generate the savename for the .json

    Returns:
        hullin (dict):
            dictionary of hull input data for gs
                {chemical space (str) :
                    {formula (str) :
                        {'E' : formation energy (float),
                         'amts' :
                            {el (str) :
                                fractional amt of el in formula (float) for el in space
                                }
                            }
                                for all relevant formulas including elements}
                    - elements are automatically given formation energy = 0
                    - chemical space is now in 'el1_el2_...' format to be jsonable
                    - each "chemical space" is a convex hull that must be computed
    """
    fjson = os.path.join(data_dir, "hullin_serial_" + chemsys + ".json")
    if not remake and os.path.exists(fjson):
        return read_json(fjson)

    ghid = GetHullInputData(gs, formation_energy_key=formation_energy_key)
    return ghid.hullin_data(fjson=fjson, remake=remake)


def serial_get_hull_output_data(
    hullin, remake=False, chemsys=CHEMSYS, data_dir=DATA_DIR
):
    """
    Now, let's solve the hull we initialized

    Args:
        hullin (dict)
            dictionary of hull input data for chemical system

        remake (bool)
            if True, re-calculate hull output data

        chemsys (str)
            chemical system to query ('-'.join([elements]))
                only used for savename

        data_dir (str)
            directory to save data

    Returns:
        stability data (dict) for all compounds in the specified chemical space
            {compound (str) :
                {'Ef' : formation energy (float),
                 'Ed' : decomposition energy (float),
                 'rxn' : decomposition reaction (str),
                 'stability' : stable (True) or unstable (False)
                 }
                 for all compounds in the chemical space
            }
    """
    fjson = os.path.join(data_dir, "hullout_serial_" + chemsys + ".json")
    if not remake and os.path.exists(fjson):
        return read_json(fjson)

    hullout = {}
    # loop through each chemical (sub)space in the hullin dictionary
    for space in hullin:
        ah = AnalyzeHull(hullin, space)
        # loop through every compound in that space
        for cmpd in hullin[space]:
            print("\n%s" % cmpd)
            # get the stability data for that compound
            hullout[cmpd] = ah.cmpd_hull_output_data(cmpd)
    return write_json(hullout, fjson)


def parallel_get_hull_input_and_output_data(
    gs, remake=False, chemsys=CHEMSYS, data_dir=DATA_DIR, n_procs=2, fresh_restart=True
):
    """
    Repeat what we did with GetHullInputData + AnalyzeHull, but now in parallel

    Args:
        gs (dict)
            dictionary of ground state data from MPQuery for chemsys
                {formula (str) : {'Ef_mp' : formation energy (eV/atom)}}
                }
        remake (bool)
            if True, re-calculate hull input and output data

        chemsys (str)
            chemical system to query ('-'.join([elements]))
                only for savename

        data_dir (str)
            directory to save data

        n_procs (int)
            number of processors to use
                could also be 'all' to use multip.cpu_count()-1 procs

        fresh_restart (bool)
            if True, restart ParallelHull process from scratch
            if False, use the files we've already written from a previous execution

    Returns:
        stability data (dict) for all compounds in the specified chemical space
            {compound (str) :
                {'Ef' : formation energy (float),
                 'Ed' : decomposition energy (float),
                 'rxn' : decomposition reaction (str),
                 'stability' : stable (True) or unstable (False)
                 }
                 for all compounds in the chemical space
            }
    """
    fjson = os.path.join(data_dir, "hullout_parallel_" + chemsys + ".json")
    if not remake and os.path.exists(fjson):
        return read_json(fjson)

    ph = ParallelHulls(gs, n_procs=n_procs, fresh_restart=fresh_restart)

    # prepare hull input data
    hullin = ph.parallel_hullin(fjson=fjson.replace("hullout", "hullin"))

    # identify the simplest hull that can be solved for each compound
    smallest_spaces = ph.smallest_spaces(
        hullin=hullin, fjson=fjson.replace("hullout", "small_spaces")
    )

    # solve these minimally complex hulls for every compound
    return ph.parallel_hullout(
        hullin=hullin, smallest_spaces=smallest_spaces, fjson=fjson, remake=True
    )


def plot_to_check_success(gs, serial_hullout, parallel_hullout):
    """
    Let's make sure things worked
        1) Parallel and serial should give the same results
        2) For unstable compounds (above the hull), our Ed results and MP E above hull should be the same
        3) For stable compounds, our Ed results will be negative; MP's should be E above hull = 0

    Args:
        gs (dict)
            dictionary of ground state data from MPQuery for chemsys

        serial_hullout (dict)
            dictionary of hull output data for gs (run serially)

        parallel_hullout (dict)
            dictionary of hull output data for gs (run in parallel)

    Returns:
        a couple scatter plots

    """
    set_rc_params()

    fig = plt.figure(figsize=(8, 3))

    params = {
        "serial": {"m": "o", "c": "blue"},
        "parallel": {"m": "^", "c": "orange"},
    }

    cmpds = sorted(gs.keys())

    cmpds = [c for c in cmpds if CompTools(c).n_els > 1]

    mp_Ehull = [gs[c]["Ehull_mp"] for c in cmpds]

    serial_decomp = [serial_hullout[c]["Ed"] for c in cmpds]
    parallel_decomp = [parallel_hullout[c]["Ed"] for c in cmpds]

    x = mp_Ehull
    y1 = serial_decomp
    y2 = parallel_decomp

    ax1 = plt.subplot(121)

    ax1 = plt.scatter(y2, y1, edgecolor="blue", marker="o", color="white")

    # ax1 = plt.xticks(xticks[1])
    # ax1 = plt.yticks(yticks[1])
    xlim, ylim = (-0.5, 1), (-0.5, 1)
    ax1 = plt.xlabel("Ed from parallel (eV/at)")
    ax1 = plt.ylabel("Ed from serial (eV/at)")
    ax1 = plt.plot(xlim, xlim, color="black", lw=1, ls="--")
    ax1 = plt.xlim(xlim)
    ax1 = plt.ylim(ylim)

    ax2 = plt.subplot(122)
    ax2 = plt.scatter(x, y1, edgecolor="blue", marker="o", color="white")

    # ax1 = plt.xticks(xticks[1])
    # ax1 = plt.yticks(yticks[1])
    xlim, ylim = (-0.1, 1), (-1, 1)
    ax2 = plt.xlabel("Ehull from MP (eV/at)")
    ax2 = plt.ylabel("")
    ax2 = plt.plot(xlim, xlim, color="black", lw=1, ls="--")
    ax2 = plt.gca().yaxis.set_ticklabels([])
    ax2 = plt.xlim(xlim)
    ax2 = plt.ylim(ylim)

    disagreements = []
    for k in serial_hullout:
        if CompTools(k).n_els == 1:
            continue
        if serial_hullout[k]["stability"] and (gs[k]["Ehull_mp"] > 0):
            disagreements.append(k)
        if not serial_hullout[k]["stability"] and (gs[k]["Ehull_mp"] == 0):
            disagreements.append(k)

        if (gs[k]["Ehull_mp"] != 0) and (
            np.round(serial_hullout[k]["Ed"], 3) != np.round(gs[k]["Ehull_mp"], 3)
        ):
            disagreements.append(k)

    for k in disagreements:
        print("\n%s" % k)
        print("my rxn = %s" % serial_hullout[k]["rxn"])
        print("my hull = %.3f" % serial_hullout[k]["Ed"])
        print("mp hull = %.3f" % gs[k]["Ehull_mp"])

    # plt.show()

    fig.savefig(os.path.join(FIG_DIR, "pd_demo_check.png"))


def demo_mixing_hull(gs, end_members, remake=False):
    """
    Now, we'll do a "mixing hull"
        A convex hull with compounds as end members

    e.g., a 2D hull bounded by LiNbO3-Li2NbO3

    Args:
        gs (dict)
            dictionary of ground state data from MPQuery for chemsys

        end_members (list)
            list of end members to use for mixing hull

        remake (bool)

    """
    print("\n now a mixing hull")
    fjson = os.path.join(
        DATA_DIR, "pd_demo_mixing_hull_%s-%s.json" % (end_members[0], end_members[1])
    )
    if not remake and os.path.exists(fjson):
        return read_json(fjson)

    mh = MixingHull(
        input_energies=gs,
        end_members=end_members,
        energy_key="Ef_mp",
    )
    out = mh.results
    for k in out:
        if out[k]["stability"]:
            print(k)
            print(out[k]["x"])
            print(out[k]["E_mix"])

    return out


def main():
    # if True, re-grab data from MP
    remake_query = True
    # if True, re-calculate hull input data
    remake_serial_hullin = True
    # if True, re-calculate hull output data
    remake_serial_hullout = True
    # if True, re-calculate hull output data in parallel
    remake_parallel_hullout = True
    # if True, generate figure to check results
    remake_hull_figure_check = True
    # if True, remake mixing hull
    remake_mixing_hull = True

    # MP query for CHEMSYS
    gs = get_mp_data_for_chemsys(CHEMSYS, remake=remake_query)

    # hull input data for CHEMSYS
    hullin = serial_get_hull_input_data(gs, remake=remake_serial_hullin)

    # hull output data for CHEMSYS
    hullout = serial_get_hull_output_data(hullin, remake=remake_serial_hullout)

    # hull output data for CHEMSYS (generated using parallelization)
    p_hullout = parallel_get_hull_input_and_output_data(
        gs, remake=remake_parallel_hullout
    )

    # generate a graph that compares serial vs parallel hull output and also compares ThermoTools hull output to MP hull data
    if remake_hull_figure_check:
        plot_to_check_success(gs, hullout, p_hullout)

    # generate a mixing hull
    mixing = demo_mixing_hull(gs, ["Li2O", "NbO2"], remake=remake_mixing_hull)
    return gs, hullin, hullout, p_hullout, mixing


if __name__ == "__main__":
    # MP Query --> hull input data (serial) --> hull output data (serial) --> hull output data (parallel)
    gs = main()

USE_INSTALLED_PYDMC = True

if USE_INSTALLED_PYDMC:
    from pydmc.VASPTools import VASPSetUp, VASPAnalysis
    from pydmc.SubmitTools import SubmitTools
    from pydmc.CompTools import CompTools
    from pydmc.handy import (
        read_json,
        write_json,
        read_yaml,
        write_yaml,
        is_slurm_job_in_queue,
    )
    from pydmc.MagTools import MagTools
    from pydmc.StrucTools import StrucTools
    from pydmc.MPQuery import MPQuery
else:
    from VASPTools import VASPSetUp, VASPAnalysis
    from SubmitTools import SubmitTools
    from CompTools import CompTools
    from handy import (
        read_json,
        write_json,
        read_yaml,
        write_yaml,
        is_slurm_job_in_queue,
    )
    from MagTools import MagTools
    from StrucTools import StrucTools
    from MPQuery import MPQuery
from pymatgen.core.structure import Structure

import os
import subprocess

"""
This demo tests the following pydmc modules:
    1) MPQuery to get structures from Materials Project
    2) MagTools to get magnetic configurations for those structures
    3) VASPSetUp to prepare VASP input files
    4) SubmitTools to submit jobs to the queue
    5) VASPAnalysis to analyze VASP output files
        - also tests SiteTools to get magnetic info during analysis

"""

# where is demo_VASPTools.py
SCRIPTS_DIR = os.getcwd()

# where to put .json files
DATA_DIR = os.path.join(SCRIPTS_DIR, "examples", "vasp_mp", "data")
if not os.path.exists(DATA_DIR):
    os.mkdir(DATA_DIR)

# where to run calculations
CALCS_DIR = DATA_DIR.replace("data", "calcs")
if not os.path.exists(CALCS_DIR):
    os.mkdir(CALCS_DIR)

# Chris' API key (replace with your own)
API_KEY = "__API_KEY__"

# how many AFM orderings I want to sample
MAX_AFM_IDX = 2

# which magnetic configurations I want to calculate (note: AFM "mag" will be "afm_0", "afm_1", ... "afm_N"
MAGS = (
    ["nm", "fm"] + ["afm_%s" % str(int(v)) for v in range(MAX_AFM_IDX)]
    if MAX_AFM_IDX != 0
    else []
)

# what kind of calculation I want to do
CALC = "relax"

# what kind of DFT I want to do
XC = "metagga"

# what "standard" settings I want to use
STANDARD = "mp"

# which MP IDs I want to calculate and what are their chemical formulas
MPIDS = {"mp-755657": "O3V1Ti1", "mp-28226": "Cr2Mn1O4"}

# MSI user name (replace w/ your user name)
USERNAME = "cbartel"


def query_mp(remake=False):
    """
    get starting structures from MP
    """
    fjson = os.path.join(DATA_DIR, "query.json")
    if os.path.exists(fjson) and not remake:
        return read_json(fjson)

    mpq = MPQuery(API_KEY)
    IDs = list(MPIDS.keys())

    d = {}
    for mpid in IDs:
        s = mpq.get_structure_by_material_id(mpid)
        formula = StrucTools(s).compact_formula
        d[mpid] = {"structure": s.as_dict()}

    return write_json(d, fjson)


def get_afm_magmoms(query, remake=False):
    """
    get MAGMOMs for AFM calculations
    """

    fjson = os.path.join(DATA_DIR, "magmoms.json")
    if os.path.exists(fjson) and not remake:
        return read_json(fjson)

    magmoms = {}
    for mpid in query:
        magmoms[mpid] = {}
        s = Structure.from_dict(query[mpid]["structure"])
        magtools = MagTools(s)
        afm_strucs = magtools.get_antiferromagnetic_structures
        if afm_strucs:
            for i in range(len(afm_strucs)):
                magmoms[mpid][i] = afm_strucs[i].site_properties["magmom"]

    return write_json(magmoms, fjson)


def get_job_name(launch_dir):
    """
    descriptive job name (formula-mpid-mag)
    """
    return "-".join(launch_dir.split("/")[-3:])


def get_launch_dir(formula, mpid, mag):
    """
    where I want to "$ sbatch sub.sh" to launch all calcs in a tree

    examples/calcs/formula/mpid/mag/
    """

    launch_dir = os.path.join(CALCS_DIR, formula, mpid, mag)
    if not os.path.exists(launch_dir):
        os.makedirs(launch_dir)
    return launch_dir


def prepare_calc_and_launch(
    mpid,
    mag,
    ready_to_launch=False,
    fresh_restart=False,
    refresh_configs_yaml=False,
    user_name=USERNAME,
):
    """
    prepare directories for launch
    make subdirectories for all calculations for a given mpid, mag
    populate with VASP input files
    populate with submission scripts
    submits calculation to the queue (if ready_to_launch = True)

    Args:

    mpid (str) - Materials Project ID
    mag (str) - 'nm' = nonmagnetic, 'fm' = ferromagnetic, 'afm_#' = antiferromagnetic with # idx ordering
    ready_to_launch (bool) - True if submit jobs now; False if you want to check directories first
    fresh_restart (bool) - if True, delete progress and start over; if False, pick up where you left off
    refresh_configs_yaml (bool) - if True, grab the base_configs.yaml from pydmc; if False, keep the one you've been editing in SCRIPTS_DIR
    user_name (str) - change to your name *******DON'T FORGET THIS ONE ********

    """

    # if you want to start from pydmc yaml in SCRIPTS_DIR
    # NOTE: user_configs will over-ride things in base_configs.yaml but never write to it
    # this option just gives you a chance to start from a fresh base_configs.yaml
    # in case you manually edited this file for some reason
    # if you haven't touched base_configs, this can always be False
    if refresh_configs_yaml:
        fyaml = os.path.join(os.getcwd(), "base_configs.yaml")
        if os.path.exists(fyaml):
            os.remove(fyaml)

    # load dictionaries from DATA_DIR/*json
    query = query_mp()
    magmoms = get_afm_magmoms(query)

    # determine launch directory
    formula = MPIDS[mpid]

    # if structure is nonmagnetic, don't run magnetic calculations
    struc = Structure.from_dict(query[mpid]["structure"])
    if (mag != "nm") and not (MagTools(struc).could_be_magnetic):
        print("%s is not magnetic; not running" % formula)
        return

    # make launch_dir (where all calculations will live for a given structure + magnetic configuration
    launch_dir = get_launch_dir(formula, mpid, mag)

    # determine magmom if calc is afm
    if "afm" in mag:
        afm_idx = mag.split("_")[1]
        if str(afm_idx) in magmoms[mpid]:
            magmom = magmoms[mpid][str(afm_idx)]
        else:
            # skip if afm_* not in magmoms (eg we didn't make that many magmoms)
            return
    else:
        # set magmom to None if not AFM (handled elsewhere in VASPSetUp)
        magmom = None

    # put POSCAR in launch_dir if it's not there already
    launch_pos = os.path.join(launch_dir, "POSCAR")
    if not os.path.exists(launch_pos) or fresh_restart:
        struc.to(filename=launch_pos, fmt="POSCAR")

    # check to see if job is in the queue (**** MAKE SURE YOUR USERNAME IS AN ARG ****)
    job_name = get_job_name(launch_dir)
    if is_slurm_job_in_queue(job_name, user_name=user_name):
        print("%s already in queue, not messing with it" % job_name)
        return

    # specify configurations that you want to change in base_configs.yaml (some may already be there - that's OK)
    user_configs = {}
    user_configs["mag"] = mag  # use the mag I'm iterating through
    user_configs["fresh_restart"] = fresh_restart  # restart or not according to arg
    user_configs["standard"] = STANDARD  # use the standard we decided on above
    user_configs["calc"] = CALC  # use the calc we decided on above
    user_configs["xc"] = XC  # use the exchange-correlation we decided above
    user_configs["job-name"] = job_name  # use a unique job name
    user_configs["partition"] = (
        "msismall"  # choose a partition (change this if jobs aren't starting quickly)
    )
    user_configs["nodes"] = 1  # choose # nodes (increase if jobs are taking too long)
    user_configs["ntasks"] = (
        16  # choose # tasks (increase if jobs are taking too long; decrease if trouble getting jobs running in a "small" queue
    )

    sub = SubmitTools(launch_dir=launch_dir, magmom=magmom, user_configs=user_configs)
    sub.write_sub
    if ready_to_launch:
        os.chdir(launch_dir)
        subprocess.call(["sbatch", "sub.sh"])
        os.chdir(SCRIPTS_DIR)


def launch_calcs(
    ready_to_launch=False, fresh_restart=False, refresh_configs_yaml=False
):
    """
    loop through calculation directories, prepare them, and launch them

    Args:
        ready_to_launch (bool) - True if submit jobs now; False if you want to check directories first
        fresh_restart (bool) - if True, delete progress and start over; if False, pick up where you left off
        refresh_configs_yaml (bool) - if True, grab the base_configs.yaml from pydmc; if False, keep the one you've been editing in SCRIPTS_DIR

    """
    for MPID in MPIDS:
        print("\nworking on %s (%s)" % (MPID, MPIDS[MPID]))
        for MAG in MAGS:
            print("mag = %s" % MAG)
            prepare_calc_and_launch(
                mpid=MPID,
                mag=MAG,
                ready_to_launch=ready_to_launch,
                fresh_restart=fresh_restart,
                refresh_configs_yaml=refresh_configs_yaml,
            )


def get_calc_dirs_for_launch_dir(mpid, mag):
    """
    Args:
        mpid (str) - Materials Project ID
        mag (str) - 'nm' = nonmagnetic, 'fm' = ferromagnetic, 'afm_#' = antiferromagnetic with # idx ordering

    finds all the calculation directories for a given structure/magnetic configuration pair (e.g., LAUNCH_DIR/gga-loose might be one of them)
    """
    formula = MPIDS[mpid]
    launch_dir = get_launch_dir(formula, mpid, mag)
    calcs = [c for c in os.listdir(launch_dir) if "gga" in c]
    calc_dirs = [os.path.join(launch_dir, c) for c in calcs]
    return calc_dirs


def purge_large_files_for_completed_calcs(mpid, mag):
    """
    gets rid of some large files to make space.. doing this mostly for pushing this example to github

    note:
    - if you are doing a bader analysis, you need AECCAR*
    - if you are doing a LOBSTER analysis, you need WAVECAR
    - if you want to plot the electron density, you need CHGCAR and ELFCAR
    """
    large_files = [
        "WAVECAR",
        "CHGCAR",
        "PROCAR",
        "LOCPOT",
        "ELFCAR",
        "CHG",
        "AECCAR0",
        "AECCAR1",
        "AECCAR2",
    ]

    calc_dirs = get_calc_dirs_for_launch_dir(mpid, mag)
    n_calcs = len(calc_dirs)
    n_converged = 0
    for calc_dir in calc_dirs:
        va = VASPAnalysis(calc_dir)
        if va.is_converged:
            n_converged += 1
    if n_converged == n_calcs:
        print("purging large files")
        for calc_dir in calc_dirs:
            for f in large_files:
                file_to_purge = os.path.join(calc_dir, f)
                print("purging %s" % file_to_purge)
                if os.path.exists(file_to_purge):
                    os.remove(file_to_purge)


def analyze_calc(mpid, mag):
    """
    Args:
        mpid (str) - Materials Project ID
        mag (str) - 'nm' = nonmagnetic, 'fm' = ferromagnetic, 'afm_#' = antiferromagnetic with # idx ordering

    Returns:
        dictionary for results of each calculation
            {'convergence' : True if calculation is converged, otherwise False,
            'E' : total energy (eV/at)
            'mag' : {el : {site index : {'mag' : magnetization of that site}}}}

    """
    d = {}
    calc_dirs = get_calc_dirs_for_launch_dir(mpid, mag)
    for calc_dir in calc_dirs:
        va = VASPAnalysis(calc_dir)
        convergence = va.is_converged
        print("converged = %s" % convergence)
        E_per_at = va.E_per_at
        mag = va.magnetization
        d[calc_dir.split("/")[-1]] = {
            "convergence": convergence,
            "E": E_per_at,
            "mag": mag,
        }
        print(d)
    return d


def analyze_calcs(remake=False):
    """
    Loops through all structure/magnetic configuration pairs and analyzes calculation results

    Returns:
        {formula : {ID : {CALCULATION RESULTS}}}

    """
    fjson = os.path.join(DATA_DIR, "results.json")
    if not remake and os.path.exists(fjson):
        return read_json
    d = {}
    for MPID in MPIDS:
        formula = MPIDS[MPID]
        print("\nanalyzing %s (%s)" % (MPID, formula))
        if formula not in d:
            d[formula] = {}
        d[formula][MPID] = {}
        for MAG in MAGS:
            print("mag = %s" % MAG)
            d[formula][MPID][MAG] = analyze_calc(MPID, MAG)
            purge_large_files_for_completed_calcs(MPID, MAG)
    return write_json(d, fjson)


def main():
    running_on_msi = True  # only launch calcs if you're on MSI

    print("\n Querying MP (or reading existing json)")
    query = query_mp(remake=True)
    print("\n Making magmoms (or reading existing json)")
    magmoms = get_afm_magmoms(query, remake=True)

    print("\n Launching calcs (or not if they are finished)")
    if running_on_msi:
        launch_calcs(
            ready_to_launch=True, fresh_restart=True, refresh_configs_yaml=False
        )

    print("\n Analyzing calcs (or reading from existing json)")
    results = analyze_calcs(remake=True)

    return query, magmoms, results


if __name__ == "__main__":
    query, magmoms, results = main()

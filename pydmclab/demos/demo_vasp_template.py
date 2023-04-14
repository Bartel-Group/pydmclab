import os
import numpy as np
import multiprocessing as multip

from pydmclab.utils.handy import read_json, write_json, make_sub_for_launcher
from pydmclab.core.query import MPQuery
from pydmclab.core.mag import MagTools
from pydmclab.core.struc import StrucTools
from pydmclab.hpc.launch import LaunchTools
from pydmclab.hpc.submit import SubmitTools
from pydmclab.hpc.analyze import AnalyzeBatch
from pydmclab.core.energies import ChemPots, FormationEnthalpy

"""
see [pydmclab docs](https://github.umn.edu/bartel-group/pydmclab/blob/main/docs.md) for help
"""

# where is this file
SCRIPTS_DIR = os.getcwd()

# where are my calculations going to live
CALCS_DIR = SCRIPTS_DIR.replace("scripts", "calcs")

# where is my data going to live
DATA_DIR = SCRIPTS_DIR.replace("scripts", "data")

for d in [CALCS_DIR, DATA_DIR]:
    if not os.path.exists(d):
        os.makedirs(d)

# if you need data from MP as a starting point (often the case), you need your API key
API_KEY = "***REMOVED***"

# lets put a tag on all the files we save
FILE_TAG = CALCS_DIR.split("/")[-2]

# what to query MP for
## e.g., 'MnO2', ['MnO2', 'TiO2'], 'Ca-Ti-O, etc
COMPOSITIONS = None

# how to transform MP structures
## e.g., [x/8 for x in range(9)]
## NOTE: you need to modify get_strucs to make this work (hard to generalize)
TRANSFORM_STRUCS = False

# whether or not you want to generate MAGMOMs
## True if you're running AFM, else False
GEN_MAGMOMS = False

# what {standard : [final_xcs]} to calculate
## e.g., {'dmc' : ['metagga', 'ggau']} if you want to run METAGGA + GGA+U at DMC standards
TO_LAUNCH = {}

# any configurations related to LaunchTools
## e.g., {'compare_to_mp' : True, 'n_afm_configs' : 4}
LAUNCH_CONFIGS = {}

# any configurations related to SubmitTools
## usually no need to change anything but n_procs... n_procs will determine how to parallelize over launch_dirs
## NOTE: do not set n_procs = 'all' unless you are running on a compute node (ie not a login node)
SUB_CONFIGS = {"n_procs": 1}

# any configurations related to Slurm
## e.g., {'ntasks' : 16, 'time' : int(24*60)}
SLURM_CONFIGS = {}

# any configurations related to VASPSetUp
## e.g., {'lobster_static' : True, 'relax_incar' : {'ENCUT' : 555}}
VASP_CONFIGS = {}

# any configurations related to AnalyzeBatch
## e.g., {'include_meta' : True, 'include_mag' : True, 'n_procs' : 4}
## NOTE: do not set n_procs = 'all' unless you are running on a compute node (ie not a login node)
ANALYSIS_CONFIGS = {"n_procs": 1}

"""
Don't forget to inspect the arguments to:
    get_query
    get_strucs
    get_magmoms
    get_launch_dirs
    submit_calcs
    get_results

You'll want to customize these depending on your calculations
"""


def get_query(
    comp,
    properties=None,
    criteria=None,
    only_gs=True,
    include_structure=True,
    supercell_structure=False,
    max_Ehull=0.05,
    max_sites_per_structure=65,
    max_strucs_per_cmpd=4,
    savename="query_%s.json" % FILE_TAG,
    remake=False,
):
    """
    Args:
        comp (list or str)
            can either be:
                - a chemical system (str) of elements joined by "-"
                - a chemical formula (str)
            can either be a list of:
                - chemical systems (str) of elements joined by "-"
                - chemical formulas (str)

        properties (list or None)
            list of properties to query
                - if None, then use typical_properties
                - if 'all', then use supported_properties

        criteria (dict or None)
            dictionary of criteria to query
                - if None, then use {}

        only_gs (bool)
            if True, remove non-ground state polymorphs for each unique composition

        include_structure (bool)
            if True, include the structure (as a dictionary) for each entry

        supercell_structure (bool)
            only runs if include_structure = True
            if False, just retrieve the MP structure
            if not False, must be specified as [a,b,c] to make an a x b x c supercell of the MP structure

        max_Ehull (float)
            if not None, remove entries with Ehull_mp > max_Ehull

        max_sites_per_structure (int)
            if not None, remove entries with more than max_sites_per_structure sites

        max_strucs_per_cmpd (int)
            if not None, only retain the lowest energy structures for each composition until you reach max_strucs_per_cmpd

        savename (str) - filename for fjson in DATA_DIR

        remake (bool) - write (True) or just read (False) fjson

    Returns:
        {mpid : {DATA}}
    """

    fjson = os.path.join(DATA_DIR, savename)
    if os.path.exists(fjson) and not remake:
        return read_json(fjson)

    # initialize MPQuery with your API key
    mpq = MPQuery(api_key=API_KEY)

    # get the data from MP
    data = mpq.get_data_for_comp(
        comp=comp,
        properties=properties,
        criteria=criteria,
        only_gs=only_gs,
        include_structure=include_structure,
        supercell_structure=supercell_structure,
        max_Ehull=max_Ehull,
        max_sites_per_structure=max_sites_per_structure,
        max_strucs_per_cmpd=max_strucs_per_cmpd,
    )

    write_json(data, fjson)
    return read_json(fjson)


def check_query(query):
    for mpid in query:
        print("\nmpid: %s" % mpid)
        print("\tcmpd: %s" % query[mpid]["cmpd"])
        print("\tstructure formula: %s" % StrucTools(query[mpid]["structure"]).formula)


def get_strucs(
    query,
    transform_strucs,
    max_strucs_per_starting_struc=1,
    ox_states=None,
    savename="strucs_%s.json" % FILE_TAG,
    remake=False,
):
    """
    Args:
        query (dict) - {mpid : {DATA}}
        transform_strucs (bool or list):
            - if False, just use the structures from MP
            - if not False:
                - this should be customized to perform whatever transformations you'd like to perform
                - *NOTE*: you should also customize species_mapping in this function
        max_strucs_per_starting_struc (int) - max number of structures to keep for each starting structure
        ox_states (dict) - {element : oxidation state} for transformations
            - if None, use AutoOxidationState algo in pymatgen
        savename (str) - filename for fjson in DATA_DIR
        remake (bool) - write (True) or just read (False) fjson

    Returns:
        if not transform_strucs:
            {formula : {mpid : structure}
        if transform_strucs:
            {formula_identifier : {index_identifier : structure}}
    """

    fjson = os.path.join(DATA_DIR, savename)
    if os.path.exists(fjson) and not remake:
        return read_json(fjson)

    # get all unique chemical formulas in the query
    formulas_in_query = sorted(list(set([query[mpid]["cmpd"] for mpid in query])))

    data = {}
    for formula in formulas_in_query:
        # get all MP IDs in your query having that formula
        mpids = [mpid for mpid in query if query[mpid]["cmpd"] == formula]
        if not transform_strucs:
            # if no transformations, just return the structures from MP
            data[formula] = {mpid: query[mpid]["structure"] for mpid in mpids}
        else:
            # now you need to customize this for whatever transformations you want
            if len(mpids) > 1:
                print("WARNING: %s has %i mpids" % (formula, len(mpids)))
                raise NotImplementedError(
                    "folder structure is not super amenable to this... Reconfiguring recommended"
                )

            # take our 1 MP ID of interest and transform it
            mpid = mpids[0]
            initial_structure = query[mpid]["structure"]
            for x in transform_strucs:
                # make a new "formula" that includes are iterator in the transformation
                key = "_".join([formula, str(x)])
                structools = StrucTools(
                    structure=initial_structure, ox_states=ox_states
                )

                species_mapping = None  # *NOTE*: you should customize this for your desired transformation

                # strucs will have the form {index_identifier : structure}
                strucs = structools.replace_species(
                    species_mapping=species_mapping,
                    n_strucs=max_strucs_per_starting_struc,
                )

                data[key] = strucs

    write_json(data, fjson)
    return read_json(fjson)


def check_strucs(strucs):
    for formula in strucs:
        for ID in strucs[formula]:
            print("\nformula: %s" % formula)
            print("\tID: %s" % ID)
            struc = strucs[formula][ID]
            print("\tstructure formula: %s" % StrucTools(struc).formula)


def get_magmoms(
    strucs,
    max_afm_combos=50,
    treat_as_nm=[],
    savename="magmoms_%s.json" % FILE_TAG,
    remake=False,
):
    """
    Args:
        strucs (dict) - {formula : {ID : structure}}
        max_afm_combos (int): maximum number of AFM spin configurations to generate
        treat_as_nm (list): any normally mag els you'd like to treat as nonmagnetic for AFM enumeration
        savename (str) - filename for fjson in DATA_DIR
        remake (bool) - write (True) or just read (False) fjson

    Returns:
        {formula : {ID : {AFM configuration index : [list of magmoms on each site]}
    """

    fjson = os.path.join(DATA_DIR, savename)
    if not remake and os.path.exists(fjson):
        return read_json(fjson)

    magmoms = {}
    for formula in strucs:
        magmoms[formula] = {}
        for ID in strucs[formula]:
            # for each unique structure, get AFM magmom orderings
            structure = strucs[formula][ID]
            magtools = MagTools(
                structure=structure,
                max_afm_combos=max_afm_combos,
                treat_as_nm=treat_as_nm,
            )
            curr_magmoms = magtools.get_afm_magmoms
            magmoms[formula][ID] = curr_magmoms

    write_json(magmoms, fjson)
    return read_json(fjson)


def check_magmoms(strucs, magmoms):
    for formula in strucs:
        for ID in strucs[formula]:
            structure_formula = StrucTools(strucs[formula][ID]).formula
            n_afm_configs = len(magmoms[formula][ID])
            print("%s: %i AFM configs\n" % (structure_formula, n_afm_configs))


def get_launch_dirs(
    strucs,
    magmoms,
    to_launch,
    user_configs,
    make_launch_dirs=True,
    refresh_configs=True,
    savename="launch_dirs_%s.json" % FILE_TAG,
    remake=False,
):
    """
    Args:
        strucs (dict) - {formula : {ID : structure}}
        magmoms (dict) - {formula : {ID : {AFM configuration index : [list of magmoms on each site]}}
        to_launch (dict) - {standard (str) : [list of final_xcs of interest]}
        user_configs (dict) - optional launch configurations
        make_launch_dirs (bool) - make launch directories (True) or just return launch dict (False)
        refresh_configs (bool) - refresh configs (True) or just use existing configs (False)
        savename (str) - filename for fjson in DATA_DIR
        remake (bool) - write (True) or just read (False) fjson

    Returns:
        {launch_dir (formula/ID/standard/mag) : {'xcs' : [list of final_xcs],
                                                 'magmoms' : [list of magmoms for each site in structure in launch_dir]}}

        also makes launch_dir and populates with POSCAR using strucs if make_dirs=True

    """

    fjson = os.path.join(DATA_DIR, savename)
    if os.path.exists(fjson) and not remake:
        return read_json(fjson)

    all_launch_dirs = {}
    for formula in strucs:
        for ID in strucs[formula]:
            # for each unique structure, generate our launch directories
            structure = strucs[formula][ID]
            if magmoms:
                curr_magmoms = magmoms[formula][ID]
            else:
                curr_magmoms = None
            top_level = formula
            unique_ID = ID

            launch = LaunchTools(
                calcs_dir=CALCS_DIR,
                to_launch=to_launch,
                user_configs=user_configs,
                magmoms=curr_magmoms,
                structure=structure,
                top_level=top_level,
                unique_ID=unique_ID,
                refresh_configs=refresh_configs,
            )

            launch_dirs = launch.launch_dirs(make_dirs=make_launch_dirs)

            for launch_dir in launch_dirs:
                all_launch_dirs[launch_dir] = launch_dirs[launch_dir]

    write_json(all_launch_dirs, fjson)
    return read_json(fjson)


def check_launch_dirs(launch_dirs):
    print("\nanalyzing launch directories")
    for d in launch_dirs:
        print("\nlaunching from %s" % d)
        print("   these final xcs: %s" % launch_dirs[d]["xcs"])


def submit_one_calc(submit_args):

    """
    Prepares VASP inputs, writes submission script, and launches job for one launch_dir

    Args:
        submit_args (dict) should contain:
        {'launch_dir' :
            launch_dir (str) - (formula/ID/standard/mag) to write and launch submission script in,
         'launch_dirs' :
            launch_dirs (dict) - {launch_dir (formula/ID/standard/mag) : {'xcs' : [list of final_xcs], 'magmoms' : [list of magmoms for each site in structure in launch_dir]}},
         'user_configs' :
            user_configs (dict) - optional sub, slurm, or VASP configurations,
         'refresh_configs' :
            refresh_configs (list) - list of which configs to refresh,
         'ready_to_launch':
            ready_to_launch (bool) - write (True) and launch (True) or just write (False) submission scripts (False)
            }

    Returns:
        None

    """
    launch_dir = submit_args["launch_dir"]
    launch_dirs = submit_args["launch_dirs"]
    user_configs = submit_args["user_configs"]
    refresh_configs = submit_args["refresh_configs"]
    ready_to_launch = submit_args["ready_to_launch"]

    # what are our terminal xcs for that launch_dir
    final_xcs = launch_dirs[launch_dir]["xcs"]

    # what magmoms apply to that launch_dir
    magmom = launch_dirs[launch_dir]["magmom"]

    try:
        sub = SubmitTools(
            launch_dir=launch_dir,
            final_xcs=final_xcs,
            magmom=magmom,
            user_configs=user_configs,
            refresh_configs=refresh_configs,
        )

        # prepare VASP directories and write submission script
        sub.write_sub

        # submit submission script to the queue
        if ready_to_launch:
            sub.launch_sub

        success = True
    except TypeError:
        print("\nERROR: %s\n   will submit without multiprocessing" % launch_dir)
        success = False

    return {"launch_dir": launch_dir, "success": success}


def submit_calcs(
    launch_dirs,
    user_configs={},
    refresh_configs=["vasp", "sub", "slurm"],
    ready_to_launch=True,
    n_procs=1,
):
    """
    Prepares VASP inputs, writes submission script, and launches job for all launch_dirs

    Args:
        launch_dirs (dict) - {launch_dir (formula/ID/standard/mag) : {'xcs' : [list of final_xcs], 'magmoms' : [list of magmoms for each site in structure in launch_dir]}}
        user_configs (dict) - optional sub, slurm, or VASP configurations
        refresh_configs (list) - list of which configs to refresh
        ready_to_launch (bool) - write (True) and launch (True) or just write (False) submission scripts (False

    Returns:
        None

    """

    submit_args = {
        "launch_dirs": launch_dirs,
        "user_configs": user_configs,
        "refresh_configs": refresh_configs,
        "ready_to_launch": ready_to_launch,
    }

    if n_procs == 1:
        print("\n\n submitting calculations in serial\n\n")
        for launch_dir in launch_dirs:
            curr_submit_args = submit_args.copy()
            curr_submit_args["launch_dir"] = launch_dir
            submit_one_calc(curr_submit_args)
        return
    elif n_procs == "all":
        n_procs = multip.cpu_count() - 1

    print("\n\n submitting calculations in parallel\n\n")
    print("not refreshing configs for parallel --> causes trouble")
    submit_args["refresh_configs"] = refresh_configs
    list_of_submit_args = []
    for launch_dir in launch_dirs:
        curr_submit_args = submit_args.copy()
        curr_submit_args["launch_dir"] = launch_dir
        list_of_submit_args.append(curr_submit_args)
    pool = multip.Pool(processes=n_procs)
    statuses = pool.map(submit_one_calc, list_of_submit_args)
    pool.close()

    submitted_w_multiprorcessing = [status for status in statuses if status["success"]]
    failed_w_multiprocessing = [status for status in statuses if not status["success"]]

    print(
        "%i/%i calculations submitted with multiprocessing"
        % (len(submitted_w_multiprorcessing), len(statuses))
    )
    for status in failed_w_multiprocessing:
        launch_dir = status["launch_dir"]
        curr_submit_args = submit_args.copy()
        curr_submit_args["launch_dir"] = launch_dir
        submit_one_calc(curr_submit_args)

    return


def get_results(
    launch_dirs,
    user_configs,
    refresh_configs=True,
    savename="results_%s.json" % FILE_TAG,
    remake=False,
):
    """
    Args:
        launch_dirs (dict) - {launch_dir (formula/ID/standard/mag) : {'xcs' : [list of final_xcs], 'magmoms' : [list of magmoms for each site in structure in launch_dir]}}
        user_configs (dict) - optional analysis configurations
        refresh_configs (bool) - refresh configs (True) or just use existing configs (False)
        savename (str) - filename for fjson in DATA_DIR
        remake (bool) - write (True) or just read (False) fjson

    Returns:
        {'formula.ID.standard.mag.xc_calc' : {scraped results from VASP calculation}}
    """

    fjson = os.path.join(DATA_DIR, savename)
    if os.path.exists(fjson) and not remake:
        return read_json(fjson)

    analyzer = AnalyzeBatch(
        launch_dirs, user_configs=user_configs, refresh_configs=refresh_configs
    )

    data = analyzer.results

    write_json(data, fjson)
    return read_json(fjson)


def check_results(results):

    keys_to_check = list(results.keys())

    converged = 0
    for key in keys_to_check:
        top_level, ID, standard, mag, xc_calc = key.split(".")
        data = results[key]
        convergence = results[key]["results"]["convergence"]
        print("\n%s" % key)
        print("convergence = %s" % convergence)
        if convergence:
            converged += 1
            print("E (static) = %.2f" % data["results"]["E_per_at"])

    print("\n\n SUMMARY: %i/%i converged" % (converged, len(keys_to_check)))


def get_gs(
    results, include_structure=False, savename="gs_%s.json" % FILE_TAG, remake=False
):
    """
    Args:
        results (dict) - {'formula.ID.standard.mag.xc_calc' : {scraped results from VASP calculation}}
        include_structure (bool) - include the structure or not
        savename (str) - filename for fjson in DATA_DIR
        remake (bool) - write (True) or just read (False) fjson

    Returns:
    {standard (str, the calculation standard) :
        {xc (str, the exchange-correlation method) :
            {formula (str) :
                {'E' : energy of the ground-structure,
                'key' : formula.ID.standard.mag.xc_calc for the ground-state structure,
                'structure' : structure of the ground-state structure,
                'n_started' : how many polymorphs you tried to calculate,
                'n_converged' : how many polymorphs are converged,
                'complete' : True if n_converged = n_started (i.e., all structures for this formula at this xc are done)}
    """
    fjson = os.path.join(DATA_DIR, savename)
    if os.path.exists(fjson) and not remake:
        return read_json(fjson)

    standards = sorted(list(set([results[key]['meta']['setup']['standard'] for key in results]))

    gs = {standard : 
            {xc : {} for xc in sorted(list(set([results[key]['meta']['setup']['xc'] 
                                                for key in results 
                                                if results[key]['meta']['setup']['standard'] == standard])))}
            for standard in standards}

    for standard in gs:
        for xc in gs[standard]:
            keys = [k for k in results 
                    if results[k]['meta']['setup']['standard'] == standard
                    if results[k]['meta']['setup']['xc'] == xc]
            
            unique_formulas = sorted(
                list(set([results[key]["results"]["formula"] for key in keys]))
            )
            for formula in unique_formulas:
                formula_keys = [k for k in keys if results[k]["results"]["formula"] == formula]
                converged_keys = [
                k for k in formula_keys if results[k]["results"]["convergence"]
            ]
                if not converged_keys:
                    gs_energy, gs_structure, gs_key = None, None, None
                else:
                    energies = [
                        results[k]["results"]["E_per_at"] for k in converged_keys
                    ]
                    gs_energy = min(energies)
                    gs_key = converged_keys[energies.index(gs_energy)]
                    gs_structure = results[gs_key]["structure"]
            complete = (
                True if len(formula_keys) == len(converged_keys) else False
            )
            gs[xc][formula] = {
                "E": gs_energy,
                "key": gs_key,
                "n_started": len(formula_keys),
                "n_converged": len(converged_keys),
                "complete": complete,
            }
            if include_structure:
                gs[xc][formula]["structure"] = gs_structure

    write_json(gs, fjson)
    return read_json(fjson)


def check_gs(gs):
    """
    checks that this dictionary is generated properly

    Args:
        gs (_type_): _description_

    """

    print("\nchecking ground-states")
    standards = gs.keys()
    print('standards = ', % standards)
    for standard in standards:
        print('\nworking on %s standard' % standard)
        xcs = list(gs[standard].keys())
        for xc in xcs:
            print('  xc = %s' % xcs)
            formulas = list(gs[standard][xc].keys())
            n_formulas = len(formulas)
            n_formulas_complete = len([k for k in formulas if gs[standard][xc][k]["complete"]])
        print(
            "%i/%i formulas with all calculations completed"
            % (n_formulas_complete, n_formulas)
        )

def get_Efs(gs, functional=None, savename="Efs_%s.json" % FILE_TAG, remake=False):
    """
    Args:
        gs (dict) - {formula (str) : {basic stuff for ground-states}}
        functional (str or None) - if None, use default functionals
        savename (str) - filename for fjson in DATA_DIR
        remake (bool) - write (True) or just read (False) fjson

    Returns:
    {xc (str, the exchange-correlation method) :}
        {formula (str) :
            {'E' : energy of the ground-structure,
             'key' : formula.ID.standard.mag.xc_calc for the ground-state structure,
             'structure' : structure of the ground-state structure,
             'n_started' : how many polymorphs you tried to calculate,
             'n_converged' : how many polymorphs are converged,
             'complete' : True if n_converged = n_started (i.e., all structures for this formula at this xc are done),
             'Ef' : formation enthalpy at 0 K (float)}
    """
    fjson = os.path.join(DATA_DIR, savename)
    if os.path.exists(fjson) and not remake:
        return read_json(fjson)
    for standard in gs:
        for xc in gs[standard]:
            if not functional:
                functional = "r2scan" if xc == "metagga" else "pbe"
            mus = ChemPots(functional=functional, standard=standard).chempots
            for formula in gs[xc]:
                E = gs[xc][formula]["E"]
                if E:
                    Ef = FormationEnthalpy(formula=formula, E_DFT=E, chempots=mus).Ef
                else:
                    Ef = None
                gs[xc][formula]["Ef"] = Ef

    write_json(gs, fjson)
    return read_json(fjson)


def check_Efs(Efs):

    print("havent written a quick check for Efs yet")
    return


def main():

    remake_query = False
    print_query_check = True

    remake_strucs = False
    print_strucs_check = True

    remake_magmoms = False
    print_magmoms_check = True

    remake_launch_dirs = False
    print_launch_dirs_check = True

    remake_subs = True
    ready_to_launch = True

    remake_results = True
    print_results_check = True

    remake_gs = True
    print_gs_check = True

    remake_Efs = True
    print_Efs_check = True

    comp = COMPOSITIONS
    query = get_query(comp=comp, remake=remake_query)
    if print_query_check:
        check_query(query)

    transform_strucs = TRANSFORM_STRUCS
    strucs = get_strucs(
        query=query, transform_strucs=transform_strucs, remake=remake_strucs
    )
    if print_strucs_check:
        check_strucs(strucs)

    if GEN_MAGMOMS:
        magmoms = get_magmoms(strucs=strucs, remake=remake_magmoms)
        if print_magmoms_check:
            check_magmoms(strucs=strucs, magmoms=magmoms)
    else:
        magmoms = None

    to_launch = TO_LAUNCH
    launch_configs = LAUNCH_CONFIGS
    launch_dirs = get_launch_dirs(
        strucs=strucs,
        magmoms=magmoms,
        to_launch=to_launch,
        user_configs=launch_configs,
        remake=remake_launch_dirs,
    )
    if print_launch_dirs_check:
        check_launch_dirs(launch_dirs)

    sub_configs = SUB_CONFIGS
    slurm_configs = SLURM_CONFIGS
    vasp_configs = VASP_CONFIGS
    user_sub_configs = {**sub_configs, **slurm_configs, **vasp_configs}
    if remake_subs:
        submit_calcs(
            launch_dirs=launch_dirs,
            user_configs=user_sub_configs,
            ready_to_launch=ready_to_launch,
            n_procs=sub_configs["n_procs"],
        )

    analysis_configs = ANALYSIS_CONFIGS
    results = get_results(
        launch_dirs=launch_dirs, user_configs=analysis_configs, remake=remake_results
    )
    if print_results_check:
        check_results(results)

    gs = get_gs(results=results, remake=remake_gs)

    if print_gs_check:
        check_gs(gs)

    Efs = get_Efs(gs=gs, remake=remake_Efs)

    if print_Efs_check:
        check_Efs(Efs)

    return


if __name__ == "__main__":
    main()

import os
import numpy as np
import multiprocessing as multip

from pydmc.utils.handy import read_json, write_json, make_sub_for_launcher
from pydmc.core.query import MPQuery
from pydmc.core.mag import MagTools
from pydmc.core.struc import StrucTools
from pydmc.hpc.launch import LaunchTools
from pydmc.hpc.submit import SubmitTools
from pydmc.hpc.analyze import AnalyzeBatch

"""
Basic framework

Initial crystal structures
    a) Sometimes these are entries in Materials Project meeting some criteria:
        - e.g., certain chemical space, chemical formula, E above hull, etc
    b) Other times, we might start with an MP structure then perform some transformations to generate new structures
        - e.g., replace x % of Ru with Ir in Ru_{1-x}Ir_{x}O_2 or extract x Li from Li_{1-x}CoO_2
            - note: now we are disordering structures, so each chemical formula will have >= 1 enumerated structure
                - i.e., Ru_0.5Ir_0.5O_2 has > 1 way to configure Ru/Ir on the cation sublattice
    
    The first two levels of our caculation directories will correspond to information relating to the crystal structure:
        - top_level: the chemical formula (or some surrogate for the chemical formula)
            - e.g., this could be RuO2 in 1(a) or it could be x in 1(b)
        - unique_ID: some unique identifier for a crystal structure belonging to that "top_level"
            - e.g., the Materials Project ID for the 1(a) case; or the index of 
        
    Each unique initial crytsal structure will be used for all directories below top_level/unique_ID
    
    Example tree #1:
    calculating RuO2 and IrO2 ground-state polymorphs
    -calcs
     -RuO2
      -mp-1234
     -IrO2
      -mp-5678
      
    Example tree #2:
    calculating Ru_{2-x}Ir_{x}O4 for x = 0, 1, 2 for 2 ordered structures at each x
    -calcs
     -0
      -0
      -1
     -1
      -0
      -1
     -2
      -0
      -1
         
VASP input settings ("standard")

    Usually we want to compute a group of structures with similar settings so that we can safely compare the resulting energies
    
    - right now, we have two standards:
        - dmc: our group standard (to be evolved collectively)
            - these are best practice settings for our group
        - mp: settings to use for strict comparison to Materials Project data
    - standards define the non-structure VASP input files (INCAR, KPOINTS, POTCAR)
    
    This will be the third level of our calculation directory: top_level/unique_ID/standard
    
    If we were calculating with dmc and mp standards, a part of the tree might look like:
    
    -calcs
     -RuO2
      -mp-1234
       -dmc
       -mp
    
Magnetic configuration (mag)

    We need to define what initial magnetic configuration we want to calculate for each initial structure
    - nm: non-magnetic (no spins)
    - fm: ferromagnetic (all spins > 0)
    - afm_*: antiferromagnetic (spins up and down with sum(spins) = 0)
        - * is a unique identifier for the ordering of spins
            - much like partially occupied / disordered structures have multiple ways to configure ions on the lattice,
                - there are often many ways to configure spins on the lattice while still being AFM (sum(spins) = 0)
        - if we want to do AFM calculations, we should first generate a "MAGMOM" (AFM ordering) file for each unique structure and save that data
        - note: if we are computing AFM, we will also compute FM
        
    This will be the fourth level of our calculation directory: top_level/unique_ID/standard/mag
    
    -calcs
     -RuO2
      -mp-1234
       -dmc
        -afm_0

        
Exchange-correlation functional (xc)

Now, we need to decide what flavor of DFT we want to use. Materials Project uses GGA+U. We will often use METAGGA.

For each xc that we use, it's generally a good idea to run calculations in a sequence:
    - a "loose" calculation that has fast-converging standards but not super accurate
    - a "relax" calculation that will optimize the crystal structure with stricter standards
    - a "static" calculation that will give us a better energy/electronic structure at that optimized geometry
    - for METAGGA calculations, we need to first converge a GGA calculation before we start to optimize the geometry with metagga
    
We don't want to have to submit each of these individually to the queue, so we will "pack" them together:

    So, if we're running gga, that means three calculations will get packed together: gga-loose --> gga-relax --> gga-static
        - if we're running metagga, we'll have 5 calcs: gga-loose --> gga-relax --> gga-static --> metagga-relax --> metagga-static
        - each packing or chain of jobs will need a submission script (i.e., something that gets submitted to the queue)
        
Each one of these "xc-calcs" (e.g., gga-relax) will require a VASP execution, meaning it needs VASP inputs and will generate VASP outputs,
    so each xc-calc gets its own directory, becoming the 5th level of our calculation directory:
    
    top_level/unique_ID/standard/mag/xc-calc
    
    -calcs
     -RuO2
      -mp-1234
       -dmc
        -afm_0
         -gga-loose
         -gga-relax
         -gga-static

"""

"""
Most of this is taken care of automatically in pydmc

1) Use MPQuery to get crystal structures from Materials Project
    - write to a dictionary: query.json
2) [OPTIONAL] Transform those structures to generate new structures using StrucTools
    - write to a dictionary: strucs.json
3) [OPTIONAL] Create AFM orderings ("magmoms") for each structure using MagTools(structure).get_afm_magmoms
    - write to a dictionary: magmoms.json
4) Create a dictionary of "launch directories" using LaunchTools
    - customizable with various _launch_configs
    - launch directories are the directories that hold submission scripts
        - these launch directories are defined by the top_level, unique_ID, standard, and mag
    - this will look like: {top_level/unique_ID/standard/mag : 
                                {'xcs' : [XCs to use for that standard],
                                'magmom' : [magmoms to use for that structure]}
    - write to a dictionary: launch_dirs.json
5) Loop through the launch directories and prepare VASP inputs + submission files and launch each chain of VASP jobs using SubmitTools
    - customizable with _sub_configs, _vasp_configs, and _slurm_configs
    - SubmitTools will figure out which jobs can be submitted together and how to order them for submission
    - It will also prepare each VASP job accordingly
6) Crawl through the launch directories, and analyze every VASP calculation using AnalyzeBatch
    - customizable with _analysis_configs
    - write to a dictionary: results.json
"""

# where is this file
SCRIPTS_DIR = os.getcwd()

# where are my calculations going to live
CALCS_DIR = SCRIPTS_DIR.replace('scripts', 'calcs')

# where is my data going to live
DATA_DIR = SCRIPTS_DIR.replace('scripts', 'data')

for d in [CALCS_DIR, DATA_DIR]:
    if not os.path.exists(d):
        os.makedirs(d)

# if you need data from MP as a starting point (often the case), you need your API key
API_KEY = '***REMOVED***'

# lets put a tag on all the files we save
FILE_TAG = CALCS_DIR.split('/')[-2]

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
SUB_CONFIGS = {'n_procs' : 1}

# any configurations related to Slurm
## e.g., {'ntasks' : 16, 'time' : int(24*60)}
SLURM_CONFIGS = {}

# any configurations related to VASPSetUp
## e.g., {'lobster_static' : True, 'relax_incar' : {'ENCUT' : 555}}
VASP_CONFIGS = {}

# any configurations related to AnalyzeBatch
## e.g., {'include_meta' : True, 'include_mag' : True, 'n_procs' : 4}
## NOTE: do not set n_procs = 'all' unless you are running on a compute node (ie not a login node)
ANALYSIS_CONFIGS = {}

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

def get_query(comp, 
              properties=None, 
              criteria=None, 
              only_gs=True, 
              include_structure=True,
              supercell_structure=False,
              max_Ehull=0.05,
              max_sites_per_structure=65,
              max_strucs_per_cmpd=4,
              savename='query_%s.json' % FILE_TAG,
              remake=False):
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
    data = mpq.get_data_for_comp(comp=comp, 
                                 properties=properties, 
                                 criteria=criteria, 
                                 only_gs=only_gs, 
                                 include_structure=include_structure,
                                 supercell_structure=supercell_structure,
                                 max_Ehull=max_Ehull,
                                 max_sites_per_structure=max_sites_per_structure,
                                 max_strucs_per_cmpd=max_strucs_per_cmpd)
    
    write_json(data, fjson) 
    return read_json(fjson)

def check_query(query):
    for mpid in query:
        print('\nmpid: %s' % mpid)
        print('\tcmpd: %s' % query[mpid]['cmpd'])
        print('\tstructure formula: %s' % len(StrucTools(query[mpid]['structure']).formula))

def get_strucs(query,
               transform_strucs,
               max_strucs_per_starting_struc=1,
               ox_states=None,
               savename='strucs_%s.json' % FILE_TAG,
               remake=False):
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
    formulas_in_query = sorted(list(set([query[mpid]['cmpd'] for mpid in query])))
    
    data = {}
    for formula in formulas_in_query:
        # get all MP IDs in your query having that formula
        mpids = [mpid for mpid in query if query[mpid]['cmpd'] == formula]
        if not transform_strucs:
            # if no transformations, just return the structures from MP
            data[formula] = {mpid : query[mpid]['structure'] for mpid in mpids}
        else:
            # now you need to customize this for whatever transformations you want
            if len(mpids) > 1:
                print('WARNING: %s has %i mpids' % (formula, len(mpids)))
                raise NotImplementedError('folder structure is not super amenable to this... Reconfiguring recommended')
            
            # take our 1 MP ID of interest and transform it
            mpid = mpids[0]
            initial_structure = query[mpid]['structure']
            for x in transform_strucs:
                # make a new "formula" that includes are iterator in the transformation
                key = '_'.join([formula, str(x)])
                structools = StrucTools(structure=initial_structure,
                                        ox_states=ox_states)
                
                species_mapping = None # *NOTE*: you should customize this for your desired transformation
                
                # strucs will have the form {index_identifier : structure}
                strucs = structools.replace_species(species_mapping=species_mapping,
                                                    n_strucs=max_strucs_per_starting_struc)
                
                data[key] = strucs       
           
    write_json(data, fjson)
    return read_json(fjson)

def check_strucs(strucs):
    for formula in strucs:
        for ID in strucs[formula]:
            print('\nformula: %s' % formula)
            print('\tID: %s' % ID)
            struc = strucs[formula][ID]
            print('\tstructure formula: %s' % StrucTools(struc).formula)

def get_magmoms(strucs,
                max_afm_combos=50,
                treat_as_nm=[],
                savename='magmoms_%s.json' % FILE_TAG,
                remake=False):
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
            magtools = MagTools(structure=structure,
                                max_afm_combos=max_afm_combos,
                                treat_as_nm=treat_as_nm)
            curr_magmoms = magtools.get_afm_magmoms
            magmoms[formula][ID] = curr_magmoms

    write_json(magmoms, fjson) 
    return read_json(fjson)


def check_magmoms(strucs,
                  magmoms):
    for formula in strucs:
        for ID in strucs[formula]:
            structure_formula = StrucTools(strucs[formula][ID]).formula
            n_afm_configs = len(magmoms[formula][ID])
            print('%s: %i AFM configs\n' % (structure_formula, n_afm_configs))            
        
def get_launch_dirs(strucs,
                    magmoms,
                    to_launch,
                    user_configs,
                    make_launch_dirs=True,
                    refresh_configs=True,
                    savename='launch_dirs_%s.json' % FILE_TAG,
                    remake=False):
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
            
            launch = LaunchTools(calcs_dir=CALCS_DIR,
                                 to_launch=to_launch,
                                 user_configs=user_configs,
                                 magmoms=curr_magmoms,                                 
                                 structure=structure,
                                 top_level=top_level,
                                 unique_ID=unique_ID,
                                 refresh_configs=refresh_configs)
            
            launch_dirs = launch.launch_dirs(make_dirs=make_launch_dirs)
            
            for launch_dir in launch_dirs:
                all_launch_dirs[launch_dir] = launch_dirs[launch_dir]  

    write_json(all_launch_dirs, fjson) 
    return read_json(fjson)  

def check_launch_dirs(launch_dirs):
    print('\nanalyzing launch directories')
    for d in launch_dirs:
        print('\nlaunching from %s' % d)
        print('   these final xcs: %s' % launch_dirs[d]['xcs'])

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
    launch_dir = submit_args['launch_dir']
    launch_dirs = submit_args['launch_dirs']
    user_configs = submit_args['user_configs']
    refresh_configs = submit_args['refresh_configs']
    ready_to_launch = submit_args['ready_to_launch']

    # what are our terminal xcs for that launch_dir
    final_xcs = launch_dirs[launch_dir]['xcs']

    # what magmoms apply to that launch_dir
    magmom = launch_dirs[launch_dir]['magmom']

    try:
        sub = SubmitTools(launch_dir=launch_dir,
                        final_xcs=final_xcs,
                        magmom=magmom,
                        user_configs=user_configs,
                        refresh_configs=refresh_configs)

        # prepare VASP directories and write submission script
        sub.write_sub

        # submit submission script to the queue
        if ready_to_launch:
            sub.launch_sub
            
        success = True
    except TypeError:
        print('\nERROR: %s\n   will submit without multiprocessing' % launch_dir)
        success = False
        
    return {'launch_dir' : launch_dir,
            'success' : success}
    
def submit_calcs(launch_dirs,
                 user_configs={},
                 refresh_configs=['vasp', 'sub', 'slurm'],
                 ready_to_launch=True,
                 n_procs=1):
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
    
    submit_args = {'launch_dirs' : launch_dirs,
                   'user_configs' : user_configs,
                   'refresh_configs' : refresh_configs,
                   'ready_to_launch' : ready_to_launch}
    
    if n_procs == 1:
        print('\n\n submitting calculations in serial\n\n')
        for launch_dir in launch_dirs:
            curr_submit_args = submit_args.copy()
            curr_submit_args['launch_dir'] = launch_dir
            submit_one_calc(curr_submit_args)            
        return
    elif n_procs == 'all':
        n_procs = multip.cpu_count()-1
    
    print('\n\n submitting calculations in parallel\n\n')
    print('not refreshing configs for parallel --> causes trouble')
    submit_args['refresh_configs'] = refresh_configs
    list_of_submit_args = []
    for launch_dir in launch_dirs:
        curr_submit_args = submit_args.copy()
        curr_submit_args['launch_dir'] = launch_dir
        list_of_submit_args.append(curr_submit_args)
    pool = multip.Pool(processes=n_procs)
    statuses = pool.map(submit_one_calc, list_of_submit_args)
    pool.close()
    
    for status in statuses:
        if not status['success']:
            launch_dir = status['launch_dir']
            curr_submit_args = submit_args.copy()
            curr_submit_args['launch_dir'] = launch_dir
            submit_one_calc(curr_submit_args)          
    
    return
                    
def get_results(launch_dirs,
                user_configs,
                refresh_configs=True,
                savename='results_%s.json' % FILE_TAG,
                remake=False):
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
    
    analyzer = AnalyzeBatch(launch_dirs,
                            user_configs=user_configs,
                            refresh_configs=refresh_configs)

    data = analyzer.results

    write_json(data, fjson) 
    return read_json(fjson)

def check_results(results):

    keys_to_check = list(results.keys())

    converged = 0
    for key in keys_to_check:
        top_level, ID, standard, mag, xc_calc = key.split('.')
        data = results[key]
        convergence = results[key]['results']['convergence']
        print('\n%s' % key)
        print('convergence = %s' % convergence)
        if convergence:
            converged += 1
            print('E (static) = %.2f' % data['results']['E_per_at'])
            if ANALYSIS_CONFIGS['include_meta']:
                print('E (relax) = %.2f' % data['meta']['E_relax'])
                print('EDIFFG = %i' % data['meta']['incar']['EDIFFG'])
                print('1st POTCAR = %s' % data['meta']['potcar'][0])
            if (mag != 'nm') and ('include_mag' in ANALYSIS_CONFIGS) and (ANALYSIS_CONFIGS['include_mag']):
                magnetization = data['magnetization']
                an_el = list(magnetization.keys())[0]
                an_idx = list(magnetization[an_el].keys())[0]
                that_mag = magnetization[an_el][an_idx]['mag']
                print('mag on %s (%s) = %.2f' % (an_el, str(an_idx), that_mag))
            if ('include_structure' not in ANALYSIS_CONFIGS) or (ANALYSIS_CONFIGS['include_structure']):
                print(data['structure'])
    
    print('\n\n %i/%i converged' % (converged, len(keys_to_check)))  
    
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
    
    comp = COMPOSITIONS
    query = get_query(comp=comp,
                      remake=remake_query)
    if print_query_check:
        check_query(query)
    
    transform_strucs = TRANSFORM_STRUCS
    strucs = get_strucs(query=query,
                        transform_strucs=transform_strucs,
                        remake=remake_strucs)
    if print_strucs_check:
        check_strucs(strucs)

    if GEN_MAGMOMS:
        magmoms = get_magmoms(strucs=strucs,
                            remake=remake_magmoms)
        if print_magmoms_check:
            check_magmoms(strucs=strucs,
                          magmoms=magmoms)
    else:
        magmoms = None
    
    to_launch = TO_LAUNCH
    launch_configs = LAUNCH_CONFIGS
    launch_dirs = get_launch_dirs(strucs=strucs,
                                  magmoms=magmoms,
                                  to_launch=to_launch,
                                  user_configs=launch_configs,
                                  remake=remake_launch_dirs)
    if print_launch_dirs_check:
        check_launch_dirs(launch_dirs)
    
    sub_configs = SUB_CONFIGS
    slurm_configs = SLURM_CONFIGS
    vasp_configs = VASP_CONFIGS
    user_sub_configs = {**sub_configs, **slurm_configs, **vasp_configs}
    if remake_subs:
        submit_calcs(launch_dirs=launch_dirs,
                     user_configs=user_sub_configs,
                     ready_to_launch=ready_to_launch,
                     n_procs=sub_configs['n_procs'])
        
    analysis_configs = ANALYSIS_CONFIGS
    results = get_results(launch_dirs=launch_dirs,
                          user_configs=analysis_configs,
                          remake=remake_results)
    if print_results_check:
        check_results(results)
    
    return

if __name__ == '__main__':
    main()
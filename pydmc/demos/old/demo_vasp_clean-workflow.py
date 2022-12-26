import os
import numpy as np

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
FILE_TAG = 'clean-workflow'

def main():
    

def get_query(comp=['MoO2', 'TiO2'],
              only_gs=True,
              include_structure=True,
              supercell_structure=[2,1,1],
              savename='query_%s.json' % FILE_TAG,
              remake=False):
    
    fjson = os.path.join(DATA_DIR, savename)
    if os.path.exists(fjson) and not remake:
       return read_json(fjson)
    
    mpq = MPQuery(api_key=API_KEY)
    
    data = mpq.get_data_for_comp(comp=comp,
                                 only_gs=only_gs,
                                 include_structure=include_structure,
                                 supercell_structure=supercell_structure)
    
    write_json(data, fjson) 
    return read_json(fjson)

def check_query(query):
    for mpid in query:
        print('\nmpid: %s' % mpid)
        print('\tcmpd: %s' % query[mpid]['cmpd'])
        print('\tstructure has %i sites' % len(StrucTools(query[mpid]['structure']).structure))

def get_magmoms(query,
                max_afm_combos=20,
                savename='magmoms_%s.json' % FILE_TAG,
                remake=False):
                          
    fjson = os.path.join(DATA_DIR, savename)
    if not remake and os.path.exists(fjson):
        return read_json(fjson)
    
    magmoms = {}
    for mpid in query:
        magmoms[mpid] = {}
        structure = query[mpid]['structure']
        magtools = MagTools(structure=structure,
                            max_afm_combos=max_afm_combos)
        curr_magmoms = magtools.get_afm_magmoms
        magmoms[mpid] = curr_magmoms

    write_json(magmoms, fjson) 
    return read_json(fjson)


def check_magmoms(query,
                  magmoms):
    for mpid in magmoms:
        cmpd = query[mpid]['cmpd']
        curr_magmoms = magmoms[mpid]
        print('\nanalyzing magmoms')
        print('%s: %i AFM configs\n' % (cmpd, len(curr_magmoms)))            
        
def get_launch_dirs(query,
                    magmoms,
                    to_launch={'dmc' : ['ggau', 'metagga']},
                    user_configs={},
                    make_launch_dirs=True,
                    refresh_configs=True,
                    savename='launch_dirs_%s.json' % FILE_TAG,
                    remake=False):
    
    fjson = os.path.join(DATA_DIR, savename)
    if os.path.exists(fjson) and not remake:
        return read_json(fjson)
    
    all_launch_dirs = {}
    for mpid in query:

        structure = query[mpid]['structure']
        curr_magmoms = magmoms[mpid]
        top_level = query[mpid]['cmpd']
        ID = mpid
        
        launch = LaunchTools(calcs_dir=CALCS_DIR,
                             structure=structure,
                             top_level=top_level,
                             unique_ID=ID,
                             to_launch=to_launch,
                             magmoms=curr_magmoms,
                             user_configs=user_configs,
                             refresh_configs=refresh_configs)

        launch_dirs = launch.launch_dirs(make_dirs=make_launch_dirs)

        all_launch_dirs = {**all_launch_dirs, **launch_dirs}

    write_json(all_launch_dirs, fjson) 
    return read_json(fjson)  

def check_launch_dirs(launch_dirs):
    print('\nanalyzing launch directories')
    for d in launch_dirs:
        print('\nlaunching from %s' % d)
        print('   these calcs: %s' % launch_dirs[d]['xcs'])
        
def submit_calcs(launch_dirs,
                 user_configs={},
                 refresh_configs=['vasp', 'sub', 'slurm'],
                 ready_to_launch=True):
    
    for launch_dir in launch_dirs:

        # these are calcs that should be chained in this launch directory
        valid_calcs = launch_dirs[launch_dir]

        # these are some configurations we'll extract from the launch directory name
        top_level, ID, standard, mag = launch_dir.split('/')[-4:]
        xcs_to_run = launch_dirs[launch_dir]['xcs']
        magmom = launch_dirs[launch_dir]['magmom']
        
        # now we'll prep the VASP directories and write the submission script
        sub = SubmitTools(launch_dir=launch_dir,
                          xcs=xcs_to_run,
                          magmom=magmom,
                          user_configs=user_configs,
                          refresh_configs=refresh_configs)

        sub.write_sub
        
        # if we're "ready to launch", let's launch
        if ready_to_launch:
            sub.launch_sub    
            
def check_subs(launch_dirs):
    print('\nanalyzing submission scripts')
    launch_dirs_to_check = list(launch_dirs.keys())
    if len(launch_dirs_to_check) > 6:
        launch_dirs_to_check = launch_dirs_to_check[:3] + launch_dirs_to_check[-3:]

    for d in launch_dirs_to_check:
        xcs = launch_dirs[d]['xcs']
        for xc in xcs:
            fsub = os.path.join(d, 'sub_%s.sh' % xc)
            with open(fsub) as f:
                print('\nanalyzing %s' % fsub)
                for line in f:
                    if 'working' in line:
                        print(line)
                    
def analyze_calcs(launch_dirs,
                  user_configs,
                  refresh_configs=True,
                  savename='results_%s.json' % FILE_TAG,
                  remake=False):
    
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
            print('E (relax) = %.2f' % data['meta']['E_relax'])
            print('EDIFFG = %i' % data['meta']['incar']['EDIFFG'])
            print('1st POTCAR = %s' % data['meta']['potcar'][0])
            if mag != 'nm':
                magnetization = data['magnetization']
                an_el = list(magnetization.keys())[0]
                an_idx = list(magnetization[an_el].keys())[0]
                that_mag = magnetization[an_el][an_idx]['mag']
                print('mag on %s (%s) = %.2f' % (an_el, str(an_idx), that_mag))
            print(data['structure'])
    
    print('\n\n %i/%i converged' % (converged, len(keys_to_check)))  
    
def main():
    """
    It's generally a good idea to set True/False statements at the top
        - this will allow you to quickly toggle whether or not to repeat certain steps
    """    
    remake_sub_launch = False
    
    remake_query = False
    print_query_check = True 
    
    remake_magmoms = False
    print_magmoms_check = True
    
    remake_launch_dirs = False
    print_launch_dirs_check = True
    
    remake_subs = True
    ready_to_launch = True
    print_subs_check = True
    
    remake_results = True
    print_results_check = True
    
    """
    Sometimes we'll need to run our launch script on a compute node if generating magmoms or analyzing directories takes a while
        here, we'll create a file called sub_launch.sh
        you can then execute this .py file on a compute node with:
            $ sbatch sub_launchs.h
    """   
    if remake_sub_launch or not os.path.exists(os.path.join(os.getcwd(), 'sub_launch.sh')):
        make_sub_for_launcher()

    query = get_query(remake=remake_query)

    if print_query_check:
        check_query(query=query)
        
    
    magmoms = get_magmoms(query=query,
                          remake=remake_magmoms)

    if print_magmoms_check:
        check_magmoms(query=query,
                      magmoms=magmoms)
    
    """
    Here, I'll specify the user_configs pertaining to setting up the launch directories
        - let's consider 1 AFM configuration
        - let's use DMC standards
        - and let's compare GGA+U to METAGGA
    """
    launch_configs = {'n_afm_configs' : 1}
    
    launch_dirs = get_launch_dirs(query=query,
                                  magmoms=magmoms,
                                  user_configs=launch_configs,
                                  remake=remake_launch_dirs)
    if print_launch_dirs_check:
        check_launch_dirs(launch_dirs=launch_dirs)
        
    """
    Now, we need to specify any configurations relevant to VASP set up or our submission scripts
    For this example, we'll do the following (on top of the defaults):
        - run on only 8 cores
        - run with a walltime of 80 hours
        - make sure we run LOBSTER
        - use a slightly higher ENCUT in all our calculations
        
    """
    user_configs = {'ntasks' : 8,
                    'time' : int(80*60),
                    'lobster_static' : True,
                    'relax_incar' : {'ENCUT' : 555},
                    'static_incar' : {'ENCUT' : 555},
                    'loose_incar' : {'ENCUT' : 555}}
    
    if remake_subs:
        submit_calcs(launch_dirs=launch_dirs,
                     user_configs=user_configs,
                     ready_to_launch=ready_to_launch)
 
    if print_subs_check:
        check_subs(launch_dirs=launch_dirs)
        
    """
    Now, we can specify what we want to collect from our calculations
        - let's run in parallel w/ 4 processors
        - include metadata
        - include magnetization results
        
    """
    
    analysis_configs = {'n_procs' : 4,
                        'include_meta' : True,
                        'include_mag' : True}
    results = analyze_calcs(launch_dirs=launch_dirs,
                            user_configs=analysis_configs,
                            remake=remake_results)
    if print_results_check:
        check_results(results)

if __name__ == '__main__':
    main()
     


import os
import numpy as np

from pydmc.utils.handy import read_json, write_json, make_sub_for_launcher
from pydmc.core.query import MPQuery
from pydmc.core.mag import MagTools
from pydmc.core.struc import StrucTools
from pydmc.hpc.launch import LaunchTools
from pydmc.hpc.submit import SubmitTools
from pydmc.hpc.analyze import AnalyzeBatch

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
        xcs = launch_dirs_to_check[d]['xcs']
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
     
from pydmc.core.comp import CompTools
from pydmc.core.query import MPQuery
from pydmc.core.hulls import GetHullInputData, AnalyzeHull, ParallelHulls
from pydmc.utils.handy import read_json, write_json
from pydmc.utils.plotting import set_rc_params, tableau_colors    
from pydmc.data.thermochem import mus_at_0K, mus_at_T

import matplotlib.pyplot as plt

import os
import numpy as np

set_rc_params()

"""
This file currently tests the following classes in ThermoTools
    - GetHullInputData
    - AnalyzeHull
    - ParallelHulls

TO-DO:
    - convert into formal "tests"
"""
# Chris B's API key for MP query
API_KEY = '***REMOVED***'

# chemical system to test on
CHEMSYS = 'Ca-Al-Ti-O-F'

# where to save data
DATA_DIR = os.path.join('output', 'hulls', 'data')
if not os.path.exists(DATA_DIR):
    os.mkdir(DATA_DIR)

# where to save figures
FIG_DIR = os.path.join('output', 'hulls', 'figures')
if not os.path.exists(FIG_DIR):
    os.mkdir(FIG_DIR)
    
def get_mp_data_for_chemsys(chemsys=CHEMSYS,
                            only_gs=True,
                            dict_key='cmpd',
                            remake=False,
                            data_dir=DATA_DIR,
                            api_key=API_KEY):
    """
    Args:
        chemsys (str): chemical system to query ('-'.join([elements]))
        only_gs (bool): if True, remove non-ground state polymorphs from MP query
            - good practice to do this before doing hull analysis b/c non-gs polymorphs trivially have Ehull=Ehull_gs + dE_polymorph-gs
        dict_key (str): key to use to orient dictionary of MPQuery results
            - 'cmpd' is the default behavior in MPQuery, meaning we get a dictionary that looks like {CHEMICAL_FORMULA : {DATA}}
        remake (bool): if True, re-query MP
        
    Returns:
        gs (dict): dictionary of ground state data from MPQuery for chemsys
    """
    fjson = os.path.join(data_dir, 'query_'+chemsys+'.json')
    if not remake and os.path.exists(fjson):
        return read_json(fjson)
    
    mpq = MPQuery(api_key)
    out = mpq.get_data_for_comp(comp=chemsys,
                                only_gs=only_gs, 
                                dict_key=dict_key)
    return write_json(out, fjson)

def serial_get_hull_input_data(gs, 
                               formation_energy_key='Ef_mp',
                               remake=False,
                               data_dir=DATA_DIR,
                               chemsys=CHEMSYS):
    """
    Args:
        gs (dict): dictionary of ground state data from MPQuery for chemsys
            - generated with get_mp_data_for_chemsys()
        formation_energy_key (str): key to use for formation energy in gs
            - 'Ef_mp' is default behavior in MPQuery
        remake (bool): if True, re-calculate hull input data
        data_dir (str): directory to save data
        chemsys (str): chemical system to query ('-'.join([elements]))
        
    Returns:
        hullin (dict): dictionary of hull input data for gs
            dict of {chemical space (str) : {formula (str) : {'E' : formation energy (float),
                                                              'amts' : {el (str) : fractional amt of el in formula (float) for el in space}} 
                                            for all relevant formulas including elements}
                - elements are automatically given formation energy = 0
                - chemical space is now in 'el1_el2_...' format to be jsonable
                - each "chemical space" is a convex hull that must be computed
    """
    fjson = os.path.join(data_dir, 'hullin_serial_'+chemsys+'.json')
    if not remake and os.path.exists(fjson):
        return read_json(fjson)
    
    ghid = GetHullInputData(gs, formation_energy_key=formation_energy_key)
    return ghid.hullin_data(fjson=fjson, remake=remake)

def serial_get_hull_output_data(hullin, 
                                remake=False,
                                chemsys=CHEMSYS,
                                data_dir=DATA_DIR):
    """
    Args:
        hullin (dict): dictionary of hull input data for gs
        remake (bool): if True, re-calculate hull output data
        chemsys (str): chemical system to query ('-'.join([elements]))
        data_dir (str): directory to save data
        
    Returns:
        stability data (dict) for all compounds in the specified chemical space
            {compound (str) : {'Ef' : formation energy (float),
                                'Ed' : decomposition energy (float),
                                'rxn' : decomposition reaction (str),
                                'stability' : stable (True) or unstable (False)}}            
    """
    fjson = os.path.join(data_dir, 'hullout_serial_'+chemsys+'.json')
    if not remake and os.path.exists(fjson):
        return read_json(fjson)
    
    hullout = {}
    for space in hullin:
        ah = AnalyzeHull(hullin, space)
        for cmpd in hullin[space]:
            print('\n%s' % cmpd)
            hullout[cmpd] = ah.cmpd_hull_output_data(cmpd)            
    return write_json(hullout, fjson)

def parallel_get_hull_input_and_output_data(gs, 
                                            remake=False,
                                            chemsys=CHEMSYS,
                                            data_dir=DATA_DIR,
                                            n_procs=2,
                                            fresh_restart=True):
    """
    Args:
        gs (dict): dictionary of ground state data from MPQuery for chemsys
        remake (bool): if True, re-calculate hull input and output data
        chemsys (str): chemical system to query ('-'.join([elements]))
        data_dir (str): directory to save data
        n_procs (int): number of processors to use (could also be 'all' to use multip.cpu_count()-1 procs)
        fresh_restart (bool): if True, restart ParallelHull process from scratch
        
    Returns:
        stability data (dict) for all compounds in the specified chemical space
            {compound (str) : {'Ef' : formation energy (float),
                                'Ed' : decomposition energy (float),
                                'rxn' : decomposition reaction (str),
                                'stability' : stable (True) or unstable (False)}}   
            written to fjson
            also writes small_spaces and hullin data resulting from ParallelHull to json
    """
    fjson = os.path.join(data_dir, 'hullout_parallel_'+chemsys+'.json')
    if not remake and os.path.exists(fjson):
        return read_json(fjson)
    ph = ParallelHulls(gs, n_procs=n_procs, fresh_restart=fresh_restart)
    hullin = ph.parallel_hullin(fjson=fjson.replace('hullout', 'hullin'))
    smallest_spaces = ph.smallest_spaces(hullin=hullin,
                                         fjson=fjson.replace('hullout', 'small_spaces'))
    return ph.parallel_hullout(hullin=hullin,
                               smallest_spaces=smallest_spaces,
                               fjson=fjson, remake=True)

def plot_to_check_success(
                          gs,
                          serial_hullout,
                          parallel_hullout):
    """
    Args:
        gs (dict): dictionary of ground state data from MPQuery for chemsys
        serial_hullout (dict): dictionary of hull output data for gs (run serially)
        parallel_hullout (dict): dictionary of hull output data for gs (run in parallel)
    
    Returns:
        compares serial and parallel hull output data (should be identica)
        compares serial and MP hull output data
            - should be identical (or very close) for unstable compounds
            - for stable compounds, MP will have Ehull = 0, whereas our code will compure Ed < 0
    
    """
    set_rc_params()
    
    fig = plt.figure(figsize=(8,3))
    
    params = {'serial' : {'m' : 'o',
                          'c' : 'blue'},
              'parallel' : {'m' : '^',
                            'c' : 'orange'},}
    
    cmpds = sorted(gs.keys())
    
    cmpds = [c for c in cmpds if CompTools(c).n_els > 1]
    
    mp_Ehull = [gs[c]['Ehull_mp'] for c in cmpds]
    
    serial_decomp = [serial_hullout[c]['Ed'] for c in cmpds]
    parallel_decomp = [parallel_hullout[c]['Ed'] for c in cmpds]
    
    
    x = mp_Ehull
    y1 = serial_decomp
    y2 = parallel_decomp
    
    ax1 = plt.subplot(121)
    
    ax1 = plt.scatter(y2, y1, 
                     edgecolor='blue',
                     marker='o',
                     color='white')
                     
    #ax1 = plt.xticks(xticks[1])
    #ax1 = plt.yticks(yticks[1])
    xlim, ylim = (-0.5, 1), (-0.5, 1)
    ax1 = plt.xlabel('Ed from parallel (eV/at)')
    ax1 = plt.ylabel('Ed from serial (eV/at)')
    ax1 = plt.plot(xlim, xlim, color='black', lw=1, ls='--')
    ax1 = plt.xlim(xlim)
    ax1 = plt.ylim(ylim)
    
    ax2 = plt.subplot(122)
    ax2 = plt.scatter(x, y1, 
                     edgecolor='blue',
                     marker='o',
                     color='white')
                     
    #ax1 = plt.xticks(xticks[1])
    #ax1 = plt.yticks(yticks[1])
    xlim, ylim = (-0.1, 1), (-1, 1)
    ax2 = plt.xlabel('Ehull from MP (eV/at)')
    ax2 = plt.ylabel('')
    ax2 = plt.plot(xlim, xlim, color='black', lw=1, ls='--')
    ax2 = plt.gca().yaxis.set_ticklabels([])
    ax2 = plt.xlim(xlim)
    ax2 = plt.ylim(ylim) 
    
    disagreements = []
    for k in serial_hullout:
        if CompTools(k).n_els == 1:
            continue
        if serial_hullout[k]['stability'] and (gs[k]['Ehull_mp'] > 0):
            disagreements.append(k)
        if not serial_hullout[k]['stability'] and (gs[k]['Ehull_mp'] == 0):
            disagreements.append(k)  
            
        if (gs[k]['Ehull_mp'] != 0) and (np.round(serial_hullout[k]['Ed'], 3) != np.round(gs[k]['Ehull_mp'], 3)):
            disagreements.append(k)
            
    for k in disagreements:
        print('\n%s' % k)
        print('my rxn = %s' % serial_hullout[k]['rxn'])
        print('my hull = %.3f' % serial_hullout[k]['Ed'])
        print('mp hull = %.3f' % gs[k]['Ehull_mp'])
    
    #plt.show()
    
    fig.savefig(os.path.join(FIG_DIR, 'pd_demo_check.png'))



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
    
    # if True, test chemical potential stuff
    
    remake_mus_figure_check = True
    
    # MP query for CHEMSYS
    gs = get_mp_data_for_chemsys(CHEMSYS, remake=remake_query)
    
    # hull input data for CHEMSYS
    hullin = serial_get_hull_input_data(gs, remake=remake_serial_hullin)
    
    # hull output data for CHEMSYS
    hullout = serial_get_hull_output_data(hullin, remake=remake_serial_hullout)
    
    # hull output data for CHEMSYS (generated using parallelization)
    p_hullout = parallel_get_hull_input_and_output_data(gs, remake=remake_parallel_hullout)
    
    # generate a graph that compares serial vs parallel hull output and also compares ThermoTools hull output to MP hull data
    if remake_hull_figure_check:
        # %%
        plot_to_check_success(gs, hullout, p_hullout)
        # %%
        
    return gs, hullin, hullout, p_hullout

if __name__ == '__main__':
    
  
    # MP Query --> hull input data (serial) --> hull output data (serial) --> hull output data (parallel)
    gs, hullin, hullout, p_hullout = main()
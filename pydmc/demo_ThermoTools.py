from CompTools import CompTools
from MPQuery import MPQuery
from ThermoTools import GetHullInputData, AnalyzeHull, ParallelHulls
from handy import read_json, write_json
import matplotlib.pyplot as plt
from plotting import set_rc_params, tableau_colors

import os
import numpy as np

API_KEY = '***REMOVED***'

CHEMSYS = 'Ca-Al-Ti-O-F'

DATA_DIR = os.path.join('..', 'examples', 'data')
if not os.path.exists(DATA_DIR):
    os.mkdir(DATA_DIR)
    
FIG_DIR = os.path.join('..', 'examples', 'figures')
if not os.path.exists(FIG_DIR):
    os.mkdir(FIG_DIR)

def get_mp_data_for_chemsys(chemsys, remake=False):
    fjson = os.path.join(DATA_DIR, 'query_'+CHEMSYS+'.json')
    if not remake and os.path.exists(fjson):
        return read_json(fjson)
    
    mpq = MPQuery(API_KEY)
    out = mpq.get_data_for_comp(chemsys, 
                                only_gs=True, 
                                dict_key='cmpd')
    return write_json(out, fjson)

def serial_get_hull_input_data(gs, remake=False):
    fjson = os.path.join(DATA_DIR, 'hullin_serial_'+CHEMSYS+'.json')
    if not remake and os.path.exists(fjson):
        return read_json(fjson)
    
    ghid = GetHullInputData(gs, 'Ef_mp')
    return ghid.hullin_data(fjson=fjson, remake=remake)

def serial_get_hull_output_data(hullin, remake=False):
    fjson = os.path.join(DATA_DIR, 'hullout_serial_'+CHEMSYS+'.json')
    if not remake and os.path.exists(fjson):
        return read_json(fjson)
    
    hullout = {}
    for space in hullin:
        ah = AnalyzeHull(hullin, space)
        for cmpd in hullin[space]:
            print('\n%s' % cmpd)
            hullout[cmpd] = ah.cmpd_hull_output_data(cmpd)            
    return write_json(hullout, fjson)

def parallel_get_hull_input_and_output_data(gs, remake=False):
    fjson = os.path.join(DATA_DIR, 'hullout_parallel_'+CHEMSYS+'.json')
    if not remake and os.path.exists(fjson):
        return read_json(fjson)
    ph = ParallelHulls(gs, n_procs=2, fresh_restart=True)
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
    remake_query = True
    remake_serial_hullin = True
    remake_serial_hullout = True
    remake_parallel_hullout = True
    remake_figure_check = True
    gs = get_mp_data_for_chemsys(CHEMSYS, remake=remake_query)
    hullin = serial_get_hull_input_data(gs, remake=remake_serial_hullin)
    hullout = serial_get_hull_output_data(hullin, remake=remake_serial_hullout)
    p_hullout = parallel_get_hull_input_and_output_data(gs, remake=remake_parallel_hullout)
    if remake_figure_check:
        # %%
        plot_to_check_success(gs, hullout, p_hullout)
        # %%
    return gs, hullin, hullout, p_hullout

if __name__ == '__main__':
    gs, hullin, hullout, p_hullout = main()
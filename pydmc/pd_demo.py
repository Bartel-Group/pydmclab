from CompTools import CompTools
from MPQuery import MPQuery
from ThermoTools import GetHullInputData, AnalyzeHull, ParallelHulls
from handy import read_json, write_json

import os

API_KEY = '***REMOVED***'

CHEMSYS = 'Mg-Al-Ti-O'

DATA_DIR = os.path.join('..', 'examples', 'data')
if not os.path.exists(DATA_DIR):
    os.mkdir(DATA_DIR)

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
    return ghid.hull_data(fjson=fjson, remake=remake)

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

def main():
    remake_query = False
    remake_serial_hullin = False
    remake_serial_hullout = False
    remake_parallel_hullout = True
    gs = get_mp_data_for_chemsys(CHEMSYS, remake=remake_query)
    hullin = serial_get_hull_input_data(gs, remake=remake_serial_hullin)
    hullout = serial_get_hull_output_data(hullin, remake=remake_serial_hullout)
    p_hullout = parallel_get_hull_input_and_output_data(gs, remake=remake_parallel_hullout)
    return gs, hullin, hullout, p_hullout

if __name__ == '__main__':
    gs, hullin, hullout, p_hullout = main()
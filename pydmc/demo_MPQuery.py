from MPQuery import MPQuery
from StrucTools import StrucTools
import os
from handy import read_json, write_json

API_KEY = '***REMOVED***'

DATA_DIR = os.path.join('..', 'examples', 'data')

CHEMSYS = 'O-Al-Mg'

def demo_get_groundstate_hull_data_for_chemsys(chemsys=CHEMSYS,
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
    fjson = os.path.join(data_dir, 'query_gs_all_'+chemsys+'.json')    
    if not remake and os.path.exists(fjson):
            return read_json(fjson)
    
    mpq = MPQuery(api_key)
    out = mpq.get_data_for_comp(comp=chemsys,
                                only_gs=only_gs, 
                                dict_key=dict_key)
    return write_json(out, fjson)

def demo_get_all_data_for_chemsys(chemsys=CHEMSYS,
                                    only_gs=False,
                                    dict_key='mpid',
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
    fjson = os.path.join(data_dir, 'query_demo_all_'+chemsys+'.json')
    if not remake and os.path.exists(fjson):
        return read_json(fjson)
    
    mpq = MPQuery(api_key)
    out = mpq.get_data_for_comp(comp=chemsys,
                                only_gs=only_gs, 
                                dict_key=dict_key)
    return write_json(out, fjson)

def demo_make_sure_gs_finder_works(d_all, d_gs):
    
    unique_compositions = sorted(list(d_gs.keys()))
    
    for cmpd in unique_compositions:
        print('\n working on %s' % cmpd)
        all_IDs = [k for k in d_all if d_all[k]['cmpd'] == cmpd]
        print('found %i MP entries' % len(all_IDs))
        all_energies = [d_all[k]['Ef_mp'] for k in all_IDs]
        min_energy_from_all = min(all_energies)
        E_gs = d_gs[cmpd]['Ef_mp']
        
        print('Ef from all data = %.2f eV/at\nEf from gs data = %.2f eV/at' % (min_energy_from_all, E_gs))
        if min_energy_from_all < E_gs:
            print('\n\n\n\nAHHHH\n\n\n')
            
def demo_entry_retrieval(mpid='mp-530748'):
    mpq = MPQuery(API_KEY)
    properties = list(mpq.typical_properties) + ['band_gap']
    entry = mpq.get_entry_by_material_id(material_id=mpid, 
                                        properties=properties, 
                                        incl_structure=True,
                                        conventional=False,
                                        compatible_only=True)
    print(entry)
    return

def demo_structure_retrieval(mpid='mp-1094961'):
    
    mpq = MPQuery(API_KEY)
    s = mpq.get_structure_by_material_id(material_id=mpid)
    
    print('\nThe formula for this structure is %s' % StrucTools(s).formula)
    print('The first site in the structure is %s' % s[0])
    
def demo_vaspinput_retrieval(mpid='mp-1938'):
    
    inputs = MPQuery(API_KEY).get_vasp_inputs(material_id=mpid)
    print('\n The %s for this calculation was %s' % ('ALGO', inputs['incar']['ALGO']))
    return inputs
    

def main():
    d_all = demo_get_all_data_for_chemsys()
    d_gs = demo_get_groundstate_hull_data_for_chemsys()
    demo_make_sure_gs_finder_works(d_all, d_gs)
    demo_entry_retrieval()
    demo_structure_retrieval()
    demo_vaspinput_retrieval()
    return d_all, d_gs

if __name__ == '__main__':
    d, gs = main()
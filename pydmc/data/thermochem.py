import numpy as np
import os, json

from pydmc.core.comp import CompTools
from pydmc.core.query import MPQuery
from pydmc.utils.handy import write_json

this_dir, this_filename = os.path.split(__file__)
DATA_PATH = os.path.join(this_dir, "data")


def mus_at_0K():
    """
    These were run by Bartel in December 2022
    """
    with open(os.path.join(DATA_PATH, "elemental_reference_energies_0K.json")) as f:
        return json.load(f)

def mus_at_T():
    """
    These come from Bartel 2018 Nat Comm
    """
    with open(os.path.join(DATA_PATH, "elemental_gibbs_energies_T.json")) as f:
        return json.load(f)   

def mp2020_compatibility_dmus():
    """
    from MP2020Compatibility (https://github.com/materialsproject/pymatgen/blob/master/pymatgen/entries/MP2020Compatibility.yaml)
    """
    data = {'U' : {'V': -1.7,
                'Cr': -1.999,
                'Mn': -1.668,
                'Fe': -2.256,
                'Co': -1.638,
                'Ni': -2.541,
                'W': -4.438,
                'Mo': -3.202},
            'anions' : {'O' : -0.687,
                        'S': -0.503,
                        'F': -0.462,
                        'Cl': -0.614,
                        'Br': -0.534,
                        'I': -0.379,
                        'N': -0.361,
                        'Se': -0.472,
                        'Si': 0.071,
                        'Sb': -0.192,
                        'Te': -0.422,
                        'H': -0.179},
            'peroxide' : {'O' : -0.465},
            'superoxide' : {'O' : -0.161}}
    
    return data

def get_mus_from_mp_no_corrections():
    fjson = os.path.join(DATA_PATH, 'mus_from_mp_no_corrections.json')
  
    mus = mus_at_0K()
    
    mp_pbe_mus = mus['mp']['pbe']
    
    mpq = MPQuery(api_key='***REMOVED***')
    
    mp_mus = {}
    for el in mp_pbe_mus:
        print(el)
        my_mu = mp_pbe_mus[el]
        el += '1'
        query = mpq.get_data_for_comp(el, 
                                      only_gs=True,
                                      dict_key='cmpd')
        
        mp_mu = query[el]['E_mp']
        mp_mus[el[:-1]] = mp_mu

    return write_json(mp_mus, fjson)    
    
def mus_from_mp_no_corrections():
    """
    Last collected Dec 2022 (with legacy MP API)
    """
    with open(os.path.join(DATA_PATH, "mus_from_mp_no_corrections.json")) as f:
        return json.load(f)  


def make_ssub_json():
    fjson = os.path.join(DATA_PATH , 'ssub.json')
    
    data = {}
    with open(os.path.join(DATA_PATH, "ssub.dat")) as f:
        for line in f: 
            if 'cmpd' in line:
                continue
            cmpd, H = line[:-1].split(' ')
            cmpd = CompTools(cmpd).clean
            if len(CompTools(cmpd).els) > 1:
                if cmpd not in data:
                    data[cmpd] = H
                else:
                    if H < data[cmpd]:
                        data[cmpd] = H
    return write_json(data, fjson)

def ssub():
    with open(os.path.join(DATA_PATH, "ssub.json")) as f:
        return json.load(f)
            
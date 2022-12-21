import numpy as np
import os, json

this_dir, this_filename = os.path.split(__file__)
DATA_PATH = os.path.join(this_dir, "data")

def mus_at_0K():
    with open(os.path.join(DATA_PATH, "elemental_reference_energies_0K.json")) as f:
        return json.load(f)

def mus_at_T():
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

def mus_from_mp_no_corrections():
    with open(os.path.join(DATA_PATH, "mus_from_mp_no_corrections.json")) as f:
        return json.load(f)  
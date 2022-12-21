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

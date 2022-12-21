import numpy as np
import os, json

this_dir, this_filename = os.path.split(__file__)
DATA_PATH = os.path.join(this_dir, "data")

def atomic_masses():
    with open(os.path.join(DATA_PATH, "atomic_masses.json")) as f:
        return json.load(f)

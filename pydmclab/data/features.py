import numpy as np
import os, json

this_dir, this_filename = os.path.split(__file__)
DATA_PATH = os.path.join(this_dir, "data")


def atomic_masses():
    with open(os.path.join(DATA_PATH, "atomic_masses.json")) as f:
        return json.load(f)


def atomic_electronegativities():
    with open(os.path.join(DATA_PATH, "atomic_electronegativities.json")) as f:
        return json.load(f)


def ionic_radii():
    with open(
        os.path.join(DATA_PATH, "shannon_revised_effective_ionic_radii.json")
    ) as f:
        return json.load(f)

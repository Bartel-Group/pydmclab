import numpy as np
import os, json

from pydmc.core.comp import CompTools
from pydmc.utils.handy import write_json

this_dir, this_filename = os.path.split(__file__)
DATA_PATH = os.path.join(this_dir, "data")

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
            
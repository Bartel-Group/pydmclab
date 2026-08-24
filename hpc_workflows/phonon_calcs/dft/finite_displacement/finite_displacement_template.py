import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from pydmclab.utils.handy import read_json, write_json, convert_numpy_to_native
from pydmclab.hpc.phonons import AnalyzePhonons
from scipy.constants import physical_constants
import pandas as pd
# from sumo.plotting.phonon_bs_plotter import SPhononBSPlotter

# from pydmclab.core.struc import StrucTools
# from pymatgen.core import Structure
# from pydmclab.hpc.helpers import get_query
# from pymatgen.io.ase import AseAtomsAdaptor

# set up some paths that will point to where your data/calculations will live
#  these are just defaults, you can change the `_DIR` variables to point to wherever you want
#
# The home directory path is used to point to your local copy of the pydmclab repo
#   pydmclab is assumed to be in /users/{number}/{username}/bin/pydmclab
#   and $HOME points to /users/{number}/{username}
HOME_PATH = os.environ["HOME"]

#importing phonon helpers
PHONON_HELPERS_DIR = "%s/bin/pydmclab/hpc_workflows/phonon_calcs" % HOME_PATH

if PHONON_HELPERS_DIR not in sys.path:
    sys.path.append(PHONON_HELPERS_DIR)

from phonon_helpers import (
    get_set_of_forces,
)

# where is this file
SCRIPTS_DIR = os.getcwd()

# where is my data going to live
DATA_DIR = SCRIPTS_DIR.replace("scripts", "data")

EV_TO_J = physical_constants['electron volt-joule relationship'][0]
AVOGADRO = physical_constants['Avogadro constant'][0]
EV_TO_J_PER_MOL = EV_TO_J * AVOGADRO
EV_TO_KJ_PER_MOL = EV_TO_J_PER_MOL / 1000.0

def compute_all_phonon_properties(results,
                                  displacements,
                                  xc_wanted = "metagga",
                                  temperatures=np.linspace(0, 2000, 101),
                                  init_kwargs={},
                                  band_structure_kwargs=None,
                                  query=None,
                                  savename='phonons.json',
                                  data_dir=DATA_DIR,
                                  remake=False,):

    """
    Compute all phonon properties from VASP results and displacements dictionary.
    Args:
        results (dict): 
            Results dictionary from DFT calcs on displaced structures. Usually generated from get_results() in pydmclab.hpc.helpers
        displacements (dict): 
            Displacements dictionary. Usually generated with get_displacements_for_phonons() in phonon_helpers in hpc_workflows --> phonon_calcs
                {
                    "unitcell": The original supercell structure pre-displacements (as dict),
                    "displaced_structures": The list of displaced structures (as dict),
                    "dataset": Only for finite displacement. The dataset containing displacement information obtained from phonopy.
                }
        xc_wanted (str): 
            Exchange-correlation functional to retrieve information from. This will grab {xc}-static data from results dictionary.
        init_kwargs (dict): 
            Initialization arguments for AnalyzePhonons. See pydmclab.hpc.phonons.AnalyzePhonons for more details.
        band_structure_kwargs (dict): 
            Arguments for band structure calculation. See pydmclab.hpc.phonons.AnalyzePhonons.band_structure() for more details.
        query (dict): 
            Query dictionary used for DFT calculations (usually from your get_query() function). This is to retrieve data from the static calculations (pre-displacements)
            If None is given, the returned phonon results will be phonon contribution to the energy only (without E0)
            Note: In QHA calculations need energy of original cell + phonon information.
            This dictionary should have the same mpids as the results dictionary but without the displacement suffixes.
            e.g. SrZrS3_needle, SrZrS3_perovskite for query keys and SrZrS3_needle_01, SrZrS3_perovskite_01 for mpid in results dictionary keys.
        savename (str): 
            Name of the output JSON file.
        data_dir (str): 
            Directory to save the output JSON file to.
        remake (bool): 
            Whether to remake the phonon properties.

    Returns:
        dict: A dictionary containing the computed phonon properties. e.g.:
        {
        'SrZrS3--SrZrS3_needle--nm--metagga-finite_displacement': {
            'phonons': {
                'frequencies': [...],
                'total_dos': [...],
                ...
            }
        },
        'SrZrS3--SrZrS3_needle--nm--metagga-static': {
            'results': {'E_per_at': ...,},
            'structure': ...,
        }

    """

    fjson = os.path.join(data_dir, savename)
    if os.path.exists(fjson) and not remake:
        return read_json(fjson)

    sets_of_forces = get_set_of_forces(results, mpid=None, xc=xc_wanted)

    out = {}

    filtered_forces = {}
    for mpid, data in sets_of_forces.items():
        forces = data.get('forces')
        if forces is None or (hasattr(forces, "__len__") and len(forces) == 0):
            print(f"Forces for mpid {mpid} not converged, skipping for now. Remake phonons once converged.")
            continue
        filtered_forces[mpid] = data

    sets_of_forces = filtered_forces

    for mpid in sets_of_forces:
        forces = sets_of_forces[mpid]['forces']
        static_key = sets_of_forces[mpid]['key']

        print(f"Forces for {mpid} found with shape {np.array(forces).shape}")
        supercell = displacements[mpid]['supercell']
        disp_strucs = displacements[mpid]['displaced_structures']
        calc_method = displacements[mpid]['calc_method']
        dataset = displacements[mpid]['dataset']

        phonon_key = static_key.replace("static", calc_method)

        E_per_at = 0
        if query:
            E_per_at = query[mpid]['E_per_at'] #assuming *base* mpids match between query and results.json
            struc = query[mpid]['structure']

            out[static_key] = {'results': 
                               {'E_per_at': E_per_at}, 
                               'structure': struc
                               }

        analyzer = AnalyzePhonons(
            unitcell=supercell,
            force_data=forces,
            dataset=dataset,
            E0=E_per_at,
            **init_kwargs
        )

        summary = analyzer.summary(temperatures=temperatures,
                                   band_structure_kwargs=band_structure_kwargs)

        out[phonon_key] = {'phonons': summary}
        out[phonon_key]['forces'] = forces

    out = convert_numpy_to_native(out)
    write_json(out, fjson)
    return read_json(fjson)


def main():
    remake_phonons = False

    results = read_json(os.path.join(DATA_DIR, "results.json"))
    displacements = read_json(os.path.join(DATA_DIR, "displacements.json"))
    query = read_json(os.path.join(DATA_DIR, "query.json"))

    xc_wanted = "metagga"

    temperatures = np.linspace(0, 2000, 101)
    init_kwargs = {}
    band_structure_kwargs = None

    phonons = compute_all_phonon_properties(results=results,
                                            displacements=displacements,
                                            xc_wanted=xc_wanted,
                                            temperatures=temperatures,
                                            init_kwargs=init_kwargs,
                                            band_structure_kwargs=band_structure_kwargs,
                                            query=query,
                                            savename='phonons_test.json',
                                            data_dir=DATA_DIR,
                                            remake=remake_phonons,)

    #See pydmclab.plotting.phonons for plotting functions

    return phonons

if __name__ == "__main__":
    main()
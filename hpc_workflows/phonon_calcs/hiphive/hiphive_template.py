import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from pydmclab.utils.handy import read_json, write_json
from pydmclab.hpc.phonons import AnalyzePhonons

# from pydmclab.core.struc import StrucTools
# from pymatgen.core import Structure
# from pydmclab.hpc.helpers import get_query
# from pymatgen.io.ase import AseAtomsAdaptor

SCRIPT_DIR = os.getcwd()
DATA_DIR = SCRIPT_DIR.replace('scripts', 'data')

HOME_PATH = os.environ["HOME"]
PHONON_HELPERS_DIR = "%s/bin/pydmclab/hpc_workflows/phonon_calcs" % HOME_PATH

if PHONON_HELPERS_DIR not in sys.path:
    sys.path.append(PHONON_HELPERS_DIR)

from phonon_helpers import (
    get_set_of_forces,
    get_fcp_hiphive,
    get_force_constants_hiphive
)


def compute_all_phonon_properties(results,
                                  displacements,
                                  xc_wanted="metagga",
                                  cutoffs=[3.5, 3.0],
                                  init_kwargs={},
                                  thermal_properties_kwargs=None,
                                  band_structure_kwargs=None,
                                  query=None,
                                  savename='phonons.json',
                                  data_dir=DATA_DIR,
                                  remake=False,
                                  remake_fcp=False,
                                  plot_band_structure=True,
                                  plot_thermal_properties=True):

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
            Exchange-correlation functional to retrieve information from. This will grab xc-static data from results dictionary.
        cutoffs (list): 
            List of cutoff distances for neighbor searching in the force constant calculations. See hiphive documentation for more details.
        init_kwargs (dict): 
            Initialization arguments for AnalyzePhonons. See pydmclab.hpc.phonons.AnalyzePhonons for more details.
        thermal_properties_kwargs (dict): 
            Arguments for thermal properties calculation. See pydmclab.hpc.phonons.AnalyzePhonons.thermal_properties() for more details.
        band_structure_kwargs (dict): 
            Arguments for band structure calculation. See pydmclab.hpc.phonons.AnalyzePhonons.band_structure() for more details.
        query (dict): 
            Query dictionary used for DFT calculations (usually from your get_query() function). 
            If None is given, information will not be retrieved for static calculations.
            This is to retrieve data from the static calculations (pre-displacements) that might be necessary in the case of running a QHA calculation.
            In QHA calculations need energy of original cell + phonon information.
            This dictionary should have the same mpids as the results dictionary but without the displacement suffixes.
            e.g. SrZrS3_needle, SrZrS3_perovskite for query keys and SrZrS3_needle_01, SrZrS3_perovskite_01 for mpid in results dictionary keys.
        savename (str): 
            Name of the output JSON file.
        data_dir (str): 
            Directory to save the output JSON file to.
        remake (bool): 
            Whether to remake the phonon properties.
        plot_band_structure (bool): 
            Whether to plot the band structure.
        plot_thermal_properties (bool): 
            Whether to plot the thermal properties.

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

    out = {}

    sets_of_forces = get_set_of_forces(results, mpid=None, xc=xc_wanted)

    for mpid in sets_of_forces:
        forces = sets_of_forces[mpid]['forces']
        static_key = sets_of_forces[mpid]['key']

        print(f"Forces for {mpid} found with shape {np.array(forces).shape}")
        supercell = displacements[mpid]['unitcell']
        disp_strucs = displacements[mpid]['displaced_structures']
        calc_method = displacements[mpid]['calc_method']

        phonon_key = static_key.replace("static", calc_method)

        fcp = get_fcp_hiphive(ideal_supercell=supercell,
                            rattled_structures=disp_strucs,
                            force_sets=forces,
                            cutoffs=cutoffs,
                            data_dir=data_dir,
                            savename=f"fcp_{mpid}.fcp",
                            remake=remake_fcp)

        force_constants = get_force_constants_hiphive(fcp, supercell)

        analyzer = AnalyzePhonons(
            unitcell=supercell,
            force_data=force_constants,
            **init_kwargs
        )

        summary = analyzer.summary(thermal_properties_kwargs=thermal_properties_kwargs,
                                    band_structure_kwargs=band_structure_kwargs)

        out[phonon_key] = {'phonons': summary}

        if query:
            E_per_at = query[mpid]['E_per_at']
            struc = query[mpid]['structure']

            out[static_key] = {'results': 
                               {'E_per_at': E_per_at}, 
                               'structure': struc}

        if plot_band_structure:
            analyzer.plot_band_structure

        if plot_thermal_properties:
            analyzer.plot_thermal_properties

    write_json(out, fjson)
    return read_json(fjson)



def main():
    remake_phonons = False
    remake_fcp = False

    results = read_json(os.path.join(DATA_DIR, "results.json"))
    displacements = read_json(os.path.join(DATA_DIR, "displacements.json"))
    query = read_json(os.path.join(DATA_DIR, "query.json"))

    xc_wanted = "metagga"
    cutoffs = [3.5, 3.0]

    init_kwargs = {}
    thermal_properties_kwargs = None
    band_structure_kwargs = None

    plot_band_structure = True
    plot_thermal_properties = True

    compute_all_phonon_properties(results=results,
                                  displacements=displacements,
                                  xc_wanted=xc_wanted,
                                  cutoffs=cutoffs,
                                  init_kwargs=init_kwargs,
                                  thermal_properties_kwargs=thermal_properties_kwargs,
                                  band_structure_kwargs=band_structure_kwargs,
                                  query=query,
                                  savename='phonons.json',
                                  data_dir=DATA_DIR,
                                  remake=remake_phonons,
                                  remake_fcp=remake_fcp,
                                  plot_band_structure=plot_band_structure,
                                  plot_thermal_properties=plot_thermal_properties)

if __name__ == "__main__":
    main()
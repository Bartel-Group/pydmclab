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
                                  savename='phonons.json',
                                  data_dir=DATA_DIR,
                                  remake=remake_phonons,
                                  remake_fcp=remake_fcp,
                                  plot_band_structure=plot_band_structure,
                                  plot_thermal_properties=plot_thermal_properties)

if __name__ == "__main__":
    main()
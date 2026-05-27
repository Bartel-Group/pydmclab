import os
import numpy as np

from pydmclab.utils.handy import read_json, write_json, convert_numpy_to_native
from pydmclab.core.struc import StrucTools
from pydmclab.core.comp import CompTools
from pydmclab.hpc.phonons import AnalyzePhonons

from pymatgen.io.ase import AseAtomsAdaptor
from pymatgen.io.phonopy import get_phonopy_structure, get_pmg_structure
from pymatgen.analysis.local_env import CrystalNN

from hiphive.structure_generation import generate_mc_rattled_structures, generate_rattled_structures

from phonopy import Phonopy

def get_displacements_for_phonons(
                    unitcell: str|dict,
                    method: str,
                    data_dir: str|None,
                    savename: str|None = 'displacements.json',
                    remake: bool|None = False,
                    supercell_matrix: list|None = None,
                    distance: float|str = 'auto',
                    mc: bool = True,
                    n_structures: int|None = None,
                    rattle_std: float|None = None,
                    minimum_distance: float|None = None,
                    ):
    """    Get the displacements for a given unitcell and method.
    Args:
        unitcell (str or dict):
            Path to the unitcell structure file (e.g., POSCAR) or a dictionary containing the structure data.
            If a dictionary is provided, it should contain 'lattice', 'species', and 'coords' keys.
        method (str):
            Method to use for displacements. Options are 'finite_displacement' or 'hiphive'.
            REMINDER: finite_displacement creates many unitcells with one displacement each, while hiphive creates many unitcells with multiple random displacements.
        data_dir (str or None):
            Path to directory where displacement data will be saved. If None, data will not be saved to disk.
        savename (str or None):
            Name of the file to save displacement data.
        remake (bool or None):
            If True, will remake the displacement data even if it exists.
        supercell_matrix (list or None):
            Supercell matrix to use for generating supercells. Highly recommend not using as to not cause confusion. Feed a structure that has already been supercelled.
        distance (float or None):
            Distance for finite displacement only. If auto will calculate as 1% of minimum interatomic distance in structure.
        mc (bool):
            If True, will use Monte Carlo method for generating displacements. For hiphive only.
        rattle_std (float or None):
            Standard deviation for random rattling displacements.
        minimum_distance (float or None):
            Minimum distance for hiphive displacement generation only if doing Monte Carlo. See hiphive.structure_generation.rattle.generate_mc_rattled_structures() for more details.

    Returns:
        displacements_data (dict):
            {
                "unitcell": The original supercell structure pre-displacements (as dict),
                "displaced_structures": The list of displaced structures (as dict),
                "dataset": Only for finite displacement. The dataset containing displacement information obtained from phonopy,
                            this is needed to feed to AnalyzePhonons if want to obtain thermal properties from finite displacement, 
                            could optionally contain forces if calculating with mlp, but this would be in a separate function.
                "calc_method": method used for calculating displacements: finite_displacement or hiphive
            }

    When creating MPIDs for the displaced structures (this would be once you are creating your get_strucs() or something), the original MPID should be used as a base, with an index appended for each displacement, always set at the end. 
    For example, if the base MPID is 'S3Sr1Zr1_needle', the displaced structures could be named 'S3Sr1Zr1_needle_01', 'S3Sr1Zr1_needle_02', etc.
    Or for QHA, where there is also a suffix for the scaling of the different volumes: 'S3Sr1Zr1_needle_1.2_01', 'S3Sr1Zr1_needle_1.2_02', etc.
    Then the helper get_set_of_forces() can be used to extract the forces within each mpid using the "raw" mpid as a key by checking against mpid minus the last underscore and everything after it.
    """
    if data_dir is not None:
        fjson = os.path.join(data_dir, savename)
        if os.path.exists(fjson) and not remake:
            return read_json(fjson)

    st = StrucTools(unitcell)
    pymatgen_struc = st.structure

    out = {}
    out['unitcell'] = st.structure_as_dict
    out['calc_method'] = method

    if method == "finite_displacement":

        unitcell = get_phonopy_structure(pymatgen_struc)
        phonon = Phonopy(unitcell=unitcell, supercell_matrix=supercell_matrix)

        if distance == 'auto':
            distance = estimate_rattle_std(st.structure_as_dict(),
                                           fraction= 0.01)

        displacement_data = phonon.generate_displacements(distance=distance)
        supercells_with_displacements = phonon.supercells_with_displacements #returns a list of PhonopyAtoms supercells
        pmg_displaced_strucs = [get_pmg_structure(struc) for struc in supercells_with_displacements]

        dataset = phonon.dataset
        out["dataset"] = dataset

    if method == "hiphive":
        #turn unitcell to AseAtoms
        atoms = AseAtomsAdaptor.get_atoms(pymatgen_struc)
        if mc:
            print("Generating displacements using Monte Carlo rattling method.")
            structures = generate_mc_rattled_structures(atoms, n_structures, rattle_std, minimum_distance)
        else:
            print("Generating displacements using simple random rattling method.")
            structures = generate_rattled_structures(atoms, n_structures, rattle_std)

        pmg_displaced_strucs = [AseAtomsAdaptor.get_structure(struc) for struc in structures]

    pmg_displaced_strucs = [struc.as_dict() for struc in pmg_displaced_strucs]
    out["displaced_structures"] = pmg_displaced_strucs

    out = convert_numpy_to_native(out)  # Make sure the output is JSON serializable

    if data_dir is not None:
        write_json(out, fjson)
        return read_json(fjson)
    else:
        return out

def estimate_rattle_std(structure: str|dict, 
                        fraction: float, 
                        include_min_dist_for_mc = False, 
                        min_dist_factor = 3.0) -> float:
    """
    Estimate the rattle standard deviation based on a fraction of the minimum interatomic distance in the structure.
    Note: at the moment just auto detects oxidation states and assigns formal charges. This could be improved in the future.
    """
    nn = CrystalNN()
    st = StrucTools(structure)
    struc = st.decorate_with_ox_states
    
    nn_info = nn.get_all_nn_info(struc)
    
    min_dist = float("inf")
    for i, neighbors in enumerate(nn_info):  # Fixed enumerate usage
        site1 = struc.sites[i]
        for neighbor in neighbors:
            site2 = neighbor['site']
            dist = site1.distance(site2)
            if dist < min_dist:
                min_dist = dist

    rattle_std = min_dist * fraction
    if include_min_dist_for_mc:
        min_dist = min_dist - min_dist_factor * rattle_std
        return rattle_std, min_dist

    return rattle_std

def get_set_of_forces(results,
                      mpid=None,
                      xc: str = "metagga"):
    '''
    Get the set of calculated forces (from DFT) from multiple structures with displacements for a specific MPID and return as a list of arrays.
    This is for the finite displacement method, where forces will be stored in the results.json under 'results' for a DFT calculation.
    Args:
        results (dict):
            Dictionary containing results from multiple calculations, usually generated with get_results().
            Keys will have mpid with displacement suffixes, e.g., 'SrZrS3--SrZrS3_needle_01--etc' or 'SrZrS3--SrZrS3_needle_1.2_01--etc' if running QHA.
        mpid (str or None):
            The base MPID of the structure for which to extract forces (without displacement suffix). E.g., 'S3Sr1Zr1_needle' or 'S3Sr1Zr1_needle_1.2' if running QHA and have a suffix for the volume scale.
            If None, will create sets of forces for all mpids and save to a dictionary.
        xc (str):
            The exchange-correlation functional used in the calculations, e.g., 'gga', 'metagga'.
    Returns:
        list or dict:
            If mpid is specified: A list of arrays (or None for missing forces) containing the forces for each structure with displacements.
            If mpid is None: A dictionary where keys follow the results.json format but use base mpids (without displacement suffixes), 
                             and each key['forces'] leads to a set of forces for all the displacements of the corresponding mpid.
                             e.g. {SrZrS3--SrZrS3_needle--etc : {'forces': [list of arrays]}}
    REMINDER: When you generate the displacements, you do STATIC calculations on those displaced structures to get the forces (no relaxation).
    '''
    # We'll collect (index, forces, representative_key) entries so we can sort by index
    if mpid is None:
        raw_sets = {}
    else:
        raw_list = []

    for key in results:
        calc_type = key.split("--")[-1]
        if calc_type != f"{xc}-static":
            continue

        r_mpid = key.split("--")[1]
        parts = r_mpid.split("_")

        mpid_minus_disp = "_".join(parts[:-1])
        index_str = parts[-1]

        # Expect the MPID to have a displacement index appended after an underscore.
        # If this is not the case, raise an error so the caller can fix the MPID naming.
        if len(parts) < 2:
            raise ValueError(f"Expected displaced MPID with an underscore and index (e.g. 'base_01'), got '{r_mpid}' from key '{key}'")

        if mpid is not None and mpid_minus_disp != mpid:
            continue

        # Extract forces
        forces = results[key].get("forces")
        if not forces:
            print(f"Warning: No forces found for {key}. Adding None to maintain indexing.")
            forces = None

        if forces is not None:
            arr = np.array(forces)
            print(f"Including forces for {key} with shape {arr.shape}")
 
        index_key = int(index_str)

        if mpid is None:
            new_key = key.replace(r_mpid, mpid_minus_disp)
            if mpid_minus_disp not in raw_sets:
                raw_sets[mpid_minus_disp] = {'raw': [], 'key': new_key}
            raw_sets[mpid_minus_disp]['raw'].append((index_key, forces))
        else:
            raw_list.append((index_key, forces))

    # Post-process collected raw entries into ordered lists by index
    if mpid is None:
        set_of_forces = {}
        for base, info in raw_sets.items():
            ordered = sorted(info['raw'], key=lambda x: x[0])
            forces_ordered = [f for (_, f) in ordered]
            set_of_forces[base] = {'forces': forces_ordered, 'key': info.get('key')}
    else:
        if not raw_list:
            print(f"No forces found for mpid: {mpid}")
            return None       
        ordered = sorted(raw_list, key=lambda x: x[0])
        set_of_forces = [f for (_, f) in ordered]
    
    return set_of_forces

def get_force_constants_dfpt(calc_dir: str, savename: str = "force_constants.json", remake: bool = False):
    '''
    workflow for getting force constants from a dfpt calculation
    Something is already implementd in pydmclab.hpc.analyze to automatically extract if analyze_phonons_dfpt=True
    '''
    fjson = os.path.join(calc_dir, savename)
    if os.path.exists(fjson) and not remake:
        return read_json(fjson)
    
    force_constants_path = os.path.join(calc_dir, "vasprun.xml")
    if not os.path.exists(force_constants_path):
        print(f"Warning: vasprun.xml file not found in {calc_dir}. Returning None.")
        return None
    
    force_constants_dict = parse_force_constants(force_constants_path)
    if not force_constants_dict:
        print("Warning: Failed to parse force constants. Returning None.")
        return None
    
    force_constants = force_constants_dict[0]
    atoms = force_constants_dict[1]
    out = {"force_constants": force_constants, "calc_method": "dfpt", "atoms": atoms}

    out = convert_numpy_to_native(out)  # Make sure the output is JSON serializable
    write_json(out, fjson)

    return read_json(fjson)

def parse_qha_results(qha_results: dict, include_structures: bool = True):
    """
    Parse the results from a results.json file for a QHA calculation.
    Args:
        qha_results (dict): Usually generated with pydmclab phonon template which grabs QHA raw results, passes them through AnalyzePhonons and produces a results.json file.
        The keys in qha_results should follow the same format as pydmclab.hpc.helper.get_results() 
        but the mpid should be base mpid without displacement suffixes, but with the volume scale suffix. i.e. phonon results for each volume point.
        mpid format should be 'base_mpid_volume-scale'

    """

    dos_dict = {}

    for key in qha_results:
        calc_method = key.split("--")[-1].split("-")[-1]
        if calc_method not in ["dfpt", "finite_displacement", "hiphive"]:
            continue

        formula, mpid = key.split("--")[0], key.split("--")[1]
        xc = key.split("--")[-1].split("-")[0]
        scale = mpid.split("_")[-1]
        mpid_minus_scale = "_".join(mpid.split("_")[:-1])

        phonon_data = qha_results[key]['phonons']
        if phonon_data is None:
            print(f"Warning: Phonon data not found for key {key}. Skipping.")
            continue

        static_key = "--".join(key.split("--")[:-1] + [f"{xc}-static"])
        if static_key not in qha_results:
            print(f"Warning: Static key {static_key} not found in results. Skipping.")
            continue
        
        structure = qha_results[static_key]['structure']
        E_per_at = qha_results[static_key]['results']['E_per_at']
        n_atoms = len(structure['sites'])
        E_electronic = n_atoms * E_per_at

        volume = structure['lattice']['volume']

        dos_data = qha_results[key]['phonons']['total_dos']
        
        thermal_props = phonon_data['helmholtz']

        if formula not in dos_dict:
            dos_dict[formula] = {}

        if mpid_minus_scale not in dos_dict[formula]:
            dos_dict[formula][mpid_minus_scale] = {}

        if volume not in dos_dict[formula][mpid_minus_scale]:
            dos_dict[formula][mpid_minus_scale][volume] = {}

        dos_dict[formula][mpid_minus_scale][volume] = {
                            'E0': E_electronic,
                            'total_dos': [dos_data],
                            'helmholtz': [thermal_props]
                        }
        if include_structures:
            dos_dict[formula][mpid_minus_scale][volume]['structure'] = structure

    return dos_dict
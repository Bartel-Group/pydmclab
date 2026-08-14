import os
import numpy as np

from pydmclab.utils.handy import read_json, write_json, convert_numpy_to_native
from pydmclab.core.struc import StrucTools
from pydmclab.core.comp import CompTools
from pydmclab.hpc.phonons import AnalyzePhonons

from pymatgen.io.ase import AseAtomsAdaptor
from pymatgen.io.phonopy import get_phonopy_structure, get_pmg_structure
from pymatgen.analysis.local_env import CrystalNN

from phonopy import Phonopy

def get_finite_displacement_strucs(query: dict, 
                                   data_dir: str,
                                   distance: str|int|float='auto',
                                   supercell_matrix: list|dict|None = None, 
                                   savename: str = "strucs.json", 
                                   savename_displacements: str = "displacements.json", 
                                   remake: bool=False):
    '''
    Args:
    distance (float or None):
        Distance for finite displacement. If auto will calculate as 1% of minimum interatomic distance in structure.
    supercell_matrix (list, dict or None):
        Supercell matrix to use when generating supercells. Usually the primitive cell is relaxed tightly before building the supercell.
        If list: apply the same supercell matrix to all structures, e.g. [2, 2, 2].
        If dict: apply a supercell matrix per mpid, e.g. {
            'S3Sr1Zr1_needle': [3, 2, 1],
            'S3Sr1Zr1_perovskite': [2, 2, 2]
        }.
        Important because you probably want each lattice parameter to be at least 10 - 15 A to avoid displacement image interactions.
    data_dir (str):
        Path to directory where displacement data will be saved. If None, data will not be saved.
    savename (str or None):
        Name of the file to save strucs to.
    savename_displacements ('str'):
        Name of the file to save displacement data. Will need this for post process calculation of phonon properties.
    remake (bool or None):
        If True, will remake the displacement data even if it exists.
    Returns:
        {formula_indicator (str) :
            {struc_indicator (str) with displacement index as suffix:
                Pymatgen Structure object as dict}}
        e.g.: 
        {S3Sr1Zr1:
            {S3Sr1Zr1_needle_01:
                Pymatgen Structure object as dict,
            S3Sr1Zr1_needle_02:
                Pymatgen Structure object as dict,
                },...}
    
    '''
    
    fjson = os.path.join(data_dir, savename)
    fjson_displacements = os.path.join(data_dir, savename_displacements)
    if os.path.exists(fjson) and os.path.exists(fjson_displacements) and not remake:
        return read_json(fjson)

    strucs = {}
    displacements_data = {}
    for mpid in query:
        struc = query[mpid]['structure']
        st = StrucTools(struc)

        formula = st.compact_formula

        if formula not in strucs:
            strucs[formula] = {}

        supercell = supercell_matrix
        if isinstance(supercell_matrix, dict):
            supercell = supercell_matrix[mpid]

        data = get_displacements_for_phonons(unitcell = struc,
                                             method = "finite_displacement",
                                             distance=distance,
                                             supercell_matrix=supercell,
                                             data_dir = None)

        displaced_supercells = data['displaced_structures']
        for i,disp in enumerate(displaced_supercells):
            new_mpid_i = f'{mpid}_{i}'
            strucs[formula][new_mpid_i] = disp

        displacements_data[mpid] = data

    write_json(displacements_data, fjson_displacements)
    write_json(strucs, fjson)
    return read_json(fjson)

def get_displacements_for_phonons(
                    unitcell: str|dict,
                    method: str,
                    data_dir: str|None,
                    savename: str|None = 'displacements.json',
                    remake: bool|None = False,
                    distance: float|str = 'auto',
                    supercell_matrix: list|None = None,
                    mc: bool = True,
                    n_structures: int|None = None,
                    rattle_std: float|None = None,
                    minimum_distance: float|None = None,
                    ):
    """    
    Get the displacements data and structures for a given unitcell and method.
    Args:
        unitcell (str or dict):
            Path to the unitcell structure file (e.g., POSCAR) or a dictionary containing the structure data.
        method (str):
            Method to use for displacements. Options are 'finite_displacement' or 'hiphive'.
            REMINDER: finite_displacement creates many unitcells with one displacement each, while hiphive creates many unitcells with multiple random displacements.
        data_dir (str or None):
            Path to directory where displacement data will be saved. If None, data will not be saved.
        savename (str or None):
            Name of the file to save displacement data.
        remake (bool or None):
            If True, will remake the displacement data even if it exists.
        distance (float or None):
            Distance for finite displacement only. If auto will calculate as 1% of minimum interatomic distance in structure.
        supercell_matrix (list or None):
            Supercell matrix to use if want to generate supercells. Usually don't use this when doing DFT as structure used to create displacements is usually already supercelled and relaxed.
        mc (bool):
            If True, will use Monte Carlo method for generating displacements. For hiphive only.
        rattle_std (float or None):
            Standard deviation for random rattling displacements. For hiphive only.
        minimum_distance (float or None):
            Minimum distance for hiphive displacement generation only if doing Monte Carlo. See hiphive.structure_generation.rattle.generate_mc_rattled_structures() for more details.

    Returns:
        displacements_data (dict):
            {
                "unitcell": The original supercell structure pre-displacements (as dict),
                "displaced_structures": The list of displaced structures (as dict),
                "dataset": Only for finite displacement. The dataset containing displacement information obtained from phonopy,
                            this is needed to feed to AnalyzePhonons if want to obtain thermal properties from finite displacement.
                "calc_method": method used for calculating displacements: finite_displacement or hiphive
            }

    When creating MPIDs for the displaced structures (this would be once you are creating your get_strucs() or something), 
    the original MPID should be used as a base, with an index appended for each displacement always set at the end (see get_finite_displacement_strucs() helper). 
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
            distance = estimate_displacement_distance(st.structure_as_dict,
                                           fraction= 0.01)

        displacement_data = phonon.generate_displacements(distance=distance)
        supercells_with_displacements = phonon.supercells_with_displacements #returns a list of PhonopyAtoms supercells
        pmg_displaced_strucs = [get_pmg_structure(struc) for struc in supercells_with_displacements]

        dataset = phonon.dataset
        out["dataset"] = dataset

    if method == "hiphive":
        from hiphive.structure_generation import generate_mc_rattled_structures, generate_rattled_structures
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

def estimate_displacement_distance(structure: str|dict, 
                                    fraction: float, 
                                    include_min_dist_for_mc = False, 
                                    min_dist_factor = 3.0) -> float:
    """
    Estimate the finite displacement distance or hiphive rattle standard deviation based on a fraction of the minimum interatomic distance in the structure.
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

def get_qha_strucs(query: dict,
                   scale=np.linspace(0.99, 1.01, 5),
                   data_dir: str = os.getcwd().replace("scripts", "data"),
                   savename="strucs.json",
                   remake=False):
    """
    Scales the structures' volumes for Quasi-Harmonic Approximation (QHA) and returns a strained structures dictionary.
    Can write the strained structures to a json file if needed.

    Args:
        query (dict)
            {ID (str) : {'structure' : Pymatgen Structure as dict,
                        '<other property>' : whatever you queried for}}
        scale (list): 
            List of scale factors to apply to the structure volume. For QHA, you need at least 5 volume points.
        data_dir (str): 
            Directory to save the JSON file.
        savename (str): 
            filename for fjson in DATA_DIR  
        remake (bool): 
            write (True) or just read (False) fjson  

    Returns:
        {formula_indicator (str) :
            {struc_indicator (str) with scale factor as suffix:
                Pymatgen Structure object as dict}}
        e.g., if you got some MP data, this might return something like:
            {'Cl3Cs1Pb1' : {'mp-1234_1.02' : Structure.as_dict}, {'mp-1234_1.04' : Structure.as_dict}, ...} 
    """

    fjson = os.path.join(data_dir, savename) if data_dir else None
    if fjson and os.path.exists(fjson) and not remake:
        return read_json(fjson)

    QHA_strucs = {}
  
    def scale_and_update(mpid, s):
        st = StrucTools(s)
        formula = st.compact_formula
        
        for i in scale:
            scaled_st = st.scale_structure(i)
            new_mpid = f"{mpid}_{np.round(i, 3)}"
            QHA_strucs.setdefault(formula, {})[new_mpid] = scaled_st.as_dict()

    for mpid in query:
        scale_and_update(mpid, query[mpid]['structure'])
    
    write_json(QHA_strucs, fjson)
    return read_json(fjson)


def parse_qha_results(qha_results: dict, include_structures: bool = True):
    """
    Parse the results from a results.json file for a QHA calculation. Needs work. 
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

from __future__ import annotations
from typing import TYPE_CHECKING
 
import os
 
from tqdm import tqdm
 
if TYPE_CHECKING:
    from matcalc._phonon import PhononCalc  # noqa: F401
 
 
def get_model_tag(user_configs: dict) -> str:
    """
    Centralizes the per-architecture "model" string used to name scripts,
    results files, and job names. Replaces the repeated
    if/elif "chgnet"/"fairchem" blocks scattered across relax_helpers.py.
 
    Add a new elif branch here any time you add a new architecture instead
    of copy-pasting the if/elif block again.
 
    Args:
        user_configs (dict): user configs (must contain "architecture" and
            "relaxer_configs")
 
    Returns:
        model (str): short string identifying the model, used in filenames
    """
 
    architecture = user_configs["architecture"]
 
    if architecture.lower() == "chgnet":
        return user_configs["relaxer_configs"]["model"].replace(".", "")
 
    elif architecture.lower() == "fairchem":
        model_name = user_configs["relaxer_configs"]["name_or_path"]
        model_task = user_configs["relaxer_configs"]["task_name"]
        return f"{model_name}-{model_task}"
 
    elif architecture.lower() == "matcalcphonon":
        return user_configs["relaxer_configs"]["framework"]
 
    raise ValueError(f"Unrecognized architecture: {architecture}")
 
 
def get_matcalc_phonon_configs(
    framework: str = "tensornet",
    calculator_kwargs: dict | None = None,
    atom_disp: float = 0.015,
    min_length: float | None = 20.0,
    supercell_matrix=None,
    t_step: float = 10,
    t_max: float = 1000,
    t_min: float = 0,
    fmax: float = 1e-05,
    max_steps: int = 5000,
    optimizer: str = "FIRE",
    relax_structure: bool = True,
    relax_calc_kwargs: dict | None = None,
    imaginary_freq_tol: float = -0.01,
    on_imaginary_modes: str = "warn",
    fix_imaginary_attempts: int = 0,
    symprec: float = 1e-05,
    write_force_constants: bool | str = False,
    write_band_structure: bool | str = False,
    write_total_dos: bool | str = False,
    write_phonon: bool | str = False,
    verbose: bool = True,
) -> dict:
    """
    Mirrors get_chgnet_configs / get_fairchem_configs, but for a
    matcalc.PhononCalc-based phonon workflow.
 
    "relaxer_configs" holds everything needed to build the calculator +
    instantiate PhononCalc (all constructor-time kwargs).
    "relax_configs" is left empty because PhononCalc.calc(structure) takes
    no extra call-time kwargs (unlike relaxer.relax(struc, fmax=..., ...)).
    Keeping both keys present (even if relax_configs is empty) keeps this
    compatible with setup_job/collect_results, which just persist whatever
    is in those two dict keys.
 
    Args:
        framework (str): mlp framework, passed to get_mlp_calculator
            ("tensornet", "fairchem", "nequix", "nequix-pft")
        calculator_kwargs (dict): kwargs for get_mlp_calculator, model-specific
        atom_disp, min_length, supercell_matrix, t_step, t_max, t_min, fmax,
        max_steps, optimizer, relax_structure, relax_calc_kwargs,
        imaginary_freq_tol, on_imaginary_modes, fix_imaginary_attempts,
        symprec, write_force_constants, write_band_structure,
        write_total_dos, write_phonon: passed straight through to
            matcalc.PhononCalc; see matcalc docs for definitions
        verbose (bool): kept for parity with the other config getters
 
    Returns:
        architecture_configs (dict): dict of architecture/relaxer_configs/
            relax_configs, same shape as get_fairchem_configs
    """
 
    architecture_configs = {
        "architecture": "MatcalcPhonon",
        "relaxer_configs": {},
        "relax_configs": {},
    }
 
    rc = architecture_configs["relaxer_configs"]
    rc["framework"] = framework
    rc["calculator_kwargs"] = calculator_kwargs
    rc["atom_disp"] = atom_disp
    rc["min_length"] = min_length
    rc["supercell_matrix"] = supercell_matrix
    rc["t_step"] = t_step
    rc["t_max"] = t_max
    rc["t_min"] = t_min
    rc["fmax"] = fmax
    rc["max_steps"] = max_steps
    rc["optimizer"] = optimizer
    rc["relax_structure"] = relax_structure
    rc["relax_calc_kwargs"] = relax_calc_kwargs
    rc["imaginary_freq_tol"] = imaginary_freq_tol
    rc["on_imaginary_modes"] = on_imaginary_modes
    rc["fix_imaginary_attempts"] = fix_imaginary_attempts
    rc["symprec"] = symprec
    rc["write_force_constants"] = write_force_constants
    rc["write_band_structure"] = write_band_structure
    rc["write_total_dos"] = write_total_dos
    rc["write_phonon"] = write_phonon
    rc["verbose"] = verbose
 
    return architecture_configs
 
 
def make_phonon_scripts(
    batching: dict, user_configs: dict, phonon_template: str, remake: bool = False
) -> None:
    """
    Mirrors make_relax_scripts, but fills in phonon_template.py instead of
    relax_template.py. Kept as a separate function (rather than branching
    inside make_relax_scripts) because the placeholder set and the
    class-instantiation pattern genuinely differ: PhononCalc is built as
    PhononCalc(calculator, **relaxer_configs) and called as
    phonon_calculator.calc(struc), not {Architecture}Relaxer(...).relax(...).
 
    Args:
        batching (dict): {"batch_id": {"launch_dir": str}}
        user_configs (dict): user configs (architecture == "MatcalcPhonon")
        phonon_template (str): path to phonon_template.py
        remake (bool): if True, remake phonon scripts
 
    Returns:
        None, writes a phonon script for each job (batch)
    """
 
    model = get_model_tag(user_configs)
    total_batches = len(batching)
 
    with tqdm(total=total_batches, desc="Making phonon scripts") as pbar:
 
        for batch_id in batching:
 
            launch_dir = batching[batch_id]["launch_dir"]
 
            phonon_script = os.path.join(launch_dir, f"matcalcphonon_{model}_phonon.py")
 
            if os.path.exists(phonon_script) and not remake:
                pbar.update(1)
                continue
 
            with open(phonon_template, "r", encoding="utf-8") as template_file:
                template_lines = template_file.readlines()
 
            phonon_script_lines = template_lines.copy()
 
            for i, line in enumerate(phonon_script_lines):
 
                indent = line[: len(line) - len(line.lstrip())]
 
                if 'intra_op_threads = "placeholder"' in line:
                    phonon_script_lines[i] = (
                        f'{indent}intra_op_threads = {user_configs["num_intraop_threads"]}\n'
                    )
 
                elif 'inter_op_threads = "placeholder"' in line:
                    phonon_script_lines[i] = (
                        f'{indent}inter_op_threads = {user_configs["num_interop_threads"]}\n'
                    )
 
                elif 'phonon_configs = "placeholder"' in line:
                    config_lines = [
                        f"{indent}{key} = {repr(value)}\n"
                        for key, value in user_configs["relaxer_configs"].items()
                        if key not in ("framework", "calculator_kwargs")
                    ]
                    phonon_script_lines[i : i + 1] = config_lines
 
                elif 'framework = "placeholder"' in line:
                    framework = user_configs["relaxer_configs"]["framework"]
                    phonon_script_lines[i] = f"{indent}framework = {repr(framework)}\n"
 
                elif 'calculator_kwargs = "placeholder"' in line:
                    kwargs = user_configs["relaxer_configs"]["calculator_kwargs"]
                    phonon_script_lines[i] = (
                        f"{indent}calculator_kwargs = {repr(kwargs)}\n"
                    )
 
                elif 'save_interval = "placeholder"' in line:
                    phonon_script_lines[i] = (
                        f"{indent}save_interval = {user_configs['save_interval']}\n"
                    )
 
                elif 'results = os.path.join(curr_dir, "placeholder")' in line:
                    phonon_script_lines[i] = (
                        f"{indent}results = os.path.join(curr_dir, "
                        f"'matcalcphonon_{model}_phonon_results.json')\n"
                    )
 
            with open(phonon_script, "w", encoding="utf-8") as script_file:
                script_file.writelines(phonon_script_lines)
 
            pbar.update(1)
 
    return
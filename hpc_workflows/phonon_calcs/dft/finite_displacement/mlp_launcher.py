import os
import sys
import numpy as np
from pydmclab.utils.handy import read_json, write_json, convert_numpy_to_native
from pydmclab.hpc.helper import get_query
from pydmclab.core.struc import StrucTools
from pydmclab.mlp.fairchem.dynamics import FAIRChemCalculator
from nequix.calculator import NequixCalculator
from scipy.constants import physical_constants
import matcalc as mtc
import matplotlib.pyplot as plt
from pydmclab.plotting.phonons import plot_phonon_bandstructure, plot_phonon_dos, plot_thermal_properties, plot_relative_prop, compare_thermal_properties, plot_phonon_dos_comparison

# HELPERS_DIR = "~/mydrive/bartel-group/pydmclab/hpc_workflows/phonon_calcs"

# # if HELPERS_DIR not in sys.path:
# #     sys.path.append(HELPERS_DIR)
# # from phonon_helpers import estimate_rattle_std

DATA_DIR = os.getcwd()
if not os.path.exists(DATA_DIR):
    os.makedirs(DATA_DIR)

TARGET_FORMULAS = ['S3Sr1Zr1']

# Conversion factors: eV/atom -> J/mol or kJ/mol
EV_TO_J = physical_constants['electron volt-joule relationship'][0]
AVOGADRO = physical_constants['Avogadro constant'][0]
EV_TO_J_PER_MOL = EV_TO_J * AVOGADRO
EV_TO_KJ_PER_MOL = EV_TO_J_PER_MOL / 1000.0


def get_mlp_calculator(framework='tensornet', calculator_kwargs=None):
    '''
    Get MLP calculator based on specified type and kwargs.
    Args:
        framework (str): Type of MLP framework to use. Options: 'tensornet', 'fairchem', 'nequix'.
        calculator_kwargs (dict): Keyword arguments specific to the chosen calculator. 
            For 'tensornet': {'name': str} e.g. {'name': 'r2scan'}
            For 'fairchem': {'name_or_path': str, 'task_name': str} e.g. {'name_or_path': "uma-s-1p2", 'task_name': "omat"}
            For 'nequix': {'model_name': str} e.g. {'model_name': 'nequix-mp-1'}
    '''
    if framework == 'tensornet':
        if calculator_kwargs is None:
            calculator_kwargs = {'name': 'r2scan'}
        mlp_calculator = mtc.load_fp(**calculator_kwargs)
    elif framework == 'fairchem':
        if calculator_kwargs is None:
            calculator_kwargs = {'name_or_path': "uma-s-1p2", 'task_name': "omat"}
        mlp_calculator = FAIRChemCalculator(**calculator_kwargs)
    elif framework == 'nequix':
        if calculator_kwargs is None:
            calculator_kwargs = {'model_name': 'nequix-mp-1-pft'}
        return NequixCalculator(**calculator_kwargs, use_kernel=False)
    else:
        raise ValueError(f"Unsupported calculator: {framework}")
    return mlp_calculator

def parse_phonon_results(phonon_results,
                         band_structure_paths=None):

    final_structure = StrucTools(phonon_results['final_structure']).structure_as_dict
    natoms = len(final_structure['sites'])
    phonon = phonon_results['phonon']
    natoms_primitive = len(phonon.primitive)
    thermal_props = phonon_results['thermal_properties']
    E0 = phonon_results['energy']/natoms
    phonon_results['E_per_at'] = E0

    helmholtz = E0 + (thermal_props['free_energy']/ EV_TO_KJ_PER_MOL / natoms_primitive) # Convert free energy from kJ/mol to eV/atom and add to E0
    entropy = thermal_props['entropy']/ EV_TO_J_PER_MOL / natoms_primitive # Convert entropy from J/mol-K to eV/atom-K
    heat_capacity = thermal_props['heat_capacity']/ EV_TO_J_PER_MOL / natoms_primitive # Convert heat capacity from J/mol-K to eV/atom-K

    thermal_properties = [
        {"T": T, "F": F, "S": S, "Cv": Cv}
        for T, F, S, Cv in zip(
            thermal_props['temperatures'],
            helmholtz,
            entropy,
            heat_capacity
        )
    ]

    phonon_results['thermal_properties'] = thermal_properties
    phonon_results['final_structure'] = final_structure
    
    h = physical_constants['Planck constant in eV/Hz'][0]
    print("Primitive cell natoms:", len(phonon.primitive))
    print("Supercell natoms:", len(phonon.supercell))
    print("Supercell matrix:\n", phonon.supercell_matrix)

    # phonon.run_mesh([30, 30, 30])
    phonon.run_total_dos()
    total_dos = phonon.get_total_dos_dict()
    tdos = total_dos['total_dos']/(h*1e12) # This is to normalize the DOS to 1/eV (phonopy default is 1/THz)
    freq_points_ev = total_dos['frequency_points']*1e12*h #convert frequencies from THz to eV

    if 'disp_supercells' in phonon_results:
        disp_supercells = phonon_results['disp_supercells']
        phonon_results['disp_supercells'] = [StrucTools(s).structure_as_dict for s in disp_supercells]

    tdos = {
                'E0': E0,
                'total_dos': [{'E': E, 'total_dos': dos} for E, dos in zip(freq_points_ev, tdos)],
                'units': {'frequency': 'eV', 'dos': 'states/eV', 'energy': 'eV/atom'}
            }
    
    phonon_results['total_dos'] = tdos

    if band_structure_paths is not None:
        Nq = 100
        
        def get_band(q_start, q_stop, N):
            """ Return path between q_start and q_stop """
            return np.array([q_start + (q_stop-q_start)*i/(N-1) for i in range(N)])

        if isinstance(band_structure_paths, dict):
            path_data = band_structure_paths
            band_structure_paths = list(paths.values())

        path_data=None
        bands = []
        for path in band_structure_paths:
            qpoints = get_band(np.array(path[0]), np.array(path[1]), Nq)
            bands.append(qpoints)
        
        phonon.run_band_structure(bands)
        bs = phonon.get_band_structure_dict()
        if path_data is not None:
            bs['path'] = list(path_data.keys())

        phonon_results['band_structure'] = bs

    phonon_results.pop('phonon')

    return convert_numpy_to_native(phonon_results)

def get_phonon_results(query,
                    phonon_calculator,
                    data_dir=DATA_DIR,
                    savename = "phonon_results.json",
                    remake=False):
    fjson = os.path.join(data_dir, savename)
    if not remake and os.path.exists(fjson):
        return read_json(fjson)
    
    results = {}
    for mpid in query:
        print(f"Calculating phonons for {mpid}...")
        my_structure = StrucTools(query[mpid]['structure']).structure
        phonon_results = phonon_calculator.calc(my_structure,)
        results[mpid] = parse_phonon_results(phonon_results)

    write_json(results, fjson)
    return read_json(fjson)

def get_all_phonon_results(query, 
                           frameworks = ['tensornet', 'fairchem', 'nequix', 'nequix-pft'], 
                            phonon_calc_kwargs = None,
                           data_dir=DATA_DIR, 
                           savename_prefix="phonons", 
                           remake=False,):
    ''' 
    if wanting to calculate using multiple models
    '''


    results = {}
    phonon_calc_kwargs = phonon_calc_kwargs or {}

    for framework in frameworks:
        fjson = os.path.join(data_dir, f"{savename_prefix}_{framework}.json")
        if not remake and os.path.exists(fjson):
            results[framework] = read_json(fjson)
            continue

        print(f"Running phonon calculations with {framework} calculator...")

        calculator_kwargs = None
        if framework == 'nequix':
            calculator_kwargs = {'model_name': 'nequix-mp-1'} 
        elif framework == 'nequix-pft':
            calculator_kwargs = {'model_name': 'nequix-mp-1-pft'}

        mlp_calculator = get_mlp_calculator(framework=framework, calculator_kwargs=calculator_kwargs)
        phonon_calculator = mtc.PhononCalc(mlp_calculator,
                                            **phonon_calc_kwargs
                                            )
        
        r = get_phonon_results(query, 
                                phonon_calculator, 
                                data_dir=data_dir,
                                savename = f"phonons_{framework}.json",
                                remake=remake)
        if calculator_kwargs is not None:
            results[f"{framework}_{calculator_kwargs['model_name']}"] = r
        else:
            results[framework] = r
    return results

def plot_all(results, 
             key_order=[], 
             plot_bs_and_dos=True,
             plot_tprops=True,
             plot_individual_properties=True,
             plot_relative_tprop=False,
             atoms_per_formula_units=None):
    
    '''
    Args:
        results (dict):
            generated with get_phonon_results
                {mpid:
                    phonon results data}
        key order (list):
            in case you want to plot relative prop and want to plot mpid2-mpid1
    '''

    tprops = {'data': {}}

    for mpid in results:
        band_structure = results[mpid]['band_structure']
        total_dos = results[mpid]['total_dos']
        thermal_properties = results[mpid]['thermal_properties']
        if plot_bs_and_dos:
            plot_phonon_bandstructure(band_structure, title=f"{mpid} Phonon Band Structure")
            plot_phonon_dos(total_dos, title=f"{mpid} Phonon DOS")
        if plot_tprops:
            plot_thermal_properties(thermal_properties, plot_props = ["F", "S"], title=f"{mpid} Thermal Properties")  
            if plot_individual_properties:
                  plot_thermal_properties(thermal_properties, plot_props = ["F"], title=f"{mpid} Thermal Properties", atoms_per_formula_units=atoms_per_formula_units) 
                  plot_thermal_properties(thermal_properties, plot_props = ["S"], title=f"{mpid} Thermal Properties", atoms_per_formula_units=atoms_per_formula_units)
                  plot_thermal_properties(thermal_properties, plot_props = ["Cv"], title=f"{mpid} Thermal Properties", atoms_per_formula_units=atoms_per_formula_units)  
        tprops['data'][mpid] = thermal_properties
    if key_order:
        tprops = {'data': {k: tprops['data'][k] for k in key_order}}
    if plot_relative_tprop:
        plot_relative_prop(tprops, prop='F', atoms_per_formula_units=atoms_per_formula_units)
        plot_relative_prop(tprops, prop='S', atoms_per_formula_units=atoms_per_formula_units)
        plot_relative_prop(tprops, prop='Cv', atoms_per_formula_units=atoms_per_formula_units)




def main():
    remake_query = False
    remake_relaxation_results = False
    remake_phonon_results = False

    query = get_query(remake=remake_query)
    mpid = 'S3Sr1Zr1_needle'

    paths = {'Γ-X': [[0.0, 0.5, 0.5], [0.0, 0.0, 0.5]],
            'X-S': [[0.0, 0.5, 0.5], [0.0, 0.0, 0.5]],
            'S-Y': [[0.0, 0.5, 0.5], [0.0, 0.0, 0.5]],
            'Y-Γ': [[0.0, 0.5, 0.5], [0.0, 0.0, 0.5]],
            'Γ-Z': [[0.0, 0.5, 0.5], [0.0, 0.0, 0.5]],
            'Z-U': [[0.0, 0.5, 0.5], [0.0, 0.0, 0.5]],
            'U-R': [[0.0, 0.5, 0.5], [0.0, 0.0, 0.5]],
            'R-T': [[0.0, 0.5, 0.5], [0.0, 0.0, 0.5]],
            'T-Z': [[0.0, 0.5, 0.5], [0.0, 0.0, 0.5]]}

    frameworks = ['tensornet', 'fairchem', 'nequix', 'nequix-pft']

    phonon_calc_kwargs = {
        'atom_disp': 0.015, #eventually do it with estimate rattle_std
        'min_length': 20.0,
        'supercell_matrix': None,
        't_step': 20,
        't_max': 2000,
        't_min': 0,
        'fmax': 1e-5,
        'max_steps': 5000,
        'optimizer': "FIRE",
        'relax_structure': True
    }

    phonon_calculator = get_mlp_calculator(framework='fairchem')
    results = get_phonon_results(query,
                                phonon_calculator,
                                data_dir=DATA_DIR,
                                savename = "phonon_results.json",
                                remake=False)

    all_results = get_all_phonon_results(query,
                                    frameworks=frameworks,
                                    phonon_calc_kwargs=phonon_calc_kwargs,
                                    remake=remake_phonon_results)
    

    plot_all(results, 
             frameworks_to_compare=frameworks, 
             key_order=['S3Sr1Zr1_needle', 'S3Sr1Zr1_dist_perovskite'], 
             plot_bs_and_dos=False,
             plot_tprops=True, 
             plot_individual_framework=False,
             plot_individual_properties=True, 
             atoms_per_formula_units=None)
            





if __name__ == "__main__":
    main()
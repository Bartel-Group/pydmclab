import os
import numpy as np

from pydmclab.utils.handy import read_json, write_json, convert_numpy_to_native
from pydmclab.core.struc import StrucTools
from pydmclab.core.comp import CompTools
from pydmclab.hpc.phonons import AnalyzePhonons

from pymatgen.core.structure import Structure
from pymatgen.io.ase import AseAtomsAdaptor
from pymatgen.io.phonopy import get_phonopy_structure, get_pmg_structure
from pymatgen.analysis.local_env import CrystalNN

from ase import Atoms

from hiphive import ForceConstantPotential
from hiphive import ClusterSpace, StructureContainer, ForceConstantPotential
from hiphive.utilities import prepare_structures
from hiphive.cutoffs import estimate_maximum_cutoff
from trainstation import Optimizer
from hiphive.structure_generation import generate_mc_rattled_structures, generate_rattled_structures

from phonopy import Phonopy
from sklearn.model_selection import KFold


def get_displacements_for_phonons(
                    unitcell: str|dict,
                    method: str,
                    data_dir: str|None,
                    savename: str|None = 'displacements.json',
                    remake: bool|None = False,
                    supercell_matrix: list|None = None,
                    distance: float|None = None,
                    mc: bool = False,
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
            Supercell matrix to use for generating supercells. Highly recommend not using as to not cause confusion. Feed a structured that has already been supercelled.
        distance (float or None):
            Distance for finite displacement only.
        mc (bool):
            If True, will use Monte Carlo method for generating displacements. For hiphive only.
        rattle_std (float or None):
            Standard deviation for random rattling displacements.
        minimum_distance (float or None):
            Minimum distance for hiphive displacement generation only if doing Monte Carlo.

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

        displacement_data = phonon.generate_displacements(distance=distance)
        supercells_with_displacements = phonon.supercells_with_displacements #returns a list of PhonopyAtoms supercells
        pmg_displaced_strucs = [get_pmg_structure(struc) for struc in supercells_with_displacements]

        dataset = phonon.dataset
        out["dataset"] = dataset

    if method == "hiphive":
        #turn unitcell to AseAtoms
        atoms = AseAtomsAdaptor.get_atoms(pymatgen_struc)
        if mc:
            structures = generate_mc_rattled_structures(atoms, n_structures, rattle_std, minimum_distance)
        else:
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

def estimate_rattle_std(structure: str|dict, fraction: float) -> float:
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

    return min_dist * fraction

def get_set_of_forces(results,
                      mpid=None,
                      xc: str = "metagga"):
    '''
    Get the set of calculated forces from multiple structures with displacements for a specific MPID and return as a list of arrays.
    This is for the finite displacement method, where forces will be stored in the results.json under 'results'.
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
    REMINDER: When you generate the displacements, you do ONLY static calculations on those displaced structures to get the forces (no relaxation).
    '''
    if mpid is None:
        set_of_forces = {}
    else:
        set_of_forces = []

    for key in results:
        calc_type = key.split("--")[-1]
        if calc_type != f"{xc}-static":
            continue

        r_mpid = key.split("--")[1]
        mpid_minus_disp = "_".join(r_mpid.split("_")[:-1])

        # Skip if we're looking for a specific mpid and this doesn't match
        if mpid is not None and mpid_minus_disp != mpid:
            continue

        # Extract forces
        forces = results[key].get("forces")
        if not forces:
            print(f"Warning: No forces found for {key}. Adding None to maintain indexing.")
            forces = None

        if forces is not None:
            print(f"Including forces for {key} with shape {np.array(forces).shape}")
        
        if mpid is None:
            new_key = key.replace(r_mpid, mpid_minus_disp)
            if mpid_minus_disp not in set_of_forces:
                set_of_forces[mpid_minus_disp] = {'forces': [], 'key': new_key}
            set_of_forces[mpid_minus_disp]['forces'].append(forces)
        else:
            set_of_forces.append(forces)

    # Return None if no forces found for specific mpid
    if mpid is not None and not set_of_forces:
        print(f"No forces found for mpid: {mpid}")
        return None
    
    return set_of_forces

def to_atoms(structure):
    """Convert various structure formats to ASE Atoms object."""
    if isinstance(structure, Atoms):
        return structure
    elif isinstance(structure, Structure):
        return AseAtomsAdaptor.get_atoms(structure)
    elif isinstance(structure, (dict, str)):
        pmg_structure = StrucTools(structure).structure
        return AseAtomsAdaptor.get_atoms(pmg_structure)
    else:
        raise TypeError(f"Unsupported structure type: {type(structure)}")

def get_cluster_space_hiphive(ideal_supercell: Atoms|dict|str,
                              cutoffs: list[float]|str = "auto",
                                safety_factor: float = 0.95,
                              primitive_cell: Atoms|dict|str|None = None,):
    
    ideal_supercell = to_atoms(ideal_supercell)
    if cutoffs == "auto":
        max_cutoff = estimate_maximum_cutoff(ideal_supercell)
        print(f"Estimated maximum cutoff: {max_cutoff} Å")
        cutoffs = [max_cutoff * safety_factor]  # Example: second order cutoffs, could add higher order if were doing third order + force constants. Right now only doing second order.
        print(f"Using cutoffs: {cutoffs} Å")
    if primitive_cell is None:
        cs = ClusterSpace(ideal_supercell, cutoffs)
    else:
        cs = ClusterSpace(primitive_cell, cutoffs)
    return cs

def get_fcp_hiphive(ideal_supercell: Atoms|dict|str, 
                    rattled_structures: list[Atoms|dict|str], 
                    force_sets: list|np.ndarray,
                    primitive_cell: Atoms | None = None,
                    cutoffs: list[float] |str = "auto",
                    safety_factor: float = 0.95,
                    data_dir: str = None,
                    savename: str = "fcp.fcp",
                    remake: bool = False):
    """
        Workflow for getting force constant potential object for a hiphive calculation. 
        With this fcp object, you can compute force constants for any size supercell, not necessarily just the size you used to create the original supercell and rattled structures.
        can generate force constant array with `fcp.get_force_constants(supercell)` implemented in get_force_constants_hiphive()
        Args:
            ideal_supercell (Atoms | dict | str): 
                The ideal supercell structure (no rattling). Can be provided as an Atoms or Structure object, a dictionary, or a path to a structure file.
            rattled_structures (list): 
                List of rattled structures as Atoms, Structure objects, dictionaries, or paths to structure files.
            force_sets (list): 
                List of force sets corresponding to the rattled structures. Must be in the same order as rattled_structures!
            primitive_cell (Atoms | MSONAtoms): 
                The primitive cell structure. If None is given then it will be calculated from the ideal supercell using spglib. Can be provided as an Atoms or Structure object, a dictionary, or a path to a structure file.
            cutoffs (list | str): 
                List of cutoff distances for the cluster space, in order of increasing order starting with second order.
                This can be either manually specified or "auto". 
                If auto is given, it will estimate the maximum cutoff based on the ideal supercell structure and takes a factor 0.95 of it for safety.
                Which is the most rigorous/expensive cutoff you can use.
            safety_factor (float):
                Safety factor to apply when estimating maximum cutoff if cutoffs="auto". Default is 0.95.
        Returns:
            ForceConstantPotential: The constructed hiphive force constant potential object.
    """
    if data_dir is not None:
        fcp_dir = os.path.join(data_dir, savename)
        if os.path.exists(fcp_dir) and not remake:
            return ForceConstantPotential.read(fcp_dir), None, None

    if len(rattled_structures) != len(force_sets):
        raise ValueError("The length of rattled_structures and force_sets must be the same.")

    # Convert all structures to Atoms objects
    ideal_supercell = to_atoms(ideal_supercell)
    rattled_structures = [to_atoms(s) for s in rattled_structures]
    if primitive_cell is not None:
        primitive_cell = to_atoms(primitive_cell)
        
    force_sets = np.array(force_sets)
    cs = get_cluster_space_hiphive(ideal_supercell, cutoffs=cutoffs, safety_factor=safety_factor, primitive_cell=primitive_cell)

    print(cs)
    cs.print_orbits()

    for i, structure in enumerate(rattled_structures):
        #remove calculator from atoms object so that it does not interfere with how hiphive prepares structures
        #See https://gitlab.com/materials-modeling/hiphive/-/blob/master/hiphive/utilities.py to see how it works
        #It seems safest to store the forces in the structure arrays and then setting calculator to None means that the forces are not recalculated and they just grab them from the stored arrays
        structure.calc = None
        structure.arrays['forces'] = force_sets[i] #This is where the forces are stored in the structure, so hiphive can use them to calculate force constants.

    # ... and structure container
    structures = prepare_structures(rattled_structures, ideal_supercell)

    sc = StructureContainer(cs)
    for structure in structures:
        sc.add_structure(structure)
    print(sc)

    # train model
    opt = Optimizer(sc.get_fit_data())
    opt.train()

    optim = {'n_parameters': opt.n_parameters,
             'n_target_values': opt.n_target_values,
             'rmse_train': opt.rmse_train,
             'rmse_test': opt.rmse_test,
             'r2_train': opt.r2_train,
             'r2_test': opt.r2_test}
    
    
    print(opt)


    # construct force constant potential
    fcp = ForceConstantPotential(cs, opt.parameters)
    print(fcp)
    
    if data_dir is not None:
        fcp.write(fcp_dir)
        return fcp.read(fcp_dir), cs, opt

    return fcp, cs, optim

def get_force_constants_hiphive(fcp, 
                                supercell, order=2):
    """
    Obtain force constants array from a hiphive force constant potential object.
    Args:
        fcp (ForceConstantPotential): 
            The force constant potential object.
        supercell (Atoms): 
            The supercell structure to compute the force constants for. 
            This does not necessarily need to match the supercell used to obtain the forces and generate the ForceConstantPotential object.
        order (int): 
            The order of the force constants to compute.
    Returns:
        np.ndarray: The computed force constants array.
    """
    supercell = to_atoms(supercell)
    fcs = fcp.get_force_constants(supercell)
    print(fcs)
    # access specific parts of the force constant matrix
    fcs.print_force_constant((0, 1))
    fcs.print_force_constant((10, 12))
    fcs = fcs.get_fc_array(order=order)
    return fcs

def get_force_data_mlp(displaced_structures: list[dict|Atoms], relaxer: object = None,
                       name_or_path: str = "uma-s-1", task_name: str = "omat",
                       data_dir: str = None, savename: str = "force_data.json", remake: bool = False):
    """
    Get force data from MLP for displaced structures.
    Args:
        displaced_structures (list or dict): 
            The displaced structures to get force data for as an Atoms object.
                If list, each element is a structure with displacements.
                If dict, must contain "displaced_structures" key. Usually generated with get_displacements_for_phonons(),
                this way it contains all of the other information in the dict (original unitcell, dataset for phonopy).
        relaxer (object): 
            The MLP relaxer object. If None, will load fairchemRelaxer model using name_or_path and task_name.
        data_dir (str or None):
        name_or_path (str): 
            The name or path to the MLP model.
        task_name (str): 
            The task name for the MLP model.

    Returns:
        dict: The force data for the displaced structures as a list of dictionaries.
            {"results": [{'structure': displaced_struc, 
                          'forces': forces, 
                          'energy': energy}, .....],
            "unitcell": unitcell,
            "dataset": dataset,
            "any other keys": "..."
            }
    """

    if data_dir is not None:
        fjson = os.path.join(data_dir, savename)
        if os.path.exists(fjson) and not remake:
            return read_json(fjson)
    
    if relaxer is None:
        from pydmclab.mlp.fairchem.dynamics import FAIRChemRelaxer #Putting in this for now bc importing this requires installing the fairchem extension
        relaxer = FAIRChemRelaxer(name_or_path=name_or_path, task_name=task_name)

    if isinstance(displaced_structures, list):
        out = {"results": []}
    elif isinstance(displaced_structures, dict):
        out = displaced_structures
        displaced_structures = out.pop("displaced_structures", None)
        out["results"] = []
        
    for displaced_struc in displaced_structures:
        if isinstance(displaced_struc, dict):
            st = StrucTools(displaced_struc)
            displaced_struc = st.structure
            atoms_displaced_struc = AseAtomsAdaptor.get_atoms(displaced_struc)
        else:
            atoms_displaced_struc = displaced_struc
            pmg_displaced_struc = AseAtomsAdaptor.get_structure(displaced_struc)
            st = StrucTools(pmg_displaced_struc)
        prediction = relaxer.predict_structure(atoms_displaced_struc)
        forces = prediction['forces']
        energy = prediction['energy']
        #could also get stresses if wanted from prediction
        out['results'].append({
            "structure": st.structure_as_dict,
            "forces": forces,
            "energy": energy
        })

    out = convert_numpy_to_native(out)  # Make sure the output is JSON serializable
    if data_dir is not None:
        write_json(out, fjson)
        return read_json(fjson)
    else:
        return out
    

def get_fcp_uncertainty(ideal_supercell, rattled_structures, force_sets, 
                    n_folds=5, calculate_phonons=False, **kwargs):
    """
    Get force constant potential with uncertainty via cross-validation.
    Args:
        ideal_supercell (Atoms | dict | str): 
            The ideal supercell structure (no rattling). Can be provided as an Atoms or Structure object, a dictionary, or a path to a structure file.
        rattled_structures (list): 
            List of rattled structures as Atoms, Structure objects, dictionaries, or paths to structure files.
        force_sets (list): 
            List of force sets corresponding to the rattled structures. Must be in the same order as rattled_structures!
        n_folds (int): 
            Number of folds for cross-validation.
        calculate_phonons (bool): 
            If True, will calculate phonon properties for each fold and return statistics.
        **kwargs: 
            Additional keyword arguments to pass to get_cluster_space_hiphive(), e.g., cutoffs.
    Returns:
        final_fcp (ForceConstantPotential): 
            The final force constant potential object trained on the mean parameters from cross-validation.
        cs (ClusterSpace): 
            The cluster space used for the force constant potential.
        out (dict): 
            Dictionary containing statistics from cross-validation:
            {
                'mean_parameters': Mean of the fitted parameters across folds,
                'std_parameters': Standard deviation of the fitted parameters across folds,
                'overall_param_std_mean': Mean of the standard deviations of the parameters,
                'overall_fc_std_mean': Mean of the standard deviations of the force constants,
                'all_parameters': Array of all fitted parameters from each fold,
                'cv_results': Force constants from the final model,
                'phonon_results': {
                    'free_energy_mean': Mean free energy across folds,
                    'free_energy_std': Standard deviation of free energy across folds,
                    'overall_free_energy_std_mean': Mean of the standard deviations of free energy,
                    'heat_capacity_mean': Mean heat capacity across folds,
                    'heat_capacity_std': Standard deviation of heat capacity across folds,
                    'overall_heat_capacity_std_mean': Mean of the standard deviations of heat capacity,
                    'entropy_mean': Mean entropy across folds,
                    'entropy_std': Standard deviation of entropy across folds,
                    'overall_entropy_std_mean': Mean of the standard deviations of entropy,
                    'total_dos_mean': Mean total DOS across folds,
                    'total_dos_std': Standard deviation of total DOS across folds,
                    'overall_total_dos_std_mean': Mean of the standard deviations of total DOS,
                }
            }
    """
    # Create the ClusterSpace once (this defines your basis)
    atoms_ideal_supercell = to_atoms(ideal_supercell)
    rattled_structures = [to_atoms(s) for s in rattled_structures]
    force_sets = np.array(force_sets)
    cutoffs = kwargs.get('cutoffs')

    cs = get_cluster_space_hiphive(atoms_ideal_supercell, cutoffs)

    # Set up cross-validation
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=42)
    force_constant_results = []
    fcp_parameters = []
    phonon_results = {'free_energy': [], 'heat_capacity': [], 'entropy': [], 'total_dos': []}


    for fold_idx, (train_idx, val_idx) in enumerate(kf.split(rattled_structures)):
        print(f"Training fold {fold_idx + 1}/{n_folds}")
        
        # Get training data for this fold
        train_structures = [rattled_structures[i] for i in train_idx]
        train_forces = [force_sets[i] for i in train_idx]

        for i, structure in enumerate(train_structures):
            structure.calc = None
            structure.arrays['forces'] = train_forces[i]

        structures = prepare_structures(train_structures, atoms_ideal_supercell)
        sc = StructureContainer(cs)  # Same ClusterSpace for all folds!
        for structure in structures:
            sc.add_structure(structure)
            
        opt = Optimizer(sc.get_fit_data())
        opt.train()
        
        # Store the trained parameters
        fcp_parameters.append(opt.parameters.copy())
        
        # Create FCP and get force constants for uncertainty analysis
        fcp = ForceConstantPotential(cs, opt.parameters)
        force_constants = fcp.get_force_constants(atoms_ideal_supercell).get_fc_array(order=2) # Example for second order
        # Store whatever you want to analyze uncertainty for
        force_constant_results.append(force_constants) # Example for second order

        if calculate_phonons:
            analyzer = AnalyzePhonons(
                unitcell=ideal_supercell,
                force_data=force_constants,
            )

            summary = analyzer.summary()

            # Normalize thermal properties by n_atoms
            n_atoms = len(ideal_supercell['sites'])
            if n_atoms:
                if 'thermal_properties' in summary and isinstance(summary['thermal_properties'], list):
                    for tp in summary['thermal_properties']:
                        if 'free_energy' in tp:
                            tp['free_energy'] /= n_atoms
                        if 'heat_capacity' in tp:
                            tp['heat_capacity'] /= n_atoms
                        if 'entropy' in tp:
                            tp['entropy'] /= n_atoms

            free_energies = [tp['free_energy'] for tp in summary['thermal_properties']]
            heat_capacities = [tp['heat_capacity'] for tp in summary['thermal_properties']]
            entropies = [tp['entropy'] for tp in summary['thermal_properties']]
            total_dos = summary['total_dos']['total_dos']
            phonon_results['free_energy'].append(free_energies)
            phonon_results['heat_capacity'].append(heat_capacities)
            phonon_results['entropy'].append(entropies)
            phonon_results['total_dos'].append(total_dos)


    # Calculate statistics
    parameters_array = np.array(fcp_parameters)
    mean_parameters = np.mean(parameters_array, axis=0)
    std_parameters = np.std(parameters_array, axis=0)
    overall_param_std_mean = np.mean(std_parameters)

    force_constants_array = np.array(force_constant_results)
    mean_force_constants = np.mean(force_constants_array, axis=0)
    std_force_constants = np.std(force_constants_array, axis=0)
    overall_fc_std_mean = np.mean(std_force_constants)

    # Create final model with mean parameters
    final_fcp = ForceConstantPotential(cs, mean_parameters)

    out = {
        'mean_parameters': mean_parameters,
        'std_parameters': std_parameters,
        'overall_param_std_mean': overall_param_std_mean,
        'overall_fc_std_mean': overall_fc_std_mean,
        'all_parameters': parameters_array,
        'cv_results': final_fcp.get_force_constants(atoms_ideal_supercell).get_fc_array(order=2) # Example for second order
    }

    if calculate_phonons:
        free_energy_array = np.array(phonon_results['free_energy'])
        heat_capacity_array = np.array(phonon_results['heat_capacity'])
        entropy_array = np.array(phonon_results['entropy'])
        total_dos_array = np.array(phonon_results['total_dos'])
        out['phonon_results'] = {
            'free_energy_mean': np.mean(free_energy_array, axis=0),
            'free_energy_std': np.std(free_energy_array, axis=0),
            'overall_free_energy_std_mean': np.mean(np.std(free_energy_array, axis=0)),
            'heat_capacity_mean': np.mean(heat_capacity_array, axis=0),
            'heat_capacity_std': np.std(heat_capacity_array, axis=0),
            'overall_heat_capacity_std_mean': np.mean(np.std(heat_capacity_array, axis=0)),
            'entropy_mean': np.mean(entropy_array, axis=0),
            'entropy_std': np.std(entropy_array, axis=0),
            'overall_entropy_std_mean': np.mean(np.std(entropy_array, axis=0)),
            'total_dos_mean': np.mean(total_dos_array, axis=0) if total_dos_array.size else None,
            'total_dos_std': np.std(total_dos_array, axis=0) if total_dos_array.size else None, 
            'overall_total_dos_std_mean': np.mean(np.std(total_dos_array, axis=0)) if total_dos_array.size else None,
        }

    return final_fcp, cs, out
import os
import numpy as np
import matplotlib.pyplot as plt
from pydmclab.plotting.utils import get_colors, set_rc_params
from matplotlib.ticker import MaxNLocator


from pydmclab.utils.handy import read_json, write_json
from pydmclab.core.struc import StrucTools

from phonopy import Phonopy, PhonopyQHA
from phonopy.interface.vasp import read_vasp, parse_force_constants

from ase.phonons import Phonons
from ase.thermochemistry import CrystalThermo

from pymatgen.io.ase import AseAtomsAdaptor
from pymatgen.analysis.eos import Murnaghan, Vinet

set_rc_params()
COLORS = get_colors(palette="tab10")

class AnalyzePhonons(object):
    def __init__(self, calc_dir: str, 
                 supercell_matrix: list = None, 
                 mesh: int|list|float=100):
        """
        Class to analyze phonon data from a VASP calculation using Phonopy, Can return force constants, dynamical matrix, mesh data, thermal properties, band structure, and total density of states.
        Args:
            calc_dir (str):
                Path to the directory containing the VASP calculation
            supercell_matrix (list):
                Supercell matrix for the phonon calculation. e.g. [[2, 0, 0], [0, 2, 0], [0, 0, 2]] for a 2x2x2 supercell.
            mesh (array-like or float):
                Mesh numbers along a, b, c axes when array_like object is given, shape=(3,).
                When float value is given, uniform mesh is generated following VASP convention by N = max(1, nint(l * |a|^*)) where 'nint' is the function to return the nearest integer. In this case, it is forced to set is_gamma_center=True.
                Default value is 100.0.

        """

        self.calc_dir = calc_dir
        self.supercell_matrix = supercell_matrix

        if isinstance(mesh, (list, tuple)) and len(mesh) != 3:
            raise ValueError(
                "Mesh should be a list of three integers or an int."
            )

        poscar_path = os.path.join(calc_dir, "POSCAR")
        if not os.path.exists(poscar_path):
            print(f"Warning: POSCAR file not found in {calc_dir}. Returning None.")
            self.phonon = None
            return  # Early exit from the initialization if POSCAR is missing

        unitcell = read_vasp(poscar_path)
        phonon = Phonopy(unitcell, supercell_matrix)
        self.unitcell = unitcell

        force_constants_path = os.path.join(calc_dir, "vasprun.xml")
        if not os.path.exists(force_constants_path):
            print(f"Warning: vasprun.xml file not found in {calc_dir}. Returning None.")
            self.phonon = None
            return  # Early exit if force constants are missing

        force_constants_dict = parse_force_constants(force_constants_path)
        if not force_constants_dict:
            print("Warning: Failed to parse force constants. Returning None.")
            self.phonon = None
            return  # Exit if force constants parsing fails

        phonon.force_constants = force_constants_dict[0]
        self.force_constants = phonon.force_constants #This is just a setter for the force constants, need it to make the dynamical matrix

        self.dynamical_matrix = phonon.dynamical_matrix # This is just a phonopy.dynamical_matrix.DynamicalMatrix object. Need it to run mesh.
        _mesh_out = phonon.run_mesh(mesh) #Need to run this in order to get mesh data, thermal properties, band structure, and total density of states

        self.phonon = phonon

    @property
    def mesh_dict(self):
        """
        Returns the mesh data for the phonon object in a dictionary
        """
        mesh_dict = self.phonon.get_mesh_dict()
        return mesh_dict

    def parse_thermal_properties(self, phonopy_data: dict):
        """
        Parses the thermal properties data from the phonopy object into a list of dictionaries
        Args:
            phonopy_data (dict):
                Thermal properties data obtained from the phonopy object self.phonon.get_thermal_properties_dict()

        Returns:
            A list of dictionaries where each dictionary corresponds to a specific temperature point.
            e.g. [{'temperature': 300, 'free_energy': float, 'entropy': float, 'heat_capacity': float},
                {'temperature': 310, 'free_energy': float, 'entropy': float, 'heat_capacity': float}, ...]
        """
        parsed_data = []

        temperatures = phonopy_data["temperatures"]
        free_energies = phonopy_data["free_energy"]
        entropies = phonopy_data["entropy"]
        heat_capacities = phonopy_data["heat_capacity"]

        for i, T in enumerate(temperatures):
            data_point = {
                "temperature": T,
                "free_energy": free_energies[i],
                "entropy": entropies[i],
                "heat_capacity": heat_capacities[i],
            }
            parsed_data.append(data_point)

        return parsed_data


    def thermal_properties(
        self,
        t_min: int|float =0,
        t_max: int|float = 2000,
        t_step: int =20,
        temperatures: list|int|float|np.ndarray = None,
        cutoff_frequency: int|float = None,
        pretend_real: bool = False,
        band_indices: list = None,
        is_projection: bool = False,
        force_rerun: bool = False,
    ):
        """
        returns the thermal properties for the phonon object in a dictionary
        Args:
            t_min, t_max, t_step (float, optional)
                Minimum and maximum temperatures and the interval in this
                temperature range. Default values are 0, 1000, and 10.
            temperatures (array_like, optional)
                Temperature points where thermal properties are calculated.
                When this is set, t_min, t_max, and t_step are ignored.
            cutoff_frequency (float, optional)
                Ignore phonon modes whose frequencies are smaller than this value.
                Default is None, which gives cutoff frequency as zero.
            pretend_real (bool, optional)
                Use absolute value of phonon frequency when True. Default is False.
            band_indices (array_like, optional)
                Band indices starting with 0. Normally the numbers correspond to
                phonon bands in ascending order of phonon frequencies. Thermal
                properties are calculated only including specified bands.
                Note that use of this results in unphysical values, and it is not
                recommended to use this feature. Default is None.
            is_projection (bool, optional)
                When True, fractions of squeared eigenvector elements are
                multiplied to mode thermal property quantities at respective phonon
                modes. Note that use of this results in unphysical values, and it
                is not recommended to use this feature. Default is False.
            force_rerun (bool, optional)
                If you already ran thermal properties but now want to change some of the arguments
                and want it to recalculate the thermal properties, set this to True. 
                Default is False.

        Returns parsed thermal properties in the following format:
        A list of dictionaries where each dictionary corresponds to a specific temperature point.
        e.g. [{'temperature': 300, 'free_energy': float, 'entropy': float, 'heat_capacity': float},
                {'temperature': 310, 'free_energy': float, 'entropy': float, 'heat_capacity': float}, ...]
        """
        if force_rerun or not hasattr(self, '_thermal_properties'):
            print("Calculating thermal properties...")
            self.phonon.run_thermal_properties(
                t_min=t_min,
                t_max=t_max,
                t_step=t_step,
                temperatures=temperatures,
                cutoff_frequency=cutoff_frequency,
                pretend_real=pretend_real,
                band_indices=band_indices,
                is_projection=is_projection,
            )
            
            tp = self.phonon.get_thermal_properties_dict()
            if tp is not None:
                self._thermal_properties = self.parse_thermal_properties(tp)
            else:
                print("Thermal properties could not be calculated.")
                return None
        
        return self._thermal_properties

    def band_structure(
        self,
        paths: list =[
            [[0.0, 0.0, 0.0], [0.5, 0.0, 0.0]],  # Γ to X
            [[0.5, 0.0, 0.0], [0.5, 0.5, 0.5]],  # X to L
            [[0.5, 0.5, 0.5], [0.0, 0.0, 0.0]],  # L to Γ
            [[0.5, 0.0, 0.0], [0.5, 0.25, 0.75]],  # X to W
        ],
    ):
        """
        Returns the band structure for the phonon object in a dictionary
        Args:
            paths (list):
                List of paths in reciprocal space. e.g. [[[0.0, 0.0, 0.0], [0.5, 0.0, 0.0]], [[0.5, 0.0, 0.0], [0.5, 0.5, 0.5]], ...]

        Returns:
            {'qpoints': arrays of q points, 'distances': arrays of distances, 'frequencies': arrays of frequencies, 'eigenvectors': arrays of eigenvectors, group_velocities': arrays of group velocities}
        """
        if not hasattr(self, '_band_structure') or self._band_structure_paths != paths:
            print("Calculating band structure...")
            _ = self.phonon.run_band_structure(paths)
            self._band_structure = self.phonon.get_band_structure_dict()
            self._band_structure_paths = paths  
        
        return self._band_structure

    @property
    def total_dos(self):
        """
        Returns the total density of states for the phonon object in a dictionary.
        Returns:
            {'frequency_points ': array of frequency points, 'total_dos': array of total density of states}
        """
        if not hasattr(self, '_total_dos'):
            print("Calculating total density of states...")
            _ = self.phonon.run_total_dos()
            self._total_dos = self.phonon.get_total_dos_dict()

        return self._total_dos


    def make_json_serializable(self, data):
        """
        Makes the data JSON serializable by converting NumPy arrays to lists.
        """

        if isinstance(data, dict):
            return {key: self.make_json_serializable(value) for key, value in data.items()}
        elif isinstance(data, list):
            return [self.make_json_serializable(item) for item in data]
        elif isinstance(data, np.ndarray):
            return data.tolist() 
        else:
            return data
  

    def summary(
        self,
        savename: str = "phonons.json",
        remake: bool = False,
        include_force_constants: bool= True,
        include_mesh: bool = True,
        include_thermal_properties: bool = True,
        include_band_structure = True,
        include_total_dos: bool = True,
        paths: list = None, 
        temperatures: list|int|float|np.ndarray = None,
        cutoff_frequency: int|float = None,
        pretend_real: bool = None,
        band_indices: bool = None,
        is_projection: bool = None,
    ):
        
        """
        Returns all desired data for post-processing DFT calculations

        Args:

            include_force_constants (bool, optional):
                Include force constants in the output. Default is True.
            include_mesh (bool, optional):  
                Include mesh data in the output. Default is True.
            include_thermal_properties (bool, optional):
                Include thermal properties in the output. Default is True.
            include_band_structure (bool, optional):
                Include band structure in the output. Default is True.
            include_total_dos (bool, optional):         
                Include total density of states in the output. Default is True.
            paths (list, optional):
                List of paths in reciprocal space. Default is None.
            temperatures (array-like, optional):
                Temperature points where thermal properties are calculated. Default is None.
            cutoff_frequency (float, optional):
                Ignore phonon modes whose frequencies are smaller than this value. Default is None.
            pretend_real (bool, optional):
                Use absolute value of phonon frequency when True. Default is False.
            band_indices (array-like, optional):    
                Band indices starting with 0. 

        Returns:
            Dictionary with the specified information
            {'force_constants': array of force constants, 
                                'mesh': mesh array, 
                                'thermal_properties': dictionary of thermal properties, 
                                'band_structure': band structure data, 
                                'total_dos': dos data}

        """
        
        calc_dir = self.calc_dir
        fjson = os.path.join(calc_dir, savename)
        if os.path.exists(fjson) and not remake:
            return read_json(fjson)
        data = {}
    

        if include_force_constants:
            fc = self.force_constants
            force_constants = self.make_json_serializable(fc)
            data["force_constants"] = force_constants

        if include_mesh:
            mesh_array = self.mesh_dict
            mesh_list = self.make_json_serializable(mesh_array)
            data["mesh"] = mesh_list

        if include_thermal_properties:
            thermal_properties_kwargs = {}

            if temperatures is not None:
                thermal_properties_kwargs["temperatures"] = temperatures
            if cutoff_frequency is not None:
                thermal_properties_kwargs["cutoff_frequency"] = cutoff_frequency
            if pretend_real is not None:
                thermal_properties_kwargs["pretend_real"] = pretend_real
            if band_indices is not None:
                thermal_properties_kwargs["band_indices"] = band_indices
            if is_projection is not None:
                thermal_properties_kwargs["is_projection"] = is_projection

            # Pass the arguments dynamically
            if thermal_properties_kwargs:
                tp = self.thermal_properties(**thermal_properties_kwargs, force_rerun=True)
            else:
                tp = self.thermal_properties()
            
            data["thermal_properties"] = tp


        if include_band_structure:
            if paths:
                band_struc = self.make_json_serializable(self.band_structure(paths=paths))
            else:
                band_struc = self.make_json_serializable(self.band_structure())

            data["band_structure"] = band_struc

        if include_total_dos:
            total_dos = self.make_json_serializable(self.total_dos)
            data["total_dos"] = total_dos

        write_json(data, fjson)
        return read_json(fjson)
    
    #Plotting functions are just using phonopy's built in plotting functions for the moment, need to updgrade this in the future
    @property
    def plot_thermal_properties(self):
        self.thermal_properties() #If thermal properties haven't been calculated, calculations will be done with defaults
        self.phonon.plot_thermal_properties()

    @property
    def plot_band_structure(self):
        self.band_structure()
        self.phonon.plot_band_structure()

    @property
    def plot_total_dos(self):
        self.total_dos
        self.phonon.plot_total_dos()
    
    @property
    def plot_band_structure_and_dos(self):
        self.band_structure(paths = self._band_structure_paths)
        self.total_dos
        self.phonon.plot_band_structure_and_dos()



class QHA(object):
    def __init__(self, results: dict, eos: str = "vinet"):
        self.results = results
        self.eos = eos
        self._parsed_results = None  # Cache for parsed results
        self._gibbs_dict_cache = None  # Cache for Gibbs free energy dictionary
        self._qha_cache = {}  # Cache for PhonopyQHA instances

    @property
    def parse_results(self):
        """
        Calls the _parse_results method to parse the results dictionary if not already parsed.
        """
        if self._parsed_results is None:  # Only parse if not cached
            self._parsed_results = self._parse_results
        return self._parsed_results

    @property
    def _parse_results(self):
        """
        Rearrange the results dictionary to group the data by formula, MPID, and volume scale.
        """
        results = self.results
        parsed_results = {}

        for key in results:
            # Only proceed with keys that have "dfpt" at the end and obtain the E_per_at from static calculations
            if key.split("--")[-1].split("-")[-1] != "dfpt":
                continue
            formula, mpid = key.split("--")[0], key.split("--")[1]
            xc = key.split("--")[-1].split("-")[0]
            scale = mpid.split("_")[-1]
            mpid_minus_scale = "_".join(mpid.split("_")[:-1])

            phonon_data = results[key]['phonons']
            structure = results[key]['structure']

            static_key = "--".join(key.split("--")[:-1] + [f"{xc}-static"])
            if static_key not in results:
                print(f"Warning: Static key {static_key} not found in results. Skipping.")
                continue

            E_per_at = results[static_key]['results']['E_per_at']
            n_atoms = len(structure['sites'])
            E_electronic = n_atoms * E_per_at

            if formula not in parsed_results:
                parsed_results[formula] = {}

            if mpid_minus_scale not in parsed_results[formula]:
                parsed_results[formula][mpid_minus_scale] = {}

            if scale not in parsed_results[formula][mpid_minus_scale]:
                parsed_results[formula][mpid_minus_scale][scale] = {}

            parsed_results[formula][mpid_minus_scale][scale] = {
                'phonons': phonon_data,
                'structure': structure,
                'E_electronic': E_electronic
            }

        return parsed_results

    @property
    def structures(self):
        """
        Returns:
            dictionary where keys are tuples of (formula, mpid) with lists of isotropically strained structures as values,
            e.g.
            {('Al1N1', 'mp-661'): [list of strained structures]}
        """
        results = self.parse_results
        structures = {
            (formula, mpid): [results[formula][mpid][scale]['structure'] for scale in results[formula][mpid]]
            for formula in results
            for mpid in results[formula]
        }
        return structures

    @property
    def volumes(self):
        """
        Returns:
            dict: A dictionary where keys are tuples of (formula, mpid)
                and values are lists of volumes (A**3) for the corresponding strained structures
                e.g. {('Al1N1', 'mp-661'): [list of volumes]}
        """
        structures_dict = self.structures
        volumes = {
            key: [StrucTools(structure).structure.volume for structure in structures]
            for key, structures in structures_dict.items()
        }
        return volumes
    
    @property
    def phonon_dos(self):
        """
        Returns:
            dict: A dictionary where keys are tuples of (formula, mpid), second key is volume key, and values are dictionaries containing
                'frequency_points' and 'total_dos'.
                e.g. dict[('Al1N1', 'mp-661')] = {
                                {scaling (float) :
                                    {'E0' : 0 K internal energy (eV/cell),
                                    'dos' :
                                        [{'E' : energy level (eV),
                                        'dos' : phonon DOS at E (float)}]
                                    }
                }
        """
        results = self.parse_results
        dos_data = {}

        for formula in results:
            for mpid in results[formula]:
                # Initialize the main dictionary for each (formula, mpid) pair
                dos_data[(formula, mpid)] = {}
                for scale, data in results[formula][mpid].items():
                    phonons_data = data['phonons']['total_dos']
                    frequency_points = phonons_data.get('frequency_points')
                    total_dos = phonons_data.get('total_dos')
                    
                # Ensure that both frequency_points and total_dos are not None
                if frequency_points is not None and total_dos is not None:
                    dos_data[(formula, mpid)][float(scale)] = {
                        'E0': data['E_electronic'],
                        'dos': [{'E': E, 'dos': d} for E, d in zip(frequency_points, total_dos)]
                    }
                else:
                    print(f"Warning: Missing 'frequency_points' or 'total_dos' for formula {formula}, mpid {mpid}, scale {scale}")

        return dos_data


    def properties_for_one_struc(self, formula, mpid):
        """
        Returns:
            dict: A dictionary where keys are volumes (A**3) and values are dictionaries containing
                'data', which is a list of dictionaries with temperature (T), Helmholtz free energy (F),
                entropy (S), heat capacity (Cv), and electronic energy (E_electronic).
                e.g. {volume(float): {'data': [{'T': 300, 'F': float, 'S': float, 'Cv': float, 'E_electronic': float}, ...]}}
        """
        results = self.parse_results
        volumes_dict = self.volumes  

        properties_dict = {}
        volume_list = volumes_dict.get((formula, mpid), [])
        
        for i, scale in enumerate(results[formula][mpid]):
            thermal_properties = results[formula][mpid][scale]['phonons']['thermal_properties']
            E_electronic = results[formula][mpid][scale]['E_electronic']
            volume = volume_list[i] if i < len(volume_list) else None
            
            if volume is not None:
                if volume not in properties_dict:
                    properties_dict[volume] = {'data': []}
                
                for prop in thermal_properties:
                    properties_dict[volume]['data'].append({
                        'T': prop['temperature'],
                        'F': prop['free_energy'],
                        'S': prop['entropy'],
                        'Cv': prop['heat_capacity'],
                        'E_electronic': E_electronic
                    })

        return properties_dict
    
    @property
    def temperatures(self):
        """
        Returns:
            list: A list of temperatures (K) from the thermal properties data
        """
        props = self.properties_for_one_struc(*list(self.parse_results.keys())[0])
        volumes = list(props.keys())
        temperatures = [i['T'] for i in props[volumes[0]]['data']]
        return temperatures
    
    def thermo_one_struc_scale(self, formula, mpid, scale):
        """
        Returns:
            ase CrystalThermo object
        """
        phonon_dos = self.phonon_dos[formula, mpid][scale]
        self.E0 = phonon_dos["E0"]

        self.phonon_energies = [
            phonon_dos["dos"][i]["E"] for i in range(len(phonon_dos["dos"]))
        ]
        self.phonon_dos = [
            phonon_dos["dos"][i]["dos"] for i in range(len(phonon_dos["dos"]))
        ]
        return CrystalThermo(
            phonon_energies=self.phonon_energies,
            phonon_DOS=self.phonon_dos,
            potentialenergy=self.E0,
            formula_units=self.formula_units,
        )

    def helmholtz_one_struc(self, formula, mpid):
        """
        Returns:
            dict
                {volume (A**3) :
                    {'data' :
                        [{'T' : temperature (K),
                        'F' : Helmholtz free energy (eV/cell)}]
                    }
                }

        """
        props = self.properties_for_one_struc(formula, mpid)
        volumes = self.volumes[(formula, mpid)]
        temperatures = self.temperatures
        out = {}

        for idx, scale in enumerate(props):
            volume = volumes[idx]  # Use the corresponding volume for each scale
            out[volume] = {}  # Set volume as the key
            thermo = self.thermo_one_struc_scale(formula, mpid, scale)
            Fs = [thermo.get_helmholtz_energy(temperature=T) for T in temperatures]
            Fs[0] = self.E0
            out[volume]['data'] = [
                {"T": temperatures[i], "F": Fs[i]} for i in range(len(temperatures))
            ]

        return out
    
    def gibbs_one_struc(self, formula, mpid, eos="vinet"):
            """
            Returns:
                {'data' :
                    [{'T' : temperature (K),
                    'G' : Gibbs free energy (eV/cell)}]
                }
            """
            # fjson = self.fjson_gibbs
            # if not self.remake_gibbs and os.path.exists(fjson):
            #     return read_json(fjson)
            temperatures = self.temperatures
            volumes = self.volumes
            Fs = self.helmholtz_one_struc(formula, mpid)
            Gs = []
            for i in range(len(temperatures)):
                T = temperatures[i]
                F = [Fs[vol]["data"][i]["F"] for vol in volumes]
                V = [float(vol) for vol in volumes]

                if eos == "vinet":
                    eos = Vinet(V, F)
                elif eos == "murnaghan":
                    eos = Murnaghan(V, F)

                try:
                    eos.fit()
                    min_F = eos.e0
                    Gs.append({"T": T, "G": min_F})
                except:
                    print(f"Failed to fit {eos} EOS at T = {T} K")
                    continue
            out = {"data": Gs}
            # write_json(out, fjson)
            return out


    def get_phonopy_qha(self, formula, mpid):
        """
        Get the cached PhonopyQHA object for a specific formula and mpid.
        """
        key = (formula, mpid)
        if key not in self._qha_cache:
            self._qha_cache[key] = self.phonopy_qha_for_one_struc(formula, mpid)
        return self._qha_cache[key]

    def phonopy_qha_for_one_struc(self, formula, mpid):
        """
        Returns:
            PhonopyQHA object for a specific formula and mpid.
        """
        eos = self.eos
        volumes = self.volumes[formula, mpid]
        properties = self.properties_for_one_struc(formula, mpid)
        temperatures = sorted([item['T'] for item in properties[volumes[0]]['data']])

        free_energies = []
        entropy = []
        cv = []
        E_electronic = []

        for volume in volumes:
            data = properties[volume]['data']
            
            F_vs_T = [entry['F'] for entry in data]
            S_vs_T = [entry['S'] for entry in data]
            Cv_vs_T = [entry['Cv'] for entry in data]
            E_electronic_value = data[0]['E_electronic']

            free_energies.append(F_vs_T)
            entropy.append(S_vs_T)
            cv.append(Cv_vs_T)
            E_electronic.append(E_electronic_value)
        
        free_energies = np.array(free_energies).T
        entropy = np.array(entropy).T
        cv = np.array(cv).T

        qha = PhonopyQHA(volumes=volumes, electronic_energies=E_electronic, temperatures=temperatures, free_energy=free_energies, cv=cv, entropy=entropy, verbose=True, eos=eos)
        return qha

    def qha_info_one_struc(self, formula, mpid):
        """
        Returns:
            A list of dictionaries with QHA information for each temperature.
            e.g. [{'temperature': 300, 'equilibrium_volume': float, 'gibbs_energy': float, 'bulk_modulus': float, 'thermal_expansion': float, 'heat_capacity': float},
                {'temperature': 310, 'equilibrium_volume': float, 'gibbs_energy': float, 'bulk_modulus': float, 'thermal_expansion': float, 'heat_capacity': float}, ...]
        """
        qha = self.get_phonopy_qha(formula, mpid)
        volumes = self.volumes[(formula, mpid)]
        equilibrium_volumes = qha.volume_temperature
        gibbs_energy = qha.gibbs_temperature
        bulk_modulus = qha.bulk_modulus_temperature
        thermal_expansion = qha.thermal_expansion
        cv = qha.heat_capacity_P_polyfit
        temperatures = qha._qha._temperatures[0:-1]

        out = [{
            "temperature": temperatures[i],
            "equilibrium_volume": equilibrium_volumes[i],
            "gibbs_energy": gibbs_energy[i],
            "bulk_modulus": bulk_modulus[i],
            "thermal_expansion": thermal_expansion[i],
            "heat_capacity": cv[i]
        } for i in range(len(temperatures))]
        return out

    def qha_dict(self, write=False, data_dir=os.getcwd().replace("scripts", "data"), savename="qha.json", remake=False):
        """
        Returns:
            A dictionary with QHA information for all structures.
            e.g. {'Al1N1': {'mp-661': [list of QHA info dictionaries]}}
        """
        fjson = os.path.join(data_dir, savename)
        if not remake and os.path.exists(fjson) and write:
            return read_json(fjson)
        
        qha_dict = {}
        for formula in self.parse_results:
            qha_dict[formula] = {}
            for mpid in self.parse_results[formula]:
                qha_dict[formula][mpid] = self.qha_info_one_struc(formula, mpid)
        
        if write:
            write_json(qha_dict, fjson)
            return read_json(fjson)
        return qha_dict

    def plot_gibbs_energy(self, formula, mpid):
        """
        Plots the Gibbs free energy vs. temperature for a specific formula and mpid.
        """
        qha_info = self.qha_info_one_struc(formula, mpid)
        temperatures = [item['temperature'] for item in qha_info]
        gibbs_energy = [item['gibbs_energy'] for item in qha_info]
        
        fig = plt.figure()
        plt.plot(temperatures, gibbs_energy, color=COLORS['red'])
        plt.xlabel("Temperature (K)", fontsize=16)
        plt.ylabel("Gibbs Free Energy (eV)", fontsize=16)
        plt.title(f"Gibbs Free Energy for {formula} {mpid}", fontsize=18)
        plt.xlim(0, max(temperatures))
        plt.xticks(fontsize=14)
        plt.yticks(fontsize=14)

        return fig

    def plot_qha_info(self, formula, mpid):
        """
        Plots the QHA information for a specific formula and mpid.
        """
        qha = self.get_phonopy_qha(formula, mpid)
        qha.plot_qha()


    def plot_all(self, save=False, fig_dir=os.getcwd().replace("scripts", "figures")):
        """
        Plots the Gibbs free energy and QHA information for all structures.
        """

        results = self.parse_results
        for formula in results:
            for mpid in results[formula]:
                fig_gibbs = self.plot_gibbs_energy(formula, mpid)
                plt.show()
                if save:
                    fig_gibbs.savefig(os.path.join(fig_dir, f"{formula}_{mpid}_gibbs.png"))

                self.plot_qha_info(formula, mpid)
                plt.show()
                if save:
                    plt.savefig(os.path.join(fig_dir, f"{formula}_{mpid}_qha.png"))


    def plot_all_gibbs_for_one_cmpd(self, formula, save=False, fig_dir=os.getcwd().replace("scripts", "figures")):
        """
        Plots the Gibbs free energy for all structures of a specific compound (formula) in the same plot.
        """
        results = self.parse_results
        fig = plt.figure()
        
        # Loop through each mpid for the given formula and plot all on the same figure
        for mpid in results[formula]:
            qha_info = self.qha_info_one_struc(formula, mpid)
            temperatures = [item['temperature'] for item in qha_info]
            gibbs_energy = [item['gibbs_energy'] for item in qha_info]
            
            # Plot the Gibbs free energy for this mpid on the same figure
            plt.plot(temperatures, gibbs_energy, label=mpid, linewidth=0.5)

        # Set the plot labels, title, and legend
        plt.xlabel("Temperature (K)", fontsize=16)
        plt.ylabel("Gibbs Free Energy (eV)", fontsize=16)
        plt.title(f"Gibbs Free Energy for {formula}", fontsize=18)
        plt.xlim(0, max(temperatures))
        plt.xticks(fontsize=14)
        plt.yticks(fontsize=14)
        plt.legend(fontsize=12, loc='best')

        # Show the plot only once, after all lines have been added
        plt.show()
        
        # Save the figure if requested
        if save:
            fig.savefig(os.path.join(fig_dir, f"{formula}_all_gibbs.png"))

        return fig


    def plot_entropy_contributions(self, formula):
        """
        Plots the entropy contributions for a specific formula and mpid.
        """

        results = self.parse_results
        fig = plt.figure()
        for mpid in results[formula]:
            qha_info = self.qha_info_one_struc(formula, mpid)
            temperatures = [item['temperature'] for item in qha_info]
            gibbs_energies = [item['gibbs_energy'] for item in qha_info]
            entropy_contributions = [gibbs-gibbs_energies[0] for gibbs in gibbs_energies]
        
            plt.plot(temperatures, entropy_contributions, label=mpid)
            
        plt.xlabel("Temperature (K)", fontsize=16)
        plt.ylabel("-ST (eV)", fontsize=16)
        plt.title(f"-ST term for {formula}", fontsize=18)
        plt.legend()
        plt.xlim(0, max(temperatures))
        plt.xticks(fontsize=14)
        plt.yticks(fontsize=14)


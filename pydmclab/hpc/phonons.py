import os
import numpy as np
import matplotlib.pyplot as plt
from pydmclab.plotting.utils import get_colors, set_rc_params
from matplotlib.ticker import MaxNLocator
# from scipy.constants import hbar, e
from scipy.constants import physical_constants
from scipy.integrate import trapezoid



from pydmclab.utils.handy import read_json, write_json
from pydmclab.core.struc import StrucTools
from pydmclab.core.comp import CompTools

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
        Units: free_energy (kJ/mol), entropy (J/mol/K), heat_capacity (J/mol/K). These are PHONON free energies!
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
            {'frequency_points ': array of frequency points (THz is the deafault), 'total_dos': array of total density of states per cell per unit of the horizontal axis (THz^-1 is default)}
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



class Helmholtz():

    def __init__(self, phonon_dos, temperatures=np.linspace(0, 2000, 100), formula_units = None):
        self.phonon_dos = phonon_dos
        self.temperatures = temperatures
        self.formula_units = formula_units
        self.E0 = phonon_dos["E0"]

        """
            Args:
                phonon_dos (Usually obtained from the phonon_dos method of the QHA class, which gets it from AnalyzePhonons.total_dos, parses it and converts it to units of eV):
                    {'E0' : 0 K internal energy (eV/cell),
                    'dos' :
                        [{'E' : energy level (eV),
                        'dos' : phonon DOS at E (float)}]
                        }

                temperatures (np.array)
                    list of temperatures (K)

                formula_units (int)
                    number of formula units in the cell. If None is given, the energy will be in eV/cell. 
                    If a number is given, the energy will be in eV/formula unit.
        """
        

    def thermo(self):
        phonon_dos = self.phonon_dos
        E0 = self.E0
        formula_units = self.formula_units

        phonon_energies = np.array([
            phonon_dos["dos"][i]["E"] for i in range(len(phonon_dos["dos"]))
        ])
        phonon_dos_values = np.array([
            phonon_dos["dos"][i]["dos"] for i in range(len(phonon_dos["dos"]))
        ])
        return CrystalThermo(
            phonon_energies=phonon_energies,
            phonon_DOS=phonon_dos_values,
            potentialenergy=self.E0,
            formula_units=formula_units,
        )

    def zero_point_energy(self):
        phonon_dos = self.phonon_dos
        formula_units = self.formula_units
        E0 = self.E0
        phonon_energies = np.array([
            phonon_dos["dos"][i]["E"] for i in range(len(phonon_dos["dos"]))
        ])
        dos_values = np.array([
            phonon_dos["dos"][i]["dos"] for i in range(len(phonon_dos["dos"]))
        ])
        zpe_list = phonon_energies / 2.
        zpe = trapezoid(zpe_list * dos_values, phonon_energies)
        if formula_units is None:
            return (zpe + E0)
        else:
            return (zpe + E0)/formula_units
    

    def helmholtz(self):
        """
        Returns:
            list
                [{'T': temperature (K), 'F': Helmholtz free energy (eV/cell or eV/formula unit), 'S': entropy (J/mol/K)}]
        """
        thermo = self.thermo()
        temperatures = self.temperatures
        S = [thermo.get_entropy(temperature=T) for T in temperatures]
        Fs = [thermo.get_helmholtz_energy(temperature=T) for T in temperatures]
        Fs[0] = self.zero_point_energy()
        out = {
            "data": [
                {"T": temperatures[i], "F": Fs[i], "S": S[i]} for i in range(len(temperatures))
            ]
        }
        return out    
    
class Gibbs():
    def __init__(self, phonon_dos_dict, eos="vinet", temperatures=np.linspace(0, 2000, 100), formula_units = None):
        self.eos = eos
        self.phonon_dos_dict = phonon_dos_dict
        self.temperatures = temperatures
        self.formula_units = formula_units

        '''
        Args:
            phonon_dos_dict (dict):
                {volume (float) :
                    {'E0' : 0 K internal energy (eV/cell),
                    'dos' :
                        [{'E' : energy level (eV),
                        'dos' : phonon DOS at E (float)}]
                    }
                }
            eos (str):
                equation of state to use for fitting the data. Default is "vinet".
            temperatures (np.array):
                list of temperatures (K)
            formula_units (int):
                number of formula units in the cell. If None is given, the energy will be in eV/cell. 
                If a number is given, the energy will be in eV/formula unit.
        '''

    @property
    def volumes(self):
        return list(self.phonon_dos_dict.keys())
    
    def helmholtz(self):
        """
        Returns:
            dict
                {volume (A**3) :
                    {'data' :
                        [{'T' : temperature (K),
                        'F' : Helmholtz free energy (eV/cell or eV/formula unit)},
                        'S' : entropy (eV/formula unit/K)]
                    }
                }
        """
        volumes = self.volumes
        dos = self.phonon_dos_dict
        formula_units = self.formula_units
        return {
            volumes[i]: Helmholtz(
                phonon_dos=dos[volumes[i]],
                temperatures=self.temperatures,
                formula_units=formula_units
            ).helmholtz()
            for i in range(len(volumes))
        }
    
    def gibbs(self):
        """
        Returns:
            dict
                {volume (A**3) :
                    {'data' :
                        [{'T' : temperature (K),
                        'G' : Gibbs free energy (eV/cell or eV/formula unit)}]
                    }
                }
        """
        eos = self.eos
        volumes = self.volumes
        temperatures = self.temperatures
        Fs = self.helmholtz()
        Gs = []
        for i in range(len(temperatures)):
            T = temperatures[i]
            F = [Fs[vol]["data"][i]["F"] for vol in volumes]
            print(f"T = {T}, F = {F}")
            V = [float(vol) for vol in volumes]
            print(V)


            if eos == "vinet":
                current_eos = Vinet(V, F)
            elif eos == "murnaghan":
                current_eos = Murnaghan(V, F)
            else:
                print(f"Invalid EOS: {eos}. Using Vinet EOS instead.")
                current_eos = Vinet(V, F)

            try:
                current_eos.fit()
                min_F = current_eos.e0
                Gs.append({"T": T, "G": min_F})
                print(f"T = {T}, eos.e0 = {current_eos.e0}, parameters = {current_eos.eos_params}")
                
            except Exception as e:
                print(f"Failed to fit {eos} EOS at T = {T} K: {e}")
                continue

        return {"data": Gs}
    
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
            

            static_key = "--".join(key.split("--")[:-1] + [f"{xc}-static"])
            if static_key not in results:
                print(f"Warning: Static key {static_key} not found in results. Skipping.")
                continue
            
            structure = results[static_key]['structure']
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
    def temperatures(self):
        """
        Returns:
            list: A list of temperatures (K) from the thermal properties data
        """
        temperatures = np.linspace(0, 2000, 201)
        return temperatures
    
    def formula_units(self, formula, mpid):
        """
        Returns:
            int: Number of formula units in the cell.
        """
        props = self.parse_results[formula][mpid]
        random_scale = list(props.keys())[0]
        structure = props[random_scale]['structure']
        st = StrucTools(structure)
        comp = st.structure.composition
        ct = CompTools(comp)
        _, formula_units = ct.get_reduced_comp_and_factor()
        return formula_units


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
                                        'dos' : phonon DOS at E (float) (normalized to 1/eV)]}]
                                    }
                }
        """
        results = self.parse_results
        dos_data = {}

        for formula in results:
            for mpid in results[formula]:
                dos_data[(formula, mpid)] = {}
                for scale, data in results[formula][mpid].items():
                    phonons_data = data['phonons']['total_dos']
                    frequency_points = np.array(phonons_data.get('frequency_points'))
                    total_dos = np.array(phonons_data.get('total_dos'))
                    struc = data['structure']
                    st = StrucTools(struc)
                    vol = st.structure.volume

                    if frequency_points is not None and total_dos is not None:
                        # Filter out imaginary frequencies with zero DOS
                        valid_indices = [
                            i for i, freq in enumerate(frequency_points)
                            if np.real(freq) > 0 or (np.real(freq) <= 0 and total_dos[i] != 0)
                        ]
                        filtered_frequencies = frequency_points[valid_indices]
                        filtered_dos = total_dos[valid_indices]/0.0041356655 #This is to normalize the DOS to 1/eV (phonopy default is 1/THz)

                        # Warning for removed imaginary frequencies
                        n_removed = len(frequency_points) - len(filtered_frequencies)
                        if n_removed > 0:
                            print(
                                f"Warning: Removed {n_removed} imaginary frequencies with zero DOS for "
                                f"formula {formula}, mpid {mpid}, scale {scale}"
                            )
                        
                        h = physical_constants['Planck constant in eV/Hz'][0]
                        hbar = h / (2 * np.pi)
                        # Convert frequencies to energy in eV
                        energy_points = filtered_frequencies * 10e12 * hbar  # in eV (output from phonopy is in THz)

                        # Store the processed DOS data
                        dos_data[(formula, mpid)][str(vol)] = {
                            'E0': data['E_electronic'],
                            'dos': [{'E': E, 'dos': d} for E, d in zip(energy_points, filtered_dos)]
                        }
                    else:
                        print(f"Warning: Missing 'frequency_points' or 'total_dos' for formula {formula}, mpid {mpid}, scale {scale}")

        return dos_data
    
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
        temperatures = self.temperatures
        phonon_dos = self.phonon_dos[(formula, mpid)]
        formula_units = self.formula_units(formula, mpid)

        F = Gibbs(phonon_dos, eos=self.eos, temperatures=temperatures, formula_units=formula_units).helmholtz()

        return F
    
    def gibbs_one_struc(self, formula, mpid, eos="vinet"):
            """
            Returns:
                {'data' :
                    [{'T' : temperature (K),
                    'G' : Gibbs free energy (eV/cell)}]
                }
            """
            temperatures = self.temperatures
            volumes = self.volumes[formula, mpid]

            phonon_dos = self.phonon_dos[formula, mpid]
            formula_units = self.formula_units(formula, mpid)
            G = Gibbs(phonon_dos, eos=eos, temperatures=temperatures, formula_units=formula_units).gibbs()
            return G

    def qha_dict(self, write=False, data_dir=os.getcwd().replace("scripts", "data"), savename="qha.json", remake=False):
        """
        Returns:
            A dictionary with QHA information for all structures.
            e.g. {'Al1N1': {'mp-661': [list of QHA info dictionaries]}}
        """
        fjson = os.path.join(data_dir, savename)
        if not remake and os.path.exists(fjson) and write:
            return read_json(fjson)
        
        results = self.parse_results
        qha_dict = {}
        for formula in results  :
            qha_dict[formula] = {}
            for mpid in results[formula]:
                F = self.helmholtz_one_struc(formula, mpid)
                G = self.gibbs_one_struc(formula, mpid)
                qha_dict[formula][mpid] = {"F": F, "G": G}    

        if write:
            write_json(qha_dict, fjson)
            return read_json(fjson)
        return qha_dict

    def plot_helmholtz_free_energy(self, formula, mpid, temp_cutoff=None):
        """
        Plot Temperature vs Helmholtz Free Energy at Different Volumes.

        Args:
            F (dict): Dictionary containing Helmholtz free energy data for different volumes.
            temp_cutoff (tuple): Optional temperature range (min_temp, max_temp) for filtering.
        """

        F = self.helmholtz_one_struc(formula, mpid)

        plt.figure(figsize=(10, 6))  

        volumes = list(F.keys())  

        for vol in volumes:
            # Extract Helmholtz free energies and temperatures
            data = F[vol]['data']
            if temp_cutoff:
                data = [d for d in data if temp_cutoff[0] <= d['T'] <= temp_cutoff[1]]

            Fs = [i['F'] for i in data]
            Ts = [i['T'] for i in data]

            plt.plot(Ts, Fs, label=f"{vol:.2f}")  # Format volume label to 2 decimals

        # Add title and axis labels
        plt.title("Temperature vs Helmholtz Free Energy at Different Volumes", fontsize=14)
        plt.xlabel("Temperature (K)", fontsize=12)
        plt.ylabel("Helmholtz Free Energy (eV/f.u.)" if self.formula_units else "Helmholtz Free Energy (eV/cell)", fontsize=12)

        # Add legend and grid
        plt.legend(title="Volumes", loc="best", fontsize=10)
        plt.grid(True)

        # Display the plot
        plt.show()

    def plot_gibbs_free_energy(self, formula, mpid, temp_cutoff=None):
        """
        Plot Temperature vs Gibbs Free Energy at Different Volumes.

        Args:
            G (dict): Dictionary containing Gibbs free energy data.
            volumes (list): List of volume values.
            temp_cutoff (tuple): Optional temperature range (min_temp, max_temp) for filtering.
        """
        G = self.gibbs_one_struc(formula, mpid)

        plt.figure(figsize=(10, 6))  # Create a new figure with a specified size

        data = G['data']
        if temp_cutoff:
            data = [d for d in data if temp_cutoff[0] <= d['T'] <= temp_cutoff[1]]

        Gs = [i['G'] for i in data]
        Ts = [i['T'] for i in data]

        plt.plot(Ts, Gs, label=f"{formula}, {mpid}") 

        # Add title and axis labels
        plt.title("Temperature vs Gibbs Free Energy at Different Volumes", fontsize=14)
        plt.xlabel("Temperature (K)", fontsize=12)
        plt.ylabel("Gibbs Free Energy (eV/f.u.)" if self.formula_units else "Gibbs Free Energy (eV/cell)", fontsize=12)

        # Add legend and grid
        plt.legend(title="Volumes", loc="best", fontsize=10)
        plt.grid(True)

        # Display the plot
        plt.show()
    
    def plot_volumes_vs_helmholtz(self, formula, mpid, skip=1, temp_cutoff=None, normalize_298K=False):
        """
        Plot Helmholtz Free Energy vs Volume at different temperatures.
        Optionally subtract the energy at 298K for normalization.

        Args:
            F (dict): Dictionary containing Helmholtz free energy data for different volumes.
            skip (int): Step size for skipping temperatures. Default is 1 (no skipping).
            temp_cutoff (tuple): Optional temperature range (min_temp, max_temp) for filtering.
            normalize_298K (bool): If True, subtract the energy at 298K for normalization. Default is True.
        """
        F = self.helmholtz_one_struc(formula, mpid)
        G = self.gibbs_one_struc(formula, mpid)
        # Extract all temperature points (assuming consistent across volumes)
        temperatures = [entry['T'] for entry in next(iter(F.values()))['data']]
        if temp_cutoff:
            temperatures = [T for T in temperatures if temp_cutoff[0] <= T <= temp_cutoff[1]]

        plt.figure(figsize=(10, 6))  # Prepare the figure


        # Loop over temperatures with the specified skip step
        for T in temperatures[1::skip]: 
            print(f"Temperature: {T} K")
            vols = []  # List to store volumes
            Fs = []  # List to store free energy values
            for vol, data in F.items():
                # Find the corresponding F for the current temperature
                for entry in data['data']:
                    if entry['T'] == T:
                        F_value = entry['F']
                        if normalize_298K:
                            G_at_T300 = next(item['G'] for item in G['data'] if item['T'] == 300)
                            F_value -= G_at_T300
                            # print(F_value)
                        vols.append(vol)
                        Fs.append(F_value)
                        break

            # Plot F vs Volume for the current temperature
            plt.plot(vols, Fs, marker='o', label=f"T = {T:.1f} K")

        # Add axis labels and title
        plt.xlabel("Volume ($\mathrm{Å}^3$)", fontsize=12)
        plt.ylabel("Helmholtz Free Energy (eV/f.u.)", fontsize=12)
        plt.title(
            "Helmholtz Free Energy vs Volume at Different Temperatures" +
            (" (Normalized to 298K)" if normalize_298K else ""),
            fontsize=14
        )

        # Add legend
        plt.legend(title="Temperatures", loc="best", fontsize=10)

        # Add grid for better readability
        plt.grid(True)

        # Show the plot
        plt.show()

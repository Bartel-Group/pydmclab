import os
import numpy as np
import matplotlib.pyplot as plt
from pydmclab.plotting.utils import get_colors, set_rc_params, get_label
from matplotlib.ticker import MaxNLocator
# from scipy.constants import hbar, e
from scipy.constants import physical_constants
from scipy.integrate import trapezoid



from pydmclab.utils.handy import read_json, write_json
from pydmclab.core.struc import StrucTools
from pydmclab.core.comp import CompTools
from pymatgen.core.composition import Composition

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
                 mesh: int|list|float=[30, 30, 30]):
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

    # def band_structure(
    #     self,
    #     paths: list =[
    #         [[0.0, 0.0, 0.0], [0.5, 0.0, 0.0]],  # Γ to X
    #         [[0.5, 0.0, 0.0], [0.5, 0.5, 0.5]],  # X to L
    #         [[0.5, 0.5, 0.5], [0.0, 0.0, 0.0]],  # L to Γ
    #         [[0.5, 0.0, 0.0], [0.5, 0.25, 0.75]],  # X to W
    #     ],
    # ):
    #     """
    #     Returns the band structure for the phonon object in a dictionary
    #     Args:
    #         paths (list):
    #             List of paths in reciprocal space. e.g. [[[0.0, 0.0, 0.0], [0.5, 0.0, 0.0]], [[0.5, 0.0, 0.0], [0.5, 0.5, 0.5]], ...]

    #     Returns:
    #         {'qpoints': arrays of q points, 'distances': arrays of distances, 'frequencies': arrays of frequencies, 'eigenvectors': arrays of eigenvectors, group_velocities': arrays of group velocities}
    #     """
    #     if not hasattr(self, '_band_structure') or self._band_structure_paths != paths:
    #         print("Calculating band structure...")
    #         _ = self.phonon.run_band_structure(paths)
    #         self._band_structure = self.phonon.get_band_structure_dict()
    #         self._band_structure_paths = paths  
        
    #     return self._band_structure

    def band_structure(
            self,
            paths: list = [
                [[0.0, 0.0, 0.0], [0.5, 0.0, 0.0]],  # Γ to X
                [[0.5, 0.0, 0.0], [0.5, 0.5, 0.5]],  # X to L
                [[0.5, 0.5, 0.5], [0.0, 0.0, 0.0]],  # L to Γ
                [[0.5, 0.0, 0.0], [0.5, 0.25, 0.75]],  # X to W
            ],
            npoints: int = 51,  # Number of interpolation points per path segment
        ):
            """
            Returns the band structure for the phonon object in a dictionary with interpolated paths
            
            Args:
                paths (list):
                            List of paths in reciprocal space. Each path is defined by start and end points at high symmetry points. 
                                e.g. [[[0.0, 0.0, 0.0], [0.5, 0.0, 0.0]], [[0.5, 0.0, 0.0], [0.5, 0.5, 0.5]], ...]
                npoints (int): 
                            Number of points to interpolate along each path segment
                
            Returns:
                dict: {'qpoints': arrays of q points, 'distances': arrays of distances, 
                    'frequencies': arrays of frequencies, 'eigenvectors': arrays of eigenvectors, 
                    'group_velocities': arrays of group velocities}
            """
            # Create a cache key that includes both paths and npoints
            cache_key = (tuple(tuple(tuple(point) for point in path) for path in paths), npoints)
            
            if not hasattr(self, '_band_structure') or self._band_structure_cache_key != cache_key:
                print("Calculating band structure...")
                
                # Method 1: Use the same interpolation approach as hiPhive example
                interpolated_paths = []
                for path in paths:
                    start_point = np.array(path[0])
                    end_point = np.array(path[1])
                    # Linear interpolation between start and end points
                    interpolated_path = np.array([
                        start_point + (end_point - start_point) * i / (npoints - 1) 
                        for i in range(npoints)
                    ])
                    interpolated_paths.append(interpolated_path)
                
                # Run band structure calculation with interpolated paths
                _ = self.phonon.run_band_structure(interpolated_paths)
                self._band_structure = self.phonon.get_band_structure_dict()
                self._band_structure_cache_key = cache_key
                
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
                phonon_dos (Usually obtained from the phonon_dos method of the QHA class, 
                which gets it from AnalyzePhonons.total_dos, parses it and converts it to units of eV):
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
                        'G' : Gibbs free energy (eV/cell or eV/formula unit)},
                        'V' : equilibrium volume (A**3)]
                    }
                }
        """
        eos = self.eos
        volumes = self.volumes
        temperatures = self.temperatures
        Fs = self.helmholtz()
        Gs = []
        fitted_F_values = {}
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
                V_eq = current_eos.v0
                Gs.append({"T": T, "G": min_F, "V": V_eq})
                print(f"T = {T}, eos.e0 = {current_eos.e0}, parameters = {current_eos.eos_params}")

                # Generate a smooth range of volumes
                smooth_volumes = np.linspace(min(V), max(V), 500) 
                fitted_F_values[T] = current_eos.func(smooth_volumes) 

            except Exception as e:
                print(f"Failed to fit {eos} EOS at T = {T} K: {e}")
                continue

        return {"data": Gs, "fitted_F_values": fitted_F_values, "volumes_for_fitting": smooth_volumes}
    
class QHA(object):
    def __init__(self, results: dict, temperatures = np.linspace(0, 2000, 201), eos: str = "vinet"):
        self.results = results
        self.temperatures = temperatures
        self.eos = eos
        self._parsed_results = None  # Cache for parsed results

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
            if phonon_data is None:
                print(f"Warning: Phonon data not found for key {key}. Skipping.")
                continue

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
    
    # @property
    # def temperatures(self):
    #     """
    #     Returns:
    #         list: A list of temperatures (K) from the thermal properties data
    #     """
    #     return self.temperatures
    
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
        reduced_comp_and_factor = Composition(
                comp
                ).get_reduced_composition_and_factor()
        formula_units = reduced_comp_and_factor[1]
        return formula_units


    def phonon_dos(self, remove_imaginary=True, move_imaginary=False):
        """
        Args:
            remove_imaginary (bool): If True, remove all DOS values and frequencies where there are imaginary frequencies. 
                                    If False, remove only imaginary frequencies where DOS values are zero.
            move_imaginary (bool): If True, move imaginary frequencies to the real axis by taking the absolute value of the frequency.
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
                        if remove_imaginary:
                            # Remove all DOS values and frequencies where there are imaginary frequencies
                            valid_indices = [
                                i for i, freq in enumerate(frequency_points)
                                if np.real(freq) > 0
                            ]
                        else:
                            # Remove only imaginary frequencies where DOS values are zero
                            valid_indices = [
                                i for i, freq in enumerate(frequency_points)
                                if np.real(freq) > 0 or (np.real(freq) <= 0 and total_dos[i] != 0)
                            ]

                        filtered_frequencies = frequency_points[valid_indices]
                        if move_imaginary:
                            filtered_frequencies = np.abs(filtered_frequencies)

                        filtered_dos = total_dos[valid_indices]/0.0041356655 #This is to normalize the DOS to 1/eV (phonopy default is 1/THz)

                        # Warning for removed imaginary frequencies
                        n_removed = len(frequency_points) - len(filtered_frequencies)
                        if n_removed > 0:
                            print(
                                f"Warning: Removed {n_removed} imaginary frequencies with zero DOS for "
                                f"formula {formula}, mpid {mpid}, scale {scale}" if not remove_imaginary
                                else f"Warning: Removed {n_removed} imaginary frequencies for " 
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

    def helmholtz_one_struc(self, formula, mpid, remove_imaginary=True):
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
        phonon_dos = self.phonon_dos(remove_imaginary=remove_imaginary)[(formula, mpid)]
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

            phonon_dos = self.phonon_dos(remove_imaginary=True)[formula, mpid]
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

    def plot_phonon_dos(self, formula, mpid, volume=None, remove_imaginary=False):
        """
        Plot phonon density of states for a specific volume.
        Args:
            formula (str): Chemical formula of the material.
            mpid (str): Materials Project ID.
            volume (float): Volume of the structure. If none, will plot phonon dos for all the volumes.
            remove_imaginary (bool): Whether to remove imaginary frequencies. Default is False.
        """
        phonon_dos_dict = self.phonon_dos(remove_imaginary=remove_imaginary)[(formula, mpid)]
        
        if not volume:
            plt.figure(figsize=(10, 6))
            for volume in phonon_dos_dict:
                phonon_dos = phonon_dos_dict[str(volume)]

                frequency_points = np.array([d['E'] for d in phonon_dos['dos']])
                total_dos = np.array([d['dos'] for d in phonon_dos['dos']])
                label = f"{mpid} - {float(volume):.2f} A^3"
                plt.plot(frequency_points, total_dos, label=label)

            plt.title(f"Phonon Density of States for {mpid}", fontsize=14)
            plt.legend(title="Volumes", loc="best", fontsize=10)

        else:
            phonon_dos = phonon_dos_dict[str(volume)]

            frequency_points = np.array([d['E'] for d in phonon_dos['dos']])
            total_dos = np.array([d['dos'] for d in phonon_dos['dos']])

            plt.figure(figsize=(10, 6))
            plt.plot(frequency_points, total_dos, label=f"{mpid} - {volume:.2f} A^3")
            plt.title(f"Phonon Density of States for {mpid} - {volume:.2f} A^3", fontsize=14)
            
        plt.xlabel("Energy (eV)", fontsize=12)
        plt.ylabel("Phonon DOS (1/eV)", fontsize=12)

    
   
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

            plt.plot(Ts, Fs, label=f"{float(vol):.2f}")  # Format volume label to 2 decimals

        # Add title and axis labels
        # plt.title("Temperature vs Helmholtz Free Energy at Different Volumes", fontsize=14)
        plt.xlabel("Temperature (K)")
        plt.ylabel("Helmholtz Free Energy (eV/f.u.)" if self.formula_units else "Helmholtz Free Energy (eV/cell)")

        # Add legend and grid
        plt.legend(title="Volumes", loc="best", fontsize=10)


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

        plt.plot(Ts, Gs, label=mpid) 

        # Add title and axis labels
        # plt.title("Temperature vs Gibbs Free Energy at Different Volumes", fontsize=14)
        plt.xlabel("Temperature (K)")
        plt.ylabel("Gibbs Free Energy (eV/f.u.)" if self.formula_units else "Gibbs Free Energy (eV/cell)")

        # Add legend and grid
        # plt.legend(title="Volumes", loc="best", fontsize=10)
        # plt.grid(True)

        # Display the plot
        plt.show()
    
    def plot_volumes_vs_helmholtz(self, formula, mpid, skip=1, temp_cutoff=None, normalize_298K=False):
        """
        Plot Helmholtz Free Energy vs Volume at different temperatures.
        Optionally subtract the energy at 298K for normalization.
        """
        F = self.helmholtz_one_struc(formula, mpid)
        G = self.gibbs_one_struc(formula, mpid)

        # Extract all temperature points (assuming consistent across volumes)
        temperatures = [entry['T'] for entry in next(iter(F.values()))['data']]
        if temp_cutoff:
            temperatures = [T for T in temperatures if temp_cutoff[0] <= T <= temp_cutoff[1]]

        plt.figure(figsize=(3, 6))  # Prepare the figure

        equil_vols = []  # List to store equilibrium volumes
        equil_F_values = []  # List to store F value (Gibbs) at equilibrium volume

        # Loop over temperatures with the specified skip step
        for i, T in enumerate(temperatures[1::skip]): 
            vols = []  # List to store volumes
            Fs = []  # List to store free energy values
            equil_vol = next(item['V'] for item in G['data'] if item['T'] == T)  # Get the equilibrium volume
            equil_vol_F = next(item['G'] for item in G['data'] if item['T'] == T)  # Get the Gibbs free energy at the equilibrium volume

            equil_vols.append(equil_vol)  # Add equilibrium volumes
            equil_F_values.append(equil_vol_F)  # Add equilibrium Gibbs free energies

            for vol, data in F.items():
                for entry in data['data']:
                    if entry['T'] == T:
                        F_value = entry['F']
                        if normalize_298K:
                            G_at_T300 = next(item['G'] for item in G['data'] if item['T'] == 300)
                            F_value -= G_at_T300
                        vols.append(float(vol))
                        Fs.append(F_value)
                        break

            # Plot Helmholtz Free Energy vs Volume for the current temperature
            plt.scatter(vols, Fs, marker='o', color='black')
                    # Plot the smooth fitted line for this temperature
            if T in G['fitted_F_values']:
                plt.plot(G['volumes_for_fitting'], G['fitted_F_values'][T], color='black')


        print("Equilibrium Volumes: ", equil_vols)
        print("Equilibrium Free Energy Values: ", equil_F_values)

        # Now plot the equilibrium volumes and Gibbs free energies at each temperature
        plt.plot(equil_vols, equil_F_values, color='red', marker='x', label="Equilibrium Volume")

        # # Add legend for first and last temperature
        # plt.text(484, -30.8, f"T = {temperatures[1::skip][0]} K", fontsize=14,
        #          color='black', ha='center', va='center')
        # plt.text(448, -37.8, f"T = {temperatures[1::skip][-1]} K", fontsize=14,
        #          color='black', ha='center', va='center')

        plt.xlabel("Volume ($\mathrm{Å}^3$)", fontsize=18)
        plt.ylabel("F (eV/f.u.)", fontsize=18)
        plt.yticks(fontsize=14)
        plt.xticks(fontsize=14)
        # plt.title(
        #     "Helmholtz Free Energy vs Volume at Different Temperatures" +
        #     (" (Normalized to 298K)" if normalize_298K else ""),
        #     fontsize=14
        # )

        # Show the plot
        plt.show()


    def plot_equilibrium_volume_vs_temperature(self, formula, mpid, temp_cutoff=None):
        """
        Plot Equilibrium Volume vs Temperature.

        Args:
            G (dict): Dictionary containing Gibbs free energy data.
            volumes (list): List of volume values.
            temp_cutoff (tuple): Optional temperature range (min_temp, max_temp) for filtering.
        """
        G = self.gibbs_one_struc(formula, mpid)

        plt.figure(figsize=(2, 6))

        data = G['data']
        if temp_cutoff:
            data = [d for d in data if temp_cutoff[0] <= d['T'] <= temp_cutoff[1]]
        
        Ts = [i['T'] for i in data]
        Vs = [i['V'] for i in data]

        plt.plot(Ts, Vs, label=mpid)
        # plt.title("Equilibrium Volume vs Temperature", fontsize=14)
        plt.xlabel("Temperature (K)", fontsize=12)
        plt.ylabel("Equilibrium Volume ($\mathrm{Å}^3$)", fontsize=12)
        plt.xticks(fontsize=12)
        plt.yticks(fontsize=12)
        # plt.legend(loc="best", fontsize=10)


    def plot_relative_gibbs(self, formula=None, mpids=None, temp_cutoff=None, experimental=None):
        """
        Plot Relative Gibbs Free Energy (compared to the ground state) vs Temperature.

        Args:
            formula (str): Formula to filter MPIDs by. If None, uses all available formulas.
            mpids (list): Specific MPIDs to compare. If None, uses all MPIDs under the given formula.
            temp_cutoff (tuple): Optional temperature range (min_temp, max_temp) for filtering.
        """
        # Get phonon DOS data
        p_dos = self.phonon_dos(remove_imaginary=False, move_imaginary=True)

        # Filter by formula and MPIDs
        if formula:
            relevant_keys = [key for key in p_dos if key[0] == formula]
        else:
            relevant_keys = list(p_dos.keys())

        if mpids:
            relevant_keys = [key for key in relevant_keys if key[1] in mpids]

        plt.figure(figsize=(8, 4))
        # Collect Gibbs free energy data
        gibbs_data = {}
        for key in relevant_keys:
            formula, mpid = key
            G = self.gibbs_one_struc(formula, mpid)
            data = G['data']
            Ts = [i['T'] for i in data]
            Gs = [i['G'] for i in data]

            # Apply temperature cutoff if specified
            if temp_cutoff:
                filtered_data = [(T, G) for T, G in zip(Ts, Gs) if temp_cutoff[0] <= T <= temp_cutoff[1]]
                Ts, Gs = zip(*filtered_data) if filtered_data else ([], [])

            gibbs_data[mpid] = {'T': Ts, 'G': Gs}

        # Ensure there are at least two datasets for comparison
        if len(gibbs_data) < 2:
            print("Error: At least two MPIDs are required for comparison.")
            return

        # Determine the ground state (lowest energy at 0K)
        ground_state_mpid = min(
            gibbs_data.keys(), key=lambda mpid: gibbs_data[mpid]['G'][0] if gibbs_data[mpid]['G'] else float('inf')
        )

        ground_state_data = gibbs_data.pop(ground_state_mpid)
        T_ref, G_ref = ground_state_data['T'], ground_state_data['G']

        fig = plt.figure(figsize=(10, 6))
        # Ensure temperature points align
        for mpid, data in gibbs_data.items():
            if data['T'] != T_ref:
                print(f"Error: Temperature points do not align for MPID {mpid}. Skipping.")
                continue

            # Calculate relative Gibbs free energy
            G_diff = [g - g_ref for g, g_ref in zip(data['G'], G_ref)]

            # Plot the difference
            label = f"$\Delta G: |G_{{{ground_state_mpid.split('_')[-1]}}}| - |G_{{{mpid.split('_')[-1]}}}|$"

            plt.plot(
                T_ref, G_diff, label=label, color=COLORS['black']
            )
            
            # Find the temperature where G_diff crosses 0 (ΔG = 0)
            for i in range(1, len(G_diff)):
                if (G_diff[i-1] > 0 and G_diff[i] < 0) or (G_diff[i-1] < 0 and G_diff[i] > 0):
                    # Interpolate between the points to get the temperature where ΔG = 0
                    T_cross = T_ref[i-1] + (0 - G_diff[i-1]) * (T_ref[i] - T_ref[i-1]) / (G_diff[i] - G_diff[i-1])
                    plt.vlines(T_cross, ymin=min(G_diff), ymax=max(G_diff), linestyle='--', color=COLORS['blue'], label="Computed transition T")

        if experimental:
            plt.vlines(experimental, ymin=min(G_diff), ymax=max(G_diff), linestyles='--', color=COLORS['red'], label="Experimental transition T")

        # Customize the plot
        # plt.title(
        #     f"Relative Gibbs Free Energy vs Temperature\n(Reference: {ground_state_mpid})",
        #     fontsize=14,
        # )
        plt.xlabel("Temperature (K)", fontsize=18)
        plt.ylabel("ΔG (eV/f.u.)", fontsize=18)
        plt.xticks(fontsize=14)
        plt.yticks(fontsize=14)
        plt.axhline(0, color="black", linestyle="--", linewidth=0.8)
        plt.legend(loc="best", fontsize=14, facecolor='white', edgecolor='black', frameon=True, framealpha=1)
        # plt.grid(True)

        # Display the plot
        plt.show()

def get_cations():
    return {
        "A_site": ["Ba", "Sr", "Ca", "Mg", "Cu", "Sn", "Pb"],
        "B_site": ["Pb", "Sn", "V", "Ti", "Nb", "Zr", "Hf"],
    }

def sort_elements(compound, two_component=True):
    '''Function to sort elements based on A_site, B_site, and Sulfur. Cations are provided in the get_cations function.
    Args:
        compound (str): Chemical formula of the compound.
        two_component (bool): Default is True.
            Whether the list of sorted elements has two or three instances in the special case of a compound that only contains two elements.
            e.g. Sn2S3 could be sorted as ['Sn', 'Sn', 'S'] or ['Sn', 'S'].

    Returns:
        list: List of sorted elements with A_site element first, B_site element second, and Sulfur last.
    '''

    cations = get_cations()
    A_site = cations["A_site"]
    B_site = cations["B_site"]

    # Extract elements and element count
    elements = CompTools(compound).els
    n_els = CompTools(compound).n_els

    # Extract the non-S elements to check for ambiguous cases
    non_s_elements = [el for el in elements if el != 'S']

    if not two_component:
        if n_els == 2:
            elements.append(non_s_elements[0]) #Hacky method of handling Sn2S3, need to find a better way to handle this

    sorted_elements = []
    for el in elements:
        if el in A_site and el in B_site:
            # If element is in both A_site and B_site, assign based on other elements
            if any(other_el in A_site for other_el in non_s_elements if other_el != el):
                sorted_elements.append((1, el))
            else:
                sorted_elements.append((0, el))
        elif el in A_site:
            sorted_elements.append((0, el))
        elif el in B_site:
            sorted_elements.append((1, el))
        elif el == 'S':
            sorted_elements.append((2, el))

    sorted_elements.sort()
    return [el for _, el in sorted_elements]
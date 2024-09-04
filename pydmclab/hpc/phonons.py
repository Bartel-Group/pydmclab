import os
import numpy as np

from pydmclab.utils.handy import read_json, write_json

from phonopy import Phonopy
from phonopy.interface.vasp import read_vasp, parse_force_constants

class AnalyzePhonons(object):
    def __init__(self, calc_dir, supercell_matrix=None, mesh=100):
        """
        Class to analyze phonon data from a VASP calculation using Phonopy, Can return force constants, dynamical matrix, mesh data, thermal properties, band structure, and total density of states.
        Args:
            calc_dir (str):
                Path to the directory containing the VASP calculation
            supercell_matrix (list):
                Supercell matrix for the phonon calculation
            mesh (array-like or float):
                Mesh numbers along a, b, c axes when array_like object is given, shape=(3,).
                When float value is given, uniform mesh is generated following VASP convention by N = max(1, nint(l * |a|^*)) where 'nint' is the function to return the nearest integer. In this case, it is forced to set is_gamma_center=True.
                Default value is 100.0.

        """

        self.calc_dir = calc_dir
        self.supercell_matrix = supercell_matrix

        if not isinstance(mesh, (list, tuple, float, int)):
            raise TypeError("Mesh should be a list, tuple, or int.")
        elif isinstance(mesh, (list, tuple)) and len(mesh) != 3:
            raise ValueError(
                "Mesh should be a list or tuple of three integers or an int."
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
    def mesh(self):
        """
        Returns the mesh data for the phonon object in a dictionary
        """
        mesh = self.phonon.get_mesh_dict()
        return mesh

    def parse_thermal_properties(self, phonopy_data):
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

        # Extract arrays from the original data
        temperatures = phonopy_data["temperatures"]
        free_energies = phonopy_data["free_energy"]
        entropies = phonopy_data["entropy"]
        heat_capacities = phonopy_data["heat_capacity"]

        # Iterate through the arrays and build the list of dictionaries
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
        t_min=0,
        t_max=2000,
        t_step=20,
        temperatures=None,
        cutoff_frequency=None,
        pretend_real=False,
        band_indices=None,
        is_projection=False,
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

        Returns parsed thermal properties in the following format:
            {T (in K) (float) :
                thermal properties as dict}}
        e.g. at 300K:
        {300 : {'temperature' : T as float}, {'free_energy' : free energy at 300K (float)}, {'entropy' : entropy at 300K (float)}, {'heat_capacity' : heat capacity at 300K (float)}}

        """
        _ = self.phonon.run_thermal_properties(
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
        if tp is None:
            print("Thermal properties could not be calculated.")
        return self.parse_thermal_properties(tp)
        # except Exception as e:
        #     print(f"Error calculating thermal properties: {e}")
        #     return None

    # def plot_thermal_properties(self):
    #     self.phonon.plot_thermal_properties()

    def band_structure(
        self,
        paths=[
            [[0.0, 0.0, 0.0], [0.5, 0.0, 0.0]],  # Γ to X
            [[0.5, 0.0, 0.0], [0.5, 0.5, 0.5]],  # X to L
            [[0.5, 0.5, 0.5], [0.0, 0.0, 0.0]],  # L to Γ
            [[0.5, 0.0, 0.0], [0.5, 0.25, 0.75]],
        ],  # X to W
    ):
        """
        Returns the band structure for the phonon object in a dictionary
        Args:
            paths (list):
                List of paths in reciprocal space

        Returns:
            {'qpoints': arrays of q points, 'distances': arrays of distances, 'frequencies': arrays of frequencies, 'eigenvectors': arrays of eigenvectors, group_velocities': arrays of group velocities}
        """

        _ = self.phonon.run_band_structure(paths)
        return self.phonon.get_band_structure_dict()

    @property
    def total_dos(self):
        """
        Returns the total density of states for the phonon object in a dictionary.
        Returns:
            {'frequency_points ': array of frequency points, 'total_dos': array of total density of states}
        """
        _ = self.phonon.run_total_dos()
        return self.phonon.get_total_dos_dict()


    def make_json_serializable(self, data):
        if isinstance(data, dict):
            return {key: self.make_json_serializable(value) for key, value in data.items()}
        elif isinstance(data, list):
            return [self.make_json_serializable(item) for item in data]
        elif isinstance(data, np.ndarray):
            return data.tolist()  # Convert NumPy arrays to lists
        else:
            return data  # Return the item as is if it's not a dict, list, or np.ndarray

# Example usage
# data = ap.band_structure or any other structure
# json_serializable_data = make_json_serializable(data)
  

    def summary(
        self,
        savename = "phonons.json",
        remake=False,
        include_force_constants=True,
        include_mesh=True,
        include_thermal_properties=True,
        include_band_structure =True,
        include_total_dos=True,
        # supercell_matrix=None,
        # mesh=None,
        paths=None, 
        temperatures=None,
        cutoff_frequency=None,
        pretend_real=None,
        band_indices=None,
        is_projection=None,
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
            supercell_matrix (list, optional):
                Supercell matrix for the phonon calculation. Default is None.
            mesh (array-like or float, optional):
                Mesh numbers along a, b, c axes when array_like object is given, shape=(3,).
                When float value is given, uniform mesh is generated following VASP convention by N = max(1, nint(l * |a|^*)) where 'nint' is the function to return the nearest integer. In this case, it is forced to set is_gamma_center=True.
                Default value is 100.0.
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
        """
        
        calc_dir = self.calc_dir
        fjson = os.path.join(calc_dir, savename)
        if os.path.exists(fjson) and not remake:
            return read_json(fjson)
        data = {}
    

        # if supercell_matrix or mesh:
        #     self.phonon = AnalyzePhonons(
        #         self.calc_dir, supercell_matrix=supercell_matrix, mesh=mesh
        #     )

        if include_force_constants:
            fc = self.force_constants
            force_constants = self.make_json_serializable(fc)
            data["force_constants"] = force_constants

        if include_mesh:
            mesh_array = self.mesh
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
                data["thermal_properties"] = self.thermal_properties(**thermal_properties_kwargs)
            else:
                data["thermal_properties"] = self.thermal_properties()


        if include_band_structure:
            if paths:
                data["band_structure"] = self.make_json_serializable(self.band_structure(paths=paths))

            data["band_structure"] = self.make_json_serializable(self.band_structure())

        if include_total_dos:
            total_dos = self.total_dos
            data["total_dos"] = self.make_json_serializable(total_dos)

        write_json(data, fjson)
        return read_json(fjson)
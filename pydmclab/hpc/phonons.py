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
from pymatgen.core.structure import Structure
from phonopy.structure.atoms import PhonopyAtoms

from phonopy import Phonopy, PhonopyQHA
from phonopy.interface.vasp import read_vasp, parse_force_constants, _get_forces_points_and_energy, check_forces

from ase.phonons import Phonons
from ase.thermochemistry import CrystalThermo

from pymatgen.io.ase import AseAtomsAdaptor
from pymatgen.analysis.eos import Murnaghan, Vinet

set_rc_params()
COLORS = get_colors(palette="tab10")

def get_forces_one_calc(
    num_atoms: int,
    calc_dir: str = None,
    use_expat: bool = True,
    verbose: bool = True,
) -> dict:
    """Parse sets of forces of files."""
    if verbose:
        print("counter (file index): ", end="")
    
    vasprun_path = os.path.join(calc_dir, "vasprun.xml")
    is_parsed = True

    with open(vasprun_path, "rb") as fp:
        forces, points, energy = _get_forces_points_and_energy(
            fp, use_expat=use_expat, filename=vasprun_path
        )

    if not forces or not points or not energy:
        is_parsed = False

    if not check_forces(forces, num_atoms, calc_dir):
        is_parsed = False

    if verbose:
        print("")

    if is_parsed:
        return {
            "forces": forces,
            "points": points,
            "supercell_energy": energy,
        }
    else:
        return {}
    
def get_set_of_forces(results, mpid, xc: str = "metagga"):
    set_of_forces = []
    for key in results:
        calc_type = key.split("--")[-1]
        if calc_type != f"{calc_type}-static":
            continue

        r_mpid = key.split("--")[1]
        mpid_minus_disp = "_".join(r_mpid.split("_")[:-1])
        if mpid_minus_disp != mpid:
            continue
            
        forces = results[key]["results"]["forces"]
        if not forces:
            print(f"Warning: No forces found for {key}. Returning None.")
            return None
        set_of_forces.append(forces)
    return set_of_forces

def get_force_constants_dfpt(calc_dir: str, savename: str = "force_constants.json", remake: bool = False):
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

    write_json(out, fjson)
    # Then make sure the collector grabs it from calc_dir and sends it to data_dir! -- Ask ChrisC, she did it for cohp calcs
    # Make sure the remake flag for results.json also remakes the forces.json file -- can possibly also ask ChrisC about this

    return read_json(fjson)

def get_force_constants_hiphive(calc_dir: str, savename: str = "force_constants.json", remake: bool = False):
    #workflow for getting force constants from a hiphive calculation
    return None #This is a placeholder, need to implement this function


class AnalyzePhonons(object):
    def __init__(self, unitcell: str|dict = None,
                 force_data: np.ndarray|list = None, 
                 supercell_matrix: list = None, 
                 mesh: int|list|float=[30, 30, 30],
                 dataset: dict = None,
):
        """
        Class to analyze phonon data from a VASP calculation using Phonopy, Can return force constants, dynamical matrix, mesh data, thermal properties, band structure, and total density of states.
        Args:
            unitcell (str or dict):
                Path to the unitcell structure file (e.g., POSCAR) or a dictionary containing the structure data.
                If a dictionary is provided, it should contain 'lattice', 'species', and 'coords' keys.
            force_data (array-like or list):
                List of forces from a set of displaced structures calculations. Shape=(N, 3*natoms) where N is the number of calculations and natoms is the number of atoms in the supercell.
                or list of force constants. Shape=(natoms*3, natoms*3).
            supercell_matrix (list):
                Supercell matrix for the phonon calculation. e.g. [[2, 0, 0], [0, 2, 0], [0, 0, 2]] for a 2x2x2 supercell.
            mesh (array-like or float):
                Mesh numbers along a, b, c axes when array_like object is given, shape=(3,).
                When float value is given, uniform mesh is generated following VASP convention by N = max(1, nint(l * |a|^*)) where 'nint' is the function to return the nearest integer. In this case, it is forced to set is_gamma_center=True.
                Default value is 100.0.

        """

        self.force_data = force_data
        self.supercell_matrix = supercell_matrix
        self.dataset = dataset

        if isinstance(mesh, (list, tuple)) and len(mesh) != 3:
            raise ValueError(
                "Mesh should be a list of three integers or an int."
            )

        struc = StrucTools(unitcell).structure

        unitcell = PhonopyAtoms(
                    symbols=[site.specie.symbol for site in struc],
                    cell=struc.lattice.matrix,
                    scaled_positions=struc.frac_coords,
                    magnetic_moments=struc.site_properties.get("magmom", None)
                )

        phonon = Phonopy(unitcell, supercell_matrix) #And also need to make sure I have this for finite displacement the way it was originally done when generating the displacements
        self.unitcell = unitcell

        # force_constants = get_force_constants_dfpt(calc_dir, savename="force_constants.json", remake=False)
        arr = np.array(force_data)
        natoms = struc.num_sites

        if arr.ndim == 3 and arr.shape[1:] == (natoms, 3):
            # Looks like forces from displacements
            print(f"Detected forces from {arr.shape[0]} displacement calculations")
            self.forces = arr
            phonon.forces = arr  # Should be assignment, not function call
            phonon.dataset = dataset if dataset is not None else phonon.dataset
            self.force_constants = phonon.force_constants

        elif arr.shape == (natoms, natoms, 3, 3):
            # Looks like force constants
            print(f"Detected force constants in compact form")
            phonon.force_constants = arr
            self.force_constants = arr
        else:
            raise ValueError(
                f"Unrecognized force_data shape: {arr.shape}\n"
                f"Expected one of:\n"
                f"  - Forces: (N_displacements, {natoms}, 3)\n"
                f"  - Force constants: ({natoms}, {natoms}, 3, 3)\n"
                f"\nBased on supercell with {natoms} atoms"
            )

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
            {'force_constants': array of force constances, 'mesh': mesh array, 'thermal_properties': dictionary of thermal properties, 'band_structure': band structure data, 'total_dos': dos data}

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
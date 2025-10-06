from pymatgen.entries.compatibility import MaterialsProject2020Compatibility
from pymatgen.entries.computed_entries import ComputedStructureEntry


def get_total_energy_correction(structure):
    """
    Compute the MP2020Compatibility correction for a structure.
    """
    params = {
        "hubbards": {
            "Co": 3.32,
            "Cr": 3.7,
            "Fe": 5.3,
            "Mn": 3.9,
            "Mo": 4.38,
            "Ni": 6.2,
            "V": 3.25,
            "W": 6.2,
        },
        "run_type": "GGA+U",
    }

    dummy_energy = 0.0
    temp_cse = ComputedStructureEntry(
        structure=structure, energy=dummy_energy, parameters=params
    )

    compat = MaterialsProject2020Compatibility(check_potcar=False)
    compat.process_entries([temp_cse])

    return temp_cse.energy

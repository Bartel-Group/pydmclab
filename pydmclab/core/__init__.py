import os
from pymatgen.core.structure import Structure


def _to_pymatgen_structure(structure: Structure | dict | str) -> Structure:
    """
    Check if the structure is a pymatgen Structure object, Structure.as_dict(), or path to structure file and return a Structure object.
    """
    if isinstance(structure, dict):
        structure = Structure.from_dict(structure)
    elif isinstance(structure, str):
        if os.path.exists(structure):
            structure = Structure.from_file(structure)
        else:
            raise ValueError(
                "The given path (string) to the structure file does not exist."
            )
    return structure

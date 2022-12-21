from pymatgen.entries.computed_entries import ComputedEntry, ComputedStructureEntry, GibbsComputedStructureEntry
from pymatgen.core.structure import Structure
from pydmc.CompTools import CompTools
from pydmc.StrucTools import StrucTools
from pydmc.VASPTools import VASPAnalysis
import os

class EntryTools(object):
    
    def __init__(self,
                 data={},
                 calc_dir=None,
                 temperature=0):
        """
        Args:
            data (dict): dictionary of data to create entry
                - expects {'formula' : formula (str),
                           'E_per_at' : energy per atom (float),
                           **optional keys}
            calc_dir (str): path to directory containing calculation to process
                - note: uses calc_dir if one is provided (ie overrides data)
            temperature (int): in Kelvin, must be in [0] + list(range(300, 2100, 100))

        """
        
        self.data = data
        self.calc_dir = calc_dir
        self.temperature = temperature

    @property
    def entry(self):
        data = self.data
        calc_dir = self.calc_dir
        
        if data and not calc_dir:
            if 'formula' not in data or not data['formula']:
                raise ValueError('No formula provided')
            if 'E_per_at' not in data or not data['E_per_at']:
                raise ValueError('No energy per atom provided')
            return ComputedEntry(composition=data['formula'],
                                 energy=data['E_per_at']*CompTools(data['formula']).n_atoms,
                                 **data)
            
        if calc_dir:
            structure = Structure.from_file(os.path.join(calc_dir, 'CONTCAR'))
            formula = StrucTools(structure).compact_formula
            n_atoms = CompTools(formula).n_atoms
            E_per_at = VASPAnalysis(calc_dir).E_per_at
            
            T = self.temperature
            
            if T == 0:
                return ComputedStructureEntry(structure=structure,
                                            energy=E_per_at*n_atoms,
                                            composition=formula,
                                            **data)
            else:
                raise NotImplementedError('need formation energy at 0 K for finite T stuff in pmg, will work on later')
            
    @property
    def mp_corrected_E_per_at(self):
        entry = self.entry
        correction_per_fu = entry.correction
        original_E_per_fu = entry.energy
        n_atoms = CompTools(entry.composition.pretty_formula).n_atoms
        return (original_E_per_fu + correction_per_fu)/n_atoms
        
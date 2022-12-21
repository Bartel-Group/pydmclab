from pymatgen.entries.computed_entries import ComputedEntry, ComputedStructureEntry
from pymatgen.entries.compatibility import MaterialsProject2020Compatibility
from pymatgen.core.structure import Structure
from pydmc.CompTools import CompTools
from pydmc.StrucTools import StrucTools
from pydmc.VASPTools import VASPAnalysis
import os

class MPCompatibilityTools(object):
    
    def __init__(self,
                 calc_dir):
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
        
        self.calc_dir = calc_dir
        if not VASPAnalysis(calc_dir).is_converged:
            raise ValueError('Calculation in %s is not converged' % calc_dir)

    @property
    def initial_entry(self):
        calc_dir = self.calc_dir
        structure = Structure.from_file(os.path.join(calc_dir, 'CONTCAR'))
        formula = StrucTools(structure).compact_formula
        n_atoms = CompTools(formula).n_atoms
        E_per_at = VASPAnalysis(calc_dir).E_per_at
        return ComputedStructureEntry(structure=structure,
                                        energy=E_per_at*n_atoms,
                                        composition=formula)
    @property
    def compatibility(self):
        return MaterialsProject2020Compatibility(compat_type='Advanced',
                                                 correct_peroxide=True,
                                                 check_potcar_hash=False,
                                                 config_file=None)
        
    @property
    def adjustments(self):
        return self.compatibility.get_adjustments(self.initial_entry)
    
    @property
    def corrected_entry(self):
        initial_entry = self.initial_entry
        return ComputedEntry(composition=initial_entry.composition,
                             energy=initial_entry.energy,
                             energy_adjustments=self.adjustments)
    
    @property
    def corrected_E_per_at(self):
        entry = self.corrected_entry
        uncorrected_E_per_at = entry.uncorrected_energy_per_atom
        correction_per_at = entry.correction_per_atom
        return uncorrected_E_per_at + correction_per_at
    
def main():
    calc_dir = '~/projects/compare_to_mp/calcs/Fe1S2/mp-1522/mp/fm/ggau-static'
    compat = MPCompatibilityTools(calc_dir=calc_dir)
    return compat

if __name__ == '__main__':
    compat = main()
    
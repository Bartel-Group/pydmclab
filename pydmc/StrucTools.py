from pymatgen.core.structure import Structure
from pydmc.CompTools import CompTools
from pymatgen.core.structure import Structure
from pymatgen.transformations.standard_transformations import OrderDisorderedStructureTransformation
from pymatgen.analysis.structure_matcher import StructureMatcher




class StrucTools(object):
    """
    Purpose: to manipulate crystal structures for DFT calculations
    """
    
    def __init__(self, structure):
        """
        Args:
            structure (Structure): pymatgen Structure object
                
        """
        
        self.s = structure
        
    @property
    def compact_formula(self):
        """
        "clean" (reduced, systematic) formula for structure
        """
        return CompTools(self.s.formula).clean
    
    @property
    def formula(self):
        """
        pretty (unreduced formula) for structure
        
        """
        return self.s.formula
    
    @property
    def els(self):
        """
        list of unique elements (str) in structure
        """
        return CompTools(self.compact_formula).els
    
    def get_ordered_structures(self):
        return 'TO DO'

            
def main():
    from pydmc.MPQuery import MPQuery

    mpq = MPQuery('***REMOVED***')
    mpid = 'mp-22584' # LiMn2O4
    #mpid = 'mp-1301329' # LiMnTiO4
    #mpid = 'mp-770495' # Li5Ti2Mn3Fe3O16
    #mpid = 'mp-772660' # NbCrO4
    #mpid = 'mp-776873' # Cr2O3
    s = mpq.get_structure_by_material_id(mpid)
    #s.make_supercell([3,3,3])
    st = StrucTools(s)
    
    #return st, st, st
    
    s_nm, s_fm = st.get_nonmagnetic_structure, st.get_ferromagnetic_structure
    
    afm_strucs = st.get_antiferromagnetic_structures
    #afm_strucs = 0
    return s_nm, s_fm, afm_strucs

if __name__ == '__main__':
    s_nm, s_fm, afm_strucs = main()
    
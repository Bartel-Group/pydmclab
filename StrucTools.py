from pymatgen.core.structure import Structure
from pymatgen.analysis.magnetism.analyzer import CollinearMagneticStructureAnalyzer
from pymatgen.analysis.magnetism.analyzer import MagneticStructureEnumerator
from CompTools import CompTools
from MPQuery import MPQuery
from pymatgen.core.structure import Structure

"""
enumeration (for site ordering + mag ordering) requires enumlib. here's how I installed it:
    - for MAC: 
        - install gfortran for MacOS (https://github.com/fxcoudert/gfortran-for-macOS/releases)
        - install xcode-select ($ xcode-select --install)
        - follow instructions on https://github.com/msg-byu/enumlib


"""
class StrucTools(object):
    
    def __init__(self, structure):
        
        self.s = structure
    
    @property
    def get_nonmagnetic_structure(self):
        
        s = self.s
        cmsa = CollinearMagneticStructureAnalyzer(s)
        return cmsa.get_nonmagnetic_structure()
    
    @property
    def get_ferromagnetic_structure(self):
        
        s = self.s
        cmsa = CollinearMagneticStructureAnalyzer(s)
        return cmsa.get_ferromagnetic_structure()
    
    @property
    def get_antiferromagnetic_structures(self):
        
        s = self.s        
        afm_strucs = MagneticStructureEnumerator(s, strategies=('ferromagnetic', 'antiferromagnetic')).ordered_structures
        return afm_strucs
    
def main():
    mpq = MPQuery('***REMOVED***')
    s = mpq.get_structure_by_material_id('mp-22584')
    #s.make_supercell([3,3,3])
    st = StrucTools(s)
    
    s_nm, s_fm = st.get_nonmagnetic_structure, st.get_ferromagnetic_structure
    
    afm_strucs = st.get_antiferromagnetic_structures
    #afm_strucs = 0
    return s_nm, s_fm, afm_strucs

if __name__ == '__main__':
    s_nm, s_fm, afm_strucs = main()
    
    
    
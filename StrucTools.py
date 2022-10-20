from msilib.schema import Feature
from pymatgen.core.structure import Structure
from pymatgen.analysis.magnetism.analyzer import CollinearMagneticStructureAnalyzer
#from pymatgen.analysis.magnetism.analyzer import MagneticStructureEnumerator
from CompTools import CompTools
from MPQuery import MPQuery
from pymatgen.core.structure import Structure
from pymatgen.transformations.standard_transformations import OrderDisorderedStructureTransformation
from pymatgen.transformations.advanced_transformations import MagOrderingTransformation
from pymatgen.analysis.structure_matcher import StructureMatcher
from pymatgen.transformations.site_transformations import ReplaceSiteSpeciesTransformation

import itertools

"""
enumeration (for site ordering + mag ordering) requires enumlib. here's how I installed it:
    - for MAC: 
        - install gfortran for MacOS (https://github.com/fxcoudert/gfortran-for-macOS/releases)
        - install xcode-select ($ xcode-select --install)
        - follow instructions on https://github.com/msg-byu/enumlib
        - ran into insurmountable OS Error related to subprocess issue exectuing makeStr.py
    - for MSI:
        - follow instructions on https://github.com/msg-byu/enumlib
        - add enumlib to path
"""
class StrucTools(object):
    
    def __init__(self, structure):
        
        self.s = structure
        
    @property
    def compact_formula(self):
        return CompTools(self.s.formula).clean
    
    @property
    def pretty_formula(self):
        return self.s.formula
    
    @property
    def els(self):
        return CompTools(self.compact_formula).els
    
    @property
    def magnetic_ions_in_struc(self):
        els = self.els
        magnetic_ions = self.magnetic_ions
        return sorted(list(set([el for el in els if el in magnetic_ions])))
        
    @property
    def magnetic_ions(self):
        # from https://github.com/materialsproject/pymatgen/blob/master/pymatgen/analysis/magnetism/default_magmoms.yaml
        return sorted(list(set(['Co', 'Cr', 'Fe', 'Mn', 'Mo', 'Ni', 'V', 
                                'W', 'Ce', 'Eu', 'Ti', 'V', 'Cr', 'Mn', 
                                'Fe', 'Co', 'Ni', 'Cu', 'Pr', 'Nd', 'Pm', 
                                'Sm', 'Gd', 'Tb', 'Dy', 'Ho', 'er', 'Tm', 
                                'Yb', 'Np', 'Ru', 'Os', 'Ir', 'U' ])))
    
    @property
    def get_nonmagnetic_structure(self):
        
        s = self.s
        cmsa = CollinearMagneticStructureAnalyzer(s)
        return cmsa.get_nonmagnetic_structure()
    
    @property
    def get_ferromagnetic_structure(self):
        magnetic_ions_in_struc = self.magnetic_ions_in_struc
        if len(magnetic_ions_in_struc) == 0:
            return None
        s = self.s        
        cmsa = CollinearMagneticStructureAnalyzer(s)
        return cmsa.get_ferromagnetic_structure()
    
    @property
    def get_antiferromagnetic_structures(self):
        magnetic_ions_in_struc = self.magnetic_ions_in_struc
        if len(magnetic_ions_in_struc) == 0:
            return None
        
        s = self.get_ferromagnetic_structure
        magnetic_sites = [i for i in range(len(s)) if s[i].species_string in magnetic_ions_in_struc]
        spins = (-5, 5)
        combos = itertools.product(range(len(spins)), repeat=len(magnetic_sites))
        combos = list(combos)
        combos = [c for c in combos if sum(c) == 0.5*len(magnetic_sites)]
        strucs = [{'s' : None, 'spin_up_indices' : [], 'spin_down_indices' : []} for i in range(len(combos))]
        for j in range(len(combos)): 
            c = combos[j]
            s = self.get_ferromagnetic_structure
            for i in range(len(magnetic_sites)):
                site = s[magnetic_sites[i]]
                if c[i] == 0:
                    site.properties['magmom'] = -5
                    strucs[0]['spin_down_indices'].append(magnetic_sites[i])
                elif c[i] == 1:
                    site.properties['magmom'] = 5
                    strucs[0]['spin_up_indices'].append(magnetic_sites[i])
            strucs[j]['s'] = s
        
        fake_strucs = []
        for i in range(len(strucs)):
            struc = strucs[i]['s']
            spin_up = strucs[i]['spin_up_indices']
            spin_down = strucs[i]['spin_down_indices']
            indices_species_map = {}
            for idx in spin_up:
                el = struc[i].species_string
                indices_species_map[idx] = el+'2+'
            for idx in spin_down:
                el = struc[i].species_string
                indices_species_map[idx] = el+'1+'               
            rsst = ReplaceSiteSpeciesTransformation(indices_species_map)
            fake_strucs.append(rsst.apply_transformation(struc))
        
        unique_fake_strucs = [fake_strucs[0]]
        sm = StructureMatcher(attempt_supercell=True)
        for i in range(len(fake_strucs)):
            same = False
            for j in range(len(unique_fake_strucs)):
                if i == j:
                    continue
                s1, s2 = fake_strucs[i], fake_strucs[j]
                same = sm.fit(s1, s2)


        
        
        
                
        ### PICK UP HERE

        
        
        fake_strucs = []
        for i in range(len(strucs)):
            s_tmp = strucs[i]
            for site in s_tmp:
                if site.properties['magmom'] == 5:
                    site.species = 'Xe'
                elif site.properties['magmom'] == -5:
                    site.species = 'Kr'
        unique_strucs = []       
        
        return strucs          
    
def main():
    mpq = MPQuery('***REMOVED***')
    s = mpq.get_structure_by_material_id('mp-22584')
    #s.make_supercell([3,3,3])
    st = StrucTools(s)
    
    #return st, st, st
    
    s_nm, s_fm = st.get_nonmagnetic_structure, st.get_ferromagnetic_structure
    
    afm_strucs = st.get_antiferromagnetic_structures
    #afm_strucs = 0
    return s_nm, s_fm, afm_strucs

if __name__ == '__main__':
    s_nm, s_fm, afm_strucs = main()
    
    
    
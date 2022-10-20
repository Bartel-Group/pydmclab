from pymatgen.core.structure import Structure
from pymatgen.analysis.magnetism.analyzer import CollinearMagneticStructureAnalyzer
#from pymatgen.analysis.magnetism.analyzer import MagneticStructureEnumerator
from CompTools import CompTools
from MPQuery import MPQuery
from pymatgen.core.structure import Structure
from pymatgen.transformations.standard_transformations import OrderDisorderedStructureTransformation
from pymatgen.analysis.structure_matcher import StructureMatcher
from pymatgen.transformations.site_transformations import ReplaceSiteSpeciesTransformation
import itertools

class StrucTools(object):
    """
    Purpose: to manipulate crystal structures for DFT calculations
    """
    
    def __init__(self, structure):
        """
        Args:
            structure (Structure): pymatgen structure object
                
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
    

    @property
    def magnetic_ions(self):
        """
        taken from what pymatgen uses to guess ox states
            from https://github.com/materialsproject/pymatgen/blob/master/pymatgen/analysis/magnetism/default_magmoms.yaml

        """
        return sorted(list(set(['Co', 'Cr', 'Fe', 'Mn', 'Mo', 'Ni', 'V', 
                                'W', 'Ce', 'Eu', 'Ti', 'V', 'Cr', 'Mn', 
                                'Fe', 'Co', 'Ni', 'Cu', 'Pr', 'Nd', 'Pm', 
                                'Sm', 'Gd', 'Tb', 'Dy', 'Ho', 'er', 'Tm', 
                                'Yb', 'Np', 'Ru', 'Os', 'Ir', 'U' ])))
    @property
    def magnetic_ions_in_struc(self):
        """
        list of elements (str) in structure that are magnetic
        """
        els = self.els
        magnetic_ions = self.magnetic_ions
        return sorted(list(set([el for el in els if el in magnetic_ions])))
        
    @property
    def get_nonmagnetic_structure(self):
        """
        Returns nonmagnetic Structure with all magnetic ions ordered ferromagnetically
        """       
        s = self.s
        cmsa = CollinearMagneticStructureAnalyzer(s)
        return cmsa.get_nonmagnetic_structure()
    
    @property
    def get_ferromagnetic_structure(self):
        """
        Returns Structure with all magnetic ions ordered ferromagnetically
            - according to pymatgen convention
        """
        magnetic_ions_in_struc = self.magnetic_ions_in_struc
        if len(magnetic_ions_in_struc) == 0:
            return None
        s = self.s        
        cmsa = CollinearMagneticStructureAnalyzer(s)
        return cmsa.get_ferromagnetic_structure()
    
    @property
    def get_antiferromagnetic_structures(self):
        """
        This is a chaotic way to get antiferromagnetic configurations 
            - but it doesn't require enumlib interaction with pymatgen
            - it seems reasonably efficient, might break down for large/complex structures
            - note: it has no idea which configurations are "most likely" to be low energy
        
        Basic workflow:
            - start from a the FM structure
            - for all sites containing ions in magnetic_ions
                - generate all possible combinations of 0 (spin down) or 1 (spin up) for each site
                    - if I had four sites w/ mag ions this might be: [(0,0,0,1), (0,0,1,1), ...]
                - retain only the combinations that sum to 0.5 (ie half spin down, half spin up) 
            - now apply all these combinations to the structure
                - generate a new structure for each combination that puts max(spin) on sites with 1 and min(spin) on sites with 0
            - now figure out which newly generated structures are symmetrically distinct
                - change the identities of sites that are spin up/down using oxidation state surrogate
                    - these ox states aren't physically meaningful, just a placeholder
                    - spin up: 2+, spin down: +
            - now use StructureMatcher to find unique structures to return
                 
        Returns:
            list of unique Structure objects with antiferromagnetic ordering
                - exhaustive
                - no idea which are most likely to be low energy
                - reasonable to randomly sample if a very large list
        """
        spins = (-5,5)
        magnetic_ions_in_struc = self.magnetic_ions_in_struc
        if len(magnetic_ions_in_struc) == 0:
            return None
        
        s = self.get_ferromagnetic_structure
        magnetic_sites = [i for i in range(len(s)) if s[i].species_string in magnetic_ions_in_struc]
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
                    site.properties['magmom'] = min(spins)
                    strucs[j]['spin_down_indices'].append(magnetic_sites[i])
                elif c[i] == 1:
                    site.properties['magmom'] = max(spins)
                    strucs[j]['spin_up_indices'].append(magnetic_sites[i])
            strucs[j]['s'] = s
        
        fake_strucs = []
        for i in range(len(strucs)):
            struc = strucs[i]['s']
            spin_up = strucs[i]['spin_up_indices']
            spin_down = strucs[i]['spin_down_indices']
            indices_species_map = {}
            for idx in spin_up:
                el = struc[idx].species_string
                indices_species_map[idx] = el+'2+'
            for idx in spin_down:
                el = struc[idx].species_string
                indices_species_map[idx] = el+'1+'               
            rsst = ReplaceSiteSpeciesTransformation(indices_species_map)
            fake_strucs.append(rsst.apply_transformation(struc))

        unique_fake_strucs = [fake_strucs[0]]
        sm = StructureMatcher(attempt_supercell=True)
        for i in range(len(fake_strucs)):
            same = False
            check = 0
            for j in range(len(unique_fake_strucs)):
                if check > 0:
                    continue
                if i == j:
                    continue
                s1, s2 = fake_strucs[i], unique_fake_strucs[j]
                same = sm.fit(s1, s2)
                if same:
                    check += 1
            if check == 0:
                print('adding you %s' % i)
                unique_fake_strucs.append(fake_strucs[i])
                            
        out = []
        for struc in unique_fake_strucs:
            struc.remove_oxidation_states()
            out.append(struc)
        return out
    
    
    def get_ordered_strucs(self):
        return 'TO DO'
            
def main():
    mpq = MPQuery('***REMOVED***')
    s = mpq.get_structure_by_material_id('mp-1301329')
    #s.make_supercell([3,3,3])
    st = StrucTools(s)
    
    #return st, st, st
    
    s_nm, s_fm = st.get_nonmagnetic_structure, st.get_ferromagnetic_structure
    
    afm_strucs = st.get_antiferromagnetic_structures
    #afm_strucs = 0
    return s_nm, s_fm, afm_strucs

if __name__ == '__main__':
    s_nm, s_fm, afm_strucs = main()
    
    
    
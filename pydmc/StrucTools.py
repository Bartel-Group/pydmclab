from pydmc.CompTools import CompTools

from pymatgen.core.structure import Structure
from pymatgen.transformations.standard_transformations import OrderDisorderedStructureTransformation, AutoOxiStateDecorationTransformation, OxidationStateDecorationTransformation
from pymatgen.analysis.structure_matcher import StructureMatcher
from pymatgen.core.composition import Element


class StrucTools(object):
    """
    Purpose: to manipulate crystal structures for DFT calculations
    """
    
    def __init__(self, structure,
                 ox_states=None):
        """
        Args:
            structure (Structure): pymatgen Structure object
            ox_states (dict): dictionary of oxidation states {el (str) : oxidation state (int)}
                - or None
                
        """
        
        self.structure = structure
        self.ox_states = ox_states
        
    @property
    def compact_formula(self):
        """
        "clean" (reduced, systematic) formula for structure
        """
        return CompTools(self.structure.formula).clean
    
    @property
    def formula(self):
        """
        pretty (unreduced formula) for structure
        
        """
        return self.structure.formula
    
    @property
    def els(self):
        """
        list of unique elements (str) in structure
        """
        return CompTools(self.compact_formula).els
    
    def make_supercell(self, 
                       grid):
        """
        Args:
            grid (list) - [nx, ny, nz]
            
        Returns:
            Structure repeated nx, ny, nz
        """
        s = self.structure
        print('making supercell with grid %s\n' % str(grid))
        s.make_supercell(grid)
        return s
        
    
    @property
    def decorate_with_ox_states(self):
        """
        Returns oxidation state decorated structure
            - uses Auto algorithm if no ox_states are provided
            - otherwise, applies ox_states
        """
        print('decorating with oxidation states\n')
        s = self.structure
        ox_states = self.ox_states
        if not ox_states:
            print('     automatically\n')
            transformer = AutoOxiStateDecorationTransformation()
        else:
            transformer = OxidationStateDecorationTransformation(ox_states=ox_states)
            print('     using %s' % str(ox_states))
        return transformer.apply_transformation(s)
    
    def get_ordered_structures(self,
                               algo=0,
                               decorate=True,
                               n_strucs=1):
        """
        Args:
            algo (int) - 0 = fast, 1 = complete, 2 = best first
            decorate (bool) - whether to decorate with oxidation states
                - if False, self.structure must already have them
            n_strucs (int) - number of ordered structures to return
            
        Returns:
            dict of ordered structures {index : structure (Structure.as_dict())}
                - index = 0 has lowest Ewald energy
        """
        transformer = OrderDisorderedStructureTransformation(algo=algo)
        if decorate:
            s = self.decorate_with_ox_states
        else:
            s = self.structure
        return_ranked_list = n_strucs if n_strucs > 1 else False
        
        print('ordering disordered structures\n')
        out = transformer.apply_transformation(s,
                                                return_ranked_list=return_ranked_list)
        out = [i['structure'] for i in out]
        print(out[0])
        if isinstance(out, list):
            print('getting unique structures\n')
            matcher = StructureMatcher()
            groups = matcher.group_structures(out)
            out = [groups[i][0] for i in range(len(groups))]
            return {i: out[i].as_dict() for i in range(len(out))}
        else:
            return {0 : out.as_dict()}
    
    def replace_species(self,
                        species_mapping,
                        n_strucs=1):
        """
        Args:
            species_mapping (dict) - {Element(el) : 
                                        {Element(el1) : fraction el1,
                                                        fraction el2}}
            n_strucs (int) - number of ordered structures to return if disordered
            
        Returns:
            dict of ordered structures {index : structure (Structure.as_dict())}
                - index = 0 has lowest Ewald energy
        """
        s = self.structure
        print('replacing species with %s\n' % str(species_mapping))
        s.replace_species(species_mapping)
        if s.is_ordered:
            return {0 : s.as_dict()}
        else:
            structools = StrucTools(s, self.ox_states)
            return structools.get_ordered_structures(n_strucs=n_strucs) 
               
    def get_structures_with_dilute_vacancy(self,
                                            el_to_replace,
                                            n_strucs=1,
                                            structure=None):
        if not structure:
            s = self.structure
        else:
            s = structure
        species_mapping = {Element(el_to_replace) : 
                            {Element(el_to_replace) : 1-1/len(s)}}
        if not structure:
            return self.replace_species(species_mapping, n_strucs=n_strucs)
        else:
            return StrucTools(structure).replace_species(species_mapping, n_strucs=n_strucs)
        
        
        
def main():
    from pydmc.MPQuery import MPQuery

    mpq = MPQuery('***REMOVED***')
    mpid = 'mp-22584' # LiMn2O4
    #mpid = 'mp-1301329' # LiMnTiO4
    #mpid = 'mp-770495' # Li5Ti2Mn3Fe3O16
    mpid = 'mp-772660' # NbCrO4
    #mpid = 'mp-776873' # Cr2O3
    s = mpq.get_structure_by_material_id(mpid)
    s.make_supercell([2,2,2])
    st = StrucTools(s)
    
    out = st.replace_species({Element('Cr') : {Element('Ti') : 1/2,
                                            Element('Fe') : 1/4}},
                             n_strucs=100)
    
    return out

if __name__ == '__main__':
    out = main()
    
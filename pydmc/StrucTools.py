from pydmc.CompTools import CompTools

from pymatgen.core.structure import Structure
from pymatgen.transformations.standard_transformations import OrderDisorderedStructureTransformation, AutoOxiStateDecorationTransformation, OxidationStateDecorationTransformation
from pymatgen.analysis.structure_matcher import StructureMatcher
from pymatgen.core.composition import Element, Composition
from pymatgen.core.ion import Ion


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
        if isinstance(structure, dict):
            structure = Structure.from_dict(structure)
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
            transformer = OxidationStateDecorationTransformation(oxidation_states=ox_states)
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
        #print(out[0])
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
        """
        Args:
            el_to_replace (str) - element to replace with vacancy
            n_strucs (int) - number of ordered structures to return if disordered
            structure (Structure) - structure to create vacancy in
                - if None, use self.structure
                
        Returns:
            dict of ordered structures {index : structure (Structure.as_dict())}
                - each structure will be missing 1 el_to_replace
        """
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
        
class SiteTools(object):
    """
    make it a little easier to get site info
    
    @Chris - TO DO
    """
    def __init__(self, structure, index):
        self.site = structure[index]
    
    @property
    def site_dict(self):
        """
        Returns:
            dict of site info (from Pymatgen)
        """
        return self.site.as_dict()
    
    @property
    def coords(self):
        """
        Returns:
            array of fractional coordinates for site ([x, y, z])
        """
        return self.site.frac_coords
    
    @property
    def magmom(self):
        """
        Returns:
            magnetic moment for site (float) or None
        """
        props = self.site.properties
        if props:
            if 'magmom' in props:
                return props['magmom']
        return None
    
    @property
    def is_fully_occ(self):
        """
        Returns:
            True if site is fully occupied else False
        """
        return self.site.is_ordered
    
    @property
    def ion(self):
        """
        Returns:
            whatever is occupying site (str)
                - could be multiple ions, multiple elements, one element, one ion, etc
        """
        return self.site.species_string
    
    @property
    def el(self):
        """
        Returns:
            just the element occupying the site (even if it has an oxidation state)
        """
        return CompTools(Composition(self.ion).formula).els[0]
    
    @property
    def ox_state(self):
        """
        Returns:
            oxidation state (float) of site
        """
        if self.is_fully_occ:
            return self.site_dict['species'][0]['oxidation_state']
        else:
            print('cant determine ox state for partially occ site')
            return None
    
def main():
    from pydmc.MPQuery import MPQuery

    mpq = MPQuery('***REMOVED***')
    mpid = 'mp-22584' # LiMn2O4
    #mpid = 'mp-1301329' # LiMnTiO4
    #mpid = 'mp-770495' # Li5Ti2Mn3Fe3O16
    mpid = 'mp-772660' # NbCrO4
    #mpid = 'mp-776873' # Cr2O3
    
    mpid = 'mp-825' # RuO2
    s = mpq.get_structure_by_material_id(mpid)
    s.make_supercell([2,1,1])
    print(s)
    st = StrucTools(s)
    
    out = st.replace_species({Element('Ru') : {Element('Ir') : 1/2,
                                            Element('Ru') : 1/2}},
                             n_strucs=100)
    
    return out

if __name__ == '__main__':
    out = main()
    

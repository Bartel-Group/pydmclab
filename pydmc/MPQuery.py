from pydmc.CompTools import CompTools

from pymatgen.ext.matproj import MPRester

import itertools

class MPQuery(object):
    """
    class to assist with downloading data from Materials Project
    
    """
    
    
    def __init__(self, api_key=None):
        """
        Args:
            api_key (str) - Materials Project API key
            
        Returns:
            self.mpr (MPRester) - Materials Project REST interface
        """
        
        api_key = api_key if api_key else 'YOUR_API_KEY'
        
        self.api_key = api_key
        self.mpr = MPRester(api_key)
          
    @property
    def supported_properties(self):
        """
        Returns list of supported properties to query for MP entries in Materials Project
        """
        supported_properties = (
            "energy",
            "energy_per_atom",
            "volume",
            "formation_energy_per_atom",
            "nsites",
            "unit_cell_formula",
            "pretty_formula",
            "is_hubbard",
            "elements",
            "nelements",
            "e_above_hull",
            "hubbards",
            "is_compatible",
            "spacegroup",
            "task_ids",
            "band_gap",
            "density",
            "icsd_id",
            "icsd_ids",
            "cif",
            "total_magnetization",
            "material_id",
            "oxide_type",
            "tags",
            "elasticity",
        )
        
        return supported_properties
    
    @property
    def supported_task_properties(self):
        """
        returns list of supported properties that can be queried for any MP task
        """
        
        supported_task_properties = (
            "energy",
            "energy_per_atom",
            "volume",
            "formation_energy_per_atom",
            "nsites",
            "unit_cell_formula",
            "pretty_formula",
            "is_hubbard",
            "elements",
            "nelements",
            "e_above_hull",
            "hubbards",
            "is_compatible",
            "spacegroup",
            "band_gap",
            "density",
            "icsd_id",
            "cif",
        )
        
        return supported_task_properties
    
    @property
    def typical_properties(self):
        """
        A list of propreties that we often query for
        
        """
        typical_properties = ('energy_per_atom', 
                                'pretty_formula',
                                'material_id',
                                'formation_energy_per_atom',
                                'e_above_hull',
                                'nsites',
                                'volume',
                                'spacegroup.number') 
        return typical_properties 
    
    @property
    def long_to_short_keys(self):
        """
        A map to nickname query properties with shorter handles
            (dict)
        """
        return {'energy_per_atom' : 'E_mp',
                'formation_energy_per_atom' : 'Ef_mp',
                'e_above_hull' : 'Ehull_mp',
                'spacegroup.number' : 'sg',
                'material_id' : 'mpid'}
         
    def get_data_for_comp(self, 
                          comp, 
                          properties=None, 
                          criteria=None, 
                          only_gs=False, 
                          dict_key=False):
        """
        Args:
            comp (list or str)
                can either be:
                    - a chemical system (str) of elements joined by "-"
                    - a chemical formula (str)
                can either be a list of:
                    - chemical systems (str) of elements joined by "-"
                    - chemical formulas (str)
            
            properties (list or None)
                list of properties to query
                    - if None, then use typical_properties
                    - if 'all', then use supported_properties
                    
            criteria (dict or None)
                dictionary of criteria to query
                    - if None, then use {}
            
            only_gs (bool)
                if True, remove non-ground state polymorphs for each unique composition
                
            dict_key (str)
                if False, return list of dicts
                if True, return dict oriented by dict_key
                    e.g., if dict_key = 'cmpd', then returns {CMPD : {query_data_for_that_cmpd}}
                        or dict_key = 'mpid' --> {MPID : {data_for_that_mpid}}
        Returns:
            list of dictionaries of properties for each material in the desired comp
        """
        key_map = self.long_to_short_keys
        if properties == 'all':
            properties = self.supported_properties
        if properties == None:
            properties = self.typical_properties
        else:
            for prop in properties:
                if prop not in self.supported_properties:
                    raise ValueError("Property %s is not supported!" % prop)
        
        if criteria == None:
            criteria = {}
        
        if isinstance(comp, str):
            if '-' in comp:
                chemsys = comp                      
                all_chemsyses = []
                elements = chemsys.split('-')
                for i in range(len(elements)):
                    for els in itertools.combinations(elements, i+1):
                        all_chemsyses.append('-'.join(sorted(els)))
                        
                criteria['chemsys'] = {'$in': all_chemsyses}
            else:
                formula = comp
                criteria['pretty_formula'] = {'$in' : [CompTools(formula).pretty]}
                
        elif isinstance(comp, list):
            if '-' in comp[0]:
                all_chemsyses = []
                for chemsys in comp:
                    elements = chemsys.split('-')
                    for i in range(len(elements)):
                        for els in itertools.combinations(elements, i+1):
                            all_chemsyses.append('-'.join(sorted(els)))
                all_chemsyses = sorted(list(set(all_chemsyses)))
                criteria['chemsys'] = {'$in': all_chemsyses}
            else:
                all_formulas = [CompTools(c).pretty for c in comp]
                criteria['pretty_formula'] = {'$in' : all_formulas}
                
        list_from_mp = self.mpr.query(criteria, properties)
        extra_keys = [k for k in list_from_mp[0] if k not in key_map]
        cleaned_list_from_mp = [{key_map[old_key] : entry[old_key] for old_key in key_map} for entry in list_from_mp]
        query = []
        for i in range(len(list_from_mp)):
            query.append({**cleaned_list_from_mp[i], 
                          **{k : list_from_mp[i][k] for k in extra_keys}, 
                          **{'cmpd' : CompTools(list_from_mp[i]['pretty_formula']).clean}})
        
        if only_gs:
            
            gs = {}
            for entry in query:
                cmpd = CompTools(entry['pretty_formula']).clean
                if cmpd not in gs:
                    gs[cmpd] = entry
                else:
                    Ef_stored = gs[cmpd]['Ef_mp']
                    Ef_check = entry['Ef_mp']
                    if Ef_check < Ef_stored:
                        gs[cmpd] = entry
            query = [gs[k] for k in gs]
        if dict_key:
            if dict_key not in query[0]:
                raise ValueError('%s not in query' % dict_key)
            query = {entry[dict_key] : entry for entry in query}
            
        return query
    
    def get_entry_by_material_id(self, 
                                 material_id, 
                                 properties=None, 
                                 incl_structure=True,
                                 conventional=False,
                                 compatible_only=True):
        """
        Args:
            material_id (str) - MP ID of entry
            properties (list) - list of properties to query
            incl_structure (bool) - whether to include structure in entry
            conventional (bool) - whether to use conventional unit cell
            compatible_only (bool) - whether to only include compatible entries (related to MP formation energies)
            
        Returns:
            ComputedEntry object
        """
        return self.mpr.get_entry_by_material_id(material_id,
                                                 compatible_only,
                                                 incl_structure,
                                                 properties,
                                                 conventional)
        
    def get_structure_by_material_id(self, material_id):
        """
        Args:
            material_id (str) - MP ID of entry
            
        Returns:
            Structure object
        """
        return self.mpr.get_structure_by_material_id(material_id)
    
    def get_incar(self, material_id):
        """
        Args:
            material_id (str) - MP ID of entry
            
        Returns:
            dict of incar settings
        """
        return self.mpr.query(material_id, ['input.incar'])[0]
        
    def get_kpoints(self, material_id):
        """
        Args:
            material_id (str) - MP ID of entry
            
        Returns:
            dict of kpoint settings
        """
        return self.mpr.query(material_id, ['input.kpoints'])[0]['input.kpoints'].as_dict()     
    
    def get_vasp_inputs(self, material_id):
        """
        Args:
            material_id (str) - MP ID of entry
        
        Returns:
            dict of vasp inputs
                - 'incar' : {setting (str) : value (mixed type)}
                - 'kpoints' : {'scheme' : (str), 'grid' : list of lists for 'A B C'}
                - 'potcar' : [list of TITELs]
                - 'structure' : Structure object as dict
        """
        
        d = self.mpr.query(material_id, ['input'])[0]['input']
        d['kpoints'] = d['kpoints'].as_dict()
        d['kpoints'] = {'scheme' : d['kpoints']['generation_style'],
                        'grid' : d['kpoints']['kpoints']}
        d['potcar'] = [d['potcar_spec'][i]['titel'] for i in range(len(d['potcar_spec']))]
        d['poscar'] = self.get_structure_by_material_id(material_id).as_dict()
        del d['potcar_spec']
        
        return d
    
def main():
    api_key = '***REMOVED***'
    mpq = MPQuery(api_key)
    data = mpq.get_vasp_inputs('mp-1009009')
    return data

if __name__ == '__main__':
    data = main()
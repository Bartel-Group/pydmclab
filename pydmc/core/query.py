from pydmc.core.comp import CompTools
from pydmc.core.struc import StrucTools

from pymatgen.ext.matproj import MPRester

from mp_api.client import MPRester as new_MPRester

import itertools
import numpy as np

class NewMPQuery(object):
    # Chris B key = ***REMOVED***
    
    """
    Trying to transition to new API but running into many issues
    """
    def __init__(self, api_key):
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
    def available_fields(self):
        return self.mpr.summary.available_fields
    
    @property
    def typical_fields(self):
        return ['formula_pretty',
                'volume',
                'nsites',
                'material_id',
                'energy_per_atom',
                'uncorrected_energy_per_atom',
                'formation_energy_per_atom',
                'energy_above_hull',
                'equilibrium_reaction_energy_per_atom',
                'decomposes_to',
                'band_gap',
                'is_magnetic',
                'symmetry']
        
    def get_data_for_comp(self, comp):
        
        all_chemsyses = None
        all_formulas = None
        if isinstance(comp, str):
            if '-' in comp:
                chemsys = comp                      
                all_chemsyses = []
                elements = chemsys.split('-')
                for i in range(len(elements)):
                    for els in itertools.combinations(elements, i+1):
                        all_chemsyses.append('-'.join(sorted(els)))        
            else:
                all_formulas = [CompTools(comp).clean]
        elif isinstance(comp, list):
            if '-' in comp[0]:
                all_chemsyses = []
                for chemsys in comp:
                    elements = chemsys.split('-')
                    for i in range(len(elements)):
                        for els in itertools.combinations(elements, i+1):
                            all_chemsyses.append('-'.join(sorted(els)))
                all_chemsyses = sorted(list(set(all_chemsyses)))   
            else:
                all_formulas = [CompTools(c).pretty for c in comp]
        if all_formulas:
            from_mp = self.mpr.summary.search(formula_pretty=all_formulas,
                                              fields=self.typical_fields)
        elif all_chemsyses:
            from_mp = self.mpr.summary.search(chemsys=all_chemsyses,
                                              fields=self.typical_fields)
        
        return from_mp
        
class MPQuery(object):
    # Chris B API KEY = 
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
            
        So after querying 'energy_per_atom' will be a key, but this map will convert that to 'E_mp'
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
                          include_structure=False,
                          supercell_structure=False,
                          max_Ehull=None,
                          max_sites_per_structure=None,
                          max_strucs_per_cmpd=None):
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
                
            include_structure (bool)
                if True, include the structure (as a dictionary) for each entry
                
            supercell_structure (bool)
                only runs if include_structure = True
                if False, just retrieve the MP structure
                if not False, must be specified as [a,b,c] to make an a x b x c supercell of the MP structure
                
            max_Ehull (float)
                if not None, remove entries with Ehull_mp > max_Ehull
                
            max_sites_per_structure (int)
                if not None, remove entries with more than max_sites_per_structure sites
            
            max_strucs_per_cmpd (int)
                if not None, only retain the lowest energy structures for each composition until you reach max_strucs_per_cmpd
                
        Returns:
            {mpid : {DATA}}
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
                    energy_key = 'E_mp' if len(CompTools(cmpd).els) == 1 else 'Ef_mp'
                    Ef_stored = gs[cmpd][energy_key]
                    Ef_check = entry[energy_key]
                    if Ef_check < Ef_stored:
                        gs[cmpd] = entry
            query = [gs[k] for k in gs]
            query = {entry['mpid'] : entry for entry in query}
            
        else:
            query = {entry['mpid'] : entry for entry in query}
            
        if include_structure:
            for entry in query:
                mpid = query[entry]['mpid']
                structure = self.get_structure_by_material_id(mpid)
                if supercell_structure:
                    if len(supercell_structure) == 3:
                        structure = StrucTools(structure).make_supercell(supercell_structure)
                query[entry]['structure'] = structure.as_dict()
                
        if max_sites_per_structure:
            query = {e : query[e] for e in query if query[e]['nsites'] <= max_sites_per_structure}
        
        if max_Ehull:
            if only_gs:
                query = {e : query[e] for e in query if query[e]['Ehull_mp'] <= max_Ehull}
            else:
                cmpds = sorted(list(set([query[e]['cmpd'] for e in query])))
                mpids = [e for e in query if query[e]['Ehull_mp'] <= max_Ehull if query[e]['cmpd'] in cmpds]
                query = {e : query[e] for e in mpids}
        
        if max_strucs_per_cmpd:
            if not only_gs:
                trimmed_query = {}
                cmpds = sorted(list(set([query[e]['cmpd'] for e in query])))
                for cmpd in cmpds:
                    mpids = [e for e in query if query[e]['cmpd'] == cmpd]
                    Ehulls = [query[e]['Ehull_mp'] for e in mpids]
                    sorted_indices = np.argsort(Ehulls)
                    relevant_ids = [mpids[i] for i in sorted_indices]
                    if len(relevant_ids) > max_strucs_per_cmpd:
                        relevant_ids = relevant_ids[:max_strucs_per_cmpd]
                    for mpid in relevant_ids:
                        trimmed_query[mpid] = query[mpid]
            
                return trimmed_query
        
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

    return 

if __name__ == '__main__':
    mpq = main()

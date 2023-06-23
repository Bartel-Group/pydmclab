from pydmclab.core.comp import CompTools
from pydmclab.core.struc import StrucTools

from pymatgen.ext.matproj import MPRester

# from mp_api.client import MPRester as new_MPRester

import itertools
import numpy as np

""" 
Purpose:
    - query the Materials Project database for data
    
Typical use:
    MPQuery.get_data_for_comp(...)
"""


class MPQuery(object):
    # Chris B API KEY =
    """
    class to assist with downloading data from Materials Project

    """

    def __init__(self, api_key=None):
        """
        Args:
            api_key (str)
                Materials Project API key

        Returns:
            self.mpr (MPRester)
                Materials Project REST interface
        """

        api_key = api_key if api_key else "YOUR_API_KEY"

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
        typical_properties = (
            "energy_per_atom",
            "pretty_formula",
            "material_id",
            "formation_energy_per_atom",
            "e_above_hull",
            "nsites",
            "volume",
            "spacegroup.number",
        )
        return typical_properties

    @property
    def long_to_short_keys(self):
        """
        A map to nickname query properties with shorter handles
            (dict)

        So after querying 'energy_per_atom' will be a key, but this map will convert that to 'E_mp'
        """
        return {
            "energy_per_atom": "E_mp",
            "formation_energy_per_atom": "Ef_mp",
            "e_above_hull": "Ehull_mp",
            "spacegroup.number": "sg",
            "material_id": "mpid",
        }

    def get_data_for_comp(
        self,
        comp,
        properties=None,
        criteria={},
        only_gs=True,
        include_structure=True,
        supercell_structure=False,
        max_Ehull=0.1,
        max_sites_per_structure=100,
        max_strucs_per_cmpd=5,
    ):
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
        # convert MP keys into shorter keys
        key_map = self.long_to_short_keys
        if properties == "all":
            # use all supported properties
            properties = self.supported_properties
        if properties == None:
            # use our typical properties
            properties = self.typical_properties
        else:
            # make sure properties are supported
            for prop in properties:
                if prop not in self.supported_properties:
                    raise ValueError("Property %s is not supported!" % prop)

        if criteria == None:
            # make criteria an empty dictionary
            criteria = {}

        if isinstance(comp, str):
            # just working with one compound or chemical system
            if "-" in comp:
                # must be a chemical system
                chemsys = comp
                # need to get all chemical (sub)systems
                all_chemsyses = []
                elements = chemsys.split("-")
                for i in range(len(elements)):
                    for els in itertools.combinations(elements, i + 1):
                        all_chemsyses.append("-".join(sorted(els)))
                # add these chemical spaces to our criteria
                criteria["chemsys"] = {"$in": all_chemsyses}
            else:
                # just working with one formula
                formula = comp
                # query only for that formula
                criteria["pretty_formula"] = {"$in": [CompTools(formula).pretty]}

        elif isinstance(comp, list):
            # now we have a list of compounds or chemical systems (should be one or the other)
            if "-" in comp[0]:
                # must be a list of chemical systems, let's get em all
                all_chemsyses = []
                for chemsys in comp:
                    elements = chemsys.split("-")
                    for i in range(len(elements)):
                        for els in itertools.combinations(elements, i + 1):
                            all_chemsyses.append("-".join(sorted(els)))
                all_chemsyses = sorted(list(set(all_chemsyses)))
                criteria["chemsys"] = {"$in": all_chemsyses}
            else:
                # get the entire list of formulas
                all_formulas = [CompTools(c).pretty for c in comp]
                criteria["pretty_formula"] = {"$in": all_formulas}

        # initalize the rester and query
        mpr = self.mpr
        list_from_mp = mpr.query(criteria, properties)
        if not list_from_mp:
            raise ValueError("No entries found for criteria %s" % criteria)

        # shorten the keys we can shorten
        cleaned_list_from_mp = [
            {key_map[old_key]: entry[old_key] for old_key in key_map}
            for entry in list_from_mp
        ]

        # grab the keys that won't get mapped to short keys
        extra_keys = [k for k in list_from_mp[0] if k not in key_map]

        # assemble all the chunked queries into one query
        query = []
        for i in range(len(list_from_mp)):
            query.append(
                {
                    **cleaned_list_from_mp[i],
                    **{k: list_from_mp[i][k] for k in extra_keys},
                    **{"cmpd": CompTools(list_from_mp[i]["pretty_formula"]).clean},
                }
            )

        if only_gs:
            # grab only the lowest energy entry for each composition
            gs = {}
            for entry in query:
                cmpd = CompTools(entry["pretty_formula"]).clean
                if cmpd not in gs:
                    gs[cmpd] = entry
                else:
                    energy_key = "E_mp" if len(CompTools(cmpd).els) == 1 else "Ef_mp"
                    Ef_stored = gs[cmpd][energy_key]
                    Ef_check = entry[energy_key]
                    if Ef_check < Ef_stored:
                        gs[cmpd] = entry
            query = [gs[k] for k in gs]

        # orient our query into a dictionary keyed by MP ID
        query = {entry["mpid"]: entry for entry in query}

        if include_structure:
            for mpid in query:
                # grab the structure for each MPID
                structure = self.get_structure_by_material_id(mpid)
                if supercell_structure:
                    if len(supercell_structure) == 3:
                        structure = StrucTools(structure).make_supercell(
                            supercell_structure
                        )
                query[mpid]["structure"] = structure.as_dict()

        if max_sites_per_structure:
            # remove entries that have too many sites
            query = {
                e: query[e]
                for e in query
                if query[e]["nsites"] <= max_sites_per_structure
            }

        if max_Ehull:
            # remove entries that are too far above the hull
            query = {e: query[e] for e in query if query[e]["Ehull_mp"] <= max_Ehull}

        if max_strucs_per_cmpd:
            if not only_gs:
                trimmed_query = {}
                cmpds = sorted(list(set([query[e]["cmpd"] for e in query])))
                for cmpd in cmpds:
                    mpids = [e for e in query if query[e]["cmpd"] == cmpd]
                    Ehulls = [query[e]["Ehull_mp"] for e in mpids]
                    sorted_indices = np.argsort(Ehulls)
                    relevant_ids = [mpids[i] for i in sorted_indices]
                    if len(relevant_ids) > max_strucs_per_cmpd:
                        relevant_ids = relevant_ids[:max_strucs_per_cmpd]
                    for mpid in relevant_ids:
                        trimmed_query[mpid] = query[mpid]

                return trimmed_query

        # close rester
        mpr.session.close()
        return query

    def get_entry_by_material_id(
        self,
        material_id,
        properties=None,
        incl_structure=True,
        conventional=False,
        compatible_only=True,
    ):
        """
        Args:
            material_id (str)
                MP ID of entry

            properties (list)
                list of properties to query

            incl_structure (bool)
                whether to include structure in entry

            conventional (bool)
                whether to use conventional unit cell

            compatible_only (bool)
                whether to only include compatible entries (related to MP formation energies)

        Returns:
            ComputedEntry object
        """
        mpr = self.mpr
        return mpr.get_entry_by_material_id(
            material_id, compatible_only, incl_structure, properties, conventional
        )

    def get_structure_by_material_id(self, material_id):
        """
        Args:
            material_id (str)
                MP ID of entry

        Returns:
            Structure object
        """
        mpr = self.mpr
        return mpr.get_structure_by_material_id(material_id)

    def get_incar(self, material_id):
        """
        Args:
            material_id (str)
                MP ID of entry

        Returns:
            dict of incar settings
        """
        mpr = self.mpr
        return mpr.query(material_id, ["input.incar"])[0]

    def get_kpoints(self, material_id):
        """
        Args:
            material_id (str)
                MP ID of entry

        Returns:
            dict of kpoint settings
        """
        mpr = self.mpr
        return mpr.query(material_id, ["input.kpoints"])[0]["input.kpoints"].as_dict()

    def get_vasp_inputs(self, material_id):
        """
        Args:
            material_id (str)
                MP ID of entry

        Returns:
            dict of vasp inputs
                - 'incar' : {setting (str) : value (mixed type)}
                - 'kpoints' : {'scheme' : (str), 'grid' : list of lists for 'A B C'}
                - 'potcar' : [list of TITELs]
                - 'structure' : Structure object as dict
        """

        mpr = self.mpr
        d = mpr.query(material_id, ["input"])[0]["input"]
        d["kpoints"] = d["kpoints"].as_dict()
        d["kpoints"] = {
            "scheme": d["kpoints"]["generation_style"],
            "grid": d["kpoints"]["kpoints"],
        }
        d["potcar"] = [
            d["potcar_spec"][i]["titel"] for i in range(len(d["potcar_spec"]))
        ]
        d["poscar"] = self.get_structure_by_material_id(material_id).as_dict()
        del d["potcar_spec"]

        return d


def main():
    return


if __name__ == "__main__":
    out = main()

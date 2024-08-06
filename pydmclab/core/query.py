from itertools import combinations
import numpy as np

from pymatgen.ext.matproj import MPRester as old_MPRester
from mp_api.client import MPRester

from pydmclab.core.comp import CompTools
from pydmclab.core.struc import StrucTools


""" 
Purpose:
    - query the Materials Project database for data
    
Typical use:
    MPQuery.get_data_for_comp(...)
"""


class MPQuery(object):
    """
    New MP API
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

        # api_key = api_key if api_key else "YOUR_API_KEY"

        self.api_key = api_key
        if not api_key:
            self.mpr = MPRester()
        else:
            self.mpr = MPRester(api_key)

    @property
    def fields(self):
        return self.mpr.summary.available_fields

    @property
    def typical_properties(self):
        props = [
            "formula_pretty",
            "material_id",
            "energy_per_atom",
            "formation_energy_per_atom",
            "uncorrected_energy_per_atom",
            "volume",
            "energy_above_hull",
            "symmetry.number",
            "nsites",
        ]
        return props

    @property
    def long_to_short_keys(self):
        return {
            "energy_per_atom": "E_mp",
            "formation_energy_per_atom": "Ef_mp",
            "energy_above_hull": "Ehull_mp",
            "material_id": "mpid",
            "symmetry.number": "sg",
        }

    def get_data(
        self,
        search_for,
        max_Ehull=0.05,
        max_polymorph_energy=0.1,
        max_strucs_per_cmpd=1,
        max_sites_per_structure=41,
        include_sub_phase_diagrams=False,
        include_structure=True,
        properties=None,
    ):
        """
        Args:
        search_for (str or list)
            can either be:
                a chemical system (str) of elements joined by "-"
                    eg 'Ca-Ti-O' for all ternary calcium titanium oxides
                a chemical formula (str)
                    eg 'CaTiO3' for this formula
                an MP ID (str)
                    eg 'mp-1234' for this ID
            or a list of:
                chemical systems (str) of elements joined by "-"
                    eg ['Ca-Ti-O', 'Sr-Ti-O']
                chemical formulas (str)
                    eg ['CaO', 'CaTiO3']
                MP IDs (str)
                    eg ['mp-1', 'mp-2']

        max_Ehull (float)
            upper bound on energy above hull for retrieved entries

        max_polymorph_energy (float)
            upper bound on polymorph energy for retrieved entries
                set to 0 to only retrieve ground-state structures for all compositions

        max_strucs_per_cmpd (int)
            upper bound on number of polymorphs to retrieve for each queried composition
                retains the lowest energy ones

        max_sites_per_structure (int)
            upper bound on number of sites in retrieved structures

        include_sub_phase_diagrams (bool)
            if True, include all sub-phase diagrams for a given composition
                e.g., if search_for = "Sr-Zr-S", then also include "Sr-S" and "Zr-S" in the query

        include_structure (bool)
            if True, include the structure (as a dictionary) for each entry

        properties (list or None)
            list of properties to query
                if None, then use pydmclab.core.query.MPQuery.typical_properties
                if 'all', then use all properties
                if a string, then add that property to typical_properties
                if a list, then add those properties to typical_properties

        Returns:
            {mpid : {property (str) : value (mixed type)}}


        """
        # start w/ typical properties
        baseline_properties = self.typical_properties

        # just use these if you don't say to add any
        if not properties:
            properties = baseline_properties

        elif isinstance(properties, str):
            if properties == "all":
                # add all possible properties
                properties = self.fields
            else:
                # if you add one property as a string, add it
                properties += [properties]
        elif isinstance(properties, list):
            # if you pass a list, add those properties
            properties += baseline_properties

        # include the structure in the query if asked for
        if include_structure and ("structure" not in properties):
            properties.append("structure")

        # map long MP keys to easier to read ones
        long_to_short_keys = self.long_to_short_keys

        # initialize MPRester
        mpr = self.mpr

        # convert a single-item list to str for convenience
        if isinstance(search_for, list) and (len(search_for) == 1):
            search_for = search_for[0]

        # figure out if you're searching based on formula, chemsys, or MP ID (or a list of any one of these)
        if isinstance(search_for, list):
            first_search = search_for[0]
            if "-" in first_search:
                if "mp" in first_search:
                    search_key = "material_id"
                else:
                    search_key = "chemsys"
            #    elif CompTools(first_search).n_els == 1:
            #        search_key = "elements"
            else:
                search_key = "formula"
        elif isinstance(search_for, str):
            if "-" in search_for:
                if "mp" in search_for:
                    search_key = "material_id"
                else:
                    search_key = "chemsys"
            else:
                search_key = "formula"

        # include sub-phase diagrams if asked for (eg for hull analysis)
        if (search_key == "chemsys") and include_sub_phase_diagrams:
            chemsyses = []
            if isinstance(search_for, str):
                search_for = [search_for]

            for chemsys in search_for:
                els = chemsys.split("-")
                for i in range(len(els)):
                    for sub_els in combinations(els, i + 1):
                        chemsyses.append("-".join(sorted(sub_els)))

            search_for = chemsyses
            docs = mpr.summary.search(
            chemsys=search_for, energy_above_hull=(0, max_Ehull)
            )

        # query MP based on a search for compounds containing at least these elements
        # if search_key == "elements":
        #    docs = mpr.summary.search(
        #        elements=search_for, energy_above_hull=(0, max_Ehull)
        #    )
        # query MP based on a search for compounds in these chemical systems
        elif search_key == "chemsys":
            docs = mpr.summary.search(
                chemsys=search_for, energy_above_hull=(0, max_Ehull)
            )
        # query MP based on a search for compounds having these formulas
        elif search_key == "formula":
            docs = mpr.summary.search(
                formula=search_for, energy_above_hull=(0, max_Ehull)
            )
        # query MP based on a search for entries w/ these IDs
        elif search_key == "material_id":
            if isinstance(search_for, str):
                search_for = [search_for]
            docs = mpr.summary.search(
                material_ids=search_for, energy_above_hull=(0, max_Ehull)
            )

        # convert returned documents into dictionaries
        d = {}
        for doc in docs:
            d_doc = doc.dict()
            mpid = d_doc["material_id"]
            # mpid will be the front key in this dict since it's unique
            tmp = {}
            for k, v in d_doc.items():
                if k == "material_id":
                    continue
                # map notable keys to shorter names
                if k in long_to_short_keys:
                    tmp[long_to_short_keys[k]] = v
                elif k in properties:
                    tmp[k] = v
            # include a clean formula
            tmp["cmpd"] = CompTools(tmp["formula_pretty"]).clean
            d[mpid] = tmp

            # get a list of all compounds in our query for parsing
            compounds = sorted(list(set([d[mpid]["cmpd"] for mpid in d])))

            # figure out which mpid belongs to the ground-state for each formula
            gs = {}
            for c in compounds:
                mpids = [mpid for mpid in d if d[mpid]["cmpd"] == c]
                energy_key = "E_mp" if len(CompTools(c).els) == 1 else "Ef_mp"
                energies = [d[mpid][energy_key] for mpid in mpids]
                gs_mpid = mpids[np.argmin(energies)]
                gs[c] = gs_mpid

            # add a flag for whether each entry is the ground state
            for mpid, data in d.items():
                d[mpid]["is_gs"] = True if mpid == gs[data["cmpd"]] else False
                energy_key = (
                    "E_mp" if len(CompTools(data["cmpd"]).els) == 1 else "Ef_mp"
                )
                d[mpid]["dE_gs"] = data[energy_key] - d[gs[data["cmpd"]]][energy_key]

            # if you ask only for ground-states, remove other polymorphs
            if max_polymorph_energy == 0:
                d = {mpid: d[mpid] for mpid in d if d[mpid]["is_gs"]}

            # remove certain higher energy polymorphs based on a max polymorphs per formula criterion
            if max_strucs_per_cmpd:
                mpids_to_keep = []
                for c in compounds:
                    mpids = [mpid for mpid in d if d[mpid]["cmpd"] == c]
                    dEs = [d[mpid]["dE_gs"] for mpid in mpids]

                    sorted_indices = np.argsort(dEs)
                    sorted_ids = [mpids[i] for i in sorted_indices]
                    mpids_to_keep += sorted_ids[:max_strucs_per_cmpd]

                d = {mpid: d[mpid] for mpid in d if mpid in mpids_to_keep}

            # remove entries that have too many sites in their structures
            if max_sites_per_structure:
                d = {
                    mpid: d[mpid]
                    for mpid in d
                    if d[mpid]["nsites"] <= max_sites_per_structure
                }

            # remove entries that have too high an energy above the ground-state polymorph
            if max_polymorph_energy:
                d = {
                    mpid: d[mpid]
                    for mpid in d
                    if np.round(d[mpid]["dE_gs"], 5) <= max_polymorph_energy
                }
        return d

    def get_structures_by_material_id(self, material_ids):
        """
        Args:
            material_ids (list)
                list of MP IDs (str)

        Returns:
            {mpid (str) : Structure (dict)}
        """

        docs = self.mpr.summary.search(
            material_ids=material_ids, fields=["structure", "material_id"]
        )
        d = {}
        for doc in docs:
            d_doc = doc.dict()
            mpid = d_doc["material_id"]
            d[mpid] = d_doc["structure"]
        return d

    def get_entries_for_chemsys(self, chemsys, thermo_types=["GGA_GGA+U", "R2SCAN"]):
        """
        Args:
            chemsys (str)
                chemical system of elements joined by "-"

        Returns:
            {chemsys (str) : [entries (dicts)]}
        """
        mpr = self.mpr
        entries = mpr.get_entries_in_chemsys(
            elements=chemsys.split("-"),
            additional_criteria={"thermo_types": thermo_types},
        )
        for entry in entries:
            entry.data["formula"] = CompTools(entry.composition.reduced_formula).clean
            entry.data["queried"] = True

        return {e.entry_id: e.as_dict() for e in entries}


class MPLegacyQuery(object):
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
        self.mpr = old_MPRester(api_key)

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
                    for els in combinations(elements, i + 1):
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
                        for els in combinations(elements, i + 1):
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
    my_new_api_key = "***REMOVED***"
    mpq = MPQuery(api_key=my_new_api_key)
    """
    out = mpq.get_data(
        search_for=["Sr-Zr-S"], include_sub_phase_diagrams=False, only_gs=True
    )
    """
    s = mpq.get_structures_by_material_id(material_ids=["mp-390"])
    return mpq, out, s


if __name__ == "__main__":
    mpq, out, s = main()

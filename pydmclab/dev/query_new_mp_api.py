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

        api_key = api_key if api_key else "YOUR_API_KEY"

        self.api_key = api_key
        self.mpr = new_MPRester(api_key)

    @property
    def available_fields(self):
        return self.mpr.summary.available_fields

    @property
    def typical_fields(self):
        return [
            "formula_pretty",
            "volume",
            "nsites",
            "material_id",
            "energy_per_atom",
            "uncorrected_energy_per_atom",
            "formation_energy_per_atom",
            "energy_above_hull",
            "equilibrium_reaction_energy_per_atom",
            "decomposes_to",
            "band_gap",
            "is_magnetic",
            "symmetry",
        ]

    def get_data_for_comp(self, comp):
        all_chemsyses = None
        all_formulas = None
        if isinstance(comp, str):
            if "-" in comp:
                chemsys = comp
                all_chemsyses = []
                elements = chemsys.split("-")
                for i in range(len(elements)):
                    for els in itertools.combinations(elements, i + 1):
                        all_chemsyses.append("-".join(sorted(els)))
            else:
                all_formulas = [CompTools(comp).clean]
        elif isinstance(comp, list):
            if "-" in comp[0]:
                all_chemsyses = []
                for chemsys in comp:
                    elements = chemsys.split("-")
                    for i in range(len(elements)):
                        for els in itertools.combinations(elements, i + 1):
                            all_chemsyses.append("-".join(sorted(els)))
                all_chemsyses = sorted(list(set(all_chemsyses)))
            else:
                all_formulas = [CompTools(c).pretty for c in comp]
        if all_formulas:
            from_mp = self.mpr.summary.search(
                formula_pretty=all_formulas, fields=self.typical_fields
            )
        elif all_chemsyses:
            from_mp = self.mpr.summary.search(
                chemsys=all_chemsyses, fields=self.typical_fields
            )

        return from_mp

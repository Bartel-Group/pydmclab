from pydmclab.core.comp import CompTools
from pydmclab.utils.handy import eVat_to_kJmol, kJmol_to_eVat


class ReactionEnergy(object):

    """
    *** This is a work in progress ***

    @TODO:
        - write tests/demo
        - incorporate filler
        - incorporate normalization
        - incorporate balance checks

    """

    def __init__(
        self,
        formation_energies,
        reactants,
        products,
        open_to=[],
        norm="atom",
        allowed_filler=["O2", "N2"],
    ):
        """

        Args:
            formation_energies (dict): {formula (str): formation energy (eV/at)}
                - formation energies should account for chemical potentials (e.g., due to partial pressures)
            reactants (dict): {formula (str) : stoichiometry (int)}
            products (dict): {formula (str) : stoichiometry (int)}
            open_to (list): list of elements to be considered "open" in the reaction. Defaults to None.
            norm (str, dict): if 'atom', then calculate reaction energy per atom of products formed
                - otherwise, specify a basis like: {'O' : 3} to normalize per three moles of O in the products formed
        """

        self.formation_energies = formation_energies
        self.reactants = reactants
        self.products = products
        self.open_to = open_to
        self.norm = norm

    @property
    def species(self):
        """
        puts the reactants and products in the same dictionary

        Returns:
            {formula (str) : {'side' : 'left' for reactants, 'right' for products},
                              'amt' : stoichiometry (float) in reaction}}
        """
        species = {}
        reactants, products = self.reactants, self.products
        energies = self.formation_energies
        for r in reactants:
            species[CompTools(r).clean] = {
                "side": "left",
                "amt": reactants[r],
                "Ef": energies[r],
            }
        for p in products:
            species[CompTools(p).clean] = {
                "side": "right",
                "amt": products[p],
                "Ef": energies[p],
            }
        return species

    def check_species_balance(self, species):
        """
        Args:
            species (dict): {formula (str) : {'side' : 'left' for reactants, 'right' for products},
                              'amt' : stoichiometry (float) in reaction}}
        Returns:
            {element (str) : 0 if balanced, else < 0 if more on left, > 0 if more on right}
        """

        involved_elements = [CompTools(formula).els for formula in species]
        involved_elements = sorted(
            list(set([item for sublist in involved_elements for item in sublist]))
        )
        balance = {}
        for el in involved_elements:
            left, right = 0, 0
            for formula in species:
                if el in CompTools(formula).els:
                    if species[formula]["side"] == "left":
                        left += CompTools(formula).els[el] * species[formula]["amt"]
                    elif species[formula]["side"] == "right":
                        right += CompTools(formula).els[el] * species[formula]["amt"]
            balance[el] = left + right

        return left + " --> " + right

    @property
    def E_rxn(self):
        species = self.species
        dE_rxn = 0
        for formula in species:
            if CompTools(formula).n_els == 1:
                continue

            if species[formula]["side"] == "left":
                sign = -1
            elif species[formula]["side"] == "right":
                sign = 1
            else:
                raise ValueError
            coef = species[formula]["amt"]
            Ef = species[formula]["Ef"]
            Ef = eVat_to_kJmol(Ef, formula)
            dE_rxn += sign * coef * Ef

        return dE_rxn

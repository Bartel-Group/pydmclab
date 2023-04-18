import os
import numpy as np

from pydmclab.core.comp import CompTools
from pydmclab.utils.handy import read_json, write_json

from scipy.spatial import ConvexHull
from scipy.optimize import minimize


class MixingHull(object):
    def __init__(
        self,
        input_energies,
        energy_key="E_per_at",
        varying_element="Li",
        end_members=["Li2FeP2S6", "FeP2S6"],
        shared_element_basis="Fe",
    ):
        input_energies = {
            CompTools(k).clean: {"E": input_energies[k][energy_key]}
            for k in input_energies
        }
        self.end_members = [CompTools(c).clean for c in end_members]
        self.varying_element = varying_element
        self.shared_element_basis = shared_element_basis
        self.input_energies = input_energies
        error_check = self._compatibility_check

    @property
    def _compatibility_check(self):

        end_members = self.end_members
        varying_element = self.varying_element

        els_end_1 = CompTools(end_members[0]).els
        els_end_2 = CompTools(end_members[1]).els
        all_els = list(set(els_end_1 + els_end_2))

        shared_els = list(set(els_end_1).intersection(els_end_2))
        unshared_els = [el for el in all_els if el not in shared_els]
        if (len(unshared_els) != 1) and (unshared_els[0] != varying_element):
            raise ValueError("The varying element must be the only unshared element.")
        return

    @property
    def shared_element_amt_basis(self):
        end_members = self.end_members
        shared_element_basis = self.shared_element_basis
        amt_1 = CompTools(end_members[0]).stoich(shared_element_basis)
        amt_2 = CompTools(end_members[1]).stoich(shared_element_basis)
        if amt_2 > amt_1:
            return amt_2
        else:
            return amt_1

    @property
    def relevant_compounds(self):
        compounds = list(self.input_energies.keys())
        endmembers = self.end_members
        varying_element = self.varying_element
        elements_in_relevant_compounds = [CompTools(e).els for e in endmembers]
        elements_in_relevant_compounds = list(
            set(
                [item for sublist in elements_in_relevant_compounds for item in sublist]
            )
        )
        elements_that_must_be_present = [
            el for el in elements_in_relevant_compounds if el != varying_element
        ]

        relevant_compounds = []
        for c in compounds:
            els = CompTools(c).els
            counter = 0
            for el in elements_that_must_be_present:
                if el in els:
                    counter += 1
            if counter == len(elements_that_must_be_present):
                relevant_compounds.append(c)
            elif counter == len(elements_in_relevant_compounds) - 1:
                if varying_element in els:
                    relevant_compounds.append(c)

        return relevant_compounds

    @property
    def energies_with_fractional_composition(self):
        shared_element_amt_basis = self.shared_element_amt_basis
        shared_element_basis = self.shared_element_basis
        varying_element = self.varying_element
        input_energies = self.input_energies
        relevant_compounds = self.relevant_compounds
        input_energies = {c: input_energies[c] for c in relevant_compounds}
        for c in input_energies:
            amt_of_shared_el = CompTools(c).stoich(shared_element_basis)
            factor = shared_element_amt_basis / amt_of_shared_el
            input_energies[c]["factor"] = factor
            n_varying = CompTools(c).stoich(varying_element) * factor
            n_atoms_basis = CompTools(c).n_atoms * factor

            input_energies[c]["factor"] = factor
            input_energies[c]["n_varying"] = n_varying
            input_energies[c]["n_atoms_basis"] = n_atoms_basis
        return input_energies

    @property
    def mixing_energies(self):
        input_energies = self.energies_with_fractional_composition
        end_members = self.end_members
        E_left = input_energies[end_members[0]]["E"]
        E_right = input_energies[end_members[1]]["E"]
        n_left = input_energies[end_members[0]]["n_varying"]
        n_right = input_energies[end_members[1]]["n_varying"]
        basis_left = input_energies[end_members[0]]["n_atoms_basis"]
        basis_right = input_energies[end_members[1]]["n_atoms_basis"]

        for c in input_energies:
            E = input_energies[c]["E"]
            n = input_energies[c]["n_varying"]
            basis = input_energies[c]["n_atoms_basis"]

            if n_left != 0:
                x = n / n_left
                zero = "right"
            elif n_right != 0:
                x = n / n_right
                zero = "left"

            if zero == "right":
                E_mix = E * basis - (
                    x * E_left * basis_left + (1 - x) * E_right * basis_right
                )
            elif zero == "left":
                E_mix = E * basis - (
                    (1 - x) * E_left * basis_left + x * E_right * basis_right
                )
            E_mix = E_mix / basis
            input_energies[c]["E_mix"] = E_mix
            input_energies[c]["x"] = x
            input_energies[c]["zero"] = zero

        return input_energies

    @property
    def sorted_compounds(self):
        """
        Returns:
            alphabetized list of compounds (str) in specified chemical space
        """
        return sorted(list(self.mixing_energies.keys()))

    def amts_matrix(self, compounds="all", chemical_space="all"):
        """
        Args:
            compounds (str or list) - if 'all', use all compounds; else use specified list
                - note: this gets modified for you as needed to minimize cpu time
            chemical_space - if 'all', use entire space; else use specified tuple
                - note: this gets modified for you as needed to minimize cpu time
        Returns:
            matrix (2D array) with the fractional composition of each element in each compound (float)
                - each row is a different compound (ordered going down alphabetically)
                - each column is a different element (ordered across alphabetically)
        """
        if chemical_space == "all":
            chemical_space = ["A", "B"]
        mixing_energies = self.mixing_energies
        if compounds == "all":
            compounds = self.sorted_compounds
        A = np.zeros((len(compounds), len(chemical_space)))
        for row in range(len(compounds)):
            compound = compounds[row]
            for col in range(len(chemical_space)):
                A[row, col] = mixing_energies[compound]["x"]
        return A

    def formation_energy_array(self, compounds="all"):
        """
        Args:
            compounds (str or list) - if 'all', use all compounds; else use specified list
                - this gets modified for you as needed to minimize cpu time
        Returns:
            1D array of formation energies (float) for each compound ordered alphabetically
        """
        mixing_energies = self.mixing_energies
        if compounds == "all":
            compounds = self.sorted_compounds
        return np.array([mixing_energies[c]["E_mix"] for c in compounds])

    def hull_input_matrix(self, compounds="all", chemical_space="all"):
        """
        Args:
            compounds (str or list) - if 'all', use all compounds; else use specified list
                - this gets modified for you as needed to minimize cpu time
            chemical_space - if 'all', use entire space; else use specified tuple
                - this gets modified for you as needed to minimize cpu time
        Returns:
            amts_matrix, but replacing the last column with the formation energy
                - this is because convex hulls are defined by (n-1) composition axes
                    - e.g., in a A-B phase diagram, specifying the fractional composition of A sets the composition of B
        """
        A = self.amts_matrix(compounds, chemical_space)
        b = self.formation_energy_array(compounds)
        X = np.zeros(np.shape(A))
        for row in range(np.shape(X)[0]):
            for col in range(np.shape(X)[1] - 1):
                X[row, col] = A[row, col]
            X[row, np.shape(X)[1] - 1] = b[row]
        return X

    @property
    def hull(self):
        """
        Returns:
            scipy.spatial.ConvexHull object for all compounds in a chemical space
        """
        return ConvexHull(self.hull_input_matrix())

    @property
    def hull_points(self):
        """
        Returns:
            array of (composition, formation energy) points (2-element tuple) fed to ConvexHull
        """
        return self.hull.points

    @property
    def hull_vertices(self):
        """
        Returns:
            array of indices (int) in hull_points corresponding with the points that are on the hull
        """
        return self.hull.vertices

    @property
    def hull_simplices(self):
        """
        Returns:
            indices of points forming the simplical facets of the convex hull.
        """
        return self.hull.simplices

    @property
    def stable_compounds(self):
        """
        Returns:
            list of compounds (str) that correspond with vertices on the hull
                - these are stable compounds
        """
        mixing_energies = self.mixing_energies
        hull_vertices = self.hull_vertices
        compounds = self.sorted_compounds
        return [
            compounds[i]
            for i in hull_vertices
            if mixing_energies[compounds[i]]["E_mix"] <= 0
        ]

    @property
    def unstable_compounds(self):
        """
        Args:
        Returns:
            list of compounds that do not correspond with vertices (str)
                - these are "above" the hull
        """
        compounds = self.sorted_compounds
        stable_compounds = self.stable_compounds
        return [c for c in compounds if c not in stable_compounds]

    @property
    def results(self):
        stable_compounds = self.stable_compounds
        unstable_compounds = self.unstable_compounds
        mixing_energies = self.mixing_energies
        for c in stable_compounds:
            mixing_energies[c]["stability"] = True
        for c in unstable_compounds:
            mixing_energies[c]["stability"] = False

        return mixing_energies


def main():
    fjson = os.path.join("/users/cbartel", "Downloads", "Li2MP2S6_gga_gs_E_per_at.json")
    d = read_json(fjson)

    out = {}
    for M in ["Mn", "Fe", "Co", "Ni"]:
        hull = MixingHull(
            d,
            energy_key="E_per_at",
            varying_element="Li",
            end_members=["Li2%sP2S6" % M, "%sP2S6" % M],
            shared_element_basis=M,
        )
        results = hull.results
        for c in results:
            out[c] = results[c]

    for c in out:
        print("\n")
        print(c)
        print("Emix : %.5f" % out[c]["E_mix"])
        print("x : %.2f" % out[c]["x"])
        print("stability : %s" % out[c]["stability"])

    write_json(out, fjson.replace(".json", "_cjb.json"))
    return out


if __name__ == "__main__":
    out = main()

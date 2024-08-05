import os
import numpy as np

from scipy.spatial import ConvexHull
from scipy.optimize import minimize

import multiprocessing as multip

from pymatgen.analysis.phase_diagram import PDEntry, CompoundPhaseDiagram
from pymatgen.core.composition import Composition

from pydmclab.core.comp import CompTools
from pydmclab.utils.handy import read_json, write_json
from pydmclab.core.energies import ReactionEnergy

from typing import Literal


# HERE = os.path.dirname(os.path.abspath(__file__))

"""
Contents:

    1) Classes allowing you to compute decomposition energies using the convex hull analysis
        Standard approach:
            - collect formation energies for all compounds in given chemical system(s)
                - using your own calculations, MPQuery, or a combination
            - map these formation energies into various phase diagrams
                - w/ GetHullInputData
            - compute convex hulls for each phase diagram
                - w/ AnalyzeHull
            - if you are doing this for many compounds / chemical spaces, you want to parallelize it
                - w/ ParallelHullAnalysis
    2) Class for having compound end-members (mixing hulls)
        - e.g., for alloys, solid solutions, voltage curves, etc

"""


class GetHullInputData(object):
    """
    Generates hull-relevant data
    Designed to be executed once all compounds and ground-state formation energies are known
    """

    def __init__(self, compound_to_energy, formation_energy_key):
        """
        Args:
            compound_to_energy (dict)
                {formula (str) :
                    {formation_energy_key (str) :
                        formation energy (float, eV/atom)}}
                it's ok to have other keys in this dictionary, just have to have a formation_energy_key that maps to formation energies
                these formation energies can be 0 K, at some temperature, or computed in any way that you want, they just have to be compatible w/ one another

            formation_energy_key (str)
                key within compound_to_energy to use for formation energy

        Returns:
            dictionary of
                {formula (str) : formation energy (float, eV/atom)}
                    note that elements are removed from this dictionary, they will be added back in later
                    this is a legacy thing that is kind of tedious but (probably) needs to be done
        """
        # sanitize
        self.compound_to_energy = {
            CompTools(k).clean: compound_to_energy[k][formation_energy_key]
            for k in compound_to_energy
            if len(CompTools(k).els) > 1
        }

    @property
    def compounds(self):
        """
        Returns:
            list of compounds (str)
        """
        return list(self.compound_to_energy.keys())

    @property
    def chemical_spaces_and_subspaces(self):
        """
        Returns:
            list of unique chemical spaces (tuple)
                a chemical space would be (Ca, O, Ti) for CaTiO3
        """
        compounds = self.compounds
        return list(set([tuple(CompTools(c).els) for c in compounds]))

    @property
    def chemical_subspaces(self):
        """
        Returns:
            list of unique chemical spaces (tuple) that do not define convex hull spaces
                (Ca, O, Ti) is the space of CaTiO3 and Ca2TiO4
                    if CaTiO3 and CaO are found in compound_to_energy, (Ca, O) is a subspace because it does not include CaTiO3
        """
        all_spaces = self.chemical_spaces_and_subspaces
        subspaces = [
            all_spaces[i]
            for i in range(len(all_spaces))
            for j in range(len(all_spaces))
            if set(all_spaces[i]).issubset(all_spaces[j])
            if all_spaces[i] != all_spaces[j]
        ]
        return list(set(subspaces))

    def hull_spaces(self, fjson=False, remake=False, write=False):
        """
        this can take a while to compute, so sometimes it's useful to write it to a json file

        Args:
            fjson (str)
                file name to write hull spaces to

            remake (bool)
                if True, repeat analysis; if False, read json

            write (bool)
                if True, write json; if False, don't write json

        Returns:
            list of unique chemical spaces (set) that do define convex hull spaces
                these are the convex hulls that must be computed
                so if you are working with binaries and ternaries, but no quaternaries, hull spaces will be maximally 3 dimensions (elements)
        """
        if not fjson:
            fjson = "hull_spaces.json"
        if not remake and os.path.exists(fjson):
            d = read_json(fjson)
            return d["hull_spaces"]
        chemical_spaces_and_subspaces = self.chemical_spaces_and_subspaces
        chemical_subspaces = self.chemical_subspaces

        # filter out subspaces and filter out single-element spaces
        d = {
            "hull_spaces": [
                s
                for s in chemical_spaces_and_subspaces
                if s not in chemical_subspaces
                if len(s) > 1
            ]
        }
        if write:
            d = write_json(d, fjson)
        return d["hull_spaces"]

    def hullin_data(self, fjson=False, remake=True):
        """

        Args:
            fjson (str)
                file name to write hull data to

            remake (bool)
                if True, run analysis + write json; if False, read json

        Returns:
            dict of
                {chemical space (str) :
                    {formula (str) :
                        {'E' : formation energy (float),
                         'amts' : {el (str) : fractional amt of el in formula (float) for el in space}}
                    for all relevant formulas including elements}
                elements are automatically given formation energy = 0
                chemical space is now in 'el1_el2_...' format to be jsonable
                each "chemical space" is a convex hull that must be computed
        """
        if not fjson:
            fjson = "hull_input_data.json"
        if os.path.exists(fjson) and not remake:
            return read_json(fjson)

        hullin_data = {}
        hull_spaces = self.hull_spaces()
        compounds = self.compounds
        compound_to_energy = self.compound_to_energy
        for space in hull_spaces:
            for el in space:
                # give elements 0 formation energy
                compound_to_energy[el] = 0

            # get the compounds belonging to this space (+ the elements that define the space)
            relevant_compounds = [
                c for c in compounds if set(CompTools(c).els).issubset(set(space))
            ] + list(space)

            # assemble into dictionary that includes the mole fraction of each element in each compound
            hullin_data["_".join(list(space))] = {
                c: {
                    "E": compound_to_energy[c],
                    "amts": {el: CompTools(c).mol_frac(el=el) for el in space},
                }
                for c in relevant_compounds
            }
        write_json(hullin_data, fjson)
        return read_json(fjson)


class AnalyzeHull(object):
    """
    Determines stability for one chemical space (hull)
    Designed to be parallelized over chemical spaces
    Output can be a dictionary with hull results for one chemical space or can be run for specified compounds
    """

    def __init__(self, hullin_data, chemical_space):
        """
        Args:
            hullin_data (dict)
                dictionary generated in GetHullInputData().hullin_data (includes at least one chemical space as a key)

            chemical_space (str)
                chemical space to analyze in 'el1_el2_...' (alphabetized) format
                    to compute Ca-Ti-O phase diagram, chemical_space would be 'Ca_O_Ti'

        Returns:
            hullin_data (dict) - hullin_data for a single chemical_space
                {compound (str) :
                    {'E' : formation energy (float),
                        'amts' :
                        {el (str) :
                            fractional amt of el in compound (float)
                                for el in chemical_space}}}})}

        """
        hullin_data = hullin_data[chemical_space]

        self.hullin_data = hullin_data

        self.chemical_space = tuple(chemical_space.split("_"))

    @property
    def sorted_compounds(self):
        """
        Returns:
            alphabetized list of compounds (str) in specified chemical space
        """
        return sorted(list(self.hullin_data.keys()))

    def amts_matrix(self, compounds="all", chemical_space="all"):
        """
        Args:
            compounds (str or list)
                if 'all', use all compounds; else use specified list
                note: this gets modified for you as needed to minimize cpu time

            chemical_space
                if 'all', use entire space; else use specified tuple
                note: this gets modified for you as needed to minimize cpu time

        Returns:
            matrix (2D array) with the fractional composition of each element in each compound (float)
                each row is a different compound (ordered going down alphabetically)
                each column is a different element (ordered across alphabetically)
        """
        # sometimes this gets called for a particular subspace
        if chemical_space == "all":
            chemical_space = self.chemical_space
        hullin_data = self.hullin_data

        # sometimes we use this with a specified set of compounds (e.g., stable ones)
        if compounds == "all":
            compounds = self.sorted_compounds

        # initialize the matrix with the format we want
        A = np.zeros((len(compounds), len(chemical_space)))
        for row in range(len(compounds)):
            compound = compounds[row]
            for col in range(len(chemical_space)):
                el = chemical_space[col]
                # populate with the mole fractions
                A[row, col] = hullin_data[compound]["amts"][el]
        return A

    def formation_energy_array(self, compounds="all"):
        """
        Args:
            compounds (str or list)
                if 'all', use all compounds; else use specified list
                    this gets modified for you as needed to minimize cpu time

        Returns:
            1D array of formation energies (float) for each compound ordered alphabetically
        """
        hullin_data = self.hullin_data
        # sometimes we grab this for a subset of compounds (e.g., stable ones)
        if compounds == "all":
            compounds = self.sorted_compounds
        return np.array([hullin_data[c]["E"] for c in compounds])

    def hull_input_matrix(self, compounds="all", chemical_space="all"):
        """
        Args:
            compounds (str or list)
                if 'all', use all compounds; else use specified list
                    this gets modified for you as needed to minimize cpu time
            chemical_space
                if 'all', use entire space; else use specified tuple
                    this gets modified for you as needed to minimize cpu time

        Returns:
            amts_matrix, but replacing the last column with the formation energy
                this is because convex hulls are defined by (n-1) composition axes
                    e.g., in a A-B phase diagram, specifying the fractional composition of A sets the composition of B
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
        return ConvexHull(self.hull_input_matrix(compounds="all", chemical_space="all"))

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
                these are stable compounds
        """
        hullin_data = self.hullin_data
        hull_vertices = self.hull_vertices
        compounds = self.sorted_compounds
        # note: we need to specify E < 0 because a mathematical hull would be enclosed above 0
        return [
            compounds[i] for i in hull_vertices if hullin_data[compounds[i]]["E"] <= 0
        ]

    @property
    def unstable_compounds(self):
        """
        Returns:
           list of compounds that do not correspond with vertices (str)
               these are "above" the hull
        """
        compounds = self.sorted_compounds
        stable_compounds = self.stable_compounds
        return [c for c in compounds if c not in stable_compounds]

    def competing_compounds(self, compound):
        """
        Args:
            compound (str)
                the compound (str) to determine competing compounds for

        Returns:
            list of compounds (str) that may participate in the decomposition reaction for the input compound
                these must have the same elements as the input compound or a subset of those elements
        """
        compounds = self.sorted_compounds
        if compound in self.unstable_compounds:
            # if a compound is unstable, its stability-defining reaction includes only stable compounds
            compounds = self.stable_compounds

        # filter based on composition
        competing_compounds = [
            c
            for c in compounds
            if c != compound
            if set(CompTools(c).els).issubset(CompTools(compound).els)
        ]
        return competing_compounds

    def A_for_decomp_solver(self, compound, competing_compounds):
        """
        Args:
            compound (str)
                the compound (str) to analyze

            competing_compounds (list)
                list of compounds (str) that may participate in the decomposition reaction for the input compound

        Returns:
            matrix (2D array) of elemental amounts (float) used for implementing molar conservation during decomposition solution
                i.e., a decomposition reaction has to conserve the total number of each element on both sides of the reaction
        """
        chemical_space = tuple(CompTools(compound).els)
        atoms_per_fu = [CompTools(c).n_atoms for c in competing_compounds]
        A = self.amts_matrix(competing_compounds, chemical_space)
        for row in range(len(competing_compounds)):
            for col in range(len(chemical_space)):
                A[row, col] *= atoms_per_fu[row]
        return A.T

    def b_for_decomp_solver(self, compound):
        """
        Args:
            compound (str)
                the compound (str) to analyze

        Returns:
            array of elemental amounts (float) used for implementing molar conservation during decomposition solution
        """
        chemical_space = tuple(CompTools(compound).els)
        return np.array([CompTools(compound).stoich(el) for el in chemical_space])

    def Es_for_decomp_solver(self, competing_compounds):
        """
        Args:
            competing_compounds (list)
                list of compounds (str) that may participate in the decomposition reaction for the input compound

        Returns:
            array of formation energies per formula unit (float) used for minimization problem during decomposition solution
                these have to be Ef per f.u. because we are computing reaction energies
        """
        atoms_per_fu = [CompTools(c).n_atoms for c in competing_compounds]
        Es_per_atom = self.formation_energy_array(competing_compounds)
        return [
            Es_per_atom[i] * atoms_per_fu[i] for i in range(len(competing_compounds))
        ]

    def decomp_solution(self, compound):
        """
        Args:
            compound (str)
                the compound (str) to analyze

        Returns:
            scipy.optimize.minimize result
                for finding the linear combination of competing compounds that minimizes the competing formation energy
                    i.e., the fractional composition of decomposition products leading to the most positive deecomposition energy
        """
        competing_compounds = self.competing_compounds(compound)
        A = self.A_for_decomp_solver(compound, competing_compounds)
        b = self.b_for_decomp_solver(compound)
        Es = self.Es_for_decomp_solver(competing_compounds)

        # initialize with zeros
        n0 = [0 for c in competing_compounds]

        # max coefficient is the number of atoms in the compound of interest
        max_bound = CompTools(compound).n_atoms
        bounds = [(0, max_bound) for c in competing_compounds]

        # function to minimize (what the compound competes with)
        def competing_formation_energy(nj):
            nj = np.array(nj)
            return np.dot(nj, Es)

        # subject to molar conservation
        constraints = [{"type": "eq", "fun": lambda x: np.dot(A, x) - b}]

        # opt paramaters
        maxiter, disp = 1000, False

        # slowly relax tolerance in case minimize can't converge
        for tol in [1e-6, 1e-4, 1e-3, 1e-2]:
            solution = minimize(
                competing_formation_energy,
                n0,
                method="SLSQP",
                bounds=bounds,
                constraints=constraints,
                tol=tol,
                options={"maxiter": maxiter, "disp": disp},
            )
            if solution.success:
                return solution
        return solution

    def decomp_products(self, compound):
        """
        Args:
            compound (str)
                the compound (str) to analyze

        Returns:
            dictionary of
                {competing compound (str) :
                    {'amt' : stoich weight in decomp rxn (float),
                     'E' : formation energy (float)}
                        for all compounds in the stability-defining reaction}

                np.nan if decomposition analysis fails
        """
        hullin_data = self.hullin_data
        competing_compounds = self.competing_compounds(compound)

        # protecting trivial case where there are no competing compounds
        if (len(competing_compounds) == 0) or (
            np.max([CompTools(c).n_els for c in competing_compounds]) == 1
        ):
            return {
                el: {"amt": CompTools(compound).stoich(el), "E": 0}
                for el in CompTools(compound).els
            }

        # result of optimization to find decomp products
        solution = self.decomp_solution(compound)
        if solution.success:
            resulting_amts = solution.x
        elif hullin_data[compound]["E"] > 0:
            # if E > 0, we don't need decomp products, just assume elemental decomposition (not strictly correct but good enough for these)
            return {
                el: {"amt": CompTools(compound).stoich(el), "E": 0}
                for el in CompTools(compound).els
            }
        else:
            # in case scipy.minimize fails to converge
            print(compound)
            print("\n\n\nFAILURE!!!!\n\n\n")
            print(compound)
            return np.nan

        # put irrelevant compounds to zero to make reaction more clear
        min_amt_to_show = 1e-4
        decomp_products = dict(zip(competing_compounds, resulting_amts))
        relevant_decomp_products = [
            k for k in decomp_products if decomp_products[k] > min_amt_to_show
        ]
        decomp_products = {
            k: (
                {"amt": decomp_products[k], "E": hullin_data[k]["E"]}
                if k in hullin_data
                else 0
            )
            for k in relevant_decomp_products
        }

        return decomp_products

    def decomp_energy(self, compound):
        """
        Args:
            compound (str)
                the compound (str) to analyze

        Returns:
            decomposition energy (float)
                if < 0, this is the magnitude of formation energy that compound could increase and still be on the hull
                if > 0, this is the energy above the hull
        """
        hullin_data = self.hullin_data
        decomp_products = self.decomp_products(compound)
        if isinstance(decomp_products, float):
            return np.nan
        decomp_energy = 0
        for k in decomp_products:
            # this is the energy of the competing phases (per mole)
            decomp_energy += (
                decomp_products[k]["amt"]
                * decomp_products[k]["E"]
                * CompTools(k).n_atoms
            )
        return (
            hullin_data[compound]["E"] * CompTools(compound).n_atoms - decomp_energy
        ) / CompTools(compound).n_atoms

    def cmpd_hull_output_data(self, compound):
        """
        Args:
            compound (str)
                formula to get data for

        Returns:
            stability data for a single compound
                {'Ef' : formation energy (float),
                'Ed' : decomposition energy (float),
                'rxn' : decomposition reaction (str),
                'stability' : stable (True) or unstable (False)}

        """
        data = {}
        hullin_data = self.hullin_data
        stable_compounds = self.stable_compounds
        c = compound
        if c in stable_compounds:
            stability = True
        else:
            stability = False
        Ef = hullin_data[c]["E"]
        Ed = self.decomp_energy(c)
        decomp_products = self.decomp_products(c)
        if isinstance(decomp_products, float):
            return np.nan

        # get the decomp reaction
        decomp_rxn = [
            "_".join([str(np.round(decomp_products[k]["amt"], 4)), k])
            for k in decomp_products
        ]
        decomp_rxn = " + ".join(decomp_rxn)
        data[c] = {"Ef": Ef, "Ed": Ed, "rxn": decomp_rxn, "stability": stability}
        return data[c]

    @property
    def hull_output_data(self):
        """
        Returns:
            stability data (dict) for all compounds in the specified chemical space
                {compound (str) :
                    {'Ef' : formation energy (float),
                     'Ed' : decomposition energy (float),
                     'rxn' : decomposition reaction (str),
                     'stability' : stable (True) or unstable (False)}}
        """
        compounds = self.stable_compounds + self.unstable_compounds

        all_hull_data = {c: self.cmpd_hull_output_data(c) for c in compounds}
        return all_hull_data


class ParallelHulls(object):
    """
    this allows you to parallelize the hullin_data and hullout_data generation over cores using multiprocessing

    """

    def __init__(
        self,
        compound_to_energy,
        formation_energy_key="Ef_mp",
        n_procs=4,
        data_dir=False,
        fresh_restart=False,
    ):
        """
         compound_to_energy (dict)
            {formula (str) :
                {formation_energy_key (str) :
                    formation energy (float, eV/atom)}}
            it's ok to have other keys in this dictionary, just have to have a formation_energy_key that maps to formation energies
            these formation energies can be 0 K, at some temperature, or computed in any way that you want, they just have to be compatible w/ one another

        formation_energy_key (str)
            key within compound_to_energy to use for formation energy

        n_procs (int) -
            number of cores to use
                if 'all', will use every core availble minus 1

        data_dir (str)
            directory to save data to (if False, writes to os.getcwd())

        fresh_restart (bool)
            if True, will re-compute everything from scratch
        """
        # how many processors
        self.n_procs = n_procs if n_procs != "all" else multip.cpu_count() - 1

        # where to save data
        if not data_dir:
            self.data_dir = os.getcwd()
        else:
            self.data_dir = data_dir

        # restart or not
        self.fresh_restart = fresh_restart

        ghid = GetHullInputData(
            compound_to_energy=compound_to_energy,
            formation_energy_key=formation_energy_key,
        )
        self.compound_to_energy = ghid.compound_to_energy
        self.compounds = ghid.compounds
        chemical_spaces = ghid.chemical_spaces_and_subspaces
        self.chemical_spaces = ["_".join(list(s)) for s in chemical_spaces]

    def hullin_from_space(self, space, verbose=True):
        """
        Function called at each parallelized generation of hull input file (same as GetHullInputData.hullin_data)

        Args:
            space (str)
                '_'.join(elements) (str) in chemical space

            verbose (bool)
                print space or not

        Returns:
            {compound (str) :
                {'E' : formation energy (float, eV/atom),
                 'amts' :
                    {el (str) : fractional amount of el in formula (float)}}}
        """
        compound_to_energy = self.compound_to_energy

        # print if you'd like
        if verbose:
            print(space)

        space = space.split("_")

        # give els zero formation energy
        for el in space:
            compound_to_energy[el] = 0

        # get the compositionally relevant compounds
        relevant_compounds = [
            c for c in compound_to_energy if set(CompTools(c).els).issubset(set(space))
        ] + list(space)

        # get the hull in data (formation energy, fractional amounts of elements)
        return {
            c: {
                "E": compound_to_energy[c],
                "amts": {el: CompTools(c).mol_frac(el=el) for el in space},
            }
            for c in relevant_compounds
        }

    def parallel_hullin(self, fjson=False, verbose=True):
        """
        Parallel generation of hull input data (same as GetHullInputData.hullin_data)

        Args:
            fjson (os.PathLike)
                path to write dictionary of hull input data

            verbose (bool)
                print space or not

        Returns:
            {chemical space (str) :
                {compound (str) :
                    {'E' : formation energy (float, eV/atom),
                     'amts' :
                        {el (str) : fractional amount of el in formula (float)}}}}
        """
        # start from scratch or not
        remake = self.fresh_restart

        # chemical spaces to loop through
        chemical_spaces = self.chemical_spaces

        if not fjson:
            # make a place to save this if not given
            fjson = os.path.join(self.data_dir, "hull_input_data.json")

        if os.path.exists(fjson) and not remake:
            # read instead of execute+write
            return read_json(fjson)

        hullin_data = {}

        # initialize multiprocessing
        pool = multip.Pool(processes=self.n_procs)

        # loop over chemical space and prepare inputs (parallel)
        results = [
            r
            for r in pool.starmap(
                self.hullin_from_space,
                [(space, verbose) for space in chemical_spaces],
            )
        ]
        # close multiprocessing
        pool.close()

        # zip spaces and results for each space
        hullin_data = dict(zip(chemical_spaces, results))

        # write
        write_json(hullin_data, fjson)
        return read_json(fjson)

    def smallest_space(self, hullin, formula, verbose=False):
        """
        function to parallelize finding the smallest hull to compute for each compound

        Args:
            hullin (dict)
                {space (str, '_'.join(elements)) :
                    {formula (str) :
                        {'E' : formation energy (float, eV/atom),
                        'amts' :
                            {element (str) : fractional amount of element in formula (float)}
                        }
                    }
                }

            formula (str)
                chemical formula

            verbose (bool)
                print formula or not

        Returns:
            chemical space (str, '_'.join(elements)) of the convex hull that is easiest to compute for a given compound
                for instance, if I want the decomp energy of CaO, there's no need to do the Ca-Ti-O hull, only the Ca-O one
        """
        # print if you like
        if verbose:
            print(formula)

        # get the unique chemical spaces
        spaces = sorted(list(hullin.keys()))

        # figure out which are relevant to the formula
        relevant = [s for s in spaces if formula in hullin[s]]

        # find the ones with the least number of elements
        sizes = [s.count("_") for s in relevant]
        small = [relevant[i] for i in range(len(sizes)) if sizes[i] == np.min(sizes)]

        # for the ones w/ equivalently small dimension, find the one with the least # of compounds
        sizes = [len(hullin[s]) for s in small]
        smallest = [small[i] for i in range(len(small)) if sizes[i] == np.min(sizes)]

        # return one of em
        return smallest[0]

    def smallest_spaces(self, hullin, fjson=False, verbose=False):
        """
        parallel generation of smallest spaces for each compound

        Args:
            hullin (dict)
                same as in smallest_space

            fjson (os.PathLike)
                path to write dictionary of smallest spaces

            verbose (bool)
                print formula or not

        Returns:
            {formula (str) :
                chemical space (str, '_'.join(elements)) of convex hull that is easiest to compute}
        """
        # start from scratch or not
        remake = self.fresh_restart

        if not fjson:
            # make a place to save this if not given
            fjson = os.path.join(self.data_dir, "smallest_spaces.json")

        if not remake and os.path.exists(fjson):
            # read instead of execute+write
            return read_json(fjson)

        # initialize multiprocessing
        pool = multip.Pool(processes=self.n_procs)

        # compounds to loop over
        compounds = self.compounds

        # list of smallest spaces
        smallest = [
            r
            for r in pool.starmap(
                self.smallest_space,
                [(hullin, compound, verbose) for compound in compounds],
            )
        ]
        # close multiprocessing
        pool.close()

        # zip compounds and smallest spaces
        data = dict(zip(compounds, smallest))

        # write it
        write_json(data, fjson)
        return read_json(fjson)

    def compound_stability(self, hullin, smallest_spaces, formula, verbose=False):
        """
        Args:
            hullin (dict)
                hull input dictionary (same as elsewhere)

            smallest_spaces (dict)
                {formula (str) :
                    smallest chemical space having formula (str, el1_el2_...)}

            formula (str)
                chemical formula to solve the hull for

            verbose (bool)
                print formula or not

        Returns:
            {'Ef' : formation energy (float, eV/atom),
             'Ed' : decomposition energy (float, eV/atom),
             'rxn' : decomposition reaction (str),
             'stability' : bool (True if on hull)}
        """
        # print if you like
        if verbose:
            print(formula)

        if CompTools(formula).n_els == 1:
            # if it's a single element, it's stable w/ 0 formation energy and decomposition energy
            return {"Ef": 0, "Ed": 0, "stability": True, "rxn": "1_%s" % formula}

        # otherwise, solve the hull for that formula
        space = smallest_spaces[formula]
        ah = AnalyzeHull(hullin, space)
        return ah.cmpd_hull_output_data(formula)

    def parallel_hullout(
        self,
        hullin,
        smallest_spaces,
        compounds="all",
        fjson=False,
        remake=False,
        verbose=False,
    ):
        """
        Args:
            hullin (dict)
                hull input dictionary (same as elsewhere)

            smallest_spaces (dict)
                {formula (str) : smallest chemical space having formula (str)}

            compounds (str or list)
                if 'all', use every compound, else specify which compounds to compute

            fjson (os.PathLike)
                path to write dictionary of hull output data

            reemake (bool)
                remake the data or not

            verbose (bool)
                print formula or not

        Returns:
            {formula (str) :
                {'Ef' : formation energy (float, eV/atom),
                'Ed' : decomposition energy (float, eV/atom),
                'rxn' : decomposition reaction (str),
                'stability' : bool (True if on hull)}
                }
        """
        if not fjson:
            # make a place to save this if not given
            fjson = os.path.join(self.data_dir, "hullout.json")

        # start from scratch or not
        if not remake and os.path.exists(fjson):
            return read_json(fjson)

        # initialize multiprocessing
        pool = multip.Pool(processes=self.n_procs)

        # use all compounds if specified
        if compounds == "all":
            compounds = sorted(list(smallest_spaces.keys()))

        # make list of stability dicts (hull results)
        results = [
            r
            for r in pool.starmap(
                self.compound_stability,
                [
                    (hullin, smallest_spaces, compound, verbose)
                    for compound in compounds
                ],
            )
        ]

        # close multiprocessing
        pool.close()

        # zip compounds and results
        data = dict(zip(compounds, results))

        # write it
        write_json(data, fjson)

        return read_json(fjson)


class MixingHull(object):
    """
    For performing the hull analysis w/ non-elemental end-members
    for computing mixing energies along one dimension
        i.e., between two end members
        if both end members are elements, this is a conventional hull so use GetHullInputData + AnalyzeHull
        this class is if one or both end members are compounds
    @TO-DO:
        - generalize to N dimensions (or at least 3)
    Some notes on normalizing energies:
        We are considering the mixing of two compounds, A (left end member) and B (right end member)
            Compunds in this mixing hull, C, must have a chemical formula that can be written as A_{1-x}B_{x}
            Therefore, mixing energies should be computed on a per A_{1-x}B_{x} formula unit basis
                (1-x) A + (x) B --> C (A_{1-x}B_{x})
        A and B should adhere to the same "basis". There are a few cases:
            1) if A and B share no elements (e.g., A = MnO2, B = Li), then they necessarily adhere to the same basis
            2) if A and B share some (or all) elements, then they may or may not adhere to the same basis
            2a) examples that do adhere to the same basis
                - e.g., A = LiMnO2, B = MnO2; basis = Li_{z}MnO2
                - e.g., A = LiMnO2, B = Li2MnO2; basis = Li_{z}MnO2
                - e.g., A = Ga2O3, B = Al2O3; basis = (Ga_{1-x}Al_{x})2O3
            2b) examples not adhering to the same basis
                - e.g., A = LiMnO2, B = Mn3O4 does not adhere to the same basis
                    - B must be re-cast as Mn_{1.5}O2, then basis = Li_{z}MnO2
                    - divide_right_by = 2
                - e.g., A = Li2MnO3, B = MnO2 does not adhere to the same basis
                    - A must be re-cast as Li_{4/3}Mn_{2/3}O2, then basis = Li_{z}MnO2
                    - divide_left_by = 3/2
                - e.g., A = Li3Fe2P4S12, B = FeP2S6 does not adhere to the same basis
                    - A must be re-cast as Li_{1.5}FeP2S6, then basis = Li_{z}FeP2S6
                    - divide_left_by = 2
                - e.g., A = Li9Mn20O40, B = LiMnO2 does not adhere to the same basis
                    - A must be re-cast as Li_{0.45}MnO2, then basis = Li_{z}MnO2
                    - divide_left_by = 20
            2c) the end members have the same basis, but you want to change both of them
                - e.g., A = Li8Fe3Mn1P8S24, B = Fe3Mn1P8S24
                    - you want a ...P2S6 basis
                    - so A must be-recast as Li_{2}Fe_{0.75}Mn_{0.25}P2S6 and B = Fe_{0.75}Mn_{0.25}P2S6
                    - divide_left_by = 4, divide_right_by = 4
        Mixing hulls are computed for mixing energies with a basis formula having A_{1-x}B_{x} atoms where A and B are compounds having the same basis
            align the basis using divide_left_by and divide_right_by
        We generally care about mixing energies per formula unit basis ('E_mix_per_fu')
    """

    def __init__(
        self,
        input_energies: dict,
        left_end_member: str,
        right_end_member: str,
        *,
        divide_left_by: int | float = 1,
        divide_right_by: int | float = 1,
        energy_key: str = "E",
    ) -> None:
        """
        Args:
            input_energies (dict):
                {formula (str) :
                    {energy_key (str) :
                        total (or formation) energy in eV/atom (float)}}}
                these should be ground-state energies for all relevant formulas
                it's OK if entries in this dictionary are not relevant to the mixing hull you are trying to calculate
                    this code will just ignore them
            left_end_member (str):
            right_end_member (str):
                this specifies the two "end members" of your mixing hull
                e.g., left = 'FeP2S6' and right='Li2FeP2S6'
                    if you were looking at the lithiation of FeP2S6 (e.g., to compute the insertion voltage)
                e.g., left = 'BaZrS3', right = 'BaNbS3'
                        if you were looking at the formation of Ba(Zr_{1-x}Nb_x)S_3 phases
            energy_key (str): Defaults to "E".
                the key inside of each formula in input_energies where the energy per atom lives
                e.g., input_energies['Li2FeP2S6']['E'] should return the DFT total energy per atom if energy_key = 'E'
            divide_left_by (int or float): Defaults to 1.
            divide_right_by (int or float): Defaults to 1.
                end members should have the same "basis" formula
                    if they do, then these can be left as 1
                    if they don't, then this is used to correct the right or left
                let A = left and B = right end member
                    e.g., A = LiMnO2, B = Mn3O4
                        divide_right_by = 2 to make both formulas have Li_{z}MnO2 basis
                    e.g., A = Li9Mn20O40, B = LiMnO2
                        divide_left_by = 20, then basis = Li_{z}MnO2
        """
        # sanitize the input_energies
        input_energies = {
            CompTools(k).clean: {"E": input_energies[k][energy_key]}
            for k in input_energies
        }

        self.left_end_member = CompTools(left_end_member).clean
        self.right_end_member = CompTools(right_end_member).clean

        print(
            "Working on mixing hulls for (1-x) %s + (x) %s "
            % (self.left_end_member, self.right_end_member)
        )
        self.end_members = [self.left_end_member, self.right_end_member]

        self.input_energies = input_energies

        left_cleaning_adj = CompTools(left_end_member).find_scaling(
            self.left_end_member
        )
        right_cleaning_adj = CompTools(right_end_member).find_scaling(
            self.right_end_member
        )
        self.divide_left_by = divide_left_by * left_cleaning_adj
        self.divide_right_by = divide_right_by * right_cleaning_adj

    @property
    def relevant_compounds(self) -> list[str]:
        """
        Returns:
            list of compounds (str) that appear in this mixing hull
        """
        input_energies = self.input_energies

        # maky pymatgen PhaseDiagram entries
        pd_entries = [
            PDEntry(Composition(c), input_energies[c]["E"]) for c in input_energies
        ]

        end_members = self.end_members

        # make pymatgen PhaseDiagram bounded by the end members
        cpd = CompoundPhaseDiagram(
            pd_entries, [Composition(c) for c in end_members], False
        )

        # get the compounds that are still in this phase diagram
        entries = cpd.entries
        return [CompTools(e.name).clean for e in entries]

    @property
    def reactions(self) -> dict[str, ReactionEnergy]:
        """
        dictionary of
            {formula (str) : ReactionEnergy object}
                the formula is some composition in the mixing hull
                the ReactionEnergy object is for the mixing reaction
                    e.g., 0.5 Ga2O3 + 0.5 Al2O3 -> AlGaO3
        """
        # every relevant compound is a reaction "target" (product)
        targets = self.relevant_compounds

        # our end-members are the reactants for all mixing energy calculations
        reactants = self.end_members
        input_energies = self.input_energies
        reactions = {
            t: ReactionEnergy(
                input_energies=input_energies,
                reactants=reactants,
                products=[t],
                energy_key="E",
                norm="rxn",
            )
            for t in targets
        }
        return reactions

    @property
    def mixing_energies(self) -> dict[str, dict[str, float]]:
        """
        Returns:
            {formula (str) :
                'x' : fraction of second end member (1-x) = fraction of first end member,
                'E' : the energy you inputted for this formula (eV/atom),
                'E_mix' : the mixing energy (eV/atom),
                'mixing_rxn' : the mixing reaction string (str)}
        """
        energies = {}
        reactions = self.reactions
        input_energies = self.input_energies
        left, right = self.end_members
        divide_left_by = self.divide_left_by
        divide_right_by = self.divide_right_by
        for target in reactions:
            E = input_energies[target]["E"]

            if target in [left, right]:
                # E_mix = 0 for end members
                E_mix_per_atom = 0
                E_mix_per_fu = 0

                # define x and number of atoms in basis
                if target == left:
                    x = 0
                    atoms_in_basis = CompTools(left).n_atoms / divide_left_by
                else:
                    x = 1
                    atoms_in_basis = CompTools(right).n_atoms / divide_right_by

                # x = 0 for first end member (left)
                # x = 0 if target == left else 1

                # no mixing reaction (trivial)
                rxn = None
                dE_rxn = 0
            else:
                coefs = reactions[target].coefs
                rxn = reactions[target].rxn_string

                # compute x based on reaction
                n_left = abs(coefs[left]) if left in coefs else 0
                n_right = abs(coefs[right]) if right in coefs else 0
                x = (n_right * divide_right_by) / (
                    (n_left * divide_left_by) + (n_right * divide_right_by)
                )

                # compute mixing energy
                # reaction energy as written to make 1 mole of target
                dE_rxn = reactions[target].dE_rxn

                # get E mix per atom of target
                E_mix_per_atom = dE_rxn / CompTools(target).n_atoms / coefs[target]

                # normalize the target to the correct formula unit
                ### target should be LEFT_{1-x}RIGHT_{x}
                ### where LEFT and RIGHT have same basis
                n_atoms_that_should_be_in_target = (
                    (1 - x) * CompTools(left).n_atoms / divide_left_by
                ) + ((x) * CompTools(right).n_atoms / divide_right_by)
                E_mix_per_fu = E_mix_per_atom * n_atoms_that_should_be_in_target

                # number of atoms in target (which is defined by the basis)
                atoms_in_basis = n_atoms_that_should_be_in_target

            energies[target] = {
                "x": x,
                "E_mix_per_at": E_mix_per_atom,
                "E": E,
                "mixing_rxn": rxn,
                "E_mix_per_fu": E_mix_per_fu,
                "dE_rxn": dE_rxn,
                "num_atoms_in_basis": atoms_in_basis,
            }

        return energies

    @property
    def sorted_compounds(self) -> list[str]:
        """
        Returns:
            alphabetized list of relevant compounds (str) in the mixing hull
        """
        return sorted(list(self.mixing_energies.keys()))

    def amts_matrix(
        self,
        compounds: Literal["all"] | list = "all",
        chemical_space: Literal["all"] | tuple = "all",
    ) -> np.ndarray:
        """
        Args:
            compounds (str or list)
                if 'all', use all compounds; else use specified list
                    note: this gets modified for you as needed to minimize cpu time
            chemical_space (str)
                if 'all', use entire space; else use specified tuple
                    note: this gets modified for you as needed to minimize cpu time
        Returns:
            matrix (2D array)
                with the fractional composition of each element in each compound (float)
                    each row is a different compound (ordered going down alphabetically)
                    each column is a different element (ordered across alphabetically)
        """
        if chemical_space == "all":
            # we're just doing binary mixing hulls so we are one-hot encoding the "chemical space" as (A, B)
            chemical_space = ["A", "B"]

        # use our computed mixing energies
        mixing_energies = self.mixing_energies

        # use all the relevant compounds
        if compounds == "all":
            compounds = self.sorted_compounds

        # make our coefficients matrix using "x" from the mixing energies dict
        A = np.zeros((len(compounds), len(chemical_space)))
        for row in range(len(compounds)):
            compound = compounds[row]
            for col in range(len(chemical_space)):
                A[row, col] = mixing_energies[compound]["x"]
        return A

    def formation_energy_array(
        self, compounds: Literal["all"] | list = "all"
    ) -> np.ndarray:
        """
        Args:
            compounds (str or list)
                if 'all', use all compounds; else use specified list
                    this gets modified for you as needed to minimize cpu time
        Returns:
            1D array of formation energies (float) for each compound ordered alphabetically
        """
        mixing_energies = self.mixing_energies
        if compounds == "all":
            compounds = self.sorted_compounds

        # our "formation energies" for the hull calculation are the mixing energies we calculated
        return np.array([mixing_energies[c]["E_mix_per_fu"] for c in compounds])

    def hull_input_matrix(
        self,
        compounds: Literal["all"] | list = "all",
        chemical_space: Literal["all"] | tuple = "all",
    ) -> np.ndarray:
        """
        Args:
            compounds (str or list)
                if 'all', use all compounds; else use specified list
                    this gets modified for you as needed to minimize cpu time
            chemical_space
                if 'all', use entire space; else use specified tuple
                    this gets modified for you as needed to minimize cpu time
        Returns:
            amts_matrix, but replacing the last column with the formation energy
                this is because convex hulls are defined by (n-1) composition axes
                    e.g., in a A-B phase diagram, specifying the fractional composition of A sets the composition of B
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
    def hull(self) -> ConvexHull:
        """
        Returns:
            scipy.spatial.ConvexHull object for all compounds in a chemical space
        """
        return ConvexHull(self.hull_input_matrix())

    @property
    def hull_points(self) -> np.ndarray:
        """
        Returns:
            array of (composition, formation energy) points (2-element tuple) fed to ConvexHull
        """
        return self.hull.points

    @property
    def hull_vertices(self) -> np.ndarray:
        """
        Returns:
            array of indices (int) in hull_points corresponding with the points that are on the hull
        """
        return self.hull.vertices

    @property
    def hull_simplices(self) -> np.ndarray:
        """
        Returns:
            indices of points forming the simplical facets of the convex hull.
        """
        return self.hull.simplices

    @property
    def stable_compounds(self) -> list[str]:
        """
        Returns:
            list of compounds (str) that correspond with vertices on the mixing hull
                these are stable compounds (within the mixing hull analysis)
                remember: the nature of a mixing hull is:
                    1) assuming that the end members are stable
                    2) assuming that there are no other stable compounds in the chemical space that are orthogonal to the mixing axis
                        - e.g., we might be mixing RuO2 and IrO2, but RuIrO5 would not appear on this hull because it is orthogonal
        """
        mixing_energies = self.mixing_energies
        hull_vertices = self.hull_vertices
        compounds = self.sorted_compounds
        end_members = self.end_members

        # compounds are stable if they have negative mixing energy and are on the hull
        vertices = [
            compounds[i]
            for i in hull_vertices
            if mixing_energies[compounds[i]]["E_mix_per_fu"] <= 0
        ]
        vertices += end_members
        return sorted(list(set(vertices)))

    @property
    def unstable_compounds(self) -> list[str]:
        """
        Returns:
            list of compounds that do not correspond with vertices (str)
                these are "above" the mixing hull
        """
        compounds = self.sorted_compounds
        stable_compounds = self.stable_compounds
        return [c for c in compounds if c not in stable_compounds]

    @property
    def results(self) -> dict[str, dict]:
        """
        this is the only method that really needs to be called by the user
        Returns:
            {formula (str) :
                {'E' : total (or formation) energy per atom (float, eV/atom),
                 'E_mix_per_fu' : mixing energy (float, eV/fu),
                 'E_mix_per_at' : mixing energy (float, eV/atom),
                 'dE_rxn' : reaction energy as determined by ReactionEnergy (float, eV/rxn),
                 'x' : fraction of the right end member (float),
                 'stability' : True if the compound is on the mixing hull else False,
                 'mixing_rxn' : mixing reaction (molar basis),
                 'num_atoms_in_basis' : number of atoms in the basis formula (float)}}
        """
        stable_compounds = self.stable_compounds
        unstable_compounds = self.unstable_compounds
        mixing_energies = self.mixing_energies
        for c in stable_compounds:
            mixing_energies[c]["stability"] = True
        for c in unstable_compounds:
            mixing_energies[c]["stability"] = False

        return mixing_energies


def main():
    gs = read_json("../demos/output/hulls/data/query_Li-Mn-Fe-O.json")
    # return gs
    mix = MixingHull(
        input_energies=gs,
        left_end_member="MnO",
        right_end_member="Li6MnO4",
        energy_key="Ef_mp",
        divide_left_by=1,
        divide_right_by=1,
    )

    return mix, mix.results


if __name__ == "__main__":
    mix, out = main()

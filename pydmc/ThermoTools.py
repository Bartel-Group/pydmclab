from pydmc.CompTools import CompTools
from pydmc.handy import read_json, write_json
from pydmc.data.reference_energies import mus_at_0K, mus_at_T
from pydmc.data.features import atomic_masses
from pydmc.StrucTools import StrucTools

import os
import numpy as np
from scipy.spatial import ConvexHull
from scipy.optimize import minimize
import multiprocessing as multip
import math
from itertools import combinations

HERE = os.path.dirname(os.path.abspath(__file__))

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

"""
class GetHullInputData(object):
    """
    Generates hull-relevant data
    Designed to be executed once all compounds and ground-state formation energies are known
    """
    
    def __init__(self, compound_to_energy, formation_energy_key):
        """
        Args:
            compound_to_energy (dict) - {formula (str) : 
                                            {formation_energy_key (str) : 
                                                formation energy (float, eV/atom)}}
                - it's ok to have other keys in this dictionary, just have to have a formation_energy_key that maps to formation energies
                - these formation energies can be 0 K, at some temperature, or computed in any way that you want, they just have to be compatible w/ one another
            formation_energy_key (str) - key within compound_to_energy to use for formation energy

        Returns:
            dictionary of {formula (str) : formation energy (float, eV/atom)}
                - note that elements are removed from this dictionary, they will be added back in later
                    - this is a legacy thing that is kind of tedious but (probably) needs to be done
        """
        self.compound_to_energy = {CompTools(k).clean : compound_to_energy[k][formation_energy_key] 
                                    for k in compound_to_energy
                                    if len(CompTools(k).els) > 1}
        
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
                - a chemical space would be (Ca, O, Ti) for CaTiO3
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
        subspaces = [all_spaces[i] for i in range(len(all_spaces)) 
                                   for j in range(len(all_spaces)) 
                                   if set(all_spaces[i]).issubset(all_spaces[j]) 
                                   if all_spaces[i] != all_spaces[j]]
        return list(set(subspaces))
    
    def hull_spaces(self, fjson=False, remake=False, write=False):
        """
        this can take a while to compute, so sometimes it's useful to write it to a json file
        
        Args:
            fjson (str) - file name to write hull spaces to
            remake (bool) - if True, repeat analysis; if False, read json
            write (bool) - if True, write json; if False, don't write json
            
        Returns:
            list of unique chemical spaces (set) that do define convex hull spaces
                these are the convex hulls that must be computed
                so if you are working with binaries and ternaries, but no quaternaries, hull spaces will be maximally 3 dimensions (elements)
        """ 
        if not fjson:
            fjson = 'hull_spaces.json'
        if not remake and os.path.exists(fjson):
            d = read_json(fjson)
            return d['hull_spaces']
        chemical_spaces_and_subspaces = self.chemical_spaces_and_subspaces
        chemical_subspaces = self.chemical_subspaces
        d = {'hull_spaces' : [s for s in chemical_spaces_and_subspaces if s not in chemical_subspaces if len(s) > 1]}
        if write:
            d = write_json(d, fjson)
        return d['hull_spaces']
    
    def hullin_data(self, fjson=False, remake=False):
        """
        
        Args:
            fjson (str) - file name to write hull data to
            remake (bool) - if True, run analysis + write json; if False, read json
            
        Returns:
            dict of {chemical space (str) : {formula (str) : {'E' : formation energy (float),
                                                              'amts' : {el (str) : fractional amt of el in formula (float) for el in space}} 
                                            for all relevant formulas including elements}
                - elements are automatically given formation energy = 0
                - chemical space is now in 'el1_el2_...' format to be jsonable
                - each "chemical space" is a convex hull that must be computed
        """
        if not fjson:
            fjson = 'hull_input_data.json'
        if (remake == True) or not os.path.exists(fjson):
            hullin_data = {}
            hull_spaces = self.hull_spaces()
            compounds = self.compounds
            compound_to_energy = self.compound_to_energy
            for space in hull_spaces:
                for el in space:
                    compound_to_energy[el] = 0
                relevant_compounds = [c for c in compounds if set(CompTools(c).els).issubset(set(space))] + list(space)
                hullin_data['_'.join(list(space))] = {c : {'E' : compound_to_energy[c],
                                                         'amts' : {el : CompTools(c).mol_frac(el=el) for el in space}}
                                                        for c in relevant_compounds}
            return write_json(hullin_data, fjson)
        else:
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
            hullin_data (dict) - dictionary generated in GetHullInputData().hullin_data
            chemical_space (str) - chemical space to analyze in 'el1_el2_...' (alphabetized) format
                - to compute Ca-Ti-O phase diagram, chemical_space would be 'Ca_O_Ti'
        Returns:

            hullin_data (dict) - dictionary of {compound (str) :
                                                    {'E' : formation energy (float), 
                                                     'amts' : 
                                                        {el (str) : 
                                                            fractional amt of el in compound (float) 
                                                                for el in chemical_space}}}})}
        
                - grabs only the relevant sub-dict from hullin_data
                
            chemical_space (tuple) - chemical space to analyze
                - changes chemical space to tuple (el1, el2, ...)
        """
        hullin_data = hullin_data[chemical_space]

        keys_to_remove = [k for k in hullin_data 
                                  if CompTools(k).n_els == 1]
        hullin_data = {k : hullin_data[k] for k in hullin_data if k not in keys_to_remove}

        els = chemical_space.split('_')
        for el in els:
            hullin_data[el] = {'E' : 0,
                             'amts' : {els[i] : 
                                        CompTools(el).mol_frac(els[i])
                                        for i in range(len(els))}}
        self.hullin_data = hullin_data
        
        self.chemical_space = tuple(els)
        
    @property 
    def sorted_compounds(self):
        """
        Returns:
            alphabetized list of compounds (str) in specified chemical space
        """
        return sorted(list(self.hullin_data.keys()))
    
    def amts_matrix(self, compounds='all', chemical_space='all'):
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
        if chemical_space == 'all':
            chemical_space = self.chemical_space
        hullin_data = self.hullin_data
        if compounds == 'all':
            compounds = self.sorted_compounds
        A = np.zeros((len(compounds), len(chemical_space)))
        for row in range(len(compounds)):
            compound = compounds[row]
            for col in range(len(chemical_space)):
                el = chemical_space[col]
                A[row, col] = hullin_data[compound]['amts'][el]
        return A
    
    def formation_energy_array(self, compounds='all'):
        """
        Args:
            compounds (str or list) - if 'all', use all compounds; else use specified list
                - this gets modified for you as needed to minimize cpu time
        
        Returns:
            1D array of formation energies (float) for each compound ordered alphabetically
        """        
        hullin_data = self.hullin_data
        if compounds == 'all':
            compounds = self.sorted_compounds
        return np.array([hullin_data[c]['E'] for c in compounds])
    
    def hull_input_matrix(self, compounds='all', chemical_space='all'):
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
            for col in range(np.shape(X)[1]-1):
                X[row, col] = A[row, col]
            X[row, np.shape(X)[1]-1] = b[row]
        return X
    
    @property
    def hull(self):
        """
        Returns:
            scipy.spatial.ConvexHull object for all compounds in a chemical space
        """
        return ConvexHull(self.hull_input_matrix(compounds='all', chemical_space='all'))
    
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
        hullin_data = self.hullin_data
        hull_vertices = self.hull_vertices
        compounds = self.sorted_compounds
        return [compounds[i] for i in hull_vertices if hullin_data[compounds[i]]['E'] <= 0]
    
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
    
    def competing_compounds(self, compound):
        """
        Args:
            compound (str) - the compound (str) to analyze
        
        Returns:
            list of compounds (str) that may participate in the decomposition reaction for the input compound
                - these must have the same elements as the input compound or a subset of those elements
        """
        compounds = self.sorted_compounds
        if compound in self.unstable_compounds:
            compounds = self.stable_compounds
        competing_compounds = [c for c in compounds if c != compound if set(CompTools(c).els).issubset(CompTools(compound).els)]
        return competing_compounds
    
    def A_for_decomp_solver(self, compound, competing_compounds):
        """
        Args:
            compound (str) - the compound (str) to analyze
            competing_compounds (list) - list of compounds (str) that may participate in the decomposition reaction for the input compound
        
        Returns:
            matrix (2D array) of elemental amounts (float) used for implementing molar conservation during decomposition solution
                - i.e., a decomposition reaction has to conserve the total number of each element on both sides of the reaction
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
            compound (str) - the compound (str) to analyze
       
        Returns:
            array of elemental amounts (float) used for implementing molar conservation during decomposition solution
        """        
        chemical_space = tuple(CompTools(compound).els)        
        return np.array([CompTools(compound).stoich(el) for el in chemical_space])
    
    def Es_for_decomp_solver(self, competing_compounds):
        """
        Args:
            competing_compounds (list) - list of compounds (str) that may participate in the decomposition reaction for the input compound
        
        Returns:
            array of formation energies per formula unit (float) used for minimization problem during decomposition solution
                - these have to be Ef per f.u. because we are computing reaction energies
        """     
        atoms_per_fu = [CompTools(c).n_atoms for c in competing_compounds]        
        Es_per_atom = self.formation_energy_array(competing_compounds)    
        return [Es_per_atom[i]*atoms_per_fu[i] for i in range(len(competing_compounds))] 
        
    def decomp_solution(self, compound):
        """
        Args:
            compound (str) - the compound (str) to analyze
        
        Returns:
            scipy.optimize.minimize result 
                for finding the linear combination of competing compounds that minimizes the competing formation energy
                    - i.e., the fractional composition of decomposition products leading to the most positive deecomposition energy
        """        
        competing_compounds = self.competing_compounds(compound)
        A = self.A_for_decomp_solver(compound, competing_compounds)
        b = self.b_for_decomp_solver(compound)
        Es = self.Es_for_decomp_solver(competing_compounds)
        n0 = [0.1 for c in competing_compounds]
        max_bound = CompTools(compound).n_atoms
        bounds = [(0,max_bound) for c in competing_compounds]
        def competing_formation_energy(nj):
            nj = np.array(nj)
            return np.dot(nj, Es)
        constraints = [{'type' : 'eq',
                        'fun' : lambda x: np.dot(A, x)-b}]
        maxiter, disp = 1000, False
        # slowly relax tolerance in case minimize can't converge
        for tol in [1e-6, 1e-4, 1e-3, 1e-2]:
            solution =  minimize(competing_formation_energy,
                                 n0,
                                 method='SLSQP',
                                 bounds=bounds,
                                 constraints=constraints,
                                 tol=tol,
                                 options={'maxiter' : maxiter,
                                          'disp' : disp})
            if solution.success:
                return solution
        return solution
        
        
    def decomp_products(self, compound):
        """
        Args:
            compound (str) - the compound (str) to analyze
        
        Returns:
            dictionary of {competing compound (str) : {'amt' : stoich weight in decomp rxn (float),
                                                       'E' : formation energy (float)}
                                                            for all compounds in the stability-defining reaction}
                np.nan if decomposition analysis fails
        """            
        hullin_data = self.hullin_data
        competing_compounds = self.competing_compounds(compound)
        
        if (len(competing_compounds) == 0) or (np.max([CompTools(c).n_els for c in competing_compounds]) == 1):
            return {el : {'amt' : CompTools(compound).stoich(el),
                          'E' : 0} for el in CompTools(compound).els}
        solution = self.decomp_solution(compound)
        if solution.success:
            resulting_amts = solution.x
        elif hullin_data[compound]['E'] > 0:
            return {el : {'amt' : CompTools(compound).stoich(el),
                          'E' : 0} for el in CompTools(compound).els}
        else:
            # in case scipy.minimize fails to converge
            print(compound)
            print('\n\n\nFAILURE!!!!\n\n\n')
            print(compound)
            return np.nan
        
        # put irrelevant compounds to zero to make reaction more clear
        min_amt_to_show = 1e-4
        decomp_products = dict(zip(competing_compounds, resulting_amts))
        relevant_decomp_products = [k for k in decomp_products if decomp_products[k] > min_amt_to_show]
        decomp_products = {k : {'amt' : decomp_products[k],
                                'E' : hullin_data[k]['E']} if k in hullin_data else 0 for k in relevant_decomp_products}
        
        # here's some stuff that's probably unnecessary if everything is working properly
        """
        el_totals_to_match = {el : CompTools(compound).stoich(el) for el in CompTools(compound).els}
        free_decomp_products = [cmpd for cmpd in decomp_products if CompTools(cmpd).n_els == 1]
        
        decomp_totals = {el : 0 for el in CompTools(compound).els}
        for decomp_product in decomp_totals:
            for el in CompTools(decomp_product).els:
                decomp_totals[el] += CompTools(decomp_product).stoich(el)
        for el in free_decomp_products:
            decomp_products[el]['amt'] += el_totals_to_match[el] - decomp_totals[el]
        """
        return decomp_products
    
    def decomp_energy(self, compound):
        """
        Args:
            compound (str) - the compound (str) to analyze
        
        Returns:
            decomposition energy (float)
                - if < 0, this is the magnitude of formation energy that compound could increase and still be on the hull
                - if > 0, this is the energy above the hull
        """
        hullin_data = self.hullin_data
        decomp_products = self.decomp_products(compound)
        if isinstance(decomp_products, float):
            return np.nan
        decomp_energy = 0
        for k in decomp_products:
            decomp_energy += decomp_products[k]['amt']*decomp_products[k]['E']*CompTools(k).n_atoms
        return (hullin_data[compound]['E']*CompTools(compound).n_atoms - decomp_energy) / CompTools(compound).n_atoms
    
    @property
    def hull_output_data(self):
        """
        Returns:
            stability data (dict) for all compounds in the specified chemical space
                {compound (str) : {'Ef' : formation energy (float),
                                   'Ed' : decomposition energy (float),
                                   'rxn' : decomposition reaction (str),
                                   'stability' : stable (True) or unstable (False)}}
        """
        data = {}
        hullin_data = self.hullin_data
        compounds, stable_compounds = self.sorted_compounds, self.stable_compounds
        for c in compounds:
            if c in stable_compounds:
                stability = True
            else:
                stability = False
            Ef = hullin_data[c]['E']
            Ed = self.decomp_energy(c)
            decomp_products = self.decomp_products(c)
         
            if isinstance(decomp_products, float):
                data[c] = np.nan
                continue
            decomp_rxn = ['_'.join([str(np.round(decomp_products[k]['amt'], 4)), k]) for k in decomp_products]
            decomp_rxn = ' + '.join(decomp_rxn)
            data[c] = {'Ef' : Ef,
                       'Ed' : Ed,
                       'rxn' : decomp_rxn,
                       'stability' : stability}
        return data
    
    def cmpd_hull_output_data(self, compound):
        """
        Args:
            compound (str) - formula to get data for
            
        Returns:
            hull_output_data but only for single compound
                {'Ef' : formation energy (float),
                'Ed' : decomposition energy (float),
                'rxn' : decomposition reaction (str),
                'stability' : stable (True) or unstable (False)}
                
            - so you could loop through compounds to get this or do things on a chemical space basis
        """
        data = {}
        hullin_data = self.hullin_data
        stable_compounds = self.stable_compounds
        c = compound
        if c in stable_compounds:
            stability = True
        else:
            stability = False
        Ef = hullin_data[c]['E']
        Ed = self.decomp_energy(c)
        decomp_products = self.decomp_products(c)
        if isinstance(decomp_products, float):
            return {c : np.nan}
        decomp_rxn = ['_'.join([str(np.round(decomp_products[k]['amt'], 4)), k]) for k in decomp_products]
        decomp_rxn = ' + '.join(decomp_rxn)
        data[c] = {'Ef' : Ef,
                   'Ed' : Ed,
                   'rxn' : decomp_rxn,
                   'stability' : stability}
        return data[c]
    
    
class ParallelHulls(object):
    """
    this allows you to parallelize the hullin_data and hullout_data generation over cores using multiprocessing
    
    """
    
    def __init__(self,
                 compound_to_energy,
                 formation_energy_key='Ef_mp',
                 n_procs=4,
                 data_dir=False,
                 fresh_restart=False):
        """
             compound_to_energy (dict) - {formula (str) : 
                                            {formation_energy_key (str) : 
                                                formation energy (float, eV/atom)}}
                - it's ok to have other keys in this dictionary, just have to have a formation_energy_key that maps to formation energies
                - these formation energies can be 0 K, at some temperature, or computed in any way that you want, they just have to be compatible w/ one another
            formation_energy_key (str) - key within compound_to_energy to use for formation energy       
            n_procs (int) - number of cores to use
                - if 'all', will use every core availble minus 1
            data_dir (str) - directory to save data to (if False, writes to os.getcwd())
            fresh_restart (bool) - if True, will re-compute everything from scratch
        """
        self.n_procs = n_procs if n_procs != 'all' else multip.cpu_count()-1
        if not data_dir:
            self.data_dir = os.getcwd()
        else:
            self.data_dir = data_dir
        self.fresh_restart = fresh_restart
        
        self.compound_to_energy = {k : compound_to_energy[k][formation_energy_key] 
                                    for k in compound_to_energy
                                    if len(CompTools(k).els) > 1}
    
    @property
    def compounds(self):
        """            
        Returns:
            list of compounds (str)
        """
        compounds = list(self.compound_to_energy.keys())
        return sorted(list([c for c in compounds if CompTools(c).n_els > 1]))
    
    @property
    def chemical_spaces(self):
        """
        Returns:
            list of unique chemical spaces (tuple)
                - a chemical space would be (Ca, O, Ti) for CaTiO3
        """
        compounds = self.compounds
        chemical_spaces = []
        for c in compounds:
            space = '_'.join(sorted(list(set(CompTools(c).els))))
            chemical_spaces.append(space)
        return sorted(list(set(chemical_spaces)))
    
    def hullin_from_space(self, space, verbose=True):
        """
        Function called at each parallelized generation of hull input file (same as GetHullInputData.hullin_data)
        
        Args:
            space (str) - '_'.join(elements) (str) in chemical space
            verbose (bool) - print space or not
        Returns:
            {compound (str) : {'E' : formation energy (float, eV/atom),
                        'amts' : {el (str) : fractional amount of el in formula (float)}}}
        """
        compound_to_energy = self.compound_to_energy
        if verbose:
            print(space)
        space = space.split('_')
        for el in space:
            compound_to_energy[el] = 0
        relevant_compounds = [c for c in compound_to_energy if set(CompTools(c).els).issubset(set(space))] + list(space)
        return {c : {'E' : compound_to_energy[c],
                    'amts' : {el : CompTools(c).mol_frac(el=el) for el in space}}
                                                for c in relevant_compounds}

    def parallel_hullin(self, 
                        fjson=False, verbose=True):
        """
        Parallel generation of hull input data (same as GetHullInputData.hullin_data)
           
        Args:
            fjson (os.PathLike) - path to write dictionary of hull input data
            verbose (bool) - print space or not
        Returns:
            {chemical space (str) : 
                {compound (str) : 
                    {'E' : formation energy (float, eV/atom),
                    'amts' : {el (str) : fractional amount of el in formula (float)}}}}
        """
        remake = self.fresh_restart
        chemical_spaces = self.chemical_spaces
        if not fjson:
            fjson = os.path.join(self.data_dir, 'hull_input_data.json')
        if (remake == True) or not os.path.exists(fjson):
            hullin_data = {}
            pool = multip.Pool(processes=self.n_procs)
            results = [r for r in pool.starmap(self.hullin_from_space, [(space, verbose) for space in chemical_spaces])]
            pool.close()
            hullin_data = dict(zip(chemical_spaces, results))
            return write_json(hullin_data, fjson)
        else:
            return read_json(fjson)
        
    def smallest_space(self, hullin, formula, 
                       verbose=False):
        """
        function to parallelize finding the smallest hull to compute for each compound
        
        Args:
            hullin (dict) - {space (str, '_'.join(elements)) : 
                                {formula (str) : 
                                    {'E' : formation energy (float, eV/atom),
                                    'amts' : 
                                        {element (str) : fractional amount of element in formula (float)}
                                    }
                                }
                            }
            formula (str) - chemical formula
            verbose (bool) - print formula or not
        
        Returns:
            chemical space (str, '_'.join(elements)) of the convex hull that is easiest to compute for a given compound
                - for instance, if I want the decomp energy of CaO, there's no need to do the Ca-Ti-O hull, only the Ca-O one
        """
        if verbose:
            print(formula)
        spaces = sorted(list(hullin.keys()))
        relevant = [s for s in spaces if formula in hullin[s]]
        sizes = [s.count('_') for s in relevant]
        small = [relevant[i] for i in range(len(sizes)) if sizes[i] == np.min(sizes)]
        sizes = [len(hullin[s]) for s in small]
        smallest = [small[i] for i in range(len(small)) if sizes[i] == np.min(sizes)]
        return smallest[0]
    
    def smallest_spaces(self,
                        hullin,
                        fjson=False,
                        verbose=False):
        """
        parallel generation of smallest spaces for each compound
        
        Args:
            hullin (dict) - same as smallest_space
            fjson (os.PathLike) - path to write dictionary of smallest spaces
            verbose (bool) - print formula or not
        
        Returns:
            {formula (str) :
                chemical space (str, '_'.join(elements)) of convex hull that is easiest to compute}
        """
        remake = self.fresh_restart
        compounds = self.compounds
        if not fjson:
            fjson = os.path.join(self.data_dir, 'smallest_spaces.json')
        if not remake and os.path.exists(fjson):
            return read_json(fjson)
        pool = multip.Pool(processes=self.n_procs)
        smallest = [r for r in pool.starmap(self.smallest_space, [(hullin, compound, verbose) for compound in compounds])]
        pool.close()
        data = dict(zip(compounds, smallest))
        return write_json(data, fjson)
    
    def compound_stability(self, 
                            hullin,
                           smallest_spaces,
                           formula,
                           verbose=False):
        """
        Args:
            hullin (dict) - hull input dictionary
            smallest_spaces (dict) - {formula (str) : smallest chemical space having formula (str)}
            formula (str) - chemical formula
            verbose (bool) - print formula or not 
        
        Returns:
            {'Ef' : formation energy (float, eV/atom),
             'Ed' : decomposition energy (float, eV/atom),
             'rxn' : decomposition reaction (str),
             'stability' : bool (True if on hull)}
        """
        if verbose:
            print(formula)
        if CompTools(formula).n_els == 1:
            return {'Ef' : 0,
                    'Ed' : 0,
                    'stability' : True,
                    'rxn' : '1_%s' % formula}
        space = smallest_spaces[formula]
        return AnalyzeHull(hullin, space).cmpd_hull_output_data(formula)
    
    def parallel_hullout(self,
                         hullin,
                         smallest_spaces,
                        compounds='all', 
                        fjson=False, remake=False,
                        verbose=False):
        """
        Args:
            hullin (dict) - hull input dictionary
            smallest_spaces (dict) - {formula (str) : smallest chemical space having formula (str)}
            compounds (str or list) - if 'all', use every compound, else specify which compounds to compute
            fjson (os.PathLike) - path to write dictionary of hull output data
            reemake (bool) - remake the data or not
            verbose (bool) - print formula or not 
        
        Returns:
            {'Ef' : formation energy (float, eV/atom),
             'Ed' : decomposition energy (float, eV/atom),
             'rxn' : decomposition reaction (str),
             'stability' : bool (True if on hull)}
        """
        if not fjson:
            fjson = os.path.join(self.data_dir, 'hullout.json')
        if not remake and os.path.exists(fjson):
            return read_json(fjson)
        pool = multip.Pool(processes=self.n_procs)
        if compounds == 'all':
            compounds = sorted(list(smallest_spaces.keys()))
        results = [r for r in pool.starmap(self.compound_stability, [(hullin, smallest_spaces, compound, verbose) for compound in compounds])]
        pool.close()
        data = dict(zip(compounds, results))
        return write_json(data, fjson)
    
class GrandPotentialPD(object):
    """
    TO DO:
    
    compute stability in open (grand canonical) systems
    
    """
    def __init__(self):
        
        pass
    
class ReactionEnergy(object):
    """
    TO DO:
    
    compute reaction energies in closed and open systems w/ various normalization
    """
    
    def __init__(self):
        
        pass
    
class GibbsEnergy(object):
    """
    TO DO:
    
    implement physical descriptor for G(T)
    """
    
    def __init__(self):
        
        pass
    
class ChemPots(object):
    """
    return dictionary of chemical potentials {el : chemical potential (eV/at)} based on user inputs
    """
    
    def __init__(self, 
                 temperature=0, 
                 functional='pbe',
                 standard='dmc',
                 partial_pressures={}, # atm
                 diatomics=['H', 'N', 'O', 'F', 'Cl'],
                 R=8.6173303e-5, # eV/K
                 user_chempots={},
                 user_dmus={}):        
        """
        Args:
            temperature (int) - temperature in Kelvin
            functional (str) - functional for DFT calculations
            standard (str) - standard for DFT calculations
            partial_pressures (dict) - {el (str) : partial pressure (atm)}
                - adjusts chemical potential of gaseous species based on RTln(p/p0)
            diatomics (list) - list of diatomic elements
                - if el is in diatomics, will use 0.5 * partial pressure effect on mu
            user_chempots (dict) - {el (str) : chemical potential (eV/at)}
                - specifies the chemical potential you want to use for el
                - will override everything
            user_dmus (dict) - {el (str) : delta_mu (eV/at)}
                - specifies the change in chemical potential you want to use for el
                - will override everything except user_chempots
        """
        self.temperature = temperature
        self.functional = functional
        self.standard = standard
        self.partial_pressures = partial_pressures
        self.diatomics = diatomics
        self.R = R
        self.user_dmus = user_dmus
        self.user_chempots = user_chempots
        
    @property
    def chempots(self):
        """
        Returns:
            dictionary of chemical potentials {el : chemical potential (eV/at)} based on user inputs
        """

        if self.temperature == 0:
            all_mus = mus_at_0K()
            els = sorted(list(all_mus[self.standard][self.functional].keys()))
            mus = {el : all_mus[self.standard][self.functional][el]['mu'] for el in els}
        else:
            allowed_Ts = list(range(300, 2100, 100))
            if self.temperature not in allowed_Ts:
                raise ValueError('Temperature must be one of %s' % allowed_Ts)
            all_mus = mus_at_T()
            mus = all_mus[str(self.temperature)]
            
        if self.partial_pressures:
            for el in self.partial_pressures:
                if el in self.diatomics:
                    factor = 1/2
                else:
                    factor = 1
                mus[el] += -self.R * self.temperature * factor * np.log(self.partial_pressures[el])
        if self.user_dmus:
            for el in self.user_dmus:
                mus[el] += self.user_dmus[el]
        if self.user_chempots:
            for el in self.user_chempots:
                mus[el] = self.user_chempots[el]
                
        return mus
        
class FormationEnergy(object):
    """
    TO DO:
        - write tests/demo
    """
    
    def __init__(self,
                 formula,
                 E_DFT, # eV/at
                 chempots, # from ThermoTools.ChemPots.chempots
                 structure=False,
                 atomic_volume=False,
                 override_Ef_0K=False): 
        
        """
        Args:
            formula (str) - chemical formula
            E_DFT (float) - DFT energy (eV/at)
            chempots (dict) - {el (str) : chemical potential (eV/at)}
                - probably generated using ChemPots.chempots
                
            Only required for getting temperature-dependent formation energies:
                structure (Structure) - pymatgen structure object
                atomic_volume (float) - atomic volume (A^3/atom)
                override_Ef_0K (float) - formation energy at 0 K (eV/at)
                    - if False, compute Ef_0K using FormationEnergy.Ef_0K
        """
        self.formula = CompTools(formula).clean
        self.E_DFT = E_DFT 
        self.chempots = chempots
        self.structure = structure
        self.atomic_volume = atomic_volume
        self.override_Ef_0K = override_Ef_0K
        
    @property
    def weighted_elemental_energies(self):
        """
        Returns:
            weighted elemental energies (eV per formula unit)
        """
        mus = self.chempots
        els_to_amts = CompTools(self.formula).amts
        return np.sum([mus[el]*els_to_amts[el] for el in els_to_amts])

    @property
    def Ef_0K(self):
        """
        Returns:
            formation energy at 0 K (eV/at)
        """
        if self.override_Ef_0K:
            return self.override_Ef_0K
        formula = self.formula
        n_atoms = CompTools(formula).n_atoms
        weighted_elemental_energies = self.weighted_elemental_energies
        E_per_fu = self.E_DFT * n_atoms
        return (1/n_atoms) * (E_per_fu - weighted_elemental_energies)
    
    @property
    def reduced_mass(self):
        """
        Returns weighted reduced mass of composition
            - only needed for G(T) see Chris B Nature Comms 2019
        """
        names = CompTools(self.formula).els
        els_to_amts = CompTools(self.formula).amts
        nums = [els_to_amts[el] for el in names]
        mass_d = atomic_masses()
        num_els = len(names)
        num_atoms = np.sum(nums)        
        denom = (num_els - 1) * num_atoms
        if denom <= 0:
            print('descriptor should not be applied to unary compounds (elements)')
            return np.nan
        masses = [mass_d[el] for el in names]
        good_masses = [m for m in masses if not math.isnan(m)]
        if len(good_masses) != len(masses):
            for el in names:
                if math.isnan(mass_d[el]):
                    print('I dont have a mass for %s...' % el)
                    return np.nan
        else:
            pairs = list(combinations(names, 2))
            pair_red_lst = []
            for i in range(len(pairs)):
                first_elem = names.index(pairs[i][0])
                second_elem = names.index(pairs[i][1])
                pair_coeff = nums[first_elem] + nums[second_elem]
                pair_prod = masses[first_elem] * masses[second_elem]
                pair_sum = masses[first_elem] + masses[second_elem]
                pair_red = pair_coeff * pair_prod / pair_sum
                pair_red_lst.append(pair_red)
            return np.sum(pair_red_lst) / denom
        
    def dGf(self, temperature=0):
        """
        Args:
            temperature (int) - temperature (K)
        Returns:
            formation energy at temperature (eV/at)
                - see Chris B Nature Comms 2019
        """
        T = temperature
        Ef_0K = self.Ef_0K
        if T == 0:
            return Ef_0K
        else:
            Ef_0K = self.Ef_0K
            m = self.reduced_mass              
            if self.atomic_volume:
                V = self.atomic_volume
            elif self.structure:
                V = self.structure.volume / len(self.structure)
            else:
                raise ValueError('Need atomic volume or structure to compute G(T)')
            
            Gd_sisso = (-2.48e-4*np.log(V) - 8.94e-5*m/V)*T + 0.181*np.log(T) - 0.882
            weighted_elemental_energies = self.weighted_elemental_energies
            G = Ef_0K + Gd_sisso
            n_atoms = CompTools(self.formula).n_atoms
            
            return (1/n_atoms) * (G*n_atoms - weighted_elemental_energies)
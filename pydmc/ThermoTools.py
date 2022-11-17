from CompTools import CompTools
from pydmc.handy import read_json, write_json

import os
import numpy as np
from scipy.spatial import ConvexHull
from scipy.optimize import minimize
import multiprocessing as multip

class ParallelHulls(object):
    
    def __init__(self,
                 compound_to_energy,
                 formation_energy_key='Ef_mp',
                 n_procs=4,
                 data_dir=False,
                 fresh_restart=False):
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
        compounds = list(self.compound_to_energy.keys())
        return sorted(list([c for c in compounds if CompTools(c).n_els > 1]))
    
    @property
    def hull_spaces(self):
        compounds = self.compounds
        hull_spaces = []
        for c in compounds:
            space = '_'.join(sorted(list(set(CompTools(c).els))))
            hull_spaces.append(space)
        return sorted(list(set(hull_spaces)))
    
    def hullin_from_space(self, space, verbose=True):
        """
        Function to parallelize in generation of hull input file
        
        Args:
            compound_to_energy (dict) - {formula (str) : formation energy (float, eV/atom)}
            compounds (list) - list of formulas (str)
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
        Parallel generation of hull input data    
        Args:
            compound_to_energy (dict) - {formula (str) : formation energy (float, eV/atom)}
            hull_spaces (list) - list of chemical spaces (tuple of elements (str))
            fjson (os.PathLike) - path to write dictionary of hull input data
            remake (bool) - regenerate or not
            Nprocs (int) - number of processors for parallelization
            verbose (bool) - print space or not
        Returns:
            {chemical space (str) : 
                {compound (str) : 
                    {'E' : formation energy (float, eV/atom),
                    'amts' : {el (str) : fractional amount of el in formula (float)}}}}
        """
        remake = self.fresh_restart
        hull_spaces = self.hull_spaces
        if not fjson:
            fjson = os.path.join(self.data_dir, 'hull_input_data.json')
        if (remake == True) or not os.path.exists(fjson):
            hull_data = {}
            pool = multip.Pool(processes=self.n_procs)
            results = [r for r in pool.starmap(self.hullin_from_space, [(space, verbose) for space in hull_spaces])]
            pool.close()
            hull_data = dict(zip(hull_spaces, results))
            return write_json(hull_data, fjson)
        else:
            return read_json(fjson)
        
    def smallest_space(self, hullin, formula, 
                       verbose=False):
        """
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
        
        Returns:
            chemical space (str, '_'.join(elements), convex hull) that is easiest to compute
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
        Args:
            Nprocs (int) - processors to parallelize over
            remake (bool) - run this (True) or read this (False)
        
        Returns:
            {formula (str) :
                chemical space (str, '_'.join(elements), convex hull) 
                that is easiest to compute}
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
            smallest_spaces (dict) - {formula (str) : smallest chemical space having formula (str)}
            hullin (dict) - hull input dictionary
            formula (str) - chemical formula
        
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
            Nprocs (int) - processors to parallelize over
            remake (bool) - run this (True) or read this (False)
        
        Returns:
            {formula (str) :
                {'Ef' : formation energy (float, eV/atom),
                'Ed' : decomposition energy (float, eV/atom),
                'rxn' : decomposition reaction (str),
                'stability' : bool (True if on hull)}
                }
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
    
class GetHullInputData(object):
    """
    Generates hull-relevant data
    Designed to be executed once all compounds and ground-state formation energies are known
    """
    
    def __init__(self, compound_to_energy, formation_energy_key):
        """
        Args:
            compound_to_energy (dict) - {formula (str) : {formation_energy_key (str) : formation energy (float)}}
            formation_energy_key (str) - key within compound_to_energy to use for formation energy
        
        Returns:
            dictionary of {formula (str) : formation energy (float)}
        """
        self.compound_to_energy = {k : compound_to_energy[k][formation_energy_key] 
                                    for k in compound_to_energy
                                    if len(CompTools(k).els) > 1}
        
    @property
    def compounds(self):
        """
        Args:
            
        Returns:
            list of compounds (str)
        """
        return list(self.compound_to_energy.keys())
    
    @property
    def chemical_spaces_and_subspaces(self):
        """
        Args:
            
        Returns:
            list of unique chemical spaces (tuple)
        """
        compounds = self.compounds
        return list(set([tuple(CompTools(c).els) for c in compounds]))
    
    @property
    def chemical_subspaces(self):
        """
        Args:
            
        Returns:
            list of unique chemical spaces (tuple) that do not define convex hull spaces
                (Ca, O, Ti) is the space of CaTiO3 and Ca2TiO4
                if CaTiO3 and CaO are found, (Ca, O) is a subspace
        """        
        all_spaces = self.chemical_spaces_and_subspaces
        subspaces = [all_spaces[i] for i in range(len(all_spaces)) 
                                   for j in range(len(all_spaces)) 
                                   if set(all_spaces[i]).issubset(all_spaces[j]) 
                                   if all_spaces[i] != all_spaces[j]]
        return list(set(subspaces))
    
    def hull_spaces(self, fjson=False, remake=False, write=False):
        """
        Args:
            
        Returns:
            list of unique chemical spaces (set) that do define convex hull spaces
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
    
    def hull_data(self, fjson=False, remake=False):
        """
        Args:
            fjson (str) - file name to write hull data to
            remake (bool) - if True, write json; if False, read json
            
        Returns:
            dict of {chemical space (str) : {formula (str) : {'E' : formation energy (float),
                                                              'amts' : {el (str) : fractional amt of el in formula (float) for el in space}} 
                                            for all relevant formulas including elements}
                elements are automatically given formation energy = 0
                chemical space is now in 'el1_el2_...' format to be jsonable
        """
        if not fjson:
            fjson = 'hull_input_data.json'
        if (remake == True) or not os.path.exists(fjson):
            hull_data = {}
            hull_spaces = self.hull_spaces()
            compounds = self.compounds
            compound_to_energy = self.compound_to_energy
            for space in hull_spaces:
                for el in space:
                    compound_to_energy[el] = 0
                relevant_compounds = [c for c in compounds if set(CompTools(c).els).issubset(set(space))] + list(space)
                hull_data['_'.join(list(space))] = {c : {'E' : compound_to_energy[c],
                                                         'amts' : {el : CompTools(c).mol_frac(el=el) for el in space}}
                                                        for c in relevant_compounds}
            return write_json(hull_data, fjson)
        else:
            return read_json(fjson)
        
        
class AnalyzeHull(object):
    """
    Determines stability for one chemical space (hull)
    Designed to be parallelized over chemical spaces
    Ultimate output is a dictionary with hull results for one chemical space
    """
    
    def __init__(self, hull_data, chemical_space):
        """
        Args:
            hull_data (dict) - dictionary generated in GetHullInputData().hull_data
            chemical_space (str) - chemical space to analyze in 'el1_el2_...' (alphabetized) format
        
        Returns:
            grabs only the relevant sub-dict from hull_data
            changes chemical space to tuple (el1, el2, ...)
        """
        hull_data = hull_data[chemical_space]

        keys_to_remove = [k for k in hull_data 
                                  if CompTools(k).n_els == 1]
        hull_data = {k : hull_data[k] for k in hull_data if k not in keys_to_remove}

        els = chemical_space.split('_')
        for el in els:
            hull_data[el] = {'E' : 0,
                             'amts' : {els[i] : 
                                        CompTools(el).mol_frac(els[i])
                                        for i in range(len(els))}}
        self.hull_data = hull_data
        
        self.chemical_space = tuple(els)
        
    @property 
    def sorted_compounds(self):
        """
        Args:
            
        Returns:
            alphabetized list of compounds (str) in specified chemical space
        """
        return sorted(list(self.hull_data.keys()))
    
    def amts_matrix(self, compounds='all', chemical_space='all'):
        """
        Args:
            compounds (str or list) - if 'all', use all compounds; else use specified list
            chemical_space - if 'all', use entire space; else use specified tuple
        
        Returns:
            matrix (2D array) with the fractional composition of each element in each compound (float)
                each row is a different compound (ordered going down alphabetically)
                each column is a different element (ordered across alphabetically)
        """
        if chemical_space == 'all':
            chemical_space = self.chemical_space
        hull_data = self.hull_data
        if compounds == 'all':
            compounds = self.sorted_compounds
        A = np.zeros((len(compounds), len(chemical_space)))
        for row in range(len(compounds)):
            compound = compounds[row]
            for col in range(len(chemical_space)):
                el = chemical_space[col]
                A[row, col] = hull_data[compound]['amts'][el]
        return A
    
    def formation_energy_array(self, compounds='all'):
        """
        Args:
            compounds (str or list) - if 'all', use all compounds; else use specified list
        
        Returns:
            array of formation energies (float) for each compound ordered alphabetically
        """        
        hull_data = self.hull_data
        if compounds == 'all':
            compounds = self.sorted_compounds
        return np.array([hull_data[c]['E'] for c in compounds])
    
    def hull_input_matrix(self, compounds='all', chemical_space='all'):
        """
        Args:
            compounds (str or list) - if 'all', use all compounds; else use specified list
            chemical_space - if 'all', use entire space; else use specified tuple
        
        Returns:
            amts_matrix, but replacing the last column with the formation energy
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
        Args:
            
        Returns:
            scipy.spatial.ConvexHull object
        """
        return ConvexHull(self.hull_input_matrix(compounds='all', chemical_space='all'))
    
    @property
    def hull_points(self):
        """
        Args:
            
        Returns:
            array of points (tuple) fed to ConvexHull
        """        
        return self.hull.points
    
    @property
    def hull_vertices(self):
        """
        Args:
            
        Returns:
            array of indices (int) corresponding with the points that are on the hull
        """         
        return self.hull.vertices
    
    @property
    def hull_simplices(self):
        
        return self.hull.simplices
    
    @property
    def stable_compounds(self):
        """
        Args:
            
        Returns:
            list of compounds that correspond with vertices (str)
        """          
        hull_data = self.hull_data
        hull_vertices = self.hull_vertices
        compounds = self.sorted_compounds
        return [compounds[i] for i in hull_vertices if hull_data[compounds[i]]['E'] <= 0]
    
    @property
    def unstable_compounds(self):
        """
        Args:
            
        Returns:
            list of compounds that do not correspond with vertices (str)
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
                                                        for all compounds in the competing reaction}
                np.nan if decomposition analysis fails
        """            
        hull_data = self.hull_data
        competing_compounds = self.competing_compounds(compound)
        
        if (len(competing_compounds) == 0) or (np.max([CompTools(c).n_els for c in competing_compounds]) == 1):
            return {el : {'amt' : CompTools(compound).stoich(el),
                          'E' : 0} for el in CompTools(compound).els}
        solution = self.decomp_solution(compound)
        if solution.success:
            resulting_amts = solution.x
        elif hull_data[compound]['E'] > 0:
            return {el : {'amt' : CompTools(compound).stoich(el),
                          'E' : 0} for el in CompTools(compound).els}
        else:
            print(compound)
            print('\n\n\nFAILURE!!!!\n\n\n')
            print(compound)
            return np.nan
        min_amt_to_show = 1e-4
        decomp_products = dict(zip(competing_compounds, resulting_amts))
        relevant_decomp_products = [k for k in decomp_products if decomp_products[k] > min_amt_to_show]
        decomp_products = {k : {'amt' : decomp_products[k],
                                'E' : hull_data[k]['E']} if k in hull_data else 0 for k in relevant_decomp_products}
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
        """
        hull_data = self.hull_data
        decomp_products = self.decomp_products(compound)
        if isinstance(decomp_products, float):
            return np.nan
        decomp_enthalpy = 0
        for k in decomp_products:
            decomp_enthalpy += decomp_products[k]['amt']*decomp_products[k]['E']*CompTools(k).n_atoms
        return (hull_data[compound]['E']*CompTools(compound).n_atoms - decomp_enthalpy) / CompTools(compound).n_atoms
    
    @property
    def hull_output_data(self):
        """
        Args:
            
        Returns:
            stability data (dict) for all compounds in the specified chemical space
                {compound (str) : {'Ef' : formation energy (float),
                                   'Ed' : decomposition energy (float),
                                   'rxn' : decomposition reaction (str),
                                   'stability' : stable (True) or unstable (False)}}
        """
        data = {}
        hull_data = self.hull_data
        compounds, stable_compounds = self.sorted_compounds, self.stable_compounds
        for c in compounds:
            if c in stable_compounds:
                stability = True
            else:
                stability = False
            Ef = hull_data[c]['E']
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
        """
        data = {}
        hull_data = self.hull_data
        stable_compounds = self.stable_compounds
        c = compound
        if c in stable_compounds:
            stability = True
        else:
            stability = False
        Ef = hull_data[c]['E']
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
    
class GrandPotentialPD(object):
    
    def __init__(self):
        
        pass
    
class ReactionEnergy(object):
    
    def __init__(self):
        
        pass
    
class GibbsEnergy(object):
    
    def __init__(self):
        
        pass
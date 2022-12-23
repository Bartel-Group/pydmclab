import os
import numpy as np

from pymatgen.io.vasp.outputs import Vasprun, Outcar, Eigenval
from pymatgen.core.structure import Structure
from pymatgen.io.vasp.inputs import Kpoints, Incar
from pymatgen.io.lobster.inputs import Lobsterin
from pymatgen.io.lobster.outputs import Doscar

from pydmc.core.struc import StrucTools, SiteTools
from pydmc.core.comp import CompTools
from pydmc.utils.handy import read_json, write_json

class AnalyzeVASP(object):
    """
    Analyze the results of one VASP calculation
    """
    def __init__(self, calc_dir, calc='from_calc_dir'):
        """
        Args:
            calc_dir (os.PathLike) - path to directory containing VASP calculation
            calc (str) = what kind of calc was done in calc_dir
                - 'from_calc_dir' (default) - determine from calc_dir
                - could also be in ['loose', 'static', 'relax']
                
        Returns:
            calc_dir, calc
        
        """
        
        self.calc_dir = calc_dir
        if calc == 'from_calc_dir':
            self.calc = os.path.split(calc_dir)[-1].split('-')[1]
        else:
            self.calc = calc
            
    @property
    def poscar(self):
        """
        Returns Structure object from POSCAR in calc_dir
        """
        return Structure.from_file(os.path.join(self.calc_dir, 'POSCAR'))
    
    @property
    def contcar(self):
        """
        Returns Structure object from CONTCAR in calc_dir
        """
        return Structure.from_file(os.path.join(self.calc_dir, 'CONTCAR'))

    @property
    def nsites(self):
        """
        Returns number of sites in POSCAR
        """
        return len(self.poscar)
          
    @property
    def vasprun(self):
        """
        Returns Vasprun object from vasprun.xml in calc_dir
        """
        fvasprun = os.path.join(self.calc_dir, 'vasprun.xml')
        if not os.path.exists(fvasprun):
            return None
        try:
            vr = Vasprun(os.path.join(self.calc_dir, 'vasprun.xml'))
            return vr
        except:
            return None     
 
    @property
    def outcar(self):
        """
        Returns Outcar object from OUTCAR in calc_dir
        """
        foutcar = os.path.join(self.calc_dir, 'OUTCAR')
        if not os.path.exists(foutcar):
            return None
        
        try:
            oc = Outcar(os.path.join(self.calc_dir, 'OUTCAR'))
            return oc
        except:
            return None
        
    @property
    def eigenval(self):
        """
        Returns Eigenval object from EIGENVAL in calc_dir
        """
        feigenval = os.path.join(self.calc_dir, 'EIGENVAL')
        if not os.path.exists(feigenval):
            return None
        
        try:
            ev = Eigenval(os.path.join(self.calc_dir, 'EIGENVAL'))
            return ev
        except:
            return None
      
    @property
    def nbands(self):
        """
        Returns number of bands from EIGENVAL
        """
        ev = self.eigenval
        if ev:
            return ev.nbands
        else:
            return None
    
    @property
    def is_converged(self):
        """
        Returns True if VASP calculation is converged, else False
        """
        vr = self.vasprun
        if vr:
            if self.calc == 'static':
                return vr.converged_electronic
            else:
                return vr.converged
        else:
            return False
    
    @property
    def E_per_at(self):
        """
        Returns energy per atom (eV/atom) from vasprun.xml or None if calc not converged
        """
        if self.is_converged:
            vr = self.vasprun
            return vr.final_energy / self.nsites
        else:
            return None
        
    @property
    def formula(self):
        """
        Returns formula of structure from POSCAR (full)
        """
        return self.poscar.formula
    
    @property
    def compact_formula(self):
        """
        Returns formula of structure from POSCAR (compact)
            - use this w/ E_per_at
        """
        return StrucTools(self.poscar).compact_formula
    
    @property
    def incar_parameters(self):
        """
        Returns dict of VASP input settings from vasprun.xml
        """
        vr = self.vasprun
        if vr:
            return vr.parameters
        else:
            return Incar.from_file(os.path.join(self.calc_dir, 'INCAR')).as_dict()
    
    @property
    def sites_to_els(self):
        """
        Returns {site index (int) : element (str) for every site in structure}
        """
        contcar = self.contcar
        return {idx : SiteTools(contcar, idx).el for idx in range(len(contcar))}

            
    @property
    def magnetization(self):
        """
        Returns {element (str) : 
                    {site index (int) : 
                        {'mag' : total magnetization on site}}}
        """
        oc = self.outcar
        if not oc:
            return {}
        mag = list(oc.magnetization)
        if not mag:
            return {}
        sites_to_els = self.sites_to_els
        els = sorted(list(set(sites_to_els.values())))
        return {el : 
                {idx : 
                    {'mag' : mag[idx]['tot']} 
                        for idx in sorted([i for i in sites_to_els if sites_to_els[i] == el])} 
                            for el in els}
        
    @property
    def fdoscar(self):
        """
        Path to DOSCAR for DOS analysis
            - for now, just DOSCAR.lobster is usable
        
        """
        fdoscar = os.path.join(self.calc_dir, 'DOSCAR.lobster')
        if not os.path.exists(fdoscar):
            fdoscar = fdoscar.replace('.lobster', '')
            if not os.path.exists(fdoscar):
                return None
        return fdoscar
    
    @property
    def els_to_orbs(self):
        """
        Returns:
            {element (str) : [list of orbitals (str) considered in calculation]}
        
        """
        if not (self.fdoscar and 'lobster' in self.fdoscar):
            return None
        s_orbs = ['s']
        p_orbs = ['p_x', 'p_y', 'p_z']
        d_orbs = ['d_xy', 'd_yz', 'd_z^2', 'd_xz', 'd_(x^2-y^2)']
        f_orbs = ['f_y(3x^2-y^2)', 'f_xyz', 'f_yz^2', 'f_z^3', 'f_xz^2', 'f_z(x^2-y^2)', 'f_x(x^2-3y^2)']
        all_orbitals = {'s' : s_orbs,
                        'p' : p_orbs,
                        'd' : d_orbs,
                        'f' : f_orbs}
        
        data = {}
        with open(os.path.join(self.calc_dir, 'lobsterin')) as f:
            for line in f:
                if 'basisfunctions' in line:
                    line = line[:-1].split(' ')
                    el = line[1]
                    basis = line[2:-1]
                    data[el] = basis
        #print(data)
        orbs = {}
        for el in data:
            orbs[el] = []
            for basis in data[el]:
                number = basis[0]
                letter = basis[1]
                #print(number)
                #print(letter)
                #print('\n')
                orbitals = [number+v for v in all_orbitals[letter]]
                orbs[el] += orbitals
        return orbs
    
    @property
    def sites_to_orbs(self):
        """
        Returns:
            {site index (int) : [list of orbitals (str) considered in calculation]}
        """
        sites_to_els = self.sites_to_els
        els_to_orbs = self.els_to_orbs
        out = {}
        for site in sites_to_els:
            el = sites_to_els[site]
            orbs = els_to_orbs[el]
            out[site] = orbs
        return out

    def pdos(self, fjson=None, remake=False):
        """
        Returns complex dict of projected DOS data
            - uses DOSCAR.lobster as of now (must run LOBSTER first)
        
        Returns:
            {element (str) :
                {site index in CONTCAR (int) :
                    {orbital (str) (e.g., '2p_x') :
                        {spin (str) (e.g., '1' or ???) : 
                            {DOS (float)}}}}}
            NOTE: there is one more key in the first level
                {'E' : 1d array of energies corresponding with DOS}
                - so when looping through elements, you must
                    for el in pdos:
                        if el != 'E':
                            ...
            
        """
        if not (self.fdoscar and 'lobster' in self.fdoscar):
            raise NotImplementedError('Need a DOSCAR.lobster file')
        if not fjson:
            fjson = os.path.join(self.calc_dir, 'pdos.json')
        if os.path.exists(fjson) and not remake:
            return read_json(fjson)
        
        fdoscar = self.fdoscar
        if not fdoscar:
            return None
        complete_dos = Doscar(doscar=fdoscar,
                                structure_file=os.path.join(self.calc_dir, 'POSCAR')).completedos
        
        sites_to_orbs = self.sites_to_orbs
        sites_to_els = self.sites_to_els
        s = self.contcar
        out = {}
        for site_idx in sites_to_orbs:
            site = s[site_idx]
            el = sites_to_els[site_idx]
            orbitals = sites_to_orbs[site_idx]
            if el not in out:
                out[el] = {site_idx : {}}
            else:
                out[el][site_idx] = {}
            for orbital in orbitals:
                out[el][site_idx][orbital] = {}
                dos = complete_dos.get_site_orbital_dos(site, orbital).as_dict()
                energies = dos['energies']
                for spin in dos['densities']:
                    out[el][site_idx][orbital][spin] = dos['densities'][spin]
        out['E'] = energies
        return write_json(out, fjson)
                
    def tdos(self, pdos=None, fjson=None, remake=False):
        """
        Returns more compact dict than pdos
            - uses DOSCAR.lobster as of now (must run LOBSTER first)
        
        Returns:
            {element (str) :
                1d array of DOS (float) summed for all orbitals, spins, and sites having that element}
            - also has a key "total" : 1d array of total DOS (summed over all orbitals over all e-)
            - also has a key "E" just like in pdos (1d array of energies aligning with each DOS)

        @TODO: add demo/test
        @TODO: explore for magnetic materials
        @TODO: add options for summing or not summing spins
        @TODO: work on generic plotting
        @TODO: work on COHPCAR/COOPCAR
        """
        if not (self.fdoscar and 'lobster' in self.fdoscar):
            raise NotImplementedError('Need a DOSCAR.lobster file')
        if not fjson:
            fjson = os.path.join(self.calc_dir, 'tdos.json')
        if os.path.exists(fjson) and not remake:
            return read_json(fjson)       
        if not pdos:
            pdos = self.pdos
        out = {}
        energies = pdos['E']
        for el in pdos:
            if el == 'E':
                continue
            out[el] = np.zeros(len(energies))
            for site in pdos[el]:
                for orb in pdos[el][site]:
                    for spin in pdos[el][site][orb]:
                        out[el] += np.array(pdos[el][site][orb][spin])
        out['total'] = np.zeros(len(energies))
        for el in out:
            if el == 'total':
                continue
            out['total'] += np.array(out[el])
            print(el)
            print(out[el][690])
            print(out['total'][690])

        out['E'] = energies
        for k in out:
            out[k] = list(out[k])
        return write_json(out, fjson)
                              

class AnalyzeBatch(object):

    def __init__(self,
                 launch_dirs_to_tags,
                 get_mag=False,
                 get_structures=False,
                 get_dos=False,
                 get_metadata=False):

        self.launch_dirs_to_tags = launch_dirs_to_tags
    
    @property
    def calc_dirs(self):
        launch_dirs = self.launch_dirs_to_tags
        calc_dirs = []
        for launch_dir in launch_dirs:
            calc_dirs += [os.path.join(launch_dir, c) for c in launch_dirs[launch_dir]]
        return calc_dirs
    
    def get_metadata(self):
        calc_dirs = self.calc_dirs
        for calc_dir in calc_dirs:
            av = AnalyzeVASP(calc_dir)

    
    def get_results(self,
                    top_level_key='formula',
                    magnetization=False,
                    relaxed_structure=False,
                    dos=None,
                    use_static=True,
                    check_relax=0.1):
        launch_dirs = self.launch_dirs_to_tags
        data = []
        for launch_dir in launch_dirs:
            print('\n~~~ analyzing %s ~~~' % launch_dir)
            top, ID, standard, xc, mag = launch_dir.split('/')[-5:]
            xc_calcs = launch_dirs[launch_dir]
            if use_static:
                xc_calcs = [c for c in xc_calcs if c.split('-')[-1] == 'static']
            for xc_calc in xc_calcs:
                print('     working on %s' % xc_calc)
                calc_data = {'info' : {},
                             'summary' : {},
                             'flags' : []}
                if magnetization:
                    calc_data['magnetization'] = {}
                if relaxed_structure:
                    calc_data['structure'] = {}
                if dos:
                    calc_data['dos'] = {}
                calc_dir = os.path.join(launch_dir, xc_calc)
                xc, calc = xc_calc.split('-')
                
                calc_data['info']['calc_dir'] = calc_dir
                calc_data['info']['mag'] = mag
                calc_data['info']['standard'] = standard
                calc_data['info'][top_level_key] = top
                calc_data['info']['ID'] = ID
                calc_data['info']['xc'] = xc
                calc_data['info']['calc'] = calc

                analyzer = AnalyzeVASP(calc_dir)
                convergence = analyzer.is_converged
                E_per_at = analyzer.E_per_at
                if convergence:
                    if (calc == 'static') and check_relax:
                        relax_calc_dir = calc_dir.replace('static', 'relax')
                        analyzer_relax = AnalyzeVASP(relax_calc_dir)
                        convergence_relax = analyzer_relax.is_converged
                        if not convergence_relax:
                            convergence = False
                            E_per_at = None
                        E_relax = analyzer_relax.E_per_at
                        if E_per_at and E_relax:
                            E_diff = abs(E_per_at - E_relax)
                            if E_diff > check_relax:
                                data['flags'].append('large E diff b/t relax and static')
                
                calc_data['summary']['E'] = E_per_at
                calc_data['summary']['convergence'] = convergence
                if not convergence:
                    calc_data['flags'].append('not converged')
                    
                if relaxed_structure:
                    if convergence:
                        structure = analyzer.contcar.as_dict()
                    else:
                        structure = None
                    calc_data['structure'] = structure
                
                if magnetization:
                    if convergence:
                        calc_data['magnetization'] = analyzer.magnetization

                if dos:
                    if dos == 'tdos':
                        calc_data['dos'] = analyzer.tdos()
                    elif dos == 'pdos':
                        calc_data['dos'] = analyzer.pdos()
                    else:
                        raise NotImplementedError('only tdos and pdos are accepted args for dos')
                
                data.append(calc_data)

        return {'data' : data}
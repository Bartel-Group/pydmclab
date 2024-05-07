import os
import numpy as np
import multiprocessing as multip
import json

from pymatgen.io.vasp.outputs import Vasprun, Outcar, Eigenval, Oszicar
from pymatgen.core.structure import Structure
from pymatgen.io.vasp.inputs import Kpoints, Incar
from pymatgen.io.lobster.outputs import Doscar, Cohpcar, Charge, MadelungEnergies

from pydmclab.core.struc import StrucTools, SiteTools
from pydmclab.core.comp import CompTools
from pydmclab.utils.handy import read_json, write_json, read_yaml, write_yaml
from pydmclab.data.configs import load_base_configs


class VASPOutputs(object):
    def __init__(self, calc_dir):
        """

        Args:
            calc_dir (str): path to where you ran VASP
        """

        self.calc_dir = calc_dir

    @property
    def vasprun(self):
        """
        Returns Vasprun object from vasprun.xml in calc_dir
        """
        fvasprun = os.path.join(self.calc_dir, "vasprun.xml")
        if not os.path.exists(fvasprun):
            return None

        try:
            vr = Vasprun(os.path.join(self.calc_dir, "vasprun.xml"))
            return vr
        except:
            return None

    @property
    def poscar(self):
        """
        Returns Structure object from POSCAR in calc_dir
        """
        fposcar = os.path.join(self.calc_dir, "POSCAR")
        if not os.path.exists(fposcar):
            return None

        try:
            return Structure.from_file(os.path.join(self.calc_dir, "POSCAR"))
        except:
            return None

    @property
    def incar(self):
        """
        Returns dict of VASP input settings from INCAR
        """
        fincar = os.path.join(self.calc_dir, "INCAR")
        if os.path.exists(fincar):
            return Incar.from_file(os.path.join(self.calc_dir, "INCAR"))
        else:
            return {}

    @property
    def all_input_settings(self):
        """
        Returns dict of VASP input settings from vasprun (ie what VASP actually used)
        """
        vr = self.vasprun
        if vr:
            return vr.parameters
        return {}

    @property
    def kpoints(self):
        """
        Returns Kpoints object from KPOINTS in calc_dir
        """
        fkpoints = os.path.join(self.calc_dir, "KPOINTS")

        if not os.path.exists(fkpoints):
            return None
        try:
            return Kpoints.from_file(fkpoints)
        except:
            return None

    @property
    def actual_kpoints(self):
        """
        Returns actual kpoints that were used (list of [a,b,c] for each kpoint)
        """
        vr = self.vasprun
        if vr:
            return vr.actual_kpoints
        else:
            return None

    @property
    def potcar(self):
        """
        Returns {el : {'name' : POTCAR name, 'pp' : pseudopotential name, 'date' : date of POTCAR, 'nval' : number of valence electrons}}}
        """
        fpot = os.path.join(self.calc_dir, "POTCAR")
        if not os.path.exists(fpot):
            return None
        out = {}
        with open(fpot) as f:
            for line in f:
                if "VRHFIN" in line:
                    el = line.split("=")[1].split(":")[0].strip()
                if "TITEL" in line:
                    tmp_dict = {}
                    line = line.split("=")[1].split(" ")
                    tmp_dict["name"] = line[1]
                    tmp_dict["pp"] = line[2]
                    tmp_dict["date"] = line[3][:-1]
                if "ZVAL" in line:
                    line = line.split(";")[1].split("=")[1].split("mass")[0]
                    tmp_dict["nval"] = int(
                        float("".join([val for val in line if val != " "]))
                    )
                    out[el] = tmp_dict
        return out

    @property
    def contcar(self):
        """
        Returns Structure object from CONTCAR in calc_dir
        """
        fcontcar = os.path.join(self.calc_dir, "CONTCAR")
        if not os.path.exists(fcontcar):
            return None
        try:
            return Structure.from_file(os.path.join(self.calc_dir, "CONTCAR"))
        except:
            return None

    @property
    def outcar(self):
        """
        Returns Outcar object from OUTCAR in calc_dir
        """
        foutcar = os.path.join(self.calc_dir, "OUTCAR")
        if not os.path.exists(foutcar):
            return None

        try:
            oc = Outcar(os.path.join(self.calc_dir, "OUTCAR"))
            return oc
        except:
            return None

    @property
    def oszicar(self):
        """
        Returns Oszicar object from OSZICAR in calc_dir
        """
        foszicar = os.path.join(self.calc_dir, "OSZICAR")
        if not os.path.exists(foszicar):
            return None

        try:
            oz = Oszicar(os.path.join(self.calc_dir, "OSZICAR"))
            return oz
        except:
            return None

    @property
    def eigenval(self):
        """
        Returns Eigenval object from EIGENVAL in calc_dir
        """
        feigenval = os.path.join(self.calc_dir, "EIGENVAL")
        if not os.path.exists(feigenval):
            return None

        try:
            ev = Eigenval(os.path.join(self.calc_dir, "EIGENVAL"))
            return ev
        except:
            return None

    def doscar(self, fdoscar="DOSCAR.lobster"):
        """
        fdoscar (str) - 'DOSCAR' or 'DOSCAR.lobster'

        Returns:
            Doscar object from DOSCAR in calc_dir
        """
        fdoscar = os.path.join(self.calc_dir, fdoscar)
        if not os.path.exists(fdoscar):
            return None

        try:
            dos = Doscar(
                doscar=fdoscar, structure_file=os.path.join(self.calc_dir, "CONTCAR")
            )
        except:
            return None

        return dos

    def cohpcar(self, are_coops=False, are_cobis=False):
        """
        Args:
            are_coops (bool) : if True, use COOPCAR
            are_cobis (bool) : if True, use COBI?
        """
        if are_coops:
            flobster = os.path.join(self.calc_dir, "COOPCAR.lobster")
        elif are_cobis:
            flobster = os.path.join(self.calc_dir, "COBICAR.lobster")
        else:
            flobster = os.path.join(self.calc_dir, "COHPCAR.lobster")

        if not os.path.exists(flobster):
            return None

        try:
            cohp = Cohpcar(are_coops=are_coops, are_cobis=are_cobis, filename=flobster)
        except:
            return None

        return cohp

    @property
    def lobsterin(self):
        """
        helps set up Lobsterin file

        Returns:
            {el (str) : {'orbs' : [orbital names], 'basis' : basis name}}
        """
        s_orbs = ["s"]
        p_orbs = ["p_x", "p_y", "p_z"]
        d_orbs = ["d_xy", "d_yz", "d_z^2", "d_xz", "d_x^2-y^2"]
        f_orbs = [
            "f_y(3x^2-y^2)",
            "f_xyz",
            "f_yz^2",
            "f_z^3",
            "f_xz^2",
            "f_z(x^2-y^2)",
            "f_x(x^2-3y^2)",
        ]
        all_orbitals = {"s": s_orbs, "p": p_orbs, "d": d_orbs, "f": f_orbs}

        basis_functions = {}
        with open(os.path.join(self.calc_dir, "lobsterin")) as f:
            for line in f:
                if "basisfunctions" in line:
                    line = line[:-1].split(" ")
                    el = line[1]
                    basis = line[2:-1]
                    basis_functions[el] = basis

        # print(data)
        orbs = {}
        for el in basis_functions:
            orbs[el] = []
            for basis in basis_functions[el]:
                number = basis[0]
                letter = basis[1]
                # print(number)
                # print(letter)
                # print('\n')
                orbitals = [number + v for v in all_orbitals[letter]]
                orbs[el] += orbitals

        data = {el: {"orbs": orbs[el], "basis": basis_functions[el]} for el in orbs}

        return data


class AnalyzeVASP(object):
    """
    Analyze the results of one VASP calculation
    """

    def __init__(self, calc_dir, calc=None):
        """
        Args:
            calc_dir (os.PathLike) - path to directory containing VASP calculation
            calc (str) = what kind of calc was done in calc_dir
                - if None, infer from calc_dir
                - could also be in ['loose', 'static', 'relax']

        Returns:
            calc_dir, calc
            outputs (VASPOutputs(calc_dir))

        """

        self.calc_dir = calc_dir
        if not calc:
            self.calc = os.path.split(calc_dir)[-1].split("-")[1]
        else:
            self.calc = calc

        self.outputs = VASPOutputs(calc_dir)

    @property
    def is_converged(self):
        """
        Returns True if VASP calculation is converged, else False
        """
        vr = self.outputs.vasprun
        incar = self.outputs.incar
        if vr:
            if int(incar["NSW"]) in [0, 1]:
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
            vr = self.outputs.vasprun
            nsites = self.nsites

            if nsites and vr.final_energy:
                return vr.final_energy / nsites
            return None
        else:
            return None

    @property
    def nsites(self):
        """
        Returns number of sites in POSCAR
        """
        poscar = self.outputs.poscar
        if poscar:
            return len(poscar)
        else:
            return None

    @property
    def nbands(self):
        """
        Returns number of bands from EIGENVAL
        """
        ev = self.outputs.eigenval
        if ev:
            return ev.nbands
        else:
            return None

    @property
    def sites_to_els(self):
        """
        Returns {site index (int) : element (str) for every site in structure}
        """
        contcar = self.outputs.contcar
        if not contcar:
            return None
        return {idx: SiteTools(contcar, idx).el for idx in range(len(contcar))}

    @property
    def magnetization(self):
        """
        Returns {element (str) :
                    {site index (int) :
                        total magnetization on site (float)}}
        """
        oc = self.outputs.outcar
        if not oc:
            return {}
        mag = list(oc.magnetization)
        if not mag:
            return {}
        sites_to_els = self.sites_to_els
        els = sorted(list(set(sites_to_els.values())))
        return {
            el: {
                str(idx): mag[idx]["tot"]
                for idx in sorted([i for i in sites_to_els if sites_to_els[i] == el])
            }
            for el in els
        }

    @property
    def gap_properties(self):
        """
        Returns {'bandgap' : bandgap (eV),
                 'is_direct' : True if gap is direct else False,
                 'cbm' : cbm (eV),
                 'vbm' : vbm (eV)}
        """
        vr = self.outputs.vasprun
        if vr:
            props = vr.eigenvalue_band_properties
            return {
                "bandgap": props[0],
                "cbm": props[1],
                "vbm": props[2],
                "is_direct": props[3],
            }
        else:
            return None

    @property
    def els_to_orbs(self):
        """
        Returns:
            {element (str) : [list of orbitals (str) considered in calculation]}

        """
        lobsterin = self.outputs.lobsterin
        return {el: lobsterin[el]["orbs"] for el in lobsterin}

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

    @property
    def compact_trajectory(self):
        """
        Returns:
            list of tuples summarizing the trajectory : (step, energy, structure (pymatgen.Structure))
        """
        if self.is_converged:
            vr = self.outputs.vasprun
            nsites = self.nsites

            if vr and nsites:
                energies = [step["e_wo_entrp"] / nsites for step in vr.ionic_steps]
                structures = [step["structure"].as_dict() for step in vr.ionic_steps]
                return list(zip(range(len(energies)), energies, structures))
            return None
        else:
            return None

    def pdos(self, fjson=None, remake=False):
        """
        @TODO: add demo/test

        Returns complex dict of projected DOS data
            - uses DOSCAR.lobster as of now (must run LOBSTER first)

        Args:
            fjson (os.PathLike) - path to json file to save pdos data to
            remake (bool) - if True, remake pdos data even if fjson exists

        Returns:
            - dictionary (let's call it d)
            - list(d.keys()) = ['E'] + [list of elements (str)]
                - e.g., if I calculate Ca2Ru2O7, the keys will be ['E', 'Ca', 'Ru', 'O']
            - d['E'] = 1d array of energies corresponding with DOS
            - d[el] =
                {site index in CONTCAR (str) :
                    {orbital (str) (e.g., '2p_x') :
                        spin (str) (e.g., '1' (spin up) or '-1' (spin down) :
                            DOS (1d array)}}
                - elaborating
                    - d[el].keys() would be all the sites where that element appears in the structure ['4', '5', '6', ...]
                    - d[el][site].keys() would be the orbitals for all valence electrons in the calculation ['4d_xy', '4d_xz', '4d_yz', ...]
                    - d[el][site][orb].keys() would be the spins ['1', '-1']

            - so if I wanted to plot the spin-up 4d_xy DOS for Ru on site 8, it would be:
                energies = d['E']
                dos = d['Ru']['8']['4d_xy']['1']
                plt.plot(dos, energies)

        """

        if not fjson:
            fjson = os.path.join(self.calc_dir, "pdos.json")
        if os.path.exists(fjson) and not remake:
            try:
                return read_json(fjson)
            except json.decoder.JSONDecodeError:
                pass

        doscar = self.outputs.doscar()
        if not doscar:
            return None

        complete_dos = doscar.completedos

        sites_to_orbs = self.sites_to_orbs
        sites_to_els = self.sites_to_els
        structure = self.outputs.contcar
        out = {}
        for site_idx in sites_to_orbs:
            site = structure[site_idx]
            el = sites_to_els[site_idx]
            orbitals = sites_to_orbs[site_idx]
            if el not in out:
                out[el] = {str(site_idx): {}}
            else:
                out[el][str(site_idx)] = {}
            for orbital in orbitals:
                out[el][str(site_idx)][orbital] = {}
                dos = complete_dos.get_site_orbital_dos(site, orbital).as_dict()
                energies = dos["energies"]
                for spin in dos["densities"]:
                    out[el][str(site_idx)][orbital][spin] = dos["densities"][spin]
        out["E"] = energies
        write_json(out, fjson)
        return read_json(fjson)

    def tdos(self, pdos=None, fjson=None, remake=False):
        """
        Returns more compact dict than pdos
            - uses DOSCAR.lobster as of now (must run LOBSTER first)

        Args:
            pdos (dict or None) - if you don't pass the pdos dictionary, it will make it before working on tdos
            fjson (os.PathLike) - path to json file to save pdos data to
            remake (bool) - if True, remake pdos data even if fjson exists

        Returns:
            - dictionary (let's call it d)
            - list(d.keys()) = ['E', 'total', 'up', 'down'] + [list of elements (str)]
            - d['E'] = 1d array of energies corresponding with DOS
            - d[el] = 1d array of DOS for that element (sums all sites, orbitals, and spins)
            - d['total'] = 1d array of DOS for structure (sums all elements, sites, orbitals, spins)
            - d['up'] or d['down']:
                - keys are ['total'] + [list of elements (str)]
                - d['up']['total'] = 1d array of spin-up DOS for structure
                - d['down'][el] = 1d array of spin-down DOS for that element
                - etc

            - so if I wanted to plot the total DOS for my structure and separate spin up (+ DOS) and spin down (- DOS)

                energies = d['E']
                dos_up = d['up']['total']
                dos_down = d['down']['total']
                plt.plot(dos_up, energies)
                plt.plot(-1*dos_down, energies)

        @TODO: add demo/test (reasonably well tested now)
        @TODO: work on generic plotting
        """
        if not fjson:
            fjson = os.path.join(self.calc_dir, "tdos.json")
        if os.path.exists(fjson) and not remake:
            try:
                return read_json(fjson)
            except json.decoder.JSONDecodeError:
                pass
        if not pdos:
            pdos = self.pdos()
        if not pdos:
            return None
        out = {"up": {}, "down": {}}
        energies = pdos["E"]
        for el in pdos:
            if el == "E":
                continue
            out[el] = np.zeros(len(energies))
            out["up"][el] = np.zeros(len(energies))
            out["down"][el] = np.zeros(len(energies))
            for site in pdos[el]:
                for orb in pdos[el][site]:
                    for spin in pdos[el][site][orb]:
                        if int(spin) == 1:
                            out["up"][el] += np.array(pdos[el][site][orb][spin])
                        elif int(spin) == -1:
                            out["down"][el] += np.array(pdos[el][site][orb][spin])
                        out[el] += np.array(pdos[el][site][orb][spin])
        out["total"] = np.zeros(len(energies))
        for el in out:
            if el in ["up", "down", "total"]:
                continue
            out["total"] += np.array(out[el])
        for spin in ["up", "down"]:
            out[spin]["total"] = np.zeros(len(energies))
            for el in out[spin]:
                if el != "total":
                    out[spin]["total"] += np.array(out[spin][el])
        out["E"] = energies
        for k in out:
            if k not in ["up", "down"]:
                out[k] = list(out[k])
        for spin in ["up", "down"]:
            for k in out[spin]:
                out[spin][k] = list(out[spin][k])
        write_json(out, fjson)
        return read_json(fjson)

    def pcohp(self, fjson=None, remake=False, are_coops=False, are_cobis=False):
        """
        @TODO: add demo/test

        Returns complex dict of projected COHP, COOP, or COBI data
            - must run LOBSTER first
            - note: COHP, COOP, and COBI have similar format so they get processed the same way. If you want all three, you might do:
                for bonding in ["cohp", "coop", "cobi"]:
                    cohp_data = self.pcohp(are_coops=False, are_cobis=False)
                    coop_data = self.pcohp(are_coops=True, are_cobis=False)
                    cobi_data = self.pcohp(are_coops=False, are_cobis=True)

        Args:
            fjson (os.PathLike) - path to json file to save pdos data to
            remake (bool) - if True, remake pchohp data even if fjson exists
            are_coops (bool) - True if you want COOP analysis
            are_cobis (bool) - True if you want COBI analysis
                - both False = COHP analysis

        Returns:
            - dictionary (let's call it d)
            - list(d.keys()) = ['E'] + [el1-el2 (str) for all pairs of elements in structure that might be bonding]
                - note: el1-el2 is sorted alphabetically
                - e.g., the keys for Ca2Ru2O7 might be: ['Ca-Ca', 'Ca-Ru', 'Ca-O', 'Ru-Ru', 'O-Ru', 'O-O', 'E']
            - d['E'] = 1d array of energies corresponding with DOS
            - for each "bond type" (el1-el2), we have particular bonds:
                - d[el1-el2].keys() might be ['16-8', '17-8', '18-8', '16-9', ...]
                    - indicating specific between el1 on site 16 and el2 on site 8, etc
                - for each specific bond, we have three keys:
                    d[el1-el2][site1-site].keys() = ['cohp', 'icohp', 'length']
                    - 'length' returns simply the bond length in Angstrom (float)
                    - 'cohp' and 'icohp' have the same structure
                        - 'cohp' refers to the absolute COHP/COOP/COBI and 'icohp' refers to the integrated COHP/COOP/COBI (don't remember exactly how this integral is done but I remember not loving it...)
                        - d[el1-el2][site1-site]['cohp'].keys() = ['1', '-1', 'total']
                            - '1' and '-1' refer to spin up and spin down, respectively
                            - 'total' means summing over these two spins
                            - d[el1-el2][site1-site]['cohp']['1' or '-1' or 'total'] gives a 1d array of populations

            - so if I wanted to plot the total COHP for the interaction between Ru on site 8 and O on site 16, I would do:
                energies = d['E']
                cohp = d['O-Ru']['16-8']['cohp']['total']
                plt.plot(cohp, energies)

        """

        if are_coops:
            savename = "pcoop.json"
        elif are_cobis:
            savename = "pcobi.json"
        else:
            savename = "pcohp.json"

        if not fjson:
            fjson = os.path.join(self.calc_dir, savename)
        if os.path.exists(fjson) and not remake:
            try:
                return read_json(fjson)
            except json.decoder.JSONDecodeError:
                pass

        cohpcar = self.outputs.cohpcar(are_coops=are_coops, are_cobis=are_cobis)
        if not cohpcar:
            return None

        energies = cohpcar.energies

        cohp = cohpcar.cohp_data

        sites_to_els = self.sites_to_els

        out = {}
        for bond_idx in cohp:
            if bond_idx == "average":
                continue
            sites = cohp[bond_idx]["sites"]
            els = [sites_to_els[site] for site in sites]
            if sorted(els) == els:
                sorted_sites = sites
            else:
                sorted_sites = (sites[1], sites[0])
            el_tag = "-".join(sorted(els))
            site_tag = "-".join([str(site) for site in sorted_sites])
            bond_length = cohp[bond_idx]["length"]
            if el_tag not in out:
                out[el_tag] = {}
            out[el_tag][site_tag] = {
                "cohp": {
                    "1": list(np.zeros(len(energies))),
                    "-1": list(np.zeros(len(energies))),
                },
                "icohp": {
                    "-1": list(np.zeros(len(energies))),
                    "1": list(np.zeros(len(energies))),
                },
                "length": bond_length,
            }
            out[el_tag][site_tag]["cohp"]["total"] = np.zeros(len(energies))
            out[el_tag][site_tag]["icohp"]["total"] = np.zeros(len(energies))
            for spin in cohp[bond_idx]["COHP"]:
                if spin.name == "up":
                    spin_tag = "1"
                else:
                    spin_tag = "-1"
                out[el_tag][site_tag]["cohp"][spin_tag] = list(
                    cohp[bond_idx]["COHP"][spin]
                )
                out[el_tag][site_tag]["icohp"][spin_tag] = list(
                    cohp[bond_idx]["ICOHP"][spin]
                )
                out[el_tag][site_tag]["cohp"]["total"] += cohp[bond_idx]["COHP"][spin]
                out[el_tag][site_tag]["icohp"]["total"] += cohp[bond_idx]["ICOHP"][spin]

        for el_tag in out:
            for site_tag in out[el_tag]:
                for key in ["cohp", "icohp"]:
                    tmp = out[el_tag][site_tag][key]["total"]
                    out[el_tag][site_tag][key]["total"] = list(tmp)

        out["E"] = list(energies)

        write_json(out, fjson)
        return read_json(fjson)

    def tcohp(
        self, pcohp=None, fjson=None, remake=False, are_coops=False, are_cobis=False
    ):
        """
        @TODO: add demo/test

        Returns simpler dict of projected COHP, COOP, or COBI data
            - must run LOBSTER first
            - note: COHP, COOP, and COBI have similar format so they get processed the same way. If you want all three, you might do:
                for bonding in ["cohp", "coop", "cobi"]:
                    cohp_data = self.pcohp(are_coops=False, are_cobis=False)
                    coop_data = self.pcohp(are_coops=True, are_cobis=False)
                    cobi_data = self.pcohp(are_coops=False, are_cobis=True)

        Args:
            pcohp (dict) - if you already have pcohp data, you can pass it in here
            fjson (os.PathLike) - path to json file to save pdos data to
            remake (bool) - if True, remake pchohp data even if fjson exists
            are_coops (bool) - True if you want COOP analysis
            are_cobis (bool) - True if you want COBI analysis
                - both False = COHP analysis

        Returns:
            - dictionary (let's call it d)
            - list(d.keys()) = ['E'] + [el1-el2 (str) for all pairs of elements in structure that might be bonding]
                - note: el1-el2 is sorted alphabetically
                - e.g., the keys for Ca2Ru2O7 might be: ['Ca-Ca', 'Ca-Ru', 'Ca-O', 'Ru-Ru', 'O-Ru', 'O-O', 'E']
            - d['E'] = 1d array of energies corresponding with DOS
            - for each "bond type" (el1-el2), we have two keys: ['cohp', 'icohp']
                - cohp --> absolute populations
                - icohp --> integrated populations
                - d[el1-el2]['cohp' or 'icohp'] will return a 1d array of populations summed over all interactions of el1-el2 in the structure, summed over all spins

            - so if I wanted to plot the total COHP for the interaction between Ru and O throughout the structure:
                energies = d['E']
                cohp = d['O-Ru']['cohp']
                plt.plot(cohp, energies)

        """

        if are_coops:
            savename = "tcoop.json"
        elif are_cobis:
            savename = "tcobi.json"
        else:
            savename = "tcohp.json"

        if not fjson:
            fjson = os.path.join(self.calc_dir, savename)

        if os.path.exists(fjson) and not remake:
            try:
                return read_json(fjson)
            except json.decoder.JSONDecodeError:
                pass
        if not pcohp:
            pcohp = self.pcohp(are_coops=are_coops, are_cobis=are_cobis)
        if not pcohp:
            return None

        out = {"E": pcohp["E"]}
        for el_tag in pcohp:
            if el_tag == "E":
                continue
            out[el_tag] = {}
            tmp_cohp = np.zeros(len(out["E"]))
            tmp_icohp = np.zeros(len(out["E"]))
            for site_tag in pcohp[el_tag]:
                for spin in pcohp[el_tag][site_tag]["cohp"]:
                    cohp_to_add = np.array(pcohp[el_tag][site_tag]["cohp"][spin])
                    #                    print(cohp_to_add)
                    if len(cohp_to_add) == len(tmp_cohp):
                        tmp_cohp = np.add(cohp_to_add, tmp_cohp)

                    icohp_to_add = np.array(pcohp[el_tag][site_tag]["icohp"][spin])
                    if len(icohp_to_add) == len(tmp_icohp):
                        tmp_icohp = np.add(icohp_to_add, tmp_icohp)

            out[el_tag]["cohp"] = tmp_cohp
            out[el_tag]["icohp"] = tmp_icohp

        for el_tag in out:
            if el_tag == "E":
                continue
            for key in ["cohp", "icohp"]:
                tmp = out[el_tag][key]
                out[el_tag][key] = list(tmp)

        write_json(out, fjson)
        return read_json(fjson)

    def charge(self, source="bader", fjson=None, remake=False):
        """

        for obtaining partial charges resulting from dft calculations

        Args:
            fjson (_type_, optional): _description_. Defaults to None.
            remake (bool, optional): _description_. Defaults to False.

        Raises:
            NotImplementedError: _description_

        Returns:
            {el (str) : {site index (str) : {charge (float)}}}
        """
        if not fjson:
            fjson = os.path.join(self.calc_dir, "charge-%s.json" % source)
        if os.path.exists(fjson) and not remake:
            return read_json(fjson)

        if source == "bader":
            fcharge = os.path.join(self.calc_dir, "ACF.dat")
            if not os.path.exists(fcharge):
                return {}
            sites_to_els = self.sites_to_els
            nsites = len(self.outputs.contcar)
            pseudos = self.outputs.potcar
            data = {}
            with open(fcharge) as f:
                count = 0
                for line in f:
                    count += 1
                    if count < 3:
                        continue
                    if count > 2 + nsites:
                        break
                    line = [v for v in line[:-1].split(" ") if v != ""]
                    idx, charge = int(line[0]) - 1, float(line[4])
                    el = sites_to_els[idx]
                    nval = pseudos[el]["nval"]
                    delta_charge = nval - charge
                    data[idx] = {"el": el, "charge": delta_charge}
            out = {el: {} for el in list(set(sites_to_els.values()))}
            for idx in data:
                el = data[idx]["el"]
                out[el][str(idx)] = data[idx]["charge"]

        elif source in ["mulliken", "lowdin"]:
            fcharge = os.path.join(self.calc_dir, "CHARGE.lobster")
            if not os.path.exists(fcharge):
                return {}
            chg = Charge(os.path.join(self.calc_dir, "CHARGE.lobster"))
            sites_to_els = self.sites_to_els
            charges = chg.Mulliken if source == "mulliken" else chg.Loewdin
            out = {el: {} for el in list(set(sites_to_els.values()))}
            for idx in sites_to_els:
                el = sites_to_els[idx]
                out[el][str(idx)] = charges[idx]

        write_json(out, fjson)
        return read_json(fjson)

    def charged_structure(self, source, structure=False):
        """
        decorates structures with optimized magnetic moments

        Args:
            source (str): source of charge, e.g. "bader", "mulliken", "lowdin"
            structure (bool, optional): if you pass a structure, it will start from it. otherwise, it will use the CONTCAR

        Returns:
            structure where each site has a "site property" called source
            - so, if s = AnalyzeVASP(calc_dir).charged_structure(source='bader'), then s[0].properties["bader"] will give you the bader charge for that site
        """
        if not structure:
            structure = self.outputs.contcar
        charges = self.charge(source=source)
        if not charges:
            return structure
        charges_by_site = {
            idx: charge for el in charges for idx, charge in charges[el].items()
        }
        sorted_charges = [charges_by_site[str(idx)] for idx in range(len(structure))]
        structure.add_site_property(source, sorted_charges)
        return structure

    def magnetic_structure(self, structure=False):
        """
        decorates structures with optimized magnetic moments

        Args:
            structure (bool, optional): if you pass a structure, it will start from it. otherwise, it will use the CONTCAR

        Returns:
            structure where each site has a "site property" called "final_magmom"
            - so, if s = AnalyzeVASP(calc_dir).magnetic_structure(), then s[0].properties["final_magmom"] will give you the optimized magnetic moment of the first site
        """
        if not structure:
            structure = self.outputs.contcar
        mags = self.magnetization
        if not mags:
            return structure
        mags_by_site = {idx: mag for el in mags for idx, mag in mags[el].items()}
        sorted_mags = [mags_by_site[str(idx)] for idx in range(len(structure))]
        structure.add_site_property("final_magmom", sorted_mags)
        return structure

    def decorated_structure(self, charge_source="bader", structure=None):
        """
        decorates structures with optimized magnetic moments

        Args:
            structure (bool, optional): if you pass a structure, it will start from it. otherwise, it will use the CONTCAR

        Returns:
            structure where each site has two "site properties": ["final_magmom", charge_source]
            - so, if s = AnalyzeVASP(calc_dir).magnetic_structure(), then s[0].properties["final_magmom"] will give you the optimized magnetic moment of the first site
        """
        if not structure:
            structure = self.outputs.contcar
        mag_struc = self.magnetic_structure(structure=structure)
        dec_struc = self.charged_structure(source=charge_source, structure=mag_struc)
        return dec_struc

    @property
    def E_madelung(self):
        """
        Returns the Madelung energy for a structure
            - must run lobster

        Returns:
            {'mulliken' : Madelung energy using Mulliken charges,
            'lowdin' : Madelung energy using Lowdin charges}
        """
        fmadelung = os.path.join(self.calc_dir, "MadelungEnergies.lobster")
        if os.path.exists(fmadelung):
            return {
                "mulliken": MadelungEnergies(
                    filename=fmadelung
                ).madelungenergies_Mulliken,
                "lowdin": MadelungEnergies(filename=fmadelung).madelungenergies_Loewdin,
            }
        return {}

    @property
    def formula(self):
        """
        returns the compact formula of the calculated structure

        Returns:
            compact formula (str) or None
        """
        contcar = self.outputs.contcar
        if contcar:
            return StrucTools(contcar).compact_formula
        poscar = self.outputs.poscar
        if poscar:
            return StrucTools(poscar).compact_formula
        return None

    @property
    def spacegroup_info(self):
        """
        Returns:
            dict of spacegroup info with 'tight' or 'loose' symmetry tolerance
            e.g.,
                data['tight']['number'] returns spacegroup number with tight tolerance
                data['loose']['symbol'] returns spacegroup symbol with loose tolerance

        """
        if self.outputs.contcar:
            return StrucTools(self.outputs.contcar).spacegroup_info
        else:
            return None

    @property
    def basic_info(self):
        """
        Returns:
            {'convergence' : True if calc converged else False,
            'E_per_at' : energy per atom if calc converged else None,
            'formula' : compact formula of computed structure,
            'sg' : dict of spacegroup information}
        """
        E_per_at = self.E_per_at
        if self.calc == "static":
            if os.path.exists(self.calc_dir.replace("static", "relax")):
                E_relax = AnalyzeVASP(self.calc_dir.replace("static", "relax")).E_per_at
                if E_relax and E_per_at and (abs(E_relax - E_per_at) < 0.1):
                    convergence = True
                else:
                    convergence = False
            else:
                convergence = True if E_per_at else False
        else:
            convergence = True if E_per_at else False
        return {
            "convergence": convergence,
            "E_per_at": E_per_at,
            "formula": self.formula,
            "sg": self.spacegroup_info,
        }

    @property
    def relaxed_structure(self):
        """

        Returns:
            pymatgen Structure.as_dict() for relaxed structure
        """
        structure = self.outputs.contcar
        if structure:
            return structure.as_dict()
        else:
            return None

    @property
    def run_stats(self):
        outcar = self.outputs.outcar
        if outcar:
            run_stats = outcar.run_stats
            run_stats.update(self.outcar_metadata)
            return run_stats
        else:
            return None

    @property
    def outcar_metadata(self):
        foutcar = os.path.join(self.calc_dir, "OUTCAR")
        if not os.path.exists(foutcar):
            return None
        vasp_version = None
        execution_date = None
        with open(foutcar, "r") as f:
            for line in f:
                if ("vasp" in line) and not vasp_version:
                    line = line.split(" ")
                    vasp_version = [v for v in line if "vasp" in v][0]
                if "executed on" in line and not execution_date:
                    line = line[:-1].split(" ")
                    date = [v for v in line if "." in v][0]
                    time = [v for v in line if ":" in v][0]
                    execution_date = "_".join([date, time])
        return {"vasp_version": vasp_version, "execution_date": execution_date}

    @property
    def metadata(self):
        """returns compiled metadata

        Returns:
            {'incar' : incar.as_dict(),
             'all_input_settings' : all_input_settings.as_dict(),
             'kpoints' : kpoints.as_dict(),
             'potcar' : potcar.as_dict(),
             'calc_dir' : calc_dir,}
        """
        outputs = self.outputs
        meta = {}
        incar_data = outputs.incar
        if not incar_data:
            meta["incar"] = {}
        else:
            meta["incar"] = (
                incar_data if isinstance(incar_data, dict) else incar_data.as_dict()
            )

        kpoints_data = outputs.kpoints
        if not kpoints_data:
            meta["kpoints"] = {}
        else:
            meta["kpoints"] = (
                kpoints_data
                if isinstance(kpoints_data, dict)
                else kpoints_data.as_dict()
            )

        potcar_data = outputs.potcar
        if not potcar_data:
            meta["potcar"] = {}
        else:
            meta["potcar"] = potcar_data

        all_input_settings = outputs.all_input_settings
        meta["all_input_settings"] = (
            all_input_settings
            if isinstance(all_input_settings, dict)
            else all_input_settings.as_dict()
        )

        meta["calc_dir"] = self.calc_dir

        meta["run_stats"] = self.run_stats

        return meta

    @property
    def calc_setup(self):
        """

        Returns:
            self explanatory?
        """
        calc_dir = self.calc_dir
        project_dir, scripts, formula, ID, mag, xc_calc = calc_dir.split("/")[-6:]
        return {
            "formula_indicator": formula,
            "struc_indicator": ID,
            "mag": mag,
            "xc": xc_calc.split("-")[0],
            "calc": xc_calc.split("-")[1],
            "project": project_dir,
        }

    @property
    def computed_structure_entry(self):
        """
        Returns:
            ComputedStructureEntry
        """
        calc_setup = self.calc_setup
        vr = self.outputs.vasprun
        if not vr:
            return None
        entry = vr.get_computed_entry()
        for key in calc_setup:
            entry.data[key] = calc_setup[key]
        entry.data["material_id"] = "--".join(
            [
                entry.data["formula_indicator"],
                entry.data["struc_indicator"],
                entry.data["mag"],
                entry.data["xc"],
                entry.data["calc"],
                entry.data["project"],
            ]
        )
        entry.data["formula"] = CompTools(entry.composition.reduced_formula).clean
        entry.data["queried"] = False
        return entry.as_dict()

    def summary(
        self,
        include_meta=False,
        include_calc_setup=False,
        include_structure=False,
        include_trajectory=False,
        include_mag=False,
        include_tdos=False,
        include_pdos=False,
        include_gap=True,
        include_charge=True,
        include_madelung=True,
        include_tcohp=False,
        include_pcohp=False,
        include_tcoop=False,
        include_pcoop=False,
        include_tcobi=False,
        include_pcobi=False,
        include_entry=False,
    ):
        """
        Returns all desired data for post-processing DFT calculations

        Args:
            include_meta (bool, optional): _description_. Defaults to False.
            include_calc_setup (bool, optional): _description_. Defaults to False.
            include_structure (bool, optional): _description_. Defaults to False.
            include_mag (bool, optional): _description_. Defaults to False.
            include_dos (bool, optional): _description_. Defaults to False.
            include_gap (bool, optional): _description_. Defaults to True.

        Raises:
            NotImplementedError: _description_

        Returns:
            _type_: _description_
        """
        data = {}
        data["results"] = self.basic_info

        convergence = data["results"]["convergence"]

        if include_meta:
            data["meta"] = self.metadata
        if include_calc_setup:
            if "meta" not in data:
                data["meta"] = {}
            data["meta"]["setup"] = self.calc_setup
        if include_structure:
            # data["structure"] = self.decorated_structure(
            #    charge_source="bader", structure=None
            # )
            data["structure"] = self.relaxed_structure
        if include_trajectory:
            if convergence:
                data["trajectory"] = self.compact_trajectory
            else:
                data["trajectory"] = None
        if include_mag:
            if convergence:
                data["magnetization"] = self.magnetization
            else:
                data["magnetization"] = None
        if include_tdos:
            if convergence:
                pdos = self.pdos()
                tdos = self.tdos(pdos=pdos)
                data["tdos"] = tdos
            else:
                data["tdos"] = None
        if include_pdos:
            if convergence:
                pdos = self.pdos()
                data["pdos"] = pdos
            else:
                data["pdos"] = None
        if include_gap:
            if convergence:
                data["gap"] = self.gap_properties
            else:
                data["gap"] = None
        if include_charge:
            data["charge"] = {}

            for source in ["bader", "mulliken", "lowdin"]:
                if convergence:
                    data["charge"][source] = self.charge(source)
                else:
                    data["charge"][source] = None
        if include_madelung:
            if convergence:
                data["madelung"] = self.E_madelung
            else:
                data["madelung"] = None
        if include_tcohp:
            if convergence:
                pcohp = self.pcohp()
                tcohp = self.tcohp(pcohp=pcohp)
                data["tcohp"] = tcohp
            else:
                data["tcohp"] = None
        if include_pcohp:
            if convergence:
                pcohp = self.pcohp()
                data["pcohp"] = pcohp
            else:
                data["pcohp"] = None
        if include_tcoop:
            if convergence:
                data["tcoop"] = self.tcohp(are_coops=True)
            else:
                data["tcoop"] = None
        if include_pcoop:
            if convergence:
                data["pcoop"] = self.pcohp(are_coops=True)
            else:
                data["pcoop"] = None
        if include_tcobi:
            if convergence:
                data["tcobi"] = self.tcohp(are_cobis=True)
            else:
                data["tcobi"] = None
        if include_pcobi:
            if convergence:
                data["pcobi"] = self.pcohp(are_cobis=True)
            else:
                data["pcobi"] = None
        if include_entry:
            data["entry"] = self.computed_structure_entry

        return data


class AnalyzeBatch(object):
    def __init__(self, launch_dirs, user_configs={}):
        """
        Args:
            launch_dirs (dict) : {launch directory : {'xcs' : [final_xcs for each chain of jobs], 'magmom' : [list of magmoms for that launch directory]}}
            user_configs (dict) : any configs relevant to analysis
                only_static: True # only retrieve data from the static calculations
                one_xc: if None, retrieve all xcs, else retrieve only the one specified
                check_relax: True # make sure the relax calculation and the static have similar energies
                include_meta: False # include metadata like INCAR, KPOINTS, POTCAR settings
                include_calc_setup: True # include things related to the calculation setup -- standard, mag, final_xc, etc
                include_structure: True # include the relaxed crystal structure as a dict
                include_mag: False # include the relaxed magnetization info as as dict
                include_dos: False # include the density of states
                verbose: True # print stuff as things get analyzed
                n_procs: all # how many cores to parallelize the analysis over
            analysis_configs_yaml (str) : path to yaml file with baseline analysis configs
            refresh_configs (bool): if True, will refresh the local baseline analysis configs with the pydmc version
        """

        # should get these from LaunchTools
        self.launch_dirs = launch_dirs

        # write baseline analysis configs locally if not there or want to be refreshed
        _base_configs = load_base_configs()

        # update configs with any user_configs
        configs = {**_base_configs, **user_configs}

        # copy configs to prevent unwanted changes
        self.configs = configs.copy()

    @property
    def calc_dirs(self):
        """
        Returns:
            a list of all calculation directories to crawl through and collect VASP output info
        """
        launch_dirs = self.launch_dirs
        all_calc_dirs = []
        for launch_dir in launch_dirs:
            stuff_in_launch_dir = os.listdir(launch_dir)
            relevant_stuff = [
                c
                for c in stuff_in_launch_dir
                if "-" in c
                if any(xc in c for xc in ["gga", "metagga", "hse"])
            ]
            calc_dirs = [os.path.join(launch_dir, c) for c in relevant_stuff]
            calc_dirs = [
                c for c in calc_dirs if os.path.exists(os.path.join(c, "POSCAR"))
            ]
            all_calc_dirs += calc_dirs
        return sorted(list(set(all_calc_dirs)))

    def _key_for_calc_dir(self, calc_dir):
        """
        Args:
            calc_dir (str) : path to a calculation directory where VASP was executed

        Returns:
            a string that can be used as a key for a dictionary
                top_level.unique_ID.standard.mag.xc_calc
        """
        return "--".join(calc_dir.split("/")[-4:])

    @property
    def results(self):
        """

        Returns:
            {calc_dir key : results for that calc_dir}
        """

        configs = self.configs.copy()

        calc_dirs = self.calc_dirs

        only_xc = configs["only_xc"]
        only_calc = configs["only_calc"]

        if only_xc:
            calc_dirs = [
                c
                for c in calc_dirs
                if self._key_for_calc_dir(c).split("--")[-1].split("-")[0] == only_xc
            ]

        if only_calc:
            calc_dirs = [
                c
                for c in calc_dirs
                if self._key_for_calc_dir(c).split("--")[-1].split("-")[1] == only_calc
            ]

        data = [_results_for_calc_dir(calc_dir, configs) for calc_dir in calc_dirs]

        # each item in data is a dictionary that looks like {key : data for that key}
        # we'll map to a dictionary of {key : data for that key} for all keys
        out = {}
        for d in data:
            for key in d:
                out[key] = d[key]

        return out


def _results_for_calc_dir(calc_dir, configs):
    """
    Args:
        calc_dir (str) : path to a calculation directory where VASP was executed

    Returns:
        a dictionary of results for that calculation directory
            - format varies based on self.configs
            - see AnalyzeVASP.summary() for more info
    """
    key = "--".join(calc_dir.split("/")[-4:])
    fjson = os.path.join(calc_dir, "results.json")
    if os.path.exists(fjson) and not configs["remake_results"]:
        result = read_json(fjson)
        return {key: result}

    xc_calc = key.split("--")[-1]
    xc, calc = xc_calc.split("-")

    configs = configs.copy()

    if calc != "relax":
        configs["include_trajectory"] = False
    if calc not in ["lobster", "bs"]:
        configs["include_tcohp"] = False
        configs["include_pcohp"] = False
        configs["include_tcoop"] = False
        configs["include_pcoop"] = False
        configs["include_tcobi"] = False
        configs["include_pcobi"] = False
        configs["include_tdos"] = False
        configs["include_pdos"] = False
    if calc != "static":
        configs["include_mag"] = False
        configs["include_entry"] = False
        configs["include_structure"] = False

    verbose = configs["verbose"]
    include_meta = configs["include_metadata"]
    include_calc_setup = configs["include_calc_setup"]
    include_structure = configs["include_structure"]
    include_trajectory = configs["include_trajectory"]
    include_mag = configs["include_mag"]
    include_tdos = configs["include_tdos"]
    include_pdos = configs["include_pdos"]
    include_tcohp = configs["include_tcohp"]
    include_pcohp = configs["include_pcohp"]
    include_tcoop = configs["include_tcoop"]
    include_pcoop = configs["include_pcoop"]
    include_tcobi = configs["include_tcobi"]
    include_pcobi = configs["include_pcobi"]
    include_entry = configs["include_entry"]
    check_relax = configs["check_relax_energy"]
    create_cif = configs["create_cif"]

    if verbose:
        print("analyzing %s" % calc_dir)
    analyzer = AnalyzeVASP(calc_dir)

    # collect the data we asked for
    summary = analyzer.summary(
        include_meta=include_meta,
        include_calc_setup=include_calc_setup,
        include_structure=include_structure,
        include_trajectory=include_trajectory,
        include_mag=include_mag,
        include_tdos=include_tdos,
        include_pdos=include_pdos,
        include_tcohp=include_tcohp,
        include_pcohp=include_pcohp,
        include_tcoop=include_tcoop,
        include_pcoop=include_pcoop,
        include_tcobi=include_tcobi,
        include_pcobi=include_pcobi,
        include_entry=include_entry,
    )

    # store the relax energy if we asked to
    if check_relax:
        relax_energy = AnalyzeVASP(calc_dir.replace(calc, "relax")).E_per_at
        summary["results"]["E_relax"] = relax_energy

    if create_cif and summary["results"]["convergence"] and include_structure:
        if summary["structure"]:
            s = StrucTools(summary["structure"]).structure
            s.to(fmt="cif", filename=os.path.join(calc_dir, key + ".cif"))
        else:
            print("no structure, cant make cif")
    return {key: summary}

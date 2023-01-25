import os
import numpy as np
import multiprocessing as multip

from pymatgen.io.vasp.outputs import Vasprun, Outcar, Eigenval
from pymatgen.core.structure import Structure
from pymatgen.io.vasp.inputs import Kpoints, Incar
from pymatgen.io.lobster.outputs import Doscar, Cohpcar, Charge, MadelungEnergies

from pydmc.core.struc import StrucTools, SiteTools
from pydmc.utils.handy import read_json, write_json, read_yaml, write_yaml
from pydmc.data.configs import load_batch_vasp_analysis_configs


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
            flobster = os.path.join(self.calc_dir, "COOPCAR")
        elif are_cobis:
            flobster = os.path.join(self.calc_dir, "COBI?")
        else:
            flobster = os.path.join(self.calc_dir, "COHPCAR")

        if not os.path.exists(flobster):
            return None

        try:
            cohp = Cohpcar(are_coops=are_coops, are_cobis=are_cobis)
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
        if vr:
            if self.calc == "static":
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
            return vr.final_energy / self.nsites
        else:
            return None

    @property
    def nsites(self):
        """
        Returns number of sites in POSCAR
        """
        return len(self.outputs.poscar)

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
    def formula(self):
        """
        Returns formula of structure from POSCAR (full)
        """
        return self.outputs.poscar.formula

    @property
    def compact_formula(self):
        """
        Returns formula of structure from POSCAR (compact)
            - use this w/ E_per_at
        """
        return StrucTools(self.outputs.poscar).compact_formula

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
                        {'mag' : total magnetization on site}}}
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
                idx: {"mag": mag[idx]["tot"]}
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

    def pdos(self, fjson=None, remake=False):
        """
        @TODO: add demo/test

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

        if not fjson:
            fjson = os.path.join(self.calc_dir, "pdos.json")
        if os.path.exists(fjson) and not remake:
            return read_json(fjson)

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
                out[el] = {site_idx: {}}
            else:
                out[el][site_idx] = {}
            for orbital in orbitals:
                out[el][site_idx][orbital] = {}
                dos = complete_dos.get_site_orbital_dos(site, orbital).as_dict()
                energies = dos["energies"]
                for spin in dos["densities"]:
                    out[el][site_idx][orbital][spin] = dos["densities"][spin]
        out["E"] = energies
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
            - also has keys "up" and "down" which separate everything by spin up or spin down

        @TODO: add demo/test
        @TODO: explore for magnetic materials
        @TODO: add options for summing or not summing spins
        @TODO: work on generic plotting
        @TODO: work on COHPCAR/COOPCAR
        """
        if not fjson:
            fjson = os.path.join(self.calc_dir, "tdos.json")
        if os.path.exists(fjson) and not remake:
            return read_json(fjson)
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
                out[spin]["total"] += np.array
        out["E"] = energies
        for k in out:
            out[k] = list(out[k])
        return write_json(out, fjson)

    def pcohp(self, fjson=None, remake=False, are_coops=False, are_cobis=False):
        """_summary_

        Args:
            fjson (_type_, optional): _description_. Defaults to None.
            remake (bool, optional): _description_. Defaults to False.

        Raises:
            NotImplementedError: _description_

        Returns:
            _type_: _description_
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
            return read_json(fjson)

        cohpcar = self.outputs.cohpcar(are_coops=are_coops, are_cobis=are_cobis)
        if not cohpcar:
            return None

        energies = cohpcar.energies

        cohp = cohpcar.cohp_data

        sites_to_els = self.sites_to_els

        out = {}
        for bond_idx in cohp:
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
                "cohp": {"1": {}, "-1": {}},
                "icohp": {"-1": {}, "1": {}},
                "length": bond_length,
            }
            for spin in cohp[bond_idx]["COHP"]:
                if spin.name == "up":
                    spin_tag = "1"
                else:
                    spin_tag = "-1"
                out[el_tag][site_tag]["cohp"][spin_tag] = cohp[bond_idx]["COHP"][spin]
                out[el_tag][site_tag]["icohp"][spin_tag] = cohp[bond_idx]["ICOHP"][spin]

        out["E"] = energies

        return write_json(out, fjson)

    def tcohp(
        self, pcohp=None, fjson=None, remake=False, are_coops=False, are_cobis=False
    ):
        """ """

        if are_coops:
            savename = "tcoop.json"
        elif are_cobis:
            savename = "tcobi.json"
        else:
            savename = "tcohp.json"

        if not fjson:
            fjson = os.path.join(self.calc_dir, savename)

        if os.path.exists(fjson) and not remake:
            return read_json(fjson)
        if not pcohp:
            pcohp = self.pcohp(are_coops=are_coops, are_cobis=are_cobis)
        if not pcohp:
            return None

        out = {"E": pcohp["E"]}
        for el_tag in pcohp:
            if el_tag == "E":
                continue
            out[el_tag] = {}
            out[el_tag]["cohp"] = np.zeros(len(out["E"]))
            out[el_tag]["icohp"] = np.zeros(len(out["E"]))
            for site_tag in pcohp[el_tag]:
                for spin in pcohp[el_tag][site_tag]["cohp"]:
                    out[el_tag]["cohp"] += np.array(
                        pcohp[el_tag][site_tag]["cohp"][spin]
                    )
                    out[el_tag]["icohp"] += np.array(
                        pcohp[el_tag][site_tag]["cohp"][spin]
                    )

        return write_json(out, fjson)

    def charge(self, source="bader", fjson=None, remake=False):
        """

        for obtaining partial charges resulting from dft calculations

        Args:
            fjson (_type_, optional): _description_. Defaults to None.
            remake (bool, optional): _description_. Defaults to False.

        Raises:
            NotImplementedError: _description_

        Returns:
            {el (str) : {site index (int) : {charge (float)}}}
        """
        if not fjson:
            fjson = os.path.join(self.calc_dir, "tdos.json")
        if os.path.exists(fjson) and not remake:
            return read_json(fjson)

        if source == "bader":
            fcharge = os.path.join(self.calc_dir, "ACF.dat")
            if not os.path.exists(fcharge):
                return None
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
                    if count > 3 + nsites:
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
                out[el][idx] = data[idx]["charge"]

        elif source in ["mulliken", "lowdin"]:
            chg = Charge(os.path.join(self.calc_dir, "CHARGE.lobster"))
            sites_to_els = self.sites_to_els
            charges = chg.Mulliken if source == "mulliken" else chg.Loewdin
            out = {el: {} for el in list(set(sites_to_els.values()))}
            for idx in sites_to_els:
                el = sites_to_els[idx]
                out[el][idx] = charges[idx]

        return write_json(out, fjson)

    def charged_structure(self, source, structure=False):
        """
        decorates structure with charges calculated from bader, mulliken, lowdin, etc

        Args:
            source (_type_): _description_
            structure (bool, optional): _description_. Defaults to False.

        Returns:
            _type_: _description_
        """
        if not structure:
            structure = self.outputs.contcar
        charges = self.charge(source=source)
        charges_by_site = {
            idx: charge for el in charges for idx, charge in charges[el].items()
        }
        sorted_charges = [charges_by_site[idx] for idx in range(len(structure))]
        structure.add_site_property(source, sorted_charges)
        return structure

    def magnetic_structure(self, structure=False):
        """
        decorates structures with optimized magnetic moments

        Args:
            structure (bool, optional): _description_. Defaults to False.

        Raises:
            NotImplementedError: _description_

        Returns:
            _type_: _description_
        """
        if not structure:
            structure = self.outputs.contcar
        mags = self.magnetization
        mags_by_site = {idx: mag for el in mags for idx, mag in mags[el].items()}
        sorted_charges = [mags_by_site[idx] for idx in range(len(structure))]
        structure.add_site_property("final_magmom", sorted_charges)
        return structure

    def E_madelung(self, source):
        fmadelung = os.path.join(self.calc_dir, "MadelungEnergies.lobster")
        return {
            "mulliken": MadelungEnergies(filename=fmadelung).madelungenergies_Mulliken,
            "lowdin": MadelungEnergies(filename=fmadelung).madelungenergies_Loewdin,
        }

    @property
    def basic_info(self):
        """
        Returns:
            {'convergence' : True if calc converged else False,
            'E_per_at' : energy per atom if calc converged else None}
        """
        E_per_at = self.E_per_at
        convergence = True if E_per_at else False
        return {"convergence": convergence, "E_per_at": E_per_at}

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

        return meta

    @property
    def calc_setup(self):
        """

        Returns:
            self explanatory?
        """
        calc_dir = self.calc_dir
        formula, ID, standard, mag, xc_calc = calc_dir.split("/")[-5:]
        return {
            "formula": formula,
            "ID": ID,
            "standard": standard,
            "mag": mag,
            "xc": xc_calc.split("-")[0],
            "calc": xc_calc.split("-")[1],
        }

    def summary(
        self,
        include_meta=False,
        include_calc_setup=False,
        include_structure=False,
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
        if include_meta:
            data["meta"] = self.metadata
        if include_calc_setup:
            if "meta" not in data:
                data["meta"] = {}
            data["meta"]["setup"] = self.calc_setup
        if include_structure:
            data["structure"] = self.relaxed_structure
        if include_mag:
            data["magnetization"] = self.magnetization
        if include_tdos:
            data["tdos"] = self.tdos()
        if include_pdos:
            data["pdos"] = self.pdos()
        if include_gap:
            data["gap"] = self.gap_properties
        if include_charge:
            for source in ["bader", "mulliken", "lowdin"]:
                data["charge"] = {source: self.charge(source)}
        if include_madelung:
            data["madelung"] = self.E_madelung
        if include_tcohp:
            data["tcohp"] = self.tcohp()
        if include_pcohp:
            data["pcohp"] = self.pcohp()
        if include_tcoop:
            data["tcoop"] = self.tcohp(are_coops=True)
        if include_pcoop:
            data["pcoop"] = self.pcohp(are_coops=True)
        if include_tcobi:
            data["tcobi"] = self.tcohp(are_cobis=True)
        if include_pcobi:
            data["pcobi"] = self.pcohp(are_cobis=True)
        return data


class AnalyzeBatch(object):
    def __init__(
        self,
        launch_dirs,
        user_configs={},
        analysis_configs_yaml=os.path.join(
            os.getcwd(), "_batch_VASP_analysis_configs.yaml"
        ),
        refresh_configs=True,
    ):
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
        if not os.path.exists(analysis_configs_yaml) or refresh_configs:
            _analysis_configs = load_batch_vasp_analysis_configs()
            write_yaml(_analysis_configs, analysis_configs_yaml)

        _analysis_configs = read_yaml(analysis_configs_yaml)

        # update configs with any user_configs
        configs = {**_analysis_configs, **user_configs}

        # figure out how many processors to use
        if configs["n_procs"] == "all":
            configs["n_procs"] = multip.cpu_count() - 1

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
        calcs = (
            ["loose", "relax", "static"]
            if not self.configs["only_static"]
            else ["static"]
        )
        for launch_dir in launch_dirs:
            files_in_launch_dir = os.listdir(launch_dir)
            calc_dirs = [
                os.path.join(launch_dir, c)
                for c in files_in_launch_dir
                if "-" in c
                if c.split("-")[1] in calcs
            ]
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
        return ".".join(calc_dir.split("/")[-5:])

    def _results_for_calc_dir(self, calc_dir):
        """
        Args:
            calc_dir (str) : path to a calculation directory where VASP was executed

        Returns:
            a dictionary of results for that calculation directory
                - format varies based on self.configs
                - see AnalyzeVASP.summary() for more info
        """

        configs = self.configs.copy()
        verbose = configs["verbose"]
        include_meta = configs["include_meta"]
        include_calc_setup = configs["include_calc_setup"]
        include_structure = configs["include_structure"]
        include_mag = configs["include_mag"]
        include_tdos = configs["include_tdos"]
        check_relax = configs["check_relax"]
        create_cif = configs["create_cif"]

        if verbose:
            print("analyzing %s" % calc_dir)
        analyzer = AnalyzeVASP(calc_dir)

        # collect the data we asked for
        summary = analyzer.summary(
            include_meta=include_meta,
            include_calc_setup=include_calc_setup,
            include_structure=include_structure,
            include_mag=include_mag,
            include_tdos=include_tdos,
        )

        # store the relax energy if we asked to
        if check_relax:
            relax_energy = AnalyzeVASP(calc_dir.replace("static", "relax")).E_per_at
            summary["meta"]["E_relax"] = relax_energy
            if not relax_energy:
                summary["results"]["convergence"] = False
                summary["results"]["E_per_at"] = None

        # save the data in a dictionary with a key for that calc_dir
        key = self._key_for_calc_dir(calc_dir)

        if create_cif and summary["results"]["convergence"] and include_structure:
            s = Structure.from_dict(summary["structure"])
            s.to(fmt="cif", filename=os.path.join(calc_dir, key + ".cif"))
        return {key: summary}

    @property
    def results(self):
        """

        Returns:
            {calc_dir key : results for that calc_dir}
        """

        n_procs = self.configs["n_procs"]

        calc_dirs = self.calc_dirs

        # run serial if only one processor
        if n_procs == 1:
            data = [self._results_for_calc_dir(calc_dir) for calc_dir in calc_dirs]

        # otherwise, run parallel
        if n_procs > 1:
            pool = multip.Pool(processes=n_procs)
            data = pool.map(self._results_for_calc_dir, calc_dirs)
            pool.close()

        # each item in data is a dictionary that looks like {key : data for that key}
        # we'll map to a dictionary of {key : data for that key} for all keys
        out = {}
        for d in data:
            for key in d:
                out[key] = d[key]

        return out

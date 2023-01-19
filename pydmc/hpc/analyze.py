import os
import numpy as np
import multiprocessing as multip

from pymatgen.io.vasp.outputs import Vasprun, Outcar, Eigenval
from pymatgen.core.structure import Structure
from pymatgen.io.vasp.inputs import Kpoints, Incar
from pymatgen.io.lobster.inputs import Lobsterin
from pymatgen.io.lobster.outputs import Doscar

from pydmc.core.struc import StrucTools, SiteTools
from pydmc.core.comp import CompTools
from pydmc.utils.handy import read_json, write_json, read_yaml, write_yaml
from pydmc.data.configs import load_batch_vasp_analysis_configs


class VASPOutputs(object):
    def __init__(self, calc_dir):

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
        Returns dict of VASP input settings from vasprun.xml
        """
        fincar = os.path.join(self.calc_dir, "INCAR")
        if os.path.exists(fincar):
            return Incar.from_file(os.path.join(self.calc_dir, "INCAR"))
        else:
            return {}

    @property
    def all_input_settings(self):
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
        Returns list of POTCAR symbols from vasprun.xml
        """
        vr = self.vasprun
        if vr:
            return vr.potcar_symbols
        else:
            return None

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

    @property
    def lobsterin(self):
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
        if calc == "from_calc_dir":
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
        @TODO: explore for magnetic materials
        @TODO: add options for summing or not summing spins
        @TODO: work on generic plotting
        @TODO: work on COHPCAR/COOPCAR


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
        out = {}
        energies = pdos["E"]
        for el in pdos:
            if el == "E":
                continue
            out[el] = np.zeros(len(energies))
            for site in pdos[el]:
                for orb in pdos[el][site]:
                    for spin in pdos[el][site][orb]:
                        out[el] += np.array(pdos[el][site][orb][spin])
        out["total"] = np.zeros(len(energies))
        for el in out:
            if el == "total":
                continue
            out["total"] += np.array(out[el])

        out["E"] = energies
        for k in out:
            out[k] = list(out[k])
        return write_json(out, fjson)

    @property
    def basic_info(self):
        E_per_at = self.E_per_at
        convergence = True if E_per_at else False
        return {"convergence": convergence, "E_per_at": E_per_at}

    @property
    def relaxed_structure(self):
        structure = self.outputs.contcar
        if structure:
            return structure.as_dict()
        else:
            return None

    @property
    def metadata(self):
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
            meta["potcar"] = (
                potcar_data if isinstance(potcar_data, dict) else potcar_data.as_dict()
            )

        input_settings_data = outputs.input_settings
        meta["input_settings"] = (
            input_settings_data
            if isinstance(input_settings_data, dict)
            else input_settings_data.as_dict()
        )

        meta["calc_dir"] = self.calc_dir

        return meta

    @property
    def calc_setup(self):
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
        include_dos=False,
        include_gap=True,
    ):
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
        if include_dos:
            raise NotImplementedError("still working on DOS processing")
        if include_gap:
            data["gap"] = self.gap_properties
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
        include_dos = configs["include_dos"]
        check_relax = configs["check_relax"]

        if verbose:
            print("analyzing %s" % calc_dir)
        analyzer = AnalyzeVASP(calc_dir)

        # collect the data we asked for
        summary = analyzer.summary(
            include_meta=include_meta,
            include_calc_setup=include_calc_setup,
            include_structure=include_structure,
            include_mag=include_mag,
            include_dos=include_dos,
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

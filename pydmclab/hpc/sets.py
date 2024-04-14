from pymatgen.io.vasp import Kpoints
from pymatgen.io.vasp.sets import MPRelaxSet, MPScanRelaxSet, MPHSERelaxSet

from pydmclab.core.struc import StrucTools


class GetSet(object):
    """
    This is how we're going to determine the VASP input files given the
        - xc (gga, metagga, etc)
        - calc (loose, relax, static, lobster, etc)
        - user specified modifications (configs, modify_incar, modify_kpoints, modify_potcar)

    """

    def __init__(
        self,
        structure,
        configs,
        potcar_functional=None,
        validate_magmom=False,
        modify_incar={},
        modify_kpoints={},
        modify_potcar={},
    ):
        """
        Args:
            structure (Structure): pymatgen structure object
            configs (dict):
                - xc_to_run (str): xc to run (gga, ggau, metagga, metaggau, hse06)
                - calc_to_run (str): calculation to run (loose, relax, static, lobster, etc)
                - standard (str): standard (mp, dmc)
                - mag (str): magnetic configuration (fm, nm, afm_*)
                - functional (str): functional to use (PBE, PBE_54, etc)
            potcar_functional (str): functional to use for POTCAR (PBE, PBE_54, etc)
            validate_magmom (bool): validate magnetic moments
            modify_incar (dict): user specified INCAR settings
            modify_kpoints (dict): user specified KPOINTS settings
            modify_potcar (dict): user specified POTCAR settings
        """
        standard = configs["standard"]
        mag = configs["mag"]

        self.xc, self.calc = configs["xc_to_run"], configs["calc_to_run"]

        self.structure = StrucTools(structure).structure
        self.standard = standard
        self.mag = mag

        self.configs = configs

        self.modify_incar = modify_incar.copy()
        self.modify_kpoints = modify_kpoints.copy()
        self.modify_potcar = modify_potcar.copy()

        self.potcar_functional = potcar_functional
        self.validate_magmom = validate_magmom

    @property
    def base_set(self):
        """
        Returns VaspSet (ie which MP set do we want to customize from)
        """
        xc, calc = self.xc, self.calc

        if xc in ["gga", "ggau"]:
            # start from MP relax for GGA or GGA+U
            return MPRelaxSet
        elif xc in ["metagga", "metaggau"]:
            # start from MP Scan for metaGGA or metaGGA+U
            return MPScanRelaxSet
        elif xc in ["hse06"]:
            # start fom MP HSE for HSE06
            return MPHSERelaxSet
        else:
            raise NotImplementedError(f"xc: {xc} not implemented")

    @property
    def user_incar_settings(self):
        """
        These are changes we want to make to a given base VaspSet

        Returns:
            {incar setting (str) : value for that setting (str, float, int, bool)}
        """
        # these are the mods we indicated in our configs
        user_passed_settings = self.modify_incar

        # need to know the kpoints mods to inform setting of KSPACING
        user_passed_kpoints_settings = self.modify_kpoints

        xc, calc, standard, mag = self.xc, self.calc, self.standard, self.mag

        new_settings = {}

        # turn magnetism on or off
        if mag == "nm":
            new_settings["ISPIN"] = 1
            new_settings["MAGMOM"] = None
        else:
            new_settings["ISPIN"] = 2
            new_settings["LORBIT"] = 11

        # set parallelization if not set explicitly
        if (
            ("NPAR" not in user_passed_settings)
            and ("NCORE" not in user_passed_settings)
            and ("KPAR" not in user_passed_settings)
        ):
            new_settings["NCORE"] = 4

        # work on the calc type

        # loose --> choose light settings
        if calc == "loose":
            new_settings["ENCUT"] = 400
            new_settings["ENAUG"] = 800
            new_settings["ISIF"] = 3
            new_settings["EDIFF"] = 1e-5
            new_settings["NELM"] = 40

        # relax --> need NSW
        elif calc == "relax":
            new_settings["NSW"] = 199

        # these three calcs are static --> turn off relaxation things
        elif calc in [
            "static",
            "lobster",
            "parchg",
        ]:
            new_settings["NSW"] = 0
            new_settings["ISIF"] = None
            new_settings["IBRION"] = None
            new_settings["POTIM"] = None
            new_settings["LCHARG"] = True
            new_settings["LORBIT"] = 0
            new_settings["LVHAR"] = True
            new_settings["ICHARG"] = 0
            new_settings["LAECHG"] = True

        # for DFPT --> set explicit requirements
        elif calc == "dfpt":
            new_settings["IBRION"] = 7
            new_settings["ISYM"] = 2
            new_settings["ALGO"] = "Normal"
            new_settings["ADDGRID"] = True
            new_settings["IALGO"] = 38
            new_settings["NPAR"] = None
            new_settings["NCORE"] = None
            new_settings["NSW"] = 1

        # for finite displacements --> set explicit requirements
        elif calc == "finite_displacements":
            new_settings["IBRION"] = 2
            new_settings["ENCUT"] = 700
            new_settings["EDIFF"] = 1e-7
            new_settings["LAECHG"] = False
            new_settings["LREAL"] = False
            new_settings["ALGO"] = "Normal"
            new_settings["NSW"] = 0
            new_settings["LCHARG"] = False

        # for dielectric --> set explicit requirements
        elif calc == "dielectric":
            new_settings["LVTOT"] = True
            new_settings["LEPSILON"] = True
            new_settings["LOPTICS"] = True
            new_settings["IBRION"] = 8

        # make sure we have a WAVECAR to pass from relax --> static and from static --> other stuff (like PARCHG)
        elif calc in ["relax", "static"]:
            new_settings["LWAVE"] = True

        # for PARCHG --> set explicit requirements
        elif calc == "parchg":
            new_settings["ISTART"] = 1
            new_settings["LPARD"] = True
            new_settings["LSEPB"] = False
            new_settings["LSEPK"] = False
            new_settings["LWAVE"] = False
            new_settings["NBMOD"] = -3
            if "EINT" not in user_passed_settings:
                print("WARNING: PARCH analysis but no EINT set. Setting to Ef - 2 eV")
                new_settings["EINT"] = "".join([str(v) for v in [-2.0, 0]])

        # for LOBSTER --> set explicit requirements
        elif calc == "lobster":
            new_settings["ISTART"] = 0
            new_settings["LAECHG"] = True
            new_settings["ISYM"] = -1
            new_settings["KSPACING"] = None
            new_settings["ISMEAR"] = 0
            new_settings["NSW"] = 0
            new_settings["LWAVE"] = True

        # for HSE06 single point (static) --> set explicit requirements
        elif calc == "sphse06":
            calc_settings = {
                "NSW": 0,
                "ALGO": "Normal",
                "GGA": "PE",
                "HFSCREEN": 0.2,
                "LHFCALC": True,
                "PRECFOCK": "Fast",
                "ISMEAR": -5,
                "LORBIT": 11,
                "LCHARG": True,
                "LASPH": True,
                "LREAL": False,
                "LDAU": False,
            }
            for key in calc_settings:
                new_settings[key] = calc_settings[key]

        # now we'll customize based on a given standard

        # dmc is the only one implemented other than MP. for MP, we leave alone
        if standard == "dmc":
            dmc_options = {
                "EDIFF": 1e-6,
                "EDIFFG": -0.03,
                "ISYM": 0,
                "LREAL": False,
                "KSPACING": 0.22,
                "ENCUT": 520,
                "ENAUG": 1040,
                "ISMEAR": 0,
                "SIGMA": 0.05,
            }
            for key in dmc_options:
                if key not in new_settings:
                    new_settings[key] = dmc_options[key]

        # now set our functional given our xc
        if xc in ["metagga", "metaggau"]:
            new_settings["GGA"] = None
            functional = self.configs["functional"]
            # r2SCAN is default
            if not functional:
                new_settings["METAGGA"] = "R2SCAN"
            else:
                new_settings["METAGGA"] = functional

        if xc in ["gga", "ggau"]:
            functional = self.configs["functional"]
            # PBE is default
            if not functional:
                new_settings["GGA"] = "PE"
            else:
                new_settings["GGA"] = functional

        if xc == "gga":
            # turn off +U b/c our base set wants to use it
            if standard != "mp":
                new_settings["LDAU"] = False
        elif xc in ["metagga", "hse06"]:
            new_settings["LDAU"] = False
        elif xc in ["ggau", "metaggau"]:
            new_settings["LDAU"] = True

        # set +U related things; NOTE: assumes d electrons (ie doesn't work for f systems)
        if (xc in ["ggau", "metaggau"]) and (standard != "mp"):
            # note: need to pass U values as eg {'LDAUU' : {'Fe' : 5}}
            new_settings["LDAU"] = True
            new_settings["LDAUTYPE"] = 2
            LDAUU = user_passed_settings["LDAUU"]
            LDAUL = {el: 2 for el in LDAUU}
            LDAUJ = {el: 0 for el in LDAUU}
            new_settings["LDAUL"] = LDAUL
            new_settings["LDAUJ"] = LDAUJ

        # if we asked for a KPOINTS file (grid, auto, etc), turn off KSPACING
        if user_passed_kpoints_settings:
            new_settings["KSPACING"] = None

        # override the default settings w/ our user-passed settings (eg incar_mods). these always take precedence
        for k, v in user_passed_settings.items():
            new_settings[k] = v

        # delete KSPACING b/c pymatgen wants to check this at some point when writing INCAR and None causes problems
        if new_settings["KSPACING"] is None:
            del new_settings["KSPACING"]

        return new_settings.copy()

    @property
    def user_kpoints_settings(self):
        """
        Returns KPOINTS object based on user passed settings

            'reciprocal_density' = N --> # kpts * volume = N
            'density' = N --> # kpts * atoms = N
            'auto' = N --> Auto N
            'grid' = [N, N, N] --> N x N x N grid
        """
        user_passed_settings = self.modify_kpoints

        xc, calc, standard, mag = self.xc, self.calc, self.standard, self.mag

        new_settings = {}

        # need a KPOINTS file for lobster, so make sure we set something
        if calc == "lobster":
            # this is pymatgen lobster default, but seems pretty dense..
            new_settings["reciprocal_density"] = 100

        for k, v in user_passed_settings.items():
            new_settings[k] = v

        if "grid" in new_settings:
            return Kpoints.gamma_automatic(kpts=new_settings["grid"])
        elif "auto" in new_settings:
            return Kpoints.automatic(subdivisions=new_settings["auto"])
        elif "density" in new_settings:
            return Kpoints.automatic_density(
                structure=self.structure, kppa=new_settings["density"]
            )
        elif "reciprocal_density" in new_settings:
            return Kpoints.automatic_density_by_vol(
                structure=self.structure, kppvol=new_settings["reciprocal_density"]
            )

    @property
    def user_potcar_settings(self):
        """
        Returns:
            {element (str) : desired potcar (str)}
        """
        user_passed_settings = self.modify_potcar

        xc, calc, standard, mag = self.xc, self.calc, self.standard, self.mag

        new_settings = {}
        if standard == "dmc":
            new_settings["W"] = "W"

        for k, v in user_passed_settings.items():
            new_settings[k] = v
        return new_settings.copy()

    @property
    def vaspset(self):
        """
        Returns:
            VaspSet object
                accounts for base_set + xc_calc related mods + user-specified mods
        """
        potcar_functional = self.potcar_functional
        validate_magmom = self.validate_magmom
        if not validate_magmom:
            validate_magmom = False
        if not potcar_functional:
            potcar_functional = "PBE" if self.standard == "mp" else "PBE_54"

        print(self.user_incar_settings)
        return self.base_set(
            structure=self.structure,
            user_incar_settings=self.user_incar_settings,
            user_kpoints_settings=self.user_kpoints_settings,
            user_potcar_settings=self.user_potcar_settings,
            user_potcar_functional=potcar_functional,
            validate_magmom=validate_magmom,
        )

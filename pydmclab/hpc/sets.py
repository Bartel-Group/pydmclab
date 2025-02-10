from __future__ import annotations
from typing import TYPE_CHECKING

from pymatgen.io.vasp import Kpoints
from pymatgen.io.vasp.sets import MPRelaxSet, MPScanRelaxSet, MPHSERelaxSet

from pydmclab.core.struc import StrucTools


if TYPE_CHECKING:
    from pymatgen.core.structure import Structure
    from pymatgen.io.vasp.sets import VaspSet
    from pymatgen.io.vasp import Kpoints


class GetSet(object):
    """
    This is how we're going to determine the VASP input files given the
        xc (gga, metagga, etc)
        calc (loose, relax, static, lobster, etc)
        user specified modifications (configs, modify_incar, modify_kpoints, modify_potcar)

    """

    def __init__(
        self,
        structure: Structure,
        configs: dict,
        potcar_functional: str | None = None,
        validate_magmom: bool = False,
        modify_incar: dict = {},
        modify_kpoints: dict = {},
        modify_potcar: dict = {},
    ) -> None:
        """
        Args:
            structure (Structure):
                pymatgen structure object

            configs (dict):
                dictionary of configs (only some of these get retrieved in Sets)
                    xc_to_run (str):
                        xc to run (gga, ggau, metagga, metaggau, hse06)
                    calc_to_run (str):
                        calculation to run (loose, relax, static, lobster, etc)
                    standard (str):
                        standard (mp, dmc)
                    mag (str):
                        magnetic configuration (fm, nm, afm_*)
                    functional (str):
                        functional to use (PE, R2SCAN, etc)

            potcar_functional (str):
                functional to use for POTCAR (PBE, PBE_54, etc)
                    None uses defaults (PBE_54 for standard = 'dmc'; PBE for standard = 'mp')

            validate_magmom (bool):
                validate magnetic moments (I've found no reason to set this True)

            modify_incar (dict):
                user specified INCAR settings
                    {incar setting (str) : value for that setting (str, float, int, bool)}

            modify_kpoints (dict):
                user specified KPOINTS settings
                    {kpoints setting (str) : value for that setting (float, int, list)}
                        'reciprocal_density' = N --> # kpts * volume = N
                        'density' = N --> # kpts * atoms = N
                        'auto' = N --> Auto N
                        'grid' = [N, N, N] --> N x N x N grid

            modify_potcar (dict):
                user specified POTCAR settings
                    {element (str) : desired potcar (str)}
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
    def base_set(self) -> VaspSet:
        """
        Returns VaspSet (ie which MP set do we want to customize from)
        """
        xc = self.xc
        calc = self.calc

        if xc in ["gga", "ggau"]:
            # start from MP relax for GGA or GGA+U
            return MPRelaxSet
        elif xc in ["metagga", "metaggau"]:
            # start from MP Scan for metaGGA or metaGGA+U
            return MPScanRelaxSet
        elif xc in ["hse06"]:
            if calc in ["preggastatic"]:
                # start fom MP HSE for HSE06
                return MPRelaxSet
            else:
                # start fom MP HSE for HSE06
                return MPHSERelaxSet
        else:
            raise NotImplementedError(f"xc: {xc} not implemented")

    @property
    def user_incar_settings(self) -> dict:
        """
        These are changes we want to make to a given base VaspSet

        Starts from our user settings, then modifies based on our xc, calc, standard, and mag

        Returns:
            {incar setting (str) : value for that setting (str, float, int, bool)}
        """
        # these are the mods we indicated in our configs
        user_passed_settings = self.modify_incar

        # need to know the kpoints mods to inform setting of KSPACING
        user_passed_kpoints_settings = self.modify_kpoints

        xc, calc, standard, mag = self.xc, self.calc, self.standard, self.mag

        new_settings = {}

        # set parallelization if not set explicitly
        if (
            ("NPAR" not in user_passed_settings)
            and ("NCORE" not in user_passed_settings)
            and ("KPAR" not in user_passed_settings)
        ):
            new_settings["NCORE"] = 4

        # work on the calc type

        # loose --> choose light settings
        if "loose" in calc:
            new_settings["ENCUT"] = 400
            new_settings["ENAUG"] = 800
            new_settings["ISIF"] = 3
            new_settings["EDIFF"] = 1e-5
            new_settings["NELM"] = 40

        # relax --> need NSW
        if calc == "relax":
            new_settings["NSW"] = 199

        # these three calcs are static --> turn off relaxation things
        if calc in [
            "static",
            #    "prelobster",
            "lobster",
            "parchg",
            "polar",
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

        # if calc == "prelobster":
        #    new_settings["ISMEAR"] = -5
        #    new_settings["NELM"] = 0
        #    new_settings["NSW"] = 0

        # for DFPT --> set explicit requirements
        if calc == "dfpt":
            new_settings["IBRION"] = 7
            new_settings["ISYM"] = 2
            new_settings["ALGO"] = "Normal"
            new_settings["ADDGRID"] = True
            new_settings["IALGO"] = 38
            new_settings["NPAR"] = None
            new_settings["NCORE"] = None
            new_settings["NSW"] = 1

        # for finite displacements --> set explicit requirements
        if calc == "finite_displacements":
            new_settings["IBRION"] = 2
            new_settings["ENCUT"] = 700
            new_settings["EDIFF"] = 1e-7
            new_settings["LAECHG"] = False
            new_settings["LREAL"] = False
            new_settings["ALGO"] = "Normal"
            new_settings["NSW"] = 0
            new_settings["LCHARG"] = False

        # for dielectric --> set explicit requirements
        if calc == "dielectric":
            new_settings["LVTOT"] = True
            new_settings["LEPSILON"] = True
            new_settings["LOPTICS"] = True
            new_settings["IBRION"] = 8

        # make sure we have a WAVECAR to pass from relax --> static and from static --> other stuff (like PARCHG)
        if calc in ["relax", "static"]:
            new_settings["LWAVE"] = True

        # for PARCHG --> set explicit requirements
        if calc == "parchg":
            new_settings["ISTART"] = 1
            new_settings["LPARD"] = True
            new_settings["LSEPB"] = False
            new_settings["LSEPK"] = False
            new_settings["LWAVE"] = False
            new_settings["NBMOD"] = -2
            print("WARNING: We now set NBMOD = -2 in PARCHG calcs, which is using the absolute energy interval (instead of the eneries relative to the Fermi level) to select contributing bands")
            if "EINT" not in user_passed_settings:
                print("WARNING: PARCHG analysis but no EINT set. Setting to Ef - 2 eV")
                new_settings["EINT"] = " ".join([str(v) for v in [-2.0, 0]])

        # for LOBSTER --> set explicit requirements
        if calc == "lobster":
            new_settings["ISTART"] = 0
            new_settings["LAECHG"] = True
            new_settings["ISYM"] = -1
            new_settings["KSPACING"] = None
            new_settings["NSW"] = 0
            new_settings["LWAVE"] = True
            new_settings["ISMEAR"] = -5

        # for computing polarization --> set explicit requirements
        # Note: VASP will only compute the macroscopic polarization if the system is insulating
        # grep "dipole moment" in your OUTCAR to see if it was computed
        if calc == "polar":
            new_settings["LCALCPOL"] = True
            new_settings["NCORE"] = None

        # for charged defect calculations --> set explicit requirements
        if "defect" in calc:

            # notes for future reference/ potential changes
            #  note 1: Doped recommends IBRION = 1 as an inital step and then switching to = 2 as...
            #    needed, this could be implemented in vasp.py as an error correction if nsw_too_low occurs
            #  note 2: NELECT is set in passer.py
            #  note 3: Unsure at this time if NUPDOWN needs to be set
            #  note 4: LVHAR setting is needed for Freysoldt finite size correction scheme
            #  note 5: ICORELEVEL setting is needed for Kumagai finite size correction scheme

            new_settings["NSW"] = 199
            new_settings["LWAVE"] = True
            new_settings["ISIF"] = 2
            new_settings["ISYM"] = 0
            new_settings["IBRION"] = 2
            new_settings["POTIM"] = 0.2
            new_settings["LREAL"] = False
            new_settings["LVHAR"] = True
            new_settings["ICORELEVEL"] = 0

            if "charged" in calc:
                pass
            elif "neutral" in calc:
                pass
            else:
                raise ValueError("defect type not recognized")

        # for HSE06 single point (static) --> set explicit requirements
        if calc == "sphse06":
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
            for setting, value in calc_settings.items():
                new_settings[setting] = value

        # for preggastatic --> set explicit requirements
        if calc == "preggastatic":
            new_settings["LWAVE"] = True
            new_settings["LDAU"] = False
            new_settings["NSW"] = 0
            new_settings["ISIF"] = None
            new_settings["IBRION"] = None
            new_settings["POTIM"] = None
            new_settings["LCHARG"] = True
            new_settings["LORBIT"] = 0
            new_settings["LVHAR"] = True
            new_settings["ICHARG"] = 0
            new_settings["LAECHG"] = True

        # now we'll customize based on a given standard

        # dmc is the only one implemented other than MP. for MP, we leave alone
        if standard == "dmc":
            dmc_settings = {
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
            for setting, value in dmc_settings.items():
                if setting not in new_settings:
                    new_settings[setting] = value

        # now set our functional given our xc
        if xc in ["metagga", "metaggau"]:
            new_settings["GGA"] = None
            functional = self.configs["functional"]
            # r2SCAN is default
            if not functional:
                new_settings["METAGGA"] = "R2SCAN"
            else:
                new_settings["METAGGA"] = functional[xc]

        if xc in ["gga", "ggau", "hse06"]:
            functional = self.configs["functional"]
            # PBE is default
            if not functional:
                new_settings["GGA"] = "PE"
            else:
                new_settings["GGA"] = functional[xc]

        if xc == "gga":
            # turn off +U b/c our base set wants to use it
            if standard != "mp":
                new_settings["LDAU"] = False
        if xc in ["metagga", "hse06"]:
            new_settings["LDAU"] = False
        if xc in ["ggau", "metaggau"]:
            new_settings["LDAU"] = True

        # set +U related things; NOTE: assumes d electrons (ie doesn't work for f systems)
        if (xc in ["ggau", "metaggau"]) and (standard != "mp"):
            # note: need to pass U values as eg {'LDAUU' : {'Fe' : 5}}
            new_settings["LDAU"] = True
            new_settings["LDAUTYPE"] = 2
            if "LDAUU" in user_passed_settings and (
                isinstance(user_passed_settings["LDAUU"], dict)
            ):
                ldauu = user_passed_settings["LDAUU"]
                ldaul = {el: 2 for el in ldauu}  # assumes d electrons
                ldauj = {el: 0 for el in ldauu}
                new_settings["LDAUL"] = ldaul
                new_settings["LDAUJ"] = ldauj

        # turn magnetism on or off
        if mag == "nm":
            new_settings["ISPIN"] = 1
            new_settings["MAGMOM"] = None
        else:
            new_settings["ISPIN"] = 2
            new_settings["LORBIT"] = 11

        if xc == "hse06":
            # speeding up the calculation by treating KPAR (number) of k-points in parallel
            new_settings["KPAR"] = 2
            if ("KPAR" in new_settings) and (calc == "parchg"):
                del new_settings["KPAR"]

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
    def user_kpoints_settings(self) -> Kpoints:
        """
        Returns KPOINTS object based on user passed settings

            'reciprocal_density' = N --> # kpts * volume = N
            'density' = N --> # kpts * atoms = N
            'auto' = N --> Auto N
            'grid' = [N, N, N] --> N x N x N grid
        """
        user_passed_settings = self.modify_kpoints

        xc, calc = self.xc, self.calc

        new_settings = {}

        # need a KPOINTS file for lobster, so make sure we set something
        if calc == "lobster":
            # reciprocal density = 100 is pymatgen default so that's our default
            new_settings["reciprocal_density"] = self.configs[
                "reciprocal_kpoints_density_for_lobster"
            ]

        # default (light) grid for HSE06 calcs (including preggastatic); user can changes this with modify_kpoints = {'hse06-all' : [X, Y, Z]}
        # if xc == "hse06":
        #     new_settings["grid"] = [2, 2, 2]

        if "1kpt" in calc:
            new_settings["grid"] = [1, 1, 1]

        for setting, value in user_passed_settings.items():
            new_settings[setting] = value

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
    def user_potcar_settings(self) -> dict:
        """
        Returns:
            {element (str) : desired potcar (str)}
        """
        user_passed_settings = self.modify_potcar

        standard = self.standard

        new_settings = {}
        if standard == "dmc":
            # default MPRelaxSet potcar gives unavailable W POTCAR in PBE_54
            new_settings["W"] = "W"

        for setting, value in user_passed_settings.items():
            new_settings[setting] = value
        return new_settings.copy()

    @property
    def vaspset(self) -> VaspSet:
        """
        Returns:
            VaspSet object
                accounts for base_set + xc_calc related mods + user-specified mods
        """
        potcar_functional = self.potcar_functional
        validate_magmom = self.validate_magmom
        if not validate_magmom:
            # I've found no reason to set this True
            validate_magmom = False
        if not potcar_functional:
            # set where POTCARs get pulled from
            potcar_functional = "PBE" if self.standard == "mp" else "PBE_54"

        return self.base_set(
            structure=self.structure,
            user_incar_settings=self.user_incar_settings,
            user_kpoints_settings=self.user_kpoints_settings,
            user_potcar_settings=self.user_potcar_settings,
            user_potcar_functional=potcar_functional,
            validate_magmom=validate_magmom,
        )

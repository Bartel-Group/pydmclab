from pymatgen.io.vasp import Incar, Kpoints, Potcar, Poscar
from pymatgen.io.vasp.sets import MPRelaxSet, MPScanRelaxSet, MPHSERelaxSet
from pydmclab.core.struc import StrucTools


class GetSet(object):

    def __init__(
        self,
        structure,
        xc_calc,
        standard,
        mag,
        potcar_functional=None,
        validate_magmom=False,
        U_values=None,
        modify_incar={},
        modify_kpoints={},
        modify_potcar={},
    ):

        xc, calc = xc_calc.split("-")
        self.xc, self.calc = xc, calc

        self.structure = StrucTools(structure).structure
        self.standard = standard
        self.mag = mag

        self.modify_incar = modify_incar.copy()
        self.modify_kpoints = modify_kpoints.copy()
        self.modify_potcar = modify_potcar.copy()

        self.potcar_functional = potcar_functional
        self.validate_magmom = validate_magmom

        self.U_values = U_values

    @property
    def base_set(self):
        xc, calc = self.xc, self.calc

        if xc in ["gga", "ggau"]:
            return MPRelaxSet
        elif xc in ["metagga", "metaggau"]:
            return MPScanRelaxSet
        elif xc in ["hse06"]:
            return MPHSERelaxSet
        else:
            raise NotImplementedError(f"xc: {xc} not implemented")

    @property
    def user_incar_settings(self):
        user_passed_settings = self.modify_incar
        user_passed_kpoints_settings = self.modify_kpoints

        xc, calc, standard, mag = self.xc, self.calc, self.standard, self.mag

        new_settings = {}
        if mag == "nm":
            new_settings["ISPIN"] = 1
            new_settings["MAGMOM"] = None
        else:
            new_settings["ISPIN"] = 2
            new_settings["LORBIT"] = 11

        if (
            ("NPAR" not in user_passed_settings)
            and ("NCORE" not in user_passed_settings)
            and ("KPAR" not in user_passed_settings)
        ):
            new_settings["NCORE"] = 4

        if calc == "relax":
            new_settings["NSW"] = 199

        if calc in ["static", "lobster", "parchg", "dfpt", "finite_displacements"]:
            new_settings["NSW"] = 0
            new_settings["ISIF"] = None
            new_settings["IBRION"] = None
            new_settings["POTIM"] = None
            new_settings["LCHARG"] = True
            new_settings["LORBIT"] = 0
            new_settings["LVHAR"] = True
            new_settings["ICHARG"] = 0
            new_settings["LAECHG"] = True

        if calc == "dfpt":
            new_settings["IBRION"] = 7
            new_settings["ISYM"] = 2
            new_settings["ALGO"] = "Normal"
            new_settings["ADDGRID"] = True
            new_settings["IALGO"] = 38
            new_settings["NPAR"] = None
            new_settings["NCORE"] = None

        if calc == "finite_displacements":
            new_settings["IBRION"] = 2
            new_settings["ENCUT"] = 700
            new_settings["EDIFF"] = 1e-7
            new_settings["LAECHG"] = False
            new_settings["LREAL"] = False
            new_settings["ALGO"] = "Normal"
            new_settings["NSW"] = 0
            new_settings["LCHARG"] = False

        if calc == "parchg":
            new_settings["ISTART"] = 1
            new_settings["LPARD"] = True
            new_settings["LSEPB"] = False
            new_settings["LSEPK"] = False
            new_settings["LWAVE"] = False
            new_settings["NBMOD"] = -3
            if "EINT" not in user_passed_settings:
                print("WARNING: PARCH analysis but no EINT set. Setting to Ef - 2 eV")
                new_settings["EINT"] = -2.0

        if standard == "dmc":
            new_settings["EDIFF"] = 1e-6
            new_settings["EDIFFG"] = -0.03

            new_settings["ISMEAR"] = 0
            new_settings["SIGMA"] = 0.05

            new_settings["ENCUT"] = 520
            new_settings["ENAUG"] = 1040

            new_settings["LREAL"] = False

            new_settings["ISYM"] = 0

            new_settings["KSPACING"] = 0.22

        if xc in ["metagga", "metaggau"]:
            new_settings["GGA"] = None
            functional = user_passed_settings["fun"]
            if not functional:
                new_settings["METAGGA"] = "R2SCAN"
            else:
                new_settings["METAGGA"] = functional

        if xc in ["gga", "ggau"]:
            functional = user_passed_settings["fun"]
            if not functional:
                new_settings["GGA"] = "PE"
            else:
                new_settings["GGA"] = functional

        if calc == "dielectric":
            new_settings["LVTOT"] = True
            new_settings["LEPSILON"] = True
            new_settings["LOPTICS"] = True
            new_settings["IBRION"] = 8

        if (xc in ["ggau", "metaggau"]) and (standard != "mp"):
            # note: need to pass U values as eg {'LDAUU' : {'Fe' : 5}}
            pass

        if calc == "loose":
            new_settings["ENCUT"] = 400
            new_settings["ENAUG"] = 800
            new_settings["ISIF"] = 3
            new_settings["EDIFF"] = 1e-5
            new_settings["NELM"] = 40

        if calc == "lobster":
            new_settings["ISTART"] = 0
            new_settings["LAECHG"] = True
            new_settings["ISYM"] = -1
            new_settings["KSPACING"] = None
            new_settings["ISMEAR"] = 0
        if user_passed_kpoints_settings:
            new_settings["KSPACING"] = None

        for k, v in user_passed_settings.items():
            new_settings[k] = v

        return new_settings.copy()

    @property
    def user_kpoints_settings(self):
        user_passed_settings = self.modify_kpoints

        xc, calc, standard, mag = self.xc, self.calc, self.standard, self.mag

        new_settings = {}

        if calc == "lobster":
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
        potcar_functional = self.potcar_functional
        validate_magmom = self.validate_magmom
        if not validate_magmom:
            validate_magmom = False
        if not potcar_functional:
            potcar_functional = "PBE" if self.standard == "mp" else "PBE_54"
        return self.base_set(
            structure=self.structure,
            user_incar_settings=self.user_incar_settings,
            user_kpoints_settings=self.user_kpoints_settings,
            user_potcar_settings=self.user_potcar_settings,
            potcar_functional=potcar_functional,
            validate_magmom=validate_magmom,
        )

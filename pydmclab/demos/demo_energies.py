from pydmclab.core.energies import (
    ChemPots,
    FormationEnthalpy,
    FormationEnergy,
    DefectFormationEnergy,
)

from pydmclab.core.struc import StrucTools

from pydmclab.hpc.analyze import AnalyzeVASP
import os

""" 
Purpose:
    Thermodynamics runs the world :)
    
Basic idea:
    ChemPots: get elemental reference energies
    FormationEnthalpy: get formation enthalpy at 0 K
    FormationEnergy: get formation energy at T 
    DefectFormationEnergy: get defect formation energy
"""


def get_0K_chemical_potentials():
    """
    These are the elemental reference energies you need to reference your DFT calculated internal energies against to get 0 K formation enthalpies (energies)
        - different functionals (and standards) give different reference energies
        - critical to use compatible reference energies w/ your calculations
    """

    # assuming DMC standards for everything

    mus = ChemPots(functional="r2scan").chempots
    print("\n XC = metagga (T = 0 K)")
    print("mu_Al = %.2f eV/at" % mus["Al"])
    print("mu_N = %.2f eV/at" % mus["N"])

    mus = ChemPots(functional="pbe").chempots
    print("\n XC = gga (T = 0 K)")
    print("mu_Al = %.2f eV/at" % mus["Al"])
    print("mu_N = %.2f eV/at" % mus["N"])


def get_T_dependent_chemical_potentials():
    """
    we can also use experimental data to get chemical potentials at T
        - from Barin's thermochemical data of pure substances
    """
    print("\n T-dependent mus")
    allowed_Ts = list(range(300, 2100, 100))
    for T in allowed_Ts:
        # note functional and standard don't matter for T > 0
        print("mu_N(%i K) = %.3f eV/at" % (T, ChemPots(temperature=T).chempots["N"]))


def get_chemical_potentials_with_gas_activity():
    """
    we can also account for the effects of gas concentration (activity) in the atmosphere
        - according to dmu = RTln(p_gas/p_gas,ref)
    """
    print("\n pN2-dependent mus")
    pN2s = [10**i for i in range(-6, 1)][::-1]
    T = 1000
    for pN2 in pN2s:
        print(
            "mu_N(1000 K, pN2 = %.0e atm) = %.3f eV/at"
            % (
                pN2,
                ChemPots(temperature=1000, partial_pressures={"N": pN2}).chempots["N"],
            )
        )


def get_mus_with_custom_shift():
    """
    we can also just specify whatever we want or shift chemical potentials as needed
    """
    print("\n custom mu shifts")
    mus = ChemPots(
        functional="r2scan", standard="dmc", user_chempots={"N": 10000}
    ).chempots
    print("user set mu_N = %.2f eV/at" % mus["N"])

    mus = ChemPots(functional="r2scan", standard="dmc", user_dmus={"N": 0.1}).chempots
    print("user shifted mu_N up by 0.1 eV/at to %.2f eV/at" % mus["N"])


def get_formation_enthalpy_at_0K():
    """
    these are DFT formation energies (at 0 K)
        Bartel npj 2019 (and others) shows these agree well with dHf(298 K)
    """
    print("\n formation enthalpy (internal energy)")
    formula = "SiC"
    E_DFT = -7.5308  # per atom
    mus = ChemPots(functional="pbe").chempots

    Ef = FormationEnthalpy(formula, E_DFT, mus).Ef
    print("Ef = %.2f eV/atom" % Ef)


def get_T_dependent_formation_energy_with_vibrational_entropy():
    """
    Bartel et al Nature Comms 2019 allows us to get DFT energies compatbile w/ finite T chemical potentials
        by approximating vibrational entropy
    """
    print("\n formation energy (Gibbs free energy from vibrational entropy)")
    data_dir = "../data/test_data/vasp/AlN"

    formula = "AlN"

    Ef = -1.6  # eV/atom (you could also compute this with FormationEnthalpy)
    path_to_poscar = os.path.join(data_dir, "CONTCAR")
    structure = StrucTools(path_to_poscar).structure

    for T in [0, 300, 900, 1500]:
        mus = ChemPots(temperature=T).chempots
        dGf = FormationEnergy(formula, Ef, mus, structure, include_Svib=True).dGf(T)
        print("dGf(%i K) = %.2f eV/atom" % (T, dGf))


def get_T_dependent_formation_energy_with_configurational_entropy():
    """
    we can also account for configurational entropy using a simple ideal mixing approximation
        S_config = -k_B * (x * ln(x) + (1-x) * ln(1-x))
        - note: this neglects any short-range order and assumes fully random occupations
    """
    print("\n formation energy (Gibbs free energy from configurational entropy)")
    formula = "Ag3Au"
    Ef = -0.041  # eV/atom (you could also compute this with FormationEnthalpy)

    for T in [0, 300, 900, 1500]:
        mus = ChemPots(temperature=T).chempots
        dGf = FormationEnergy(
            formula,
            Ef,
            mus,
            include_Svib=False,
            include_Sconfig=True,
            x_config=0.75,  # 3 Ag and 1 Au can occupy the same site
            n_config=1,  # one site having mixed occupation
        ).dGf(T)
        print("dGf(%i K) = %.2f eV/atom" % (T, dGf))


def get_defect_formation_energy():
    """
    defect formation energies tell me the energy gain or penalty of creating defects
        - vacancies, interstitials, substitutions
        - antisites not implemented yet...
    """
    pristine_formula = "GaN"
    defect_formula = "AlGa3N4"

    Ef_pristine = -0.668  # eV/atom
    Ef_defect = -0.896  # eV/atom

    Eg_pristine = 1.74  # eV

    q = 0  # neutral substitutional defect

    charge_correction = 0  # not charged

    shared_el = "N"  # for normalizing formula

    mus = ChemPots(functional="pbe", standard="mp").chempots  # need mus (for Ga and Al)

    E_Fermi = 0  # neutral defect, so let's evaluate at VBM

    dfe = DefectFormationEnergy(
        formula_defect=defect_formula,
        formula_pristine=pristine_formula,
        E_defect=Ef_defect,
        E_pristine=Ef_pristine,
        Eg_pristine=Eg_pristine,
        charge_defect=q,
        charge_correction=charge_correction,
        shared_el_for_basis=shared_el,
        chempots=mus,
    ).Efs[E_Fermi]

    print("defect formation energy for Al_Ga in GaN = %.2f eV/atom" % dfe)


def main():
    get_0K_chemical_potentials()
    get_T_dependent_chemical_potentials()
    get_chemical_potentials_with_gas_activity()
    get_mus_with_custom_shift()
    get_formation_enthalpy_at_0K()
    get_T_dependent_formation_energy_with_vibrational_entropy()
    get_T_dependent_formation_energy_with_configurational_entropy()
    get_defect_formation_energy()
    return


if __name__ == "__main__":
    main()

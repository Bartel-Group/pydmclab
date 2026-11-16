import os
import unittest
from pydmclab.core.energies import (
    ChemPots,
    FormationEnthalpy,
    FormationEnergy,
    DefectFormationEnergy,
    ReactionEnergy,
    UMAFormationEnergy,
)
from pydmclab.core.struc import StrucTools
from pydmclab.data.thermochem import uma_s1p2p1_mp_elemental_references

from pymatgen.core import Lattice, Structure

import numpy as np

TEST_DATA = os.path.join("..", "pydmclab", "data", "test_data", "energies")


class UnitTestChemPots(unittest.TestCase):
    def test_zeroK_mus(self):
        cp = ChemPots(functional="pbe", standard="dmc")
        muTi = cp.chempots["Ti"]
        muO = cp.chempots["O"]

        # from 230122_dmc-mus.json
        self.assertEqual(muTi, -7.805884913333333)
        self.assertEqual(muO, -4.942339795)

        cp = ChemPots(functional="r2scan", standard="dmc")
        muTi = cp.chempots["Ti"]
        muO = cp.chempots["O"]
        self.assertEqual(muTi, -12.935879826666666)
        self.assertEqual(muO, -5.93051377)

        # from mus_from_mp_no_corrections.json + mp2020_compatibility_dmus()
        cp = ChemPots(functional="pbe", standard="mp")
        muTi = cp.chempots["Ti"]
        muO = cp.chempots["O"]
        self.assertEqual(muTi, -7.895052840000001)
        self.assertEqual(muO, -4.94795546875 - -0.687)

    def test_Tdep_mus(self):
        cp = ChemPots(temperature=1000)
        muTi = cp.chempots["Ti"]
        muO = cp.chempots["O"]

        # from elemental_gibbs_energies_T.json
        self.assertEqual(muTi, -0.4631694045706587)
        self.assertEqual(muO, -1.1440332694201172)

    def test_partial_pressure_effect(self):
        cp = ChemPots(temperature=1000, partial_pressures={"O": 1 / 1000})

        muO_at_1000ppm = cp.chempots["O"]

        cp = ChemPots(temperature=1000, partial_pressures={"O": 1})
        muO_at_1atm = cp.chempots["O"]

        # muO2(T, PO2) = muO2(0 K, 1 atm) + 1/2 * RTln(PO2)
        muO_check = muO_at_1atm + 0.008314 / 96.485 * 1000 * 0.5 * np.log(1 / 1000)

        self.assertAlmostEqual(muO_at_1000ppm, muO_check, places=3)

    def test_user_modifications(self):
        cp = ChemPots(user_chempots={"Re": -123})
        muRe = cp.chempots["Re"]
        self.assertEqual(muRe, -123)

        cp = ChemPots(functional="r2scan", standard="dmc", user_dmus={"Ti": 0.23})
        self.assertEqual(cp.chempots["Ti"], -12.935879826666666 + 0.23)


class UnitTestFormationEnthalpy(unittest.TestCase):
    def test_Ef_zeroK(self):
        formula = "Li2MnO2F"
        E_DFT = -6.0355
        mus = ChemPots(functional="pbe", standard="dmc").chempots

        Ef_auto = FormationEnthalpy(formula, E_DFT, mus).Ef
        Ef_hard = (
            1 / 6 * (E_DFT * 6 - mus["Li"] * 2 - mus["Mn"] - mus["O"] * 2 - mus["F"])
        )

        self.assertAlmostEqual(Ef_auto, Ef_hard, places=3)

    def test_dGf_Sconfig(self):
        formula = "Ba2ZrNbS6"
        mixed_formula = "BaZr0.5Nb0.5S3"
        Ef = -3
        x, n, T = 0.5, 2, 1000
        chempots = ChemPots().chempots
        dGf = FormationEnergy(
            formula=formula,
            Ef=Ef,
            chempots=chempots,
            include_Sconfig=True,
            include_Svib=False,
            n_config=n,
            x_config=x,
        ).dGf(T)
        S_config_hard = -8.617e-5 * n * 2 * 0.5 * np.log(0.5) / 10
        dGf_hard = Ef - 1000 * S_config_hard
        self.assertAlmostEqual(dGf, dGf_hard, places=3)

    def test_dGf_Svib(self):
        formula = "NaCl"
        Ef_0K = -2.11

        T = 1500
        mus = ChemPots(temperature=T).chempots

        dGf = FormationEnergy(
            formula=formula,
            Ef=Ef_0K,
            chempots=mus,
            include_Sconfig=False,
            include_Svib=True,
            atomic_volume=174.50 / 8,
        ).dGf(T)

        dGf_MP = -1.166

        self.assertAlmostEqual(dGf, dGf_MP, places=1)

    def test_dGf_Svib_and_Sconfig(self):
        formula = "NaCl"
        Ef_0K = -2.11

        T = 1500
        mus = ChemPots(temperature=T).chempots

        dGf_just_Svib = FormationEnergy(
            formula=formula,
            Ef=Ef_0K,
            chempots=mus,
            include_Sconfig=False,
            include_Svib=True,
            atomic_volume=174.50 / 8,
        ).dGf(T)

        x, n = 0.125, 1
        chempots = ChemPots().chempots
        dGf_just_Sconfig = FormationEnergy(
            formula=formula,
            Ef=Ef_0K,
            chempots=chempots,
            include_Sconfig=True,
            include_Svib=False,
            n_config=n,
            x_config=x,
        ).dGf(T)

        difference_from_Svib = dGf_just_Svib - Ef_0K
        difference_from_Sconfig = dGf_just_Sconfig - Ef_0K

        dGf_hard = Ef_0K + difference_from_Sconfig + difference_from_Svib

        chempots = ChemPots(temperature=T).chempots
        dGf = FormationEnergy(
            formula=formula,
            Ef=Ef_0K,
            chempots=chempots,
            include_Sconfig=True,
            include_Svib=True,
            n_config=n,
            x_config=x,
            atomic_volume=174.50 / 8,
        ).dGf(T)

        difference_from_both = dGf - Ef_0K

        self.assertAlmostEqual(dGf, dGf_hard, places=1)


class UnitTestDefects(unittest.TestCase):
    def test_defects(self):
        chempots = ChemPots(functional="r2scan", standard="dmc").chempots

        E_pristine = -1.744
        E_defect = -1.461
        formula_pristine = "LiCoO2"
        formula_defect = "CoO2"
        charge_defect = 0
        Eg_pristine = 0.66
        charge_correction = 0
        fixed_els = ["Co", "O"]

        dfe = DefectFormationEnergy(
            E_pristine=E_pristine,
            formula_pristine=formula_pristine,
            Eg_pristine=Eg_pristine,
            E_defect=E_defect,
            formula_defect=formula_defect,
            charge_defect=charge_defect,
            chempots=chempots,
            charge_correction=charge_correction,
            fixed_els=fixed_els,
        )

        self.assertEqual(dfe.rxn_string, "CoO2 + Li -> LiCoO2")
        self.assertEqual(dfe.n_defects, 1)
        self.assertEqual(dfe.defect_concentration, 1)
        self.assertEqual(dfe.charge_contribution[0.0], 0)
        self.assertEqual(dfe.Efs[0.0], dfe.dE_rxn / 1)

        Ef1 = dfe.Efs[0.0]

        dfe = DefectFormationEnergy(
            E_pristine=E_pristine,
            formula_pristine=formula_pristine,
            Eg_pristine=Eg_pristine,
            E_defect=E_defect,
            formula_defect="Li97Co100O200",
            charge_defect=charge_defect,
            chempots=chempots,
            charge_correction=charge_correction,
            fixed_els=fixed_els,
        )

        Ef2 = dfe.Efs[0.0]

        self.assertEqual(dfe.rxn_string, "0.01 Li97(CoO2)100 + 0.03 Li -> LiCoO2")
        self.assertEqual(dfe.n_defects, 3)
        self.assertEqual(dfe.defect_concentration, 0.03)
        self.assertEqual(dfe.charge_contribution[0.0], 0)
        self.assertEqual(dfe.Efs[0.0], dfe.dE_rxn / 3)

        dfe = DefectFormationEnergy(
            E_pristine=E_pristine,
            formula_pristine=formula_pristine,
            Eg_pristine=Eg_pristine,
            E_defect=E_defect,
            formula_defect="Li97Na3Co100O200",
            charge_defect=charge_defect,
            chempots=chempots,
            charge_correction=charge_correction,
            fixed_els=fixed_els,
        )

        Ef3 = dfe.Efs[0.0]

        self.assertEqual(
            dfe.rxn_string, "0.01 Na3Li97(CoO2)100 + 0.03 Li -> 0.03 Na + LiCoO2"
        )
        self.assertEqual(dfe.n_defects, 3)
        self.assertEqual(dfe.defect_type, "substitutional")
        self.assertEqual(dfe.defect_concentration, 0.03)
        self.assertEqual(dfe.charge_contribution[0.0], 0)
        self.assertEqual(dfe.Efs[0.0], dfe.dE_rxn / 3)

        dfe = DefectFormationEnergy(
            E_pristine=E_pristine,
            formula_pristine=formula_pristine,
            Eg_pristine=Eg_pristine,
            E_defect=E_defect,
            formula_defect="CoO2",
            charge_defect=1,
            chempots=chempots,
            charge_correction=0,
            fixed_els=fixed_els,
        )

        Ef5 = dfe.Efs[0.1]

        self.assertAlmostEqual(Ef1, Ef5 - 0.1 * 1, places=5)


class UnitTestReactions(unittest.TestCase):
    def test_rxn_string(self):
        reactants = ["RuO2", "FeO"]
        products = ["RuFeO3"]
        rxn = ReactionEnergy({}, reactants, products).rxn_string
        self.assertEqual(rxn, "RuO2 + FeO -> FeRuO3")

        products = ["RuFeO3", "O2"]
        rxn = ReactionEnergy({}, reactants, products).rxn_string
        self.assertEqual(rxn, "RuO2 + FeO -> FeRuO3")

        products = ["RuFeO4", "O2"]
        rxn = ReactionEnergy({}, reactants, products).rxn_string
        self.assertEqual(rxn, "RuO2 + FeO + 0.5 O2 -> FeRuO4")

        products = ["RuFeO2", "O2"]
        rxn = ReactionEnergy({}, reactants, products).rxn_string
        self.assertEqual(rxn, "RuO2 + FeO -> FeRuO2 + 0.5 O2")

        products = ["RuFeO2", "O2", "H2O", "N2", "CO2"]
        rxn = ReactionEnergy({}, reactants, products).rxn_string
        self.assertEqual(rxn, "RuO2 + FeO -> FeRuO2 + 0.5 O2")

        input_energies = {
            "CaO": {"E": -6.618 / 2},
            "TiO2": {"E": -10.505 / 3},
            "CaTiO3": {"E": -17.787 / 5},
        }

        reactants = ["CaO", "TiO2"]
        products = ["CaTiO3"]
        rxn = ReactionEnergy(input_energies, reactants, products, energy_key="E")
        self.assertEqual(rxn.rxn_string, "CaO + TiO2 -> CaTiO3")
        self.assertEqual(rxn.coefs["O2Ti1"], -1)
        self.assertEqual(rxn.coefs["Ca1O3Ti1"], 1)
        self.assertAlmostEqual(rxn.dE_rxn, -0.664 / 1, places=3)

        rxn = ReactionEnergy(
            input_energies, reactants, products, energy_key="E", norm="atom"
        )

        self.assertAlmostEqual(rxn.dE_rxn, -0.664 / 5, places=3)

        reactants = ["CaO", "TiO2"]
        products = ["CaTi2O4", "O2"]
        input_energies["CaTi2O4"] = {"E": -17.787 / 5}
        input_energies["O2"] = {"E": -1}
        rxn = ReactionEnergy(input_energies, reactants, products, energy_key="E")
        self.assertEqual(rxn.rxn_string, "CaO + 2 TiO2 -> CaTi2O4 + 0.5 O2")
        self.assertEqual(rxn.coefs["O1"], 1)

        input_energies = {
            "Co1La1O3": {"E": -7.939448833465576},
            "Co8La8O23": {"E": -7.974939346313477},
            "Co4La4O11": {"E": -8.00416374206543},
            "Co8La8O21": {"E": -8.012917518615723},
            "Co2La2O5": {"E": -8.028905868530273},
        }

        reactants = ["LaCoO3", "LaCoO2.5"]
        products = ["LaCoO2.875"]
        rxn = ReactionEnergy(input_energies, reactants, products)
        self.assertEqual(rxn.rxn_string, "6 LaCoO3 + La2Co2O5 -> La8Co8O23")
        self.assertEqual(rxn.coefs["Co2La2O5"], -1)
        self.assertEqual(rxn.coefs["Co8La8O23"], 1)
        self.assertAlmostEqual(rxn.species["Co8La8O23"]["E"], -7.9749393, places=5)
        self.assertAlmostEqual(rxn.dE_rxn, -0.57901667, places=3)


class UnitTestUMAFormationEnergy(unittest.TestCase):
    def setUp(self):
        self.uma = UMAFormationEnergy()
        self.mus = uma_s1p2p1_mp_elemental_references()

        # LiCoO2 (R-3m), 4 atoms
        self.licoo2 = Structure.from_spacegroup(
            "R-3m",
            Lattice.hexagonal(2.82, 14.05),
            ["Li", "Co", "O"],
            [[0, 0, 0], [0, 0, 0.5], [0, 0, 0.24]],
        )
        self.E_licoo2 = -5.7276 - 0.030

        # fcc Cu, 4 atoms
        self.cu = Structure.from_spacegroup(
            "Fm-3m", Lattice.cubic(3.6), ["Cu"], [[0, 0, 0]]
        )

    def test_packaged_references(self):
        self.assertEqual(self.uma.elemental_references, self.mus)
        self.assertEqual(self.mus["Co"], -7.036621506169756)
        self.assertEqual(self.mus["Cl"], -1.8500821359903892)

    def test_correction(self):
        # OMat24 corrections: oxide (-0.657 eV/O) + Co GGA/GGA+U mixing (-2.151 eV/Co)
        correction = self.uma.correction_per_atom(self.licoo2, self.E_licoo2)
        self.assertAlmostEqual(correction, (2 * -0.657 + -2.151) / 4, places=6)

        # no correction for an element that is not an anion
        self.assertEqual(self.uma.correction_per_atom(self.cu, self.mus["Cu"]), 0)

    def test_Ef(self):
        Ef = self.uma.formation_energy(self.licoo2, self.E_licoo2)
        correction = (2 * -0.657 + -2.151) / 4
        mus = self.mus
        Ef_hard = (
            self.E_licoo2
            + correction
            - (mus["Li"] + mus["Co"] + 2 * mus["O"]) / 4
        )
        self.assertAlmostEqual(Ef, Ef_hard, places=6)

        # an element at its reference energy has Ef = 0
        self.assertAlmostEqual(
            self.uma.formation_energy(self.cu, self.mus["Cu"]), 0, places=8
        )

    def test_structure_input_types(self):
        Ef = self.uma.formation_energy(self.licoo2, self.E_licoo2)
        for s in [
            self.licoo2.as_dict(),
            StrucTools(self.licoo2).structure,
        ]:
            self.assertAlmostEqual(
                self.uma.formation_energy(s, self.E_licoo2), Ef, places=8
            )

    def test_user_references(self):
        mus = {"Li": -1.0, "Co": -2.0, "O": -3.0}
        uma = UMAFormationEnergy(elemental_references=mus)
        Ef = uma.formation_energy(self.licoo2, -5)
        correction = (2 * -0.657 + -2.151) / 4
        self.assertAlmostEqual(Ef, -5 + correction - (-1 - 2 - 6) / 4, places=6)

    def test_user_compatibility_yaml(self):
        # fake corrections: oxide (-1.0 eV/O) + Co GGA/GGA+U mixing (-2.0 eV/Co)
        uma = UMAFormationEnergy(
            compatibility_yaml=os.path.join(
                TEST_DATA, "fake_mp2020-style_compatibility.yaml"
            )
        )
        correction = uma.correction_per_atom(self.licoo2, self.E_licoo2)
        self.assertAlmostEqual(correction, (2 * -1.0 + -2.0) / 4, places=6)

        # packaged elemental references are still used
        Ef = uma.formation_energy(self.licoo2, self.E_licoo2)
        mus = self.mus
        Ef_hard = (
            self.E_licoo2
            + correction
            - (mus["Li"] + mus["Co"] + 2 * mus["O"]) / 4
        )
        self.assertAlmostEqual(Ef, Ef_hard, places=6)

    def test_missing_reference(self):
        uma = UMAFormationEnergy(elemental_references={"Li": -1.0, "O": -3.0})
        with self.assertRaises(KeyError):
            uma.formation_energy(self.licoo2, self.E_licoo2)


if __name__ == "__main__":
    unittest.main()

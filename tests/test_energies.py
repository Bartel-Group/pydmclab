import unittest
from pydmclab.core.energies import ChemPots, FormationEnthalpy, FormationEnergy

import numpy as np


class UnitTestChemPots(unittest.TestCase):
    def test_zeroK_mus(self):
        cp = ChemPots(functional="pbe", standard="dmc")
        muTi = cp.chempots["Ti"]
        muO = cp.chempots["O"]

        # from 230122_dmc-mus.json
        self.assertEqual(muTi, -7.74480818)
        self.assertEqual(muO, -4.71400617)

        cp = ChemPots(functional="r2scan", standard="dmc")
        muTi = cp.chempots["Ti"]
        muO = cp.chempots["O"]
        self.assertEqual(muTi, -12.852719325)
        self.assertEqual(muO, -5.585154205)

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
        self.assertEqual(cp.chempots["Ti"], -12.852719325 + 0.23)


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


if __name__ == "__main__":
    unittest.main()

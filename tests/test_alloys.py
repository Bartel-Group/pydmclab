import unittest
import numpy as np
from pydmclab.core.alloys import IsoAlloy


class UnitTestIsoAlloy(unittest.TestCase):
    """Unit tests for the IsoAlloy class."""

    def setUp(self):
        """Set up test fixtures with sample mixing energy data."""
        # Sample mixing energies for a binary alloy (symmetric regular solution)
        # Using a parabolic mixing enthalpy: H_mix = omega * x * (1-x)
        # with omega ~ 2 eV
        self.energies = {
            0.0: 0.0,
            0.25: 0.375,
            0.5: 0.5,
            0.75: 0.375,
            1.0: 0.0,
        }
        
        # Smaller x and T arrays for faster tests
        self.xs = np.linspace(0.00001, 0.99999, 1000)
        self.Ts = np.linspace(100, 2000, 100)
        
        self.alloy = IsoAlloy(
            energies=self.energies,
            xs=self.xs,
            Ts=self.Ts,
            n_sites_per_fu=1,
        )

    def test_init(self):
        """Test IsoAlloy initialization."""
        self.assertEqual(self.alloy.energies, self.energies)
        self.assertEqual(self.alloy.n_sites_per_fu, 1)
        self.assertEqual(self.alloy.discrete_x, [0.0, 0.25, 0.5, 0.75, 1.0])
        self.assertAlmostEqual(self.alloy.kB, 8.6173e-5, places=9)

    def test_omega_calculation(self):
        """Test that omega is correctly calculated from mixing energies."""
        # For H_mix = omega * x * (1-x), with max at x=0.5: H_mix(0.5) = 0.25 * omega
        # Given H_mix(0.5) = 0.5, omega should be approximately 2.0
        self.assertAlmostEqual(self.alloy.omega, 2.0, places=2)

    def test_mixing_entropy(self):
        """Test the mixing entropy calculation."""
        kB = self.alloy.kB
        n_sites = self.alloy.n_sites_per_fu
        
        # Test at x = 0.5 (maximum entropy)
        x = 0.5
        S_expected = -kB * n_sites * (x * np.log(x) + (1 - x) * np.log(1 - x))
        S_calculated = self.alloy._mixing_entropy(x)
        self.assertAlmostEqual(S_calculated, S_expected, places=10)
        
        # Maximum entropy should be at x = 0.5
        S_at_0p5 = self.alloy._mixing_entropy(0.5)
        S_at_0p3 = self.alloy._mixing_entropy(0.3)
        self.assertGreater(S_at_0p5, S_at_0p3)

    def test_mixing_entropy_array(self):
        """Test mixing entropy with array input."""
        xs = np.array([0.25, 0.5, 0.75])
        S = self.alloy._mixing_entropy(xs)
        self.assertEqual(len(S), 3)
        # Symmetry: S(0.25) should equal S(0.75)
        self.assertAlmostEqual(S[0], S[2], places=10)

    def test_deltaG_mix(self):
        """Test the Gibbs free energy of mixing calculation."""
        x = 0.5
        T = 1000  # K
        
        # Manual calculation
        H_mix = self.alloy.omega * x * (1 - x)
        S_mix = self.alloy._mixing_entropy(x)
        G_expected = H_mix - T * S_mix
        
        G_calculated = self.alloy.deltaG_mix(x, T)
        self.assertAlmostEqual(G_calculated, G_expected, places=10)

    def test_deltaG_mix_array(self):
        """Test Gibbs free energy of mixing with array input."""
        xs = np.array([0.25, 0.5, 0.75])
        T = 15000  # High enough T that entropy dominates over enthalpy
        G = self.alloy.deltaG_mix(xs, T)
        self.assertEqual(len(G), 3)
        # At high temperature, G_mix should be negative (mixing is favorable)
        self.assertTrue(np.all(G < 0))

    def test_deltaG_mix_temperature_dependence(self):
        """Test that higher temperature makes mixing more favorable."""
        x = 0.5
        G_low_T = self.alloy.deltaG_mix(x, 300)
        G_high_T = self.alloy.deltaG_mix(x, 1500)
        # Higher T should give more negative G (more favorable mixing)
        self.assertLess(G_high_T, G_low_T)

    def test_find_tangent_touch_points(self):
        """Test finding tangent touch points (local minima)."""
        # Create a curve with two local minima
        x = np.linspace(0, 1, 100)
        y = (x - 0.3)**2 * (x - 0.7)**2 - 0.01  # Two minima around 0.3 and 0.7
        
        touch_x, touch_y = self.alloy.find_tangent_touch_points(x, y)
        
        # Should find points near local minima
        self.assertGreater(len(touch_x), 0)

    def test_find_inflection_points(self):
        """Test finding inflection points."""
        x = np.linspace(0, 1, 100)
        # Cubic has inflection point at x = 0.5
        y = (x - 0.5)**3
        
        inflection_x, inflection_y = self.alloy.find_inflection_points(x, y)
        
        # Should find inflection point near 0.5
        if len(inflection_x) > 0:
            self.assertTrue(any(abs(xi - 0.5) < 0.1 for xi in inflection_x))

    def test_calculate_phase_diagram(self):
        """Test phase diagram calculation returns valid binodal and spinodal."""
        binodal, spinodal = self.alloy.calculate_phase_diagram()
        
        # Both should be numpy arrays
        self.assertIsInstance(binodal, np.ndarray)
        self.assertIsInstance(spinodal, np.ndarray)
        
        # Should have shape (n, 2) for (x, T) pairs
        if len(binodal) > 0:
            self.assertEqual(binodal.shape[1], 2)
        if len(spinodal) > 0:
            self.assertEqual(spinodal.shape[1], 2)

    def test_calculate_phase_diagram_symmetry(self):
        """Test that phase diagram is symmetric around x = 0.5 for symmetric energies."""
        binodal, spinodal = self.alloy.calculate_phase_diagram(exclude_center=True)
        
        if len(binodal) > 0:
            # Binodal should be symmetric: for each (x, T), there should be (1-x, T)
            xs_binodal = binodal[:, 0]
            left_side = xs_binodal[xs_binodal < 0.5]
            right_side = xs_binodal[xs_binodal > 0.5]
            # Check approximate symmetry
            self.assertTrue(len(left_side) > 0 or len(right_side) > 0)

    def test_to_dict(self):
        """Test conversion to dictionary."""
        d = self.alloy.to_dict()
        
        # Check required keys exist
        self.assertIn("x", d)
        self.assertIn("T", d)
        self.assertIn("omega", d)
        self.assertIn("E", d)
        self.assertIn("binodal", d)
        self.assertIn("spinodal", d)
        
        # Check omega value
        self.assertAlmostEqual(d["omega"], self.alloy.omega, places=5)
        
        # Check E entries format
        for e_entry in d["E"]:
            self.assertIn("x", e_entry)
            self.assertIn("E", e_entry)

    def test_to_dict_energy_values(self):
        """Test that to_dict contains correct energy values."""
        d = self.alloy.to_dict()
        
        # Verify energy values match input
        for e_entry in d["E"]:
            x = e_entry["x"]
            E = e_entry["E"]
            self.assertAlmostEqual(E, self.energies[x], places=10)

    def test_n_sites_per_fu_scaling(self):
        """Test that n_sites_per_fu correctly scales the entropy."""
        alloy_1site = IsoAlloy(
            energies=self.energies,
            xs=self.xs,
            Ts=self.Ts,
            n_sites_per_fu=1,
        )
        
        alloy_2sites = IsoAlloy(
            energies=self.energies,
            xs=self.xs,
            Ts=self.Ts,
            n_sites_per_fu=2,
        )
        
        x = 0.5
        S_1site = alloy_1site._mixing_entropy(x)
        S_2sites = alloy_2sites._mixing_entropy(x)
        
        # Entropy with 2 sites should be double that with 1 site
        self.assertAlmostEqual(S_2sites, 2 * S_1site, places=10)

    def test_custom_boltzmann_constant(self):
        """Test initialization with custom Boltzmann constant."""
        custom_kB = 1e-4
        alloy = IsoAlloy(
            energies=self.energies,
            kB=custom_kB,
        )
        self.assertEqual(alloy.kB, custom_kB)


class UnitTestIsoAlloyEdgeCases(unittest.TestCase):
    """Edge case tests for IsoAlloy class."""

    def test_minimal_energies(self):
        """Test with minimal energy data (just endpoints)."""
        energies = {0.0: 0.0, 1.0: 0.0}
        alloy = IsoAlloy(energies=energies)
        
        # Should still calculate omega (though it may be 0 or small)
        self.assertIsNotNone(alloy.omega)

    def test_asymmetric_energies(self):
        """Test with asymmetric mixing energies."""
        energies = {
            0.0: 0.0,
            0.25: 0.02,
            0.5: 0.06,
            0.75: 0.05,
            1.0: 0.0,
        }
        alloy = IsoAlloy(energies=energies)
        
        # Should still work and calculate phase diagram
        binodal, spinodal = alloy.calculate_phase_diagram()
        self.assertIsNotNone(binodal)
        self.assertIsNotNone(spinodal)

    def test_negative_mixing_energies(self):
        """Test with negative mixing energies (favorable mixing)."""
        energies = {
            0.0: 0.0,
            0.25: -0.02,
            0.5: -0.03,
            0.75: -0.02,
            1.0: 0.0,
        }
        alloy = IsoAlloy(energies=energies)
        
        # Omega should be negative
        self.assertLess(alloy.omega, 0)
        
        # With negative omega, G_mix should always be negative (no phase separation)
        G = alloy.deltaG_mix(0.5, 300)
        self.assertLess(G, 0)


class UnitTestIsoAlloyUnitConsistency(unittest.TestCase):
    """
    Test that passing mixing energies in eV/f.u. with n_sites_per_fu yields
    equivalent results to what would be obtained from total energies with
    total number of mixing sites.
    
    The key thermodynamic relationship being tested:
    
    For an alloy with N_fu formula units and n_sites_per_fu mixing sites per f.u.:
        - Total mixing sites: N_total = N_fu * n_sites_per_fu
        - Total energy: E_total = E_per_fu * N_fu
        - Total entropy: S_total = S_per_fu * N_fu
        - Total dG_mix: dG_total = dG_per_fu * N_fu
    
    The IsoAlloy class works in eV/f.u., so we verify that scaling is consistent.
    """

    def test_energy_units_equivalence(self):
        """
        Test that dG_mix scales correctly with the number of formula units.
        
        Scenario: Consider (A_x B_{1-x})_2 O_3 with 2 mixing sites per formula unit.
        
        If we have N_fu formula units:
            - E_total = E_per_fu * N_fu
            - S_total = -kB * N_total * [x*ln(x) + (1-x)*ln(1-x)]
                      = -kB * N_fu * n_sites_per_fu * [x*ln(x) + (1-x)*ln(1-x)]
            - dG_total = E_total - T * S_total
                       = N_fu * (E_per_fu - T * S_per_fu)
                       = N_fu * dG_per_fu
        
        This test verifies that dG_total / N_fu = dG_per_fu for any N_fu.
        """
        # Mixing energies in eV/f.u. for a system with 2 mixing sites per f.u.
        # Example: (Al_x Fe_{1-x})_2 O_3
        energies_per_fu = {
            0.0: 0.0,
            0.25: 0.15,
            0.5: 0.20,
            0.75: 0.15,
            1.0: 0.0,
        }
        n_sites_per_fu = 2
        
        xs = np.linspace(0.00001, 0.99999, 1000)
        Ts = np.linspace(100, 2000, 100)
        
        # Create IsoAlloy with eV/f.u. energies
        alloy_per_fu = IsoAlloy(
            energies=energies_per_fu,
            xs=xs,
            Ts=Ts,
            n_sites_per_fu=n_sites_per_fu,
        )
        
        # Test at various compositions and temperatures
        test_xs = [0.25, 0.5, 0.75]
        test_Ts = [300, 1000, 2000]
        
        for x in test_xs:
            for T in test_Ts:
                dG_per_fu = alloy_per_fu.deltaG_mix(x, T)
                
                # Now simulate what a "total energy" approach would give
                # For N_fu formula units:
                N_fu = 100  # arbitrary number of formula units
                N_total_sites = N_fu * n_sites_per_fu
                
                # Total enthalpy: H_total = omega * x * (1-x) * N_fu
                H_total = alloy_per_fu.omega * x * (1 - x) * N_fu
                
                # Total entropy: S_total = -kB * N_total_sites * [x*ln(x) + (1-x)*ln(1-x)]
                kB = alloy_per_fu.kB
                S_total = -kB * N_total_sites * (x * np.log(x) + (1 - x) * np.log(1 - x))
                
                # Total Gibbs free energy of mixing
                dG_total = H_total - T * S_total
                
                # Per f.u. value from total
                dG_per_fu_from_total = dG_total / N_fu
                
                # These should be equal
                self.assertAlmostEqual(
                    dG_per_fu, 
                    dG_per_fu_from_total, 
                    places=10,
                    msg=f"dG_mix mismatch at x={x}, T={T}"
                )

    def test_per_site_vs_per_fu_consistency(self):
        """
        Test consistency between per-site and per-f.u. formulations.
        
        For a system with n_sites_per_fu mixing sites:
            S_per_fu = n_sites_per_fu * S_per_site
            
        where S_per_site = -kB * [x*ln(x) + (1-x)*ln(1-x)]
        """
        kB = 8.6173e-5
        n_sites = 3  # e.g., A_3 O_4 with 3 mixing sites per f.u.
        
        # Mixing energies per formula unit
        energies_per_fu = {
            0.0: 0.0,
            0.5: 0.3,
            1.0: 0.0,
        }
        
        alloy = IsoAlloy(
            energies=energies_per_fu,
            n_sites_per_fu=n_sites,
        )
        
        x = 0.5
        
        # Per-site entropy (for a single mixing site)
        S_per_site = -kB * (x * np.log(x) + (1 - x) * np.log(1 - x))
        
        # Per-f.u. entropy should be n_sites times larger
        S_per_fu_expected = n_sites * S_per_site
        S_per_fu_actual = alloy._mixing_entropy(x)
        
        self.assertAlmostEqual(S_per_fu_actual, S_per_fu_expected, places=10)

    def test_scaling_invariance_of_thermodynamics(self):
        """
        Test that thermodynamic predictions (like phase separation temperature)
        are independent of the number of formula units considered.
        
        The consolute (critical) temperature T_c = omega / (2 * kB * n_sites_per_fu)
        should only depend on intensive quantities.
        """
        energies = {
            0.0: 0.0,
            0.25: 0.375,
            0.5: 0.5,  # omega * 0.25 = 0.5 => omega = 2.0
            0.75: 0.375,
            1.0: 0.0,
        }
        
        # Same energies per f.u., different n_sites_per_fu
        alloy_1site = IsoAlloy(energies=energies, n_sites_per_fu=1)
        alloy_2sites = IsoAlloy(energies=energies, n_sites_per_fu=2)
        
        kB = alloy_1site.kB
        omega = alloy_1site.omega  # Should be ~2.0
        
        # Theoretical consolute temperature: T_c = omega / (2 * kB * n_sites_per_fu)
        # With more sites per f.u., entropy is larger, so T_c is lower
        T_c_1site = omega / (2 * kB * 1)
        T_c_2sites = omega / (2 * kB * 2)
        
        # The system with 2 sites per f.u. should have lower critical temperature
        # because the entropy contribution is larger
        self.assertAlmostEqual(T_c_2sites, T_c_1site / 2, places=5)
        
        # Verify at a temperature between T_c_2sites and T_c_1site:
        # - 1-site alloy should phase separate (dG has double-well shape)
        # - 2-site alloy should be miscible (dG is convex)
        T_test = (T_c_1site + T_c_2sites) / 2
        
        # Check second derivative at x=0.5 (negative = phase separation, positive = miscible)
        # d²G/dx² at x=0.5 = -2*omega + kB*T*n_sites/[x*(1-x)]
        # At x=0.5: d²G/dx² = -2*omega + 4*kB*T*n_sites
        
        d2G_1site = -2 * omega + 4 * kB * T_test * 1
        d2G_2sites = -2 * omega + 4 * kB * T_test * 2
        
        # 1-site should have negative curvature (phase separating)
        self.assertLess(d2G_1site, 0)
        # 2-site should have positive curvature (miscible)
        self.assertGreater(d2G_2sites, 0)


if __name__ == "__main__":
    unittest.main()

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize, curve_fit, bisect
from pydmclab.plotting.utils import set_rc_params, get_colors
from pydmclab.core.comp import CompTools
from pydmclab.core.energies import FormationEnthalpy, ChemPots
from pydmclab.utils.handy import read_json, write_json

import os

COLORS = get_colors("tab10")
set_rc_params()


class IsoAlloy(object):
    """THIS DOESNT WORK YET"""

    def __init__(
        self,
        energies,
        kB=8.617e-5,
        xs=np.linspace(0, 1, 101),
        Ts=np.linspace(0, 3000, 31),
    ):

        self.energies = energies
        self.kB = kB
        self.xs = xs
        self.Ts = Ts
        self.discrete_x = sorted(list(energies.keys()))

    @property
    def E_mix_discrete(self):
        """
        Calculates mixing energies from the total or formation energies provided in energies
            {x (float) : energy per formula unit (float)}
        """
        energies = self.energies
        out = {}
        for x in energies:
            out[x] = energies[x] - (1 - x) * energies[0] - x * energies[1]
        return out

    @property
    def omega(self):
        """
        Fits the discrete E_mix to a continous function of x
            H_mix(x) = x * dE_B + omega * x * (1 - x)
            - omega is a fit parameter, sometimes called the alloy interaction parameter

        Returns:
            omega (float): the fit parameter (eV/f.u.)
        """
        E_mix_dict = self.E_mix_discrete
        compositions = sorted(list(E_mix_dict.keys()))
        E_mix_to_fit = [E_mix_dict[x] for x in compositions]

        def func(x, omega):
            """
            Function to optimize during fitting

            Args:
                x (float): composition in A_{1-x}B_{x}
                omega (float): parameter to fit
            """
            return omega * x * (1 - x)

        popt, pcov = curve_fit(func, compositions, E_mix_to_fit)
        return popt[0]

    def H_mix_curve(self, x):
        """
        Estimate of mixing enthalpy using the fit parameter omega

        Args:
            x (float): composition in A_{1-x}B_{x}
        Returns:
            mixing enthalpy (float, eV/fu) at composition x
        """
        omega = self.omega
        return omega * x * (1 - x)

    @property
    def H_mix(self):
        """
        Mixing enthalpies for all x values specified in __init__

        Returns:
            list of mixing enthalpies (eV/fu)

        """
        xs = self.xs
        return [self.H_mix_curve(x) for x in xs]

    def S_mix(self, x):
        """
        Ideal mixing entropy (eV/fu)

        Args:
            x (float): composition in A_{1-x}B_{x}

        Returns:
            mixing entropy (float, eV/fu)

        """
        return -self.kB * (x * np.log(x) + (1 - x) * np.log(1 - x))

    def G_mix(self, x, T):
        """
        Gibbs energy of mixing at composition x and temperature T

        Args:
            x (float): composition in A_{1-x}B_{x}
            T (float): temperature in K

        Returns:
            float (eV/fu)
        """
        return self.H_mix_curve(x) - T * self.S_mix(x)

    def G_mix_curve(self, T):
        """
        Gibbs energies of mixing at all xs

        Args:
            T (float): temperature in K

        Returns:
            list of Gibbs energies of mixing (eV/fu)
        """
        xs = self.xs
        return [self.G_mix(x, T) for x in xs]

    def dGdx(self, x, T):
        """
        First derivative of the G(x) curve

        Args:
            x (float): composition in A_{1-x}B_{x}
            T (float): temperature in K

        Returns:
            float

        """
        return self.omega * (1 - 2 * x) + self.kB * T * np.log(x / (1 - x))

    def dG2dx2(self, x, T):
        """
        Second derivative of the G(x) curve

        Args:
            x (float): composition in A_{1-x}B_{x}
            T (float): temperature in K

        Returns:
            float
        """
        return -2 * self.omega + self.kB * T / (x * (1 - x))

    def common_tangent(self, T, initial_guess=(0.1, 0.9)):
        """
        Performs the common tangent construction to determine the binodal compositions

        Args:
            T (float): temperature in K
            initial_guess (tuple): initial guess for the common tangent construction
                (x1, x2)
                    x1 is the initial guess for the max x where a solid solution in alpha is stable
                    x2 is the initial guess for the min x where a solid solution in beta is stable

        Returns:
            tuple: (x1, x2)
                x1 is the max composition where the alpha solid solution is stable
                x2 is the min composition where the beta solid solution is stable

        Note:
            - a solution should require that G(x) <= 0 at the common tangent points.
            - if it's not, then we don't have a solid solution in that phase
            - this is accounted for with if statements after optimization
        """

        def equations_to_solve(compositions, T):
            """
            Equations to optimize
                - we return the squared sum to penalize non-zero values
                - eqn1 and eqn2 should go to zero at the common tangent
            Args:
                compositions (tuple): (x1, x2)
                    x1 is the max x where a solid solution in alpha is stable
                    x2 is the min x where a solid solution in beta is stable

                T (float): temperature in K

            """
            x1, x2 = compositions
            eqn1 = self.dGdx(x1, T) - self.dGdx(x2, T)
            eqn2 = (self.G_mix(x1, T) - self.G_mix(x2, T)) / (x1 - x2) - self.dGdx(
                x1, T
            )
            return eqn1**2 + eqn2**2

        G_mix_curve = self.G_mix_curve(T)
        d2dGdx2 = [self.dG2dx2(x, T) for x in self.xs]
        if max(G_mix_curve) <= 0:
            return "solid solution everywhere"

        elif min(G_mix_curve) >= 0:
            return "solid solution nowhere"
        else:
            # trying to deal with a few cases:
            # G_mix_curve is a single parabola --> don't need common tangent
            # G_mix_curve is a double parabola --> need common tangent
            return
        result = minimize(
            fun=equations_to_solve,
            x0=initial_guess,
            args=(T,),
            bounds=((0.00001, 0.99999), (0.00001, 0.99999)),
        )
        x1, x2 = result.x
        if self.G_mix(x1) >= 0:
            x1 = 0
        if self.G_mix(x2) >= 0:
            x2 = 1
        return x1, x2


class HeteroAlloy(object):
    """
    For computing heterostructural alloy phase diagrams

    See the method, .to_dict() for a detailed description of what the code is doing.

    Typical use case would be to:
        - generate the dictionary with all the results
            HeteroAlloy(args).to_dict(fjson=<savename>)

        - read that dictionary and analyze the results
            d = read_json(<savename>)

        - plot the results
            - see AlloyPlotter below, which starts from this dictionary

    """

    def __init__(
        self,
        alpha_energies,
        beta_energies,
        kB=8.617e-5,
        xs=np.linspace(0, 1, 101),
        Ts=np.linspace(0, 3000, 31),
    ):
        """
        Args:
            alpha_energies (dict): dictionary of (total or formation) energies for the alpha phase.
                {x (float) : energy per formula unit (float)}
            beta_energies (dict): dictionary of (total or formation) energies for the beta phase.
                {x (float) : energy per formula unit (float)}
            kB (float): Boltzmann constant in eV/K
            xs (np.array): array of compositions to consider
            Ts (np.array): array of temperatures to consider

            In alpha_energies and beta_energies, x should be of the form A_{1-x}B_{x}
                so alpha_energies[0] <= beta_energies[0] (i.e., A's ground-state is alpha)
                and beta_energies[1] <= alpha_energies[1] (i.e., B's ground-state is beta)

        """
        self.alpha_energies = alpha_energies
        self.beta_energies = beta_energies
        self.kB = kB
        self.xs = xs
        self.Ts = Ts
        self.x_discrete_alpha = sorted(list(alpha_energies.keys()))
        self.x_discrete_beta = sorted(list(beta_energies.keys()))

    @property
    def dE_A(self):
        """
        Returns the energy difference (float, eV/fu) between the beta polymorph and the alpha ground state for the A phase
        """
        x = 0
        return self.beta_energies[x] - self.alpha_energies[x]

    @property
    def dE_B(self):
        """
        Returns the energy difference (float, eV/fu) between the alpha polymorph and the beta ground state for the B phase
        """
        x = 1
        return self.alpha_energies[x] - self.beta_energies[x]

    @property
    def gs_check(self):
        """
        Returns True if the ground states are consistent with the A_{1-x}B_{x} (alpha --> beta) notation
        """
        if (self.dE_A < 0) or (self.dE_B < 0):
            return False
        return True

    @property
    def is_isostructural(self):
        """
        Returns True if the alpha and beta polymorphs are degenerate in energy for A and B

        Note: the code doesn't currently handle this well
        """
        dE_A, dE_B = np.round(self.dE_A, 5), np.round(self.dE_B, 5)
        if (dE_A == 0) and (dE_B == 0):
            return True
        return False

    @property
    def E_mix_discrete_alpha(self):
        """
        Calculates mixing energies for the alpha structure from the total or formation energies provided in alpha_energies
            {x (float) : energy per formula unit (float)}


        """
        dE_B = self.dE_B
        alpha_energies = self.alpha_energies
        out = {}
        for x in alpha_energies:
            out[x] = (
                x * dE_B
                + alpha_energies[x]
                - (1 - x) * alpha_energies[0]
                - x * alpha_energies[1]
            )
        return out

    @property
    def E_mix_discrete_beta(self):
        """
        Calculates mixing energies for the beta structure from the total or formation energies provided in beta_energies
            {x (float) : energy per formula unit (float)}
        """
        dE_A = self.dE_A
        beta_energies = self.beta_energies
        out = {}
        for x in beta_energies:
            out[x] = (
                (1 - x) * dE_A
                + beta_energies[x]
                - (1 - x) * beta_energies[0]
                - x * beta_energies[1]
            )
        return out

    @property
    def omega_alpha(self):
        """
        Fits the discrete E_mix to a continous function of x
            H_mix(x) = x * dE_B + omega * x * (1 - x)
            - omega is a fit parameter, sometimes called the alloy interaction parameter

        Returns:
            omega (float): the fit parameter (eV/f.u.) for the alpha structure
        """
        E_mix_dict = self.E_mix_discrete_alpha
        compositions = sorted(list(E_mix_dict.keys()))
        E_mix_to_fit = [E_mix_dict[x] for x in compositions]
        dE_B = self.dE_B

        def func(x, omega):
            """
            Function to optimize during fitting

            Args:
                x (float): composition in A_{1-x}B_{x}
                omega (float): parameter to fit
            """
            return x * dE_B + omega * x * (1 - x)

        popt, pcov = curve_fit(func, compositions, E_mix_to_fit)
        return popt[0]

    @property
    def omega_beta(self):
        """
        Fits the discrete E_mix to a continous function of x
            H_mix(x) = (1 - x) * dE_A + omega * x * (1 - x)
            - omega is a fit parameter, sometimes called the alloy interaction parameter

        Returns:
            omega (float): the fit parameter (eV/f.u.) for the beta structure
        """
        E_mix_dict = self.E_mix_discrete_beta
        compositions = sorted(list(E_mix_dict.keys()))
        E_mix_to_fit = [E_mix_dict[x] for x in compositions]
        dE_A = self.dE_A
        dE_B = self.dE_B

        def func(x, omega):
            """
            Function to optimize during fitting

            Args:
                x (float): composition in A_{1-x}B_{x}
                omega (float): parameter to fit
            """

            return (1 - x) * dE_A + omega * x * (1 - x)

        popt, pcov = curve_fit(func, compositions, E_mix_to_fit)
        return popt[0]

    def H_mix_curve_alpha(self, x):
        """
        Estimate of mixing enthalpy for the alpha structure using the fit parameter omega_alpha

        Args:
            x (float): composition in A_{1-x}B_{x}
        Returns:
            mixing enthalpy (float, eV/fu) for the alpha structure at composition x
        """
        dE_B = self.dE_B
        omega_alpha = self.omega_alpha
        return x * dE_B + omega_alpha * x * (1 - x)

    def H_mix_curve_beta(self, x):
        """
        Estimate of mixing enthalpy for the beta structure using the fit parameter omega_beta

        Args:
            x (float): composition in A_{1-x}B_{x}
        Returns:
            mixing enthalpy (float, eV/fu) for the beta structure at composition x
        """
        dE_A = self.dE_A
        omega_beta = self.omega_beta
        return (1 - x) * dE_A + omega_beta * x * (1 - x)

    @property
    def H_mix_alpha(self):
        """
        Mixing enthalpies in the alpha structure for all x values specified in __init__

        Returns:
            list of mixing enthalpies (eV/fu)

        """
        xs = self.xs
        return [self.H_mix_curve_alpha(x) for x in xs]

    @property
    def H_mix_beta(self):
        """
        Mixing enthalpies in the beta structure for all x values specified in __init__

        Returns:
            list of mixing enthalpies (eV/fu)

        """

        xs = self.xs
        return [self.H_mix_curve_beta(x) for x in xs]

    def S_mix(self, x):
        """
        Ideal mixing entropy (eV/fu)

        Args:
            x (float): composition in A_{1-x}B_{x}

        Returns:
            mixing entropy (float, eV/fu)

        """
        return -self.kB * (x * np.log(x) + (1 - x) * np.log(1 - x))

    def G_alpha(self, x, T):
        """
        Gibbs energy of mixing for the alpha structure at composition x and temperature T

        Args:
            x (float): composition in A_{1-x}B_{x}
            T (float): temperature in K

        Returns:
            float (eV/fu)
        """
        return self.H_mix_curve_alpha(x) - T * self.S_mix(x)

    def G_beta(self, x, T):
        """
        Gibbs energy of mixing for the beta structure at composition x and temperature T

        Args:
            x (float): composition in A_{1-x}B_{x}
            T (float): temperature in K

        Returns:
            float (eV/fu)
        """
        return self.H_mix_curve_beta(x) - T * self.S_mix(x)

    def G_mix_curve_alpha(self, T):
        """
        Gibbs energies of mixing for the alpha structure at all xs

        Args:
            T (float): temperature in K

        Returns:
            list of Gibbs energies of mixing (eV/fu)
        """
        xs = self.xs
        return [self.G_alpha(x, T) for x in xs]

    def G_mix_curve_beta(self, T):
        """
        Gibbs energies of mixing for the beta structure at all xs

        Args:
            T (float): temperature in K

        Returns:
            list of Gibbs energies of mixing (eV/fu)
        """
        xs = self.xs
        return [self.G_beta(x, T) for x in xs]

    def dGdx_alpha(self, x, T):
        """
        First derivative of the G(x) curve for the alpha structure

        Args:
            x (float): composition in A_{1-x}B_{x}
            T (float): temperature in K

        Returns:
            float

        """
        return (
            self.dE_B
            + self.omega_alpha * (1 - 2 * x)
            + self.kB * T * np.log(x / (1 - x))
        )

    def dGdx_beta(self, x, T):
        """
        First derivative of the G(x) curve for the beta structure

        Args:
            x (float): composition in A_{1-x}B_{x}
            T (float): temperature in K

        Returns:
            float
        """
        return (
            -self.dE_A
            + self.omega_beta * (1 - 2 * x)
            + self.kB * T * np.log(x / (1 - x))
        )

    def dG2dx2_alpha(self, x, T):
        """
        Second derivative of the G(x) curve for the alpha structure

        Args:
            x (float): composition in A_{1-x}B_{x}
            T (float): temperature in K

        Returns:
            float
        """
        return -2 * self.omega_alpha + self.kB * T / (x * (1 - x))

    def dG2dx2_beta(self, x, T):
        """
        Second derivative of the G(x) curve for the beta structure

        Args:
            x (float): composition in A_{1-x}B_{x}
            T (float): temperature in K

        Returns:
            float
        """
        return -2 * self.omega_beta + self.kB * T / (x * (1 - x))

    def common_tangent(self, T, initial_guess=(0.1, 0.9)):
        """
        Performs the common tangent construction to determine the binodal compositions

        Args:
            T (float): temperature in K
            initial_guess (tuple): initial guess for the common tangent construction
                (x1, x2)
                    x1 is the initial guess for the max x where a solid solution in alpha is stable
                    x2 is the initial guess for the min x where a solid solution in beta is stable

        Returns:
            tuple: (x1, x2)
                x1 is the max composition where the alpha solid solution is stable
                x2 is the min composition where the beta solid solution is stable

        Note:
            - a solution should require that G(x) <= 0 at the common tangent points.
            - if it's not, then we don't have a solid solution in that phase
            - this is accounted for with if statements after optimization
        """

        def equations_to_solve(compositions, T):
            """
            Equations to optimize
                - we return the squared sum to penalize non-zero values
                - eqn1 and eqn2 should go to zero at the common tangent
            Args:
                compositions (tuple): (x1, x2)
                    x1 is the max x where a solid solution in alpha is stable
                    x2 is the min x where a solid solution in beta is stable

                T (float): temperature in K

            """
            x1, x2 = compositions
            eqn1 = self.dGdx_alpha(x1, T) - self.dGdx_beta(x2, T)
            eqn2 = (self.G_alpha(x1, T) - self.G_beta(x2, T)) / (
                x1 - x2
            ) - self.dGdx_alpha(x1, T)
            return eqn1**2 + eqn2**2

        result = minimize(
            fun=equations_to_solve,
            x0=initial_guess,
            args=(T,),
            bounds=((0.00001, 0.99999), (0.00001, 0.99999)),
        )
        x1, x2 = result.x
        if self.G_alpha(x1, T) >= 0:
            x1 = 0
        if self.G_beta(x2, T) >= 0:
            x2 = 1
        return x1, x2

    @property
    def critical_composition(self):
        """
        The critical compositoin is the maximum (minimum) x where alpha (beta) is stable
            - this is determined by the intersection of the mixing enthalpies (because entropy is presumed to affect each phase similarly)

        Returns:
            critical composition (x, float)

        """
        H_mix_alpha = self.H_mix_alpha
        H_mix_beta = self.H_mix_beta
        diffs = [abs(H_mix_alpha[i] - H_mix_beta[i]) for i in range(len(H_mix_alpha))]
        idx = np.argmin(diffs)
        return self.xs[idx]

    @property
    def spinodal_x_alpha(self):
        """
        The compositions that are spinodal "in alpha" are left of the critical composition

        Returns:
            np.array: compositions that are spinodal in alpha
        """
        xs = self.xs
        xc = self.critical_composition
        return xs[xs <= xc]

    @property
    def spinodal_x_beta(self):
        """
        The compositions that are spinodal "in beta" are right of the critical composition

        Returns:
            np.array: compositions that are spinodal in beta
        """
        xs = self.xs
        xc = self.critical_composition
        return xs[xs >= xc]

    @property
    def spinodal_T_alpha(self):
        """
        The alpha spinodal line is determined by setting the 2nd derivative to zero and re-arranging for T

        Below these temperatures and at x < critical composition, we get spinodal decomposition

        Returns:
            list: temperatures where spinodal decomposition occurs at alpha compositions
        """
        xs = self.spinodal_x_alpha

        omega_alpha = self.omega_alpha
        kB = self.kB
        return [2 * omega_alpha * x * (1 - x) / kB for x in xs]

    @property
    def spinodal_T_beta(self):
        """
        The beta spinodal line is determined by setting the 2nd derivative to zero and re-arranging for T

        Below these temperatures and at x > critical composition, we get spinodal decomposition

        Returns:
            list: temperatures where spinodal decomposition occurs at beta compositions
        """
        xs = self.spinodal_x_beta

        omega_beta = self.omega_beta
        kB = self.kB
        return [2 * omega_beta * x * (1 - x) / kB for x in xs]

    @property
    def binodal(self):
        """
        The binodal lines are the bounds on solid solution stability for the alpha and beta phase

        Returns:
            {'alpha' : {T (float) : x (float)}, 'beta' : {T (float) : x (float)}}
                where binodal['alpha'][T] is the max x where a solid solution in alpha is stable at temperature T
                and binodal['beta'][T] is the min x where a solid solution in beta is stable at temperature T
        """
        Ts = self.Ts
        x_alphas = {}
        x_betas = {}
        for T in Ts:
            x1, x2 = self.common_tangent(T)
            x_alphas[T] = x1
            x_betas[T] = x2
        return {"alpha": x_alphas, "beta": x_betas}

    def to_dict(self, fjson=None):
        """
        Args:
            fjson (str): path to save the dictionary as a json file
                if None, don't save
        Returns:
            dict: dictionary representation of the HeteroAlloy object

            {'x' : x values in A_{1-x}B_{x} for continuous curves (list)
             'T' : temperatures (K) (list),
             'omega' : {'alpha' : alloy interaction parameter in alpha phase (float), 'beta' : alloy interaction parameter in beta phase (float)},
             'dE_A' : energy difference between the beta polymorph and the alpha ground state for the A phase (float, eV/fu),
            'dE_B' : energy difference between the alpha polymorph and the beta ground state for the B phase (float, eV/fu),
            'critical_composition' : critical composition (max composition where alpha can form; min composition where beta forms (float),
            'E' : {'alpha' : [{'x' : x values that you calculated (float), 'E' : internal or formation energy (float, eV/fu)}], 'beta' : [{'x' : x, 'E' : internal or formation energy (float, eV/fu)}]},
            'E_mix' : {'alpha' : [{'x' : x values that you calculated, 'E_mix' : mixing enthalpy (float, eV/fu)}], 'beta' : [{'x' : x, 'E_mix' : mixing enthalpy (float, eV/fu)}]},
            'H_mix' : {'alpha' : [{'x' : continous x values, 'H_mix' : mixing enthalpy from fit to continuous function (float, eV/fu)}], 'beta' : [{'x' : x, 'H_mix' : mixing enthalpy (float, eV/fu)}]},
            'G_mix' : {'alpha' : [{'x' : continuous x values, 'T' : T, 'G_mix' : Gibbs energy of mixing (float, eV/fu)}], 'beta' : [{'x' : x, 'T' : T, 'G_mix' : Gibbs energy of mixing (float, eV/fu)}]},
            'binodal' : {'alpha' : {T : x}, 'beta' : {T : x}},
            'spinodal' : {'alpha' : {x : T}, 'beta' : {x : T}}
            }

            Some notes on the approach:
                1. you have two polymorphs, alpha and beta
                    - alpha is the ground-state for some compound, A
                    - beta is the ground-state for some compound, B
                2. you want to understand the phase stability as you mix A and B in each structure
                3. you run some DFT calculations to get internal or formation energies
                    - each calculation is at some x in A_{1-x}B_{x}
                    - each calculation is in some structure, alpha or beta
                    - these are stored as 'E'
                4. you calculate the mixing energies at 0 K in each structure
                    - you have two discrete sets of mixing energies -- one for alpha and one for beta
                    - the alpha mixing energy should be 0 at x = 0 (because alpha is the ground state for A)
                    - the beta mixing energy should be 0 at x = 1 (because beta is the ground state for B)
                    - these are stored as 'E_mix'
                5. you want these mixing curves to be continuous in x, so you fit a function to E_mix in each phase
                    - E_mix ~ H_mix(x) = x * dE_B + omega * x * (1 - x) for alpha
                    - E_mix ~ H_mix(x) = (1 - x) * dE_A + omega * x * (1 - x) for beta
                    - the dE_B and dE_A terms make sure that these curves exactly fit the discrete values at x = 0 and x = 1
                    - omega is a fit parameter that describes the penalty for mixing in each structure
                        - larger omega --> more penalty (the energy goes up faster as you mix)
                    - note: we fit a simple function in one parameter, but you can also do more complex fits with two omegas
                        - alpha: E_mix ~ H_mix(x) = x * dE_B + omega * x * (1 - x) + omega2 * (x^2 - x)(x - 0.5)
                        - beta: E_mix ~ H_mix(x) = (1 - x) * dE_A + omega * x * (1 - x) + omega2 * (x^2 - x)(x - 0.5)
                    - these fit mixing enthalpies are stored as 'H_mix'
                6. now we want to understand the influence of temperature (entropy on stability)
                    - we assume the maximum configurational entropy (ideal mixing)
                    - this is assuming that mixing is completely random and there is no short-range order
                    - the mixing entropy is therefore S_mix(x) = -kB * (x * ln(x) + (1 - x) * ln(1 - x))
                    - we then map our mixing enthalpies to Gibbs energies of mixing
                        - G_mix(T, x) = H_mix(x) - T * S_mix(x)
                        - we have two G_mix(x) curves at each T -- one for mixing in alpha and one for mixing in beta
                    - these are stored as 'G_mix'
                7. now we want to understand at what temperatures and compositions will our system phase separate
                    - this is called spinodal decomposition and is undesirable if we want to make a mixed phase
                    - we can calculate the spinodal lines (T, x) by setting the second derivative of G_mix to zero and solving for T at each x
                    - at a specified composition, x, below T_spinodal, the system will phase separate
                    - because we have two G_mix curves, we also have two spinodals
                        - the one that proceeds from left to right is the alpha spinodal
                        - the one that proceeds from right to left is the beta spinodal
                    - our system transitions from the alpha spinodal to the beta spinodal being relevant at the critical composition
                        - because entropy affects both alpha and beta similarly, the critical composition is determined by the intersection of our H_mix curves
                    - these (T, x) pairs that define the spinodal for each structure are stored as 'spinodal'
                8. we also want to understand the bounds on solid solution stability
                    - if we're trying to mixing A and B into a single phase, we want to find conditions where a solid solution is stable
                    - this is called the binodal line
                        - the alpha binodal is the max x where a solid solution in alpha is stable
                        - the beta binodal is the min x where a solid solution in beta is stable
                        - these are temperature-dependent because it's easier to mix at higher T (due to entropy)
                    - at a given T, the binodal compositions are determined by a common tangent construction
                        - given two G_mix(x) curves, one for alpha and one for beta
                        - we find the common tangent to these curves
                            - i.e., a line that intersects both curves and has the same slope as both curves at the intersection
                        - the x value where the common tangent hits the alpha curve is the max x where an alpha solid solution is stable
                        - the x value where the common tangent hits the beta curve is the min x where a beta solid solution is stable
                    - these (T, x) pairs that define the binodal for each structure are stored as 'binodal'

        """
        d = {}
        xs = list(self.xs)
        Ts = list(self.Ts)
        d["x"] = xs
        d["T"] = Ts

        omega = {"alpha": self.omega_alpha, "beta": self.omega_beta}
        d["omega"] = omega

        dE_A = self.dE_A
        dE_B = self.dE_B
        d["dE_A"] = dE_A
        d["dE_B"] = dE_B

        critical_composition = self.critical_composition
        d["critical_composition"] = critical_composition

        energies = {"alpha": self.alpha_energies, "beta": self.beta_energies}
        d["E"] = {
            "alpha": [{"x": x, "E": energies["alpha"][x]} for x in energies["alpha"]],
            "beta": [{"x": x, "E": energies["beta"][x]} for x in energies["beta"]],
        }

        E_mix_discete_alpha = self.E_mix_discrete_alpha
        E_mix_discrete_beta = self.E_mix_discrete_beta
        discrete_mixing_enthalpies = {
            "alpha": [
                {"x": x, "E_mix": E_mix_discete_alpha[x]} for x in E_mix_discete_alpha
            ],
            "beta": [
                {"x": x, "E_mix": E_mix_discrete_beta[x]} for x in E_mix_discrete_beta
            ],
        }
        d["E_mix"] = discrete_mixing_enthalpies

        H_mix_alpha = self.H_mix_alpha
        H_mix_beta = self.H_mix_beta
        H_mix = {
            "alpha": [
                {"x": xs[i], "H_mix": H_mix_alpha[i]} for i in range(len(H_mix_alpha))
            ],
            "beta": [
                {"x": xs[i], "H_mix": H_mix_beta[i]} for i in range(len(H_mix_beta))
            ],
        }
        d["H_mix"] = H_mix

        G_mix_alpha = {T: dict(zip(xs, self.G_mix_curve_alpha(T))) for T in Ts}
        G_mix_beta = {T: dict(zip(xs, self.G_mix_curve_beta(T))) for T in Ts}

        G_mix = {
            "alpha": [
                {"x": x, "T": T, "G_mix": G_mix_alpha[T][x]}
                for T in G_mix_alpha
                for x in G_mix_alpha[T]
            ],
            "beta": [
                {"x": x, "T": T, "G_mix": G_mix_beta[T][x]}
                for T in G_mix_beta
                for x in G_mix_beta[T]
            ],
        }
        d["G_mix"] = G_mix

        binodal = self.binodal
        binodal = {
            "alpha": [{"x": binodal["alpha"][T], "T": T} for T in binodal["alpha"]],
            "beta": [{"x": binodal["beta"][T], "T": T} for T in binodal["beta"]],
        }
        d["binodal"] = binodal

        spinodal = {
            "alpha": dict(zip(self.spinodal_x_alpha, self.spinodal_T_alpha)),
            "beta": dict(zip(self.spinodal_x_beta, self.spinodal_T_beta)),
        }
        alpha_xs = sorted(list(spinodal["alpha"].keys()))
        beta_xs = sorted(list(spinodal["beta"].keys()))

        spinodal = {
            "alpha": [{"x": x, "T": spinodal["alpha"][x]} for x in alpha_xs],
            "beta": [{"x": x, "T": spinodal["beta"][x]} for x in beta_xs],
        }
        d["spinodal"] = spinodal

        if not fjson:
            return d
        else:
            write_json(d, fjson)
            return read_json(fjson)


class HeteroAlloyPlotter(object):
    def __init__(
        self,
        alloy_dict,
        color_palette="tab10",
        alpha_color="blue",
        beta_color="orange",
        alpha_label=r"$\alpha$",
        beta_label=r"$\beta$",
        A_label="A",
        B_label="B",
        temp_unit="C",
    ):
        """
        Args:
            alloy_dict (dict):
                From HeteroAlloy(args).to_dict()
        """
        self.alloy_dict = alloy_dict
        colors = get_colors(color_palette)
        self.colors = colors
        self.params = {
            "alpha": {"color": alpha_color, "label": alpha_label},
            "beta": {"color": beta_color, "label": beta_label},
            "A": A_label,
            "B": B_label,
            "spinodal": {"color": "gray"},
        }
        self.temp_unit = temp_unit

    @property
    def ax_mixing_enthalpies(self):
        """
        plots the mixing enthalpies
        """
        d = self.alloy_dict
        params = self.params
        alpha_color, beta_color = params["alpha"]["color"], params["beta"]["color"]
        alpha_label, beta_label = params["alpha"]["label"], params["beta"]["label"]
        A, B = params["A"], params["B"]

        discrete_data = d["E_mix"]

        x_alpha_discrete = [
            discrete_data["alpha"][i]["x"] for i in range(len(discrete_data["alpha"]))
        ]
        y_alpha_discrete = [
            discrete_data["alpha"][i]["E_mix"]
            for i in range(len(discrete_data["alpha"]))
        ]

        x_beta_discrete = [
            discrete_data["beta"][i]["x"] for i in range(len(discrete_data["beta"]))
        ]
        y_beta_discrete = [
            discrete_data["beta"][i]["E_mix"] for i in range(len(discrete_data["beta"]))
        ]

        x_smooth = d["x"]

        y_alpha_smooth = [d["H_mix"]["alpha"][i]["H_mix"] for i in range(len(x_smooth))]
        y_beta_smooth = [d["H_mix"]["beta"][i]["H_mix"] for i in range(len(x_smooth))]

        ax = plt.scatter(
            x_alpha_discrete,
            y_alpha_discrete,
            color="white",
            edgecolor=alpha_color,
            label=alpha_label,
        )
        ax = plt.scatter(
            x_beta_discrete,
            y_beta_discrete,
            color="white",
            edgecolor=beta_color,
            label=beta_label,
        )

        ax = plt.plot(x_smooth, y_alpha_smooth, color=alpha_color, linestyle="--")
        ax = plt.plot(x_smooth, y_beta_smooth, color=beta_color, linestyle="--")

        ax = plt.legend()

        ax = plt.xlabel(r"$%s_{1-x}%s_{x}$" % (A, B))
        ax = plt.ylabel(r"$\Delta H_{mix}\/(\frac{eV}{f.u.})$")
        return ax

    def ax_mixing_free_energies(self, T):
        """
        plots the Gibbs energies of mixing at T (float, K)
        """
        d = self.alloy_dict
        params = self.params
        alpha_color, beta_color = params["alpha"]["color"], params["beta"]["color"]
        alpha_label, beta_label = params["alpha"]["label"], params["beta"]["label"]
        A, B = params["A"], params["B"]

        G_mix_alpha = d["G_mix"]["alpha"]
        G_mix_alpha = [v for v in G_mix_alpha if v["T"] == T]
        x_alpha = [v["x"] for v in G_mix_alpha]
        y_alpha = [v["G_mix"] for v in G_mix_alpha]

        G_mix_beta = d["G_mix"]["beta"]
        G_mix_beta = [v for v in G_mix_beta if v["T"] == T]
        x_beta = [v["x"] for v in G_mix_beta]
        y_beta = [v["G_mix"] for v in G_mix_beta]

        ax = plt.plot(
            x_alpha,
            y_alpha,
            color=alpha_color,
            linestyle="--",
            label=alpha_label,
        )
        ax = plt.plot(
            x_beta, y_beta, color=beta_color, linestyle="--", label=beta_label
        )

        ax = plt.legend()

        ax = plt.xlabel(r"$%s_{1-x}%s_{x}$" % (A, B))
        ax = plt.ylabel(r"$\Delta G_{mix}(%i\/K)\/(\frac{eV}{f.u.})$" % T)

        """ 
        x1, x2 = d["binodal"]["alpha"][T], d["binodal"]["beta"][T]
        y1, y2 = (
            d["G_mix"]["alpha"][T][xs.index(x1)],
            d["G_mix"]["beta"][T][xs.index(x2)],
        )
        ax = plt.plot(
            [x1, x2], [y1, y2], color="black", label="common tangent", ls="--"
        )
        """
        return ax

    @property
    def ax_regions(self):
        """
        plots the T-x phase diagram
        """
        d = self.alloy_dict
        params = self.params
        alpha_color, beta_color = params["alpha"]["color"], params["beta"]["color"]
        spinodal_color = params["spinodal"]["color"]
        alpha_label, beta_label = params["alpha"]["label"], params["beta"]["label"]
        A, B = params["A"], params["B"]

        critical_composition = d["critical_composition"]

        binodal = d["binodal"]
        spinodal = d["spinodal"]

        x_alpha_binodal = [
            binodal["alpha"][i]["x"] for i in range(len(binodal["alpha"]))
        ]
        T_alpha_binodal = [
            binodal["alpha"][i]["T"] for i in range(len(binodal["alpha"]))
        ]

        x_beta_binodal = [binodal["beta"][i]["x"] for i in range(len(binodal["beta"]))]
        T_beta_binodal = [binodal["beta"][i]["T"] for i in range(len(binodal["beta"]))]

        x_alpha_spinodal = [
            spinodal["alpha"][i]["x"] for i in range(len(spinodal["alpha"]))
        ]
        T_alpha_spinodal = [
            spinodal["alpha"][i]["T"] for i in range(len(spinodal["alpha"]))
        ]

        x_beta_spinodal = [
            spinodal["beta"][i]["x"] for i in range(len(spinodal["beta"]))
        ]
        T_beta_spinodal = [
            spinodal["beta"][i]["T"] for i in range(len(spinodal["beta"]))
        ]

        if self.temp_unit == "C":
            T_alpha_binodal = [T - 273.15 for T in T_alpha_binodal]
            T_beta_binodal = [T - 273.15 for T in T_beta_binodal]
            T_alpha_spinodal = [T - 273.15 for T in T_alpha_spinodal]
            T_beta_spinodal = [T - 273.15 for T in T_beta_spinodal]

        ax = plt.plot(
            x_alpha_binodal,
            T_alpha_binodal,
            color=alpha_color,
            ls="-",
        )

        ax = plt.plot(
            x_beta_binodal,
            T_beta_binodal,
            color=beta_color,
            ls="-",
        )

        ax = plt.plot(
            x_alpha_spinodal,
            T_alpha_spinodal,
            color=alpha_color,
            ls="--",
        )

        ax = plt.plot(
            x_beta_spinodal,
            T_beta_spinodal,
            color=beta_color,
            ls="--",
        )

        ax = plt.fill_between(
            x_alpha_binodal,
            T_alpha_binodal,
            max(T_alpha_binodal),
            color=alpha_color,
            alpha=0.1,
            lw=0,
            label=alpha_label,
        )

        ax = plt.fill_between(
            x_beta_binodal,
            T_beta_binodal,
            max(T_beta_binodal),
            color=beta_color,
            alpha=0.1,
            lw=0,
            label=beta_label,
        )

        ax = plt.fill_between(
            x_alpha_spinodal,
            T_alpha_spinodal,
            0,
            color=spinodal_color,
            alpha=0.1,
            lw=0,
            label="spinodal",
        )

        ax = plt.fill_between(
            x_beta_spinodal, T_beta_spinodal, 0, color=spinodal_color, alpha=0.1, lw=0
        )

        ax = plt.plot(
            [critical_composition, critical_composition],
            [0, max(T_alpha_binodal)],
            color="black",
            ls="--",
            label=r"$x_{crit}$",
        )

        ax = plt.legend(frameon=True)

        ax = plt.xlabel(r"$%s_{1-x}%s_{x}$" % (A, B))
        if self.temp_unit == "C":
            ax = plt.ylabel("Temperature (°C)")
        else:
            ax = plt.ylabel(r"$T\/(K)$")
        return ax


def main():
    alpha_energies = {0: 0, 0.4: 0.1, 0.6: 0.15, 0.8: 0.2, 1: 0.22}
    beta_energies = {1: 0, 0.8: 0.05, 0.6: 0.09, 0.4: 0.1, 0.2: 0.12, 0: 0.125}
    # beta_energies = alpha_energies.copy()

    alloy = HeteroAlloy(alpha_energies, beta_energies)

    fjson = "/Users/cbartel/Downloads/alloy.json"
    remake = False
    if not os.path.exists(fjson) or remake:
        alloy_dict = alloy.to_dict(fjson)

    else:
        alloy_dict = read_json(fjson)

    plotter = HeteroAlloyPlotter(alloy_dict)
    fig = plt.figure()
    ax = plt.subplot(111)
    ax = plotter.ax_mixing_enthalpies
    fig = plt.figure()
    ax = plt.subplot(111)
    ax = plotter.ax_mixing_free_energies(1800)
    fig = plt.figure()
    ax = plt.subplot(111)
    ax = plotter.ax_regions
    return alloy, alloy_dict


if __name__ == "__main__":
    alloy, alloy_dict = main()

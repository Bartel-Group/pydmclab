import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize, curve_fit, bisect
from pydmclab.plotting.utils import set_rc_params, get_colors
from pydmclab.core.comp import CompTools
from pydmclab.core.energies import FormationEnthalpy, ChemPots

COLORS = get_colors("tab10")
set_rc_params()


class AlloyThermo(object):

    def __init__(self, alpha_energies, beta_energies, A, B):
        """
        Args:
            alpha_energies (dict): Dictionary of energies for the alpha phase.
                {x (float) : energy per formula unit (float)}
            beta_energies (dict): Dictionary of energies for the beta phase.
                {x (float) : energy per formula unit (float)}
            A (formula): formula of compound whose ground-state is alpha
            B (formula): formula of compound whose ground-state is beta

            In alpha_energies and beta_energies, x should be of the form A_{1-x}B_{x}
                so alpha_energies[0] <= beta_energies[0]
                and beta_energies[1] <= alpha_energies[1]

        """
        self.alpha_energies = alpha_energies
        self.beta_energies = beta_energies
        self.A = A
        self.B = B
        self.kB = 8.617e-5  # eV/K
        self.xs = np.linspace(0, 1, 101)
        self.Ts = np.linspace(0, 3030, 30)
        self.x_discrete_alpha = sorted(list(alpha_energies.keys()))
        self.x_discrete_beta = sorted(list(beta_energies.keys()))

    @property
    def dE_A(self):
        x = 0
        return self.beta_energies[x] - self.alpha_energies[x]

    @property
    def dE_B(self):
        x = 1
        return self.alpha_energies[x] - self.beta_energies[x]

    @property
    def gs_check(self):
        if (self.dE_A < 0) or (self.dE_B < 0):
            return False
        return True

    @property
    def is_isostructural(self):
        dE_A, dE_B = np.round(self.dE_A, 5), np.round(self.dE_B, 5)
        if (dE_A == 0) and (dE_B == 0):
            return True
        return False

    @property
    def E_mixes_alpha_for_fitting(self):
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
    def E_mixes_beta_for_fitting(self):
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

    def dH_mix_curve_alpha(self, x):
        dE_B = self.dE_B
        omega_alpha = self.omega_alpha
        return x * dE_B + omega_alpha * x * (1 - x)

    def dH_mix_curve_beta(self, x):
        dE_A = self.dE_A
        omega_beta = self.omega_beta
        return (1 - x) * dE_A + omega_beta * x * (1 - x)

    @property
    def omega_alpha(self):
        E_mix_dict = self.E_mixes_alpha_for_fitting
        compositions = sorted(list(E_mix_dict.keys()))
        E_mix_to_fit = [E_mix_dict[x] for x in compositions]
        dE_A = self.dE_A
        dE_B = self.dE_B

        def func(x, omega):

            return x * dE_B + omega * x * (1 - x)

        popt, pcov = curve_fit(func, compositions, E_mix_to_fit)
        return popt[0]

    @property
    def omega_beta(self):
        E_mix_dict = self.E_mixes_beta_for_fitting
        compositions = sorted(list(E_mix_dict.keys()))
        E_mix_to_fit = [E_mix_dict[x] for x in compositions]
        dE_A = self.dE_A
        dE_B = self.dE_B

        def func(x, omega):

            return (1 - x) * dE_A + omega * x * (1 - x)

        popt, pcov = curve_fit(func, compositions, E_mix_to_fit)
        return popt[0]

    @property
    def dH_mix_alpha(self):
        xs = self.xs
        return [self.dH_mix_curve_alpha(x) for x in xs]

    @property
    def dH_mix_beta(self):

        xs = self.xs
        return [self.dH_mix_curve_beta(x) for x in xs]

    def entropy(self, x):
        return -self.kB * (x * np.log(x) + (1 - x) * np.log(1 - x))

    def G_alpha(self, x, T):
        return self.dH_mix_curve_alpha(x) - T * self.entropy(x)

    def G_beta(self, x, T):
        return self.dH_mix_curve_beta(x) - T * self.entropy(x)

    def dG_mix_curve_alpha(self, T):
        xs = self.xs
        return [self.G_alpha(x, T) for x in xs]

    def dG_mix_curve_beta(self, T):
        xs = self.xs
        return [self.G_beta(x, T) for x in xs]

    def dGdx_alpha(self, x, T):
        return (
            self.dE_B
            + self.omega_alpha * (1 - 2 * x)
            + self.kB * T * np.log(x / (1 - x))
        )

    def dGdx_beta(self, x, T):
        return (
            -self.dE_A
            + self.omega_beta * (1 - 2 * x)
            + self.kB * T * np.log(x / (1 - x))
        )

    def dG2dx2_alpha(self, x, T):
        return -2 * self.omega_alpha + self.kB * T / (x * (1 - x))

    def dG2dx2_beta(self, x, T):
        return -2 * self.omega_beta + self.kB * T / (x * (1 - x))

    def common_tangent(self, T, initial_guess=(0.1, 0.9)):
        def equations_to_solve(p, T):
            x1, x2 = p
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
        dH_mix_alpha = self.dH_mix_alpha
        dH_mix_beta = self.dH_mix_beta
        diffs = [
            abs(dH_mix_alpha[i] - dH_mix_beta[i]) for i in range(len(dH_mix_alpha))
        ]
        idx = np.argmin(diffs)
        return self.xs[idx]

    @property
    def spinodal_x_alpha(self):
        xs = self.xs
        xc = self.critical_composition
        return xs[xs <= xc]

    @property
    def spinodal_x_beta(self):
        xs = self.xs
        xc = self.critical_composition
        return xs[xs >= xc]

    @property
    def spinodal_T_alpha(self):
        xs = self.spinodal_x_alpha

        omega_alpha = self.omega_alpha
        kB = self.kB
        return [2 * omega_alpha * x * (1 - x) / kB for x in xs]

    @property
    def spinodal_T_beta(self):
        xs = self.spinodal_x_beta

        omega_beta = self.omega_beta
        kB = self.kB
        return [2 * omega_beta * x * (1 - x) / kB for x in xs]

    @property
    def binodal(self):
        Ts = self.Ts
        x_alphas = {}
        x_betas = {}
        for T in Ts:
            x1, x2 = self.common_tangent(T)
            x_alphas[T] = x1
            x_betas[T] = x2
        return {"alpha": x_alphas, "beta": x_betas}

    @property
    def ax_dHmixes(self):
        alpha_color = COLORS["blue"]
        beta_color = COLORS["orange"]
        xs = self.xs
        y_alpha_smooth = self.dH_mix_alpha
        y_beta_smooth = self.dH_mix_beta

        x_discrete_alpha = self.x_discrete_alpha
        alpha_energies = self.alpha_energies
        y_alpha_discrete = [alpha_energies[x] for x in x_discrete_alpha]

        x_discrete_beta = self.x_discrete_beta
        beta_energies = self.beta_energies
        y_beta_discrete = [beta_energies[x] for x in x_discrete_beta]

        ax = plt.scatter(
            x_discrete_alpha,
            y_alpha_discrete,
            color="white",
            edgecolor=alpha_color,
            label="alpha",
        )
        ax = plt.scatter(
            x_discrete_beta,
            y_beta_discrete,
            color="white",
            edgecolor=beta_color,
            label="beta",
        )

        ax = plt.plot(xs, y_alpha_smooth, color=alpha_color, linestyle="--")
        ax = plt.plot(xs, y_beta_smooth, color=beta_color, linestyle="--")

        ax = plt.legend()

        A = self.A
        B = self.B

        ax = plt.xlabel(r"$%s_{1-x}%s_{x}$" % (A, B))
        ax = plt.ylabel(r"$\Delta H_{mix}\/(\frac{eV}{f.u.})$")
        return ax

    def ax_dGmix(self, T):
        alpha_color = COLORS["blue"]
        beta_color = COLORS["orange"]
        xs = self.xs
        y_alpha_smooth = self.dG_mix_curve_alpha(T)
        y_beta_smooth = self.dG_mix_curve_beta(T)

        ax = plt.plot(
            xs, y_alpha_smooth, color=alpha_color, linestyle="--", label="alpha"
        )
        ax = plt.plot(xs, y_beta_smooth, color=beta_color, linestyle="--", label="beta")

        ax = plt.legend()

        A = self.A
        B = self.B

        ax = plt.xlabel(r"$%s_{1-x}%s_{x}$" % (A, B))
        ax = plt.ylabel(r"$\Delta G_{mix}\/(\frac{eV}{f.u.})$")

        x1, x2 = self.common_tangent(T)
        y1, y2 = self.G_alpha(x1, T), self.G_beta(x2, T)
        ax = plt.plot(
            [x1, x2], [y1, y2], color="black", label="common tangent", ls="--"
        )
        return ax

    @property
    def ax_regions(self):
        alpha_color = COLORS["blue"]
        beta_color = COLORS["orange"]
        spinodal_color = COLORS["red"]
        Ts = self.Ts
        binodal = self.binodal

        ax = plt.plot(
            self.spinodal_x_alpha,
            self.spinodal_T_alpha,
            color=spinodal_color,
            label="spinodal",
        )

        ax = plt.plot(self.spinodal_x_beta, self.spinodal_T_beta, color=spinodal_color)

        ax = plt.plot(
            [binodal["alpha"][T] for T in Ts], Ts, color=alpha_color, label="alpha"
        )
        ax = plt.plot(
            [binodal["beta"][T] for T in Ts], Ts, color=beta_color, label="beta"
        )
        ax = plt.legend()

        ax = plt.xlabel(r"$%s_{1-x}%s_{x}$" % (self.A, self.B))
        ax = plt.ylabel(r"$T\/(K)$")
        return ax


def main():
    alpha_energies = {0: 0, 0.4: 0.1, 0.6: 0.15, 0.8: 0.2, 1: 0.22}
    beta_energies = {1: 0, 0.8: 0.05, 0.6: 0.09, 0.4: 0.1, 0.2: 0.12, 0: 0.125}
    A = "Al"
    B = "Fe"
    alloy = AlloyThermo(alpha_energies, beta_energies, A, B)

    """
    fig = plt.figure()
    ax = plt.subplot(111)
    # ax = alloy.ax_dHmixes
    T = 0
    ax = alloy.ax_dGmix(T)
    # print(alloy.common_tangent(2000))
    """
    fig = plt.figure()
    ax = plt.subplot(111)
    ax = alloy.ax_regions

    return alloy


if __name__ == "__main__":
    alloy = main()

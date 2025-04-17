from __future__ import annotations
from pymatgen.core.structure import Structure
import numpy as np
from scipy.ndimage import gaussian_filter1d
from matplotlib import pyplot as plt
import matplotlib.animation as animation
from pydmclab.core.struc import StrucTools
from pydmclab.plotting.utils import set_rc_params, get_colors

set_rc_params()

WAVELENGTHS = {
    "CuKa": 1.54184,
    "CuKa2": 1.54439,
    "CuKa1": 1.54056,
    "CuKb1": 1.39222,
    "MoKa": 0.71073,
    "MoKa2": 0.71359,
    "MoKa1": 0.70930,
    "MoKb1": 0.63229,
    "CrKa": 2.29100,
    "CrKa2": 2.29361,
    "CrKa1": 2.28970,
    "CrKb1": 2.08487,
    "FeKa": 1.93735,
    "FeKa2": 1.93998,
    "FeKa1": 1.93604,
    "FeKb1": 1.75661,
    "CoKa": 1.79026,
    "CoKa2": 1.79285,
    "CoKa1": 1.78896,
    "CoKb1": 1.63079,
    "AgKa": 0.560885,
    "AgKa2": 0.563813,
    "AgKa1": 0.559421,
    "AgKb1": 0.497082,
}


class PlotXRD(object):
    """
    Used to plot XRD patterns.
    """

    def __init__(
        self,
        xrd_data: (
            Structure
            | tuple[Structure]
            | dict
            | tuple[dict]
            | str
            | tuple[str]
            | tuple[np.ndarray, np.ndarray]
            | tuple[tuple[np.ndarray, np.ndarray], ...]
        ),
        wavelength: str | float = "CuKa",
    ) -> None:
        """
        Args:
            xrd_data: input data for XRD patterns
                - Structure object or Structure as dict or path to file
                - tuple of Structure objects or Structures as dicts or paths to files
                - tuple of (two-theta, intensity) data for a single XRD pattern
                - tuple of tuples of (two-theta, intensity) data for multiple XRD patterns
            wavelength: wavelength of X-ray radiation, the default is "CuKa"
        """
        if isinstance(wavelength, str):
            self.wavelength = WAVELENGTHS[wavelength]
        else:
            self.wavelength = wavelength

        if isinstance(xrd_data, (Structure, dict, str)):
            self.xrd_data = [self.compute_xrd(xrd_data)]
        elif isinstance(xrd_data, tuple) and xrd_data:
            if isinstance(xrd_data[0], (Structure, dict, str)):
                self.xrd_data = [self.compute_xrd(struc) for struc in xrd_data]
            elif isinstance(xrd_data[0], np.ndarray):
                self.xrd_data = [
                    xrd_data,
                ]
            elif isinstance(xrd_data[0], tuple):
                self.xrd_data = list(xrd_data)
        else:
            raise ValueError("Invalid input data for XRD patterns")

    def compute_xrd(
        self, struc: Structure | dict | str
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Args:
            structure: pymatgen Structure object or Structure as dict or path to file

        Returns:
            tuple of (two-theta, intensity) data for the XRD pattern
        """
        xrd_pattern = StrucTools(struc).get_xrd_pattern(wavelength=self.wavelength)
        return (xrd_pattern["two_thetas"], xrd_pattern["intensities"])

    def simulated_plot(
        self,
        pattern_to_plot: int = 0,
        broadened: bool = False,
        savename: str | None = None,
        show: bool = False,
        **kwargs,
    ) -> tuple[plt.Figure, plt.Axes]:
        """
        Args:
            pattern_to_plot: index of the XRD pattern to plot
            broadened: whether to plot the broadened XRD pattern
            savename: save the plot to specified file location if not None
            show: whether to display the plot
            kwargs: additional keyword arguments for customizing the plot
                - color (tab10), lw, xlim, ylim, xticks, yticks, xlabel, ylabel, etc.
                - for broadened plot: tau, mesh_steps
        Return:
            fig, ax: figure and axis objects of the plot
        """
        if np.abs(pattern_to_plot) >= len(self.xrd_data):
            raise ValueError("Invalid index for XRD pattern, out of range")

        tt, i = self.xrd_data[pattern_to_plot]

        fig, ax = plt.subplots(figsize=kwargs.get("figsize", (8, 6)))

        colors = get_colors("tab10")
        color = kwargs.get("color", "red")
        lw = kwargs.get("lw", 2)

        if not broadened:
            ax.vlines(tt, [0], i, color=colors[color], lw=lw)
        else:
            tt, i = self.broaden_spectrum(tt, i, **kwargs)
            ax.plot(tt, i, color=colors[color], lw=lw)

        ax.set_xlim(kwargs.get("xlim", (10, 80)))
        ax.set_ylim(kwargs.get("ylim", (0, 105)))
        xticks = kwargs.get("xticks", np.arange(10, 81, 10))
        if xticks is not None:
            ax.set_xticks(xticks)
        yticks = kwargs.get("yticks", np.arange(0, 101, 10))
        if yticks is not None:
            ax.set_yticks(yticks)
        ax.set_xlabel(kwargs.get("xlabel", "2θ (degrees)"))
        ax.set_ylabel(kwargs.get("ylabel", "Intensity (arb. unit)"))

        if kwargs.get("grid", True):
            ax.grid(True)

        if savename is not None:
            fig.savefig(savename)

        if show:
            plt.show()

        return fig, ax

    def all_simulated_plots(
        self,
        savenames: tuple[str, ...],
        broadened: bool = False,
        show: bool = False,
        **kwargs,
    ) -> None:
        """
        Args:
            savenames: tuple of file locations to save the plots
            broadened: whether to plot the broadened XRD pattern
            show: whether to display the plots
            kwargs: additional keyword arguments for customizing the plots
                - color (tab10), lw, xlim, ylim, xticks, yticks, xlabel, ylabel, etc.
        """
        if len(savenames) != len(self.xrd_data):
            raise ValueError(
                "Provided number of file locations does not match the number of XRD patterns"
            )

        for i, savename in enumerate(savenames):
            self.simulated_plot(
                i, savename=savename, broadened=broadened, show=show, **kwargs
            )

    def overlay_simulated_plots(
        self,
        patterns_to_plot: tuple[int],
        broadened: bool = False,
        savename: str | None = None,
        show: bool = False,
        **kwargs,
    ) -> tuple[plt.Figure, plt.Axes]:
        """
        Args:
            patterns_to_plot: indices of the XRD patterns to overlay
            broadened: whether to plot the broadened XRD pattern
            savename: save the plot to specified file location if not None
            show: whether to display the plot
            kwargs: additional keyword arguments for customizing the plot
                - color, lw, xlim, ylim, xticks, yticks, xlabel, ylabel, etc.
        Return:
            fig, ax: figure and axis objects of the plot
        """
        for index in patterns_to_plot:
            if np.abs(index) >= len(self.xrd_data):
                raise ValueError("Invalid index for XRD pattern, out of range")

        fig, ax = plt.subplots(figsize=kwargs.get("figsize", (8, 6)))

        colors = list(get_colors("tab10").values())
        lw = kwargs.get("lw", 2)

        for idx, index in enumerate(patterns_to_plot):
            tt, i = self.xrd_data[index]

            color = colors[idx % len(colors)]

            if not broadened:
                ax.vlines(tt, [0], i, color=color, lw=lw)
            else:
                tt, i = self.broaden_spectrum(tt, i, **kwargs)
                ax.plot(tt, i, color=color, lw=lw)

        ax.set_xlim(kwargs.get("xlim", (10, 80)))
        ax.set_ylim(kwargs.get("ylim", (0, 105)))
        xticks = kwargs.get("xticks", np.arange(10, 81, 10))
        if xticks is not None:
            ax.set_xticks(xticks)
        yticks = kwargs.get("yticks", np.arange(0, 101, 10))
        if yticks is not None:
            ax.set_yticks(yticks)
        ax.set_xlabel(kwargs.get("xlabel", "2θ (degrees)"))
        ax.set_ylabel(kwargs.get("ylabel", "Intensity (arb. unit)"))

        if kwargs.get("grid", True):
            ax.grid(True)

        if savename is not None:
            fig.savefig(savename)

        if show:
            plt.show()

        return fig, ax

    def animated_plot(
        self,
        savename: str,
        writer: str = "ffmpeg",
        interval: int = 200,
        reference_pattern: int | tuple[np.ndarray, np.ndarray] | None = None,
        broadened: bool = False,
        **kwargs,
    ) -> None:
        """
        Note: requires ffmpeg to be installed on the system.
            for macOS: brew install ffmpeg

        Args:
            savename: save the animation to specified file location
            writer: writer to use for saving the animation, the default is "ffmpeg"
            interval: time interval between frames in milliseconds
            reference_pattern: index of the reference XRD pattern or (two-theta, intensity) data
            broadened: whether to plot the broadened XRD pattern
            kwargs: additional keyword arguments for customizing the plots
                - color (tab10), lw, xlim, ylim, xticks, yticks, xlabel, ylabel, etc.
        """
        fig, ax = plt.subplots(figsize=kwargs.get("figsize", (8, 6)))

        colors = get_colors("tab10")
        color = kwargs.get("color", "red")
        lw = kwargs.get("lw", 2)

        ax.set_xlim(kwargs.get("xlim", (10, 80)))
        ax.set_ylim(kwargs.get("ylim", (0, 105)))
        xticks = kwargs.get("xticks", np.arange(10, 81, 10))
        if xticks is not None:
            ax.set_xticks(xticks)
        yticks = kwargs.get("yticks", np.arange(0, 101, 10))
        if yticks is not None:
            ax.set_yticks(yticks)
        ax.set_xlabel(kwargs.get("xlabel", "2θ (degrees)"))
        ax.set_ylabel(kwargs.get("ylabel", "Intensity (arb. unit)"))

        if kwargs.get("grid", True):
            ax.grid(True)

        if reference_pattern is not None:
            if isinstance(reference_pattern, int):
                tt, i = self.xrd_data[reference_pattern]
            elif isinstance(reference_pattern, tuple):
                tt, i = reference_pattern
            else:
                raise ValueError("Invalid input for reference pattern")

            ref_label = kwargs.get("ref_label", "Reference")

            if not broadened:
                ax.vlines(tt, 0, i, color=colors["blue"], lw=lw, label=ref_label)
            else:
                tt, i = self.broaden_spectrum(tt, i, **kwargs)
                ax.plot(tt, i, color=colors["blue"], lw=lw, label=ref_label)

        if not broadened:
            plotting = ax.vlines([], 0, [], color=colors[color], lw=lw)
        else:
            (plotting,) = ax.plot([], [], color=colors[color], lw=lw)

        def update(frame):
            tt, i = self.xrd_data[frame]
            if not broadened:
                plotting.set_segments([[[t, 0], [t, i]] for t, i in zip(tt, i)])
            else:
                tt, i = self.broaden_spectrum(tt, i, **kwargs)
                plotting.set_data(tt, i)
            return (plotting,)

        ani = animation.FuncAnimation(
            fig, update, frames=len(self.xrd_data), interval=interval, blit=True
        )

        ani.save(savename, writer)

    def calc_var(self, two_theta: float, tau: float) -> float:
        """
        Args:
            two_theta: 2-theta value for peak (in degrees)
            tau: domain size (in nm)
        Returns:
            variance for gaussian kernel
        """
        # calculate FWHM using Scherrer eq
        K = 0.9
        wavelength = self.wavelength * 0.1  # angstrom to nm
        theta = np.radians(two_theta / 2)  # two theta degrees to theta radians
        beta = (K * wavelength) / (np.cos(theta) * tau)  # in radians

        # convert FWHM to std deviation of gaussian
        sigma = np.sqrt(1 / (2 * np.log(2))) * 0.5 * np.degrees(beta)

        return sigma**2

    def broaden_spectrum(
        self,
        two_thetas: np.ndarray,
        intensities: np.ndarray,
        tau=20,
        mesh_steps=5000,
        **kwargs,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        calculate broadened XRD pattern

        Args:
            two_thetas: 2-theta values for peaks (in degrees)
            intensities: intensities for peaks
            tau: domain size (in nm)
        Returns:
            broadened 2-theta and intensity values
        """
        # override default tau and mesh_step values
        # if provided as kwarg in another function call
        tau = kwargs.get("tau", tau)
        mesh_steps = kwargs.get("mesh_steps", mesh_steps)

        min_two_theta = np.min(two_thetas)
        max_two_theta = np.max(two_thetas)

        # create a fine mesh of two theta (tt) values
        tt_fine = np.linspace(min_two_theta, max_two_theta, mesh_steps)

        # create an intensity array with rows for each input tt value
        # and columns for the tt fine mesh
        i_array = np.zeros((len(two_thetas), len(tt_fine)))

        # map intensity at each input tt to the tt fine mesh
        for idx, angle in enumerate(two_thetas):
            nearest_idx = np.argmin(np.abs(angle - tt_fine))
            i_array[idx, nearest_idx] = intensities[idx]

        mesh_step_size = (max_two_theta - min_two_theta) / mesh_steps
        for idx in range(len(two_thetas)):
            # initial tt fine mesh intensity values for given tt input
            tt_fine_i = i_array[idx, :]
            # tt fine mesh angle with max intensity
            tt_fine_with_max_i = tt_fine[np.argmax(tt_fine_i)]
            # calculate variance for gaussian kernel
            var = self.calc_var(tt_fine_with_max_i, tau)
            # apply gaussian filter to the intensity values
            std_dev = np.sqrt(var) * (1 / mesh_step_size)
            i_array[idx, :] = gaussian_filter1d(tt_fine_i, std_dev, mode="constant")

        # sum intensity contributions for each tt fine mesh value
        i_fine = np.sum(i_array, axis=0)

        # normalize signal
        norm_fine_i = 100 * (i_fine / np.max(i_fine))

        return tt_fine, norm_fine_i

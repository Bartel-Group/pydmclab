from __future__ import annotations

import os
import numpy as np
import matplotlib.pyplot as plt

from ase.io.trajectory import Trajectory
from ase.io import write

from pymatgen.core.structure import Structure
from pymatgen.io.ase import AseAtomsAdaptor
from pymatgen.analysis.diffusion.aimd.pathway import ProbabilityDensityAnalysis

from pydmclab.core.struc import StrucTools
from pydmclab.plotting.xrd import PlotXRD
from pydmclab.utils.handy import convert_numpy_to_native
from pydmclab.plotting.utils import set_rc_params, get_colors
from pydmclab.mlp.utils import clean_md_log_and_traj_files

set_rc_params()


class AnalyzeMD:
    def __init__(
        self,
        logfile: str = "md.log",
        trajfile: str = "md.traj",
        clean_files: bool = True,
        full_summary: list[dict] | None = None,
    ) -> None:
        """
        Args:
            logfile (str): The filename for the log file.
            trajfile (str): The filename for the trajectory.
            clean_files (bool): Whether to clean the log and trajectory files.
            full_summary (list[dict] | None): Optionally, a pre-generated summary containing both log and traj info.
        """

        required_keys = {"t", "T", "Etot", "Epot", "Ekin", "run", "structure"}

        if full_summary is not None:
            for i, entry in enumerate(full_summary):
                if not isinstance(entry, dict):
                    raise ValueError(f"Entry {i} in full_summary is not a dictionary.")
                missing_keys = required_keys - entry.keys()
                if missing_keys:
                    raise ValueError(
                        f"Entry {i} in full_summary is missing keys: {missing_keys}"
                    )
            self._full_summary = full_summary
        else:
            if clean_files:
                clean_md_log_and_traj_files(logfile, trajfile)

            self.logfile = logfile
            self.trajfile = trajfile
            self._full_summary = None

    @property
    def log_summary(self) -> list[dict]:
        """
        Returns:
            list[dict]: A summary of the log file.
        """
        if self._full_summary is not None:
            return [
                {k: v for k, v in d.items() if k != "structure"}
                for d in self._full_summary
            ]

        data = []
        run_count = 0
        t_adj = 0.0
        change_time_adjust = False
        with open(self.logfile, "r", encoding="utf-8") as logf:
            for line in logf:
                line = line[:-1]
                if "Time" in line:
                    run_count += 1
                    if run_count > 1:
                        change_time_adjust = True
                    continue
                if change_time_adjust:
                    t_adj = data[-1]["t"]
                    change_time_adjust = False
                t, Etot, Epot, Ekin, T = line.split()
                data.append(
                    {
                        "t": float(t) + t_adj,
                        "T": float(T),
                        "Etot": float(Etot),
                        "Epot": float(Epot),
                        "Ekin": float(Ekin),
                        "run": run_count,
                    }
                )
        return data

    @property
    def traj_summary(self) -> list[dict]:
        """
        Returns:
            list[dict]: A summary of the trajectory file.
                each item of the list is a Structure.as_dict()
                corresponds with the log_summary dict
        """
        if self._full_summary is not None:
            return [d["structure"] for d in self._full_summary]

        with Trajectory(self.trajfile, "r") as traj:
            return [AseAtomsAdaptor().get_structure(atoms).as_dict() for atoms in traj]

    @property
    def full_summary(self) -> list[dict]:
        """
        Returns:
            list[dict]: A summary of the log and trajectory files.
                each item of the list is a dict with the log_summary and corresponding structure at that time step
        """
        if self._full_summary is not None:
            return self._full_summary

        log_summary = self.log_summary
        traj_summary = self.traj_summary

        if len(log_summary) != len(traj_summary):
            raise Warning(
                "log and trajectory files have different lengths, data may be mismatched"
            )

        for i, structure in enumerate(traj_summary):
            log_summary[i]["structure"] = convert_numpy_to_native(structure)

        return log_summary

    @property
    def get_E_T_t_data(self) -> tuple[list[float], list[float], list[float]]:
        """
        Returns:
            tuple[list[float], list[float], list[float]]: The time, potential energy, and temperature data.
        """
        data = self.log_summary

        times = [d["t"] for d in data]
        temps = [d["T"] for d in data]
        Epots = [d["Epot"] for d in data]

        return times, Epots, temps

    def plot_E_T_t(
        self,
        T_setpoint: float | None = None,
        xlim: tuple[float, float] | None = None,
        ylim_T: tuple[float, float] | None = None,
        savename: str | None = None,
        show: bool = False,
        return_fig: bool = False,
    ) -> tuple[plt.Figure, tuple[plt.Axes, plt.Axes]] | None:
        """
        plots E vs t and T vs t

        Args:
            T_setpoint (float | None): The temperature setpoint to include in the plot.
            xlim (tuple[float, float] | None): The x-axis limits of the subplots.
            ylim_T (tuple[float, float] | None): The y-axis limits of the temperature subplot.
            save_name (str): The file path to save the plot.
            show (bool): Whether to show the plot.
            return_fig (bool): Whether to return the figure and axes objects.

        Returns:
            if return_fig is True, returns the figure and axes objects
        """
        times, Epots, temps = self.get_E_T_t_data

        colors = get_colors("tab10")

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 6), sharex=True)
        ax1.plot(times, Epots, label="E", color=colors["blue"])
        ax1.set_ylabel("E (eV)")
        ax2.plot(times, temps, label="T", color=colors["orange"])
        ax2.set_ylabel("T (K)")
        ax2.set_xlabel("time (ps)")

        if T_setpoint is not None:
            ax2.axhline(y=T_setpoint, color=colors["red"], linestyle="--", linewidth=2)

        if xlim is not None:
            ax1.set_xlim(xlim)
            ax2.set_xlim(xlim)

        if ylim_T is not None:
            ax2.set_ylim(ylim_T)

        fig.tight_layout()

        if savename is not None:
            fig.savefig(savename)

        if show:
            fig.show()

        if return_fig:
            return fig, (ax1, ax2)

    def get_T_distribution_data(
        self,
        time_range: tuple[float, float] | None = None,
        remove_outliers: float | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        gets data for plotting the temperature distribution

        Args:
            time_range (tuple[float, float] | float | None): The time range to plot the temperature distribution.
                - If None, the entire simulation is used.
                - If float, the percentage of the simulation to use (starting from the end); i.e. 0.1 is the last 10% of the simulation.
                - If tuple, the start and end times of the simulation to use.
            remove_outliers (float | None): Don't plot data points with z-scores greater than this value. Does not affect mean and std calculations.

        Returns:
            tuple[np.ndarray, np.ndarray]: The full temperature data and the inlier temperature data (remaining temperatures after removing outliers).
        """
        data = self.log_summary

        if time_range is None:
            temps = np.array([d["T"] for d in data])
        elif isinstance(time_range, float):
            if time_range < 0 or time_range > 1:
                raise ValueError(
                    "if setting time_range as a float, it is a percentage, so it should be between 0 and 1"
                )
            temps = np.array([d["T"] for d in data[-int(time_range * len(data)) :]])
        elif isinstance(time_range, tuple):
            if time_range[0] < data[0]["t"] or time_range[1] > data[-1]["t"]:
                raise ValueError("time_range is outside the simulation time range")
            temps = np.array(
                [d["T"] for d in data if time_range[0] <= d["t"] <= time_range[1]]
            )
        else:
            raise ValueError("time_range should be None, float, or tuple[float, float]")

        if remove_outliers is not None:
            z_scores = (temps - np.mean(temps)) / np.std(temps)
            inlier_temps = temps[np.abs(z_scores) <= remove_outliers]
        else:
            inlier_temps = temps

        return temps, inlier_temps

    def plot_T_distribution(
        self,
        time_range: tuple[float, float] | float | None = None,
        num_bins: int = 50,
        density: bool = True,
        remove_outliers: float | None = None,
        include_mean: bool = False,
        T_setpoint: float | None = None,
        savename: str | None = None,
        show: bool = False,
        return_fig: bool = False,
    ) -> tuple[plt.Figure, plt.Axes] | None:
        """
        plots the temperature distribution

        Args:
            time_range (tuple[float, float] | float | None): The time range to plot the temperature distribution.
                - If None, the entire simulation is used.
                - If float, the percentage of the simulation to use (starting from the end); i.e. 0.1 is the last 10% of the simulation.
                - If tuple, the start and end times of the simulation to use.
            num_bins (int): The number of bins for the histogram.
            density (bool): If True, plot the density instead of the frequency.
            remove_outliers (float | None): Don't plot data points with z-scores greater than this value. Does not affect mean and std calculations.
            include_mean (bool): Whether to include the mean temperature in the plot.
            T_setpoint (float | None): The temperature setpoint to include in the plot.
            savename (str): The file path to save the plot.
            show (bool): Whether to show the plot.
            return_fig (bool): Whether to return the figure and axes objects.

        Returns:
            If return_fig is True, returns the figure and axes objects.
        """
        temps, inlier_temps = self.get_T_distribution_data(
            time_range=time_range, remove_outliers=remove_outliers
        )

        colors = get_colors("tab10")

        fig, ax = plt.subplots(1, 1, figsize=(8, 6))
        ax.hist(inlier_temps, bins=num_bins, density=density, color=colors["orange"])
        ax.set_xlabel("T (K)")
        ax.set_ylabel("Density" if density else "Frequency")

        mean_plotting_temp = np.mean(temps)
        max_deviation = np.max(np.abs(temps - mean_plotting_temp))
        ax.set_xlim(
            np.floor((mean_plotting_temp - max_deviation) / 100) * 100,
            np.ceil((mean_plotting_temp + max_deviation) / 100) * 100,
        )

        if include_mean:
            ax.axvline(
                np.mean(temps),
                color=colors["blue"],
                linestyle="--",
                linewidth=2,
                label="mean",
            )

        if T_setpoint is not None:
            ax.axvline(
                T_setpoint,
                color=colors["red"],
                linestyle="--",
                linewidth=2,
                label="setpoint",
            )

        if include_mean or T_setpoint is not None:
            ax.legend(loc="upper right")

        fig.tight_layout()

        if savename is not None:
            fig.savefig(savename)

        if show:
            fig.show()

        if return_fig:
            return fig, ax

    def get_xrd_data(
        self,
        time_range: tuple[float, float] | float | None = None,
        data_density: float = 0.2,
    ) -> tuple[Structure]:
        """

        Args:
            time_range (tuple[float, float] | float | None): The time range to examine the XRD evolution.
                - If None, the entire simulation is used.
                - If float, the percentage of the simulation to use (starting from the end); i.e. 0.1 is the last 10% of the simulation.
                - If tuple, the start and end times of the simulation to use.
            data_density (float): The percentage of data points to generate the XRD pattern for and include in the movie.

        Returns:
            tuple[Structure]: The structures to use for the XRD analysis
        """
        data = self.full_summary

        if time_range is None:
            strucs = [d["structure"] for d in data]
        elif isinstance(time_range, float):
            if time_range < 0 or time_range > 1:
                raise ValueError(
                    "if setting time_range as a float, it is a percentage, so it should be between 0 and 1"
                )
            strucs = [d["structure"] for d in data[-int(time_range * len(data)) :]]
        elif isinstance(time_range, tuple):
            if time_range[0] < data[0]["t"] or time_range[1] > data[-1]["t"]:
                raise ValueError("time_range is outside the simulation time range")
            strucs = [
                d["structure"] for d in data if time_range[0] <= d["t"] <= time_range[1]
            ]
        else:
            raise ValueError("time_range should be None, float, or tuple[float, float]")

        reduced_strucs = tuple(strucs[:: int(1 / data_density)])

        return reduced_strucs

    def xrd_movie(
        self,
        time_range: tuple[float, float] | float | None = None,
        data_density: float = 0.2,
        reference_pattern: int | tuple[np.ndarray, np.ndarray] | None = 0,
        broadened: bool = True,
        savename: str = "md_xrd_evolution.mp4",
    ) -> None:
        """
        makes a movie of the XRD evolution

        note: this method requires 'ffmpeg' to be installed on your system

        Args:
            time_range (tuple[float, float] | float | None): The time range to examine the XRD evolution.
                - If None, the entire simulation is used.
                - If float, the percentage of the simulation to use (starting from the end); i.e. 0.1 is the last 10% of the simulation.
                - If tuple, the start and end times of the simulation to use.
            data_density (float): The percentage of data points to generate the XRD pattern for and include in the movie.
            reference_pattern (int | tuple[np.ndarray, np.ndarray] | None): The reference XRD pattern to include in the movie.
                - If int, the indice of xrd pattern to use (most likely 0 or -1).
                - If tuple, input should be tuple of (2-theta, intensity) to use as the reference pattern.
                - If None, no reference pattern is included.
            broadened (bool): Whether to include broadening in the XRD pattern.
            savename (str): The filename to save the movie.
        """
        reduced_strucs = self.get_xrd_data(
            time_range=time_range, data_density=data_density
        )

        plotXRD = PlotXRD(xrd_data=reduced_strucs)

        plotXRD.animated_plot(
            savename=savename,
            time_interval=100,
            reference_pattern=reference_pattern,
            broadened=broadened,
        )

    def write_pdb(self, savename: str = "chgnet_md.pdb", remake: bool = False) -> None:
        """
        writes the trajectory to a pdb file

        Args:
            savename (str): The file path to save the pdb file.
            remake (bool): Whether to remake the pdb file.
        """
        if os.path.exists(savename) and not remake:
            print(f"{savename} already exists. Skipping.")
            return

        ase_atoms = [
            AseAtomsAdaptor().get_atoms(StrucTools(struc).structure)
            for struc in self.traj_summary
        ]
        write(savename, ase_atoms)

    def make_probability_density_analysis_chgcar(
        self,
        species: str | tuple[str],
        ref_idx: int = 0,
        interval: float = 0.5,
        savename: str = "chgnet_md_pda.vasp",
        remake: bool = False,
    ) -> None:
        """
        sends probability density to chgcar format for visualization (e.g., in VESTA)

        Args:
            species (str | tuple[str]): The species to analyze.
            ref_idx (int): The index of the reference structure to use (most likely 0 or -1).
            interval (float): Interval between nearest grid points (Å).
            savename (str): The file path to save the chgcar file.
            remake (bool): Whether to remake the chgcar file.
        """
        if os.path.exists(savename) and not remake:
            print(f"{savename} already exists. Skipping.")
            return

        # load in trajectories as list of Structure.as_dict()
        trajs = self.traj_summary

        # check ref_idx is within bounds
        if not (-len(trajs) <= ref_idx < len(trajs)):
            raise IndexError(
                f"ref_idx {ref_idx} is out of bounds for number of trajectories ({len(trajs)})"
            )

        # grab structure to serve as reference
        ini_struc = StrucTools(trajs[ref_idx]).structure

        # make array of fractional coordinates
        num_timesteps = len(trajs)
        num_ions = len(trajs[0]["sites"])
        traj_array = np.zeros((num_timesteps, num_ions, 3))
        for i, traj in enumerate(trajs):
            for j in range(num_ions):
                traj_array[i, j] = traj["sites"][j]["abc"]

        # initialize pda
        pda = ProbabilityDensityAnalysis(
            structure=ini_struc,
            trajectories=traj_array,
            interval=interval,
            species=species,
        )

        # make chgcar for pda visualization
        pda.to_chgcar(filename=savename)

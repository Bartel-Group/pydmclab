from __future__ import annotations

import os
import io
import sys
import inspect
import contextlib
import pickle as pkl
from enum import Enum
from collections.abc import Sequence
from typing import TYPE_CHECKING, Literal

from chgnet.model import CHGNet
from chgnet.model.dynamics import MolecularDynamics
from chgnet.utils import determine_device

import numpy as np
import ase.optimize as opt
from ase import Atoms
from ase import filters
from ase.filters import Filter
from ase.calculators.calculator import Calculator, all_changes, all_properties
from ase.io.trajectory import Trajectory
from ase.io.jsonio import encode, decode
from ase.io import write

from pymatgen.core.structure import Structure
from pymatgen.io.ase import AseAtomsAdaptor
from pymatgen.analysis.diffusion.aimd.pathway import ProbabilityDensityAnalysis

from torch import Tensor

import matplotlib.pyplot as plt
from pydmclab.core.struc import StrucTools
from pydmclab.plotting.xrd import PlotXRD
from pydmclab.plotting.utils import set_rc_params, get_colors
from pydmclab.mlp.chgnet import clean_md_log_and_traj_files

from pydmclab.utils.handy import convert_numpy_to_native

set_rc_params()

if TYPE_CHECKING:
    from pydmclab.mlp.chgnet import Versions, Devices, PredTask
    from typing_extensions import Self
    from ase.optimize.optimize import Optimizer as ASEOptimizer


class OPTIMIZERS(Enum):
    """An enumeration of optimizers available in ASE."""

    fire = opt.fire.FIRE
    bfgs = opt.bfgs.BFGS
    lbfgs = opt.lbfgs.LBFGS
    lbfgslinesearch = opt.lbfgs.LBFGSLineSearch
    mdmin = opt.mdmin.MDMin
    scipyfmincg = opt.sciopt.SciPyFminCG
    scipyfminbfgs = opt.sciopt.SciPyFminBFGS
    bfgslinesearch = opt.bfgslinesearch.BFGSLineSearch


class CHGNetCalculator(Calculator):
    """CHGNet Calculator for ASE applications."""

    implemented_properties = (
        "energy",
        "forces",
        "stress",
        "magmoms",
    )  # Needed for ASE compatibility (Do not remove)

    def __init__(
        self,
        model: CHGNet | Versions | None = None,
        *,
        use_device: Devices | None = None,
        check_cuda_mem: bool = False,
        stress_weight: float | None = 1 / 160.21766208,
        on_isolated_atoms: Literal["ignore", "warn", "error"] = "warn",
        **kwargs,
    ) -> None:

        super().__init__(**kwargs)

        device = determine_device(use_device=use_device, check_cuda_mem=check_cuda_mem)
        self.device = device
        self.stress_weight = stress_weight

        if isinstance(model, str):
            self.model = CHGNet.load(model_name=model, verbose=False).to(self.device)
        elif model is None:
            self.model = CHGNet.load(use_device=self.device, verbose=False)
        else:
            self.model = model.to(self.device)

        self.model.graph_converter.set_isolated_atom_response(on_isolated_atoms)
        print(f"CHGNet will run on {self.device}")

    @classmethod
    def from_file(
        cls,
        path: str,
        use_device: Devices | None = None,
        check_cuda_mem: bool = False,
        stress_weight: float | None = 1 / 160.21766208,
        on_isolated_atoms: Literal["ignore", "warn", "error"] = "warn",
        **kwargs,
    ) -> Self:
        """Load a user's CHGNet model and initialize the Calculator."""
        return cls(
            model=CHGNet.from_file(path),
            use_device=use_device,
            check_cuda_mem=check_cuda_mem,
            stress_weight=stress_weight,
            on_isolated_atoms=on_isolated_atoms,
            **kwargs,
        )

    @property
    def version(self) -> str | None:
        """The version of CHGNet."""
        return self.model.version

    @property
    def n_params(self) -> int:
        """The number of parameters in CHGNet."""
        return self.model.n_params

    def calculate(
        self,
        atoms: Atoms | None = None,
        properties: list | None = None,
        system_changes: list | None = None,
    ) -> None:
        """Calculate various properties of the atoms using CHGNet.

        Args:
            atoms (Atoms | None): The atoms object to calculate properties for.
            properties (list | None): The properties to calculate.
                Default is all properties.
            system_changes (list | None): The changes made to the system.
                Default is all changes.
        """
        properties = properties or all_properties
        system_changes = system_changes or all_changes
        super().calculate(
            atoms=atoms,
            properties=properties,
            system_changes=system_changes,
        )

        # Run CHGNet
        structure = AseAtomsAdaptor.get_structure(atoms)
        graph = self.model.graph_converter(structure)
        model_prediction = self.model.predict_graph(
            graph.to(self.device), task="efsm", return_crystal_feas=True
        )

        # Convert Result
        factor = 1 if not self.model.is_intensive else structure.composition.num_atoms
        self.results.update(
            energy=model_prediction["e"] * factor,
            forces=model_prediction["f"],
            magmoms=model_prediction["m"],
            stress=model_prediction["s"] * self.stress_weight,
            crystal_fea=model_prediction["crystal_fea"],
        )


class CHGNetObserver:
    """Trajectory observer is a hook in the relaxation process that saves the
    intermediate structures.
    """

    def __init__(self, atoms: Atoms) -> None:
        """Create a TrajectoryObserver from an Atoms object.

        Args:
            atoms (Atoms): the structure to observe.
        """
        self.atoms = atoms
        self.energies: list[float] = []
        self.forces: list[np.ndarray] = []
        self.stresses: list[np.ndarray] = []
        self.magmoms: list[np.ndarray] = []
        self.atomic_numbers: list[int] = []
        self.atom_positions: list[np.ndarray] = []
        self.cells: list[np.ndarray] = []

    def __call__(self) -> None:
        """The logic for saving the properties of an Atoms during the relaxation."""
        self.energies.append(self.atoms.get_potential_energy())
        self.forces.append(self.atoms.get_forces())
        self.stresses.append(self.atoms.get_stress())
        self.magmoms.append(self.atoms.get_magnetic_moments())
        self.atomic_numbers.append(self.atoms.get_atomic_numbers())
        self.atom_positions.append(self.atoms.get_positions())
        self.cells.append(self.atoms.get_cell()[:])

    def __len__(self) -> int:
        """The number of steps in the trajectory."""
        return len(self.energies)

    def as_dict(self) -> dict[str, list]:
        """Return the trajectory as a dictionary."""
        return {
            "atoms": encode(
                self.atoms
            ),  # returns the atoms object as a str representation
            "energies": self.energies,
            "forces": self.forces,
            "stresses": self.stresses,
            "magmoms": self.magmoms,
            "atomic_numbers": self.atomic_numbers,
            "atom_positions": self.atom_positions,
            "cells": self.cells,
        }

    @classmethod
    def from_dict(cls, data: dict[str, list]) -> Self:
        """Create a TrajectoryObserver from a dictionary."""
        obs = cls(decode(data["atoms"]))
        obs.energies = data["energies"]
        obs.forces = data["forces"]
        obs.stresses = data["stresses"]
        obs.magmoms = data["magmoms"]
        obs.atomic_numbers = data["atomic_numbers"]
        obs.atom_positions = data["atom_positions"]
        obs.cells = data["cells"]
        return obs

    def save(self, filename: str) -> None:
        """Save the trajectory to file.

        Args:
            filename (str): filename to save the trajectory
        """
        out_pkl = self.as_dict()
        with open(filename, "wb") as file:
            pkl.dump(out_pkl, file)


class CHGNetRelaxer:
    """Wrapper class for structural relaxation."""

    def __init__(
        self,
        model: CHGNet | CHGNetCalculator | Versions | None = None,
        optimizer: ASEOptimizer | str = "FIRE",
        use_device: Devices | None = None,
        check_cuda_mem: bool = False,
        stress_weight: float | None = 1 / 160.21766208,
        on_isolated_atoms: Literal["ignore", "warn", "error"] = "warn",
    ) -> None:

        self.optimizer: ASEOptimizer = (
            OPTIMIZERS[optimizer.lower()].value
            if isinstance(optimizer, str)
            else optimizer
        )

        if isinstance(model, CHGNetCalculator):
            self.calculator = model
            self.model = self.calculator.model
        else:
            self.calculator = CHGNetCalculator(
                model=model,
                use_device=use_device,
                check_cuda_mem=check_cuda_mem,
                stress_weight=stress_weight,
                on_isolated_atoms=on_isolated_atoms,
            )
            self.model = self.calculator.model

    @classmethod
    def from_file(
        cls,
        path: str | os.PathLike[str],
        optimizer: ASEOptimizer | str = "FIRE",
        use_device: Devices | None = None,
        check_cuda_mem: bool = False,
        stress_weight: float | None = 1 / 160.21766208,
        on_isolated_atoms: Literal["ignore", "warn", "error"] = "warn",
        **kwargs,
    ) -> Self:
        """Load a user's CHGNet model and initialize the Calculator."""
        return cls(
            model=CHGNet.from_file(path),
            optimizer=optimizer,
            use_device=use_device,
            check_cuda_mem=check_cuda_mem,
            stress_weight=stress_weight,
            on_isolated_atoms=on_isolated_atoms,
            **kwargs,
        )

    @property
    def version(self) -> str | None:
        """The version of CHGNet."""
        return self.model.version

    @property
    def n_params(self) -> int:
        """The number of parameters in CHGNet."""
        return self.model.n_params

    def predict_structure(
        self,
        structure: Structure | Sequence[Structure],
        *,
        task: PredTask = "efsm",
        return_site_energies: bool = False,
        return_atom_feas: bool = False,
        return_crystal_feas: bool = False,
        batch_size: int = 16,
    ) -> dict[str, Tensor] | list[dict[str, Tensor]]:
        """Predict the properties of a structure or list of structures."""
        return self.model.predict_structure(
            structure,
            task=task,
            return_site_energies=return_site_energies,
            return_atom_feas=return_atom_feas,
            return_crystal_feas=return_crystal_feas,
            batch_size=batch_size,
        )

    def relax(
        self,
        atoms: Structure | Atoms,
        *,
        fmax: float | None = 0.1,
        steps: int | None = 500,
        relax_cell: bool | None = True,
        ase_filter: str | None = "FrechetCellFilter",
        params_asefilter: dict | None = None,
        traj_path: str | None = None,
        interval: int | None = 1,
        verbose: bool = True,
        convert_to_native_types: bool = True,
        **kwargs,
    ) -> dict[str, Structure | CHGNetObserver]:
        """Relax the Structure/Atoms until maximum force is smaller than fmax."""

        valid_filter_names = [
            name
            for name, cls in inspect.getmembers(filters, inspect.isclass)
            if issubclass(cls, Filter)
        ]

        if isinstance(ase_filter, str):
            if ase_filter in valid_filter_names:
                ase_filter = getattr(filters, ase_filter)
            else:
                raise ValueError(
                    f"Invalid {ase_filter=}, must be one of {valid_filter_names}. "
                )

        if isinstance(atoms, Structure):
            atoms = AseAtomsAdaptor().get_atoms(atoms)
        atoms.calc = self.calculator

        stream = sys.stdout if verbose else io.StringIO()
        params_asefilter = params_asefilter or {}
        with contextlib.redirect_stdout(stream):
            obs = CHGNetObserver(atoms)

            if relax_cell:
                atoms = ase_filter(atoms, **params_asefilter)

            optimizer: ASEOptimizer = self.optimizer(atoms, **kwargs)
            optimizer.attach(obs, interval=interval)
            optimizer.run(fmax=fmax, steps=steps)
            obs()

        if traj_path is not None:
            obs.save(traj_path)

        if isinstance(atoms, Filter):
            atoms = atoms.atoms

        struc = AseAtomsAdaptor.get_structure(atoms)

        if convert_to_native_types:
            native_struc = convert_numpy_to_native(struc.as_dict())
            struc = Structure.from_dict(native_struc)

            native_obs = convert_numpy_to_native(obs.as_dict())
            obs = CHGNetObserver.from_dict(native_obs)

        return {
            "final_structure": struc,
            "final_energy": obs.energies[-1],
            "trajectory": obs,
        }


class CHGNetMD:
    def __init__(
        self,
        structure: Structure | Atoms,
        *,
        model: CHGNet | CHGNetCalculator | Versions | None = None,
        relax_first: bool = False,
        temperature: int = 300,
        pressure: float = 1.01325e-4,
        ensemble: str = "nvt",
        thermostat: str = "Berendsen_inhomogeneous",
        timestep: float = 2.0,
        taut: float | None = None,
        use_device: Devices | None = None,
        check_cuda_mem: bool = False,
        stress_weight: float | None = 1 / 160.21766208,
        on_isolated_atoms: Literal["ignore", "warn", "error"] = "warn",
        logfile: str = "md.log",
        trajfile: str = "md.traj",
        loginterval: int = 10,
        append_trajectory: bool = False,
        **kwargs,
    ) -> None:
        """
        Args:
            structure (Structure | Atoms): The initial structure to simulate.
            model (CHGNet | CHGNetCalculator): The CHGNet model to use.
            relax_first (bool): Whether to relax the structure before running the MD.
            temperature (int): The temperature of the simulation.
            pressure (float): The pressure of the simulation.
            ensemble (str): The ensemble to use for the simulation.
            thermostat (str): The thermostat to use for the simulation.
            timestep (float): The timestep for the simulation.
            tau_t (float | None): The time constant for the thermostat.
            use_device (Devices | None): The device to run the simulation on.
            check_cuda_mem (bool): Whether to check the available CUDA memory.
            stress_weight (float | None): The weight of the stress in the loss function.
            on_isolated_atoms (Literal["ignore", "warn", "error"]): How to handle isolated atoms.
            logfile (str | None): The filename for the log file.
            trajectory (str | None): The filename for the trajectory file.
            loginterval (int): The interval to log the simulation.
            append_trajectory (bool): Whether to append to the trajectory file.
            **kwargs: Additional keyword arguments.
        """

        self.relaxer: CHGNetRelaxer | None = None

        if relax_first:
            self.relaxer = CHGNetRelaxer(
                model=model,
                use_device=use_device,
                check_cuda_mem=check_cuda_mem,
                stress_weight=stress_weight,
                on_isolated_atoms=on_isolated_atoms,
            )
            structure = self.relaxer.relax(structure, **kwargs)["final_structure"]

        if isinstance(structure, Structure):
            self.structure = structure
            self.atoms = AseAtomsAdaptor().get_atoms(structure)
        else:
            self.atoms = structure
            self.structure = AseAtomsAdaptor().get_structure(structure)

        if isinstance(model, CHGNetCalculator):
            self.calculator = model
            self.model = self.calculator.model
        else:
            self.calculator = CHGNetCalculator(
                model=model,
                use_device=use_device,
                check_cuda_mem=check_cuda_mem,
                stress_weight=stress_weight,
                on_isolated_atoms=on_isolated_atoms,
            )
            self.model = self.calculator.model

        self.atoms.calc = self.calculator

        self.logfile = logfile
        self.trajectory = trajfile
        self.md = MolecularDynamics(
            atoms=self.atoms,
            model=self.model,
            ensemble=ensemble,
            temperature=temperature,
            thermostat=thermostat,
            timestep=timestep,
            taut=taut,
            pressure=pressure,
            use_device=use_device,
            on_isolated_atoms=on_isolated_atoms,
            logfile=self.logfile,
            trajectory=self.trajectory,
            loginterval=loginterval,
            append_trajectory=append_trajectory,
        )

    @classmethod
    def continue_from_traj(
        cls,
        *,
        model: CHGNet | CHGNetCalculator | Versions | None = None,
        relax_first: bool = False,
        temperature: int = 300,
        pressure: float = 1.01325e-4,
        ensemble: str = "nvt",
        thermostat: str = "Berendsen_inhomogeneous",
        timestep: float = 2.0,
        taut: float | None = None,
        use_device: Devices | None = None,
        check_cuda_mem: bool = False,
        stress_weight: float | None = 1 / 160.21766208,
        on_isolated_atoms: Literal["ignore", "warn", "error"] = "warn",
        trajfile: str = "md.traj",
        logfile: str = "md.log",
        loginterval: int = 10,
        append_trajectory: bool = True,
        **kwargs,
    ) -> Self:
        """Continue an MD simulation from a trajectory file."""
        if not os.path.exists(trajfile):
            raise FileNotFoundError(f"{trajfile} not found")
        with Trajectory(trajfile, "r") as traj:
            starting_structure = traj[-1]
        return cls(
            structure=starting_structure,
            model=model,
            relax_first=relax_first,
            temperature=temperature,
            pressure=pressure,
            ensemble=ensemble,
            thermostat=thermostat,
            timestep=timestep,
            taut=taut,
            use_device=use_device,
            check_cuda_mem=check_cuda_mem,
            stress_weight=stress_weight,
            on_isolated_atoms=on_isolated_atoms,
            logfile=logfile,
            trajfile=trajfile,
            loginterval=loginterval,
            append_trajectory=append_trajectory,
            **kwargs,
        )

    def run(self, steps: int = 1000):
        """
        Args:
            steps (int): The number of steps to run the simulation.

        """
        if steps < 1:
            raise ValueError("number of steps should be greater than 0")
        self.md.run(steps=steps)


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
            return [AseAtomsAdaptor.get_structure(atoms).as_dict() for atoms in traj]

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
            log_summary[i]["structure"] = structure

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
            AseAtomsAdaptor.get_atoms(StrucTools(struc).structure)
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

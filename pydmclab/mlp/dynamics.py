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

from pymatgen.core.structure import Structure
from pymatgen.io.ase import AseAtomsAdaptor

from torch import Tensor

import matplotlib.pyplot as plt
from pydmclab.plotting.utils import set_rc_params

set_rc_params()


if TYPE_CHECKING:
    from pydmclab.mlp import Versions, Devices, PredTask
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

    def save(self, filename: str) -> None:
        """Save the trajectory to file.

        Args:
            filename (str): filename to save the trajectory
        """
        out_pkl = {
            "energy": self.energies,
            "forces": self.forces,
            "stresses": self.stresses,
            "magmoms": self.magmoms,
            "atomic_number": self.atomic_numbers,
            "atom_positions": self.atom_positions,
            "cell": self.cells,
        }
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

        struct = AseAtomsAdaptor.get_structure(atoms)

        return {
            "final_structure": struct,
            "final_energy": float(obs.energies[-1]),
            "trajectory": obs,
        }


class CHGNetMD:
    def __init__(
        self,
        structure: Structure | Atoms,
        model: CHGNet | CHGNetCalculator,
        relax_first: bool = False,
        temperature: int = 300,
        pressure: float = 1.01325e-4,
        ensemble: str = "nvt",
        thermostat: str = "Berendsen_inhomogeneous",
        timestep: float = 2.0,
        use_device: Devices | None = None,
        check_cuda_mem: bool = False,
        stress_weight: float | None = 1 / 160.21766208,
        on_isolated_atoms: Literal["ignore", "warn", "error"] = "warn",
        logfile: str | None = None,
        trajectory: str | None = None,
        loginterval: int = 10,
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
            use_device (Devices | None): The device to run the simulation on.
            check_cuda_mem (bool): Whether to check the available CUDA memory.
            stress_weight (float | None): The weight of the stress in the loss function.
            on_isolated_atoms (Literal["ignore", "warn", "error"]): How to handle isolated atoms.
            logfile (str | None): The filename for the log file.
            trajectory (str | None): The filename for the trajectory file.
            loginterval (int): The interval to log the simulation.
            **kwargs: Additional keyword arguments.
        """
        if isinstance(structure, Structure):
            structure = AseAtomsAdaptor().get_atoms(structure)
        structure.calc = model
        if isinstance(model, CHGNetCalculator):
            calculator = model
        else:
            calculator = CHGNetCalculator(
                model=model,
                use_device=use_device,
                check_cuda_mem=check_cuda_mem,
                stress_weight=stress_weight,
                on_isolated_atoms=on_isolated_atoms,
            )
            model = calculator.model
        if relax_first:
            relaxer = CHGNetRelaxer(
                model=model,
                use_device=use_device,
                check_cuda_mem=check_cuda_mem,
                stress_weight=stress_weight,
                on_isolated_atoms=on_isolated_atoms,
            )
            structure = relaxer.relax(structure, **kwargs)["final_structure"]

        self.structure = structure
        self.logfile = logfile if logfile else "md.log"
        self.trajectory = trajectory if trajectory else "md.traj"
        self.md = MolecularDynamics(
            atoms=structure,
            model=model,
            ensemble=ensemble,
            temperature=temperature,
            thermostat=thermostat,
            timestep=timestep,
            pressure=pressure,
            use_device=use_device,
            on_isolated_atoms=on_isolated_atoms,
            logfile=self.logfile,
            trajectory=self.trajectory,
            loginterval=loginterval,
        )

    def run(self, steps: int = 1000):
        """
        Args:
            steps (int): The number of steps to run the simulation.

        """
        self.md.run(steps=steps)


class AnalyzeMD:
    def __init__(self, logfile, trajfile):
        """
        Args:
            logfile (str): The filename for the log file.
            trajfile (str): The filename for the trajectory
        """
        self.logfile = logfile
        self.trajfile = trajfile

    @property
    def log_summary(self):
        """
        Returns:
            list[dict]: A summary of the log file.
        """
        data = []
        with open(self.logfile, "r") as f:
            for line in f:
                line = line[:-1]
                if "Time" in line:
                    continue
                t, Etot, Epot, Ekin, T = line.split()
                data.append(
                    {
                        "t": float(t),
                        "T": float(T),
                        "Etot": float(Etot),
                        "Epot": float(Epot),
                        "Ekin": float(Ekin),
                    }
                )
        return data

    @property
    def traj_summary(self):
        """
        Returns:
            list[dict]: A summary of the trajectory file.
                each item of the list is a Structure.as_dict()
                corresponds with the log_summary dict
        """
        traj = Trajectory(self.trajfile)
        return [AseAtomsAdaptor.get_structure(atoms).as_dict() for atoms in traj]

    @property
    def full_summary(self):
        """
        Returns:
            list[dict]: A summary of the log and trajectory files.
                each item of the list is a dict with the log_summary and corresponding structure at that time step
        """
        log_summary = self.log_summary
        traj_summary = self.traj_summary
        for i in range(len(log_summary)):
            log_summary[i]["structure"] = traj_summary[i]
        return log_summary

    @property
    def plot_E_T_t(self):
        """
        Returns:
            plots E vs t and T vs t
        """
        data = self.log_summary

        times = [d["t"] for d in data]
        temps = [d["T"] for d in data]
        Epots = [d["Epot"] for d in data]

        fig = plt.figure()
        ax1 = plt.subplot(211)
        ax1 = plt.plot(times, Epots, label="E")
        ax1 = plt.ylabel("E (eV)")
        ax1 = plt.gca().xaxis.set_ticklabels([])
        # ax1 = plt.legend()
        ax2 = plt.subplot(212)
        ax2 = plt.plot(times, temps, label="T", color="orange")
        ax2 = plt.ylabel("T (K)")
        ax2 = plt.xlabel("time (ps)")
        ax2 = plt.legend()


def main():
    rerun_MD = False
    ftraj = "/Users/cbartel/Downloads/md.traj"
    flog = "/Users/cbartel/Downloads/md.log"
    if rerun_MD:
        from pydmclab.core.struc import StrucTools
        import os

        if os.path.exists(ftraj):
            os.remove(ftraj)
        if os.path.exists(flog):
            os.remove(flog)

        s = StrucTools(
            os.path.join("..", "data", "test_data", "launch", "Mn1O1.vasp")
        ).structure

        T, nsteps, loginterval = 1800, 10000, 100
        md = CHGNetMD(
            structure=s,
            model="0.3.0",
            temperature=T,
            relax_first=False,
            trajectory=ftraj,
            logfile=flog,
            loginterval=loginterval,
        )
        md.run(steps=nsteps)

    amd = AnalyzeMD(flog, ftraj)

    return amd


if __name__ == "__main__":
    amd = main()

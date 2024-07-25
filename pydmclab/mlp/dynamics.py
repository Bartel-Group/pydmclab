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
from chgnet.utils import determine_device

import numpy as np
import ase.optimize as opt
from ase import Atoms
from ase import filters
from ase.filters import Filter
from ase.calculators.calculator import Calculator, all_changes, all_properties

from pymatgen.core.structure import Structure
from pymatgen.io.ase import AseAtomsAdaptor

from torch import Tensor

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
            free_energy=model_prediction["e"] * factor,
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
        self.atom_positions: list[np.ndarray] = []
        self.cells: list[np.ndarray] = []
        self.atomic_numbers: list[int] = []

    def __call__(self) -> None:
        """The logic for saving the properties of an Atoms during the relaxation."""
        self.energies.append(self.atoms.get_potential_energy())
        self.forces.append(self.atoms.get_forces())
        self.stresses.append(self.atoms.get_stress())
        self.magmoms.append(self.atoms.get_magnetic_moments())
        self.atom_positions.append(self.atoms.get_positions())
        self.cells.append(self.atoms.get_cell()[:])
        self.atomic_numbers.append(self.atoms.get_atomic_numbers())

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
            "atom_positions": self.atom_positions,
            "cell": self.cells,
            "atomic_number": self.atomic_numbers,
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
        assign_magmoms: bool = True,
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
        if assign_magmoms:
            for key in struct.site_properties:
                struct.remove_site_property(property_name=key)
            struct.add_site_property(
                "magmom", [float(magmom) for magmom in atoms.get_magnetic_moments()]
            )

        return {"final_structure": struct, "trajectory": obs}

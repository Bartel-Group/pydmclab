from __future__ import annotations

import logging
import os
import sys
import io
import contextlib
import inspect
import pickle as pkl
from enum import Enum
from functools import partial
from typing import TYPE_CHECKING, Literal

import numpy as np
import ase.optimize as opt
from ase import filters
from ase.filters import Filter
from ase.calculators.calculator import Calculator
from ase.stress import full_3x3_to_voigt_6_stress
from ase.io.jsonio import encode, decode

from fairchem.core.calculate import pretrained_mlip
from fairchem.core.datasets import data_list_collater
from fairchem.core.datasets.atomic_data import AtomicData
from fairchem.core.units.mlip_unit.api.inference import (
    CHARGE_RANGE,
    DEFAULT_CHARGE,
    DEFAULT_SPIN,
    DEFAULT_SPIN_OMOL,
    SPIN_RANGE,
    InferenceSettings,
    UMATask,
)

from pymatgen.core.structure import Structure
from pymatgen.io.ase import AseAtomsAdaptor

from pydmclab.mlp.fairchem.utils import (
    MixedPBCError,
    AllZeroUnitCellError,
)
from pydmclab.utils.handy import convert_numpy_to_native

if TYPE_CHECKING:
    from ase import Atoms
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


class FAIRChemCalculator(Calculator):
    def __init__(
        self,
        name_or_path: str = "uma-s-1",
        task_name: UMATask | str | None = "omat",
        inference_settings: InferenceSettings | str = "default",
        overrides: dict | None = None,
        device: Literal["cuda", "cpu"] | None = None,
        seed: int = 42,
    ):
        """
        UMA ASE Calculator

        See fairchem-core package for more details

        Args:
            name_or_path: A model name from fairchem.core.pretrained.available_models or a path to a checkpoint file.
            task_name (UMATask or str, optional): Name of the task to use if using a UMA checkpoint.
                Determines default key names for energy, forces, and stress.
                Can be one of 'omol', 'omat', 'oc20', 'odac', or 'omc'.
            inference_settings: Settings for inference. Can be "default" (general purpose) or "turbo"
                (optimized for speed but requires fixed atomic composition).
            overrides: Optional dictionary of settings to override default inference settings.
            device: Optional torch device to load the model onto. If None, uses the default device.
            seed (int, optional): Random seed for reproducibility.
        """

        super().__init__()

        available_models = pretrained_mlip.available_models

        if name_or_path in available_models:
            predict_unit = pretrained_mlip.get_predict_unit(
                name_or_path,
                inference_settings=inference_settings,
                overrides=overrides,
                device=device,
            )
        elif os.path.isfile(name_or_path):
            predict_unit = pretrained_mlip.load_predict_unit(
                name_or_path,
                inference_settings=inference_settings,
                overrides=overrides,
                device=device,
            )
        else:
            raise ValueError(
                f"{name_or_path=} is not a valid model name or checkpoint path, pretrained models include {available_models}"
            )

        if predict_unit.inference_mode.external_graph_gen is not False:
            raise RuntimeError(
                "FAIRChemCalculator can only be used with external_graph_gen True inference settings."
            )

        if predict_unit.model.module.backbone.direct_forces:
            logging.warning(
                "This is a direct-force model. Direct force predictions may lead to discontinuities in the potential "
                "energy surface and energy conservation errors."
            )

        if isinstance(task_name, UMATask):
            task_name = task_name.value

        if task_name is not None:
            assert (
                task_name in predict_unit.datasets
            ), f"Given: {task_name}, Valid options are {predict_unit.datasets}"
            self._task_name = task_name
        elif len(predict_unit.datasets) == 1:
            self._task_name = predict_unit.datasets[0]
        else:
            raise RuntimeError(
                f"A task name must be provided. Valid options are {predict_unit.datasets}"
            )

        self.implemented_properties = [
            task.property for task in predict_unit.dataset_to_tasks[self.task_name]
        ]
        if "energy" in self.implemented_properties:
            self.implemented_properties.append(
                "free_energy"
            )  # free_energy is a copy of energy, see calculate method docstring

        self.predictor = predict_unit
        self.predictor.seed(seed)

        self.a2g = partial(
            AtomicData.from_ase,
            max_neigh=self.predictor.model.module.backbone.max_neighbors,
            radius=self.predictor.model.module.backbone.cutoff,
            task_name=self.task_name,
            r_edges=False,
            r_data_keys=["spin", "charge"],
        )

    @property
    def task_name(self) -> str:
        return self._task_name

    def check_state(self, atoms: Atoms, tol: float = 1e-15) -> list:
        """
        Check for any system changes since the last calculation.

        Args:
            atoms (ase.Atoms): The atomic structure to check.
            tol (float): Tolerance for detecting changes.

        Returns:
            list: A list of changes detected in the system.
        """
        state = super().check_state(atoms, tol=tol)
        if (not state) and (self.atoms.info != atoms.info):
            state.append("info")
        return state

    def calculate(
        self, atoms: Atoms, properties: list[str], system_changes: list[str]
    ) -> None:
        """
        Perform the calculation for the given atomic structure.

        Args:
            atoms (Atoms): The atomic structure to calculate properties for.
            properties (list[str]): The list of properties to calculate.
            system_changes (list[str]): The list of changes in the system.

        Notes:
            - `charge` must be an integer representing the total charge on the system and can range from -100 to 100.
            - `spin` must be an integer representing the spin multiplicity and can range from 0 to 100.
            - If `task_name="omol"`, and `charge` or `spin` are not set in `atoms.info`, they will default to `0`.
            - `charge` and `spin` are currently only used for the `omol` head.
            - The `free_energy` is simply a copy of the `energy` and is not the actual electronic free energy.
              It is only set for ASE routines/optimizers that are hard-coded to use this rather than the `energy` key.
        """

        # Our calculators won't work if natoms=0
        if len(atoms) == 0:
            raise ValueError("Atoms object has no atoms inside.")

        # Check if the atoms object has periodic boundary conditions (PBC) set correctly
        self._check_atoms_pbc(atoms)

        # Validate that charge/spin are set correctly for omol, or default to 0 otherwise
        self._validate_charge_and_spin(atoms)

        # Standard call to check system_changes etc
        Calculator.calculate(self, atoms, properties, system_changes)

        if len(atoms) == 1 and sum(atoms.pbc) == 0:
            self.results = self._get_single_atom_energies(atoms)
        else:
            # Convert using the current a2g object
            data_object = self.a2g(atoms)

            # Batch and predict
            batch = data_list_collater([data_object], otf_graph=True)
            pred = self.predictor.predict(batch)

            # Collect the results into self.results
            self.results = {}
            for calc_key in self.implemented_properties:
                if calc_key == "energy":
                    energy = float(pred[calc_key].detach().cpu().numpy()[0])

                    self.results["energy"] = self.results["free_energy"] = (
                        energy  # Free energy is a copy of energy
                    )
                if calc_key == "forces":
                    forces = pred[calc_key].detach().cpu().numpy()
                    self.results["forces"] = forces
                if calc_key == "stress":
                    stress = pred[calc_key].detach().cpu().numpy().reshape(3, 3)
                    stress_voigt = full_3x3_to_voigt_6_stress(stress)
                    self.results["stress"] = stress_voigt

    def _get_single_atom_energies(self, atoms) -> dict:
        """
        Populate output with single atom energies
        """
        if self.predictor.atom_refs is None:
            raise ValueError(
                "Single atom system but no atomic references present. "
                "Please call fairchem.core.pretrained_mlip.get_predict_unit() "
                "with an appropriate checkpoint name."
            )
        logging.warning(
            "Single atom systems are not handled by the model; "
            "the precomputed DFT result is returned. "
            "Spin multiplicity is ignored for monoatomic systems."
        )
        elt = atoms.get_atomic_numbers()[0]
        results = {}

        atom_refs = self.predictor.atom_refs[self.task_name]
        try:
            energy = atom_refs.get(int(elt), {}).get(atoms.info["charge"])
        except AttributeError:
            energy = atom_refs[int(elt)]
        if energy is None:
            raise ValueError("This model has not stored this element with this charge.")
        results["energy"] = energy
        results["forces"] = np.array([[0.0] * 3])
        results["stress"] = np.array([0.0] * 6)
        return results

    def _check_atoms_pbc(self, atoms) -> None:
        """
        Check for invalid PBC conditions

        Args:
            atoms (ase.Atoms): The atomic structure to check.
        """
        if np.all(atoms.pbc) and np.allclose(atoms.cell, 0):
            raise AllZeroUnitCellError
        if np.any(atoms.pbc) and not np.all(atoms.pbc):
            raise MixedPBCError

    def _validate_charge_and_spin(self, atoms: Atoms) -> None:
        """
        Validate and set default values for charge and spin.

        Args:
            atoms (Atoms): The atomic structure containing charge and spin information.
        """

        if "charge" not in atoms.info:
            if self.task_name == UMATask.OMOL.value:
                logging.warning(
                    "task_name='omol' detected, but charge is not set in atoms.info. Defaulting to charge=0. "
                    "Ensure charge is an integer representing the total charge on the system and is within the range -100 to 100."
                )
            atoms.info["charge"] = DEFAULT_CHARGE

        if "spin" not in atoms.info:
            if self.task_name == UMATask.OMOL.value:
                atoms.info["spin"] = DEFAULT_SPIN_OMOL
                logging.warning(
                    "task_name='omol' detected, but spin multiplicity is not set in atoms.info. Defaulting to spin=1. "
                    "Ensure spin is an integer representing the spin multiplicity from 0 to 100."
                )
            else:
                atoms.info["spin"] = DEFAULT_SPIN

        # Validate charge
        charge = atoms.info["charge"]
        if not isinstance(charge, int):
            raise TypeError(
                f"Invalid type for charge: {type(charge)}. Charge must be an integer representing the total charge on the system."
            )
        if not (CHARGE_RANGE[0] <= charge <= CHARGE_RANGE[1]):
            raise ValueError(
                f"Invalid value for charge: {charge}. Charge must be within the range {CHARGE_RANGE[0]} to {CHARGE_RANGE[1]}."
            )

        # Validate spin
        spin = atoms.info["spin"]
        if not isinstance(spin, int):
            raise TypeError(
                f"Invalid type for spin: {type(spin)}. Spin must be an integer representing the spin multiplicity."
            )
        if not (SPIN_RANGE[0] <= spin <= SPIN_RANGE[1]):
            raise ValueError(
                f"Invalid value for spin: {spin}. Spin must be within the range {SPIN_RANGE[0]} to {SPIN_RANGE[1]}."
            )


class FAIRChemObserver:
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
        self.atomic_numbers: list[int] = []
        self.atom_positions: list[np.ndarray] = []
        self.cells: list[np.ndarray] = []

    def __call__(self) -> None:
        """The logic for saving the properties of an Atoms during the relaxation."""
        self.energies.append(self.atoms.get_potential_energy())
        self.forces.append(self.atoms.get_forces())
        self.stresses.append(self.atoms.get_stress())
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
            "atomic_numbers": self.atomic_numbers,
            "atom_positions": self.atom_positions,
            "cells": self.cells,
        }

    @classmethod
    def from_dict(cls, data_dict: dict[str, list]) -> Self:
        """Create a TrajectoryObserver from a dictionary."""
        obs = cls(decode(data_dict["atoms"]))
        obs.energies = data_dict["energies"]
        obs.forces = data_dict["forces"]
        obs.stresses = data_dict["stresses"]
        obs.atomic_numbers = data_dict["atomic_numbers"]
        obs.atom_positions = data_dict["atom_positions"]
        obs.cells = data_dict["cells"]
        return obs

    def save(self, filename: str) -> None:
        """Save the trajectory to file.

        Args:
            filename (str): filename to save the trajectory
        """
        out_pkl = self.as_dict()
        with open(filename, "wb") as file:
            pkl.dump(out_pkl, file)


class FAIRChemRelaxer:
    """Wrapper class for structural relaxation."""

    def __init__(
        self,
        name_or_path: str = "uma-s-1",
        task_name: UMATask | str | None = "omat",
        inference_settings: InferenceSettings | str = "default",
        overrides: dict | None = None,
        device: Literal["cuda", "cpu"] | None = None,
        seed: int = 42,
        optimizer: ASEOptimizer | str = "FIRE",
    ) -> None:

        self.optimizer: ASEOptimizer = (
            OPTIMIZERS[optimizer.lower()].value
            if isinstance(optimizer, str)
            else optimizer
        )
        self.calculator = FAIRChemCalculator(
            name_or_path=name_or_path,
            task_name=task_name,
            inference_settings=inference_settings,
            overrides=overrides,
            device=device,
            seed=seed,
        )

    def predict_structure(
        self,
        atoms: Structure | Atoms,
    ) -> dict:

        if isinstance(atoms, Structure):
            atoms = AseAtomsAdaptor().get_atoms(atoms)

        atoms.calc = self.calculator

        results = {
            "energy": atoms.get_potential_energy(),
            "forces": atoms.get_forces(),
            "stresses": atoms.get_stress(),
        }

        return results

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
    ) -> dict:
        """Relax the Structure/ Atoms until max force is less than fmax"""

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

            obs = FAIRChemObserver(atoms)

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
            obs = FAIRChemObserver.from_dict(native_obs)

        return {
            "final_structure": struc,
            "final_energy": obs.energies[-1],
            "trajectory": obs,
        }

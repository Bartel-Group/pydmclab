from __future__ import annotations

import logging
import os
import sys
import io
import contextlib
import inspect
import warnings
import pickle as pkl
from enum import Enum
from math import sqrt
from functools import partial
from typing import TYPE_CHECKING, Literal

import numpy as np
import ase.optimize as opt
from ase import filters, units
from ase.filters import Filter
from ase.calculators.calculator import Calculator
from ase.stress import full_3x3_to_voigt_6_stress
from ase.io.trajectory import Trajectory
from ase.io.jsonio import encode, decode
from ase.io import write
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution, Stationary
from ase.md.verlet import VelocityVerlet
from ase.md.npt import NPT
from ase.md.nptberendsen import Inhomogeneous_NPTBerendsen, NPTBerendsen
from ase.md.nvtberendsen import NVTBerendsen

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
from pymatgen.analysis.eos import BirchMurnaghan

# from pymatgen.analysis.diffusion.aimd.pathway import ProbabilityDensityAnalysis

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
    """
    Trajectory observer is a hook in the relaxation process that saves the
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
        self.fmaxs: list[float] = []
        self.stresses: list[np.ndarray] = []
        self.atomic_numbers: list[int] = []
        self.atom_positions: list[np.ndarray] = []
        self.cells: list[np.ndarray] = []

    def __call__(self) -> None:
        """The logic for saving the properties of an Atoms during the relaxation."""
        self.energies.append(self.atoms.get_potential_energy())
        self.forces.append(self.atoms.get_forces())
        self.fmaxs.append(sqrt((self.atoms.get_forces() ** 2).sum(axis=1).max()))
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
            "fmaxs": self.fmaxs,
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
        obs.fmaxs = data_dict["fmaxs"]
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
    """Structure relaxation class"""

    def __init__(
        self,
        name_or_path: str = "uma-s-1",
        task_name: UMATask | str | None = "omat",
        calculator: FAIRChemCalculator | None = None,
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
        if isinstance(calculator, FAIRChemCalculator):
            self.calculator = calculator
            raise UserWarning(
                "Using provided Calculator instead of generating new one."
            )
        else:
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
        include_obs_in_results: bool = True,
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

        struc = AseAtomsAdaptor().get_structure(atoms)

        if convert_to_native_types:
            native_struc = convert_numpy_to_native(struc.as_dict())
            struc = Structure.from_dict(native_struc)

            native_obs = convert_numpy_to_native(obs.as_dict())
            obs = FAIRChemObserver.from_dict(native_obs)
            
         # Ensure structure is JSON serializable by converting to dict
        serializable_structure = struc.as_dict() if hasattr(struc, 'as_dict') else struc

        return {
            "final_structure": serializable_structure,
            "final_energy": obs.energies[-1],
            "converged": obs.fmaxs[-1] < fmax if obs.fmaxs else False,
            "trajectory": obs if include_obs_in_results else None,
        }


class EquationOfState:
    """Class to calculate equation of state."""

    def __init__(
        self,
        name_or_path: str = "uma-s-1",
        task_name: UMATask | str | None = "omat",
        calculator: FAIRChemCalculator | None = None,
        inference_settings: InferenceSettings | str = "default",
        overrides: dict | None = None,
        device: Literal["cuda", "cpu"] | None = None,
        seed: int = 42,
        optimizer: ASEOptimizer | str = "FIRE",
    ) -> None:
        """Initialize a structure optimizer object for calculation of bulk modulus.

        Args:
            name_or_path (str): A model name from fairchem.core.pretrained.available_models or a path to a checkpoint file.
            task_name (UMATask or str, optional): Name of the task to use if using a UMA checkpoint.
                Determines default key names for energy, forces, and stress.
                Can be one of 'omol', 'omat', 'oc20', 'odac', or 'omc'.
            calculator: if provided, use input FAIRChemCalculator
            inference_settings (InferenceSettings | str): Settings for inference. Can be "default" (general purpose) or "turbo"
                (optimized for speed but requires fixed atomic composition).
            overrides (dict | None): Optional dictionary of settings to override default inference settings.
            device (Literal["cuda", "cpu"] | None): Optional torch device to load the model onto. If None, uses the default device.
            seed (int, optional): Random seed for reproducibility.
            optimizer (ASEOptimizer | str): The ASE optimizer to use for relaxation.
        """

        if isinstance(calculator, FAIRChemCalculator):
            self.relaxer = FAIRChemRelaxer(
                calculator=calculator,
                optimizer=optimizer,
            )
        else:
            self.relaxer = FAIRChemRelaxer(
                name_or_path=name_or_path,
                task_name=task_name,
                inference_settings=inference_settings,
                overrides=overrides,
                device=device,
                seed=seed,
                optimizer=optimizer,
            )

        self.fitted = False

    def fit(
        self,
        atoms: Structure | Atoms,
        *,
        n_points: int = 11,
        fmax: float | None = 0.1,
        steps: int | None = 500,
        verbose: bool | None = False,
        **kwargs,
    ) -> None:
        """Relax the Structure/Atoms and fit the Birch-Murnaghan equation of state.

        Args:
            atoms (Structure | Atoms): A Structure or Atoms object to relax.
            n_points (int): Number of structures used in fitting the equation of states
            fmax (float | None): The maximum force tolerance for relaxation.
                Default = 0.1
            steps (int | None): The maximum number of steps for relaxation.
                Default = 500
            verbose (bool): Whether to print the output of the ASE optimizer.
                Default = False
            **kwargs: Additional parameters for the optimizer.
        """
        if isinstance(atoms, Atoms):
            atoms = AseAtomsAdaptor.get_structure(atoms)
        primitive_cell = atoms.get_primitive_structure()
        local_minima = self.relaxer.relax(
            primitive_cell,
            relax_cell=True,
            fmax=fmax,
            steps=steps,
            verbose=verbose,
            **kwargs,
        )

        volumes, energies = [], []
        for idx in np.linspace(-0.1, 0.1, n_points):
            structure_strained = local_minima["final_structure"].copy()
            structure_strained.apply_strain([idx, idx, idx])
            result = self.relaxer.relax(
                structure_strained,
                relax_cell=False,
                fmax=fmax,
                steps=steps,
                verbose=verbose,
                **kwargs,
            )
            volumes.append(result["final_structure"].volume)
            energies.append(result["trajectory"].energies[-1])
        self.bm = BirchMurnaghan(volumes=volumes, energies=energies)
        self.bm.fit()
        self.fitted = True

    def get_bulk_modulus(self, unit: Literal["eV/A^3", "GPa"] = "eV/A^3") -> float:
        """Get the bulk modulus of from the fitted Birch-Murnaghan equation of state.

        Args:
            unit (str): The unit of bulk modulus. Can be "eV/A^3" or "GPa"
                Default = "eV/A^3"

        Returns:
            float: Bulk Modulus

        Raises:
            ValueError: If the equation of state is not fitted.
        """
        if self.fitted is False:
            raise ValueError(
                "Equation of state needs to be fitted first through self.fit()"
            )
        if unit == "eV/A^3":
            return self.bm.b0
        if unit == "GPa":
            return self.bm.b0_GPa
        raise ValueError("unit has to be eV/A^3 or GPa")

    def get_compressibility(self, unit: str = "A^3/eV") -> float:
        """Get the bulk modulus of from the fitted Birch-Murnaghan equation of state.

        Args:
            unit (str): The unit of bulk modulus. Can be "A^3/eV",
            "GPa^-1" "Pa^-1" or "m^2/N"
                Default = "A^3/eV"

        Returns:
            Bulk Modulus (float)
        """
        if self.fitted is False:
            raise ValueError(
                "Equation of state needs to be fitted first through self.fit()"
            )
        if unit == "A^3/eV":
            return 1 / self.bm.b0
        if unit == "GPa^-1":
            return 1 / self.bm.b0_GPa
        if unit in {"Pa^-1", "m^2/N"}:
            return 1 / (self.bm.b0_GPa * 1e9)
        raise NotImplementedError("unit has to be one of A^3/eV, GPa^-1 Pa^-1 or m^2/N")


class FAIRChemMD:
    """Molecular dynamics class"""

    def __init__(
        self,
        atoms: Atoms | Structure,
        *,
        name_or_path: str = "uma-s-1",
        task_name: UMATask | str | None = "omat",
        calculator: FAIRChemCalculator | None = None,
        ensemble: str = "nvt",
        thermostat: str = "Berendsen_inhomogeneous",
        starting_temperature: int | None = None,
        temperature: int = 300,
        pressure: float = 1.01325e-4,
        timestep: float = 2.0,
        taut: float | None = None,
        taup: float | None = None,
        bulk_modulus: float | None = None,
        loginterval: int = 10,
        logfile: str = "md.log",
        trajfile: str = "md.traj",
        append_trajectory: bool = False,
        inference_settings: InferenceSettings | str = "default",
        overrides: dict | None = None,
        device: Literal["cuda", "cpu"] | None = None,
        seed: int = 42,
    ) -> None:

        self.ensemble = ensemble
        self.thermostat = thermostat

        if isinstance(atoms, Structure):
            atoms = AseAtomsAdaptor().get_atoms(atoms)

        if starting_temperature is not None:
            MaxwellBoltzmannDistribution(
                atoms, temperature_K=starting_temperature, force_temp=True
            )
            Stationary(atoms)

        self.atoms = atoms
        if isinstance(calculator, FAIRChemCalculator):
            self.atoms.calc = calculator
            raise UserWarning(
                "Using provided Calculator instead of generating new one."
            )
        else:
            self.atoms.calc = FAIRChemCalculator(
                name_or_path=name_or_path,
                task_name=task_name,
                inference_settings=inference_settings,
                overrides=overrides,
                device=device,
                seed=seed,
            )

        if taut is None:
            taut = 100 * timestep
        if taup is None:
            taup = 1000 * timestep

        if ensemble.lower() == "nve":

            self.dyn = VelocityVerlet(
                atoms=self.atoms,
                timestep=timestep * units.fs,
                trajectory=trajfile,
                logfile=logfile,
                loginterval=loginterval,
                append_trajectory=append_trajectory,
            )
            print("NVE-MD created")

        elif ensemble.lower() == "nvt":

            if thermostat.lower() == "nose-hoover":

                self.upper_triangular_cell()
                self.dyn = NPT(
                    atoms=self.atoms,
                    timestep=timestep * units.fs,
                    temperature_K=temperature,
                    externalstress=pressure * units.GPa,
                    ttime=taut * units.fs,
                    pfactor=None,
                    trajectory=trajfile,
                    logfile=logfile,
                    loginterval=loginterval,
                    append_trajectory=append_trajectory,
                )
                print("NVT-Nose-Hoover MD created")

            elif thermostat.lower().startswith("berendsen"):

                self.dyn = NVTBerendsen(
                    atoms=self.atoms,
                    timestep=timestep * units.fs,
                    temperature_K=temperature,
                    taut=taut * units.fs,
                    trajectory=trajfile,
                    logfile=logfile,
                    loginterval=loginterval,
                    append_trajectory=append_trajectory,
                )
                print("NVT-Berendsen-MD created")

            else:

                raise ValueError(
                    "Thermostat not supported, choose in 'Nose-Hoover', 'Berendsen', "
                    "'Berendsen_inhomogeneous'"
                )

        elif ensemble.lower() == "npt":

            if bulk_modulus is not None:
                bulk_modulus_au = bulk_modulus / 160.2176  # GPa to eV/A^3
                compressibility_au = 1 / bulk_modulus_au
            else:
                try:
                    eos = EquationOfState(calculator=self.atoms.calc)
                    eos.fit(atoms=atoms, steps=500, fmax=0.1, verbose=False)
                    bulk_modulus = eos.get_bulk_modulus(unit="GPa")
                    bulk_modulus_au = eos.get_bulk_modulus(unit="eV/A^3")
                    compressibility_au = eos.get_compressibility(unit="A^3/eV")
                    print(
                        f"Completed bulk modulus calculation: "
                        f"k = {bulk_modulus:.3}GPa, {bulk_modulus_au:.3}eV/A^3"
                    )
                except Exception:
                    bulk_modulus_au = 2 / 160.2176
                    compressibility_au = 1 / bulk_modulus_au
                    warnings.warn(
                        "Warning!!! Equation of State fitting failed, setting bulk "
                        "modulus to 2 GPa. NPT simulation can proceed with incorrect "
                        "pressure relaxation time."
                        "User input for bulk modulus is recommended.",
                        stacklevel=2,
                    )
            self.bulk_modulus = bulk_modulus

            if thermostat.lower() == "nose-hoover":

                self.upper_triangular_cell()
                ptime = taup * units.fs
                self.dyn = NPT(
                    atoms=self.atoms,
                    timestep=timestep * units.fs,
                    temperature_K=temperature,
                    externalstress=pressure * units.GPa,
                    ttime=taut * units.fs,
                    pfactor=bulk_modulus * units.GPa * ptime * ptime,
                    trajectory=trajfile,
                    logfile=logfile,
                    loginterval=loginterval,
                    append_trajectory=append_trajectory,
                )
                print("NPT-Nose-Hoover MD created")

            elif thermostat.lower() == "berendsen_inhomogeneous":

                # Inhomogeneous_NPTBerendsen thermo/barostat
                # This is a more flexible scheme that fixes three angles of the unit
                # cell but allows three lattice parameter to change independently.
                # see: https://gitlab.com/ase/ase/-/blob/master/ase/md/nptberendsen.py

                self.dyn = Inhomogeneous_NPTBerendsen(
                    atoms=self.atoms,
                    timestep=timestep * units.fs,
                    temperature_K=temperature,
                    pressure_au=pressure * units.GPa,
                    taut=taut * units.fs,
                    taup=taup * units.fs,
                    compressibility_au=compressibility_au,
                    trajectory=trajfile,
                    logfile=logfile,
                    loginterval=loginterval,
                )
                print("NPT-Berendsen-inhomogeneous-MD created")

            elif thermostat.lower() == "npt_berendsen":

                # This is a similar scheme to the Inhomogeneous_NPTBerendsen.
                # This is a less flexible scheme that fixes the shape of the
                # cell - three angles are fixed and the ratios between the three
                # lattice constants.
                # see: https://gitlab.com/ase/ase/-/blob/master/ase/md/nptberendsen.py

                self.dyn = NPTBerendsen(
                    atoms=self.atoms,
                    timestep=timestep * units.fs,
                    temperature_K=temperature,
                    pressure_au=pressure * units.GPa,
                    taut=taut * units.fs,
                    taup=taup * units.fs,
                    compressibility_au=compressibility_au,
                    trajectory=trajfile,
                    logfile=logfile,
                    loginterval=loginterval,
                    append_trajectory=append_trajectory,
                )
                print("NPT-Berendsen-MD created")

            else:

                raise ValueError(
                    "Thermostat not supported, choose in 'Nose-Hoover', 'Berendsen', "
                    "'Berendsen_inhomogeneous'"
                )

        self.trajectory = trajfile
        self.logfile = logfile
        self.loginterval = loginterval
        self.timestep = timestep

        return

    @classmethod
    def continue_from_traj(
        cls,
        *,
        name_or_path: str,
        task_name: UMATask | str | None = "omat",
        ensemble: str = "nvt",
        thermostat: str = "Berendsen_inhomogeneous",
        starting_temperature: int | None = None,
        temperature: int = 300,
        pressure: float = 1.01325e-4,
        timestep: float = 2.0,
        taut: float | None = None,
        taup: float | None = None,
        bulk_modulus: float | None = None,
        loginterval: int = 10,
        logfile: str = "md.log",
        trajfile: str = "md.traj",
        append_trajectory: bool = True,
        inference_settings: InferenceSettings | str = "default",
        overrides: dict | None = None,
        device: Literal["cuda", "cpu"] | None = None,
        seed: int = 42,
    ) -> Self:
        """Continue an MD simulation from a trajectory file."""
        if not os.path.exists(trajfile):
            raise FileNotFoundError(f"{trajfile} not found")
        with Trajectory(trajfile, "r") as traj:
            starting_structure = traj[-1]
        return cls(
            atoms=starting_structure,
            name_or_path=name_or_path,
            task_name=task_name,
            ensemble=ensemble,
            thermostat=thermostat,
            starting_temperature=starting_temperature,
            temperature=temperature,
            pressure=pressure,
            timestep=timestep,
            taut=taut,
            taup=taup,
            bulk_modulus=bulk_modulus,
            loginterval=loginterval,
            logfile=logfile,
            trajfile=trajfile,
            append_trajectory=append_trajectory,
            inference_settings=inference_settings,
            overrides=overrides,
            device=device,
            seed=seed,
        )

    def run(self, steps: int) -> None:
        """
        hin wrapper of ase MD run.

        Args:
            steps (int): number of MD steps
        """

        self.dyn.run(steps)

    def set_atoms(self, atoms: Atoms) -> None:
        """
        Set new atoms to run MD.

        Args:
            atoms (Atoms): new atoms for running MD
        """

        calculator = self.atoms.calc
        self.atoms = atoms
        self.dyn.atoms = atoms
        self.dyn.atoms.calc = calculator

    def upper_triangular_cell(self, *, verbose: bool | None = False) -> None:
        """Transform to upper-triangular cell.
        ASE Nose-Hoover implementation only supports upper-triangular cell
        while ASE's canonical description is lower-triangular cell.

        Args:
            verbose (bool): Whether to notify user about upper-triangular cell
                transformation. Default = False
        """
        if not NPT._isuppertriangular(self.atoms.get_cell()):  # noqa: SLF001
            a, b, c, alpha, beta, gamma = self.atoms.cell.cellpar()
            angles = np.radians((alpha, beta, gamma))
            sin_a, sin_b, _sin_g = np.sin(angles)
            cos_a, cos_b, cos_g = np.cos(angles)
            cos_p = (cos_g - cos_a * cos_b) / (sin_a * sin_b)
            cos_p = np.clip(cos_p, -1, 1)
            sin_p = (1 - cos_p**2) ** 0.5

            new_basis = [
                (a * sin_b * sin_p, a * sin_b * cos_p, a * cos_b),
                (0, b * sin_a, b * cos_a),
                (0, 0, c),
            ]

            self.atoms.set_cell(new_basis, scale_atoms=True)

            if verbose:
                print("Transformed to upper triangular unit cell.", flush=True)

from __future__ import annotations

import io
import sys
import pickle
import contextlib
import collections

from math import sqrt
from enum import Enum
from typing import TYPE_CHECKING, Literal

import matgl
import numpy as np
import pandas as pd
import scipy.sparse as sp

import ase.optimize as opt

from ase import Atoms, units
from ase.io.jsonio import encode, decode
from ase.calculators.calculator import Calculator, all_changes
from ase.filters import FrechetCellFilter
from ase.stress import full_3x3_to_voigt_6_stress
from pymatgen.core.structure import Molecule, Structure
from pymatgen.io.ase import AseAtomsAdaptor

from pydmclab.utils.handy import convert_numpy_to_native

from matgl.ext.ase import Atoms2Graph

if TYPE_CHECKING:
    from typing import Any

    from pydmclab.mlp.matgl import PretrainedPotentials

    import dgl
    import torch
    from ase.optimize.optimize import Optimizer

    from typing_extensions import Self

    from matgl.apps.pes import Potential


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


class MatGLCalculator(Calculator):
    """Potential calculator for ASE."""

    implemented_properties = [
        "energy",
        "free_energy",
        "forces",
        "stress",
        "hessian",
        "magmoms",
    ]  # noqa:RUF012

    def __init__(
        self,
        potential: Potential | PretrainedPotentials,
        *,
        state_attr: torch.Tensor | None = None,
        stress_unit: Literal["eV/A3", "GPa"] = "GPa",
        stress_weight: float = 1.0,
        use_voigt: bool = False,
        **kwargs,
    ):
        """
        Init PESCalculator with a Potential from matgl.

        Args:
            potential (Potential): matgl.apps.pes.Potential
            state_attr (tensor): State attribute
            compute_stress (bool): whether to calculate the stress
            stress_unit (str): stress unit. Default: "GPa"
            stress_weight (float): conversion factor from GPa to eV/A^3, if it is set to 1.0, the unit is in GPa
            use_voigt (bool): whether the voigt notation is used for stress output
            **kwargs: Kwargs pass through to super().__init__().
        """
        super().__init__(**kwargs)

        if isinstance(potential, str):
            self.potential = matgl.load_model(potential)
        elif isinstance(potential, Potential):
            self.potential = potential
        else:
            raise TypeError(
                f"Unsupported potential type: {type(potential)}. Must be a str or Potential."
            )

        self.compute_stress = self.potential.calc_stresses
        self.compute_hessian = self.potential.calc_hessian
        self.compute_magmom = self.potential.calc_magmom

        self.graph_converter = Atoms2Graph(
            self.potential.model.element_types, self.potential.model.cutoff
        )

        # Handle stress unit conversion
        if stress_unit == "eV/A3":
            conversion_factor = units.GPa / (
                units.eV / units.Angstrom**3
            )  # Conversion factor from GPa to eV/A^3
        elif stress_unit == "GPa":
            conversion_factor = 1.0  # No conversion needed if stress is already in GPa
        else:
            raise ValueError(
                f"Unsupported stress_unit: {stress_unit}. Must be 'GPa' or 'eV/A3'."
            )

        self.stress_weight = stress_weight * conversion_factor
        self.state_attr = state_attr
        self.element_types = self.potential.model.element_types  # type: ignore
        self.cutoff = self.potential.model.cutoff
        self.use_voigt = use_voigt

    def calculate(  # type:ignore[override]
        self,
        atoms: Atoms,
        properties: list | None = None,
        system_changes: list | None = None,
    ):
        """
        Perform calculation for an input Atoms.

        Args:
            atoms (ase.Atoms): ase Atoms object
            properties (list): list of properties to calculate
            system_changes (list): monitor which properties of atoms were
                changed for new calculation. If not, the previous calculation
                results will be loaded.
        """
        properties = properties or ["energy"]
        system_changes = system_changes or all_changes
        super().calculate(
            atoms=atoms, properties=properties, system_changes=system_changes
        )
        graph, lattice, state_attr_default = self.graph_converter.get_graph(atoms)
        # type: ignore
        if self.state_attr is not None:
            calc_result = self.potential(graph, lattice, self.state_attr)
        else:
            calc_result = self.potential(graph, lattice, state_attr_default)
        self.results.update(
            energy=calc_result[0].detach().cpu().numpy().item(),
            free_energy=calc_result[0].detach().cpu().numpy().item(),
            forces=calc_result[1].detach().cpu().numpy(),
        )
        if self.compute_stress:
            stresses_np = (
                full_3x3_to_voigt_6_stress(calc_result[2].detach().cpu().numpy())
                if self.use_voigt
                else calc_result[2].detach().cpu().numpy()
            )
            self.results.update(stress=stresses_np * self.stress_weight)
        if self.compute_hessian:
            self.results.update(hessian=calc_result[3].detach().cpu().numpy())
        if self.compute_magmom:
            self.results.update(magmoms=calc_result[4].detach().cpu().numpy())


class MatGLObserver(collections.abc.Sequence):
    """Trajectory observer is a hook in the relaxation process that saves the
    intermediate structures.
    """

    def __init__(self, atoms: Atoms) -> None:
        """
        Init the Trajectory Observer from a Atoms.

        Args:
            atoms (Atoms): Structure to observe.
        """
        self.atoms = atoms
        self.energies: list[float] = []
        self.forces: list[np.ndarray] = []
        self.fmaxs: list[float] = []
        self.stresses: list[np.ndarray] = []
        self.atom_positions: list[np.ndarray] = []
        self.cells: list[np.ndarray] = []

    def __call__(self) -> None:
        """The logic for saving the properties of an Atoms during the relaxation."""
        self.energies.append(float(self.atoms.get_potential_energy()))
        self.forces.append(self.atoms.get_forces())
        self.fmaxs.append(sqrt((self.atoms.get_forces() ** 2).sum(axis=1).max()))
        if self.atoms.calc.compute_stress:
            self.stresses.append(self.atoms.get_stress())
        self.atom_positions.append(self.atoms.get_positions())
        if self.atoms.pbc.any():
            self.cells.append(self.atoms.get_cell()[:])

    def __getitem__(self, item):
        return (
            self.energies[item],
            self.forces[item],
            self.stresses[item],
            self.cells[item],
            self.atom_positions[item],
        )

    def __len__(self):
        return len(self.energies)

    def as_pandas(self) -> pd.DataFrame:
        """Returns: DataFrame of energies, forces, stresses, cells and atom_positions."""
        return pd.DataFrame(
            {
                "energies": self.energies,
                "forces": self.forces,
                "stresses": self.stresses,
                "cells": self.cells,
                "atom_positions": self.atom_positions,
            }
        )

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
            "atom_positions": self.atom_positions,
            "cells": self.cells,
        }

    @classmethod
    def from_dict(cls, data: dict[str, list]) -> Self:
        """Create a TrajectoryObserver from a dictionary."""
        obs = cls(decode(data["atoms"]))
        obs.energies = data["energies"]
        obs.forces = data["forces"]
        obs.fmaxs = data["fmaxs"]
        obs.stresses = data["stresses"]
        obs.atom_positions = data["atom_positions"]
        obs.cells = data["cells"]
        return obs

    def save(self, filename: str) -> None:
        """Save the trajectory to file.

        Args:
            filename (str): filename to save the trajectory.
        """
        out = self.as_dict()
        with open(filename, "wb") as file:
            pickle.dump(out, file)


class MatGLRelaxer:
    """Relaxer is a class for structural relaxation."""

    def __init__(
        self,
        potential: Potential | PretrainedPotentials,
        state_attr: torch.Tensor | None = None,
        optimizer: Optimizer | str = "FIRE",
        stress_weight: float = 1 / 160.21766208,
    ):
        """
        Args:
            potential (Potential): a M3GNet potential, a str path to a saved model or a short name for saved model
            that comes with M3GNet distribution
            state_attr (torch.Tensor): State attr.
            optimizer (str or ase Optimizer): the optimization algorithm.
            Defaults to "FIRE"
            relax_cell (bool): whether to relax the lattice cell
            stress_weight (float): conversion factor from GPa to eV/A^3.
        """
        self.optimizer: Optimizer = (
            OPTIMIZERS[optimizer.lower()].value
            if isinstance(optimizer, str)
            else optimizer
        )
        self.calculator = MatGLCalculator(
            potential=potential,
            state_attr=state_attr,
            stress_weight=stress_weight,  # type: ignore
        )
        self.ase_adaptor = AseAtomsAdaptor()


    def predict_structure(
        self,
        atoms: Structure | Atoms,
    ) -> dict[str, Structure]:

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
        atoms: Atoms | Structure | Molecule,
        fmax: float = 0.1,
        steps: int = 500,
        relax_cell: bool = True,
        traj_path: str | None = None,
        include_obs_in_results: bool = True,
        interval: int = 1,
        verbose: bool = False,
        ase_filter: Literal["Frechet", "Exp"] = "Frechet",
        params_asefilter: dict | None = None,
        convert_to_native_types: bool = True,
        **kwargs,
    ):
        """
        Relax an input Atoms.

        Args:
            atoms (Atoms | Structure | Molecule): the atoms for relaxation
            fmax (float): total force tolerance for relaxation convergence.
            Here fmax is a sum of force and stress forces
            steps (int): max number of steps for relaxation
            traj_file (str): the trajectory file for saving
            interval (int): the step interval for saving the trajectories
            verbose (bool): Whether to have verbose output.
            ase_cellfilter (literal): which filter is used for variable cell relaxation. Default is Frechet.
            params_asecellfilter (dict): Parameters to be passed to FrechetCellFilter. Allows
                setting of constant pressure or constant volume relaxations, for example. Refer to
                https://wiki.fysik.dtu.dk/ase/ase/filters.html#FrechetCellFilter for more information.
            **kwargs: Kwargs pass-through to optimizer.
        """
        if isinstance(atoms, Structure | Molecule):
            atoms = self.ase_adaptor.get_atoms(atoms)
        atoms.set_calculator(self.calculator)
        stream = sys.stdout if verbose else io.StringIO()
        params_asecellfilter = params_asecellfilter or {}
        with contextlib.redirect_stdout(stream):
            obs = MatGLObserver(atoms)
            if relax_cell:
                atoms = FrechetCellFilter(
                    atoms, **params_asecellfilter
                )  # type:ignore[assignment]

            optimizer = self.optimizer(atoms, **kwargs)  # type:ignore[operator]
            optimizer.attach(obs, interval=interval)
            optimizer.run(fmax=fmax, steps=steps)
            obs()
        if traj_path is not None:
            obs.save(traj_path)

        if isinstance(atoms, FrechetCellFilter):
            atoms = atoms.atoms

        final_structure: Structure | Molecule
        if isinstance(atoms, Atoms):
            if np.array(atoms.pbc).any():
                final_structure = self.ase_adaptor.get_structure(atoms)
            else:
                final_structure = self.ase_adaptor.get_molecule(atoms)
        elif isinstance(atoms, Structure | Molecule):
            final_structure = atoms
        else:
            raise TypeError(f"Unsupported atoms type: {type(atoms)}")

        if convert_to_native_types:
            native_struc = convert_numpy_to_native(final_structure.as_dict())
            final_structure = Structure.from_dict(native_struc)

            native_obs = convert_numpy_to_native(obs.as_dict())
            obs = MatGLObserver.from_dict(native_obs)

        return {
            "final_structure": final_structure,
            "final_energy": obs.energies[-1],
            "coverged": obs.fmaxs[-1] < fmax if obs.fmaxs else False,
            "trajectory": obs if include_obs_in_results else None,
        }

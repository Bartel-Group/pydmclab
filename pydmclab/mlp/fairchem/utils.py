from __future__ import annotations

import numpy as np
from typing import TYPE_CHECKING, Literal

from fairchem.core.units.mlip_unit.api.inference import (
    InferenceSettings,
    UMATask,
)

from pymatgen.core.structure import Structure
from pymatgen.io.ase import AseAtomsAdaptor
from pymatgen.analysis.eos import BirchMurnaghan
from pydmclab.mlp.fairchem.dynamics import FAIRChemRelaxer

if TYPE_CHECKING:
    from ase import Atoms
    from ase.optimize.optimize import Optimizer as ASEOptimizer


class MixedPBCError(ValueError):
    """Specific exception example."""

    def __init__(
        self,
        message="Attempted to guess PBC for an atoms object, but the atoms object has PBC set to True for some"
        "dimensions but not others. Please ensure that the atoms object has PBC set to True for all dimensions.",
    ):
        self.message = message
        super().__init__(self.message)


class AllZeroUnitCellError(ValueError):
    """Specific exception example."""

    def __init__(
        self,
        message="Atoms object claims to have PBC set, but the unit cell is identically 0. Please ensure that the atoms"
        "object has a non-zero unit cell.",
    ):
        self.message = message
        super().__init__(self.message)


class EquationOfState:
    """Class to calculate equation of state."""

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
        """Initialize a structure optimizer object for calculation of bulk modulus.

        Args:
            name_or_path (str): A model name from fairchem.core.pretrained.available_models or a path to a checkpoint file.
            task_name (UMATask or str, optional): Name of the task to use if using a UMA checkpoint.
                Determines default key names for energy, forces, and stress.
                Can be one of 'omol', 'omat', 'oc20', 'odac', or 'omc'.
            inference_settings (InferenceSettings | str): Settings for inference. Can be "default" (general purpose) or "turbo"
                (optimized for speed but requires fixed atomic composition).
            overrides (dict | None): Optional dictionary of settings to override default inference settings.
            device (Literal["cuda", "cpu"] | None): Optional torch device to load the model onto. If None, uses the default device.
            seed (int, optional): Random seed for reproducibility.
            optimizer (ASEOptimizer | str): The ASE optimizer to use for relaxation.
        """
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

from __future__ import annotations
from typing import TYPE_CHECKING, Literal

import os
import collections
import warnings
from math import sqrt

from matgl.ext.ase import Relaxer

from ase import Atoms
from ase.filters import FrechetCellFilter
from ase.constraints import ExpCellFilter

from pymatgen.core.structure import Structure
from pymatgen.io.vasp import Poscar
from pymatgen.io.ase import AseAtomsAdaptor

from pydmclab.data.configs import load_base_configs
from pydmclab.core.struc import StrucTools
from pydmclab.utils.handy import read_json, write_json, convert_numpy_to_native

if TYPE_CHECKING:
    import torch
    from matgl.apps.pes import Potential
    from ase.optimize.optimize import Optimizer


class FPSetUp(object):
    """
    Use to write inputs for a single FP calculation
    """

    def __init__(self, calc_dir: str, user_configs: dict | None = None) -> None:
        """
        Args:
            calc_dir (str):
                path to directory where FP calculation will be run
            user_configs (dict or None):
                if None, will use default configs
                if dict, will use user_configs
                    these will override any default configs
                    see pydmclab.data.data._hpc_configs for defaults that can be changed
        """

        if user_configs is None:
            user_configs = {}

        self.calc_dir = calc_dir

        fpos = os.path.join(calc_dir, "POSCAR")
        if not os.path.exists(fpos):
            raise FileNotFoundError(f"POSCAR not found in {calc_dir}")
        else:
            structure = Structure.from_file(fpos)
            self.structure = structure

        self.default_configs = load_base_configs()

        self.user_configs = user_configs

    # Don't think need a configs method
    # Need method to prepare the calc dir
    # Need method to check if a calc dir is clean
    # Possibly need some type of error correction?

    @property
    def prepare_calc(self) -> None:
        """
        Write input files (settings and template...)
        """
        pass


class FPRelaxer(Relaxer):
    """
    Custom relaxer building on top of matgl relaxer

    This relaxer extends the following functionality:
    - writes CONTCAR file after each optimization step
    - incrementally updates traj.json file with full trajectory data
    """

    def __init__(
        self,
        potential: Potential | str,
        calc_dir: str,
        state_attr: torch.Tensor | None = None,
        optimizer: Optimizer | str = "FIRE",
        relax_cell: bool = True,
        stress_weight: float = 1 / 160.21766208,
    ):
        """
        Args:
            potential (Potential):
                a matgl potential, a short name for a model that comes with the matgl distribution, or a str path to a saved model
            calc_dir (str):
                directory where output files will be saved (and the FP calc will run)
            state_attr (torch.Tensor):
                state attribute
            optimizer (str or ase Optimizer):
                ase optimization algorithm to use (e.g., "FIRE", "BFGS", "LBFGS", etc.), defaults to "FIRE"
                see https://github.com/materialyzeai/matgl/blob/v1.3.0/src/matgl/ext/ase.py for implemented options
            relax_cell (bool):
                whether to relax the lattice cell
            stress_weight (float):
                conversion factor from GPa to eV/A^3
        """
        super().__init__(
            potential=potential,
            state_attr=state_attr,
            optimizer=optimizer,
            relax_cell=relax_cell,
            stress_weight=stress_weight,
        )
        self.calc_dir = calc_dir

    def relax(
        self,
        atoms: Atoms | Structure,
        fmax: float = 0.03,
        steps: int = 500,
        interval: int = 1,
        traj_file: str = "traj.json",
        ase_cellfilter: Literal["Frechet", "Exp"] = "Frechet",
        params_asecellfilter: dict | None = None,
        **kwargs,
    ):
        """
        Relax an input Atoms

        Implements the custom FPObserver class for writing output files during the relaxation.

        Args:
            atoms (Atoms | Structure):
                the atoms for relaxation
            fmax (float):
                force tolerance for relaxation convergence
            steps (int):
                maximum number of steps to attempt for relaxation
            interval (int):
                step interval for recording the trajectory
            traj_file (str):
                filename to save trajectory information, defaults to "traj.json"
            ase_cellfilter (str):
                ASE cell filter used to control cell relaxation, options are "Frechet" or "Exp, default is "Frechet"
            params_asecellfilter (dict):
                parameters to pass to cell filter to control cell relaxation
                e.g., {"mask":[False,False,True,False,False,False]} to relax in only the z-direction
            **kwargs:
                kwargs to pass to optimizer
        """
        if isinstance(atoms, Structure):
            atoms = self.ase_adaptor.get_atoms(atoms)
        atoms.calc = self.calculator

        if params_asecellfilter is None:
            params_asecellfilter = {}

        obs = FPObserver(atoms=atoms, traj_file=traj_file, calc_dir=self.calc_dir)

        if self.relax_cell:
            atoms = (
                FrechetCellFilter(atoms, **params_asecellfilter)
                if ase_cellfilter == "Frechet"
                else ExpCellFilter(atoms, **params_asecellfilter)
            )

        optimizer = self.optimizer(atoms, **kwargs)
        optimizer.attach(obs, interval=interval)
        optimizer.run(fmax=fmax, steps=steps)
        obs()

        if isinstance(atoms, (FrechetCellFilter, ExpCellFilter)):
            atoms = atoms.atoms

        final_structure = self.ase_adaptor.get_structure(atoms)

        return {
            "final_structure": final_structure,
            "trajectory": obs.as_dict(),
        }


class FPObserver(collections.abc.Sequence):
    """
    Custom trajectory observer that hooks into the relaxation process to save intermediate data.

    This observer not only records energy, force, stress, fmax, and structure data, but also:
    - writes a CONTCAR file each time it is called
    - incrementally updates traj.json file with the full trajectory data
    """

    def __init__(self, atoms: Atoms, traj_file: str, calc_dir: str) -> None:
        """
        Args:
            atoms (Atoms):
                the Atoms object to observe
            traj_file (str):
                name of the .json file for saving trajectory data in calc_dir
            calc_dir (str):
                name of directory where all info relevant to the FP calculation lives
        """
        self.atoms = atoms
        self.traj_file = traj_file
        self.calc_dir = calc_dir
        self.ase_adaptor = AseAtomsAdaptor()

        traj_path = os.path.join(calc_dir, traj_file)
        if os.path.exists(traj_path):
            existing_data = read_json(traj_path)
            required_keys = {"energies", "forces", "stresses", "fmaxs", "structures"}
            if required_keys.issubset(existing_data):
                self.energies = existing_data.get("energies", [])
                self.forces = existing_data.get("forces", [])
                self.stresses = existing_data.get("stresses", [])
                self.fmaxs = existing_data.get("fmaxs", [])
                self.structures = existing_data.get("structures", [])
                if (
                    len(
                        {
                            len(self.energies),
                            len(self.forces),
                            len(self.stresses),
                            len(self.fmaxs),
                            len(self.structures),
                        }
                    )
                    > 1
                ):
                    reason = "data lengths are not consistent"
                    is_valid = False
                else:
                    is_valid = True
            else:
                reason = "required keys are missing"
                is_valid = False
            if not is_valid:
                warnings.warn(
                    f"Removing existing {traj_file} b/c {reason}.", stacklevel=2
                )
                os.remove(traj_path)
                self.energies = []
                self.forces = []
                self.stresses = []
                self.fmaxs = []
                self.structures = []
        else:
            self.energies = []
            self.forces = []
            self.stresses = []
            self.fmaxs = []
            self.structures = []

    def __call__(self) -> None:
        """The logic for saving the properties of an Atoms during the relaxation."""
        atoms = self.atoms
        self.energies.append(float(self.atoms.get_potential_energy()))
        self.forces.append(convert_numpy_to_native(self.atoms.get_forces()))
        self.stresses.append(convert_numpy_to_native(self.atoms.get_stress()))
        self.fmaxs.append(
            sqrt((self.atoms.get_forces() ** 2).sum(axis=1).max())
        )  # this isn't matching the output fmax, NEED TO FIX
        if isinstance(atoms, (FrechetCellFilter, ExpCellFilter)):
            atoms = atoms.atoms
        st = StrucTools(self.ase_adaptor.get_structure(atoms))
        self.structures.append(convert_numpy_to_native(st.structure_as_dict))

        Poscar(st.structure).write_file(os.path.join(self.calc_dir, "CONTCAR"))

        traj_data = {
            "energies": self.energies,
            "forces": self.forces,
            "stresses": self.stresses,
            "fmaxs": self.fmaxs,
            "structures": self.structures,
        }
        traj_data = write_json(traj_data, os.path.join(self.calc_dir, self.traj_file))

    def __getitem__(self, item):
        """Get trajectory data at a specific step by index"""
        return (
            self.energies[item],
            self.forces[item],
            self.stresses[item],
            self.structures[item],
        )

    def __len__(self):
        """The number of steps in the complete trajectory."""
        return len(self.energies)

    def as_dict(self) -> dict[str, list]:
        """Return the trajectory as a dictionary."""
        return {
            "energies": self.energies,
            "forces": self.forces,
            "stresses": self.stresses,
            "fmaxs": self.fmaxs,
            "structures": self.structures,
        }

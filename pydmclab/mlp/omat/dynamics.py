from __future__ import annotations

import os
from enum import Enum
from typing import TYPE_CHECKING, Literal

from fairchem.core import OCPCalculator

from pymatgen.io.ase import AseAtomsAdaptor

if TYPE_CHECKING:
    from ase.atoms import Atoms
    from pymatgen.core.structure import Structure
    from torch_geometric.data import Batch


class CheckpointPaths(Enum):

    small_omat = os.path.join("~", "eqV2_31M_omat.pt")
    medium_omat = os.path.join("~", "eqV2_86M_omat.pt")
    large_omat = os.path.join("~", "eqV2_153M_omat.pt")

    small_omat_mpalex = os.path.join("~", "eqV2_31M_omat_mp_salex.pt")
    medium_omat_mpalex = os.path.join("~", "eqV2_86M_omat_mp_salex.pt")
    large_omat_mpalex = os.path.join("~", "eqV2_153M_omat_mp_salex.pt")

    small_mp = os.path.join("~", "eqV2_31M_mp.pt")
    small_mp_dens = os.path.join("~", "eqV2_dens_31M_mp.pt")
    medium_mp_dens = os.path.join("~", "eqV2_dens_86M_mp.pt")
    large_mp_dens = os.path.join("~", "eqV2_dens_153M_mp.pt")


class OMatCalculator(OCPCalculator):
    """Wrapper for Omat models as calculators for ASE Atoms objects"""

    def __init__(
        self,
        model_size: Literal["small", "medium", "large"] = "small",
        pretrain_data: Literal["mp", "omat"] = "omat",
        finetune_data: Literal["mpalex", "dens"] | None = None,
        trainer: str | None = None,
        use_cpu: bool = True,
        seed: int | None = None,
    ) -> None:

        if not isinstance(model_size, str) or model_size not in [
            "small",
            "medium",
            "large",
        ]:
            raise ValueError("model_size must be one of 'small', 'medium', or 'large'")
        if not isinstance(pretrain_data, str) or pretrain_data not in ["mp", "omat"]:
            raise ValueError("pretrain_data must be one of 'mp' or 'omat'")

        checkpoint = f"{model_size}_{pretrain_data}"

        if finetune_data is not None:
            if not isinstance(finetune_data, str) or finetune_data not in [
                "mpalex",
                "dens",
            ]:
                raise ValueError(
                    "finetune_data must be one of 'mpalex', 'dens', or None"
                )
            checkpoint += f"_{finetune_data}"

        self.checkpoint_path: str = CheckpointPaths[checkpoint].value

        super().__init__(
            checkpoint_path=self.checkpoint_path,
            trainer=trainer,
            cpu=use_cpu,
            seed=seed,
        )

    def calculate(
        self, atoms: Structure | Atoms | Batch, properties, system_changes
    ) -> dict:

        if isinstance(atoms, Structure):
            atoms = AseAtomsAdaptor.get_atoms(atoms)

        super().calculate(atoms, properties, system_changes)

        return self.results

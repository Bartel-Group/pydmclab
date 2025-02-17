from __future__ import annotations

import os
import warnings
import urllib.request
from enum import Enum
from glob import glob
from typing import TYPE_CHECKING, Literal

import torch
import numpy as np

from ase import units
from ase.calculators.mixing import SumCalculator
from ase.calculators.calculator import Calculator, all_changes
from ase.stress import full_3x3_to_voigt_6_stress
from e3nn import o3

from mace import data
from mace.cli.convert_e3nn_cueq import run as run_e3nn_to_cueq
from mace.modules.utils import extract_invariant
from mace.tools import torch_geometric, torch_tools, utils
from mace.tools.compile import prepare
from mace.tools.scripts_utils import extract_model

from pydmclab.mlp.mace.utils import get_model_dtype

if TYPE_CHECKING:
    from ase import Atoms

os.environ["TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD"] = "1"


class MACECHECKPOINTS(Enum):
    """URLS for MACE checkpoints"""

    small = "https://github.com/ACEsuit/mace-mp/releases/download/mace_mp_0/2023-12-10-mace-128-L0_energy_epoch-249.model"
    medium = "https://github.com/ACEsuit/mace-mp/releases/download/mace_mp_0/2023-12-03-mace-128-L1_epoch-199.model"
    large = "https://github.com/ACEsuit/mace-mp/releases/download/mace_mp_0/MACE_MPtrj_2022.9.model"
    small0b = "https://github.com/ACEsuit/mace-mp/releases/download/mace_mp_0b/mace_agnesi_small.model"
    medium0b = "https://github.com/ACEsuit/mace-mp/releases/download/mace_mp_0b/mace_agnesi_medium.model"
    small0b2 = "https://github.com/ACEsuit/mace-mp/releases/download/mace_mp_0b2/mace-small-density-agnesi-stress.model"
    medium0b2 = "https://github.com/ACEsuit/mace-mp/releases/download/mace_mp_0b2/mace-medium-density-agnesi-stress.model"
    large0b2 = "https://github.com/ACEsuit/mace-mp/releases/download/mace_mp_0b2/mace-large-density-agnesi-stress.model"
    medium0b3 = "https://github.com/ACEsuit/mace-mp/releases/download/mace_mp_0b3/mace-mp-0b3-medium.model"
    mediummpa0 = "https://github.com/ACEsuit/mace-mp/releases/download/mace_mpa_0/mace-mpa-0-medium.model"
    mediumomat0 = "https://github.com/ACEsuit/mace-mp/releases/download/mace_omat_0/mace-omat-0-medium.model"


class MACECalculator(Calculator):
    """MACE ASE Calculator
    args:
        models (list[torch.nn.Module] | torch.nn.Module | list[str] | str): path to model or models if a committee is produced
            To make a committee use multiple models, paths, or a wild card notation like mace_*.model
        device (Literal["cpu", "cuda"]): device to run on
            Defaults to "cpu"
        default_dtype (Literal["float32", "float64", "auto"]): default dtype to use
            Defaults to "auto" which will match the model dtype
        model_type (Literal["MACE", "DipoleMACE", "EnergyDipoleMACE"]): type of model to load
            Defaults to "MACE"
        energy_units_to_eV (float): conversion factor for energy units
            Defaults to 1.0
        length_units_to_A (float): conversion factor for length units
            Defaults to 1.0
        charges_key (str): key to use for charges in the ASE Atoms object
            Defaults to "Qs"
        compile_mode (Literal["default", "reduce-overhead", "max-autotune", "max-autotune-no-cudagraphs"] | None): compile mode to use
            Defaults to None which will not attempt to optimize with TorchDynamo
        fullgraph (bool): whether to use fullgraph mode in TorchDynamo
            Defaults to True
        enable_cueq (bool): whether to convert models to CuEq for acceleration
            Defaults to False, only works with MACE models
        **kwargs: additional arguments to pass to the parent ASE Calculator instance

    Dipoles are returned in units of Debye
    """

    def __init__(
        self,
        models: list[torch.nn.Module] | torch.nn.Module | list[str] | str,
        *,
        device: Literal["cpu", "cuda"] = "cpu",
        default_dtype: Literal["float32", "float64", "auto"] = "auto",
        model_type: Literal["MACE", "DipoleMACE", "EnergyDipoleMace"] = "MACE",
        energy_units_to_eV: float = 1.0,
        length_units_to_A: float = 1.0,
        charges_key: str = "Qs",
        compile_mode: (
            Literal[
                "default",
                "reduce-overhead",
                "max-autotune",
                "max-autotune-no-cudagraphs",
            ]
            | None
        ) = None,
        fullgraph: bool = True,
        enable_cueq: bool = False,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.results = {}

        if model_type not in ["MACE", "DipoleMACE", "EnergyDipoleMACE"]:
            raise ValueError(
                f"{model_type} is not supported. Please use one of MACE, DipoleMACE, or EnergyDipoleMACE"
            )

        self.model_type = model_type

        if self.model_type == "MACE":
            self.implemented_properties = [
                "energy",
                "free_energy",
                "node_energy",
                "forces",
                "stress",
            ]
        elif self.model_type == "DipoleMACE":
            self.implemented_properties = ["dipole"]
        elif self.model_type == "EnergyDipoleMACE":
            self.implemented_properties = [
                "energy",
                "free_energy",
                "node_energy",
                "forces",
                "stress",
                "dipole",
            ]

        self.device = torch_tools.init_device(device)

        if not any(isinstance(models, t) for t in [list, str, torch.nn.Module]):
            raise ValueError(
                "Models must be a list of torch.nn.models/paths, a model path, or a torch.nn.module object"
            )

        if isinstance(models, list):
            if all(isinstance(model, str) for model in models):
                self.model_paths = models

                if not all(
                    os.path.exists(model_path) for model_path in self.model_paths
                ):
                    raise ValueError(
                        "Couldn't find one of the MACE model files, please check the model paths"
                    )

                self.num_models = len(self.model_paths)
                self.models = [
                    torch.load(f=model_path, map_location=self.device)
                    for model_path in self.model_paths
                ]

            elif all(isinstance(model, torch.nn.Module) for model in models):
                self.models = models
                self.num_models = len(self.models)
                self.model_paths = None

            else:
                raise ValueError(
                    "List could not be processed. Mixing model paths and models is not supported"
                )

        if isinstance(models, str):
            globbed_model_paths = glob(models)

            if not globbed_model_paths:
                raise ValueError(f"Couldn't find MACE model files: {models}")

            self.model_paths = globbed_model_paths
            self.num_models = len(self.model_paths)
            self.models = [
                torch.load(f=model_path, map_location=self.device)
                for model_path in self.model_paths
            ]

        if isinstance(models, torch.nn.Module):
            self.models = [models]
            self.num_models = 1
            self.model_paths = None

        self.compile_mode = compile_mode

        if enable_cueq:
            assert self.model_type == "MACE", "CuEq only supports MACE models"

            self.compile_mode = None

            print("Attempting to convert models to CuEq for acceleration")

            self.models = [
                run_e3nn_to_cueq(model, device=self.device).to(self.device)
                for model in self.models
            ]

            print("Conversion to CuEq successful")

        if self.num_models > 1:
            print(f"Committee MACE will run with {self.num_models} models")

            if self.model_type in ["MACE", "EnergyDipoleMACE"]:
                self.implemented_properties.extend(
                    ["energies", "energy_var", "forces_comm", "stress_var"]
                )
            elif self.model_type == "DipoleMACE":
                self.implemented_properties.extend(["dipole_var"])

        self.fullgraph = fullgraph

        if self.compile_mode is not None:
            print(f"Attempting to enable torch compile with mode: {compile_mode}")
            self.models = [
                torch.compile(
                    prepare(extract_model)(model=model, map_location=self.device),
                    mode=self.compile_mode,
                    fullgraph=self.fullgraph,
                )
                for model in self.models
            ]
            self.use_compile = True
            print("Compile successful")
        else:
            self.use_compile = False

        # Ensure all models are on the same device
        for model in self.models:
            model.to(self.device)

        r_maxs = [model.r_max.cpu() for model in self.models]
        r_maxs = np.array(r_maxs)
        if not np.all(r_maxs == r_maxs[0]):
            raise ValueError(f"Committee r_max are not all the same {' '.join(r_maxs)}")
        self.r_max = float(r_maxs[0])

        self.energy_units_to_eV = energy_units_to_eV
        self.length_units_to_A = length_units_to_A
        self.z_table = utils.AtomicNumberTable(
            [int(z) for z in self.models[0].atomic_numbers]
        )
        self.charges_key = charges_key

        try:
            self.heads = self.models[0].heads
        except AttributeError:
            self.heads = ["Default"]

        self.default_dtype = default_dtype
        self.model_dtype = get_model_dtype(self.models[0])

        if self.default_dtype == "auto":
            print(
                f"Automatic dtype selected, switching to {self.model_dtype} to match model dtype."
            )
            self.default_dtype = self.model_dtype

        elif self.model_dtype != self.default_dtype:
            print(
                f"Default dtype {self.default_dtype} does not match model dtype {self.model_dtype}, converting models to {self.default_dtype}."
            )
            if self.default_dtype == "float64":
                print(
                    "Using float64 for MACECalculator, which is slower but more accurate. Recommended for geometry optimization."
                )
                self.models = [model.double() for model in self.models]
            elif self.default_dtype == "float32":
                print(
                    "Using float32 for MACECalculator, which is faster but less accurate. Recommended for MD. Use float64 for geometry optimization."
                )
                self.models = [model.float() for model in self.models]

        torch_tools.set_default_dtype(self.default_dtype)

        for model in self.models:
            for param in model.parameters():
                param.requires_grad = False

    def _create_result_tensors(
        self,
        model_type: Literal["MACE", "DipoleMACE", "EnergyDipoleMace"],
        num_models: int,
        num_atoms: int,
    ) -> dict:
        """Create tensors to store the results of the committee

        Args:
            model_type (Literal[MACE, DipoleMACE, EnergyDipoleMACE]): type of model to load
            num_models (int): number of models in the committee
            num_atoms (int): number of atoms in the system

        Returns:
            dict: dictionary of tensors to store the results of the committee
        """
        dict_of_tensors = {}
        if model_type in ["MACE", "EnergyDipoleMACE"]:
            energies = torch.zeros(num_models, device=self.device)
            node_energy = torch.zeros(num_models, num_atoms, device=self.device)
            forces = torch.zeros(num_models, num_atoms, 3, device=self.device)
            stress = torch.zeros(num_models, 3, 3, device=self.device)
            dict_of_tensors.update(
                {
                    "energies": energies,
                    "node_energy": node_energy,
                    "forces": forces,
                    "stress": stress,
                }
            )
        if model_type in ["EnergyDipoleMACE", "DipoleMACE"]:
            dipole = torch.zeros(num_models, 3, device=self.device)
            dict_of_tensors.update({"dipole": dipole})
        return dict_of_tensors

    def _atoms_to_batch(self, atoms: Atoms) -> torch_geometric.data.Data:
        """Convert ASE Atoms object to PyTorch Geometric Data object"""
        config = data.config_from_atoms(atoms, charges_key=self.charges_key)
        data_loader = torch_geometric.dataloader.DataLoader(
            dataset=[
                data.AtomicData.from_config(
                    config, z_table=self.z_table, cutoff=self.r_max, heads=self.heads
                )
            ],
            batch_size=1,
            shuffle=False,
            drop_last=False,
        )
        batch = next(iter(data_loader)).to(self.device)
        return batch

    def _clone_batch(
        self, batch: torch_geometric.data.Data
    ) -> torch_geometric.data.Data:
        """Clone the batch to allow for gradients to be calculated"""
        batch_clone = batch.clone()
        if self.use_compile:
            batch_clone["node_attrs"].requires_grad_(True)
            batch_clone["positions"].requires_grad_(True)
        return batch_clone

    # pylint: disable=dangerous-default-value
    def calculate(
        self,
        atoms: Atoms | None = None,
        properties: list | None = None,
        system_changes: list | None = all_changes,
    ) -> None:
        """Calculate various properties of the atoms.

        Args:
            atoms (Atoms | None): The atoms object to calculate properties for.
            properties (list | None): The properties to calculate.
                Default is None.
            system_changes (list | None): The changes made to the system.
                Default is all changes.
        """
        # call to base-class to set atoms attribute
        super().calculate(self, atoms)

        batch_base = self._atoms_to_batch(atoms)

        if self.model_type in ["MACE", "EnergyDipoleMACE"]:
            batch = self._clone_batch(batch_base)
            node_heads = batch["head"][batch["batch"]]
            num_atoms_arange = torch.arange(batch["positions"].shape[0])
            node_e0 = self.models[0].atomic_energies_fn(batch["node_attrs"])[
                num_atoms_arange, node_heads
            ]
            compute_stress = not self.use_compile
        else:
            compute_stress = False

        ret_tensors = self._create_result_tensors(
            self.model_type, self.num_models, len(atoms)
        )
        for i, model in enumerate(self.models):
            batch = self._clone_batch(batch_base)
            out = model(
                batch.to_dict(),
                compute_stress=compute_stress,
                training=self.use_compile,
            )
            if self.model_type in ["MACE", "EnergyDipoleMACE"]:
                ret_tensors["energies"][i] = out["energy"].detach()
                ret_tensors["node_energy"][i] = (out["node_energy"] - node_e0).detach()
                ret_tensors["forces"][i] = out["forces"].detach()
                if out["stress"] is not None:
                    ret_tensors["stress"][i] = out["stress"].detach()
            if self.model_type in ["DipoleMACE", "EnergyDipoleMACE"]:
                ret_tensors["dipole"][i] = out["dipole"].detach()

        self.results = {}
        if self.model_type in ["MACE", "EnergyDipoleMACE"]:
            self.results["energy"] = (
                torch.mean(ret_tensors["energies"], dim=0).cpu().item()
                * self.energy_units_to_eV
            )
            self.results["free_energy"] = self.results["energy"]
            self.results["node_energy"] = (
                torch.mean(ret_tensors["node_energy"], dim=0).cpu().numpy()
            )
            self.results["forces"] = (
                torch.mean(ret_tensors["forces"], dim=0).cpu().numpy()
                * self.energy_units_to_eV
                / self.length_units_to_A
            )
            if self.num_models > 1:
                self.results["energies"] = (
                    ret_tensors["energies"].cpu().numpy() * self.energy_units_to_eV
                )
                self.results["energy_var"] = (
                    torch.var(ret_tensors["energies"], dim=0, unbiased=False)
                    .cpu()
                    .item()
                    * self.energy_units_to_eV
                )
                self.results["forces_comm"] = (
                    ret_tensors["forces"].cpu().numpy()
                    * self.energy_units_to_eV
                    / self.length_units_to_A
                )
            if out["stress"] is not None:
                self.results["stress"] = full_3x3_to_voigt_6_stress(
                    torch.mean(ret_tensors["stress"], dim=0).cpu().numpy()
                    * self.energy_units_to_eV
                    / self.length_units_to_A**3
                )
                if self.num_models > 1:
                    self.results["stress_var"] = full_3x3_to_voigt_6_stress(
                        torch.var(ret_tensors["stress"], dim=0, unbiased=False)
                        .cpu()
                        .numpy()
                        * self.energy_units_to_eV
                        / self.length_units_to_A**3
                    )
        if self.model_type in ["DipoleMACE", "EnergyDipoleMACE"]:
            self.results["dipole"] = (
                torch.mean(ret_tensors["dipole"], dim=0).cpu().numpy()
            )
            if self.num_models > 1:
                self.results["dipole_var"] = (
                    torch.var(ret_tensors["dipole"], dim=0, unbiased=False)
                    .cpu()
                    .numpy()
                )

    def get_hessian(self, atoms: Atoms | None = None) -> np.ndarray | list[np.ndarray]:
        if atoms is None and self.atoms is None:
            raise ValueError("atoms not set")
        if atoms is None:
            atoms = self.atoms
        if self.model_type != "MACE":
            raise NotImplementedError("Only implemented for MACE models")
        batch = self._atoms_to_batch(atoms)
        hessians = [
            model(
                self._clone_batch(batch).to_dict(),
                compute_hessian=True,
                compute_stress=False,
                training=self.use_compile,
            )["hessian"]
            for model in self.models
        ]
        hessians = [hessian.detach().cpu().numpy() for hessian in hessians]
        if self.num_models == 1:
            return hessians[0]
        return hessians

    def get_descriptors(
        self,
        atoms: Atoms | None = None,
        invariants_only: bool = True,
        num_layers: int = -1,
    ) -> np.ndarray | list[np.ndarray]:
        """Get descriptors from the model

        Args:
            atoms (Atoms | None, optional): Atoms object to get descriptors for
            invariants_only (bool, optional): Whether to return only the invariant features
            num_layers (int, optional): Number of layers to return descriptors for

        Raises:
            ValueError: atoms not set
            NotImplementedError: Only implemented for MACE models

        Returns:
            np.ndarray | list[np.ndarray]: descriptors from the model
        """
        if atoms is None and self.atoms is None:
            raise ValueError("atoms not set")
        if atoms is None:
            atoms = self.atoms
        if self.model_type != "MACE":
            raise NotImplementedError("Only implemented for MACE models")
        num_interactions = int(self.models[0].num_interactions)
        if num_layers == -1:
            num_layers = num_interactions
        batch = self._atoms_to_batch(atoms)
        descriptors = [model(batch.to_dict())["node_feats"] for model in self.models]

        irreps_out = o3.Irreps(str(self.models[0].products[0].linear.irreps_out))
        l_max = irreps_out.lmax
        num_invariant_features = irreps_out.dim // (l_max + 1) ** 2
        per_layer_features = [irreps_out.dim for _ in range(num_interactions)]
        per_layer_features[-1] = (
            num_invariant_features  # Equivariant features not created for the last layer
        )

        if invariants_only:
            descriptors = [
                extract_invariant(
                    descriptor,
                    num_layers=num_layers,
                    num_features=num_invariant_features,
                    l_max=l_max,
                )
                for descriptor in descriptors
            ]
        to_keep = np.sum(per_layer_features[:num_layers])
        descriptors = [
            descriptor[:, :to_keep].detach().cpu().numpy() for descriptor in descriptors
        ]

        if self.num_models == 1:
            return descriptors[0]
        return descriptors


class MACELoader(object):
    """Class to load MACE calculators from pretrained models or user specified models"""

    def __init__(self, remake_cache: bool = False) -> None:
        self.pretrained_models = [model.name for model in MACECHECKPOINTS]

        self.cache_dir = os.path.expanduser("~/.cache/mace")
        if remake_cache and os.path.exists(self.cache_dir):
            warnings.warn(
                "Remake cache was set to True. The contents of the cache directory have been deleted."
            )
            os.removedirs(self.cache_dir)
            return

        os.makedirs(self.cache_dir, exist_ok=True)

    def _check_model_cache(self, model: str) -> bool:
        """Check if the model is cached"""
        return os.path.exists(os.path.join(self.cache_dir, f"{model}.model"))

    def _cache_pretrained_model(self, model: str) -> os.PathLike:
        """Cache the specified pretrained model if not already cached"""

        cached_model_path = os.path.join(self.cache_dir, f"{model}.model")

        if not self._check_model_cache(model):
            model_url = MACECHECKPOINTS[model].value
            print(f"Downloading {model} model from {model_url}...")
            _, http_msg = urllib.request.urlretrieve(model_url, cached_model_path)
            if "Content-Type: text/html" in http_msg:
                raise RuntimeError(
                    f"Model download failed, please check the URL {model_url}"
                )
            print(f"Cached MACE model to {cached_model_path}")
        elif self._check_model_cache(model):
            print(f"Cached model found at {cached_model_path}")
        else:
            raise RuntimeError(
                f"Failed to cache model {model}. Double check the model name and URL."
            )

        if model == "mediummpa0":
            print(
                "Using medium MPA-0 model as default MACE-MP model, to use previous (before 3.10) default model please specify 'medium' as model argument"
            )
        elif model == "mediumomat0":
            print(
                "Using medium OMAT-0 model under Academic Software License (ASL) license, see https://github.com/gabor1/ASL \n To use this model you accept the terms of the license."
            )

        return cached_model_path

    def load_calculator(
        self,
        models: list[torch.nn.Module] | torch.nn.Module | list[str] | str,
        *,
        device: Literal["cpu", "cuda"] = "cpu",
        default_dtype: Literal["float32", "float64", "auto"] = "auto",
        model_type: Literal["MACE", "DipoleMACE", "EnergyDipoleMace"] = "MACE",
        energy_units_to_eV: float = 1.0,
        length_units_to_A: float = 1.0,
        charges_key: str = "Qs",
        compile_mode: (
            Literal[
                "default",
                "reduce-overhead",
                "max-autotune",
                "max-autotune-no-cudagraphs",
            ]
            | None
        ) = None,
        fullgraph: bool = True,
        enable_cueq: bool = False,
        include_dispersion: bool = False,
        damping_function: Literal["zero", "bj", "zerom", "bjm"] = "bj",
        dispersion_xc: str = "pbe",
        dispersion_cutoff: float = 40.0 * units.Bohr,
        **kwargs,
    ) -> MACECalculator:

        if isinstance(models, str) and models.lower() in self.pretrained_models:
            self.models = self._cache_pretrained_model(models.lower())
        elif isinstance(models, list) and all(
            isinstance(model, str) for model in models
        ):
            self.models = [
                (
                    self._cache_pretrained_model(model.lower())
                    if model.lower() in self.pretrained_models
                    else model
                )
                for model in models
            ]
        else:
            self.models = models

        self.mace_calculator = MACECalculator(
            models=self.models,
            device=device,
            default_dtype=default_dtype,
            model_type=model_type,
            energy_units_to_eV=energy_units_to_eV,
            length_units_to_A=length_units_to_A,
            charges_key=charges_key,
            compile_mode=compile_mode,
            fullgraph=fullgraph,
            enable_cueq=enable_cueq,
            **kwargs,
        )

        if include_dispersion:

            try:
                from torch_dftd.torch_dftd3_calculator import TorchDFTD3Calculator
            except ImportError as exc:
                raise RuntimeError(
                    "Please install torch-dftd to use dispersion corrections (see https://github.com/pfnet-research/torch-dftd)"
                ) from exc

            print("Using TorchDFTD3Calculator for D3 dispersion corrections")
            self.d3_calc = TorchDFTD3Calculator(
                device=device,
                damping=damping_function,
                dtype=self.mace_calculator.default_dtype,
                xc=dispersion_xc,
                cutoff=dispersion_cutoff,
                **kwargs,
            )

            self.calculator = SumCalculator([self.mace_calculator, self.d3_calc])
            return self.calculator

        self.calculator = self.mace_calculator
        return self.mace_calculator

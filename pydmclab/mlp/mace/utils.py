from __future__ import annotations

import os
import warnings
import urllib.request
from enum import Enum
from typing import TYPE_CHECKING, Literal

from torch import float32, float64

from ase import units
from ase.calculators.mixing import SumCalculator

from pydmclab.mlp.mace.dynamics import MACECalculator

if TYPE_CHECKING:
    from torch import dtype
    from torch.nn import Module


def get_model_dtype(model: Module) -> dtype:
    """Get the dtype of the model"""
    mode_dtype = next(model.parameters()).dtype
    if mode_dtype == float64:
        return "float64"
    if mode_dtype == float32:
        return "float32"
    raise ValueError(f"Unknown dtype {mode_dtype}")


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


class MACELoader(object):
    """Class to load MACE calculators from pretrained models or user specified models"""

    def __init__(self, remake_cache: bool = False) -> None:
        self.pretrained_models = [model.name for model in MACECHECKPOINTS]

        self.cache_dir = os.path.expanduser("~/.mace/cache")
        if remake_cache and os.path.exists(self.cache_dir):
            warnings.warn(
                "Remake cache was set to True. The contents of the cache directory have been deleted."
            )
            os.removedirs(self.cache_dir)
            return

        os.makedirs(self.cache_dir, exist_ok=True)

    def _check_model_cache(self, model: str) -> bool:
        """Check if the model is cached"""
        return os.path.exists(os.path.join(self.cache_dir, model, ".model"))

    def _cache_pretrained_model(self, model: str) -> os.PathLike:
        """Cache the specified pretrained model if not already cached"""

        cached_model_path = os.path.join(self.cache_dir, model, ".model")

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

    def load_calulator(
        self,
        models: list[Module] | Module | list[str] | str,
        *,
        device: Literal["cpu", "mps", "cuda"] = "mps",
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

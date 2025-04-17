from __future__ import annotations

from enum import Enum

from typing import TYPE_CHECKING, Sequence

if TYPE_CHECKING:
    from torch.nn import Parameter
    from torch.optim.optimizer import Optimizer as TorchOptimizer
    from torch.optim.lr_scheduler import LRScheduler as TorchLrScheduler


class TorchOptimizerEnum(Enum):
    def __new__(
        cls,
        optimizer_object: TorchOptimizer,
        default_kwargs: dict[str, (float | int | bool)] | None = None,
    ) -> TorchOptimizerEnum:
        obj = object.__new__(cls)
        obj._value_ = optimizer_object
        obj._default_kwargs_ = (
            default_kwargs if isinstance(default_kwargs, dict) else {}
        )
        return obj

    def __call__(
        self, params: Sequence[Parameter], learning_rate: float, **kwargs
    ) -> TorchOptimizer:

        # Merge default kwargs with provided kwargs
        all_kwargs = {**self._default_kwargs_, **kwargs}
        return self.value(params, learning_rate, **all_kwargs)


class TorchLrSchedulerEnum(Enum):
    def __new__(
        cls,
        scheduler_object: TorchLrScheduler,
        default_kwargs: dict[str, (float | int | bool)] | None = None,
    ) -> TorchOptimizerEnum:
        obj = object.__new__(cls)
        obj._value_ = scheduler_object
        obj._default_kwargs_ = (
            default_kwargs if isinstance(default_kwargs, dict) else {}
        )
        return obj

    def __call__(
        self,
        optimizer: TorchOptimizer,
        context: dict[str, (float | int | bool)],
        **kwargs,
    ) -> TorchOptimizer:

        # Merge default kwargs with provided kwargs
        all_kwargs = {**self._default_kwargs_, **kwargs}

        # Evaluate all kwargs if they are callable, this would be set in the script that uses this enum
        evaluated_kwargs = {
            k: (v(context, all_kwargs) if callable(v) else v)
            for k, v in all_kwargs.copy().items()
        }

        # Clean up kwargs that are not part of the optimizer's signature (e.g. if any of the kwargs were popped)
        evaluated_kwargs = {
            k: v for k, v in evaluated_kwargs.items() if k in all_kwargs
        }
        return self.value(optimizer, **evaluated_kwargs)

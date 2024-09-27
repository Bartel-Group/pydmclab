from __future__ import annotations

from typing import Literal, get_args

import numpy as np

Versions = Literal["0.2.0", "0.3.0"]
Devices = Literal["mps", "cuda", "cpu"]

TrainTask = Literal["ef", "efs", "efsm"]
PredTask = Literal["e", "ef", "em", "efs", "efsm"]

LogFreq = Literal["epoch", "batch"]
LogEachEpoch, LogEachBatch = get_args(LogFreq)


def convert_numpy_to_native(obj):
    if isinstance(obj, np.float32) or isinstance(obj, np.float64):
        return float(obj)
    elif isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {k: convert_numpy_to_native(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_to_native(i) for i in obj]
    else:
        return obj

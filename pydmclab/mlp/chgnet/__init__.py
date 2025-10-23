from __future__ import annotations

from typing import Literal, get_args

Versions = Literal["0.2.0", "0.3.0"]
Devices = Literal["mps", "cuda", "cpu"]

TrainTask = Literal["ef", "efs", "efsm"]
PredTask = Literal["e", "ef", "em", "efs", "efsm"]

LogFreq = Literal["epoch", "batch"]
LogEachEpoch, LogEachBatch = get_args(LogFreq)

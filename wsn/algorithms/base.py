from __future__ import annotations
from dataclasses import dataclass
from typing import Protocol, Dict, Any, Tuple
import numpy as np

from ..models import Network
from ..fitness import fitness, FitnessParams


class Optimizer(Protocol):
    def run(self, net: Network, fit_params: FitnessParams, n_iter: int) -> Tuple[np.ndarray, float]:
        ...


@dataclass
class OptimizationResult:
    best_ch_indices: np.ndarray
    best_fitness: float
    history: Dict[str, Any]
    nfe: float = 0.0
    

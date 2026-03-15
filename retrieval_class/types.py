"""Shared datatypes used across the retrieval package.

This module defines light-weight dataclasses for axes, traces, configuration,
and retrieval outputs.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Optional
import numpy as np

Array = np.ndarray


@dataclass(slots=True)
class FrogGrid:
    """Sampling grid for delay and frequency axes."""

    delay: Array
    frequency: Array

    @property
    def n(self) -> int:
        """Number of samples on the grid."""
        return int(self.delay.size)


@dataclass(slots=True)
class FrogTrace:
    """Measured trace bundle with geometry metadata."""

    intensity: Array
    grid: FrogGrid
    geometry: str = "shg-frog"

    def __post_init__(self) -> None:
        """Validate trace dimensionality against the provided grid."""
        if self.intensity.ndim != 2:
            raise ValueError("trace.intensity must be 2D")
        n = self.grid.n
        if self.intensity.shape != (n, n):
            raise ValueError(f"trace shape must be {(n, n)}, got {self.intensity.shape}")


@dataclass(slots=True)
class RetrievalConfig:
    """Tunable parameters grouped by algorithm usage.

    Shared:
    - ``verbose``, ``rng_seed``
    - ``progress_callback``, ``progress_interval``, ``stop_requested``

    RANA-only:
    - ``rana_g_cutoff``, ``rana_gp_cutoff``
    - ``rana_full_iter_cap``
    - ``rana_weight_factor``
    """

    # Shared options
    verbose: bool = True
    rng_seed: Optional[int] = None
    progress_callback: Optional[Callable[[Dict[str, Any]], None]] = None
    progress_interval: int = 1
    stop_requested: Optional[Callable[[], bool]] = None

    # RANA-only options
    rana_g_cutoff: float = 1e-3
    rana_gp_cutoff: float = 0.01
    rana_full_iter_cap: int = 160
    rana_weight_factor: float = 1.0


@dataclass(slots=True)
class RetrievalResult:
    """Outputs and diagnostics produced by a retrieval run."""

    field: Array
    retrieved_trace: Array
    errors: Array
    best_iteration: int
    best_sigma: float
    perturbations: int
    diagnostics: Dict[str, Any] = field(default_factory=dict)

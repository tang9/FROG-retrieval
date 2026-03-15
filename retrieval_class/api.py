"""Top-level user API for running pulse retrieval.

This module validates basic input structure, constructs the retrieval model,
and dispatches to a configured retriever implementation.
"""

from __future__ import annotations

import numpy as np
from .factory import RetrieverFactory
from .field2trace import build_forward_model, normalize_geometry_name
from .types import Array, FrogGrid, FrogTrace, RetrievalConfig, RetrievalResult


def retrieve_pulse(
    trace: Array,
    delay: Array,
    frequency: Array,
    algorithm: str = "rana",
    guess: Array | None = None,
    geometry: str = "shg-frog",
    config: RetrievalConfig | None = None,
) -> RetrievalResult:
    """Run pulse retrieval from a measured FROG trace.

    Args:
        trace: Measured FROG intensity matrix with shape ``(N, N)``.
        delay: Delay axis used for the trace.
        frequency: Frequency axis used for the trace.
        algorithm: Retriever name registered in ``RetrieverFactory``.
        guess: Optional initial complex field guess.
        geometry: Measurement geometry name such as ``"shg-frog"``,
            ``"pg-frog"``, ``"thg-frog"``, or ``"sd-frog"``.
        config: Optional retrieval configuration.

    Returns:
        RetrievalResult containing the retrieved field, trace, and diagnostics.
    """
    config = config or RetrievalConfig()
    geometry_name = normalize_geometry_name(geometry)
    grid = FrogGrid(delay=np.asarray(delay), frequency=np.asarray(frequency))
    frog_trace = FrogTrace(
        intensity=np.ascontiguousarray(trace, dtype=np.float64),
        grid=grid,
        geometry=geometry_name,
    )

    model = build_forward_model(geometry_name)
    retriever = RetrieverFactory.create(algorithm, model=model, config=config)
    return retriever.retrieve(frog_trace, guess=guess)

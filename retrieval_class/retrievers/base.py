"""Abstract retriever interface shared by all algorithm implementations."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Optional
import numpy as np
from ..field2trace import ForwardModel
from ..types import Array, FrogTrace, RetrievalConfig, RetrievalResult


class Retriever(ABC):
    """Base class for retrieval algorithms."""

    name: str = "base"

    def __init__(self, model: ForwardModel, config: Optional[RetrievalConfig] = None):
        """Store forward model/config and initialize RNG."""
        self.model = model
        self.config = config or RetrievalConfig()
        self.rng = np.random.default_rng(self.config.rng_seed)

    @abstractmethod
    def retrieve(self, frog_trace: FrogTrace, guess: Optional[Array] = None) -> RetrievalResult:
        """Run retrieval and return the best solution found."""
        raise NotImplementedError

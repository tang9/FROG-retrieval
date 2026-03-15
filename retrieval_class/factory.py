"""Retriever registry and construction helpers.

This module maps retriever names to implementation classes and constructs
instances with a chosen forward model/config.
"""

from __future__ import annotations

from typing import Optional, Type
from .field2trace import ForwardModel
from .retrievers.base import Retriever
from .retrievers.rana import RANARetriever
from .types import RetrievalConfig


class RetrieverFactory:
    """Factory for registering and instantiating retriever algorithms."""

    _registry: dict[str, Type[Retriever]] = {
        "rana": RANARetriever,
        # future:
        # "pcgpa": PCGPARetriever,
        # "epie": EPIERetriever,
        # "copra": COPRARetriever,
    }

    @classmethod
    def register(cls, name: str, retriever_cls: Type[Retriever]) -> None:
        """Register a retriever class under a string name."""
        cls._registry[name.lower()] = retriever_cls

    @classmethod
    def create(cls, name: str, model: ForwardModel, config: Optional[RetrievalConfig] = None) -> Retriever:
        """Create a retriever instance by name."""
        key = name.lower()
        if key not in cls._registry:
            raise ValueError(f"Unknown retriever '{name}'. Available: {sorted(cls._registry)}")
        return cls._registry[key](model=model, config=config)

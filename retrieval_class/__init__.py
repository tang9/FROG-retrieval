"""Public package exports for the FROG retrieval library.

This module re-exports the primary API entry points and core dataclasses so
users can import from ``retrieval_class`` directly.
"""

from .api import retrieve_pulse
from .factory import RetrieverFactory
from .field2trace import (
    PGFrogForwardModel,
    SDFrogForwardModel,
    SHGFrogForwardModel,
    THGFrogForwardModel,
    build_forward_model,
)
from .types import FrogGrid, FrogTrace, RetrievalConfig, RetrievalResult

__all__ = [
    "retrieve_pulse",
    "RetrieverFactory",
    "build_forward_model",
    "SHGFrogForwardModel",
    "PGFrogForwardModel",
    "THGFrogForwardModel",
    "SDFrogForwardModel",
    "FrogGrid",
    "FrogTrace",
    "RetrievalConfig",
    "RetrievalResult",
]

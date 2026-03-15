"""Forward FROG simulation kernels.

This module defines the forward-model interface and NumPy implementations for
the supported delay-gated FROG geometries.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
import numpy as np

from .types import Array, FrogGrid


SUPPORTED_GEOMETRIES = (
    "shg-frog",
    "pg-frog",
    "thg-frog",
    "sd-frog",
)


def normalize_geometry_name(geometry: str | None) -> str:
    """Normalize user-facing geometry names to canonical internal values."""
    value = str(geometry or "shg-frog").strip().lower().replace("_", "-")
    aliases = {
        "shg": "shg-frog",
        "shgfrog": "shg-frog",
        "pg": "pg-frog",
        "pgfrog": "pg-frog",
        "thg": "thg-frog",
        "thgfrog": "thg-frog",
        "sd": "sd-frog",
        "sdfrog": "sd-frog",
    }
    value = aliases.get(value, value)
    if value not in SUPPORTED_GEOMETRIES:
        raise ValueError(f"Unsupported FROG geometry '{geometry}'. Available: {list(SUPPORTED_GEOMETRIES)}")
    return value


def probe_gate_from_field(field: Array, geometry: str) -> tuple[Array, Array]:
    """Return the probe and gate pulses used by CalcEsig(probe, gate)."""
    f = np.asarray(field, dtype=np.complex128).ravel()
    geom = normalize_geometry_name(geometry)
    if geom == "shg-frog":
        return f, f
    if geom == "pg-frog":
        return f, np.abs(f) ** 2
    if geom == "thg-frog":
        return f, f * f
    if geom == "sd-frog":
        return np.conj(f), f * f
    raise ValueError(f"Unsupported FROG geometry '{geometry}'.")


class ForwardModel(ABC):
    """Abstract interface for forward models used by retrievers."""

    geometry: str = "shg-frog"

    @abstractmethod
    def simulate_trace(self, field: Array, grid: FrogGrid) -> Array:
        """Simulate one FROG trace as a delay-by-frequency matrix."""
        raise NotImplementedError


class _DelayGateForwardModel(ForwardModel):
    """Common implementation for probe(t) * gate(t - tau) style FROG traces."""

    geometry: str = "shg-frog"

    def __init__(self, geometry: str | None = None) -> None:
        self.geometry = normalize_geometry_name(geometry or self.geometry)
        self._cached_grid_id: int | None = None
        self._cached_idx: np.ndarray | None = None
        self._cached_valid: np.ndarray | None = None
        self._cached_idx_clip: np.ndarray | None = None

    def _prepare_grid(self, grid: FrogGrid) -> None:
        gid = id(grid)
        if gid != self._cached_grid_id:
            self._cached_grid_id = gid
            n = grid.n
            shifts = np.arange(-n // 2, n // 2, dtype=np.int64)
            idx = np.arange(n, dtype=np.int64)[None, :] - shifts[:, None]
            self._cached_valid = (idx >= 0) & (idx < n)
            self._cached_idx_clip = np.clip(idx, 0, n - 1)

    def _probe_gate(self, field: Array) -> tuple[Array, Array]:
        return probe_gate_from_field(field, self.geometry)

    def simulate_trace(self, field: Array, grid: FrogGrid) -> Array:
        field_arr = np.asarray(field, dtype=np.complex128).ravel()
        n = grid.n
        if field_arr.size != n:
            raise ValueError(f"field length {field_arr.size} does not match grid size {n}")

        self._prepare_grid(grid)
        probe, gate = self._probe_gate(field_arr)
        delayed_gate = np.where(self._cached_valid, gate[self._cached_idx_clip], 0.0)
        esig_t = probe[None, :] * delayed_gate
        spec = np.fft.fft(esig_t, axis=1) / n
        intensity = np.fft.fftshift(spec.real * spec.real + spec.imag * spec.imag, axes=1)

        peak = float(np.max(intensity))
        if peak > 1e-15:
            intensity = intensity / peak
        return intensity


class SHGFrogForwardModel(_DelayGateForwardModel):
    """SHG-FROG forward model."""

    geometry = "shg-frog"


class PGFrogForwardModel(_DelayGateForwardModel):
    """PG-FROG forward model."""

    geometry = "pg-frog"


class THGFrogForwardModel(_DelayGateForwardModel):
    """THG-FROG forward model."""

    geometry = "thg-frog"


class SDFrogForwardModel(_DelayGateForwardModel):
    """SD-FROG forward model."""

    geometry = "sd-frog"


def build_forward_model(geometry: str | None = None) -> ForwardModel:
    """Construct the forward model matching the requested FROG geometry."""
    geom = normalize_geometry_name(geometry)
    models = {
        "shg-frog": SHGFrogForwardModel,
        "pg-frog": PGFrogForwardModel,
        "thg-frog": THGFrogForwardModel,
        "sd-frog": SDFrogForwardModel,
    }
    return models[geom]()

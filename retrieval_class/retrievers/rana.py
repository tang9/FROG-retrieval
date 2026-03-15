"""RANA-inspired delay-gated FROG retriever.

This implementation follows the MATLAB RANA workflow at a practical level:
- estimate one-to-few spectra from the frequency marginal,
- build a coarse-to-fine multi-grid pyramid of the trace and spectral guesses,
- run multi-initial retrieval on each grid and promote the best candidates,
- refine on the full grid with dual stopping criteria on G and G'.

Algorithm sketch:
- `retrieve()` first normalizes the measured FROG trace and converts it to
  the internal `[frequency, delay]` layout used by the MATLAB code.
- The frequency marginal is passed to `_spec_ret_marginal()` to generate one or
  more spectral-amplitude seeds; random phases are then attached to create the
  initial fields on the coarsest grid.
- `_multi_grid()` downsamples the trace, time/frequency axes, and spectral
  guesses so retrieval starts on a small problem and is then promoted toward
  the full-resolution problem.
- `_promote_initial_guesses()` upsamples the best coarse-grid solutions and
  ranks them with the same FROG error metric used during retrieval.
- The inner solver is `_quickfrog()`: construct `Esig(t, tau)`, FFT to the
  spectral domain, replace the magnitude with the measured one, compute the
  geometry-specific descent direction, perform the exact polynomial line
  search used by the RANA workflow, re-center the pulse, and evaluate `G`
  and `G'`.
- Retrieval stops when the error targets are met, the error stagnates, the
  iteration budget is exhausted, or the caller requests cancellation.

Weighted zero-pixel handling:
- `weight_factor` is converted to a per-pixel `weights` array once the target
  amplitude is known, and the same weighted formulas are then used everywhere.
- When `weight_factor == 1`, `weights` becomes an all-ones mask and the update
  rules reduce to the standard unweighted RANA expressions without a special
  branch.
- `weight_factor < 1` softens the magnitude replacement and error evaluation on
  pixels where the measured amplitude is zero, which can make retrieval less
  brittle on sparse or aggressively thresholded traces.

Notes:
- The original MATLAB package relies on MEX kernels. This module provides a
  NumPy/SciPy implementation of the same core updates.
- Iteration counts are capped by default to keep pure-Python runtimes usable.

Variable naming conventions (following the MATLAB RANA code):
- et, e        : E(t), the electric field in the time domain.  complex 1-D array [N].
- esig, esig_t : E_sig(t, tau), the signal field in time domain.  complex 2-D [freq x delay].
- esig_w       : E_sig(w, tau), the signal field in frequency domain (FFT of esig along freq axis).
- asig         : A_sig = sqrt(I_measured), amplitude of measured FROG trace.  real 2-D [freq x delay].
- i_frog       : I_FROG, the measured FROG trace intensity (normalized).  real 2-D [freq x delay].
- isig         : same as i_frog at each multi-grid level.
- g, g_best    : G error = RMS difference between |E_sig|^2 and measured, normalized by max.
- gp, gp_best  : G' error = same residual as G but normalized by integral(A_sig^4).
- z, z_best    : Z error = geometry-specific gradient-descent objective.
- dz           : dZ/dE, the descent direction for the Z minimization.
- swr          : spectral weight / seed spectrum, estimated from the frequency marginal.
- mw           : marginal in the frequency direction = integral of I_FROG over delay.
- weights, w   : per-pixel weight array (1.0 for nonzero measured pixels, weight_factor for zero).
- kernel       : _GeometryKernel, bundles geometry-specific functions (calc_esig, dZ/dE, etc.).

https://frog.gatech.edu/code.html
R. Jafari and R. Trebino, "Extremely Robust Pulse Retrieval From Even Noisy Second-Harmonic-Generation
Frequency-Resolved Optical Gating Traces," IEEE J. of Quant. Electr. 56, 1-8 (2020).
"""

from __future__ import annotations

import importlib
from pathlib import Path
import subprocess
import sys
import time
from dataclasses import dataclass
from typing import Callable, Optional
import warnings

import numpy as np
from scipy.interpolate import PchipInterpolator, RegularGridInterpolator
from scipy.signal import find_peaks, savgol_filter

from ..field2trace import normalize_geometry_name, probe_gate_from_field
from ..types import Array, FrogTrace, RetrievalResult
from .base import Retriever

def _import_rana_cython_module():
    module_name = f"{__package__}.rana_cython"
    try:
        return importlib.import_module(".rana_cython", __package__), None
    except Exception as first_exc:
        setup_path = Path(__file__).resolve().parents[2] / "setup_cython.py"
        if not setup_path.is_file():
            return None, first_exc
        try:
            subprocess.run(
                [sys.executable, str(setup_path), "build_ext", "--inplace"],
                cwd=str(setup_path.parent),
                check=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )
            importlib.invalidate_caches()
            sys.modules.pop(module_name, None)
            return importlib.import_module(".rana_cython", __package__), None
        except Exception as second_exc:
            return None, RuntimeError(
                f"initial import failed: {first_exc}; auto-build retry failed: {second_exc}"
            )


_rana_cython_module, _rana_cython_error = _import_rana_cython_module()
if _rana_cython_module is not None:
    _quickfrog_cy = _rana_cython_module.quickfrog_cy
    _calc_esig_geom_cy = _rana_cython_module._calc_esig_geom
    _g_error_cy = _rana_cython_module._g_error
    _g_gprime_error_cy = _rana_cython_module._g_gprime_error
    _compare_gerror_cy = _rana_cython_module.compare_gerror_cy
    HAS_RANA_CYTHON = True
else:
    HAS_RANA_CYTHON = False
    _quickfrog_cy = None
    _calc_esig_geom_cy = None
    _g_error_cy = None
    _g_gprime_error_cy = None
    _compare_gerror_cy = None
    warnings.warn(
        f"Cython kernel retrieval_class.retrievers.rana_cython unavailable; using NumPy fallback. ({_rana_cython_error})",
        RuntimeWarning,
    )


DEFAULT_WEIGHT_FACTOR = 1.0
DEFAULT_SNR_LEVEL = 4
DEFAULT_MAX_SPECTRA = 8
DEFAULT_COARSE_ITER_CAP = 40
DEFAULT_STALL_ABS_COARSE = 1e-6
DEFAULT_STALL_RATIO_COARSE = 1e-2
DEFAULT_STALL_ABS_FULL = 5e-8
DEFAULT_STALL_RATIO_FULL = 1e-3


def normalize_field(field: Array) -> Array:
    """Scale a complex field so its peak magnitude is 1."""
    field_arr = np.asarray(field, dtype=np.complex128)
    peak = float(np.max(np.abs(field_arr))) if field_arr.size else 0.0
    if peak < 1e-15:
        return field_arr.copy()
    return field_arr / peak


def _safe_sqrt(x: Array) -> Array:
    """Element-wise sqrt after clipping negatives to zero."""
    return np.sqrt(np.clip(np.asarray(x, dtype=np.float64), 0.0, None))


def _quickscale(x: Array) -> Array:
    x_arr = np.asarray(x)
    peak = float(np.max(np.abs(x_arr))) if x_arr.size else 0.0
    if peak <= 1e-30:
        return x_arr.copy()
    return x_arr / peak


def _fftc(x: Array, axis: int = 0) -> Array:
    return np.fft.fftshift(np.fft.fft(np.fft.ifftshift(x, axes=axis), axis=axis), axes=axis)


def _ifftc(x: Array, axis: int = 0) -> Array:
    n = int(np.asarray(x).shape[axis])
    if n % 2:
        return np.fft.fftshift(np.fft.ifft(np.fft.ifftshift(x, axes=axis), axis=axis), axes=axis)
    return np.fft.ifftshift(np.fft.ifft(np.fft.fftshift(x, axes=axis), axis=axis), axes=axis)


def _center_moment(x: Array) -> Array:
    x_arr = np.asarray(x)
    w = np.abs(x_arr).astype(np.float64, copy=False)
    s = float(np.sum(w))
    if s <= 1e-30:
        return x_arr.copy()
    n = int(w.size)
    ic = float(np.sum((np.arange(n, dtype=np.float64) + 1.0) * w) / s)
    shift = int(np.floor(n / 2.0) - np.round(ic) + 1)
    return np.roll(x_arr, shift)


def _calc_esig_probe_gate(probe: Array, gate: Array) -> Array:
    """Return Esig(t, tau) = probe(t) * gate(t - tau) on integer delays."""
    p = np.asarray(probe, dtype=np.complex128).ravel()
    g = np.asarray(gate, dtype=np.complex128).ravel()
    if p.size != g.size:
        raise ValueError("probe and gate must have the same length.")
    n = p.size
    shifts = np.arange(-n // 2, n // 2, dtype=np.int64)
    t = np.arange(n, dtype=np.int64)[:, None]
    tp = t - shifts[None, :]
    valid = (tp >= 0) & (tp < n)
    tp_clip = np.clip(tp, 0, n - 1)
    return np.where(valid, p[t] * g[tp_clip], 0.0j)


def _calc_esig_geom(et: Array, geometry: str) -> Array:
    probe, gate = probe_gate_from_field(et, geometry)
    return _calc_esig_probe_gate(probe, gate)



def _signal_coeffs_shg(e_t: complex, e_tp: complex, d_t: complex, d_tp: complex) -> Array:
    return np.asarray(
        [
            e_t * e_tp,
            d_t * e_tp + e_t * d_tp,
            d_t * d_tp,
        ],
        dtype=np.complex128,
    )


def _signal_coeffs_pg(e_t: complex, e_tp: complex, d_t: complex, d_tp: complex) -> Array:
    c0 = float(np.abs(e_tp) ** 2)
    c1 = 2.0 * float(np.real(d_tp * np.conj(e_tp)))
    c2 = float(np.abs(d_tp) ** 2)
    return np.asarray(
        [
            e_t * c0,
            d_t * c0 + e_t * c1,
            d_t * c1 + e_t * c2,
            d_t * c2,
        ],
        dtype=np.complex128,
    )


def _signal_coeffs_thg(e_t: complex, e_tp: complex, d_t: complex, d_tp: complex) -> Array:
    return np.asarray(
        [
            e_t * e_tp * e_tp,
            2.0 * e_t * e_tp * d_tp + d_t * e_tp * e_tp,
            e_t * d_tp * d_tp + 2.0 * d_t * e_tp * d_tp,
            d_t * d_tp * d_tp,
        ],
        dtype=np.complex128,
    )


def _signal_coeffs_sd(e_t: complex, e_tp: complex, d_t: complex, d_tp: complex) -> Array:
    ce_t = np.conj(e_t)
    cd_t = np.conj(d_t)
    return np.asarray(
        [
            ce_t * e_tp * e_tp,
            2.0 * ce_t * e_tp * d_tp + cd_t * e_tp * e_tp,
            ce_t * d_tp * d_tp + 2.0 * cd_t * e_tp * d_tp,
            cd_t * d_tp * d_tp,
        ],
        dtype=np.complex128,
    )


def _dzdE_shg(esigp: Array, et: Array) -> Array:
    """Vectorized NumPy port of dZdE_shg MEX kernel."""
    es = np.asarray(esigp, dtype=np.complex128)
    e = np.asarray(et, dtype=np.complex128).ravel()
    n = e.size
    shifts = np.arange(-n // 2, n // 2, dtype=np.int64)
    d = np.zeros(n, dtype=np.complex128)

    for j, s in enumerate(shifts):
        if s >= 0:
            t1 = np.arange(s, n, dtype=np.int64)
            tp1 = t1 - s
            t2 = np.arange(0, n - s, dtype=np.int64)
            tp2 = t2 + s
        else:
            t1 = np.arange(0, n + s, dtype=np.int64)
            tp1 = t1 - s
            t2 = np.arange(-s, n, dtype=np.int64)
            tp2 = t2 + s

        if t1.size:
            d[t1] += (e[t1] * e[tp1] - es[t1, j]) * np.conj(e[tp1])
        if t2.size:
            d[t2] += (e[t2] * e[tp2] - es[tp2, j]) * np.conj(e[tp2])

    return d / float(es.size)


def _dzdE_pg(esigp: Array, et: Array) -> Array:
    es = np.asarray(esigp, dtype=np.complex128)
    e = np.asarray(et, dtype=np.complex128).ravel()
    n = e.size
    shifts = np.arange(-n // 2, n // 2, dtype=np.int64)
    d = np.zeros(n, dtype=np.complex128)

    for t0 in range(n):
        total = 0.0j
        for j, s in enumerate(shifts):
            tp = t0 - s
            if 0 <= tp < n:
                total += (e[t0] * (np.abs(e[tp]) ** 2) - es[t0, j]) * (np.abs(e[tp]) ** 2)
            tp = t0 + s
            if 0 <= tp < n:
                total += 2.0 * ((np.abs(e[tp]) ** 2) * (np.abs(e[t0]) ** 2) - np.real(es[tp, j] * np.conj(e[tp]))) * e[t0]
        d[t0] = total / float(es.size)
    return d


def _dzdE_thg(esigp: Array, et: Array) -> Array:
    es = np.asarray(esigp, dtype=np.complex128)
    e = np.asarray(et, dtype=np.complex128).ravel()
    n = e.size
    shifts = np.arange(-n // 2, n // 2, dtype=np.int64)
    d = np.zeros(n, dtype=np.complex128)

    for t0 in range(n):
        total = 0.0j
        for j, s in enumerate(shifts):
            tp = t0 - s
            if 0 <= tp < n:
                total += (e[t0] * e[tp] * e[tp] - es[t0, j]) * np.conj(e[tp] * e[tp])
            tp = t0 + s
            if 0 <= tp < n:
                total += 2.0 * np.conj(e[tp] * e[t0]) * (e[tp] * e[t0] * e[t0] - es[tp, j])
        d[t0] = total / float(es.size)
    return d


def _dzdE_sd(esigp: Array, et: Array) -> Array:
    es = np.asarray(esigp, dtype=np.complex128)
    e = np.asarray(et, dtype=np.complex128).ravel()
    n = e.size
    shifts = np.arange(-n // 2, n // 2, dtype=np.int64)
    d = np.zeros(n, dtype=np.complex128)

    for t0 in range(n):
        total = 0.0j
        for j, s in enumerate(shifts):
            tp = t0 - s
            if 0 <= tp < n:
                total += (e[t0] * np.conj(e[tp]) * np.conj(e[tp]) - np.conj(es[t0, j])) * (e[tp] * e[tp])
            tp = t0 + s
            if 0 <= tp < n:
                total += (np.conj(e[tp]) * e[t0] * e[t0] - es[tp, j]) * 2.0 * e[tp] * np.conj(e[t0])
        d[t0] = total / float(es.size)
    return d


@dataclass(frozen=True)
class _GeometryKernel:
    geometry: str
    calc_esig: Callable[[Array], Array]
    dzde: Callable[[Array, Array], Array]
    signal_coeffs: Callable[[complex, complex, complex, complex], Array]
    g_scale_power: float
    rana_marginal_init: bool

    def apply_g_factor(self, et: Array, scale: float) -> Array:
        if scale <= 0:
            return np.asarray(et, dtype=np.complex128)
        return np.asarray(et, dtype=np.complex128) * (float(scale) ** self.g_scale_power)


GEOMETRY_KERNELS: dict[str, _GeometryKernel] = {
    "shg-frog": _GeometryKernel("shg-frog", lambda et: _calc_esig_geom(et, "shg-frog"), _dzdE_shg, _signal_coeffs_shg, 0.25, True),
    "pg-frog": _GeometryKernel("pg-frog", lambda et: _calc_esig_geom(et, "pg-frog"), _dzdE_pg, _signal_coeffs_pg, 1.0 / 6.0, False),
    "thg-frog": _GeometryKernel("thg-frog", lambda et: _calc_esig_geom(et, "thg-frog"), _dzdE_thg, _signal_coeffs_thg, 1.0 / 6.0, False),
    "sd-frog": _GeometryKernel("sd-frog", lambda et: _calc_esig_geom(et, "sd-frog"), _dzdE_sd, _signal_coeffs_sd, 1.0 / 6.0, False),
}


def _mag_repl(esig: Array, asig: Array, weights: Array) -> Array:
    """Replace |Esig| with Asig while preserving phase."""
    es = np.asarray(esig, dtype=np.complex128)
    a = np.asarray(asig, dtype=np.float64)
    w = np.asarray(weights, dtype=np.float64)
    temp = np.abs(es)
    target = es.copy()
    nz = temp > 0
    target[nz] = a[nz] * es[nz] / temp[nz]
    target[~nz] = a[~nz] + 0.0j
    return w * target + (1.0 - w) * es


def _min_gerr(esig: Array, asig: Array, weights: Array) -> tuple[float, float]:
    """Return (G, scale) where scale minimizes G."""
    e2 = np.abs(np.asarray(esig, dtype=np.complex128)) ** 2
    a2 = np.abs(np.asarray(asig, dtype=np.float64)) ** 2
    w = np.asarray(weights, dtype=np.float64)
    denom = float(np.sum(w * e2 * e2))
    if denom <= 1e-30:
        return float("inf"), 0.0
    scale = float(np.sum(w * e2 * a2) / denom)
    mx = float(np.max(a2))
    if mx <= 1e-30:
        return 0.0, scale
    n = float(np.sqrt(a2.size))
    if n <= 0:
        return float("inf"), 0.0
    g = float(np.sqrt(np.sum(w * (a2 - scale * e2) ** 2)) / mx / n)
    return g, scale


def _gprime_from_amp(asig: Array, esig: Array, scale: float, weights: Array) -> float:
    a = np.asarray(asig, dtype=np.float64)
    w = np.asarray(weights, dtype=np.float64)
    diff = np.abs(a ** 2 - scale * (np.abs(esig) ** 2)) ** 2
    num = float(np.trapezoid(np.trapezoid(w * diff, axis=0), axis=0))
    den = float(np.trapezoid(np.trapezoid(w * (a ** 4), axis=0), axis=0))
    if den <= 1e-30:
        return float("inf")
    val = num / den
    return float(np.sqrt(val)) if val > 0 else 0.0


def _g_gprime_error(asig: Array, et: Array, weights: Array, kernel: _GeometryKernel) -> tuple[float, float]:
    """Return (G, G') for a given field."""
    esig = _fftc(kernel.calc_esig(et), axis=0)
    g, a = _min_gerr(esig, asig, weights=weights)
    if not np.isfinite(g):
        return float("inf"), float("inf")
    gp = _gprime_from_amp(asig, esig, a, weights=weights)
    return float(g), float(gp)


def _g_error(asig: Array, et: Array, weights: Array, kernel: _GeometryKernel) -> float:
    g, _ = _g_gprime_error(asig, et, weights=weights, kernel=kernel)
    return float(g)


def _gprime_error(asig: Array, et: Array, weights: Array, kernel: _GeometryKernel) -> float:
    _, gp = _g_gprime_error(asig, et, weights=weights, kernel=kernel)
    return float(gp)


def _g_gprime_error_fast(asig: Array, et: Array, weights: Array, kernel: _GeometryKernel) -> tuple[float, float]:
    if HAS_RANA_CYTHON:
        return _g_gprime_error_cy(asig, et, weights, kernel.geometry)
    return _g_gprime_error(asig, et, weights=weights, kernel=kernel)


def _g_error_fast(asig: Array, et: Array, weights: Array, kernel: _GeometryKernel) -> float:
    if HAS_RANA_CYTHON:
        return float(_g_error_cy(asig, et, weights, kernel.geometry))
    return _g_error(asig, et, weights=weights, kernel=kernel)


def _pixel_weights(asig: Array, weight_factor: float = DEFAULT_WEIGHT_FACTOR) -> Array:
    """Return per-pixel weights: 1.0 for nonzero measured, weight_factor for zero."""
    a = np.asarray(asig, dtype=np.float64)
    nonzero = np.isfinite(a) & (a > 0.0)
    weights = np.full_like(a, weight_factor)
    weights[nonzero] = 1.0
    return weights


def _min_zerr(esig: Array, et: Array, dz: Array, kernel: _GeometryKernel) -> tuple[Array, float]:
    """Exact polynomial line search for the geometry-specific signal model."""
    es = np.asarray(esig, dtype=np.complex128)
    e = np.asarray(et, dtype=np.complex128).ravel()
    d = np.asarray(dz, dtype=np.complex128).ravel()
    n = e.size

    mx = float(np.max(np.abs(es) ** 2)) if es.size else 0.0
    if mx <= 1e-30:
        return e.copy(), 0.0

    coeff = np.zeros(2 * (len(kernel.signal_coeffs(1.0 + 0.0j, 1.0 + 0.0j, 1.0 + 0.0j, 1.0 + 0.0j)) - 1) + 1, dtype=np.float64)
    shifts = np.arange(-n // 2, n // 2, dtype=np.int64)

    for j, s in enumerate(shifts):
        if s >= 0:
            t = np.arange(s, n, dtype=np.int64)
            tp = t - s
        else:
            t = np.arange(0, n + s, dtype=np.int64)
            tp = t - s

        if not t.size:
            continue

        for tt, ttp in zip(t, tp):
            poly = np.asarray(kernel.signal_coeffs(e[tt], e[ttp], d[tt], d[ttp]), dtype=np.complex128)
            poly[0] -= es[tt, j]
            coeff += np.convolve(poly, np.conj(poly)).real

    scale = float(es.size) * mx
    coeff /= max(scale, 1e-30)

    if np.all(np.abs(coeff) <= 1e-30):
        return e.copy(), 0.0

    desc = coeff[::-1]
    dcoeff = np.polyder(desc)
    roots = np.roots(dcoeff)
    real_roots = roots[np.isclose(np.imag(roots), 0.0, atol=1e-10)].real
    if real_roots.size == 0:
        ridx = int(np.argmin(np.abs(np.imag(roots))))
        real_roots = np.asarray([float(np.real(roots[ridx]))], dtype=np.float64)

    zvals = np.polyval(desc, real_roots)
    m = int(np.argmin(zvals))
    x = float(real_roots[m])
    z = float(zvals[m])

    zmin = float(np.finfo(np.float64).eps * coeff[0])
    if z < zmin:
        z = zmin
    z = float(np.sqrt(max(z, 0.0)))

    return e + x * d, z


def _smooth_sgolay(y: Array) -> Array:
    y_arr = np.asarray(y, dtype=np.float64).ravel()
    n = y_arr.size
    win = min(11, n if n % 2 else n - 1)
    if win < 5:
        return y_arr.copy()
    return savgol_filter(y_arr, window_length=win, polyorder=2, mode="interp")


def _super_gaussian_1d(in_arr: Array, coeff: float, power: float) -> Array:
    x = np.asarray(in_arr)
    n = x.size
    h = np.arange(-n // 2, n // 2, dtype=np.float64)
    wn = max((n / 2.0) * float(coeff), 1e-12)
    sgauss = np.exp(-((h * h) / (wn * wn)) ** float(power))
    sgauss = _quickscale(sgauss)
    return x * sgauss


def _side_noise_1d(x: Array, n: int) -> Array:
    arr = np.asarray(x, dtype=np.float64).ravel().copy()
    n_use = int(max(1, min(n, arr.size // 2)))
    a = float(np.mean(np.concatenate((arr[:n_use], arr[-n_use:]))))
    arr -= a
    arr[arr < 0] = 1e-5
    return arr


def _side(mw: Array, st: Array, pos_neg: Array, left: bool) -> tuple[Array, float, float]:
    n = int(np.asarray(pos_neg).size)
    signs = np.asarray(pos_neg, dtype=np.float64).copy()
    if left:
        signs[n // 2 + 1 :] = signs[1 : n // 2][::-1]
    else:
        signs[1 : n // 2] = signs[n // 2 + 1 :][::-1]

    stc = np.asarray(st, dtype=np.complex128) * signs
    swc = _fftc(stc, axis=0)
    swc = _super_gaussian_1d(swc, 0.65, 5.0)

    a = np.abs(swc)
    autos = _quickscale(np.abs(np.convolve(a, a, mode="same")))
    rms_err1 = float(np.mean(np.abs(autos - mw) ** 2))

    indmax = int(np.argmax(np.abs(np.real(swc))))
    sgn = np.sign(np.real(swc[indmax]))
    if sgn == 0:
        sgn = 1.0
    swc = sgn * swc

    neg_vals = np.zeros_like(swc)
    mask = np.real(swc) < 0
    neg_vals[mask] = swc[mask]
    rms_err2 = float(np.trapezoid(np.abs(neg_vals)))

    swc = _quickscale(np.abs(swc))
    return swc, rms_err1, rms_err2


def _make_toggle_patterns(indices: Array, n: int, left: bool) -> Array:
    idx = np.asarray(indices, dtype=np.int64)
    if idx.size == 0:
        return np.ones((1, n), dtype=np.float64)

    idx = np.sort(idx)
    num = 1 << int(idx.size)
    patterns = np.ones((num, n), dtype=np.float64)

    for mask in range(num):
        p = patterns[mask]
        for bit, start in enumerate(idx):
            if ((mask >> bit) & 1) == 0:
                continue
            if bit + 1 < idx.size:
                end = int(idx[bit + 1])
            else:
                end = (n // 2 + 1) if left else n
            if end > start:
                p[start:end] *= -1.0

    if left:
        patterns[:, n // 2 + 1 :] = patterns[:, 1 : n // 2][:, ::-1]
    else:
        patterns[:, 1 : n // 2] = patterns[:, n // 2 + 1 :][:, ::-1]

    return patterns


def _select_diverse(cands: Array, score1: Array, score2: Array, max_keep: int) -> Array:
    c = np.asarray(cands, dtype=np.float64)
    if c.ndim != 2 or c.shape[0] == 0:
        return np.ones((1, c.shape[1] if c.ndim == 2 else 1), dtype=np.float64)

    s1 = np.asarray(score1, dtype=np.float64).ravel()
    s2 = np.asarray(score2, dtype=np.float64).ravel()
    combined = s1 + 0.05 * s2

    order = np.argsort(combined)
    keep = min(max_keep, c.shape[0])
    if keep <= 1:
        return c[order[:1], :]
    if c.shape[0] <= keep:
        return c[order, :]

    selected = [int(order[0])]
    pool = order[: min(c.shape[0], max(keep * 8, 12))]

    while len(selected) < keep:
        best_idx = None
        best_delta = -1.0
        for idx in pool:
            ii = int(idx)
            if ii in selected:
                continue
            delta = 1.0
            ci = _quickscale(c[ii, :])
            for jj in selected:
                cj = _quickscale(c[jj, :])
                d = float(np.mean(np.abs(ci - cj) ** 2))
                delta *= np.sqrt(max(d, 1e-16))
            if delta > best_delta:
                best_delta = delta
                best_idx = ii

        if best_idx is None:
            break
        selected.append(best_idx)

    return c[selected, :]


def _spec_ret_marginal(mwin: Array, noise: bool, max_keep: int) -> Array:
    mw = _quickscale(np.asarray(mwin, dtype=np.float64).ravel())
    n = int(mw.size)

    if noise:
        mw = _side_noise_1d(mw, 5)
        mw[mw < 0] = 1e-6
        mw = _smooth_sgolay(mw)
        mw = _super_gaussian_1d(mw, 0.9, 10.0)
        mw[mw < 0] = 1e-6

    mw = _quickscale(mw)

    st2 = _ifftc((mw / (2.0 * np.pi)).astype(np.complex128), axis=0)
    st = np.sqrt(st2)

    real_st = np.real(st).copy()
    imag_st = np.imag(st).copy()

    pos_neg = np.ones(n, dtype=np.float64)
    c = np.asarray([0.09, 0.425, 1.0], dtype=np.float64)

    for i_step in range(4, n):
        real_pre = real_st[: i_step + 1]
        imag_pre = imag_st[: i_step + 1]

        real_flip = real_pre.copy()
        imag_flip = imag_pre.copy()
        real_flip[i_step] = -real_flip[i_step]
        imag_flip[i_step] = -imag_flip[i_step]

        d3_real = np.diff(np.diff(np.diff(real_pre)))
        d3_imag = np.diff(np.diff(np.diff(imag_pre)))
        d3_real_f = np.diff(np.diff(np.diff(real_flip)))
        d3_imag_f = np.diff(np.diff(np.diff(imag_flip)))

        d12_r = real_st[i_step] - real_st[i_step - 1]
        d01_r = real_st[i_step - 1] - real_st[i_step - 2]
        d12_rf = real_flip[i_step] - real_st[i_step - 1]
        d01_rf = real_st[i_step - 1] - real_st[i_step - 2]

        d12_i = imag_st[i_step] - imag_st[i_step - 1]
        d01_i = imag_st[i_step - 1] - imag_st[i_step - 2]
        d12_if = imag_flip[i_step] - imag_st[i_step - 1]
        d01_if = imag_st[i_step - 1] - imag_st[i_step - 2]

        wp_r = (abs(d12_r) ** 2) * c[0] + (abs(d12_r - d01_r) ** 2) * c[1] + (abs(d3_real[i_step - 3]) ** 2) * c[2]
        wn_r = (abs(d12_rf) ** 2) * c[0] + (abs(d12_rf - d01_rf) ** 2) * c[1] + (abs(d3_real_f[i_step - 3]) ** 2) * c[2]

        wp_i = (abs(d12_i) ** 2) * c[0] + (abs(d12_i - d01_i) ** 2) * c[1] + (abs(d3_imag[i_step - 3]) ** 2) * c[2]
        wn_i = (abs(d12_if) ** 2) * c[0] + (abs(d12_if - d01_if) ** 2) * c[1] + (abs(d3_imag_f[i_step - 3]) ** 2) * c[2]

        if (wp_r + wp_i) > (wn_r + wn_i):
            real_st[i_step:] *= -1.0
            imag_st[i_step:] *= -1.0
            pos_neg[i_step:] *= -1.0

    pos_neg1 = pos_neg.copy()
    pos_neg1[n // 2 + 1 :] = pos_neg1[1 : n // 2][::-1]
    pos_neg2 = pos_neg.copy()
    pos_neg2[1 : n // 2] = pos_neg2[n // 2 + 1 :][::-1]

    cands = []
    score_main = []
    score_neg = []

    sw, r1, r2 = _side(mw, st, pos_neg1, left=True)
    cands.append(sw)
    score_main.append(r1)
    score_neg.append(r2)

    sw, r1, r2 = _side(mw, st, pos_neg2, left=False)
    cands.append(sw)
    score_main.append(r1)
    score_neg.append(r2)

    minima_idx, _ = find_peaks(-np.real(st))
    if minima_idx.size:
        lo = int(round(0.2 * n))
        hi = int(round(0.8 * n))
        minima_idx = minima_idx[(minima_idx >= lo) & (minima_idx <= hi)]

    left_idx = minima_idx[minima_idx < n // 2]
    right_idx = minima_idx[minima_idx > n // 2]

    if left_idx.size > 5:
        ord_l = np.argsort(np.abs(np.real(st[left_idx])))[::-1][:5]
        left_idx = np.sort(left_idx[ord_l])
    if right_idx.size > 5:
        ord_r = np.argsort(np.abs(np.real(st[right_idx])))[::-1][:5]
        right_idx = np.sort(right_idx[ord_r])

    left_patterns = _make_toggle_patterns(left_idx, n, left=True)
    right_patterns = _make_toggle_patterns(right_idx, n, left=False)

    for p in left_patterns:
        sw, r1, r2 = _side(mw, st, pos_neg1 * p, left=True)
        cands.append(sw)
        score_main.append(r1)
        score_neg.append(r2)

    for p in right_patterns:
        sw, r1, r2 = _side(mw, st, pos_neg2 * p, left=False)
        cands.append(sw)
        score_main.append(r1)
        score_neg.append(r2)

    cands_arr = np.asarray(cands, dtype=np.float64)
    main_arr = np.asarray(score_main, dtype=np.float64)
    neg_arr = np.asarray(score_neg, dtype=np.float64)

    selected = _select_diverse(cands_arr, main_arr, neg_arr, max_keep=max(1, int(max_keep)))
    return np.abs(selected)


def _iteration_number_exp(n: int, snr_level: int) -> tuple[np.ndarray, np.ndarray]:
    s = int(np.clip(snr_level, 1, 4))

    # Empirical schedule used by the original RANA workflow.
    # For each grid size `n` and SNR bucket `s`, the table stores:
    # - iteration counts [full grid, intermediate grid, coarsest grid]
    # - seed populations [full grid, intermediate grid, coarsest grid]
    # Larger/lower-SNR problems keep more seeds alive on coarse grids so the
    # search is broader before the best candidates are promoted upward.
    if n <= 64:
        table = {
            1: ([800, 10, 20], [4, 8, 16]),
            2: ([800, 25, 25], [4, 12, 16]),
            3: ([800, 25, 25], [4, 12, 20]),
            4: ([800, 30, 35], [4, 20, 32]),
        }
    elif n == 128:
        table = {
            1: ([800, 15, 15], [4, 12, 20]),
            2: ([800, 25, 25], [4, 12, 24]),
            3: ([800, 30, 35], [4, 16, 28]),
            4: ([800, 30, 35], [4, 20, 40]),
        }
    elif n == 256:
        table = {
            1: ([800, 15, 15], [4, 12, 24]),
            2: ([800, 30, 30], [4, 16, 32]),
            3: ([800, 30, 35], [4, 20, 36]),
            4: ([800, 30, 35], [4, 24, 48]),
        }
    elif n == 512:
        table = {
            1: ([800, 25, 30], [4, 20, 32]),
            2: ([800, 30, 35], [4, 20, 36]),
            3: ([800, 30, 35], [4, 28, 44]),
            4: ([800, 30, 35], [4, 32, 52]),
        }
    elif n == 1024:
        table = {
            1: ([800, 25, 30], [4, 20, 32]),
            2: ([800, 30, 30], [4, 24, 36]),
            3: ([800, 30, 35], [4, 24, 40]),
            4: ([800, 30, 35], [8, 32, 52]),
        }
    elif n == 2048:
        table = {
            1: ([1000, 30, 30], [4, 28, 42]),
            2: ([1200, 30, 30], [4, 28, 48]),
            3: ([1200, 30, 35], [4, 28, 52]),
            4: ([1200, 35, 35], [8, 36, 60]),
        }
    else:
        table = {1: ([1200, 35, 40], [8, 36, 60]), 2: ([1200, 35, 40], [8, 40, 80]), 3: ([1200, 35, 40], [8, 40, 80]), 4: ([1200, 35, 40], [8, 40, 80])}

    it, ti = table[s]
    return np.asarray(it, dtype=np.int64), np.asarray(ti, dtype=np.int64)


def _bin_axis(coeff: float, dim: int, dt: float, option: int) -> Array:
    h = np.arange(-dim // 2, dim // 2, dtype=np.float64)
    if option == 1:
        return (float(coeff) * float(dt)) * h
    df = 1.0 / (max(float(dim), 1.0) * float(dt) * float(coeff))
    return df * h


def _bin_trace(a: Array, t1: Array, f1: Array, t2: Array, f2: Array) -> Array:
    interp = RegularGridInterpolator((np.asarray(f1, dtype=np.float64), np.asarray(t1, dtype=np.float64)), np.asarray(a, dtype=np.float64), method="linear", bounds_error=False, fill_value=1e-50)
    ff, tt = np.meshgrid(np.asarray(f2, dtype=np.float64), np.asarray(t2, dtype=np.float64), indexing="ij")
    pts = np.column_stack((ff.ravel(), tt.ravel()))
    out = interp(pts).reshape((len(f2), len(t2)))
    out[~np.isfinite(out)] = 1e-50
    return out


def _bin_spectrum(var_in: Array, f1: Array, f2: Array) -> Array:
    s = np.asarray(var_in, dtype=np.float64)
    if s.ndim == 1:
        s = s[None, :]

    f1_arr = np.asarray(f1, dtype=np.float64)
    f2_arr = np.asarray(f2, dtype=np.float64)
    out = np.zeros((s.shape[0], f2_arr.size), dtype=np.float64)

    for i in range(s.shape[0]):
        row = s[i, :]
        try:
            itp = PchipInterpolator(f1_arr, row, extrapolate=False)
            y = itp(f2_arr)
        except Exception:
            y = np.interp(f2_arr, f1_arr, row, left=0.0, right=0.0)
        y = np.asarray(y, dtype=np.float64)
        y[~np.isfinite(y)] = 0.0
        out[i, :] = y

    return out


def _multi_grid(isig_full: Array, t_full: Array, f_full: Array, swr_full: Array) -> tuple[list[Array], list[Array], list[Array], list[Array]]:
    isig_levels = [np.asarray(isig_full, dtype=np.float64)]
    t_levels = [np.asarray(t_full, dtype=np.float64)]
    f_levels = [np.asarray(f_full, dtype=np.float64)]
    swr_levels = [np.asarray(swr_full, dtype=np.float64)]

    n = int(isig_levels[0].shape[0])
    dt = float(np.mean(np.diff(t_levels[0]))) if t_levels[0].size > 1 else 1.0

    a = 2.0
    for k in (1, 2):
        coeff = a ** (k / 2.0)
        dim = max(2, n // (2**k))
        t_new = _bin_axis(coeff, dim, dt, option=1)
        f_new = _bin_axis(coeff, dim, dt, option=2)

        isig_new = _bin_trace(isig_levels[0], t_levels[0], f_levels[0], t_new, f_new)
        swr_new = _bin_spectrum(swr_levels[0], f_levels[0], f_new)

        isig_levels.append(isig_new)
        t_levels.append(t_new)
        f_levels.append(f_new)
        swr_levels.append(swr_new)

    return isig_levels, t_levels, f_levels, swr_levels


def _initial_spectra_from_marginal(mw: Array, geometry: str, max_keep: int) -> Array:
    geom = normalize_geometry_name(geometry)
    mw_arr = _quickscale(np.clip(np.asarray(mw, dtype=np.float64).ravel(), 0.0, None))
    keep = max(1, int(max_keep))
    if GEOMETRY_KERNELS[geom].rana_marginal_init:
        return _spec_ret_marginal(mw_arr, noise=True, max_keep=keep)
    if not np.any(mw_arr > 0):
        mw_arr = np.ones_like(mw_arr)
    return np.tile(mw_arr[None, :], (keep, 1))


def _compare_gerror(ew_int: Array, asig_amp: Array, weights: Array, kernel: _GeometryKernel) -> float:
    if HAS_RANA_CYTHON:
        return float(_compare_gerror_cy(ew_int, asig_amp, weights, kernel.geometry))
    field = _ifftc(np.asarray(ew_int, dtype=np.complex128), axis=0)
    esig = _fftc(kernel.calc_esig(field), axis=0)
    asig_wt = _quickscale(np.abs(esig))
    g, _ = _min_gerr(asig_wt, asig_amp, weights=weights)
    return float(g)


def _promote_initial_guesses(
    e_out: Array,
    measure: Array,
    f_curr: Array,
    f_next: Array,
    swr_next: Array,
    isig_next_amp: Array,
    isig_next_weights: Array,
    total_next: int,
    kernel: _GeometryKernel,
) -> Array:
    e_arr = np.asarray(e_out, dtype=np.complex128)
    m_arr = np.asarray(measure, dtype=np.float64)
    if e_arr.ndim != 2 or e_arr.shape[0] == 0:
        n_next = int(np.asarray(f_next).size)
        return np.zeros((max(1, total_next), n_next), dtype=np.complex128)

    order = np.argsort(m_arr)
    n_next = int(np.asarray(f_next).size)
    num_spec = int(np.asarray(swr_next).shape[0])
    swr = np.asarray(swr_next, dtype=np.float64)

    out = np.zeros((max(1, total_next), n_next), dtype=np.complex128)
    f_curr_arr = np.asarray(f_curr, dtype=np.float64)
    f_next_arr = np.asarray(f_next, dtype=np.float64)
    spec_amp = np.sqrt(np.abs(swr))

    for i_case in range(out.shape[0]):
        src = int(order[i_case % order.size])
        e_proc = _center_moment(e_arr[src, :])
        e_w = _fftc(e_proc, axis=0)

        ew_b = np.interp(f_next_arr, f_curr_arr, e_w, left=0.0 + 0.0j, right=0.0 + 0.0j)
        ew_b = np.asarray(ew_b, dtype=np.complex128)
        ew_b[~np.isfinite(ew_b)] = 0.0 + 0.0j

        # Convention: E = sqrt(I) * exp(-i*phase)
        phase = -np.angle(ew_b)
        phase_factor = np.exp(-1j * phase)
        rms_b = _compare_gerror(ew_b, isig_next_amp, weights=isig_next_weights, kernel=kernel)

        best_alt = None
        best_rms = float("inf")
        for ii in range(num_spec):
            candidate = spec_amp[ii, :] * phase_factor
            rms_candidate = _compare_gerror(candidate, isig_next_amp, weights=isig_next_weights, kernel=kernel)
            if rms_candidate < best_rms:
                best_rms = float(rms_candidate)
                best_alt = candidate

        ew = best_alt if (best_alt is not None and best_rms < rms_b) else ew_b
        out[i_case, :] = _ifftc(ew, axis=0)

    return out


def _quickfrog_py(
    asig_amp: Array,
    et0: Array,
    max_iter: int,
    g_cutoff: float,
    gp_cutoff: float,
    stall_abs: float,
    stall_ratio: float,
    weights: Array,
    kernel: _GeometryKernel,
    stop_requested: Optional[Callable[[], bool]] = None,
) -> tuple[Array, Array, Array, float, float, int, bool, Array]:
    e = np.asarray(et0, dtype=np.complex128).ravel().copy()
    n = int(max(1, max_iter))

    g_hist = [float("inf")]
    gp_hist = [float("inf")]
    z_hist = [float("inf")]

    g_best = float("inf")
    gp_best = float("inf")
    z_best = float("inf")

    et_best_g = e.copy()
    et_best_gp = e.copy()

    esig = kernel.calc_esig(e)
    esig_w = _fftc(esig, axis=0)
    esig_w = _mag_repl(esig_w, asig_amp, weights=weights)
    esig = _ifftc(esig_w, axis=0)

    k = 0
    stopped = False

    while (min(g_hist) > g_cutoff) and (min(gp_hist) > gp_cutoff) and (k < n - 1):
        if stop_requested is not None and bool(stop_requested()):
            stopped = True
            break

        k += 1
        dz = -kernel.dzde(esig, e)
        e, z = _min_zerr(esig, e, dz, kernel)
        e = _center_moment(e)

        esig = kernel.calc_esig(e)
        esig_w = _fftc(esig, axis=0)

        g, a = _min_gerr(esig_w, asig_amp, weights=weights)
        gp = _gprime_from_amp(asig_amp, esig_w, a, weights=weights)

        if a > 0:
            e = kernel.apply_g_factor(e, a)

        g_hist.append(float(g))
        gp_hist.append(float(gp))
        z_hist.append(float(z))

        if g <= g_best or z <= z_best:
            g_best = float(g)
            z_best = float(z)
            et_best_g = e.copy()

        if gp <= gp_best:
            gp_best = float(gp)
            et_best_gp = e.copy()

        esig_w = _mag_repl(esig_w, asig_amp, weights=weights)
        esig = _ifftc(esig_w, axis=0)

        if k > 50 and len(g_hist) >= 11:
            recent = np.asarray(g_hist[-10:], dtype=np.float64)
            stag = float(np.mean(np.abs(np.diff(recent))))
            if stag < stall_abs and stag < max(g_cutoff * stall_ratio, 1e-15):
                break

    if not stopped:
        k += 1
        dz = -kernel.dzde(esig, e)
        e, _ = _min_zerr(esig, e, dz, kernel)

        esig = kernel.calc_esig(e)
        esig_w = _fftc(esig, axis=0)
        g, a = _min_gerr(esig_w, asig_amp, weights=weights)
        gp = _gprime_from_amp(asig_amp, esig_w, a, weights=weights)

        if a > 0:
            e = kernel.apply_g_factor(e, a)

        g_hist.append(float(g))
        gp_hist.append(float(gp))

        if g <= g_best:
            g_best = float(g)
            et_best_g = e.copy()
        if gp <= gp_best:
            gp_best = float(gp)
            et_best_gp = e.copy()

    if not np.isfinite(g_best):
        g_best = _g_error(asig_amp, e, weights=weights, kernel=kernel)
        et_best_g = e.copy()
    if not np.isfinite(gp_best):
        gp_best = _gprime_error(asig_amp, e, weights=weights, kernel=kernel)
        et_best_gp = e.copy()

    return (
        et_best_g,
        e,
        et_best_gp,
        float(g_best),
        float(gp_best),
        int(k),
        bool(stopped),
        np.asarray(g_hist[1:], dtype=np.float64),
    )


def _quickfrog(
    asig_amp: Array,
    et0: Array,
    max_iter: int,
    g_cutoff: float,
    gp_cutoff: float,
    stall_abs: float,
    stall_ratio: float,
    weights: Array,
    kernel: _GeometryKernel,
    stop_requested: Optional[Callable[[], bool]] = None,
) -> tuple[Array, Array, Array, float, float, int, bool, Array]:
    if HAS_RANA_CYTHON:
        return _quickfrog_cy(
            asig_amp,
            et0,
            max_iter,
            g_cutoff,
            gp_cutoff,
            stall_abs,
            stall_ratio,
            weights,
            kernel.geometry,
            stop_requested,
        )
    return _quickfrog_py(
        asig_amp,
        et0,
        max_iter,
        g_cutoff,
        gp_cutoff,
        stall_abs,
        stall_ratio,
        weights=weights,
        kernel=kernel,
        stop_requested=stop_requested,
    )


class RANARetriever(Retriever):
    """RANA-inspired multi-grid retriever."""

    name = "rana"

    def _emit_progress(
        self,
        progress_cb: Optional[Callable[[dict], None]],
        stage: str,
        iter_num: int,
        iter_step: int,
        idx: int,
        total: int,
        field: Array,
        best_field: Optional[Array],
        grid,
        measured: Array,
        g_val: float,
        g_best: float,
        t0: float,
        g_hist: Optional[Array] = None,
    ) -> None:
        if progress_cb is None:
            return
        try:
            current_trace = np.ascontiguousarray(self.model.simulate_trace(field, grid), dtype=np.float64)
            line = f"[RANA] {stage} {iter_num} {idx}/{total} G={g_val:.5g}, best G={g_best:.5g}, t={time.perf_counter() - t0:.2f}s"
            best_payload = None
            if best_field is not None:
                best_arr = np.ascontiguousarray(np.asarray(best_field, dtype=np.complex128))
                if best_arr.size == np.asarray(field, dtype=np.complex128).size:
                    best_payload = best_arr
            progress_cb(
                {
                    "iter": float(iter_num),
                    "iter_step": int(iter_step),
                    "seed_index": int(idx),
                    "seed_total": int(total),
                    "stage": str(stage),
                    "current_gprime": float(np.nan),
                    "best_gprime": float(np.nan),
                    "current_g": float(g_val),
                    "best_g": float(g_best),
                    "elapsed_s": float(time.perf_counter() - t0),
                    "best_sigma": float(np.nan),
                    "line_text": line,
                    "field": np.ascontiguousarray(np.asarray(field, dtype=np.complex128)),
                    "best_field": best_payload,
                    "current_trace": current_trace,
                    "g_hist": None if g_hist is None else np.asarray(g_hist, dtype=np.float64).copy(),
                }
            )
        except Exception:
            return

    def retrieve(self, frog_trace: FrogTrace, guess: Optional[Array] = None) -> RetrievalResult:
        start = time.perf_counter()
        geometry = normalize_geometry_name(getattr(self.model, "geometry", frog_trace.geometry))
        kernel = GEOMETRY_KERNELS[geometry]
        # Package convention is [delay, frequency], while the RANA core below
        # follows the original [frequency, delay] layout used by the MATLAB code.
        measured = np.ascontiguousarray(
            np.clip(np.asarray(frog_trace.intensity, dtype=np.float64).T, 0.0, None)
        )
        grid = frog_trace.grid
        n = grid.n
        if n % 2:
            raise ValueError("RANA retriever currently requires an even grid size.")

        progress_cb = self.config.progress_callback if callable(getattr(self.config, "progress_callback", None)) else None
        stop_requested = self.config.stop_requested if callable(getattr(self.config, "stop_requested", None)) else None

        snr_level = DEFAULT_SNR_LEVEL
        g_cutoff = float(getattr(self.config, "rana_g_cutoff", 4e-3))
        gp_cutoff = float(getattr(self.config, "rana_gp_cutoff", 0.1))
        max_spectra = DEFAULT_MAX_SPECTRA
        full_iter_cap = int(getattr(self.config, "rana_full_iter_cap", 160))
        coarse_iter_cap = DEFAULT_COARSE_ITER_CAP
        stall_abs_coarse = DEFAULT_STALL_ABS_COARSE
        stall_ratio_coarse = DEFAULT_STALL_RATIO_COARSE
        stall_abs_full = DEFAULT_STALL_ABS_FULL
        stall_ratio_full = DEFAULT_STALL_RATIO_FULL
        weight_factor = float(getattr(self.config, "rana_weight_factor", DEFAULT_WEIGHT_FACTOR))

        i_frog = _quickscale(np.clip(measured, 0.0, None))
        i_frog_mw = i_frog

        t0 = np.sort(np.asarray(grid.delay, dtype=np.float64).ravel())
        if t0.size != n:
            raise ValueError("Delay axis length does not match trace size.")
        dt = float(np.mean(np.diff(t0))) if n > 1 else 1.0
        if not np.isfinite(dt) or dt == 0:
            dt = 1.0

        h = np.arange(-n // 2, n // 2, dtype=np.float64)
        f0 = (1.0 / (n * dt)) * h

        mw = np.trapezoid(_quickscale(i_frog_mw), axis=1)
        swr0 = _initial_spectra_from_marginal(mw, geometry, max_keep=max_spectra)

        isig_levels, t_levels, f_levels, swr_levels = _multi_grid(i_frog, t0, f0, swr0)
        asig_levels = [_safe_sqrt(level) for level in isig_levels]
        weight_levels = [_pixel_weights(asig, weight_factor) for asig in asig_levels]
        k_levels = len(t_levels)

        iter_array, tot_initial = _iteration_number_exp(n, snr_level)
        iter_array = iter_array.astype(np.int64)
        tot_initial = tot_initial.astype(np.int64)

        iter_array[0] = min(iter_array[0], max(10, full_iter_cap))
        iter_array[1:] = np.minimum(iter_array[1:], max(5, coarse_iter_cap))

        seeds = guess
        if seeds is not None:
            seed_vec = normalize_field(np.asarray(seeds, dtype=np.complex128).ravel())
            if seed_vec.size != t_levels[-1].size:
                seed_vec = None
        else:
            seed_vec = None

        if seed_vec is None:
            small_spec = np.asarray(swr_levels[-1], dtype=np.float64)
            num_init = int(max(1, tot_initial[-1]))
            num_spec = int(max(1, small_spec.shape[0]))
            n_small = int(t_levels[-1].size)
            seed_mat = np.zeros((num_init, n_small), dtype=np.complex128)
            for i in range(num_init):
                amp = np.sqrt(np.abs(small_spec[i % num_spec, :]))
                ph = np.exp(1j * 2.0 * np.pi * self.rng.random(n_small))
                seed_mat[i, :] = _ifftc(amp * ph, axis=0)
        else:
            seed_mat = seed_vec[None, :]

        best_field_full: Optional[Array] = None
        best_g_full = float("inf")
        best_iter = 0
        errors = []
        stopped_by_user = False

        for level in range(k_levels - 1, -1, -1):
            asig = asig_levels[level]
            weights = weight_levels[level]
            n_init = int(max(1, tot_initial[level]))

            if seed_mat.shape[0] < n_init:
                reps = int(np.ceil(n_init / seed_mat.shape[0]))
                seed_use = np.tile(seed_mat, (reps, 1))[:n_init, :]
            else:
                seed_use = seed_mat[:n_init, :]

            best_g_coarse = float("inf")
            if level >= 1:
                e_out = np.zeros((n_init, seed_use.shape[1]), dtype=np.complex128)
                measure = np.full(n_init, np.inf, dtype=np.float64)
                for i in range(n_init):
                    etb, _, _, g, _, _, stopped, _ = _quickfrog(
                        asig,
                        seed_use[i, :],
                        max_iter=int(iter_array[level]),
                        g_cutoff=1e-6,
                        gp_cutoff=gp_cutoff,
                        stall_abs=stall_abs_coarse,
                        stall_ratio=stall_ratio_coarse,
                        weights=weights,
                        kernel=kernel,
                        stop_requested=None,
                    )
                    e_out[i, :] = etb
                    measure[i] = g
                    errors.append(float(g))

                    if g < best_g_coarse:
                        best_g_coarse = float(g)

                    self._emit_progress(
                        progress_cb,
                        stage=f"grid {level+1}/{k_levels}",
                        iter_num = iter_array[level],
                        iter_step = iter_array[level],
                        idx=i + 1,
                        total=n_init,
                        field=etb,
                        best_field=None,
                        grid=grid,
                        measured=measured,
                        g_val=float(g),
                        g_best=float(best_g_coarse),
                        t0=start,
                    )

                next_total = int(max(1, tot_initial[level - 1]))
                seed_mat = _promote_initial_guesses(
                    e_out,
                    measure,
                    f_levels[level],
                    f_levels[level - 1],
                    swr_levels[level - 1],
                    asig_levels[level - 1],
                    weight_levels[level - 1],
                    next_total,
                    kernel,
                )
            else:
                iter_total = int(iter_array[0])
                chunk = 10
                fields = seed_use.astype(np.complex128).copy()
                g_vals = np.full(n_init, np.inf, dtype=np.float64)
                iters_done = np.zeros(n_init, dtype=np.int64)
                converged = False
                iter_used = 0

                while (not converged) and (iter_used < iter_total) and (not stopped_by_user):
                    cur_chunk = int(min(chunk, iter_total - iter_used))
                    first_round = (iter_used == 0)
                    processed_count = 0
                    chunk_histories: list[Optional[Array]] = [None] * n_init
                    for i in range(n_init):
                        etb, _, etbp, g, gp, k_it, stopped, g_hist = _quickfrog(
                            asig,
                            fields[i, :],
                            max_iter=cur_chunk,
                            g_cutoff=g_cutoff,
                            gp_cutoff=gp_cutoff,
                            stall_abs=stall_abs_full,
                            stall_ratio=stall_ratio_full,
                            weights=weights,
                            kernel=kernel,
                            stop_requested=None if (first_round and i == 0) else stop_requested,
                        )
                        if gp <= gp_cutoff and g > g_cutoff:
                            etb = np.asarray(etbp, dtype=np.complex128)
                            g = float(_g_error_fast(asig, etb, weights=weights, kernel=kernel))
                        fields[i, :] = etb
                        g_vals[i] = float(g)
                        iters_done[i] += int(k_it)
                        errors.append(float(g))
                        processed_count = i + 1
                        chunk_histories[i] = np.asarray(g_hist, dtype=np.float64)

                        if g < best_g_full:
                            best_g_full = float(g)
                            best_field_full = np.asarray(etb, dtype=np.complex128).copy()

                        if g <= g_cutoff:
                            converged = True
                        if stopped:
                            stopped_by_user = True
                            break

                    if processed_count > 0:
                        valid_g = np.asarray(g_vals[:processed_count], dtype=np.float64)
                        chunk_best_local = int(np.argmin(valid_g))
                        chunk_best_field = np.asarray(fields[chunk_best_local, :], dtype=np.complex128)
                        chunk_best_g = float(valid_g[chunk_best_local])
                        chunk_best_hist = chunk_histories[chunk_best_local]
                        self._emit_progress(
                            progress_cb,
                            stage="full grid",
                            iter_num=iter_used + cur_chunk,
                            iter_step=cur_chunk,
                            idx=chunk_best_local + 1,
                            total=processed_count,
                            field=chunk_best_field,
                            best_field=best_field_full,
                            grid=grid,
                            measured=measured,
                            g_val=chunk_best_g,
                            g_best=float(best_g_full),
                            t0=start,
                            g_hist=chunk_best_hist,
                        )

                    iter_used += cur_chunk

                if stopped_by_user:
                    break

                # Pick best seed.
                ind_best = int(np.argmin(g_vals))
                g_out_dum = float(g_vals[ind_best])
                best_iter = int(iters_done[ind_best])
                best_g_full = min(best_g_full, g_out_dum)

        field_candidate = np.asarray(best_field_full, dtype=np.complex128)
        if field_candidate.size != n:
            raise ValueError(
                f"Internal RANA size error: full-grid field length {field_candidate.size} does not match grid size {n}"
            )

        field = normalize_field(field_candidate)
        retrieved_trace = np.ascontiguousarray(self.model.simulate_trace(field, grid), dtype=np.float64)

        asig_full = asig_levels[0]
        weights_full = weight_levels[0]
        g_out, gp_out = _g_gprime_error_fast(asig_full, field, weights=weights_full, kernel=kernel)

        return RetrievalResult(
            field=np.ascontiguousarray(field, dtype=np.complex128),
            retrieved_trace=retrieved_trace,
            errors=np.asarray(errors, dtype=np.float64),
            best_iteration=int(best_iter),
            best_sigma=float("nan"),
            perturbations=0,
            diagnostics={
                "best_error_g": float(g_out),
                "best_error_gprime": float(gp_out),
                "rana_cython_enabled": bool(HAS_RANA_CYTHON),
                "geometry": geometry,
                "rana_weight_factor": float(weight_factor),
                "rana_zero_pixels": int(
                    np.count_nonzero(~(np.clip(np.asarray(frog_trace.intensity, dtype=np.float64), 0.0, None) > 0.0))
                ),
                "rana_nonzero_pixels": int(
                    np.count_nonzero(np.clip(np.asarray(frog_trace.intensity, dtype=np.float64), 0.0, None) > 0.0)
                ),
                "stopped_by_user": bool(stopped_by_user),
                "rana_spectra_count": int(np.asarray(swr0).shape[0]),
                "rana_iter_array": np.asarray(iter_array, dtype=np.int64),
                "rana_tot_initial": np.asarray(tot_initial, dtype=np.int64),
                "rana_g_cutoff": float(g_cutoff),
                "rana_gp_cutoff": float(gp_cutoff),
                "rana_stall_abs_coarse": float(stall_abs_coarse),
                "rana_stall_ratio_coarse": float(stall_ratio_coarse),
                "rana_stall_abs_full": float(stall_abs_full),
                "rana_stall_ratio_full": float(stall_ratio_full),
                "elapsed_s": float(time.perf_counter() - start),
            },
        )

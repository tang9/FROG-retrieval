# cython: boundscheck=False, wraparound=False, cdivision=True, language_level=3
"""Cython kernels mirroring the geometry-aware RANA hot path.

This module keeps the same public API as ``rana.py`` while implementing the
hot loops with typed memoryviews and fixed-size coefficient accumulation.
"""

import numpy as np
cimport numpy as cnp
from libc.math cimport sqrt

cnp.import_array()

ctypedef cnp.complex128_t complex_t

cdef int GEOM_SHG = 0
cdef int GEOM_PG = 1
cdef int GEOM_THG = 2
cdef int GEOM_SD = 3


cdef inline double _abs2(complex_t z):
    return z.real * z.real + z.imag * z.imag


cdef inline double _real_prod_conj(complex_t a, complex_t b):
    return a.real * b.real + a.imag * b.imag


cdef inline void _accumulate_quadratic(double* coeff, complex_t b0, complex_t b1, complex_t b2):
    coeff[0] += _abs2(b0)
    coeff[1] += 2.0 * _real_prod_conj(b1, b0)
    coeff[2] += _abs2(b1) + 2.0 * _real_prod_conj(b2, b0)
    coeff[3] += 2.0 * _real_prod_conj(b2, b1)
    coeff[4] += _abs2(b2)


cdef inline void _accumulate_cubic(double* coeff, complex_t b0, complex_t b1, complex_t b2, complex_t b3):
    coeff[0] += _abs2(b0)
    coeff[1] += 2.0 * _real_prod_conj(b1, b0)
    coeff[2] += _abs2(b1) + 2.0 * _real_prod_conj(b2, b0)
    coeff[3] += 2.0 * _real_prod_conj(b2, b1) + 2.0 * _real_prod_conj(b3, b0)
    coeff[4] += _abs2(b2) + 2.0 * _real_prod_conj(b3, b1)
    coeff[5] += 2.0 * _real_prod_conj(b3, b2)
    coeff[6] += _abs2(b3)


def _normalize_geometry_name(geometry):
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
    if value not in ("shg-frog", "pg-frog", "thg-frog", "sd-frog"):
        raise ValueError(f"Unsupported FROG geometry '{geometry}'.")
    return value


cdef int _geometry_code(geometry) except -1:
    cdef str value = _normalize_geometry_name(geometry)
    if value == "shg-frog":
        return GEOM_SHG
    if value == "pg-frog":
        return GEOM_PG
    if value == "thg-frog":
        return GEOM_THG
    return GEOM_SD


def _quickscale(x):
    x_arr = np.asarray(x)
    peak = float(np.max(np.abs(x_arr))) if x_arr.size else 0.0
    if peak <= 1e-30:
        return x_arr.copy()
    return x_arr / peak


def _fftc(x, int axis=0):
    return np.fft.fftshift(np.fft.fft(np.fft.ifftshift(x, axes=axis), axis=axis), axes=axis)


def _ifftc(x, int axis=0):
    cdef int n = int(np.asarray(x).shape[axis])
    if n % 2:
        return np.fft.fftshift(np.fft.ifft(np.fft.ifftshift(x, axes=axis), axis=axis), axes=axis)
    return np.fft.ifftshift(np.fft.ifft(np.fft.fftshift(x, axes=axis), axis=axis), axes=axis)


def _center_moment(x):
    x_arr = np.asarray(x)
    w = np.abs(x_arr).astype(np.float64, copy=False)
    s = float(np.sum(w))
    if s <= 1e-30:
        return x_arr.copy()
    n = int(w.size)
    ic = float(np.sum((np.arange(n, dtype=np.float64) + 1.0) * w) / s)
    shift = int(np.floor(n / 2.0) - np.round(ic) + 1)
    return np.roll(x_arr, shift)


def _probe_gate_from_field(field, geometry):
    f = np.asarray(field, dtype=np.complex128).ravel()
    geom = _normalize_geometry_name(geometry)
    if geom == "shg-frog":
        return f, f
    if geom == "pg-frog":
        return f, np.abs(f) ** 2
    if geom == "thg-frog":
        return f, f * f
    if geom == "sd-frog":
        return np.conj(f), f * f
    raise ValueError(f"Unsupported FROG geometry '{geometry}'.")


cdef cnp.ndarray[complex_t, ndim=2] _calc_esig_geom_code(object et, int geom):
    cdef cnp.ndarray[complex_t, ndim=1] e = np.ascontiguousarray(np.asarray(et, dtype=np.complex128).ravel())
    cdef int n = e.shape[0]
    cdef int half = n // 2
    cdef cnp.ndarray[complex_t, ndim=2] esig = np.zeros((n, n), dtype=np.complex128)
    cdef complex_t[::1] ev = e
    cdef complex_t[:, ::1] esv = esig
    cdef int j, s, t, tp, t_start, t_end
    cdef complex_t etp
    cdef double gate_abs2

    for j in range(n):
        s = j - half
        if s >= 0:
            t_start = s
            t_end = n
        else:
            t_start = 0
            t_end = n + s
        for t in range(t_start, t_end):
            tp = t - s
            if geom == GEOM_SHG:
                esv[t, j] = ev[t] * ev[tp]
            elif geom == GEOM_PG:
                gate_abs2 = _abs2(ev[tp])
                esv[t, j] = ev[t] * gate_abs2
            elif geom == GEOM_THG:
                etp = ev[tp] * ev[tp]
                esv[t, j] = ev[t] * etp
            else:
                etp = ev[tp] * ev[tp]
                esv[t, j] = ev[t].conjugate() * etp
    return esig


def _calc_esig_probe_gate(probe, gate):
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


def _calc_esig_geom(et, geometry):
    return _calc_esig_geom_code(et, _geometry_code(geometry))


def _pixel_weights(asig, double weight_factor=1.0):
    a = np.asarray(asig, dtype=np.float64)
    nonzero = np.isfinite(a) & (a > 0.0)
    weights = np.full_like(a, weight_factor)
    weights[nonzero] = 1.0
    return weights


def _mag_repl(esig, asig, weights):
    es = np.asarray(esig, dtype=np.complex128)
    a = np.asarray(asig, dtype=np.float64)
    w = np.asarray(weights, dtype=np.float64)
    temp = np.abs(es)
    target = es.copy()
    mask = temp > 0
    target[mask] = a[mask] * es[mask] / temp[mask]
    target[~mask] = a[~mask] + 0.0j
    return w * target + (1.0 - w) * es


def _min_gerr(esig, asig, weights):
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
    w_sum = float(np.sum(w))
    if w_sum <= 1e-30:
        return float("inf"), 0.0
    g = float(np.sqrt(np.sum(w * (a2 - scale * e2) ** 2) / w_sum) / mx)
    return g, scale


def _gprime_from_amp(asig, esig, double scale, weights):
    a = np.asarray(asig, dtype=np.float64)
    w = np.asarray(weights, dtype=np.float64)
    diff = np.abs(a ** 2 - scale * (np.abs(esig) ** 2)) ** 2
    num = float(np.trapezoid(np.trapezoid(w * diff, axis=0), axis=0))
    den = float(np.trapezoid(np.trapezoid(w * (a ** 4), axis=0), axis=0))
    if den <= 1e-30:
        return float("inf")
    return float(np.sqrt(num / den)) if num > 0.0 else 0.0


def _g_gprime_error(asig, et, weights, geometry):
    cdef int geom = _geometry_code(geometry)
    esig = _fftc(_calc_esig_geom_code(et, geom), axis=0)
    g, a = _min_gerr(esig, asig, weights)
    if not np.isfinite(g):
        return float("inf"), float("inf")
    return float(g), float(_gprime_from_amp(asig, esig, a, weights))


def _g_error(asig, et, weights, geometry):
    g, _ = _g_gprime_error(asig, et, weights, geometry)
    return float(g)


def _gprime_error(asig, et, weights, geometry):
    _, gp = _g_gprime_error(asig, et, weights, geometry)
    return float(gp)


def compare_gerror_cy(ew_int, asig_amp, weights, geometry):
    field = _ifftc(np.asarray(ew_int, dtype=np.complex128), axis=0)
    esig = _fftc(_calc_esig_geom_code(field, _geometry_code(geometry)), axis=0)
    asig_wt = _quickscale(np.abs(esig))
    g, _ = _min_gerr(asig_wt, asig_amp, weights)
    return float(g)


cdef cnp.ndarray[complex_t, ndim=1] _dzde_code(object esigp, object et, int geom):
    cdef cnp.ndarray[complex_t, ndim=2] es = np.asarray(esigp, dtype=np.complex128)
    cdef cnp.ndarray[complex_t, ndim=1] e = np.ascontiguousarray(np.asarray(et, dtype=np.complex128).ravel())
    cdef int n = e.shape[0]
    cdef int half = n // 2
    cdef cnp.ndarray[complex_t, ndim=1] d = np.zeros(n, dtype=np.complex128)
    cdef complex_t[:, ::1] esv = es
    cdef complex_t[::1] ev = e
    cdef complex_t[::1] dv = d
    cdef int j, s, t, tp, t0
    cdef double scale = 1.0 / float(es.shape[0] * es.shape[1])
    cdef double gate_abs2, term_real
    cdef complex_t et0, etp, total

    if geom == GEOM_SHG:
        for j in range(n):
            s = j - half
            if s >= 0:
                for t in range(s, n):
                    tp = t - s
                    dv[t] = dv[t] + (ev[t] * ev[tp] - esv[t, j]) * ev[tp].conjugate()
                for t in range(0, n - s):
                    tp = t + s
                    dv[t] = dv[t] + (ev[t] * ev[tp] - esv[tp, j]) * ev[tp].conjugate()
            else:
                for t in range(0, n + s):
                    tp = t - s
                    dv[t] = dv[t] + (ev[t] * ev[tp] - esv[t, j]) * ev[tp].conjugate()
                for t in range(-s, n):
                    tp = t + s
                    dv[t] = dv[t] + (ev[t] * ev[tp] - esv[tp, j]) * ev[tp].conjugate()
        for t in range(n):
            dv[t] = dv[t] * scale
        return d

    for t0 in range(n):
        et0 = ev[t0]
        total = 0.0 + 0.0j
        for j in range(n):
            s = j - half
            tp = t0 - s
            if 0 <= tp < n:
                etp = ev[tp]
                if geom == GEOM_PG:
                    gate_abs2 = _abs2(etp)
                    total = total + (et0 * gate_abs2 - esv[t0, j]) * gate_abs2
                elif geom == GEOM_THG:
                    total = total + (et0 * etp * etp - esv[t0, j]) * (etp * etp).conjugate()
                else:
                    total = total + (et0 * etp.conjugate() * etp.conjugate() - esv[t0, j].conjugate()) * (etp * etp)
            tp = t0 + s
            if 0 <= tp < n:
                etp = ev[tp]
                if geom == GEOM_PG:
                    term_real = _abs2(etp) * _abs2(et0) - _real_prod_conj(esv[tp, j], etp)
                    total = total + 2.0 * term_real * et0
                elif geom == GEOM_THG:
                    total = total + 2.0 * (etp * et0).conjugate() * (etp * et0 * et0 - esv[tp, j])
                else:
                    total = total + (etp.conjugate() * et0 * et0 - esv[tp, j]) * 2.0 * etp * et0.conjugate()
        dv[t0] = total * scale
    return d


def _dzdE_shg(esigp, et):
    return _dzde_code(esigp, et, GEOM_SHG)


def _dzdE_pg(esigp, et):
    return _dzde_code(esigp, et, GEOM_PG)


def _dzdE_thg(esigp, et):
    return _dzde_code(esigp, et, GEOM_THG)


def _dzdE_sd(esigp, et):
    return _dzde_code(esigp, et, GEOM_SD)


def _signal_coeffs_shg(e_t, e_tp, d_t, d_tp):
    return np.asarray([e_t * e_tp, d_t * e_tp + e_t * d_tp, d_t * d_tp], dtype=np.complex128)


def _signal_coeffs_pg(e_t, e_tp, d_t, d_tp):
    c0 = float(np.abs(e_tp) ** 2)
    c1 = 2.0 * float(np.real(d_tp * np.conj(e_tp)))
    c2 = float(np.abs(d_tp) ** 2)
    return np.asarray([e_t * c0, d_t * c0 + e_t * c1, d_t * c1 + e_t * c2, d_t * c2], dtype=np.complex128)


def _signal_coeffs_thg(e_t, e_tp, d_t, d_tp):
    return np.asarray(
        [
            e_t * e_tp * e_tp,
            2.0 * e_t * e_tp * d_tp + d_t * e_tp * e_tp,
            e_t * d_tp * d_tp + 2.0 * d_t * e_tp * d_tp,
            d_t * d_tp * d_tp,
        ],
        dtype=np.complex128,
    )


def _signal_coeffs_sd(e_t, e_tp, d_t, d_tp):
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


def _signal_coeffs_for_geometry(e_t, e_tp, d_t, d_tp, geometry):
    cdef int geom = _geometry_code(geometry)
    if geom == GEOM_SHG:
        return _signal_coeffs_shg(e_t, e_tp, d_t, d_tp)
    if geom == GEOM_PG:
        return _signal_coeffs_pg(e_t, e_tp, d_t, d_tp)
    if geom == GEOM_THG:
        return _signal_coeffs_thg(e_t, e_tp, d_t, d_tp)
    return _signal_coeffs_sd(e_t, e_tp, d_t, d_tp)


def _dzde(esigp, et, geometry):
    return _dzde_code(esigp, et, _geometry_code(geometry))


cdef tuple _min_zerr_code(object esig, object et, object dz, int geom):
    cdef cnp.ndarray[complex_t, ndim=2] es = np.asarray(esig, dtype=np.complex128)
    cdef cnp.ndarray[complex_t, ndim=1] e = np.ascontiguousarray(np.asarray(et, dtype=np.complex128).ravel())
    cdef cnp.ndarray[complex_t, ndim=1] d = np.ascontiguousarray(np.asarray(dz, dtype=np.complex128).ravel())
    cdef int n = e.shape[0]
    cdef int half = n // 2
    cdef double mx = float(np.max(np.abs(es) ** 2)) if es.size else 0.0
    cdef complex_t[:, ::1] esv = es
    cdef complex_t[::1] ev = e
    cdef complex_t[::1] dv = d
    cdef int j, s, t, tp, t_start, t_end, i, coeff_len, degree
    cdef double scale, zmin, x, z
    cdef double coeff[7]
    cdef cnp.ndarray[cnp.float64_t, ndim=1] coeff_np, dcoeff, real_roots, zvals
    cdef object roots
    cdef complex_t e_t, e_tp, d_t, d_tp, b0, b1, b2, b3
    cdef double c0, c1, c2

    if mx <= 1e-30:
        return e.copy(), 0.0

    for i in range(7):
        coeff[i] = 0.0

    if geom == GEOM_SHG:
        coeff_len = 5
    else:
        coeff_len = 7

    for j in range(n):
        s = j - half
        if s >= 0:
            t_start = s
            t_end = n
        else:
            t_start = 0
            t_end = n + s
        for t in range(t_start, t_end):
            tp = t - s
            e_t = ev[t]
            e_tp = ev[tp]
            d_t = dv[t]
            d_tp = dv[tp]
            if geom == GEOM_SHG:
                b0 = e_t * e_tp - esv[t, j]
                b1 = d_t * e_tp + e_t * d_tp
                b2 = d_t * d_tp
                _accumulate_quadratic(coeff, b0, b1, b2)
            elif geom == GEOM_PG:
                c0 = _abs2(e_tp)
                c1 = 2.0 * _real_prod_conj(d_tp, e_tp)
                c2 = _abs2(d_tp)
                b0 = e_t * c0 - esv[t, j]
                b1 = d_t * c0 + e_t * c1
                b2 = d_t * c1 + e_t * c2
                b3 = d_t * c2
                _accumulate_cubic(coeff, b0, b1, b2, b3)
            elif geom == GEOM_THG:
                b0 = e_t * e_tp * e_tp - esv[t, j]
                b1 = 2.0 * e_t * e_tp * d_tp + d_t * e_tp * e_tp
                b2 = e_t * d_tp * d_tp + 2.0 * d_t * e_tp * d_tp
                b3 = d_t * d_tp * d_tp
                _accumulate_cubic(coeff, b0, b1, b2, b3)
            else:
                b0 = e_t.conjugate() * e_tp * e_tp - esv[t, j]
                b1 = 2.0 * e_t.conjugate() * e_tp * d_tp + d_t.conjugate() * e_tp * e_tp
                b2 = e_t.conjugate() * d_tp * d_tp + 2.0 * d_t.conjugate() * e_tp * d_tp
                b3 = d_t.conjugate() * d_tp * d_tp
                _accumulate_cubic(coeff, b0, b1, b2, b3)

    scale = float(es.shape[0] * es.shape[1]) * mx
    if scale <= 1e-30:
        return e.copy(), 0.0

    coeff_np = np.empty(coeff_len, dtype=np.float64)
    for i in range(coeff_len):
        coeff_np[coeff_len - 1 - i] = coeff[i] / scale

    if np.all(np.abs(coeff_np) <= 1e-30):
        return e.copy(), 0.0

    degree = coeff_len - 1
    dcoeff = np.empty(degree, dtype=np.float64)
    for i in range(degree):
        dcoeff[i] = coeff_np[i] * (degree - i)

    roots = np.roots(dcoeff)
    real_roots = roots[np.isclose(np.imag(roots), 0.0, atol=1e-10)].real
    if real_roots.size == 0:
        real_roots = np.asarray([float(np.real(roots[int(np.argmin(np.abs(np.imag(roots))))]))], dtype=np.float64)

    zvals = np.polyval(coeff_np, real_roots)
    i = int(np.argmin(zvals))
    x = float(real_roots[i])
    z = float(zvals[i])
    zmin = float(np.finfo(np.float64).eps * (coeff[0] / scale))
    if z < zmin:
        z = zmin
    z = sqrt(max(z, 0.0))
    return e + x * d, z


def _min_zerr(esig, et, dz, geometry):
    return _min_zerr_code(esig, et, dz, _geometry_code(geometry))


cdef cnp.ndarray[complex_t, ndim=1] _apply_g_factor_code(object et, double scale, int geom):
    cdef cnp.ndarray[complex_t, ndim=1] out = np.ascontiguousarray(np.asarray(et, dtype=np.complex128).ravel())
    cdef double power
    if scale <= 0.0:
        return out
    power = 0.25 if geom == GEOM_SHG else (1.0 / 6.0)
    return out * (scale ** power)


def _apply_g_factor(et, double scale, geometry):
    return _apply_g_factor_code(et, scale, _geometry_code(geometry))


def quickfrog_cy(
    asig_amp,
    et0,
    int max_iter,
    double g_cutoff,
    double gp_cutoff,
    double stall_abs,
    double stall_ratio,
    weights,
    geometry="shg-frog",
    stop_requested=None,
):
    """Geometry-aware Cython version of the quickfrog inner loop."""
    cdef int geom = _geometry_code(geometry)
    e = np.ascontiguousarray(np.asarray(et0, dtype=np.complex128).ravel()).copy()
    n = int(max(1, max_iter))

    g_hist = [float("inf")]
    gp_hist = [float("inf")]
    z_hist = [float("inf")]

    g_best = float("inf")
    gp_best = float("inf")
    z_best = float("inf")

    et_best_g = e.copy()
    et_best_gp = e.copy()

    esig = _calc_esig_geom_code(e, geom)
    esig_w = _fftc(esig, axis=0)
    esig_w = _mag_repl(esig_w, asig_amp, weights)
    esig = _ifftc(esig_w, axis=0)

    k = 0
    stopped = False

    while (min(g_hist) > g_cutoff) and (min(gp_hist) > gp_cutoff) and (k < n - 1):
        if stop_requested is not None and bool(stop_requested()):
            stopped = True
            break

        k += 1
        dz = -_dzde_code(esig, e, geom)
        e, z = _min_zerr_code(esig, e, dz, geom)
        e = _center_moment(e)

        esig = _calc_esig_geom_code(e, geom)
        esig_w = _fftc(esig, axis=0)

        g, a = _min_gerr(esig_w, asig_amp, weights)
        gp = _gprime_from_amp(asig_amp, esig_w, a, weights)

        if a > 0:
            e = _apply_g_factor_code(e, a, geom)

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

        esig_w = _mag_repl(esig_w, asig_amp, weights)
        esig = _ifftc(esig_w, axis=0)

        if k > 50 and len(g_hist) >= 11:
            recent = np.asarray(g_hist[len(g_hist) - 10 :], dtype=np.float64)
            if float(np.mean(np.abs(np.diff(recent)))) < stall_abs and float(np.mean(np.abs(np.diff(recent)))) < max(g_cutoff * stall_ratio, 1e-15):
                break

    if not stopped:
        k += 1
        dz = -_dzde_code(esig, e, geom)
        e, _ = _min_zerr_code(esig, e, dz, geom)

        esig = _calc_esig_geom_code(e, geom)
        esig_w = _fftc(esig, axis=0)
        g, a = _min_gerr(esig_w, asig_amp, weights)
        gp = _gprime_from_amp(asig_amp, esig_w, a, weights)

        if a > 0:
            e = _apply_g_factor_code(e, a, geom)

        if g <= g_best:
            g_best = float(g)
            et_best_g = e.copy()
        if gp <= gp_best:
            gp_best = float(gp)
            et_best_gp = e.copy()

    if not np.isfinite(g_best):
        g_best = _g_error(asig_amp, e, weights, geometry)
        et_best_g = e.copy()
    if not np.isfinite(gp_best):
        gp_best = _gprime_error(asig_amp, e, weights, geometry)
        et_best_gp = e.copy()

    return et_best_g, e, et_best_gp, float(g_best), float(gp_best), int(k), bool(stopped)

"""Local shared utilities copied from common_use to avoid external dependency."""

from __future__ import annotations
from typing import Any, Sequence
import matplotlib.colors as colors
import matplotlib.pyplot as plt
import numpy as np
import scipy.optimize as opt
from matplotlib import ticker
from matplotlib.colors import LinearSegmentedColormap

StrArrayLike = Sequence[str] | np.ndarray
FloatArrayLike = Sequence[float] | np.ndarray
IntArrayLike = Sequence[int] | np.ndarray
AnyArrayLike = Sequence[Any] | np.ndarray
Float2DArrayLike = Sequence[Sequence[float]] | np.ndarray

plt.rcParams.update({"font.size": 11, "font.family": "sans-serif"})
plt.rcParams.update({"figure.dpi": 100})
plt.rcParams["savefig.dpi"] = 200
plt.rcParams["legend.frameon"] = False

def ImageColorMap(cmap_name: str, num_colors: int = 256):
    if cmap_name == "Wh_jet":
        jet_cmap = plt.get_cmap("jet")
        jet_colors = jet_cmap(np.linspace(0, 1, round(num_colors / 16)))
        jet_colors[0] = [1, 1, 1, 1]
        return LinearSegmentedColormap.from_list("custom_jet", jet_colors, num_colors)
    return plt.get_cmap("jet")

custom_cmap = ImageColorMap("Wh_jet")

def plot_1d(
    ax: Any,
    x: FloatArrayLike,
    y: Float2DArrayLike,
    group: IntArrayLike | None = None,
    xerror: Float2DArrayLike | None = None,
    yerror: Float2DArrayLike | None = None,
    fmt: StrArrayLike | None = None,
    colors: AnyArrayLike | None = None,
    title: str | None = None,
    xlabel: str | None = None,
    ylabel: str | None = None,
    xlim: FloatArrayLike | None = None,
    ylim: FloatArrayLike | None = None,
    legends: StrArrayLike | None = None,
    show_legend: bool = False,
    labelsize: int | float | None = None,
    grid: bool | str = False,
    xscale: str | None = None,
    yscale: str | None = None,
    scientific: str | None = None,
    markersize: float = 5,
    capsize: float = 5,
    zorder: FloatArrayLike | None = None,
    linewidth: float | FloatArrayLike | None = None,
) -> Any:
    group = [0] if group is None else group
    if ax is None:
        fig = plt.figure(figsize=(8, 6))
        ax = fig.add_subplot(111)
    if xerror is None:
        xerror = [None] * len(group)
    if yerror is None:
        yerror = [None] * len(group)

    for ig, iy in enumerate(group):
        errorbar_container = None
        label = legends[ig] if legends and len(legends) > ig else None
        has_error = not (yerror[ig] is None and xerror[ig] is None)
        trace_color = None
        if colors is not None:
            if isinstance(colors, str) or np.isscalar(colors):
                trace_color = colors
            else:
                trace_color = colors[ig]

        if has_error:
            if fmt:
                errorbar_container = ax.errorbar(
                    x,
                    y[iy],
                    xerr=xerror[ig],
                    yerr=yerror[ig],
                    fmt=fmt[ig],
                    markersize=markersize,
                    capsize=capsize,
                    elinewidth=linewidth,
                    linewidth=linewidth,
                    color=trace_color,
                    ecolor=trace_color,
                    label=label,
                )
            else:
                errorbar_container = ax.errorbar(
                    x,
                    y[iy],
                    xerr=xerror[ig],
                    yerr=yerror[ig],
                    markersize=markersize,
                    capsize=capsize,
                    elinewidth=linewidth,
                    linewidth=linewidth,
                    color=trace_color,
                    ecolor=trace_color,
                    label=label,
                )
        else:
            if fmt:
                ax.plot(
                    x,
                    y[iy],
                    fmt[ig],
                    markersize=markersize,
                    linewidth=linewidth,
                    color=trace_color,
                    label=label,
                )
            else:
                ax.plot(
                    x,
                    y[iy],
                    markersize=markersize,
                    linewidth=linewidth,
                    color=trace_color,
                    label=label,
                )

        if trace_color is not None:
            lines = ax.get_lines()
            lines[-1].set_color(trace_color)
            if has_error and errorbar_container is not None:
                data_line, caplines, barlinecols = errorbar_container.lines
                data_line.set_color(trace_color)
                for capline in caplines:
                    capline.set_color(trace_color)
                for barlinecol in barlinecols:
                    barlinecol.set_color(trace_color)
        if zorder:
            lines = ax.get_lines()
            lines[-1].zorder = zorder[ig]
        if linewidth:
            lines = ax.get_lines()
            line_width = linewidth if not isinstance(linewidth, (list, tuple)) else linewidth[ig]
            lines[-1].set_linewidth(line_width)

    if xlim:
        ax.set_xlim(xlim)
    if ylim:
        ax.set_ylim(ylim)
    if title:
        ax.set_title(title, fontsize=labelsize * 1.1 if labelsize else None)
    if show_legend:
        ax.legend(fontsize=labelsize if labelsize else None)
    if grid:
        ax.grid(axis=grid if isinstance(grid, str) else "both")
    if xlabel:
        ax.set_xlabel(xlabel, fontsize=labelsize * 1.1 if labelsize else None)
    if ylabel:
        ax.set_ylabel(ylabel, fontsize=labelsize * 1.1 if labelsize else None)
    if xscale:
        ax.set_xscale(xscale)
    if yscale:
        ax.set_yscale(yscale)
    if scientific:
        ax.ticklabel_format(axis=scientific, style="scientific", scilimits=(-2.5, 2.5), useMathText=True)

    ax.tick_params(axis="both", which="major", direction="in", length=3, width=1, labelsize=labelsize if labelsize else None)
    ax.tick_params(axis="both", which="minor", direction="in", length=2, width=1, labelsize=labelsize if labelsize else None)
    for axis in ["top", "bottom", "left", "right"]:
        ax.spines[axis].set_linewidth(1)
    plt.tight_layout()
    return ax


def plot_2d(
    ax: Any,
    x: FloatArrayLike,
    y: FloatArrayLike,
    z: Any,
    levels: int | FloatArrayLike = 20,
    extend: str = "neither",
    logscale: bool = True,
    colorbar_ticks: FloatArrayLike | None = None,
    xlabel: str | None = None,
    ylabel: str | None = None,
    xlim: FloatArrayLike | None = None,
    ylim: FloatArrayLike | None = None,
    title: str | None = None,
    cmap: Any = custom_cmap,
    shrink: float = 1,
    labelsize: int | float | None = None,
    grid: bool | str = False,
    xscale: str | None = None,
    yscale: str | None = None,
    scientific: str = "",
    style: str = "contourf",
    colorbar: bool = True,
    cbarLoc: str = "right",
) -> Any:
    _ = cbarLoc
    if ax is None:
        fig = plt.figure(figsize=(8, 6))
        ax = fig.add_subplot(111)

    vmin = np.min(z)
    vmax = np.max(z)
    if not isinstance(levels, int):
        vmin = levels[0]
        vmax = levels[-1]

    if logscale:
        z = np.where(z <= 0, 1e-100, z)
        if style == "contourf":
            cs = ax.contourf(x, y, z.T, levels, locator=ticker.LogLocator(), cmap=cmap, extend=extend)
        else:
            cs = ax.pcolormesh(x, y, z.T, norm=colors.LogNorm(vmin, vmax), cmap=cmap, shading="nearest")
    else:
        if style == "contourf":
            cs = ax.contourf(x, y, z.T, levels, cmap=cmap, extend=extend)
        else:
            cs = ax.pcolormesh(x, y, z.T, norm=colors.Normalize(vmin, vmax), cmap=cmap, shading="nearest")

    cb = None
    if colorbar:
        if colorbar_ticks is not None:
            cb = plt.colorbar(cs, ticks=colorbar_ticks, shrink=shrink, pad=0.02, fraction=0.08, extend=extend, aspect=40)
        else:
            cb = plt.colorbar(cs, pad=0.02, fraction=0.08, extend=extend, aspect=40, shrink=shrink)
        for t in cb.ax.get_yticklabels():
            t.set_fontsize(12)
        cb.ax.tick_params(labelsize=labelsize if labelsize else None)

    if xlim:
        ax.set_xlim(xlim)
    if ylim:
        ax.set_ylim(ylim)
    if title:
        ax.set_title(title, fontsize=labelsize * 1.1 if labelsize else None)
    if grid:
        ax.grid(axis=grid if isinstance(grid, str) else "both")
    if xlabel:
        ax.set_xlabel(xlabel, fontsize=labelsize * 1.1 if labelsize else None)
    if ylabel:
        ax.set_ylabel(ylabel, fontsize=labelsize * 1.1 if labelsize else None)
    if xscale:
        ax.set_xscale(xscale)
    if yscale:
        ax.set_yscale(yscale)
    if "x" in scientific or "both" in scientific:
        ax.ticklabel_format(axis="x", style="scientific", scilimits=(-2, 2), useMathText=True)
    if "y" in scientific or "both" in scientific:
        ax.ticklabel_format(axis="y", style="scientific", scilimits=(-2, 2), useMathText=True)
    if "z" in scientific and cb is not None:
        cb.formatter.set_powerlimits((-3, 3))
        cb.formatter.set_useMathText(True)
        cb.ax.yaxis.set_offset_position("left")

    ax.tick_params(axis="both", which="major", direction="in", length=3, width=1, labelsize=labelsize if labelsize else None)
    ax.tick_params(axis="both", which="minor", direction="in", length=2, width=1, labelsize=labelsize if labelsize else None)
    ax.xaxis.set_ticks_position("both")
    ax.yaxis.set_ticks_position("both")
    for axis in ["top", "bottom", "left", "right"]:
        ax.spines[axis].set_linewidth(1)
    plt.tight_layout()
    return ax, cs


def find_index(L: FloatArrayLike, x: float) -> int:
    for i, t in enumerate(L):
        if x <= t:
            return i if (i == 0 or (L[i] - x) <= (x - L[i - 1])) else i - 1
    return len(L)-1


def gaussian_function(x: Any, area: float, cen: float, sig: float) -> Any:
    return area / np.sqrt(2 * np.pi) / sig * np.exp(-(x - cen) ** 2 / 2 / sig**2)


def find_peak_fwhm(x: FloatArrayLike, y: FloatArrayLike) -> tuple[float, float, float, float]:
    i_max, x_fwhm_1, x_fwhm_2 = 0, x[0], x[-1]
    for i in range(len(x)):
        if y[i_max] < y[i]:
            i_max = i
    y_fwhm = y[i_max] / 2
    for i in range(0, i_max-1):
        if y[i] <= y_fwhm and y[i + 1] > y_fwhm:
            x_fwhm_1 = x[i] + (x[i + 1] - x[i]) * (y_fwhm - y[i]) / (y[i + 1] - y[i])
            break
    for i in range(len(x) - 2, i_max, -1):
        if y[i] >= y_fwhm and y[i + 1] < y_fwhm:
            x_fwhm_2 = x[i] + (x[i + 1] - x[i]) * (y_fwhm - y[i]) / (y[i + 1] - y[i])
            break
    return x[i_max], y[i_max], x_fwhm_2 - x_fwhm_1, (x_fwhm_2 + x_fwhm_1) / 2

def find_peak_ratio(x: FloatArrayLike, y: FloatArrayLike, ratio) -> tuple[float, float]:
    """Return the max-value point (xi, yi) and FWHM."""
    i_max, x1, x2 = 0, x[0], x[-1]
    for i in range(len(x)):
        if y[i_max] < y[i]:
            i_max = i
    y_ratio = y[i_max] * ratio
    for i in range(0, i_max-1):
        if y[i] <= y_ratio and y[i + 1] > y_ratio:
            x1 = x[i] + (x[i + 1] - x[i]) * (y_ratio - y[i]) / (y[i + 1] - y[i])
            break
    for i in range(len(x) - 2, i_max, -1):
        if y[i] >= y_ratio and y[i + 1] < y_ratio:
            x2 = x[i] + (x[i + 1] - x[i]) * (y_ratio - y[i]) / (y[i + 1] - y[i])
            break
    return x[i_max], y[i_max], x1, x2

def fit_peak(x: FloatArrayLike, y: FloatArrayLike) -> tuple[Any, Any]:
    _, maximum, fwhm, center = find_peak_fwhm(x, y)
    p0 = [maximum * fwhm, center, fwhm / 2.35482]
    try:
        param, pcov = opt.curve_fit(gaussian_function, x, y, p0=p0)
    except Exception:
        print("Gaussian fit failed, returning initial guess.")
        param = p0
        pcov = np.zeros((3, 3))
    return param, pcov


def bin3(
    x: FloatArrayLike,
    y: FloatArrayLike,
    z: np.ndarray,
    N_bin: int,
    axis: int = 0,
    result: str = "average",
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if axis == 1:
        y, x = x, y
        z = z.T
    x_new = []
    z_new = []
    for i in range(len(x) // N_bin):
        xt, zt = 0, 0
        for ib in range(N_bin):
            xt += x[i * N_bin + ib]
            zt += z[i * N_bin + ib, :]
        x_new.append(xt / N_bin)
        if result[:3] in ["ave", "Ave", "AVE"]:
            z_new.append(zt / N_bin)
        else:
            z_new.append(zt)
    if axis == 0:
        return np.array(x_new), y, np.array(z_new)
    if axis == 1:
        return y, np.array(x_new), np.array(z_new).T
    return bin3(np.array(x_new), y, np.array(z_new), N_bin, axis=1)

def get_Et_from_Ef(freq, Ef, time_interval=None):
    #calculate E(t) from E(f) by inverse fourier transform
    index_sorted = np.argsort(freq) # sort the frequency in ascending order
    freq = freq[index_sorted]
    Ef = Ef[index_sorted]
    d_freq = np.diff(freq)
    if np.any(np.abs(d_freq - d_freq[0]) > 1e-6*np.median(d_freq)):
        print('Warning: frequency grid is not uniform, interpolation is needed')
        print("grid the frequency to uniform grid")
        freq_grid = np.linspace(freq[0], freq[-1], len(freq))
        Ef = np.interp(freq_grid, freq, Ef, left=0, right=0)
        freq = freq_grid
    df = np.median(np.diff(freq))
    if time_interval is not None:
        freq_length = 1/(time_interval)
        param,_ = fit_peak(freq, np.abs(Ef)**2)
        f0 = param[1]
        fmin_target = f0 - freq_length/2
        fmax_target = f0 + freq_length/2
        left_freq_add = np.arange(freq[0]-df, fmin_target-df, -df)[::-1]
        right_freq_add = np.arange(freq[-1]+df, fmax_target+df, df)
        freq = np.concatenate([left_freq_add, freq, right_freq_add])
        Ef = np.concatenate([np.zeros_like(left_freq_add), Ef, np.zeros_like(right_freq_add)])

    # E(t) = iFFT(E(f)) = FFT(E(-f))
    v_neg = -freq[::-1]
    Gv = Ef[::-1]
    fft = np.fft.fft(Gv)
    fft = np.fft.fftshift(fft)
    fft_int = fft.real**2+fft.imag**2
    fft_norm = fft/np.sqrt(max(fft_int))
    time = np.fft.fftfreq(len(freq), d = (freq[1]-freq[0]))
    time = np.fft.fftshift(time)
    #compensate the phase
    phase_compensate = np.exp(-1.0j*2*np.pi*v_neg[0]*time)
    Et = fft_norm * phase_compensate
    return time, Et

def get_Ef_from_Et(time: np.ndarray, Et: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Convert time-domain field E(t) to frequency-domain field E(f).

    Inputs:
    - time in fs (uniformly sampled)
    - Et complex field

    Returns:
    - frequency axis in 1/fs (equivalent to PHz)
    - normalized complex Ef with FFT shifted to centered frequency bins
    """
    t = np.asarray(time, dtype=np.float64).ravel()
    et = np.asarray(Et, dtype=np.complex128).ravel()
    if t.size != et.size:
        raise ValueError(f"time and Et length mismatch: {t.size} vs {et.size}")
    if t.size == 0:
        return np.asarray([], dtype=np.float64), np.asarray([], dtype=np.complex128)
    if t.size == 1:
        return np.asarray([0.0], dtype=np.float64), np.asarray([1.0 + 0.0j], dtype=np.complex128)

    dt = float(np.median(np.diff(t)))
    if not np.isfinite(dt) or abs(dt) < 1e-30:
        raise ValueError("Invalid time axis spacing for FFT.")

    fft = np.fft.fftshift(np.fft.fft(et))
    power = np.abs(fft) ** 2
    amp = np.sqrt(max(float(np.max(power)), 1e-30))
    fft_norm = fft / amp

    freq = np.fft.fftshift(np.fft.fftfreq(t.size, d=dt))
    # Compensate time-origin offset so phase is not dominated by a linear term from t[0].
    phase_compensate = np.exp(-1.0j * 2.0 * np.pi * t[0] * freq)
    Ef = fft_norm * phase_compensate
    return freq, Ef

__all__ = [
    "custom_cmap",
    "plot_1d",
    "plot_2d",
    "find_index",
    "gaussian_function",
    "find_peak_fwhm",
    "find_peak_ratio",
    "fit_peak",
    "bin3",
    "get_Et_from_Ef",
    "get_Ef_from_Et",
]

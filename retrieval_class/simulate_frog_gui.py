"""PySide6 + pyqtgraph GUI for SHG/PG/THG/SD FROG simulation."""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from typing import Callable, Optional

import numpy as np
import pyqtgraph as pg
from PySide6 import QtCore, QtWidgets

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_DIR = os.path.dirname(_THIS_DIR)

from .common import custom_cmap, find_peak_fwhm, get_Ef_from_Et, get_Et_from_Ef
from .field2trace import build_forward_model
from .types import FrogGrid

SPEED_LIGHT = 299792458.0
LIGHT_PHz_um = SPEED_LIGHT * 1e-9

_TRACE_COLORMAP = None


def _safe_normalize(y: np.ndarray) -> np.ndarray:
    out = np.asarray(y, dtype=np.float64).copy()
    ymax = float(np.max(out)) if out.size else 0.0
    if ymax > 1e-15:
        out /= ymax
    return out


def _normalize_field(field: np.ndarray) -> np.ndarray:
    out = np.asarray(field, dtype=np.complex128).copy()
    peak = float(np.max(np.abs(out))) if out.size else 0.0
    if peak > 1e-15:
        out /= peak
    return out


def _phase_for_display(complex_signal: np.ndarray, intensity: np.ndarray, threshold: float = 0.02) -> np.ndarray:
    phase = -np.unwrap(np.angle(np.asarray(complex_signal, dtype=np.complex128)))
    if phase.size:
        phase -= phase[int(np.argmax(np.asarray(intensity, dtype=np.float64)))]
    mask = np.asarray(intensity, dtype=np.float64) > float(threshold)
    return np.where(mask, phase, np.nan)


def _estimate_fwhm(x: np.ndarray, y: np.ndarray) -> float:
    x_arr = np.asarray(x, dtype=np.float64)
    y_arr = np.asarray(y, dtype=np.float64)
    if x_arr.size < 3 or y_arr.size < 3 or float(np.max(y_arr)) <= 0:
        return float("nan")
    try:
        out = find_peak_fwhm(x_arr, y_arr)
        if isinstance(out, tuple) and len(out) >= 3:
            return float(out[2])
    except Exception:
        pass
    y_norm = _safe_normalize(y_arr)
    idx = np.flatnonzero(y_norm >= 0.5)
    if idx.size < 2:
        return float("nan")
    return float(abs(x_arr[idx[-1]] - x_arr[idx[0]]))


def _get_trace_colormap():
    global _TRACE_COLORMAP
    if _TRACE_COLORMAP is not None:
        return _TRACE_COLORMAP
    if custom_cmap is not None:
        try:
            x = np.linspace(0.0, 1.0, 256)
            rgba = np.asarray(custom_cmap(x), dtype=np.float64)
            if rgba.ndim == 2 and rgba.shape[1] in (3, 4):
                if rgba.shape[1] == 3:
                    rgba = np.hstack((rgba, np.ones((rgba.shape[0], 1), dtype=np.float64)))
                colors = np.clip(rgba * 255, 0.0, 255.0).astype(np.ubyte)
                pos = np.linspace(0.0, 1.0, colors.shape[0], dtype=np.float64)
                _TRACE_COLORMAP = pg.ColorMap(pos, colors)
                return _TRACE_COLORMAP
        except Exception:
            pass
    _TRACE_COLORMAP = pg.colormap.get("jet", source="matplotlib")
    return _TRACE_COLORMAP


def _wavelength_um_to_frequency_phz(wavelength_um: np.ndarray | float) -> np.ndarray | float:
    return LIGHT_PHz_um / np.asarray(wavelength_um, dtype=np.float64)


def _frequency_phz_to_wavelength_um(freq_phz: np.ndarray | float) -> np.ndarray | float:
    return LIGHT_PHz_um / np.asarray(freq_phz, dtype=np.float64)


def _abs_dlambda_df_um_per_phz(freq_phz: np.ndarray | float) -> np.ndarray | float:
    freq = np.asarray(freq_phz, dtype=np.float64)
    out = np.zeros_like(freq, dtype=np.float64)
    positive = freq > 1e-12
    out[positive] = LIGHT_PHz_um / (freq[positive] ** 2)
    return out


def _abs_df_dlambda_phz_per_um(wavelength_um: np.ndarray | float) -> np.ndarray | float:
    wavelength = np.asarray(wavelength_um, dtype=np.float64)
    out = np.zeros_like(wavelength, dtype=np.float64)
    positive = wavelength > 1e-12
    out[positive] = LIGHT_PHz_um / (wavelength[positive] ** 2)
    return out


def _build_centered_time_axis(delay_range_fs: float, n: int) -> np.ndarray:
    dt = 2.0 * float(delay_range_fs) / float(n)
    return (np.arange(int(n), dtype=np.float64) - int(n) // 2) * dt


def _build_frequency_axis_from_time(time_fs: np.ndarray) -> np.ndarray:
    t = np.asarray(time_fs, dtype=np.float64)
    dt = float(np.median(np.diff(t)))
    return np.fft.fftshift(np.fft.fftfreq(t.size, d=dt))


def _load_three_column_profile(path: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    data = np.loadtxt(path, dtype=np.float64)
    data = np.atleast_2d(data)
    if data.shape[1] < 2:
        raise ValueError(f"{os.path.basename(path)} must contain at least two columns.")
    axis = np.asarray(data[:, 0], dtype=np.float64).ravel()
    intensity = np.clip(np.asarray(data[:, 1], dtype=np.float64).ravel(), 0.0, None)
    phase = np.asarray(data[:, 2], dtype=np.float64).ravel() if data.shape[1] >= 3 else np.zeros_like(intensity)
    if axis.size < 2:
        raise ValueError(f"{os.path.basename(path)} must contain at least two rows.")
    return axis, intensity, phase


def _interp_amplitude_phase(
    x_new: np.ndarray,
    x_old: np.ndarray,
    amplitude_old: np.ndarray,
    phase_old: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    x = np.asarray(x_old, dtype=np.float64).ravel()
    amp = np.asarray(amplitude_old, dtype=np.float64).ravel()
    phase = np.asarray(phase_old, dtype=np.float64).ravel()
    order = np.argsort(x)
    x = x[order]
    amp = amp[order]
    phase = phase[order]
    x_unique, idx = np.unique(x, return_index=True)
    amp = amp[idx]
    phase = np.unwrap(phase[idx])
    amp_new = np.interp(x_new, x_unique, amp, left=0.0, right=0.0)
    phase_new = np.interp(x_new, x_unique, phase, left=phase[0], right=phase[-1])
    return amp_new, phase_new


def _build_time_field_from_params(time_fs: np.ndarray, duration_fs: float, shape: str) -> np.ndarray:
    t = np.asarray(time_fs, dtype=np.float64)
    tau = float(duration_fs)
    if tau <= 0:
        raise ValueError("Pulse duration must be positive.")
    shape_key = str(shape).strip().lower()
    if shape_key == "gaussian":
        intensity = np.exp(-4.0 * np.log(2.0) * (t / tau) ** 2)
        field = np.sqrt(intensity).astype(np.complex128)
    elif shape_key == "sech":
        t0 = tau / 1.763
        field = (1.0 / np.cosh(t / t0)).astype(np.complex128)
    else:
        raise ValueError(f"Unsupported time-domain pulse shape: {shape}")
    return _normalize_field(field)


def _build_time_field_from_file(time_fs: np.ndarray, path: str) -> np.ndarray:
    axis, intensity, phase = _load_three_column_profile(path)
    amplitude = np.sqrt(np.clip(intensity, 0.0, None))
    amp_interp, phase_interp = _interp_amplitude_phase(np.asarray(time_fs, dtype=np.float64), axis, amplitude, phase)
    field = amp_interp * np.exp(-1.0j * phase_interp)
    return _normalize_field(field)


def _build_spectrum_from_params(
    freq_rel_phz: np.ndarray,
    center_wavelength_um: float,
    bandwidth_um: float,
) -> tuple[float, np.ndarray]:
    center_wl = float(center_wavelength_um)
    bandwidth = float(bandwidth_um)
    if center_wl <= 0:
        raise ValueError("Center wavelength must be positive.")
    if bandwidth <= 0:
        raise ValueError("Bandwidth must be positive.")
    center_freq = float(_wavelength_um_to_frequency_phz(center_wl))
    freq_abs = center_freq + np.asarray(freq_rel_phz, dtype=np.float64)
    intensity = np.zeros_like(freq_abs, dtype=np.float64)
    positive = freq_abs > 1e-12
    wavelength_um = np.empty_like(freq_abs, dtype=np.float64)
    wavelength_um[:] = np.nan
    wavelength_um[positive] = _frequency_phz_to_wavelength_um(freq_abs[positive])
    intensity_lambda = np.exp(-4.0 * np.log(2.0) * ((wavelength_um[positive] - center_wl) / bandwidth) ** 2)
    intensity[positive] = intensity_lambda * _abs_dlambda_df_um_per_phz(freq_abs[positive])
    return center_freq, _normalize_field(np.sqrt(intensity).astype(np.complex128))


def _build_spectrum_from_file(freq_rel_phz: np.ndarray, path: str) -> tuple[float, np.ndarray]:
    wavelength_um, intensity, phase = _load_three_column_profile(path)
    wavelength_um = np.asarray(wavelength_um, dtype=np.float64).ravel()
    positive = wavelength_um > 0
    if np.count_nonzero(positive) < 2:
        raise ValueError(f"{os.path.basename(path)} must contain at least two positive wavelengths.")
    wavelength_um = wavelength_um[positive]
    intensity = np.asarray(intensity, dtype=np.float64).ravel()[positive]
    phase = np.asarray(phase, dtype=np.float64).ravel()[positive]
    intensity_lambda = np.clip(intensity, 0.0, None)
    freq_abs_file = np.asarray(_wavelength_um_to_frequency_phz(wavelength_um), dtype=np.float64)
    intensity_freq_file = intensity_lambda * _abs_dlambda_df_um_per_phz(freq_abs_file)
    amplitude = np.sqrt(np.clip(intensity_freq_file, 0.0, None))
    center_freq = float(np.sum(freq_abs_file * intensity_freq_file) / max(np.sum(intensity_freq_file), 1e-15))
    freq_abs_target = center_freq + np.asarray(freq_rel_phz, dtype=np.float64)
    amp_interp, phase_interp = _interp_amplitude_phase(freq_abs_target, freq_abs_file, amplitude, phase)
    field = amp_interp * np.exp(-1.0j * phase_interp)
    if float(np.max(np.abs(field))) <= 1e-15:
        raise ValueError("Spectrum file does not overlap the selected frequency grid. Reduce delay range or increase N.")
    return center_freq, _normalize_field(field)


def _apply_chirp(freq_rel_phz: np.ndarray, spectrum: np.ndarray, gdd_fs2: float, tod_fs3: float) -> np.ndarray:
    omega = 2.0 * np.pi * np.asarray(freq_rel_phz, dtype=np.float64)
    phase = 0.5 * float(gdd_fs2) * omega**2 + (1.0 / 6.0) * float(tod_fs3) * omega**3
    return np.asarray(spectrum, dtype=np.complex128) * np.exp(-1.0j * phase)


def _map_trace_to_wavelength(freq_abs_phz: np.ndarray, trace: np.ndarray) -> tuple[np.ndarray, np.ndarray, str]:
    freq_abs = np.asarray(freq_abs_phz, dtype=np.float64).ravel()
    z = np.asarray(trace, dtype=np.float64)
    positive = freq_abs > 1e-12
    if np.count_nonzero(positive) < 2:
        raise ValueError("Trace frequency axis has fewer than two positive points; cannot build wavelength trace.")
    wavelength_um = np.asarray(_frequency_phz_to_wavelength_um(freq_abs[positive]), dtype=np.float64)
    order = np.argsort(wavelength_um)
    wavelength_sorted = wavelength_um[order]
    trace_sorted = z[:, positive][:, order] * _abs_df_dlambda_phz_per_um(wavelength_sorted)[None, :]
    note = ""
    if np.count_nonzero(positive) != freq_abs.size:
        note = f"Wavelength trace uses {np.count_nonzero(positive)}/{freq_abs.size} positive SHG frequency bins."
    return wavelength_sorted, _safe_normalize(trace_sorted), note


def _centers_to_edges(centers: np.ndarray) -> np.ndarray:
    c = np.asarray(centers, dtype=np.float64).ravel()
    if c.size < 2:
        raise ValueError("Need at least two centers to build cell edges.")
    edges = np.empty(c.size + 1, dtype=np.float64)
    edges[1:-1] = 0.5 * (c[:-1] + c[1:])
    edges[0] = c[0] - 0.5 * (c[1] - c[0])
    edges[-1] = c[-1] + 0.5 * (c[-1] - c[-2])
    return edges


@dataclass(slots=True)
class SimulationResult:
    time_fs: np.ndarray
    field_t: np.ndarray
    freq_rel_phz: np.ndarray
    field_f: np.ndarray
    freq_abs_phz: np.ndarray
    center_freq_phz: float
    geometry: str
    trace_delay_fs: np.ndarray
    trace_freq_phz: np.ndarray
    trace: np.ndarray
    trace_wave_um: np.ndarray
    trace_wave: np.ndarray
    wavelength_note: str


def _trace_frequency_multiplier(geometry: str) -> float:
    geom = str(geometry).strip().lower().replace("_", "-")
    if geom == "shg-frog":
        return 2.0
    if geom == "pg-frog":
        return 1.0
    if geom == "thg-frog":
        return 3.0
    if geom == "sd-frog":
        return 1.0
    raise ValueError(f"Unsupported geometry '{geometry}'.")


def _trace_label_prefix(geometry: str) -> str:
    geom = str(geometry).strip().lower().replace("_", "-")
    labels = {
        "shg-frog": "SHG",
        "pg-frog": "PG",
        "thg-frog": "THG",
        "sd-frog": "SD",
    }
    return labels.get(geom, geom.upper())


class _DualAxisCurve:
    def __init__(
        self,
        plot_item: pg.PlotItem,
        x_label: str,
        left_label: str,
        right_label: str,
        top_label: str | None = None,
        top_transform: Optional[Callable[[float], float]] = None,
    ):
        self.plot_item = plot_item
        self.top_transform = top_transform
        self.plot_item.showAxis("right")
        self.plot_item.getAxis("left").enableAutoSIPrefix(False)
        self.plot_item.getAxis("bottom").enableAutoSIPrefix(False)
        self.plot_item.getAxis("right").enableAutoSIPrefix(False)
        self.plot_item.setLabel("bottom", x_label)
        self.plot_item.setLabel("left", left_label)
        self.plot_item.getAxis("right").setLabel(right_label)
        self.plot_item.showGrid(x=True, y=True, alpha=0.25)
        self.plot_item.addLegend()
        if top_label is not None:
            self.plot_item.showAxis("top")
            self.plot_item.getAxis("top").enableAutoSIPrefix(False)
            self.plot_item.getAxis("top").setLabel(top_label)

        self.phase_vb = pg.ViewBox()
        self.plot_item.scene().addItem(self.phase_vb)
        self.plot_item.getAxis("right").linkToView(self.phase_vb)
        self.phase_vb.setXLink(self.plot_item)
        self.plot_item.vb.sigResized.connect(self._update_views)
        self.plot_item.vb.sigRangeChanged.connect(self._on_range_changed)
        self._update_views()

        self._intensity = self.plot_item.plot([], [], pen=pg.mkPen("#111111", width=3), name="Intensity")
        self._phase = pg.PlotCurveItem([], [], pen=pg.mkPen("#1F77B4", width=2.5, style=QtCore.Qt.PenStyle.DashLine))
        self.phase_vb.addItem(self._phase)

    def _update_views(self) -> None:
        self.phase_vb.setGeometry(self.plot_item.vb.sceneBoundingRect())
        self.phase_vb.linkedViewChanged(self.plot_item.vb, self.phase_vb.XAxis)

    def _on_range_changed(self, *_args) -> None:
        self._update_top_ticks()

    def _update_top_ticks(self) -> None:
        if self.top_transform is None:
            return
        axis = self.plot_item.getAxis("top")
        x_range, _ = self.plot_item.vb.viewRange()
        xmin = float(x_range[0])
        xmax = float(x_range[1])
        if not np.isfinite(xmin) or not np.isfinite(xmax) or abs(xmax - xmin) < 1e-15:
            return
        ticks = np.linspace(xmin, xmax, 6)
        top_ticks = []
        for tick in ticks:
            try:
                mapped = float(self.top_transform(float(tick)))
            except Exception:
                continue
            if not np.isfinite(mapped):
                continue
            top_ticks.append((float(tick), f"{mapped:.4g}"))
        if top_ticks:
            axis.setTicks([top_ticks])

    def clear(self, title: str) -> None:
        self.plot_item.setTitle(title)
        self._intensity.setData([], [])
        self._phase.setData([], [])
        self._update_top_ticks()

    def set_data(self, x: np.ndarray, intensity: np.ndarray, phase: np.ndarray, title: str) -> None:
        x_arr = np.asarray(x, dtype=np.float64)
        self.plot_item.setTitle(title)
        self._intensity.setData(x_arr, np.asarray(intensity, dtype=np.float64))
        self._phase.setData(x_arr, np.asarray(phase, dtype=np.float64))
        QtCore.QTimer.singleShot(0, self._update_top_ticks)


class _TraceImageView:
    def __init__(self, plot_item: pg.PlotItem, bottom_label: str, left_label: str):
        self.plot_item = plot_item
        self.plot_item.getAxis("bottom").enableAutoSIPrefix(False)
        self.plot_item.getAxis("left").enableAutoSIPrefix(False)
        self.plot_item.disableAutoRange()
        self.plot_item.setLabel("bottom", bottom_label)
        self.plot_item.setLabel("left", left_label)
        self.plot_item.showGrid(x=True, y=True, alpha=0.2)
        self.image = pg.ImageItem()
        self.image.setColorMap(_get_trace_colormap())
        self.plot_item.addItem(self.image)
        self.colorbar = pg.ColorBarItem(values=(0.0, 1.0), colorMap=_get_trace_colormap(), interactive=False, width=12)
        self.colorbar.setImageItem(self.image, insert_in=self.plot_item)

    def clear(self, title: str) -> None:
        self.plot_item.setTitle(title)
        self.image.setImage(np.zeros((2, 2), dtype=np.float64), autoLevels=False)
        self.image.setLevels((0.0, 1.0))
        self.colorbar.setLevels(values=(0.0, 1.0))

    def set_data(self, x: np.ndarray, y: np.ndarray, z: np.ndarray, title: str) -> None:
        x_arr = np.asarray(x, dtype=np.float64).ravel()
        y_arr = np.asarray(y, dtype=np.float64).ravel()
        z_arr = np.asarray(z, dtype=np.float64)
        if x_arr.size < 2 or y_arr.size < 2:
            raise ValueError("Trace axes must contain at least two points.")
        self.image.setImage(z_arr, autoLevels=False)
        x0, x1 = float(np.min(x_arr)), float(np.max(x_arr))
        y0, y1 = float(np.min(y_arr)), float(np.max(y_arr))
        rect = QtCore.QRectF(min(x0, x1), min(y0, y1), max(abs(x1 - x0), 1e-12), max(abs(y1 - y0), 1e-12))
        self.image.setRect(rect)
        zmin = float(np.min(z_arr)) if z_arr.size else 0.0
        zmax = float(np.max(z_arr)) if z_arr.size else 1.0
        if zmax - zmin < 1e-12:
            zmax = zmin + 1e-12
        self.image.setLevels((zmin, zmax))
        self.colorbar.setLevels(values=(zmin, zmax))
        self.plot_item.setTitle(title)
        self.plot_item.setRange(
            xRange=(float(np.min(x_arr)), float(np.max(x_arr))),
            yRange=(float(np.min(y_arr)), float(np.max(y_arr))),
            padding=0.0,
        )
        self.plot_item.setLimits(
            xMin=float(np.min(x_arr)),
            xMax=float(np.max(x_arr)),
            yMin=float(np.min(y_arr)),
            yMax=float(np.max(y_arr)),
        )


class _TraceMeshView:
    def __init__(self, plot_item: pg.PlotItem, bottom_label: str, left_label: str):
        self.plot_item = plot_item
        self.plot_item.getAxis("bottom").enableAutoSIPrefix(False)
        self.plot_item.getAxis("left").enableAutoSIPrefix(False)
        self.plot_item.disableAutoRange()
        self.plot_item.setLabel("bottom", bottom_label)
        self.plot_item.setLabel("left", left_label)
        self.plot_item.showGrid(x=True, y=True, alpha=0.2)
        self.mesh = pg.PColorMeshItem()
        self.mesh.setColorMap(_get_trace_colormap())
        self.plot_item.addItem(self.mesh)
        self.colorbar = pg.ColorBarItem(values=(0.0, 1.0), colorMap=_get_trace_colormap(), interactive=False, width=12)
        self.colorbar.setImageItem(self.mesh, insert_in=self.plot_item)

    def clear(self, title: str) -> None:
        self.plot_item.setTitle(title)
        x = np.array([[0.0, 1.0], [0.0, 1.0]], dtype=np.float64)
        y = np.array([[0.0, 0.0], [1.0, 1.0]], dtype=np.float64)
        z = np.array([[0.0]], dtype=np.float64)
        self.mesh.setData(x, y, z, autoLevels=False)
        self.mesh.setLevels((0.0, 1.0))
        self.colorbar.setLevels(values=(0.0, 1.0))

    def set_data(
        self,
        x: np.ndarray,
        y: np.ndarray,
        z: np.ndarray,
        title: str,
        display_y_range: tuple[float, float] | None = None,
    ) -> None:
        x_arr = np.asarray(x, dtype=np.float64).ravel()
        y_arr = np.asarray(y, dtype=np.float64).ravel()
        z_arr = np.asarray(z, dtype=np.float64)
        if z_arr.shape != (x_arr.size, y_arr.size):
            raise ValueError(f"Trace mesh shape mismatch: {z_arr.shape} vs {(x_arr.size, y_arr.size)}.")
        x_edges = _centers_to_edges(x_arr)
        y_edges = _centers_to_edges(y_arr)
        x_mesh, y_mesh = np.meshgrid(x_edges, y_edges, indexing="ij")
        self.mesh.setData(x_mesh, y_mesh, z_arr, autoLevels=False)
        zmin = float(np.min(z_arr)) if z_arr.size else 0.0
        zmax = float(np.max(z_arr)) if z_arr.size else 1.0
        if zmax - zmin < 1e-12:
            zmax = zmin + 1e-12
        self.mesh.setLevels((zmin, zmax))
        self.colorbar.setLevels(values=(zmin, zmax))
        self.plot_item.setTitle(title)
        y_min = float(np.min(y_edges))
        y_max = float(np.max(y_edges))
        if display_y_range is not None:
            y_min = max(y_min, float(display_y_range[0]))
            y_max = min(y_max, float(display_y_range[1]))
        self.plot_item.setRange(
            xRange=(float(np.min(x_edges)), float(np.max(x_edges))),
            yRange=(y_min, y_max),
            padding=0.0,
        )
        self.plot_item.setLimits(
            xMin=float(np.min(x_edges)),
            xMax=float(np.max(x_edges)),
            yMin=float(np.min(y_edges)),
            yMax=float(np.max(y_edges)),
        )


class SimulateFrogGUI(QtWidgets.QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("Simulate FROG GUI")
        self.resize(1280, 820)
        self._initial_size_applied = False
        self._last_dir = _REPO_DIR if os.path.isdir(_REPO_DIR) else os.getcwd()
        self._sim_result: Optional[SimulationResult] = None

        pg.setConfigOptions(antialias=True, background="w", foreground="k")
        self._forward_model = build_forward_model("shg-frog")
        self._build_ui()
        self._set_status("Ready.")

    def showEvent(self, event) -> None:
        super().showEvent(event)
        if self._initial_size_applied:
            return
        self._initial_size_applied = True
        QtCore.QTimer.singleShot(0, self._apply_initial_splitter_sizes)

    def _build_ui(self) -> None:
        root = QtWidgets.QWidget()
        self.setCentralWidget(root)
        vbox = QtWidgets.QVBoxLayout(root)
        vbox.setContentsMargins(4, 4, 4, 4)
        vbox.setSpacing(4)

        top = QtWidgets.QHBoxLayout()
        top.setContentsMargins(0, 0, 0, 0)
        top.setSpacing(4)
        vbox.addLayout(top)

        self.btn_simulate = QtWidgets.QPushButton("Simulate")
        self.btn_simulate.setStyleSheet("background-color: #90EE90;")
        self.btn_save_time = QtWidgets.QPushButton("Save time")
        self.btn_save_frequency = QtWidgets.QPushButton("Save frequency")
        self.btn_save_frg = QtWidgets.QPushButton("Save frg")
        self.btn_save_trace = QtWidgets.QPushButton("Save trace")
        self.btn_save_all = QtWidgets.QPushButton("Save all")
        self.model_combo = QtWidgets.QComboBox()
        self.model_combo.addItems(["SHG", "PG", "THG", "SD"])
        self.chk_trace_log = QtWidgets.QCheckBox("Trace log scale")
        self.trace_log_vmin_edit = QtWidgets.QLineEdit("1e-4")
        self.trace_log_vmin_edit.setMaximumWidth(70)
        top.addWidget(self.btn_simulate)
        top.addWidget(self.btn_save_time)
        top.addWidget(self.btn_save_frequency)
        top.addWidget(self.btn_save_frg)
        top.addWidget(self.btn_save_trace)
        top.addWidget(self.btn_save_all)
        top.addWidget(QtWidgets.QLabel("Model"))
        top.addWidget(self.model_combo)
        top.addWidget(self.chk_trace_log)
        top.addWidget(QtWidgets.QLabel("vmin"))
        top.addWidget(self.trace_log_vmin_edit)
        top.addStretch(1)

        splitter = QtWidgets.QSplitter(QtCore.Qt.Horizontal)
        splitter.setHandleWidth(2)
        splitter.setChildrenCollapsible(False)
        vbox.addWidget(splitter, 1)
        self._main_splitter = splitter

        ctrl_scroll = QtWidgets.QScrollArea()
        ctrl_scroll.setWidgetResizable(True)
        ctrl_scroll.setFrameShape(QtWidgets.QFrame.Shape.NoFrame)
        ctrl_scroll.setFixedWidth(250)
        splitter.addWidget(ctrl_scroll)
        self._ctrl_scroll = ctrl_scroll

        ctrl_container = QtWidgets.QWidget()
        ctrl_scroll.setWidget(ctrl_container)
        ctrl_layout = QtWidgets.QVBoxLayout(ctrl_container)
        ctrl_layout.setContentsMargins(0, 0, 0, 0)
        ctrl_layout.setSpacing(6)
        ctrl_layout.setAlignment(QtCore.Qt.AlignmentFlag.AlignTop)

        input_group = QtWidgets.QGroupBox("Input")
        input_form = QtWidgets.QFormLayout(input_group)
        input_form.setContentsMargins(8, 8, 8, 8)
        input_form.setSpacing(4)

        self.domain_combo = QtWidgets.QComboBox()
        self.domain_combo.addItems(["Time domain", "Frequency domain"])
        self.source_combo = QtWidgets.QComboBox()
        self.source_combo.addItems(["Analytic", "From file"])
        input_form.addRow("Domain", self.domain_combo)
        input_form.addRow("Source", self.source_combo)

        self.input_stack = QtWidgets.QStackedWidget()
        input_form.addRow(self.input_stack)
        ctrl_layout.addWidget(input_group)

        self._build_input_pages()

        sim_group = QtWidgets.QGroupBox("Simulation")
        sim_form = QtWidgets.QVBoxLayout(sim_group)
        sim_form.setContentsMargins(8, 8, 8, 8)
        sim_form.setSpacing(4)
        self.gdd_edit = QtWidgets.QLineEdit("0")
        self.tod_edit = QtWidgets.QLineEdit("0")
        self.noise_edit = QtWidgets.QLineEdit("0")
        self.delay_range_edit = QtWidgets.QLineEdit("250")
        self.n_spin = QtWidgets.QSpinBox()
        self.n_spin.setRange(32, 4096)
        self.n_spin.setSingleStep(32)
        self.n_spin.setValue(256)
        self._set_compact_field_width(self.gdd_edit)
        self._set_compact_field_width(self.tod_edit)
        self._set_compact_field_width(self.noise_edit)
        self._set_compact_field_width(self.delay_range_edit)
        self.n_spin.setMaximumWidth(120)
        sim_form.addWidget(self._build_compact_row("GDD (fs^2)", self.gdd_edit))
        sim_form.addWidget(self._build_compact_row("TOD (fs^3)", self.tod_edit))
        sim_form.addWidget(self._build_compact_row("Noise (%)", self.noise_edit))
        sim_form.addWidget(self._build_compact_row("delay_range (fs)", self.delay_range_edit))
        sim_form.addWidget(self._build_compact_row("N", self.n_spin))
        sim_form.addStretch(1)
        ctrl_layout.addWidget(sim_group)

        note_group = QtWidgets.QGroupBox("Notes")
        note_layout = QtWidgets.QVBoxLayout(note_group)
        note_layout.setContentsMargins(8, 8, 8, 8)
        note_layout.setSpacing(4)
        note = QtWidgets.QLabel(
            "Time file: \n(time_fs, intensity, [phase_rad])\n"
            "Spectrum file: \n(wavelength_um, intensity, [phase_rad])\n"
            "If the phase column is missing, \nphase = 0.\n"
            "Pulse duration and bandwidth \nare treated as intensity FWHM."
        )
        note.setStyleSheet("color:#444;")
        note.setTextInteractionFlags(QtCore.Qt.TextInteractionFlag.TextSelectableByMouse)
        note_layout.addWidget(note)
        note_layout.addStretch(1)
        ctrl_layout.addWidget(note_group)
        ctrl_layout.addStretch(1)

        self.graphics = pg.GraphicsLayoutWidget()
        self.graphics.setBackground("w")
        splitter.addWidget(self.graphics)
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)

        self.plot_time = self.graphics.addPlot(row=0, col=0, title="Time profile")
        self.plot_spectrum = self.graphics.addPlot(row=0, col=1, title="Spectrum")
        self.plot_trace_freq = self.graphics.addPlot(row=1, col=0, title="FROG trace (frequency)")
        self.plot_trace_wave = self.graphics.addPlot(row=1, col=1, title="FROG trace (wavelength)")

        self.time_view = _DualAxisCurve(self.plot_time, "Time (fs)", "Intensity", "Phase (rad)")
        self.spectrum_view = _DualAxisCurve(
            self.plot_spectrum,
            "Frequency (PHz)",
            "Intensity",
            "Phase (rad)",
            top_label="Wavelength (um)",
            top_transform=lambda x: float(_frequency_phz_to_wavelength_um(x)) if x > 1e-12 else float("nan"),
        )
        self.trace_freq_view = _TraceImageView(self.plot_trace_freq, "Delay (fs)", "SHG Frequency (PHz)")
        self.trace_wave_view = _TraceMeshView(self.plot_trace_wave, "Delay (fs)", "SHG Wavelength (um)")

        self.domain_combo.currentTextChanged.connect(self._update_input_page)
        self.source_combo.currentTextChanged.connect(self._update_input_page)
        self.btn_simulate.clicked.connect(self._on_simulate_clicked)
        self.btn_save_time.clicked.connect(lambda: self._save_selected_outputs(("time",)))
        self.btn_save_frequency.clicked.connect(lambda: self._save_selected_outputs(("frequency",)))
        self.btn_save_frg.clicked.connect(lambda: self._save_selected_outputs(("frg",)))
        self.btn_save_trace.clicked.connect(lambda: self._save_selected_outputs(("trace",)))
        self.btn_save_all.clicked.connect(lambda: self._save_selected_outputs(("time", "frequency", "frg", "trace")))
        self.chk_trace_log.toggled.connect(self._on_trace_log_toggled)
        self.trace_log_vmin_edit.editingFinished.connect(self._on_trace_log_vmin_changed)
        self._update_input_page()
        self._clear_plots()

    def _apply_initial_splitter_sizes(self) -> None:
        splitter = getattr(self, "_main_splitter", None)
        ctrl_scroll = getattr(self, "_ctrl_scroll", None)
        if splitter is None or ctrl_scroll is None:
            return
        left_width = int(ctrl_scroll.width()) if ctrl_scroll.width() > 0 else 250
        total_width = int(splitter.size().width()) if splitter.size().width() > 0 else max(self.width() - 8, left_width + 200)
        splitter.setSizes([left_width, max(200, total_width - left_width)])

    def _build_input_pages(self) -> None:
        self.page_time_analytic = QtWidgets.QWidget()
        form = QtWidgets.QVBoxLayout(self.page_time_analytic)
        form.setContentsMargins(0, 0, 0, 0)
        form.setSpacing(4)
        self.time_center_wl_edit = QtWidgets.QLineEdit("0.8")
        self.time_duration_edit = QtWidgets.QLineEdit("30")
        self.time_shape_combo = QtWidgets.QComboBox()
        self.time_shape_combo.addItems(["Gaussian", "Sech"])
        self._set_compact_field_width(self.time_center_wl_edit)
        self._set_compact_field_width(self.time_duration_edit)
        self._set_compact_field_width(self.time_shape_combo)
        form.addWidget(self._build_compact_row("Center wavelength (um)", self.time_center_wl_edit))
        form.addWidget(self._build_compact_row("Pulse duration (fs)", self.time_duration_edit))
        form.addWidget(self._build_compact_row("Pulse shape", self.time_shape_combo))
        form.addStretch(1)
        self.input_stack.addWidget(self.page_time_analytic)

        self.page_time_file = QtWidgets.QWidget()
        form = QtWidgets.QVBoxLayout(self.page_time_file)
        form.setContentsMargins(0, 0, 0, 0)
        form.setSpacing(4)
        self.time_file_center_wl_edit = QtWidgets.QLineEdit("0.8")
        self._set_compact_field_width(self.time_file_center_wl_edit)
        self.time_file_path = QtWidgets.QLineEdit("")
        self.time_file_path.setPlaceholderText("time, intensity, [phase]")
        btn = QtWidgets.QPushButton("Browse")
        btn.setMaximumWidth(110)
        btn.clicked.connect(lambda: self._browse_into(self.time_file_path, "Open Time-Domain Field File"))
        form.addWidget(self._build_compact_row("Center wavelength (um)", self.time_file_center_wl_edit))
        form.addWidget(self._build_file_row("File", self.time_file_path, btn))
        form.addStretch(1)
        self.input_stack.addWidget(self.page_time_file)

        self.page_freq_analytic = QtWidgets.QWidget()
        form = QtWidgets.QVBoxLayout(self.page_freq_analytic)
        form.setContentsMargins(0, 0, 0, 0)
        form.setSpacing(4)
        self.freq_center_wl_edit = QtWidgets.QLineEdit("0.8")
        self.freq_bandwidth_edit = QtWidgets.QLineEdit("0.03")
        self.freq_shape_combo = QtWidgets.QComboBox()
        self.freq_shape_combo.addItems(["Gaussian"])
        self._set_compact_field_width(self.freq_center_wl_edit)
        self._set_compact_field_width(self.freq_bandwidth_edit)
        self._set_compact_field_width(self.freq_shape_combo)
        form.addWidget(self._build_compact_row("Center wavelength (um)", self.freq_center_wl_edit))
        form.addWidget(self._build_compact_row("Bandwidth (um)", self.freq_bandwidth_edit))
        form.addWidget(self._build_compact_row("Pulse shape", self.freq_shape_combo))
        form.addStretch(1)
        self.input_stack.addWidget(self.page_freq_analytic)

        self.page_freq_file = QtWidgets.QWidget()
        form = QtWidgets.QVBoxLayout(self.page_freq_file)
        form.setContentsMargins(0, 0, 0, 0)
        form.setSpacing(4)
        self.freq_file_path = QtWidgets.QLineEdit("")
        self.freq_file_path.setPlaceholderText("wavelength_um, intensity, [phase]")
        btn = QtWidgets.QPushButton("Browse")
        btn.setMaximumWidth(110)
        btn.clicked.connect(lambda: self._browse_into(self.freq_file_path, "Open Spectrum File"))
        form.addWidget(self._build_file_row("File", self.freq_file_path, btn))
        form.addStretch(1)
        self.input_stack.addWidget(self.page_freq_file)

    @staticmethod
    def _wrap_layout(layout: QtWidgets.QLayout) -> QtWidgets.QWidget:
        widget = QtWidgets.QWidget()
        widget.setLayout(layout)
        return widget

    @staticmethod
    def _set_compact_field_width(widget: QtWidgets.QWidget) -> None:
        if isinstance(widget, QtWidgets.QLineEdit):
            widget.setMaximumWidth(140)
        elif isinstance(widget, QtWidgets.QComboBox):
            widget.setMaximumWidth(180)

    def _build_compact_row(self, label_text: str, field: QtWidgets.QWidget) -> QtWidgets.QWidget:
        row = QtWidgets.QWidget()
        layout = QtWidgets.QHBoxLayout(row)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(8)
        label = QtWidgets.QLabel(label_text)
        label.setSizePolicy(QtWidgets.QSizePolicy.Policy.Maximum, QtWidgets.QSizePolicy.Policy.Preferred)
        layout.addWidget(label)
        layout.addWidget(field, 0, QtCore.Qt.AlignmentFlag.AlignLeft)
        layout.addStretch(1)
        return row

    def _build_file_row(
        self,
        label_text: str,
        line_edit: QtWidgets.QLineEdit,
        button: QtWidgets.QPushButton,
    ) -> QtWidgets.QWidget:
        row = QtWidgets.QWidget()
        layout = QtWidgets.QHBoxLayout(row)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(8)
        label = QtWidgets.QLabel(label_text)
        label.setSizePolicy(QtWidgets.QSizePolicy.Policy.Maximum, QtWidgets.QSizePolicy.Policy.Preferred)
        line_edit.setMaximumWidth(220)
        layout.addWidget(label)
        layout.addWidget(line_edit, 0, QtCore.Qt.AlignmentFlag.AlignLeft)
        layout.addWidget(button, 0, QtCore.Qt.AlignmentFlag.AlignLeft)
        layout.addStretch(1)
        return row

    def _browse_into(self, line_edit: QtWidgets.QLineEdit, title: str) -> None:
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self,
            title,
            self._last_dir,
            "Data files (*.txt *.dat *.csv);;All files (*)",
        )
        if not path:
            return
        self._last_dir = os.path.dirname(path) or self._last_dir
        line_edit.setText(path)

    def _update_input_page(self) -> None:
        key = (self.domain_combo.currentText(), self.source_combo.currentText())
        mapping = {
            ("Time domain", "Analytic"): 0,
            ("Time domain", "From file"): 1,
            ("Frequency domain", "Analytic"): 2,
            ("Frequency domain", "From file"): 3,
        }
        self.input_stack.setCurrentIndex(mapping[key])

    def _clear_plots(self) -> None:
        self.time_view.clear("Time profile")
        self.spectrum_view.clear("Spectrum")
        self.trace_freq_view.clear("FROG trace (frequency)")
        self.trace_wave_view.clear("FROG trace (wavelength)")

    def _trace_display_array(self, z: np.ndarray) -> np.ndarray:
        z_arr = np.asarray(z, dtype=np.float64)
        if not self.chk_trace_log.isChecked():
            return z_arr
        return np.log10(np.clip(z_arr, self._trace_log_vmin(), None))

    def _trace_log_vmin(self) -> float:
        text = self.trace_log_vmin_edit.text().strip()
        default = 1e-4
        try:
            value = float(text)
        except Exception:
            value = default
        if not np.isfinite(value) or value <= 0:
            value = default
        return float(value)

    @QtCore.Slot(bool)
    def _on_trace_log_toggled(self, _checked: bool) -> None:
        if self._sim_result is not None:
            self._draw_simulation(self._sim_result)

    @QtCore.Slot()
    def _on_trace_log_vmin_changed(self) -> None:
        value = self._trace_log_vmin()
        self.trace_log_vmin_edit.setText(f"{value:.4g}")
        if self.chk_trace_log.isChecked() and self._sim_result is not None:
            self._draw_simulation(self._sim_result)

    def _set_status(self, text: str, timeout_ms: int = 0) -> None:
        self.statusBar().showMessage(str(text), int(timeout_ms))

    def _require_simulation_result(self) -> SimulationResult:
        if self._sim_result is None:
            raise RuntimeError("Please run Simulate first.")
        return self._sim_result

    def _default_save_prefix(self) -> str:
        return os.path.join(self._last_dir, "simulated_frog")

    def _prompt_save_prefix(self) -> Optional[str]:
        default_prefix = self._default_save_prefix()
        path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self,
            "Save Prefix",
            default_prefix,
            "All files (*)",
        )
        if not path:
            return None
        prefix = str(path).strip()
        if not prefix:
            QtWidgets.QMessageBox.warning(self, "Invalid Prefix", "Prefix cannot be empty.")
            return None
        lower = prefix.lower()
        known_suffixes = (
            "_temporal.txt",
            "_frequency.txt",
            "_frog_trace.txt",
            ".frg",
        )
        for suffix in known_suffixes:
            if lower.endswith(suffix.lower()):
                prefix = prefix[: -len(suffix)]
                lower = prefix.lower()
                break
        binned_tag = lower.rfind("_binned")
        if binned_tag != -1:
            prefix = prefix[:binned_tag]
            lower = prefix.lower()
        if lower.endswith(".txt") or lower.endswith(".dat") or lower.endswith(".csv"):
            prefix = prefix[:-4]
        save_dir = os.path.dirname(os.path.abspath(prefix))
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
            self._last_dir = save_dir
        return prefix

    @staticmethod
    def _save_table(path: str, data: np.ndarray, header: str) -> None:
        np.savetxt(path, np.asarray(data, dtype=np.float64), fmt="%.12e", delimiter="\t", header=header, comments="")

    @staticmethod
    def _save_frg_matrix(path: str, delay: np.ndarray, frequency: np.ndarray, trace: np.ndarray) -> None:
        delay_arr = np.asarray(delay, dtype=np.float64).ravel()
        freq_arr = np.asarray(frequency, dtype=np.float64).ravel()
        z = np.asarray(trace, dtype=np.float64)
        if delay_arr.size < 2 or freq_arr.size < 2:
            raise ValueError("FRG export requires at least two delay/frequency points.")
        if z.shape != (delay_arr.size, freq_arr.size):
            raise ValueError(f"FRG trace shape mismatch: {z.shape} vs {(delay_arr.size, freq_arr.size)}.")
        width = int(delay_arr.size)
        height = int(freq_arr.size)
        dt = float(np.median(np.diff(delay_arr)))
        dv = float(np.median(np.diff(freq_arr)))
        v0 = float(freq_arr[height // 2])
        with open(path, "w", encoding="utf-8") as f:
            f.write(f"{width}\t{height}\t{dt:.12e}\t{dv:.12e}\t{v0:.12e}\n")
            for j in range(height):
                row = "\t".join(f"{float(z[i, j]):.12e}" for i in range(width))
                f.write(row + "\n")

    @staticmethod
    def _save_wavelength_trace(path: str, delay: np.ndarray, wavelength_um: np.ndarray, trace: np.ndarray) -> None:
        delay_arr = np.asarray(delay, dtype=np.float64).ravel()
        wave_nm = 1000.0 * np.asarray(wavelength_um, dtype=np.float64).ravel()
        z = np.asarray(trace, dtype=np.float64)
        if z.shape != (delay_arr.size, wave_nm.size):
            raise ValueError(f"Trace export shape mismatch: {z.shape} vs {(delay_arr.size, wave_nm.size)}.")
        with open(path, "w", encoding="utf-8") as f:
            f.write("\t".join(f"{float(v):.12e}" for v in delay_arr) + "\n")
            f.write("\t".join(f"{float(v):.12e}" for v in wave_nm) + "\n")
            for i in range(delay_arr.size):
                f.write("\t".join(f"{float(z[i, j]):.12e}" for j in range(wave_nm.size)) + "\n")

    def _build_time_export(self, result: SimulationResult) -> np.ndarray:
        time_intensity = _safe_normalize(np.abs(result.field_t) ** 2)
        time_phase = _phase_for_display(result.field_t, time_intensity)
        return np.column_stack(
            (
                np.asarray(result.time_fs, dtype=np.float64),
                time_intensity,
                np.asarray(time_phase, dtype=np.float64),
            )
        )

    def _build_frequency_export(self, result: SimulationResult) -> np.ndarray:
        freq_abs = np.asarray(result.freq_abs_phz, dtype=np.float64)
        wavelength_um = np.full_like(freq_abs, np.nan)
        positive = freq_abs > 1e-12
        wavelength_um[positive] = _frequency_phz_to_wavelength_um(freq_abs[positive])
        spec_intensity = _safe_normalize(np.abs(result.field_f) ** 2)
        spec_phase = _phase_for_display(result.field_f, spec_intensity)
        return np.column_stack(
            (
                freq_abs,
                spec_intensity,
                np.asarray(spec_phase, dtype=np.float64),
                wavelength_um,
            )
        )

    @staticmethod
    def _build_trace_export(x: np.ndarray, y: np.ndarray, z: np.ndarray) -> np.ndarray:
        xx, yy = np.meshgrid(np.asarray(x, dtype=np.float64), np.asarray(y, dtype=np.float64), indexing="ij")
        return np.column_stack((xx.ravel(), yy.ravel(), np.asarray(z, dtype=np.float64).ravel()))

    def _save_selected_outputs(self, kinds: tuple[str, ...]) -> None:
        try:
            result = self._require_simulation_result()
        except Exception as exc:
            QtWidgets.QMessageBox.warning(self, "No Simulation", str(exc))
            return

        prefix = self._prompt_save_prefix()
        if not prefix:
            return

        trace_freq_abs = _trace_frequency_multiplier(result.geometry) * float(result.center_freq_phz) + np.asarray(result.freq_rel_phz, dtype=np.float64)
        wave_axis, wave_trace, _ = _map_trace_to_wavelength(trace_freq_abs, result.trace)
        written: list[str] = []
        try:
            for kind in kinds:
                if kind == "time":
                    path = f"{prefix}_temporal.txt"
                    self._save_table(
                        path,
                        self._build_time_export(result),
                        "time_fs\tintensity_norm\tphase_rad",
                    )
                elif kind == "frequency":
                    path = f"{prefix}_frequency.txt"
                    self._save_table(
                        path,
                        self._build_frequency_export(result),
                        "frequency_PHz\tintensity_norm\tphase_rad\twavelength_um",
                    )
                elif kind == "frg":
                    path = f"{prefix}_binned{int(result.trace.shape[0])}.frg"
                    self._save_frg_matrix(
                        path,
                        result.trace_delay_fs,
                        trace_freq_abs,
                        result.trace,
                    )
                elif kind == "trace":
                    path = f"{prefix}_frog_trace.txt"
                    self._save_wavelength_trace(
                        path,
                        result.trace_delay_fs,
                        wave_axis,
                        wave_trace,
                    )
                else:
                    continue
                written.append(path)
        except Exception as exc:
            self._set_status(f"Save error: {exc}", timeout_ms=12000)
            QtWidgets.QMessageBox.critical(self, "Save Error", str(exc))
            return

        if written:
            self._set_status(f"Saved {len(written)} file(s) with prefix: {prefix}", timeout_ms=10000)
            QtWidgets.QMessageBox.information(self, "Saved", "\n".join(written))

    def _parse_float(self, widget: QtWidgets.QLineEdit, label: str) -> float:
        text = widget.text().strip()
        if not text:
            raise ValueError(f"{label} cannot be empty.")
        return float(text)

    def _build_simulation(self) -> SimulationResult:
        delay_range_fs = self._parse_float(self.delay_range_edit, "delay_range")
        if delay_range_fs <= 0:
            raise ValueError("delay_range must be positive.")
        n = int(self.n_spin.value())
        time_axis = _build_centered_time_axis(delay_range_fs, n)
        freq_rel = _build_frequency_axis_from_time(time_axis)

        domain = self.domain_combo.currentText()
        source = self.source_combo.currentText()

        if domain == "Time domain":
            if source == "Analytic":
                center_wl_um = self._parse_float(self.time_center_wl_edit, "center wavelength")
                center_freq = float(_wavelength_um_to_frequency_phz(center_wl_um))
                duration_fs = self._parse_float(self.time_duration_edit, "pulse duration")
                seed_field_t = _build_time_field_from_params(time_axis, duration_fs, self.time_shape_combo.currentText())
            else:
                center_wl_um = self._parse_float(self.time_file_center_wl_edit, "center wavelength")
                center_freq = float(_wavelength_um_to_frequency_phz(center_wl_um))
                path = self.time_file_path.text().strip()
                if not path:
                    raise ValueError("Time-domain file path cannot be empty.")
                seed_field_t = _build_time_field_from_file(time_axis, path)
            freq_rel, seed_field_f = get_Ef_from_Et(time_axis, seed_field_t)
        else:
            if source == "Analytic":
                if self.freq_shape_combo.currentText().strip().lower() != "gaussian":
                    raise ValueError("Only Gaussian analytic spectrum is implemented.")
                center_wl_um = self._parse_float(self.freq_center_wl_edit, "center wavelength")
                bandwidth_um = self._parse_float(self.freq_bandwidth_edit, "bandwidth")
                center_freq, seed_field_f = _build_spectrum_from_params(freq_rel, center_wl_um, bandwidth_um)
            else:
                path = self.freq_file_path.text().strip()
                if not path:
                    raise ValueError("Spectrum file path cannot be empty.")
                center_freq, seed_field_f = _build_spectrum_from_file(freq_rel, path)

        gdd_fs2 = self._parse_float(self.gdd_edit, "GDD")
        tod_fs3 = self._parse_float(self.tod_edit, "TOD")
        noise_percent = self._parse_float(self.noise_edit, "Noise")
        if noise_percent < 0:
            raise ValueError("Noise must be >= 0.")

        field_f = _apply_chirp(freq_rel, seed_field_f, gdd_fs2, tod_fs3)
        time_fs, field_t = get_Et_from_Ef(freq_rel, field_f)
        field_t = _normalize_field(field_t)
        freq_rel, field_f = get_Ef_from_Et(time_fs, field_t)
        field_f = _normalize_field(field_f)

        geometry = str(getattr(self._forward_model, "geometry", "shg-frog"))
        trace_freq_phz = _trace_frequency_multiplier(geometry) * float(center_freq) + np.asarray(freq_rel, dtype=np.float64)
        grid = FrogGrid(delay=np.asarray(time_fs, dtype=np.float64), frequency=trace_freq_phz)
        trace = self._forward_model.simulate_trace(field_t, grid)
        trace = _safe_normalize(trace)
        if noise_percent > 0:
            rng = np.random.default_rng()
            noise_sigma = float(noise_percent) / 100.0
            trace = np.clip(trace + noise_sigma * rng.normal(size=trace.shape), 0.0, None)
            trace = _safe_normalize(trace)

        trace_wave_um, trace_wave, note = _map_trace_to_wavelength(trace_freq_phz, trace)
        freq_abs = float(center_freq) + np.asarray(freq_rel, dtype=np.float64)
        return SimulationResult(
            time_fs=np.asarray(time_fs, dtype=np.float64),
            field_t=np.asarray(field_t, dtype=np.complex128),
            freq_rel_phz=np.asarray(freq_rel, dtype=np.float64),
            field_f=np.asarray(field_f, dtype=np.complex128),
            freq_abs_phz=freq_abs,
            center_freq_phz=float(center_freq),
            geometry=geometry,
            trace_delay_fs=np.asarray(time_fs, dtype=np.float64),
            trace_freq_phz=trace_freq_phz,
            trace=np.asarray(trace, dtype=np.float64),
            trace_wave_um=np.asarray(trace_wave_um, dtype=np.float64),
            trace_wave=np.asarray(trace_wave, dtype=np.float64),
            wavelength_note=note,
        )

    def _draw_simulation(self, result: SimulationResult) -> None:
        trace_freq_abs = _trace_frequency_multiplier(result.geometry) * float(result.center_freq_phz) + np.asarray(result.freq_rel_phz, dtype=np.float64)
        time_intensity = _safe_normalize(np.abs(result.field_t) ** 2)
        time_phase = _phase_for_display(result.field_t, time_intensity)
        time_fwhm = _estimate_fwhm(result.time_fs, time_intensity)
        time_title = "Time profile"
        if np.isfinite(time_fwhm):
            time_title = f"Time profile (FWHM={time_fwhm:.4g} fs)"
        self.time_view.set_data(result.time_fs, time_intensity, time_phase, time_title)

        spec_intensity = _safe_normalize(np.abs(result.field_f) ** 2)
        spec_phase = _phase_for_display(result.field_f, spec_intensity)
        positive_freq = result.freq_abs_phz > 1e-12
        spec_fwhm = _estimate_fwhm(result.freq_abs_phz[positive_freq], spec_intensity[positive_freq]) if np.any(positive_freq) else float("nan")
        spec_fwhm_nm = float("nan")
        if np.any(positive_freq):
            wavelength_nm = 1000.0 * np.asarray(_frequency_phz_to_wavelength_um(result.freq_abs_phz[positive_freq]), dtype=np.float64)
            order = np.argsort(wavelength_nm)
            spec_intensity_lambda = spec_intensity[positive_freq] * _abs_df_dlambda_phz_per_um(wavelength_nm / 1000.0)
            spec_fwhm_nm = _estimate_fwhm(wavelength_nm[order], _safe_normalize(spec_intensity_lambda[order]))
        spec_title = "Spectrum"
        if np.isfinite(spec_fwhm) and np.isfinite(spec_fwhm_nm):
            spec_title = f"Spectrum (FWHM={spec_fwhm:.4g} PHz, {spec_fwhm_nm:.4g} nm)"
        elif np.isfinite(spec_fwhm):
            spec_title = f"Spectrum (FWHM={spec_fwhm:.4g} PHz)"
        elif np.isfinite(spec_fwhm_nm):
            spec_title = f"Spectrum (FWHM={spec_fwhm_nm:.4g} nm)"
        self.spectrum_view.set_data(result.freq_abs_phz, spec_intensity, spec_phase, spec_title)

        self.trace_freq_view.set_data(
            result.trace_delay_fs,
            trace_freq_abs,
            self._trace_display_array(result.trace),
            f"FROG trace: delay vs {_trace_label_prefix(result.geometry)} frequency"
            + (f" (log10, vmin={self._trace_log_vmin():.4g})" if self.chk_trace_log.isChecked() else ""),
        )
        wave_axis, wave_trace, _ = _map_trace_to_wavelength(trace_freq_abs, result.trace)
        wave_marginal = np.max(wave_trace, axis=0) if wave_trace.size else np.asarray([], dtype=np.float64)
        display_y_range = None
        if wave_axis.size >= 2 and wave_marginal.size == wave_axis.size and float(np.max(wave_marginal)) > 0:
            mask = wave_marginal >= 1e-3 * float(np.max(wave_marginal))
            if np.count_nonzero(mask) >= 2:
                y0 = float(np.min(wave_axis[mask]))
                y1 = float(np.max(wave_axis[mask]))
                pad = 0.08 * max(y1 - y0, 1e-12)
                display_y_range = (max(float(np.min(wave_axis)), y0 - pad), min(float(np.max(wave_axis)), y1 + pad))
        self.trace_wave_view.set_data(
            result.trace_delay_fs,
            wave_axis,
            self._trace_display_array(wave_trace),
            f"FROG trace: delay vs {_trace_label_prefix(result.geometry)} wavelength"
            + (f" (log10, vmin={self._trace_log_vmin():.4g})" if self.chk_trace_log.isChecked() else ""),
            display_y_range=display_y_range,
        )

    @QtCore.Slot()
    def _on_simulate_clicked(self) -> None:
        try:
            geometry = f"{self.model_combo.currentText().strip().lower()}-frog"
            self._forward_model = build_forward_model(geometry)
            self._sim_result = self._build_simulation()
            self._draw_simulation(self._sim_result)
            time_intensity = _safe_normalize(np.abs(self._sim_result.field_t) ** 2)
            spec_intensity = _safe_normalize(np.abs(self._sim_result.field_f) ** 2)
            pulse_fwhm = _estimate_fwhm(self._sim_result.time_fs, time_intensity)
            spec_fwhm = _estimate_fwhm(self._sim_result.freq_abs_phz[self._sim_result.freq_abs_phz > 0], spec_intensity[self._sim_result.freq_abs_phz > 0])
            note = f" | {self._sim_result.wavelength_note}" if self._sim_result.wavelength_note else ""
            self._set_status(
                f"Simulated {geometry}. Pulse FWHM={pulse_fwhm:.4g} fs, spectrum FWHM={spec_fwhm:.4g} PHz{note}",
                timeout_ms=12000,
            )
        except Exception as exc:
            self._set_status(f"Simulation error: {exc}", timeout_ms=15000)
            QtWidgets.QMessageBox.critical(self, "Simulation Error", str(exc))


    def export_convert_payload(self) -> Optional[dict]:
        """Build a wavelength-domain payload dict for the Convert GUI.

        Returns a dict with keys ``delay`` (fs), ``wavelength`` (nm),
        ``frog_trace`` (2-D, [delay x wavelength]) — the same format
        accepted by ``FROG.__init__`` via dict input.
        """
        if self._sim_result is None:
            return None
        r = self._sim_result
        trace_freq_abs = (
            _trace_frequency_multiplier(r.geometry) * float(r.center_freq_phz)
            + np.asarray(r.freq_rel_phz, dtype=np.float64)
        )
        wave_um, wave_trace, _ = _map_trace_to_wavelength(trace_freq_abs, r.trace)
        delay_fs = np.asarray(r.trace_delay_fs, dtype=np.float64).ravel()
        wavelength_nm = 1000.0 * np.asarray(wave_um, dtype=np.float64).ravel()
        if delay_fs.size < 2 or wavelength_nm.size < 2 or wave_trace.size == 0:
            return None
        return {
            "delay": delay_fs.copy(),
            "wavelength": wavelength_nm.copy(),
            "frog_trace": np.asarray(wave_trace, dtype=np.float64).copy(),
            "geometry": r.geometry,
            "source": "simulate_gui",
        }


def main() -> None:
    app = QtWidgets.QApplication(sys.argv)
    win = SimulateFrogGUI()
    win.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()

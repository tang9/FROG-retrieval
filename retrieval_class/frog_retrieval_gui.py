"""PySide6 + pyqtgraph GUI for SHG/PG/THG/SD FROG pulse retrieval."""

from __future__ import annotations

import os
import sys
import threading
import traceback
from dataclasses import dataclass
from typing import Optional

import numpy as np
import pyqtgraph as pg
from PySide6 import QtCore, QtWidgets

from .api import retrieve_pulse
from .types import RetrievalConfig, RetrievalResult
from .common import (
    custom_cmap,
    find_peak_fwhm,
    fit_peak,
    get_Ef_from_Et,
    get_Et_from_Ef,
)

_TRACE_COLORMAP = None
_TRACE_DIFF_COLORMAP = pg.colormap.get("bwr", source="matplotlib")
SPEED_LIGHT = 299792458.0  # m/s

@dataclass(slots=True)
class FrgData:
    """Parsed FRG container used by the GUI."""

    path: str
    delay: np.ndarray
    frequency: np.ndarray
    frequency_center: float
    trace: np.ndarray
    geometry: str = "shg-frog"


def _center_bin_value(axis: np.ndarray) -> float:
    arr = np.asarray(axis, dtype=np.float64).ravel()
    if arr.size == 0:
        return 0.0
    return float(arr[arr.size // 2])


def _normalize_geometry_name(geometry: str) -> str:
    geom = str(geometry).strip().lower().replace("_", "-")
    aliases = {
        "shg": "shg-frog",
        "pg": "pg-frog",
        "thg": "thg-frog",
        "sd": "sd-frog",
    }
    return aliases.get(geom, geom)


def _trace_frequency_multiplier(geometry: str) -> float:
    geom = _normalize_geometry_name(geometry)
    if geom == "shg-frog":
        return 2.0
    if geom == "pg-frog":
        return 1.0
    if geom == "thg-frog":
        return 3.0
    if geom == "sd-frog":
        return 1.0
    raise ValueError(f"Unsupported geometry '{geometry}'.")


def _field_center_frequency(trace_center: float, geometry: str) -> float:
    return float(trace_center) / _trace_frequency_multiplier(geometry)


def _parse_text_number(token: str) -> float:
    text = str(token).strip()
    if not text:
        raise ValueError("Empty numeric token.")
    # Accept locale-style decimal commas in imported trace files.
    if "," in text and "." not in text:
        text = text.replace(",", ".")
    return float(text)


def _parse_number_line(line: str) -> list[float]:
    return [_parse_text_number(part) for part in str(line).split()]


def _read_frg_file(path: str) -> FrgData:
    """Load a `.frg` or `.dat` FROG trace into the GUI container."""
    lower_path = str(path).strip().lower()
    if lower_path.endswith(".dat"):
        with open(path, "r", encoding="utf-8") as f:
            lines = [ln.strip() for ln in f if ln.strip()]
        if len(lines) < 3:
            raise ValueError("Invalid .dat file: expected header, limits, and data.")

        header = _parse_number_line(lines[0])
        if len(header) < 2:
            raise ValueError("Invalid .dat header. Expected width and height.")

        width = int(round(header[0]))
        height = int(round(header[1]))
        if width <= 1 or height <= 1:
            raise ValueError(f"Invalid dimensions in .dat header: width={width}, height={height}")
        if width != height:
            raise ValueError(f"Retrieval currently requires square traces. Got {width}x{height}.")

        values: list[float] = []
        for line in lines[1:]:
            values.extend(_parse_number_line(line))

        required = 2 + height + width + width * height
        if len(values) < required:
            raise ValueError(
                f".dat data too short: expected at least {required} values after the header, got {len(values)}."
            )

        offset = 2  # z_min, z_max
        wavelength_nm = np.asarray(values[offset : offset + height], dtype=np.float64)
        offset += height
        delay = np.asarray(values[offset : offset + width], dtype=np.float64)
        offset += width
        trace = np.asarray(values[offset : offset + width * height], dtype=np.float64).reshape((width, height))

        wavelength_nm = np.clip(wavelength_nm, 1e-12, None)
        frequency = SPEED_LIGHT / (wavelength_nm * 1e-9) / 1e15
        trace = np.nan_to_num(trace, nan=0.0, posinf=0.0, neginf=0.0)
        trace = np.clip(trace, 0.0, None)
        max_trace = float(np.max(trace))
        if max_trace > 1e-15:
            trace /= max_trace

        return FrgData(
            path=path,
            delay=delay,
            frequency=frequency,
            frequency_center=_center_bin_value(frequency),
            trace=trace,
        )

    # Load a .frg file using the formats emitted by frog_convert.py.
    with open(path, "r", encoding="utf-8") as f:
        lines = [ln.strip() for ln in f if ln.strip()]
    if not lines:
        raise ValueError("Empty .frg file.")

    header = _parse_number_line(lines[0])
    if len(header) < 5:
        raise ValueError("Invalid .frg header. Expected at least 5 numbers.")

    width = int(round(header[0]))
    height = int(round(header[1]))
    if width <= 1 or height <= 1:
        raise ValueError(f"Invalid dimensions in .frg header: width={width}, height={height}")
    if width != height:
        raise ValueError(f"Retrieval currently requires square traces. Got {width}x{height}.")

    values: list[float] = []
    for line in lines[1:]:
        values.extend(_parse_number_line(line))
    expected = width * height
    if len(values) < expected:
        raise ValueError(f"Trace matrix too short: expected {expected} values, got {len(values)}.")

    # frog_convert.py writes one fixed-frequency slice per row in the file.
    # Normalize the in-memory convention to [delay, frequency].
    matrix = np.asarray(values[:expected], dtype=np.float64).reshape((height, width))
    trace = matrix.T
    trace = np.nan_to_num(trace, nan=0.0, posinf=0.0, neginf=0.0)
    trace = np.clip(trace, 0.0, None)
    max_trace = float(np.max(trace))
    if max_trace > 1e-15:
        trace /= max_trace

    dt_or_calib = float(header[2])
    dv_or_wave = float(header[3])
    center_or_wave = float(header[4])

    delay = (np.arange(width, dtype=np.float64) - width // 2) * dt_or_calib
    # Heuristic: binned FRG stores center frequency (PHz) around ~0.1-2;
    # raw FRG stores center wavelength (nm) usually >> 10.
    if abs(center_or_wave) < 10.0:
        freq_center = float(center_or_wave)
        frequency = (np.arange(height, dtype=np.float64) - height // 2) * dv_or_wave + center_or_wave
    else:
        freq_center = float(SPEED_LIGHT / (center_or_wave * 1e-9) / 1e15)
        wavelength_nm = (np.arange(height, dtype=np.float64) - height // 2) * dv_or_wave + center_or_wave
        wavelength_nm = np.clip(wavelength_nm, 1e-9, None)
        frequency = SPEED_LIGHT / (wavelength_nm * 1e-9) / 1e15

    return FrgData(path=path, delay=delay, frequency=frequency, frequency_center=freq_center, trace=trace)


def _safe_normalize(y: np.ndarray) -> np.ndarray:
    out = np.asarray(y, dtype=np.float64).copy()
    ymax = float(np.max(out)) if out.size else 0.0
    if ymax > 1e-15:
        out /= ymax
    return out


def _get_trace_colormap():
    """Build a pyqtgraph colormap from local common.custom_cmap when available."""
    global _TRACE_COLORMAP
    if _TRACE_COLORMAP is not None:
        return _TRACE_COLORMAP

    if custom_cmap is not None:
        try:
            x = np.linspace(0.0, 1.0, 256)
            rgba = np.asarray(custom_cmap(x), dtype=np.float64)
            if rgba.ndim == 2 and rgba.shape[0] >= 2 and rgba.shape[1] in (3, 4):
                if rgba.shape[1] == 3:
                    alpha = np.ones((rgba.shape[0], 1), dtype=np.float64)
                    rgba = np.hstack((rgba, alpha))
                rgba = np.clip(rgba, 0.0, 1.0)
                colors = (rgba * 255).astype(np.ubyte)
                pos = np.linspace(0.0, 1.0, colors.shape[0], dtype=np.float64)
                _TRACE_COLORMAP = pg.ColorMap(pos, colors)
                return _TRACE_COLORMAP
        except Exception:
            pass

    _TRACE_COLORMAP = pg.colormap.get("jet", source="matplotlib")
    return _TRACE_COLORMAP

def _phase_for_display(complex_signal: np.ndarray, intensity: np.ndarray, threshold: float = 0.02) -> np.ndarray:
    # Convention: E = sqrt(I) * exp(-i*phase)
    phase = -np.unwrap(np.angle(np.asarray(complex_signal, dtype=np.complex128)))
    if phase.size:
        phase -= phase[int(np.argmax(np.asarray(intensity, dtype=np.float64)))]
    mask = np.asarray(intensity, dtype=np.float64) > float(threshold)
    phase = np.where(mask, phase, np.nan)
    return phase


def _estimate_fwhm(x: np.ndarray, y: np.ndarray) -> float:
    x_arr = np.asarray(x, dtype=np.float64)
    y_arr = np.asarray(y, dtype=np.float64)
    if x_arr.size < 3 or y_arr.size < 3 or np.max(y_arr) <= 0:
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

def _center_time_axis(delay: np.ndarray, field: np.ndarray) -> np.ndarray:
    """Return a shifted copy of delay so that the intensity peak of field is at t=0."""
    t = np.asarray(delay, dtype=np.float64).ravel()
    et = np.asarray(field, dtype=np.complex128).ravel()
    if et.size < 3:
        return t.copy()

    intensity = np.abs(et) ** 2
    try:
        param, _ = fit_peak(t, intensity)
        center_t = float(param[1])
    except Exception:
        return t.copy()

    if not np.isfinite(center_t):
        return t.copy()

    return t - center_t


class _DualAxisCurve:
    """One plot with intensity on left axis and phase on right axis."""

    def __init__(self, plot_item: pg.PlotItem, x_label: str, left_label: str, right_label: str):
        self.plot_item = plot_item
        self.plot_item.showAxis("right")
        self.plot_item.getAxis("left").enableAutoSIPrefix(False)
        self.plot_item.getAxis("bottom").enableAutoSIPrefix(False)
        self.plot_item.getAxis("right").enableAutoSIPrefix(False)
        self.plot_item.setLabel("bottom", x_label)
        self.plot_item.setLabel("left", left_label)
        self.plot_item.getAxis("right").setLabel(right_label)
        self.plot_item.showGrid(x=True, y=True, alpha=0.3)
        self.plot_item.addLegend()

        self.phase_vb = pg.ViewBox()
        self.plot_item.scene().addItem(self.phase_vb)
        self.plot_item.getAxis("right").linkToView(self.phase_vb)
        self.phase_vb.setXLink(self.plot_item)
        self.plot_item.vb.sigResized.connect(self._update_views)
        self._update_views()

        self._intensity_best = self.plot_item.plot([], [], pen=pg.mkPen("#111111", width=3), name="best G")
        self._intensity_current = self.plot_item.plot(
            [],
            [],
            pen=pg.mkPen("#D62728", width=2.5),
            name="current",
        )
        self._phase_best = pg.PlotCurveItem([], [], pen=pg.mkPen("#1F77B4", width=2.5, style=QtCore.Qt.PenStyle.DashLine))
        self._phase_current = pg.PlotCurveItem([], [], pen=pg.mkPen("#9467BD", width=2.5, style=QtCore.Qt.PenStyle.DashLine))
        self.phase_vb.addItem(self._phase_best)
        self.phase_vb.addItem(self._phase_current)

    def _update_views(self) -> None:
        self.phase_vb.setGeometry(self.plot_item.vb.sceneBoundingRect())
        self.phase_vb.linkedViewChanged(self.plot_item.vb, self.phase_vb.XAxis)

    def clear(self, title: str) -> None:
        self.plot_item.setTitle(title)
        self._intensity_best.setData([], [])
        self._intensity_current.setData([], [])
        self._phase_best.setData([], [])
        self._phase_current.setData([], [])

    def set_data(
        self,
        x: np.ndarray,
        best_intensity: np.ndarray,
        best_phase: np.ndarray,
        current_intensity: Optional[np.ndarray],
        current_phase: Optional[np.ndarray],
        title: str,
    ) -> None:
        self.plot_item.setTitle(title)
        x_arr = np.asarray(x, dtype=np.float64)
        self._intensity_best.setData(x_arr, np.asarray(best_intensity, dtype=np.float64))
        self._phase_best.setData(x_arr, np.asarray(best_phase, dtype=np.float64))
        if current_intensity is not None and current_phase is not None:
            self._intensity_current.setData(x_arr, np.asarray(current_intensity, dtype=np.float64))
            self._phase_current.setData(x_arr, np.asarray(current_phase, dtype=np.float64))
        else:
            self._intensity_current.setData([], [])
            self._phase_current.setData([], [])


class RetrievalWorker(QtCore.QObject):
    """Background worker running retrieval on a QThread."""

    progress = QtCore.Signal(object)  # progress payload dict
    finished = QtCore.Signal(object)  # RetrievalResult
    failed = QtCore.Signal(str)

    def __init__(self, frg: FrgData, geometry: str, algorithm: str, config_dict: dict):
        super().__init__()
        self.frg = frg
        self.geometry = geometry
        self.algorithm = algorithm
        self.config_dict = config_dict
        self._stop_event = threading.Event()

    @QtCore.Slot()
    def request_stop(self) -> None:
        self._stop_event.set()

    @QtCore.Slot()
    def run(self) -> None:
        try:
            config = RetrievalConfig(**self.config_dict)

            def _progress(payload: dict) -> None:
                self.progress.emit(payload)

            config.progress_callback = _progress
            config.progress_interval = 50
            config.verbose = False
            config.stop_requested = self._stop_event.is_set

            result = retrieve_pulse(
                trace=self.frg.trace,
                delay=self.frg.delay,
                frequency=self.frg.frequency,
                algorithm=self.algorithm.lower(),
                geometry=self.geometry,
                config=config,
            )
            self.finished.emit(result)
        except Exception:
            self.failed.emit(traceback.format_exc())


class FrogRetrievalGUI(QtWidgets.QMainWindow):
    """Main window for FROG retrieval."""
    result_data_ready = QtCore.Signal(object)

    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("FROG Retrieval GUI")
        self._initial_window_width = 1200
        self._initial_window_height = 700
        self._initial_size_applied = False
        self.resize(self._initial_window_width, self._initial_window_height)

        self.frg_data: Optional[FrgData] = None
        self.result: Optional[RetrievalResult] = None
        self._thread: Optional[QtCore.QThread] = None
        self._worker: Optional[RetrievalWorker] = None
        self._last_dir = os.getcwd()
        self._active_algorithm = "RANA"
        self._trace_layers: dict[int, dict] = {}
        self._trace_original_raw: Optional[np.ndarray] = None
        self._trace_current_raw: Optional[np.ndarray] = None
        self._trace_diff_raw: Optional[np.ndarray] = None
        self._last_progress_iter: int = 0
        self._last_progress_g: float = float("nan")
        self._current_g_best_x: list[int] = []
        self._current_g_best_y: list[float] = []

        pg.setConfigOptions(antialias=True)
        self._build_ui()
        self._build_status_output()
        self._set_status("Ready.")

    def showEvent(self, event) -> None:
        super().showEvent(event)
        if self._initial_size_applied:
            return
        self._initial_size_applied = True
        # First show can be stretched by transient minimumSizeHint from plot widgets.
        # Re-apply desired startup size after layout settles.
        QtCore.QTimer.singleShot(
            0,
            lambda: self.resize(self._initial_window_width, self._initial_window_height),
        )

    def _build_ui(self) -> None:
        root = QtWidgets.QWidget()
        self.setCentralWidget(root)
        vbox = QtWidgets.QVBoxLayout(root)
        vbox.setContentsMargins(2, 2, 2, 2)
        vbox.setSpacing(2)

        top_row = QtWidgets.QHBoxLayout()
        top_row.setContentsMargins(2, 2, 2, 2)
        top_row.setSpacing(4)
        vbox.addLayout(top_row)

        self.btn_open = QtWidgets.QPushButton("Open Binned Trace")
        self.btn_open.setStyleSheet("background-color: #90EE90;")  # light green
        self.btn_run = QtWidgets.QPushButton("Run Retrieval")
        self.btn_run.setStyleSheet("background-color: #90EE90;")
        self.btn_stop = QtWidgets.QPushButton("Stop")
        self.btn_stop.setEnabled(False)
        self.btn_save = QtWidgets.QPushButton("Save")
        self.btn_save.setEnabled(False)
        self.file_label = QtWidgets.QLabel("No file loaded")
        self.file_label.setStyleSheet("color:#444;")
        self.combo_model = QtWidgets.QComboBox()
        self.combo_model.addItems(["SHG", "PG", "THG", "SD"])
        self.combo_algo = QtWidgets.QComboBox()
        self.combo_algo.addItems(["RANA"])
        self.chk_trace_log = QtWidgets.QCheckBox("Trace Log Scale")
        self.chk_trace_log.setChecked(False)
        self.trace_log_vmin = QtWidgets.QLineEdit("1e-4")
        self.trace_log_vmin.setMaximumWidth(70)
        self.trace_log_vmin.setToolTip("Minimum trace intensity used for log-scale display.")

        top_row.addWidget(self.btn_open)
        top_row.addWidget(QtWidgets.QLabel("Model:"))
        top_row.addWidget(self.combo_model)
        top_row.addWidget(QtWidgets.QLabel("Retrieval:"))
        top_row.addWidget(self.combo_algo)
        top_row.addWidget(self.chk_trace_log)
        top_row.addWidget(QtWidgets.QLabel("vmin:"))
        top_row.addWidget(self.trace_log_vmin)
        top_row.addWidget(self.btn_run)
        top_row.addWidget(self.btn_stop)
        top_row.addWidget(self.btn_save)
        top_row.addWidget(self.file_label, 1)

        split = QtWidgets.QSplitter(QtCore.Qt.Vertical)
        split.setHandleWidth(2)
        split.setChildrenCollapsible(False)
        vbox.addWidget(split, 1)
        self._main_splitter = split

        param_scroll = QtWidgets.QScrollArea()
        param_scroll.setWidgetResizable(True)
        param_scroll.setFrameShape(QtWidgets.QFrame.Shape.NoFrame)
        param_scroll.setSizePolicy(
            QtWidgets.QSizePolicy.Policy.Expanding,
            QtWidgets.QSizePolicy.Policy.Maximum,
        )
        split.addWidget(param_scroll)
        self._param_scroll = param_scroll

        param_container = QtWidgets.QWidget()
        param_container.setSizePolicy(
            QtWidgets.QSizePolicy.Policy.Expanding,
            QtWidgets.QSizePolicy.Policy.Maximum,
        )
        param_scroll.setWidget(param_container)
        self._param_container = param_container
        param_layout = QtWidgets.QHBoxLayout(param_container)
        param_layout.setContentsMargins(0, 0, 0, 0)
        param_layout.setSpacing(0)

        param_group = QtWidgets.QGroupBox("Parameters")
        param_group.setSizePolicy(
            QtWidgets.QSizePolicy.Policy.Expanding,
            QtWidgets.QSizePolicy.Policy.Fixed,
        )
        self._param_group = param_group
        grid = QtWidgets.QGridLayout(param_group)
        grid.setHorizontalSpacing(1)
        grid.setVerticalSpacing(1)
        grid.setContentsMargins(0, 0, 0, 0)
        for _c in range(7):
            grid.setColumnStretch(_c, 1)
        param_group.setSizePolicy(
            QtWidgets.QSizePolicy.Policy.Expanding,
            QtWidgets.QSizePolicy.Policy.Maximum,
        )
        self._param_grid = grid
        self._param_label_map: dict[QtWidgets.QWidget, QtWidgets.QWidget] = {}
        self._param_cell_map: dict[QtWidgets.QWidget, QtWidgets.QWidget] = {}
        self._param_entries: list[QtWidgets.QWidget] = []
        _param_index = 0

        def add_param(name: str, widget: QtWidgets.QWidget) -> None:
            nonlocal _param_index
            row = _param_index // 7
            col = _param_index % 7
            cell = QtWidgets.QWidget()
            cell_l = QtWidgets.QHBoxLayout(cell)
            cell_l.setContentsMargins(0, 0, 0, 0)
            cell_l.setSpacing(2)
            label = QtWidgets.QLabel(name)
            cell_l.addWidget(label)
            if isinstance(widget, (QtWidgets.QLineEdit, QtWidgets.QSpinBox, QtWidgets.QComboBox)):
                widget.setMaximumWidth(120)
            cell_l.addWidget(widget)
            cell_l.addStretch(1)
            grid.addWidget(cell, row, col)
            self._param_label_map[widget] = label
            self._param_cell_map[widget] = cell
            self._param_entries.append(widget)
            _param_index += 1

        self.rana_g_cutoff = QtWidgets.QLineEdit("1e-3")
        add_param("G_cutoff", self.rana_g_cutoff)

        self.rana_gp_cutoff = QtWidgets.QLineEdit("0.01")
        add_param("G_prime_cutoff", self.rana_gp_cutoff)

        self.rana_full_iter_cap = QtWidgets.QSpinBox()
        self.rana_full_iter_cap.setRange(1, 1_000_000)
        self.rana_full_iter_cap.setValue(100)
        add_param("Max iterations", self.rana_full_iter_cap)

        self.rana_weight_factor = QtWidgets.QLineEdit("1.0")
        add_param("Zero-pixel weight factor", self.rana_weight_factor)

        param_layout.addWidget(param_group)

        self._shared_param_widgets = []
        self._rana_param_widgets = [
            self.rana_g_cutoff,
            self.rana_gp_cutoff,
            self.rana_full_iter_cap,
        ]
        self._rana_extra_widgets = [
            self.rana_weight_factor,
        ]

        self.graphics = pg.GraphicsLayoutWidget()
        self.graphics.setBackground("w")
        split.addWidget(self.graphics)
        split.setStretchFactor(0, 0)
        split.setStretchFactor(1, 1)
        QtCore.QTimer.singleShot(0, self._apply_minimum_param_height)

        self.plot_original = self.graphics.addPlot(row=0, col=0, title="Original trace")
        self.plot_current = self.graphics.addPlot(row=0, col=1, title="Current trace")
        self.plot_diff = self.graphics.addPlot(row=0, col=2, title="Trace difference")
        self.plot_time = self.graphics.addPlot(row=1, col=0, title="Time profile")
        self.plot_freq = self.graphics.addPlot(row=1, col=1, title="Frequency")
        self.plot_spec = self.graphics.addPlot(row=1, col=2, title="Current G vs interaction number")

        for p in (self.plot_original, self.plot_current, self.plot_diff):
            p.getAxis("left").enableAutoSIPrefix(False)
            p.getAxis("bottom").enableAutoSIPrefix(False)
            p.setLabel("bottom", "Delay (fs)")
            p.setLabel("left", "Frequency (PHz)")
            p.showGrid(x=True, y=True, alpha=0.2)

        self.time_dual = _DualAxisCurve(self.plot_time, "Time (fs)", "Intensity", "Phase (rad)")
        self.freq_dual = _DualAxisCurve(self.plot_freq, "Frequency (PHz)", "Intensity", "Phase (rad)")

        self.plot_spec.setLabel("bottom", "Interaction number")
        self.plot_spec.setLabel("left", "Current G")
        self.plot_spec.getAxis("bottom").enableAutoSIPrefix(False)
        self.plot_spec.getAxis("left").enableAutoSIPrefix(False)
        self.plot_spec.showGrid(x=True, y=True, alpha=0.2)
        self._current_g_curve = self.plot_spec.plot([], [], pen=pg.mkPen("#111111", width=2.5))

        self.btn_open.clicked.connect(self._on_open_clicked)
        self.btn_run.clicked.connect(self._on_run_clicked)
        self.btn_stop.clicked.connect(self._on_stop_clicked)
        self.btn_save.clicked.connect(self._on_save_clicked)
        self.combo_algo.currentTextChanged.connect(self._on_algo_changed)
        self.chk_trace_log.toggled.connect(self._on_trace_scale_toggled)
        self.trace_log_vmin.editingFinished.connect(self._on_trace_vmin_changed)
        self._on_algo_changed(self.combo_algo.currentText())

    def _build_status_output(self) -> None:
        sb = self.statusBar()
        box = QtWidgets.QWidget()
        h = QtWidgets.QHBoxLayout(box)
        h.setContentsMargins(0, 0, 0, 0)
        h.setSpacing(8)

        h.addWidget(QtWidgets.QLabel("Best G:"))
        self.best_g_display = QtWidgets.QLabel("--")
        self.best_g_display.setMinimumWidth(90)
        self.best_g_display.setAlignment(QtCore.Qt.AlignmentFlag.AlignRight | QtCore.Qt.AlignmentFlag.AlignVCenter)
        h.addWidget(self.best_g_display)

        h.addWidget(QtWidgets.QLabel("FWHM:"))
        self.fwhm_display = QtWidgets.QLabel("--")
        self.fwhm_display.setMinimumWidth(70)
        self.fwhm_display.setAlignment(QtCore.Qt.AlignmentFlag.AlignRight | QtCore.Qt.AlignmentFlag.AlignVCenter)
        h.addWidget(self.fwhm_display)

        sb.addPermanentWidget(box)

    def _set_status(self, text: str, timeout_ms: int = 0) -> None:
        self.statusBar().showMessage(str(text), int(timeout_ms))

    def _parse_float(self, line: QtWidgets.QLineEdit, name: str) -> float:
        text = line.text().strip()
        if not text:
            raise ValueError(f"{name} cannot be empty.")
        return float(text)

    def _set_widget_row_visible(self, widget: QtWidgets.QWidget, visible: bool) -> None:
        cell = self._param_cell_map.get(widget)
        if cell is not None:
            cell.setVisible(bool(visible))
        label = self._param_label_map.get(widget)
        if label is not None:
            label.setVisible(bool(visible))
        widget.setVisible(bool(visible))

    def _reflow_param_widgets(self, visible_widgets: list[QtWidgets.QWidget]) -> None:
        for w in self._param_entries:
            cell = self._param_cell_map.get(w)
            if cell is None:
                continue
            self._param_grid.removeWidget(cell)
            cell.setVisible(False)

        # Reset column stretches so old columns don't hold stale widths.
        ncols = 7
        for c in range(ncols):
            self._param_grid.setColumnStretch(c, 0)

        used_cols = min(len(visible_widgets), ncols)
        for c in range(used_cols):
            self._param_grid.setColumnStretch(c, 1)

        for idx, w in enumerate(visible_widgets):
            row = idx // ncols
            col = idx % ncols
            cell = self._param_cell_map.get(w)
            if cell is None:
                continue
            self._param_grid.addWidget(cell, row, col)
            cell.setVisible(True)
            w.setVisible(True)
        self._schedule_minimum_param_height_update()

    def _schedule_minimum_param_height_update(self) -> None:
        QtCore.QTimer.singleShot(0, self._apply_minimum_param_height)

    def _apply_minimum_param_height(self) -> None:
        splitter = getattr(self, "_main_splitter", None)
        param_scroll = getattr(self, "_param_scroll", None)
        param_container = getattr(self, "_param_container", None)
        param_group = getattr(self, "_param_group", None)
        if splitter is None or param_scroll is None or param_container is None or param_group is None:
            return

        # Force the grid layout to recalculate before querying sizes.
        self._param_grid.invalidate()
        self._param_grid.activate()
        param_container.adjustSize()
        param_group.adjustSize()
        # Process pending layout events so sizeHint is up-to-date.
        QtWidgets.QApplication.processEvents()
        target_h = int(max(1, param_group.sizeHint().height() + 1))
        param_scroll.setMinimumHeight(target_h)
        param_scroll.setMaximumHeight(target_h)
        sizes = splitter.sizes()
        total = int(sum(sizes)) if sizes else int(self.height())
        if total <= target_h:
            total = target_h + 1
        splitter.setSizes([target_h, total - target_h])

    @QtCore.Slot(str)
    def _on_algo_changed(self, algo_text: str) -> None:
        visible_widgets = list(self._shared_param_widgets)
        visible_widgets.extend(self._rana_param_widgets)
        visible_widgets.extend(self._rana_extra_widgets)
        self._reflow_param_widgets(visible_widgets)

    def _collect_config_dict(self) -> dict:
        base = {}
        base.update(
            {
                "rana_g_cutoff": self._parse_float(self.rana_g_cutoff, "rana_g_cutoff"),
                "rana_gp_cutoff": self._parse_float(self.rana_gp_cutoff, "rana_gp_cutoff"),
                "rana_full_iter_cap": int(self.rana_full_iter_cap.value()),
                "rana_weight_factor": self._parse_float(self.rana_weight_factor, "rana_weight_factor"),
            }
        )
        return base

    def _reset_metric_displays(self) -> None:
        self.best_g_display.setText("--")
        self.fwhm_display.setText("--")

    def _reset_current_g_history(self) -> None:
        self._current_g_best_x = []
        self._current_g_best_y = []
        self._refresh_current_g_plot()

    def _refresh_current_g_plot(self) -> None:
        self.plot_spec.setTitle("Best current G of 4 trials vs interaction number")
        self.plot_spec.setLabel("bottom", "Interaction number")
        self.plot_spec.setLabel("left", "Current G")
        self._current_g_curve.setData(
            np.asarray(self._current_g_best_x, dtype=np.float64),
            np.asarray(self._current_g_best_y, dtype=np.float64),
        )
        self.plot_spec.autoRange()

    def _append_current_g_history(
        self,
        iteration: int,
        current_g: Optional[float] = None,
        step_count: int = 1,
        g_hist: Optional[np.ndarray] = None,
    ) -> None:
        try:
            it = int(iteration)
            steps = int(step_count)
        except Exception:
            return
        if it <= 0:
            return

        hist_arr = None
        if g_hist is not None:
            try:
                hist_arr = np.asarray(g_hist, dtype=np.float64).ravel()
            except Exception:
                hist_arr = None
            if hist_arr is not None:
                hist_arr = hist_arr[np.isfinite(hist_arr)]

        if hist_arr is not None and hist_arr.size > 0:
            new_y = hist_arr.tolist()
            new_x = list(range(max(1, it - hist_arr.size + 1), it + 1))
        else:
            if current_g is None:
                return
            g_val = float(current_g)
            if not np.isfinite(g_val):
                return
            if steps <= 0:
                steps = 1
            new_x = list(range(max(1, it - steps + 1), it + 1))
            new_y = [g_val] * len(new_x)

        if not self._current_g_best_x:
            self._current_g_best_x = new_x
            self._current_g_best_y = new_y
            self._refresh_current_g_plot()
            return

        keep = 0
        start_x = new_x[0]
        while keep < len(self._current_g_best_x) and self._current_g_best_x[keep] < start_x:
            keep += 1
        self._current_g_best_x = self._current_g_best_x[:keep]
        self._current_g_best_y = self._current_g_best_y[:keep]

        x_to_idx = {x: idx for idx, x in enumerate(self._current_g_best_x)}
        for x, y in zip(new_x, new_y):
            idx = x_to_idx.get(x)
            if idx is None:
                self._current_g_best_x.append(x)
                self._current_g_best_y.append(float(y))
                x_to_idx[x] = len(self._current_g_best_x) - 1
            else:
                self._current_g_best_y[idx] = float(y)
        self._refresh_current_g_plot()

    def _update_metric_displays(self, best_g: Optional[float], best_field: Optional[np.ndarray]) -> None:
        if best_g is not None and np.isfinite(float(best_g)):
            self.best_g_display.setText(f"{float(best_g):.6g}")
        else:
            self.best_g_display.setText("--")

        if self.frg_data is None or best_field is None:
            self.fwhm_display.setText("--")
            return

        field = np.asarray(best_field, dtype=np.complex128).ravel()
        delay = np.asarray(self.frg_data.delay, dtype=np.float64).ravel()
        if field.size != delay.size:
            self.fwhm_display.setText("--")
            return
        intensity = _safe_normalize(np.abs(field) ** 2)
        fwhm = _estimate_fwhm(delay, intensity)
        if np.isfinite(fwhm):
            self.fwhm_display.setText(f"{float(fwhm):.4g}")
        else:
            self.fwhm_display.setText("--")

    @QtCore.Slot(bool)
    def _on_trace_scale_toggled(self, _checked: bool) -> None:
        self._refresh_trace_views()

    @QtCore.Slot()
    def _on_trace_vmin_changed(self) -> None:
        self.trace_log_vmin.setText(f"{self._trace_log_vmin():.6g}")
        self._refresh_trace_views()

    def _trace_log_vmin(self) -> float:
        text = self.trace_log_vmin.text().strip()
        try:
            value = float(text)
        except Exception:
            value = 1e-4
        if not np.isfinite(value) or value <= 0.0:
            value = 1e-4
        return float(value)

    def _refresh_trace_views(self) -> None:
        if self.frg_data is None:
            return
        if self._trace_original_raw is not None:
            self._draw_trace(self.plot_original, self.frg_data.delay, self.frg_data.frequency, self._trace_original_raw, "Original trace")
        if self._trace_current_raw is not None:
            self._draw_trace(self.plot_current, self.frg_data.delay, self.frg_data.frequency, self._trace_current_raw, "Current trace")
        if self._trace_diff_raw is not None:
            self._draw_trace(self.plot_diff, self.frg_data.delay, self.frg_data.frequency, self._trace_diff_raw, "Trace difference")

    def _apply_loaded_frg_data(self, frg: FrgData, display_name: str) -> None:
        self.frg_data = frg
        self.result = None
        self.btn_save.setEnabled(False)
        self.file_label.setText(str(display_name))
        self._set_status(f"Loaded {display_name}")
        self._reset_progress_labels()
        self._reset_metric_displays()
        self._trace_original_raw = np.asarray(self.frg_data.trace, dtype=np.float64).copy()
        self._trace_current_raw = None
        self._trace_diff_raw = None
        self._draw_trace(self.plot_original, self.frg_data.delay, self.frg_data.frequency, self.frg_data.trace, "Original trace")
        self._clear_result_views()

    def load_frg_data(
        self,
        frg: FrgData,
        source_label: Optional[str] = None,
        trace_order: Optional[str] = None,
    ) -> None:
        """Load FRG data directly from memory (no file I/O)."""
        delay = np.asarray(frg.delay, dtype=np.float64).ravel()
        freq = np.asarray(frg.frequency, dtype=np.float64).ravel()
        trace = np.asarray(frg.trace, dtype=np.float64)
        order = str(trace_order).strip().lower() if trace_order is not None else ""
        if order in ("frequency_delay", "freq_delay", "fd"):
            if trace.shape != (freq.size, delay.size):
                raise ValueError(
                    f"FRG shape mismatch for frequency_delay: trace={trace.shape}, "
                    f"expected {(freq.size, delay.size)}"
                )
            trace = trace.T
        elif order in ("delay_frequency", "delay_freq", "df", ""):
            if trace.shape == (delay.size, freq.size):
                pass
            elif trace.shape == (freq.size, delay.size) and delay.size != freq.size:
                # Non-square case can be inferred safely.
                trace = trace.T
            else:
                raise ValueError(
                    f"FRG shape mismatch for delay_frequency: trace={trace.shape}, "
                    f"expected {(delay.size, freq.size)} or {(freq.size, delay.size)}"
                )
        else:
            raise ValueError("trace_order must be one of delay_frequency/frequency_delay.")
        trace = np.nan_to_num(trace, nan=0.0, posinf=0.0, neginf=0.0)
        trace = np.clip(trace, 0.0, None)
        zmax = float(np.max(trace))
        if zmax > 1e-15:
            trace = trace / zmax

        safe_path = str(getattr(frg, "path", source_label or "memory.frg"))
        freq_center = float(getattr(frg, "frequency_center", _center_bin_value(freq)))
        loaded = FrgData(
            path=safe_path,
            delay=delay.copy(),
            frequency=freq.copy(),
            frequency_center=freq_center,
            trace=trace.copy(),
            geometry=str(getattr(frg, "geometry", "shg-frog")),
        )
        display = source_label or os.path.basename(safe_path) or "memory.frg"
        self._apply_loaded_frg_data(loaded, display)

    def load_frg_payload(self, payload: dict, source_label: Optional[str] = None) -> None:
        """Load FRG payload dict with keys path/delay/frequency/frequency_center/trace."""
        if not isinstance(payload, dict):
            raise ValueError("FRG payload must be a dict.")
        frg = FrgData(
            path=str(payload.get("path", source_label or "memory.frg")),
            delay=np.asarray(payload.get("delay", []), dtype=np.float64).ravel(),
            frequency=np.asarray(payload.get("frequency", []), dtype=np.float64).ravel(),
            frequency_center=float(payload.get("frequency_center", np.nan)),
            trace=np.asarray(payload.get("trace", []), dtype=np.float64),
            geometry=str(payload.get("geometry", "shg-frog")),
        )
        if not np.isfinite(frg.frequency_center):
            frg.frequency_center = _center_bin_value(frg.frequency)
        self.load_frg_data(
            frg,
            source_label=source_label,
            trace_order=str(payload.get("trace_order", "delay_frequency")),
        )

    def _on_open_clicked(self) -> None:
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self,
            "Open Trace File",
            self._last_dir,
            "Trace files (*.frg *.dat);;FRG files (*.frg);;A.dat files (*.dat);;All files (*)",
        )
        if not path:
            return
        self._last_dir = os.path.dirname(path) or self._last_dir
        try:
            self._apply_loaded_frg_data(_read_frg_file(path), os.path.basename(path))
        except Exception as e:
            self._set_status(f"Open error: {e}", timeout_ms=10000)
            QtWidgets.QMessageBox.critical(self, "Open Error", str(e))

    def _on_run_clicked(self) -> None:
        if self.frg_data is None:
            QtWidgets.QMessageBox.warning(self, "No Data", "Please open a .frg or .dat file first.")
            return
        if self._thread is not None and self._thread.isRunning():
            QtWidgets.QMessageBox.information(self, "Busy", "Retrieval is already running.")
            return

        try:
            config_dict = self._collect_config_dict()
        except Exception as e:
            QtWidgets.QMessageBox.warning(self, "Invalid Parameters", str(e))
            return

        self._reset_progress_labels()
        self._reset_metric_displays()
        self._reset_current_g_history()
        self._last_progress_iter = 0
        self._last_progress_g = float("nan")
        self.btn_run.setEnabled(False)
        self.btn_stop.setEnabled(True)
        self.btn_save.setEnabled(False)
        self._active_algorithm = self.combo_algo.currentText().strip().upper()
        geometry = f"{self.combo_model.currentText().strip().lower()}-frog"
        self.frg_data.geometry = geometry
        self._set_status(f"{self._active_algorithm} retrieval running ({geometry})...")

        self._thread = QtCore.QThread(self)
        self._worker = RetrievalWorker(self.frg_data, geometry, self.combo_algo.currentText(), config_dict)
        self._worker.moveToThread(self._thread)

        self._thread.started.connect(self._worker.run)
        self._worker.progress.connect(self._on_worker_progress)
        self._worker.finished.connect(self._on_worker_finished)
        self._worker.failed.connect(self._on_worker_failed)
        self._worker.finished.connect(self._thread.quit)
        self._worker.failed.connect(self._thread.quit)
        self._thread.finished.connect(self._cleanup_worker)

        self._thread.start()

    def _on_stop_clicked(self) -> None:
        if self._thread is None or not self._thread.isRunning() or self._worker is None:
            self._set_status("No running retrieval to stop.", timeout_ms=4000)
            return
        self._worker.request_stop()
        self.btn_stop.setEnabled(False)
        self._set_status("Stop requested... waiting for current iteration to finish.")

    @QtCore.Slot()
    def _cleanup_worker(self) -> None:
        self.btn_run.setEnabled(True)
        self.btn_stop.setEnabled(False)
        self.btn_save.setEnabled(self.result is not None)
        if self._worker is not None:
            self._worker.deleteLater()
            self._worker = None
        if self._thread is not None:
            self._thread.deleteLater()
            self._thread = None

    def _reset_progress_labels(self) -> None:
        # Top progress labels were removed; keep method for call-site compatibility.
        return

    @QtCore.Slot(object)
    def _on_worker_progress(self, payload_obj: object) -> None:
        if not isinstance(payload_obj, dict):
            return
        it = int(payload_obj.get("iter", 0))
        if it > self._last_progress_iter:
            self._last_progress_iter = it
        g_now = payload_obj.get("current_g")
        if g_now is not None:
            try:
                g_val = float(g_now)
                if np.isfinite(g_val):
                    self._last_progress_g = g_val
            except Exception:
                pass

        line_text = payload_obj.get("line_text")
        if isinstance(line_text, str) and line_text:
            self._set_status(line_text)

        if self.frg_data is None:
            return
        current_trace = payload_obj.get("current_trace")
        field = payload_obj.get("field")
        best_field = payload_obj.get("best_field")
        if current_trace is None or (field is None and best_field is None):
            return

        measured = np.asarray(self.frg_data.trace, dtype=np.float64)
        current_trace_arr = np.asarray(current_trace, dtype=np.float64)
        self._trace_current_raw = current_trace_arr.copy()
        field_arr = np.asarray(field, dtype=np.complex128) if field is not None else None
        best_field_arr = np.asarray(best_field, dtype=np.complex128) if best_field is not None else None
        best_g = payload_obj.get("best_g")
        current_g = payload_obj.get("current_g")
        step_count = int(payload_obj.get("iter_step", 1))
        g_hist = payload_obj.get("g_hist")
        stage = str(payload_obj.get("stage", ""))
        if stage == "full grid" and current_g is not None:
            self._append_current_g_history(it, float(current_g), step_count=step_count, g_hist=g_hist)
        self._update_metric_displays(float(best_g) if best_g is not None else None, best_field_arr)

        self._draw_trace(
            self.plot_current,
            self.frg_data.delay,
            self.frg_data.frequency,
            current_trace_arr,
            f"Current trace (iter {it})",
        )

        denom = float(np.sum(current_trace_arr * current_trace_arr))
        mu = float(np.sum(measured * current_trace_arr) / denom) if denom > 1e-15 else 0.0
        diff = measured - mu * current_trace_arr
        self._trace_diff_raw = diff.copy()
        self._draw_trace(
            self.plot_diff,
            self.frg_data.delay,
            self.frg_data.frequency,
            diff,
            f"Trace difference (iter {it})",
        )

        # Keep 1D panels on best-so-far plus current solution.
        if best_field_arr is not None:
            self._draw_profiles_from_fields(best_field_arr, measured, current_field=field_arr)
        elif field_arr is not None:
            self._draw_profiles_from_fields(field_arr, measured, current_field=field_arr)

    @QtCore.Slot(object)
    def _on_worker_finished(self, result_obj: object) -> None:
        if not isinstance(result_obj, RetrievalResult):
            self._set_status("Retrieval returned unexpected result type.", timeout_ms=12000)
            return
        if self.frg_data is None:
            self._set_status("Loaded trace disappeared unexpectedly.", timeout_ms=12000)
            return

        self.result = result_obj
        self.btn_save.setEnabled(True)
        best_g = float(result_obj.diagnostics.get("best_error_g", np.nan))
        stopped_by_user = bool(result_obj.diagnostics.get("stopped_by_user", False))

        measured = np.asarray(self.frg_data.trace, dtype=np.float64)
        current_trace = np.asarray(result_obj.retrieved_trace, dtype=np.float64)
        self._trace_current_raw = current_trace.copy()
        self._draw_trace(self.plot_current, self.frg_data.delay, self.frg_data.frequency, current_trace, "Current trace")

        denom = float(np.sum(current_trace * current_trace))
        mu = float(np.sum(measured * current_trace) / denom) if denom > 1e-15 else 0.0
        diff = measured - mu * current_trace
        self._trace_diff_raw = diff.copy()
        self._draw_trace(self.plot_diff, self.frg_data.delay, self.frg_data.frequency, diff, "Trace difference")

        final_field = np.asarray(result_obj.field, dtype=np.complex128)
        self._update_metric_displays(best_g, final_field)
        self._draw_profiles_from_fields(final_field, measured, current_field=final_field)

        timing = result_obj.diagnostics.get("timing_per_iteration", {})
        elapsed_s = float(np.sum(np.asarray(timing.get("total_s", []), dtype=np.float64))) if timing else float("nan")
        if not np.isfinite(elapsed_s):
            elapsed_s = float(result_obj.diagnostics.get("elapsed_s", np.nan))
        sigma_text = f", sigma={result_obj.best_sigma:.4g}" if np.isfinite(result_obj.best_sigma) else ""
        elapsed_text = f"{elapsed_s:.2f}" if np.isfinite(elapsed_s) else "nan"
        algo_tag = self._active_algorithm if self._active_algorithm else self.combo_algo.currentText().strip().upper()
        report_iter = int(self._last_progress_iter)
        g_display = float(self._last_progress_g)
        if stopped_by_user:
            self._set_status(
                f"Retrieval stopped by user at iter={report_iter}, "
                f"G={g_display:.5g}, best G={best_g:.5g}, t={elapsed_text}s{sigma_text}"
            )
        else:
            self._set_status(
                f"[{algo_tag}] iter={report_iter} G={g_display:.5g}, "
                f"best G={best_g:.5g}, t={elapsed_text}s{sigma_text}"
            )
        try:
            payload = self.export_result_payload()
            if payload is not None:
                self.result_data_ready.emit(payload)
        except Exception:
            pass

    @QtCore.Slot(str)
    def _on_worker_failed(self, err_text: str) -> None:
        self._set_status("Retrieval failed.", timeout_ms=12000)
        QtWidgets.QMessageBox.critical(self, "Retrieval Failed", err_text)

    def _suggest_save_prefix(self) -> str:
        if self.frg_data is not None and getattr(self.frg_data, "path", ""):
            p = str(self.frg_data.path)
            if p.lower().endswith(".frg"):
                return p[:-4]
            return p
        return os.path.join(self._last_dir, "retrieval_result")

    @staticmethod
    def _normalize_prefix_path(path: str) -> str:
        prefix = str(path).strip()
        if not prefix:
            return prefix
        known = (".A.dat", ".Arecon.dat", ".Ek.dat", ".Ew.dat", ".Speck.dat")
        lower = prefix.lower()
        for s in known:
            if lower.endswith(s.lower()):
                return prefix[: -len(s)]
        if lower.endswith(".dat"):
            return prefix[:-4]
        return prefix

    @staticmethod
    def _align_trace_for_export(delay: np.ndarray, frequency: np.ndarray, trace: np.ndarray) -> np.ndarray:
        width = int(np.asarray(delay).size)
        height = int(np.asarray(frequency).size)
        z = np.asarray(trace, dtype=np.float64)
        if z.shape == (width, height):
            out = z
        elif z.shape == (height, width):
            out = z.T
        else:
            raise ValueError(f"Trace shape mismatch: got {z.shape}, expected {(width, height)} or {(height, width)}.")
        out = np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)
        return np.clip(out, 0.0, None)

    def _save_trace_dat(self, path: str, delay: np.ndarray, frequency: np.ndarray, trace: np.ndarray) -> None:
        delay_arr = np.asarray(delay, dtype=np.float64).ravel()
        freq_arr = np.asarray(frequency, dtype=np.float64).ravel()
        z = self._align_trace_for_export(delay_arr, freq_arr, trace)

        width = int(delay_arr.size)
        height = int(freq_arr.size)
        freq_safe = np.where(freq_arr > 1e-12, freq_arr, 1e-12)
        wavelength_nm = SPEED_LIGHT / (freq_safe * 1e15) * 1e9
        z_min = float(np.min(z)) if z.size else 0.0
        z_max = float(np.max(z)) if z.size else 0.0

        with open(path, "w", encoding="utf-8") as f:
            f.write(f"{width}\t{height}\n")
            f.write(f"{z_min:.6e}\t{z_max:.6e}\n")
            for val in wavelength_nm:
                f.write(f"{float(val):.6e}\n")
            for val in delay_arr:
                f.write(f"{float(val):.6e}\n")
            for val in z.reshape(-1):
                f.write(f"{float(val):.6e}\n")

    @staticmethod
    def _save_profile_dat(path: str, axis: np.ndarray, intensity: np.ndarray, phase: np.ndarray, field_complex: np.ndarray) -> None:
        x = np.asarray(axis, dtype=np.float64).ravel()
        inten = np.asarray(intensity, dtype=np.float64).ravel()
        ph = np.asarray(phase, dtype=np.float64).ravel()
        ec = np.asarray(field_complex, dtype=np.complex128).ravel()
        n = x.size
        if inten.size != n or ph.size != n or ec.size != n:
            raise ValueError(f"Profile length mismatch for {os.path.basename(path)}.")
        data = np.column_stack((x, inten, ph, np.real(ec), np.imag(ec)))
        np.savetxt(path, data, fmt="%.12e", delimiter="\t")

    def _build_export_profiles(self, best_field: np.ndarray) -> dict:
        if self.frg_data is None:
            raise RuntimeError("No FRG data loaded.")
        field = np.asarray(best_field, dtype=np.complex128).ravel()
        n = field.size
        delay = _center_time_axis(self.frg_data.delay, field)
        if delay.size != n:
            raise ValueError(f"Field length {n} does not match delay axis length {delay.size}.")

        # Time-domain profile.
        amp_t = np.sqrt(max(float(np.max(np.abs(field) ** 2)), 1e-30))
        time_field = field / amp_t
        time_intensity = _safe_normalize(np.abs(time_field) ** 2)
        time_phase = -np.unwrap(np.angle(time_field))
        if time_phase.size:
            time_phase -= time_phase[int(np.argmax(time_intensity))]

        # Frequency-domain profile on the fundamental field axis derived from the trace center.
        freq_rel, freq_field = get_Ef_from_Et(delay, field)
        trace_center = float(
            getattr(
                self.frg_data,
                "frequency_center",
                _center_bin_value(np.asarray(self.frg_data.frequency, dtype=np.float64)),
            )
        )
        freq_axis = freq_rel + _field_center_frequency(trace_center, getattr(self.frg_data, "geometry", "shg-frog"))

        freq_intensity = _safe_normalize(np.abs(freq_field) ** 2)
        # Convention: Ef = sqrt(I) * exp(-i*phase)  =>  phase = -arg(Ef)
        freq_phase = -np.unwrap(np.angle(freq_field))
        if freq_phase.size:
            freq_phase -= freq_phase[int(np.argmax(freq_intensity))]

        # Wavelength-domain profile from the same spectrum.
        freq_safe = np.where(freq_axis > 1e-12, freq_axis, 1e-12)
        wavelength_nm = SPEED_LIGHT / (freq_safe * 1e15) * 1e9
        order = np.argsort(wavelength_nm)
        wavelength_nm = wavelength_nm[order]
        spec_field = freq_field[order]
        spec_intensity = _safe_normalize(np.abs(spec_field) ** 2)
        spec_phase = -np.unwrap(np.angle(spec_field))
        if spec_phase.size:
            spec_phase -= spec_phase[int(np.argmax(spec_intensity))]

        return {
            "time_axis": delay,
            "time_intensity": time_intensity,
            "time_phase": time_phase,
            "time_field": time_field,
            "freq_axis": freq_axis,
            "freq_intensity": freq_intensity,
            "freq_phase": freq_phase,
            "freq_field": freq_field,
            "wave_axis": wavelength_nm,
            "wave_intensity": spec_intensity,
            "wave_phase": spec_phase,
            "wave_field": spec_field,
        }

    def export_result_payload(self) -> Optional[dict]:
        """Build in-memory result payload for FrogResultGUI."""
        if self.frg_data is None or self.result is None:
            return None
        profiles = self._build_export_profiles(np.asarray(self.result.field, dtype=np.complex128))
        freq_axis = np.asarray(profiles["freq_axis"], dtype=np.float64)
        freq_intensity = np.asarray(profiles["freq_intensity"], dtype=np.float64)
        time_ftl = np.asarray([], dtype=np.float64)
        time_ftl_intensity = np.asarray([], dtype=np.float64)
        pulse_duration_ftl = float("nan")
        if freq_axis.size >= 2 and freq_intensity.size == freq_axis.size:
            ef_ftl = np.sqrt(np.clip(np.asarray(freq_intensity, dtype=np.float64), 0.0, None)).astype(np.complex128)
            time_ftl, et_ftl = get_Et_from_Ef(
                np.asarray(freq_axis, dtype=np.float64),
                ef_ftl,
            )
            time_ftl = np.asarray(time_ftl, dtype=np.float64)
            time_ftl_intensity = _safe_normalize(np.abs(np.asarray(et_ftl, dtype=np.complex128)) ** 2)
            pulse_duration_ftl = _estimate_fwhm(time_ftl, time_ftl_intensity)
        best_g = float(self.result.diagnostics.get("best_error_g", np.nan))
        return {
            "prefix": self._suggest_save_prefix(),
            "frg_delay": np.asarray(self.frg_data.delay, dtype=np.float64).copy(),
            "frg_freq": np.asarray(self.frg_data.frequency, dtype=np.float64).copy(),
            "frg_trace": np.asarray(self.frg_data.trace, dtype=np.float64).copy(),
            "frg_trace_reconstructed": np.asarray(self.result.retrieved_trace, dtype=np.float64).copy(),
            "time": np.asarray(profiles["time_axis"], dtype=np.float64).copy(),
            "time_intensity": np.asarray(profiles["time_intensity"], dtype=np.float64).copy(),
            "time_phase": np.asarray(profiles["time_phase"], dtype=np.float64).copy(),
            "freq": np.asarray(profiles["freq_axis"], dtype=np.float64).copy(),
            "freq_intensity": np.asarray(profiles["freq_intensity"], dtype=np.float64).copy(),
            "freq_phase": np.asarray(profiles["freq_phase"], dtype=np.float64).copy(),
            "wavelength": np.asarray(profiles["wave_axis"], dtype=np.float64).copy(),
            "wavelength_intensity": np.asarray(profiles["wave_intensity"], dtype=np.float64).copy(),
            "wavelength_phase": np.asarray(profiles["wave_phase"], dtype=np.float64).copy(),
            "time_FTL": time_ftl,
            "time_FTL_intensity": time_ftl_intensity,
            "pulse_duration": _estimate_fwhm(
                np.asarray(profiles["time_axis"], dtype=np.float64),
                np.asarray(profiles["time_intensity"], dtype=np.float64),
            ),
            "pulse_duration_FTL": pulse_duration_ftl,
            "best_g": best_g,
            "best_iteration": int(self.result.best_iteration),
            "best_sigma": float(self.result.best_sigma),
            "algorithm": str(self._active_algorithm),
            "source": "retrieval_gui",
        }

    def _on_save_clicked(self) -> None:
        if self.frg_data is None:
            QtWidgets.QMessageBox.warning(self, "No Data", "Please open a .frg or .dat file first.")
            return
        if self.result is None:
            QtWidgets.QMessageBox.warning(self, "No Result", "Please run retrieval before saving.")
            return

        try:
            prefix = self._suggest_save_prefix()
            if not prefix:
                raise ValueError("Save prefix cannot be empty.")
            source_dir = os.path.dirname(os.path.abspath(prefix))
            retrieval_dir = os.path.join(source_dir, "retrieval_result")
            os.makedirs(retrieval_dir, exist_ok=True)
            save_prefix = os.path.join(retrieval_dir, os.path.basename(prefix))

            field = np.asarray(self.result.field, dtype=np.complex128)
            profiles = self._build_export_profiles(field)
            self._save_trace_dat(f"{save_prefix}.A.dat", self.frg_data.delay, self.frg_data.frequency, self.frg_data.trace)
            self._save_trace_dat(
                f"{save_prefix}.Arecon.dat",
                self.frg_data.delay,
                self.frg_data.frequency,
                np.asarray(self.result.retrieved_trace, dtype=np.float64),
            )
            self._save_profile_dat(
                f"{save_prefix}.Ek.dat",
                profiles["time_axis"],
                profiles["time_intensity"],
                profiles["time_phase"],
                profiles["time_field"],
            )
            self._save_profile_dat(
                f"{save_prefix}.Ew.dat",
                profiles["freq_axis"],
                profiles["freq_intensity"],
                profiles["freq_phase"],
                profiles["freq_field"],
            )
            self._save_profile_dat(
                f"{save_prefix}.Speck.dat",
                profiles["wave_axis"],
                profiles["wave_intensity"],
                profiles["wave_phase"],
                profiles["wave_field"],
            )
            self._set_status(
                f"Saved to: {retrieval_dir} | Prefix: {save_prefix}",
                timeout_ms=10000,
            )
        except Exception as e:
            self._set_status(f"Save error: {e}", timeout_ms=12000)
            QtWidgets.QMessageBox.critical(self, "Save Error", str(e))

    def _clear_result_views(self) -> None:
        self._trace_current_raw = None
        self._trace_diff_raw = None
        self._reset_metric_displays()
        if self.frg_data is not None:
            empty = np.zeros_like(np.asarray(self.frg_data.trace, dtype=np.float64))
            self._draw_trace(self.plot_current, self.frg_data.delay, self.frg_data.frequency, empty, "Current trace")
            self._draw_trace(self.plot_diff, self.frg_data.delay, self.frg_data.frequency, empty, "Trace difference")
        else:
            self.plot_current.clear()
            self.plot_current.setTitle("Current trace")
            self.plot_diff.clear()
            self.plot_diff.setTitle("Trace difference")
        self.time_dual.clear("Time profile")
        self.freq_dual.clear("Frequency")
        self._reset_current_g_history()

    def _trace_colormap_for_plot(self, plot_item: pg.PlotItem):
        if plot_item is self.plot_diff:
            return _TRACE_DIFF_COLORMAP
        return _get_trace_colormap()

    def _ensure_trace_layer(self, plot_item: pg.PlotItem) -> dict:
        key = id(plot_item)
        layer = self._trace_layers.get(key)
        if layer is not None:
            return layer

        cmap = self._trace_colormap_for_plot(plot_item)
        img = pg.ImageItem()
        img.setColorMap(cmap)
        plot_item.addItem(img)
        cbar = pg.ColorBarItem(values=(0.0, 1.0), colorMap=cmap, interactive=False, width=12)
        cbar.setImageItem(img, insert_in=plot_item)
        layer = {"img": img, "cbar": cbar}
        self._trace_layers[key] = layer
        return layer

    def _draw_trace(self, plot_item: pg.PlotItem, x: np.ndarray, y: np.ndarray, z: np.ndarray, title: str) -> None:
        layer = self._ensure_trace_layer(plot_item)
        z_arr = np.asarray(z, dtype=np.float64)
        # Log-scale display applies only to original/reconstructed traces, not difference.
        if self.chk_trace_log.isChecked() and plot_item in (self.plot_original, self.plot_current):
            z_arr = np.log10(np.clip(z_arr, self._trace_log_vmin(), None))
        img = layer["img"]
        cbar = layer["cbar"]
        img.setImage(z_arr, autoLevels=False)

        x0, x1 = float(np.min(x)), float(np.max(x))
        y0, y1 = float(np.min(y)), float(np.max(y))
        rect = QtCore.QRectF(min(x0, x1), min(y0, y1), max(abs(x1 - x0), 1e-12), max(abs(y1 - y0), 1e-12))
        img.setRect(rect)

        if plot_item is self.plot_diff:
            vmax = float(max(abs(np.min(z_arr)), abs(np.max(z_arr))))
            vmax = max(vmax, 1e-12)
            levels = (-vmax, vmax)
        else:
            zmin = float(np.min(z_arr))
            zmax = float(np.max(z_arr))
            if zmax - zmin < 1e-12:
                zmax = zmin + 1e-12
            levels = (zmin, zmax)
        img.setLevels(levels)
        cbar.setLevels(values=levels)

        plot_item.setTitle(title)
        plot_item.setLabel("bottom", "Delay (fs)")
        plot_item.setLabel("left", "Frequency (PHz)")

    def _draw_profiles_from_fields(
        self,
        best_field: np.ndarray,
        measured_trace: np.ndarray,
        current_field: Optional[np.ndarray] = None,
    ) -> None:
        if self.frg_data is None:
            return
        delay = np.asarray(self.frg_data.delay, dtype=np.float64)
        best_field = np.asarray(best_field, dtype=np.complex128)
        curr_field = np.asarray(current_field, dtype=np.complex128) if current_field is not None else None

        best_time_int = _safe_normalize(np.abs(best_field) ** 2)
        best_time_phase = _phase_for_display(best_field, best_time_int)
        curr_time_int = _safe_normalize(np.abs(curr_field) ** 2) if curr_field is not None else None
        curr_time_phase = _phase_for_display(curr_field, curr_time_int) if curr_field is not None else None

        fwhm_t = _estimate_fwhm(delay, best_time_int)
        self.time_dual.set_data(
            delay,
            best_time_int,
            best_time_phase,
            curr_time_int,
            curr_time_phase,
            f"Time profile (best G FWHM={fwhm_t:.3g} fs)",
        )

        freq_rel, best_spectrum = get_Ef_from_Et(delay, best_field)
        # SHG-FROG trace axis is around 2*w0; field axis should be centered at w0.
        trace_center = float(
            getattr(
                self.frg_data,
                "frequency_center",
                _center_bin_value(np.asarray(self.frg_data.frequency, dtype=np.float64)),
            )
        )
        freq_center = _field_center_frequency(trace_center, getattr(self.frg_data, "geometry", "shg-frog"))
        freq = freq_rel + freq_center
        best_freq_int = _safe_normalize(np.abs(best_spectrum) ** 2)
        best_freq_phase = _phase_for_display(best_spectrum, best_freq_int)

        if curr_field is not None:
            _, curr_spectrum = get_Ef_from_Et(delay, curr_field)
            curr_freq_int = _safe_normalize(np.abs(curr_spectrum) ** 2)
            curr_freq_phase = _phase_for_display(curr_spectrum, curr_freq_int)
        else:
            curr_freq_int = None
            curr_freq_phase = None

        fwhm_f = _estimate_fwhm(freq_rel, best_freq_int)
        self.freq_dual.set_data(
            freq,
            best_freq_int,
            best_freq_phase,
            curr_freq_int,
            curr_freq_phase,
            f"Frequency (best G FWHM={fwhm_f:.3g} PHz)",
        )

        self._refresh_current_g_plot()


def main() -> None:
    app = QtWidgets.QApplication(sys.argv)
    win = FrogRetrievalGUI()
    win.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()

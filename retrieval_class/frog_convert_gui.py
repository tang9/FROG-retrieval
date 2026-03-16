"""
FROG Convert GUI

Purpose:
- Open raw FROG data and display raw/processed/binned/autocorrelation panels.
- Adjust parameters, process data, save figures, and export binned traces.
- Load and save parameter files.

Parameter files:
- Save: writes the processed figure and matching parameter file.
- Save Param: writes current parameters to a user-selected file.
- On startup, if `retrieval_class/frog_convert_default_param.txt` exists,
  it is loaded automatically.

writer: Tang, 2026-02
"""

import sys
import os
import json
os.environ.setdefault("QT_API", "pyside6")
import matplotlib
matplotlib.use("QtAgg")
import numpy as np
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.colors import LogNorm
from matplotlib.figure import Figure
from PySide6 import QtCore, QtWidgets
import pyqtgraph as pg
from .common import custom_cmap, fit_peak, gaussian_function
from .frog_convert import FROG
class Show1DDialog(QtWidgets.QDialog):
    def __init__(self, gui_parent):
        super().__init__(gui_parent)
        self.gui_parent = gui_parent
        self.setWindowTitle("Show 1D")
        self.resize(700, 750)

        layout = QtWidgets.QVBoxLayout(self)
        ctrl = QtWidgets.QHBoxLayout()
        layout.addLayout(ctrl)

        ctrl.addWidget(QtWidgets.QLabel("delay"))
        self.delay_combo = QtWidgets.QComboBox()
        ctrl.addWidget(self.delay_combo)

        ctrl.addWidget(QtWidgets.QLabel("wavelength"))
        self.wave_combo = QtWidgets.QComboBox()
        ctrl.addWidget(self.wave_combo)

        ctrl.addStretch(1)

        self.pg_widget = pg.GraphicsLayoutWidget()
        self.pg_widget.setBackground("w")
        layout.addWidget(self.pg_widget, 1)
        self.plot_spec = self.pg_widget.addPlot(row=0, col=0, title="Spectrum Slice (wavelength axis)")
        self.plot_spec.getAxis("left").setTextPen("k")
        self.plot_spec.getAxis("bottom").setTextPen("k")
        self.plot_spec.getAxis("left").setPen("k")
        self.plot_spec.getAxis("bottom").setPen("k")
        self.plot_spec.showGrid(x=True, y=True, alpha=0.25)
        self.plot_spec.addLegend()
        self.plot_spec.setLabel("bottom", "Wavelength (nm)")
        self.plot_spec.setLabel("left", "Intensity")

        self.plot_delay = self.pg_widget.addPlot(row=1, col=0, title="Delay Slice (delay axis)")
        self.plot_delay.getAxis("left").setTextPen("k")
        self.plot_delay.getAxis("bottom").setTextPen("k")
        self.plot_delay.getAxis("left").setPen("k")
        self.plot_delay.getAxis("bottom").setPen("k")
        self.plot_delay.showGrid(x=True, y=True, alpha=0.25)
        self.plot_delay.addLegend()
        self.plot_delay.setLabel("bottom", "Delay (fs)")
        self.plot_delay.setLabel("left", "Intensity")

        self.delay_combo.currentIndexChanged.connect(self.redraw)
        self.wave_combo.currentIndexChanged.connect(self.redraw)

    def refresh_options(self):
        raw = self.gui_parent.raw_obj
        if raw is None or np.size(getattr(raw, "frog_trace", [])) == 0:
            return

        prev_delay = self.delay_combo.currentText()
        prev_wave = self.wave_combo.currentText()

        self.delay_combo.blockSignals(True)
        self.wave_combo.blockSignals(True)
        self.delay_combo.clear()
        self.wave_combo.clear()
        for i, d in enumerate(np.array(raw.delay).ravel()):
            self.delay_combo.addItem(f"{i}: {float(d):.3f} fs")
        for j, w in enumerate(np.array(raw.wavelength).ravel()):
            self.wave_combo.addItem(f"{j}: {float(w):.3f} nm")
        self.delay_combo.blockSignals(False)
        self.wave_combo.blockSignals(False)

        def restore_or_default(combo, prev_text):
            if prev_text:
                idx = combo.findText(prev_text)
                if idx >= 0:
                    combo.setCurrentIndex(idx)
                    return
            if combo.count() > 0:
                combo.setCurrentIndex(combo.count() // 2)

        restore_or_default(self.delay_combo, prev_delay)
        restore_or_default(self.wave_combo, prev_wave)
        self.redraw()

    def clear_view(self):
        self.delay_combo.blockSignals(True)
        self.wave_combo.blockSignals(True)
        self.delay_combo.clear()
        self.wave_combo.clear()
        self.delay_combo.blockSignals(False)
        self.wave_combo.blockSignals(False)
        self.plot_spec.clear()
        self.plot_delay.clear()
        self.plot_spec.setTitle("No raw data")
        self.plot_delay.setTitle("No raw data")

    def _nearest_idx(self, arr, value):
        a = np.asarray(arr, dtype=float).ravel()
        return int(np.argmin(np.abs(a - float(value))))

    def redraw(self):
        raw = self.gui_parent.raw_obj
        self.plot_spec.clear()
        self.plot_delay.clear()
        if raw is None or np.size(getattr(raw, "frog_trace", [])) == 0:
            self.plot_spec.setTitle("No raw data")
            self.plot_delay.setTitle("No raw data")
            return

        i_d = max(0, self.delay_combo.currentIndex())
        i_w = max(0, self.wave_combo.currentIndex())
        i_d = min(i_d, len(raw.delay) - 1)
        i_w = min(i_w, len(raw.wavelength) - 1)

        delay_val = float(raw.delay[i_d])
        wave_val = float(raw.wavelength[i_w])

        self.plot_spec.plot(
            np.asarray(raw.wavelength, dtype=float),
            np.asarray(raw.frog_trace[i_d, :], dtype=float),
            pen=pg.mkPen((30, 100, 220), width=1.2),
            symbol="o",
            symbolSize=4,
            symbolBrush=(30, 100, 220),
            name=f"Raw @ delay={delay_val:.2f} fs",
        )
        self.plot_delay.plot(
            np.asarray(raw.delay, dtype=float),
            np.asarray(raw.frog_trace[:, i_w], dtype=float),
            pen=pg.mkPen((30, 100, 220), width=1.2),
            symbol="o",
            symbolSize=4,
            symbolBrush=(30, 100, 220),
            name=f"Raw @ wl={wave_val:.2f} nm",
        )

        proc = self.gui_parent.proc_obj
        if proc is not None and np.size(getattr(proc, "frog_trace", [])) > 0:
            proc_trace = getattr(proc, "frog_trace_pre_norm", None)
            if proc_trace is None or np.size(proc_trace) == 0:
                proc_trace = proc.frog_trace
            proc_delay = np.array(proc.delay)
            proc_wave = np.array(proc.wavelength)
            pi_d = self._nearest_idx(proc_delay, delay_val)
            pi_w = self._nearest_idx(proc_wave, wave_val)
            self.plot_spec.plot(
                np.asarray(proc_wave, dtype=float),
                np.asarray(proc_trace[pi_d, :], dtype=float),
                pen=pg.mkPen((220, 70, 50), width=1.6),
                name=f"Processed @ delay={float(proc_delay[pi_d]):.2f} fs",
            )
            self.plot_delay.plot(
                np.asarray(proc_delay, dtype=float),
                np.asarray(proc_trace[:, pi_w], dtype=float),
                pen=pg.mkPen((220, 70, 50), width=1.6),
                name=f"Processed @ wl={float(proc_wave[pi_w]):.2f} nm",
            )
        self.plot_spec.setTitle("Spectrum Slice (wavelength axis)")
        self.plot_delay.setTitle("Delay Slice (delay axis)")


class FROGConvertGUI(QtWidgets.QMainWindow):
    retrieval_data_ready = QtCore.Signal(object)
    DEFAULT_PARAM_FILE = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "frog_convert_default_param.txt",
    )

    def __init__(self):
        super().__init__()
        self.setWindowTitle("FROG Convert GUI (PySide6)")
        self.resize(900, 700)

        self.raw_obj = None
        self.proc_obj = None
        self._raw_data_dict = None
        self._cbars = {}
        self.last_dir = os.getcwd()
        self.one_d_dialog = None

        self._build_ui()
        self.statusBar().showMessage("Ready", 0)
        self._autoload_default_params()

    def _set_status(self, text, timeout_ms=0):
        self.statusBar().showMessage(str(text), int(timeout_ms))

    def _status_info(self, text, timeout_ms=0):
        self._set_status(f"INFO: {text}", timeout_ms)

    def _status_warn(self, text, timeout_ms=0):
        self._set_status(f"WARN: {text}", timeout_ms)

    def _status_error(self, text, timeout_ms=0):
        self._set_status(f"ERROR: {text}", timeout_ms)

    def _reset_cached_state(self):
        self.raw_obj = None
        self.proc_obj = None
        self._raw_data_dict = None
        self.redraw()
        if hasattr(self, "one_d_dialog") and self.one_d_dialog is not None:
            self.one_d_dialog.clear_view()

    def _build_ui(self):
        central = QtWidgets.QWidget()
        self.setCentralWidget(central)
        root_layout = QtWidgets.QHBoxLayout(central)
        root_layout.setContentsMargins(2, 2, 2, 2)
        root_layout.setSpacing(2)
        split = QtWidgets.QSplitter(QtCore.Qt.Orientation.Horizontal)
        split.setHandleWidth(2)
        split.setChildrenCollapsible(False)
        root_layout.addWidget(split, 1)

        left_panel = QtWidgets.QWidget()
        left_panel.setFixedWidth(300)
        left_layout = QtWidgets.QVBoxLayout(left_panel)
        left_layout.setContentsMargins(2, 2, 2, 2)
        left_layout.setSpacing(2)
        split.addWidget(left_panel)

        right_panel = QtWidgets.QWidget()
        right_layout = QtWidgets.QVBoxLayout(right_panel)
        right_layout.setContentsMargins(2, 2, 2, 2)
        right_layout.setSpacing(2)
        split.addWidget(right_panel)
        split.setStretchFactor(0, 0)
        split.setStretchFactor(1, 1)
        QtCore.QTimer.singleShot(0, lambda: split.setSizes([250, 10_000]))

        params_box = QtWidgets.QGroupBox("Parameters")
        left_layout.addWidget(params_box)
        g = QtWidgets.QGridLayout(params_box)
        g.setHorizontalSpacing(8)
        g.setVerticalSpacing(6)
        for col in range(2):
            g.setColumnStretch(col, 1)

        def add_pair(row, text, widget):
            g.addWidget(QtWidgets.QLabel(text), row, 0)
            g.addWidget(widget, row, 1)

        def add_check(row, checkbox):
            g.addWidget(checkbox, row, 0, 1, 2)

        self.range_enable = QtWidgets.QCheckBox("wavelength_range")
        self.range_enable.setChecked(False)
        self.wavelength_min = QtWidgets.QLineEdit("200")
        self.wavelength_max = QtWidgets.QLineEdit("2600")
        add_check(0, self.range_enable)

        self.wavelength_bin = QtWidgets.QLineEdit("1")
        self.noise_filter = QtWidgets.QComboBox()
        self.noise_filter.addItems(["", "Gaussian", "Low pass"])
        self.sigma = QtWidgets.QLineEdit("1")
        self.cutoff = QtWidgets.QLineEdit("8")
        self.constant_bkg = QtWidgets.QLineEdit("0")
        self.filter_axis = QtWidgets.QComboBox()
        self.filter_axis.addItems(["wavelength", "delay", "both"])
        add_pair(1, "wavelength_min", self.wavelength_min)
        add_pair(2, "wavelength_max", self.wavelength_max)
        add_pair(3, "wavelength_bin", self.wavelength_bin)
        add_pair(4, "noise_filter", self.noise_filter)
        add_pair(5, "filter_axis", self.filter_axis)
        add_pair(6, "sigma (Gaussian)", self.sigma)
        add_pair(7, "cutoff ratio (Low pass)", self.cutoff)

        self.delay_step = QtWidgets.QComboBox()
        self.delay_step.addItems(["fs", "mm", "nm", "um", "m", "ps", "as"])
        self.delay_correction = QtWidgets.QLineEdit("1.0")
        self.edge = QtWidgets.QLineEdit("20")
        self.subx = QtWidgets.QCheckBox("subtract bkg (delay)")
        self.subx.setChecked(True)
        self.suby = QtWidgets.QCheckBox("subtract bkg (wavelength)")
        self.suby.setChecked(False)
        self.corner_suppression = QtWidgets.QCheckBox("corner_suppression")
        self.mask_frg = QtWidgets.QLineEdit("")
        self.mask_frg.setPlaceholderText("None")
        self.time_zero = QtWidgets.QCheckBox("time_zero")
        self.time_zero.setChecked(True)
        add_pair(8, "delay_step", self.delay_step)
        add_pair(9, "delay_correction", self.delay_correction)
        add_pair(10, "edge pixels", self.edge)
        add_pair(11, "constant_bkg", self.constant_bkg)
        add_pair(12, "mask_frg", self.mask_frg)
        add_check(13, self.time_zero)
        add_check(14, self.corner_suppression)
        add_check(15, self.subx)
        add_check(16, self.suby)

        self.time_enable = QtWidgets.QCheckBox("time_range")
        self.time_enable.setChecked(False)
        self.time_min = QtWidgets.QLineEdit("-900")
        self.time_max = QtWidgets.QLineEdit("900")
        self.N = QtWidgets.QLineEdit("256")
        add_check(17, self.time_enable)
        add_pair(18, "time_min", self.time_min)
        add_pair(19, "time_max", self.time_max)
        add_pair(20, "N", self.N)

        self.save_default_param_btn = QtWidgets.QPushButton("Save Default Param")
        self.save_default_param_btn.clicked.connect(self.save_default_params)
        left_layout.addWidget(self.save_default_param_btn)
        left_layout.addStretch(1)

        file_layout = QtWidgets.QHBoxLayout()
        right_layout.addLayout(file_layout)
        file_layout.addWidget(QtWidgets.QLabel("File"))
        self.file_edit = QtWidgets.QLineEdit()
        file_layout.addWidget(self.file_edit, 1)
        self.open_btn = QtWidgets.QPushButton("Open")
        self.reload_btn = QtWidgets.QPushButton("Reload")
        self.process_btn = QtWidgets.QPushButton("Run convert")
        self.process_btn.setStyleSheet("background-color: #90EE90;")
        self.load_param_btn = QtWidgets.QPushButton("Load Param")
        self.save_param_btn = QtWidgets.QPushButton("Save Param")
        self.save_btn = QtWidgets.QPushButton("Save Convert")
        file_layout.addWidget(self.open_btn)
        file_layout.addWidget(self.reload_btn)
        self.open_btn.clicked.connect(self.open_file)
        self.reload_btn.clicked.connect(self.reload_raw)
        self.process_btn.clicked.connect(self.process)
        self.load_param_btn.clicked.connect(self.load_params_file)
        self.save_param_btn.clicked.connect(self.save_params_file)
        self.save_btn.clicked.connect(self.save_results)

        logs_layout = QtWidgets.QHBoxLayout()
        right_layout.addLayout(logs_layout)
        self.trace_log = QtWidgets.QCheckBox("Trace log scale")
        self.trace_log.setChecked(False)
        self.show_1d_btn = QtWidgets.QPushButton("show 1d")
        logs_layout.addWidget(self.trace_log)
        logs_layout.addWidget(self.show_1d_btn)
        logs_layout.addWidget(self.load_param_btn)
        logs_layout.addWidget(self.save_param_btn)
        logs_layout.addWidget(self.process_btn)
        logs_layout.addWidget(self.save_btn)
        #logs_layout.addStretch(1)
        self.trace_log.toggled.connect(self.redraw)
        self.show_1d_btn.clicked.connect(self.show_1d_dialog)

        self.fig = Figure(figsize=(11, 7), dpi=100)
        self.ax_raw = self.fig.add_subplot(221)
        self.ax_frog = self.fig.add_subplot(222)
        self.ax_frg = self.fig.add_subplot(223)
        self.ax_auto = self.fig.add_subplot(224)
        self.canvas = FigureCanvasQTAgg(self.fig)
        right_layout.addWidget(self.canvas, 1)

    def _read_range(self, enable_box: QtWidgets.QCheckBox, min_edit: QtWidgets.QLineEdit, max_edit: QtWidgets.QLineEdit):
        if not enable_box.isChecked():
            return None
        return float(min_edit.text().strip()), float(max_edit.text().strip())

    def _collect_params(self):
        mask_text = self.mask_frg.text().strip()
        mask_frg = None if mask_text == "" else float(mask_text)
        constant_text = self.constant_bkg.text().strip()
        constant_bkg = float(constant_text) if constant_text else 0.0
        return {
            "wavelength_range": self._read_range(self.range_enable, self.wavelength_min, self.wavelength_max),
            "wavelength_bin": int(self.wavelength_bin.text().strip()),
            "noise_filter": self.noise_filter.currentText(),
            "sigma": float(self.sigma.text().strip()),
            "cutoff": float(self.cutoff.text().strip()),
            "constant_bkg": constant_bkg,
            "filter_axis": self.filter_axis.currentText(),
            "delay_step": self.delay_step.currentText(),  # default fs
            "delay_correction": float(self.delay_correction.text().strip()),
            "edge": int(self.edge.text().strip()),
            "subx": self.subx.isChecked(),
            "suby": self.suby.isChecked(),
            "corner_suppression": self.corner_suppression.isChecked(),
            "mask_frg": mask_frg,
            "time_zero": self.time_zero.isChecked(),
            "time_range": self._read_range(self.time_enable, self.time_min, self.time_max),
            "N": int(self.N.text().strip()),
        }

    def _apply_params(self, p):
        if "wavelength_range" in p:
            wr = p["wavelength_range"]
            self.range_enable.setChecked(wr is not None)
            if wr is not None and len(wr) == 2:
                self.wavelength_min.setText(str(wr[0]))
                self.wavelength_max.setText(str(wr[1]))
        if "wavelength_bin" in p:
            self.wavelength_bin.setText(str(p["wavelength_bin"]))
        if "noise_filter" in p:
            idx = self.noise_filter.findText(str(p["noise_filter"]))
            if idx >= 0:
                self.noise_filter.setCurrentIndex(idx)
        if "sigma" in p:
            self.sigma.setText(str(p["sigma"]))
        if "cutoff" in p:
            self.cutoff.setText(str(p["cutoff"]))
        if "constant_bkg" in p:
            self.constant_bkg.setText(str(p["constant_bkg"]))
        if "filter_axis" in p:
            idx = self.filter_axis.findText(str(p["filter_axis"]))
            if idx >= 0:
                self.filter_axis.setCurrentIndex(idx)
        if "delay_step" in p:
            idx = self.delay_step.findText(str(p["delay_step"]))
            if idx >= 0:
                self.delay_step.setCurrentIndex(idx)
        if "delay_correction" in p:
            self.delay_correction.setText(str(p["delay_correction"]))
        if "edge" in p:
            self.edge.setText(str(p["edge"]))
        if "subx" in p:
            self.subx.setChecked(bool(p["subx"]))
        if "suby" in p:
            self.suby.setChecked(bool(p["suby"]))
        if "corner_suppression" in p:
            self.corner_suppression.setChecked(bool(p["corner_suppression"]))
        if "mask_frg" in p:
            mf = p["mask_frg"]
            self.mask_frg.setText("" if mf is None else str(mf))
        if "time_zero" in p:
            self.time_zero.setChecked(bool(p["time_zero"]))
        if "time_range" in p:
            tr = p["time_range"]
            self.time_enable.setChecked(tr is not None)
            if tr is not None and len(tr) == 2:
                self.time_min.setText(str(tr[0]))
                self.time_max.setText(str(tr[1]))
        if "N" in p:
            self.N.setText(str(p["N"]))

    def _default_param_path(self):
        return self.DEFAULT_PARAM_FILE

    def _load_params_from_path(self, param_path):
        with open(param_path, "r", encoding="utf-8") as f:
            p = json.load(f)
        if not isinstance(p, dict):
            raise ValueError("Parameter file must be a JSON object.")
        self._apply_params(p)
        self._status_info(f"Loaded parameters: {param_path}")

    def _autoload_default_params(self):
        param_path = self._default_param_path()
        if not os.path.isfile(param_path):
            return
        try:
            self._load_params_from_path(param_path)
        except Exception as e:
            self._status_warn(f"Default param load failed: {param_path} ({e})")

    def load_params_file(self):
        param_path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, "Load parameter file", self.last_dir, "Text/JSON files (*.txt *.json);;All files (*)"
        )
        if not param_path:
            return
        try:
            self._update_last_dir_from_path(param_path)
            self._load_params_from_path(param_path)
        except Exception as e:
            self._status_error(f"Load param error: {e}")

    def save_params_file(self):
        try:
            if self.proc_obj is not None and getattr(self.proc_obj, "prefix", ""):
                prefix = str(self.proc_obj.prefix).strip()
            else:
                file_path = self.file_edit.text().strip()
                if file_path:
                    prefix = os.path.splitext(os.path.abspath(file_path))[0]
                else:
                    prefix = os.path.join(self.last_dir, "frog_convert")
            source_dir = os.path.dirname(os.path.abspath(prefix))
            retrieval_dir = os.path.join(source_dir, "retrieval_result")
            os.makedirs(retrieval_dir, exist_ok=True)
            save_prefix = os.path.join(retrieval_dir, os.path.basename(prefix))
            param_path = f"{save_prefix}_param.json"
            with open(param_path, "w", encoding="utf-8") as f:
                json.dump(self._collect_params(), f, indent=2, ensure_ascii=True)
            self.last_dir = retrieval_dir
            self._status_info(f"Saved parameters: {param_path}")
        except Exception as e:
            self._status_error(f"Save param error: {e}")

    def save_default_params(self):
        try:
            param_path = self._default_param_path()
            with open(param_path, "w", encoding="utf-8") as f:
                json.dump(self._collect_params(), f, indent=2, ensure_ascii=True)
            self._status_info(f"Saved default parameters: {param_path}")
        except Exception as e:
            self._status_error(f"Save default param error: {e}")

    def open_file(self):
        open_dir = self._open_dialog_dir()
        file_path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, "Open FROG file", open_dir, "All files (*)"
        )
        if not file_path:
            return
        self._update_last_dir_from_path(file_path)
        self.file_edit.setText(file_path)
        self.load_raw()

    def _open_dialog_dir(self):
        folder = os.path.abspath(self.last_dir)
        probe = folder
        while True:
            if os.path.basename(probe) == "retrieval_result":
                parent = os.path.dirname(probe)
                return parent if os.path.isdir(parent) else folder
            parent = os.path.dirname(probe)
            if parent == probe:
                return folder
            probe = parent

    def _update_last_dir_from_path(self, path):
        folder = os.path.dirname(os.path.abspath(path))
        if os.path.isdir(folder):
            self.last_dir = folder

    def reload_raw(self):
        """Reload raw data: re-read from file if path is valid, otherwise rebuild from in-memory dict."""
        path = self.file_edit.text().strip()
        if path and os.path.isfile(path):
            self.load_raw()
        elif self._raw_data_dict is not None:
            self._reload_from_dict()
        else:
            self.load_raw()  # will show appropriate error

    def _reload_from_dict(self):
        """Rebuild raw_obj from the stored _raw_data_dict."""
        d = self._raw_data_dict
        old_proc = self.proc_obj
        self.raw_obj = None
        self.proc_obj = None
        try:
            p = self._collect_params()
            self.raw_obj = FROG(
                d,
                wavelength_range=p["wavelength_range"],
                wavelength_bin=p["wavelength_bin"],
                delay_step=p["delay_step"],
                delay_correction=p["delay_correction"],
                edge=p["edge"],
                subx=p["subx"],
                suby=p["suby"],
                time_zero=p["time_zero"],
                time_range=p["time_range"],
                raw=True,
            )
            self.redraw()
            if hasattr(self, "one_d_dialog") and self.one_d_dialog is not None:
                self.one_d_dialog.refresh_options()
            label = d.get("prefix", "in-memory")
            self._status_info(f"Reloaded: {label}")
        except Exception as e:
            self.raw_obj = None
            self.proc_obj = old_proc
            self._status_error(f"Reload error: {e}")

    def load_raw(self):
        path = self.file_edit.text().strip()
        if not path:
            self._status_warn("Missing file: please choose a file first.")
            return
        self._reset_cached_state()
        try:
            p = self._collect_params()
            self.raw_obj = FROG(
                path,
                wavelength_range=p["wavelength_range"],
                wavelength_bin=p["wavelength_bin"],
                delay_step=p["delay_step"],
                delay_correction=p["delay_correction"],
                edge=p["edge"],
                subx=p["subx"],
                suby=p["suby"],
                time_zero=p["time_zero"],
                time_range=p["time_range"],
                raw=True,
            )
            raw_delay = np.asarray(getattr(self.raw_obj, "delay", []), dtype=float).ravel()
            raw_wave = np.asarray(getattr(self.raw_obj, "wavelength", []), dtype=float).ravel()
            raw_trace = np.asarray(getattr(self.raw_obj, "frog_trace", []), dtype=float)
            if raw_trace.ndim != 2 or raw_trace.size == 0:
                raise ValueError("No raw trace loaded")
            if raw_trace.shape == (raw_delay.size, raw_wave.size):
                pass
            elif raw_trace.shape == (raw_wave.size, raw_delay.size):
                raw_trace = raw_trace.T
                self.raw_obj.frog_trace = raw_trace
            else:
                raise ValueError(
                    f"Raw data shape mismatch: trace={raw_trace.shape}, "
                    f"delay={raw_delay.size}, wavelength={raw_wave.size}"
                )
            self._raw_data_dict = {
                "delay": raw_delay.copy(),
                "wavelength": raw_wave.copy(),
                "frog_trace": raw_trace.copy(),
                "prefix": str(getattr(self.raw_obj, "prefix", "frog")),
            }
            self.redraw()
            if hasattr(self, "one_d_dialog") and self.one_d_dialog is not None:
                self.one_d_dialog.refresh_options()
            self._status_info(f"Raw loaded: {path}")
        except Exception as e:
            self._reset_cached_state()
            self._status_error(f"Load error: {e}")

    def load_simulate_payload(self, payload: dict) -> None:
        """Load a simulated FROG trace dict (delay fs, wavelength nm, frog_trace)."""
        if not isinstance(payload, dict):
            raise ValueError("Payload must be a dict.")
        delay = np.asarray(payload["delay"], dtype=float).ravel()
        wavelength = np.asarray(payload["wavelength"], dtype=float).ravel()
        trace = np.asarray(payload["frog_trace"], dtype=float)
        if trace.shape != (delay.size, wavelength.size):
            raise ValueError(f"Trace shape {trace.shape} vs delay {delay.size} x wavelength {wavelength.size}")
        self._reset_cached_state()
        self._raw_data_dict = {
            "delay": delay.copy(),
            "wavelength": wavelength.copy(),
            "frog_trace": trace.copy(),
            "prefix": "simulated",
        }
        self.raw_obj = FROG(
            self._raw_data_dict,
            delay_step="fs",
            delay_correction=1.0,
            raw=True,
        )
        self.file_edit.setText(f"[simulated, {delay.size} pts]")
        self.redraw()
        if hasattr(self, "one_d_dialog") and self.one_d_dialog is not None:
            self.one_d_dialog.refresh_options()
        self._status_info("Loaded simulated FROG trace from Simulate tab.")

    def process(self):
        if self._raw_data_dict is None:
            self._status_warn("No raw data loaded: please Open or Reload first.")
            return
        try:
            p = self._collect_params()
            self.proc_obj = FROG(
                self._raw_data_dict,
                wavelength_range=p["wavelength_range"],
                wavelength_bin=p["wavelength_bin"],
                noise_filter=p["noise_filter"],
                sigma=p["sigma"],
                cutoff=p["cutoff"],
                constant_bkg=p["constant_bkg"],
                filter_axis=p["filter_axis"],
                delay_step=p["delay_step"],
                delay_correction=p["delay_correction"],
                edge=p["edge"],
                subx=p["subx"],
                suby=p["suby"],
                corner_suppression=p["corner_suppression"],
                mask_frg=p["mask_frg"],
                time_zero=p["time_zero"],
                time_range=p["time_range"],
                N=p["N"],
                profile=True,
            )
            self.redraw()
            if hasattr(self, "one_d_dialog") and self.one_d_dialog is not None:
                self.one_d_dialog.redraw()
            payload = self.export_retrieval_payload()
            if payload is not None:
                self.retrieval_data_ready.emit(payload)
            label = self._raw_data_dict.get("prefix", "frog")
            profile_timing = getattr(self.proc_obj, "profile_timing", [])
            if profile_timing:
                top3 = sorted(profile_timing, key=lambda t: t[1], reverse=True)[:3]
                top3_text = ", ".join([f"{k}:{v:.2f}s" for k, v in top3])
                self._status_info(f"Processed: {label} | Top steps: {top3_text}")
            else:
                self._status_info(f"Processed: {label}")
        except Exception as e:
            self._status_error(f"Process error: {e}")

    def export_retrieval_payload(self):
        """Build an in-memory FRG payload for downstream retrieval GUI."""
        if self.proc_obj is None:
            return None

        delay = np.asarray(getattr(self.proc_obj, "delay_binned", []), dtype=float).ravel()
        freq = np.asarray(getattr(self.proc_obj, "freq_binned", []), dtype=float).ravel()
        trace = np.asarray(getattr(self.proc_obj, "frg", []), dtype=float)
        if delay.size < 2 or freq.size < 2 or trace.size == 0:
            return None

        if trace.shape != (delay.size, freq.size):
            return None

        trace = np.nan_to_num(trace, nan=0.0, posinf=0.0, neginf=0.0)
        trace = np.clip(trace, 0.0, None)
        zmax = float(np.max(trace))
        if zmax > 1e-15:
            trace = trace / zmax

        freq_center = float(getattr(self.proc_obj, "v0", freq[freq.size // 2]))
        prefix = str(getattr(self.proc_obj, "prefix", "frog")).strip() or "frog"
        n = int(getattr(self.proc_obj, "frg_bin_size", delay.size))
        path = f"{prefix}_binned{n}.frg"
        return {
            "path": path,
            "delay": delay.copy(),
            "frequency": freq.copy(),
            "frequency_center": freq_center,
            "trace": trace.copy(),
            "trace_order": "delay_frequency",
            "source": "convert_gui",
        }

    def show_1d_dialog(self):
        if self.raw_obj is None or np.size(getattr(self.raw_obj, "frog_trace", [])) == 0:
            self._status_warn("No raw data: please click Open/Reload first.")
            return
        if not hasattr(self, "one_d_dialog") or self.one_d_dialog is None:
            self.one_d_dialog = Show1DDialog(self)
        self.one_d_dialog.refresh_options()
        self.one_d_dialog.show()
        self.one_d_dialog.raise_()
        self.one_d_dialog.activateWindow()

    def _safe_remove_cbar(self, ax):
        cbar = self._cbars.pop(ax, None)
        if cbar is None:
            return
        # Matplotlib can occasionally fail inside cbar.remove() during repeated relayout.
        # Fall back to removing the colorbar axes directly.
        try:
            cbar.remove()
        except Exception:
            try:
                if cbar.ax in self.fig.axes:
                    cbar.ax.remove()
            except Exception:
                pass

    def _plot_2d(self, ax, x, y, z, title, xlabel, ylabel, logscale=False, log_vmin=None):
        # Remove old colorbar for this axes before drawing a new one.
        self._safe_remove_cbar(ax)

        ax.clear()
        if z is None or np.size(z) == 0:
            ax.set_title(title + " (empty)")
            ax.set_xlabel(xlabel)
            ax.set_ylabel(ylabel)
            return
        zz = np.array(z, dtype=float).T
        xx, yy = np.meshgrid(x, y, indexing="xy")
        if logscale:
            pos = zz[zz > 0]
            if pos.size > 0:
                if log_vmin is None:
                    vmin = max(np.min(pos), 1e-12)
                else:
                    vmin = max(float(log_vmin), 1e-12)
                im = ax.pcolormesh(
                    xx, yy, zz, shading="nearest", cmap=custom_cmap,
                    norm=LogNorm(vmin=vmin, vmax=np.max(pos))
                )
            else:
                im = ax.pcolormesh(xx, yy, zz, shading="nearest", cmap=custom_cmap)
        else:
            im = ax.pcolormesh(xx, yy, zz, shading="nearest", cmap=custom_cmap)
        self._cbars[ax] = self.fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
        ax.set_title(title)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)

    def redraw(self):
        # Keep colorbar cleanup robust across frequent reload/process redraw calls.
        for _ax in (self.ax_raw, self.ax_frog, self.ax_frg):
            self._safe_remove_cbar(_ax)

        self.ax_raw.clear()
        self.ax_frog.clear()
        self.ax_frg.clear()
        self.ax_auto.clear()

        if self.raw_obj is not None and np.size(self.raw_obj.frog_trace) > 0:
            raw_title = getattr(self.raw_obj, "prefix", "Raw FROG Trace")
            raw_title = raw_title.split("/")[-1].split("\\")[-1]
            self._plot_2d(
                self.ax_raw,
                self.raw_obj.delay,
                self.raw_obj.wavelength,
                self.raw_obj.frog_trace,
                raw_title,
                "Delay (fs)",
                "Wavelength (nm)",
                self.trace_log.isChecked(),
                log_vmin=np.max(self.raw_obj.frog_trace) * 1e-4,
            )
        else:
            self.ax_raw.set_title("Raw FROG Trace")

        if self.proc_obj is not None and np.size(self.proc_obj.frog_trace) > 0:
            self._plot_2d(
                self.ax_frog,
                self.proc_obj.delay,
                self.proc_obj.wavelength,
                self.proc_obj.frog_trace,
                "Processed FROG Trace",
                "Delay (fs)",
                "Wavelength (nm)",
                self.trace_log.isChecked(),
                log_vmin=1e-4,
            )
            mask_thr = 0.0
            if self.proc_obj is not None:
                try:
                    _p = self._collect_params()
                    mask_val = _p.get("mask_frg", None)
                    mask_thr = float(mask_val) if mask_val is not None else 0.0
                except Exception:
                    mask_thr = 0.0
            self._plot_2d(
                self.ax_frg,
                self.proc_obj.delay_binned,
                self.proc_obj.freq_binned,
                self.proc_obj.frg,
                "Binned Trace",
                "Delay (fs)",
                "Frequency (PHz)",
                self.trace_log.isChecked(),
                log_vmin=max(mask_thr, 1e-4),
            )
            auto = np.array(self.proc_obj.autocorrelation, dtype=float)
            if auto.size > 0 and np.max(auto) > 0:
                auto_norm = auto / np.max(auto)
                self.ax_auto.plot(self.proc_obj.delay, auto_norm, ".-", lw=1, label="Autocorrelation")
                try:
                    param, _ = fit_peak(np.asarray(self.proc_obj.delay, dtype=float), auto_norm)
                    param = np.asarray(param, dtype=float)
                    if param.size == 3 and np.isfinite(param[2]) and abs(param[2]) > 0:
                        fit_y = gaussian_function(np.asarray(self.proc_obj.delay, dtype=float), param[0], param[1], param[2])
                        fwhm = param[2] * 2.35482
                        self.ax_auto.plot(
                            self.proc_obj.delay,
                            fit_y,
                            "-",
                            lw=1.2,
                            label=f"Gaussian fit (FWHM={fwhm:.1f} fs)",
                        )
                except Exception:
                    pass
            else:
                self.ax_auto.plot(self.proc_obj.delay, auto, ".-", lw=1, label="Autocorrelation")
            self.ax_auto.set_title("Autocorrelation")
            self.ax_auto.set_xlabel("Delay (fs)")
            self.ax_auto.set_ylabel("Intensity")
            self.ax_auto.grid(True, axis="x", alpha=0.3)
            self.ax_auto.legend()
        else:
            self.ax_frog.set_title("Processed FROG Trace")
            self.ax_frg.set_title("Binned Trace")
            self.ax_auto.set_title("Autocorrelation")

        self.fig.tight_layout()
        self.canvas.draw_idle()

    def save_results(self):
        if self.proc_obj is None or np.size(getattr(self.proc_obj, "frog_trace", [])) == 0:
            self._status_warn("No processed data: please click Process first.")
            return
        try:
            prefix = self.proc_obj.prefix
            source_dir = os.path.dirname(os.path.abspath(prefix))
            retrieval_dir = os.path.join(source_dir, "retrieval_result")
            os.makedirs(retrieval_dir, exist_ok=True)
            save_prefix = os.path.join(retrieval_dir, os.path.basename(prefix))
            N = self.proc_obj.frg_bin_size
            fig_path = f"{save_prefix}_processed_N{N}.png"
            param_path = f"{save_prefix}_processed_N{N}.txt"
            self.fig.savefig(fig_path, dpi=200, bbox_inches="tight")
            with open(param_path, "w", encoding="utf-8") as f:
                json.dump(self._collect_params(), f, indent=2, ensure_ascii=True)
            old_prefix = self.proc_obj.prefix
            self.proc_obj.prefix = save_prefix
            try:
                self.proc_obj.output_binned()
            finally:
                self.proc_obj.prefix = old_prefix
            self._status_info(
                f"Saved to: {retrieval_dir} | Figure: {fig_path} | Params: {param_path} | Binned file exported."
            )
        except Exception as e:
            self._status_error(f"Save error: {e}")


def main():
    app = QtWidgets.QApplication(sys.argv)
    win = FROGConvertGUI()
    win.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()

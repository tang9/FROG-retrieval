import os
import re
import sys
import importlib

os.environ.setdefault("QT_API", "pyside6")

import matplotlib
matplotlib.use("QtAgg")
import numpy as np
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.figure import Figure
from matplotlib.colors import LogNorm, Normalize
from PySide6 import QtWidgets, QtCore

from .frog_result import FROG_result
from .common import custom_cmap, find_peak_fwhm
from .output_paths import build_retrieval_save_prefix

C_LIGHT = 299792458.0

def _build_retrieval_save_prefix(prefix: str):
    retrieval_dir, save_prefix = build_retrieval_save_prefix(prefix)
    os.makedirs(retrieval_dir, exist_ok=True)
    return retrieval_dir, save_prefix


def _open_dialog_dir(path: str) -> str:
    folder = os.path.abspath(path)
    probe = folder
    while True:
        if os.path.basename(probe) == "retrieval_result":
            parent = os.path.dirname(probe)
            return parent if os.path.isdir(parent) else folder
        parent = os.path.dirname(probe)
        if parent == probe:
            return folder
        probe = parent

def import_data_1d(
    filename: str,
    have_title: bool = False
):
    """Read 1D tabular data and return x, y arrays (and optional titles)."""
    with open(filename, "r") as f:
        lines = f.readlines()
   # Skip title/comments until first numeric row.
    start_idx = None
    for i_line, line in enumerate(lines):
        line = line.strip()
        if not line:
            continue
        try:
            [float(x) for x in line.replace(",", " ").split()]
            start_idx = i_line
            break
        except Exception:
            continue
    if start_idx is None or (len(lines) - start_idx) < 2:
        raise ValueError(f"No data found in file: {filename}")
    if have_title:
        titles = lines[start_idx-1].replace(",", " ").split()
    lines = lines[start_idx:]
    test = lines[0].replace(",", " ").split()
    N_y = len(test) - 1
    x = []
    y = [[] for _ in range(N_y)]
    iline = 0
    while iline < len(lines):
        if not lines[iline].strip():
            iline += 1
            continue
        try:
            l = lines[iline].replace(",", " ").split()
            x.append(float(l[0]))
            for iP in range(N_y):
                y[iP].append(float(l[iP + 1]))
            iline += 1
        except Exception:
            iline += 1
            continue
    if have_title:
        return np.array(x), np.array(y), titles
    return np.array(x), np.array(y)


class FrequencyAnalysisWidget(QtWidgets.QWidget):
    def __init__(
        self,
        result_obj,
        parent=None,
        status_sink=None,
        time_limits_provider=None,
    ):
        super().__init__(parent)
        self.result_obj = result_obj
        self._material_module_cache = {}
        self._status_sink = status_sink
        self._time_limits_provider = time_limits_provider

        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(2, 2, 2, 2)
        layout.setSpacing(2)
        ctrl = QtWidgets.QHBoxLayout()
        layout.addLayout(ctrl)
        ctrl.addWidget(QtWidgets.QLabel("freq min (PHz)"))
        self.freq_min_edit = QtWidgets.QLineEdit("")
        ctrl.addWidget(self.freq_min_edit)
        ctrl.addWidget(QtWidgets.QLabel("freq max (PHz)"))
        self.freq_max_edit = QtWidgets.QLineEdit("")
        ctrl.addWidget(self.freq_max_edit)
        ctrl.addWidget(QtWidgets.QLabel("add_GDD (fs^2)"))
        self.add_gdd_edit = QtWidgets.QLineEdit("0")
        ctrl.addWidget(self.add_gdd_edit)
        ctrl.addWidget(QtWidgets.QLabel("add_TOD (fs^3)"))
        self.add_tod_edit = QtWidgets.QLineEdit("0")
        ctrl.addWidget(self.add_tod_edit)
        self.update_btn = QtWidgets.QPushButton("Update")
        self.save_btn = QtWidgets.QPushButton("Save")
        ctrl.addWidget(self.update_btn)
        ctrl.addWidget(self.save_btn)
        ctrl.addStretch(1)

        mat_row = QtWidgets.QHBoxLayout()
        layout.addLayout(mat_row)
        mat_row.addWidget(QtWidgets.QLabel("Fit order"))
        self.fit_order_edit = QtWidgets.QLineEdit("3")
        self.fit_order_edit.setMaximumWidth(80)
        mat_row.addWidget(self.fit_order_edit)
        mat_row.addWidget(QtWidgets.QLabel("Material"))
        self.material_combo = QtWidgets.QComboBox()
        self.material_combo.addItem("None")
        for mat in self._list_dispersion_materials():
            self.material_combo.addItem(mat)
        if self.material_combo.findText("FS") >= 0:
            self.material_combo.setCurrentText("FS")
        mat_row.addWidget(self.material_combo)
        mat_row.addWidget(QtWidgets.QLabel("thickness (mm)"))
        self.material_thickness_edit = QtWidgets.QLineEdit("0")
        self.material_thickness_edit.setMaximumWidth(120)
        mat_row.addWidget(self.material_thickness_edit)
        self.material_disp_label = QtWidgets.QLabel("Material GDD/TOD: --")
        self.material_disp_label.setStyleSheet("color: #404040;")
        mat_row.addWidget(self.material_disp_label)
        mat_row.addStretch(1)

        self.fig = Figure(figsize=(12, 5.5), dpi=90)
        self.freq_ax = self.fig.add_subplot(121)
        self.intensity_ax = self.freq_ax.twinx()
        self.angular_freq_ax = self.freq_ax.secondary_xaxis(
            "top",
            functions=(lambda value: 2.0 * np.pi * value, lambda value: value / (2.0 * np.pi)),
        )
        self.time_ax = self.fig.add_subplot(122)
        self.canvas = FigureCanvasQTAgg(self.fig)
        layout.addWidget(self.canvas, 1)

        self.update_btn.clicked.connect(self.redraw)
        self.save_btn.clicked.connect(self.save_phase_analysis)
        self.freq_min_edit.editingFinished.connect(self.redraw)
        self.freq_max_edit.editingFinished.connect(self.redraw)
        self.add_gdd_edit.editingFinished.connect(self.redraw)
        self.add_tod_edit.editingFinished.connect(self.redraw)
        self.fit_order_edit.editingFinished.connect(self.redraw)
        self.material_combo.currentTextChanged.connect(self.redraw)
        self.material_thickness_edit.editingFinished.connect(self.redraw)
        self.redraw()

    def _notify(self, text: str, error: bool = False):
        if callable(self._status_sink):
            self._status_sink(text, error=error)
            return
        parent = self.parent()
        if parent is not None and hasattr(parent, "_show_status"):
            try:
                parent._show_status(text, error=error)
                return
            except Exception:
                pass

    @staticmethod
    def _list_dispersion_materials():
        _repo_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        disp_dir = os.path.join(_repo_dir, "refractive_index")
        if not os.path.isdir(disp_dir):
            return []
        mats = []
        for fname in os.listdir(disp_dir):
            if not fname.endswith("_dis.py"):
                continue
            mat = fname[:-7]
            if not re.match(r"^[A-Za-z_][A-Za-z0-9_]*$", mat):
                continue
            mats.append(mat)
        return sorted(set(mats), key=str.lower)

    def _load_material_module(self, material: str):
        if material in self._material_module_cache:
            return self._material_module_cache[material]
        _repo_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        if _repo_dir not in sys.path:
            sys.path.insert(0, _repo_dir)
        mod_name = f"refractive_index.{material}_dis"
        mod = importlib.import_module(mod_name)
        if not hasattr(mod, "n"):
            raise AttributeError(f"Material '{material}' has no n(lam) function.")
        self._material_module_cache[material] = mod
        return mod

    @staticmethod
    def _effective_index(n_raw, n_points: int):
        n_arr = np.asarray(n_raw, dtype=float)
        if n_arr.ndim == 0:
            return np.full(n_points, float(n_arr), dtype=float)
        if n_arr.ndim == 1:
            if n_arr.size == n_points:
                return n_arr.astype(float)
            return np.full(n_points, float(n_arr[0]), dtype=float)
        first_axis = np.asarray(n_arr[0], dtype=float).reshape(-1)
        if first_axis.size == n_points:
            return first_axis.astype(float)
        if first_axis.size == 1:
            return np.full(n_points, float(first_axis[0]), dtype=float)
        raise ValueError("Unsupported refractive-index array shape.")

    def _parse_material_addition(self):
        material = self.material_combo.currentText().strip()
        if material == "" or material == "None":
            return None, 0.0
        txt = self.material_thickness_edit.text().strip()
        try:
            thickness_mm = float(txt) if txt else 0.0
        except ValueError:
            self._notify("Invalid input: material thickness must be numeric.", error=True)
            return None, 0.0
        return material, thickness_mm

    @staticmethod
    def _extract_gdd_tod_from_phase(x_rad_per_s, phase_rad):
        x = np.asarray(x_rad_per_s, dtype=float)
        y = np.asarray(phase_rad, dtype=float)
        valid = np.isfinite(x) & np.isfinite(y)
        if np.count_nonzero(valid) < 3:
            return 0.0, 0.0
        x = x[valid]
        y = y[valid]

        # Local derivative around x=0 (angular_freq0), not a global fit across the full spectrum.
        # x is in rad/s, so convert derivatives to fs^2 / fs^3 at the end.
        idx = np.argsort(np.abs(x))
        n_local = min(121, len(x))
        if n_local < 3:
            return 0.0, 0.0
        sel = np.sort(idx[:n_local])
        xl = x[sel]
        yl = y[sel]
        deg = min(5, n_local - 1)
        poly = np.poly1d(np.polyfit(xl, yl, deg))
        gdd = float(np.polyder(poly, 2)(0.0) * 1e30) if deg >= 2 else 0.0
        tod = float(np.polyder(poly, 3)(0.0) * 1e45) if deg >= 3 else 0.0
        return gdd, tod

    @staticmethod
    def _phase_at_frequency(freq, phase, freq0):
        freq = np.asarray(freq, dtype=float)
        phase = np.asarray(phase, dtype=float)
        valid = np.isfinite(freq) & np.isfinite(phase)
        if np.count_nonzero(valid) == 0:
            return 0.0
        f = freq[valid]
        p = phase[valid]
        idx = np.argsort(f)
        f = f[idx]
        p = p[idx]
        f_unique, uniq_idx = np.unique(f, return_index=True)
        p_unique = p[uniq_idx]
        if f_unique.size == 1:
            return float(p_unique[0])
        if freq0 <= f_unique[0]:
            return float(p_unique[0])
        if freq0 >= f_unique[-1]:
            return float(p_unique[-1])
        return float(np.interp(freq0, f_unique, p_unique))

    def _material_phase_add(self, freq_phz, angular_freq0, material, thickness_mm):
        freq_phz = np.asarray(freq_phz, dtype=float)
        phi_add = np.zeros_like(freq_phz, dtype=float)
        mat_gdd = 0.0
        mat_tod = 0.0
        if material is None or abs(float(thickness_mm)) == 0.0:
            return phi_add, mat_gdd, mat_tod

        freq_hz = freq_phz * 1e15
        omega = 2 * np.pi * freq_hz
        valid = np.isfinite(freq_hz) & (freq_hz > 0)
        if np.count_nonzero(valid) < 2:
            raise ValueError("Frequency axis must contain at least two positive points.")

        lam_um = C_LIGHT / freq_hz[valid] * 1e6
        mat_mod = self._load_material_module(material)
        n_eff = self._effective_index(mat_mod.n(lam_um), lam_um.size)
        phase_valid = omega[valid] * n_eff * (thickness_mm * 1e-3) / C_LIGHT

        # Keep dispersion-only component for stable plotting/compensation:
        # remove global phase + linear group-delay term using local values at angular_freq0.
        sort_idx = np.argsort(omega[valid])
        omega_sorted = omega[valid][sort_idx]
        phase_sorted = phase_valid[sort_idx]
        # angular_freq0 in this GUI is rad/fs (because freq axis is PHz),
        # convert it to rad/s before mixing with omega in SI units.
        angular_freq0_si = float(angular_freq0) * 1e15
        x = omega_sorted - angular_freq0_si
        phase_center = float(
            np.interp(
                angular_freq0_si,
                omega_sorted,
                phase_sorted,
                left=phase_sorted[0],
                right=phase_sorted[-1],
            )
        )
        if omega_sorted.size >= 3:
            dphase_domega = np.gradient(phase_sorted, omega_sorted, edge_order=2)
        else:
            dphase_domega = np.gradient(phase_sorted, omega_sorted)
        delay_center = float(
            np.interp(
                angular_freq0_si,
                omega_sorted,
                dphase_domega,
                left=dphase_domega[0],
                right=dphase_domega[-1],
            )
        )
        phase_disp_only = phase_sorted - (phase_center + delay_center * x)
        # Also enforce phase(angular_freq0)=0 for the material-added phase.
        phase0 = float(
            np.interp(
                angular_freq0_si,
                omega_sorted,
                phase_disp_only,
                left=phase_disp_only[0],
                right=phase_disp_only[-1],
            )
        )
        phase_disp_only = phase_disp_only - phase0
        mat_gdd, mat_tod = self._extract_gdd_tod_from_phase(x, phase_disp_only)

        phase_out = np.empty_like(phase_disp_only)
        phase_out[sort_idx] = phase_disp_only
        phi_add[valid] = phase_out
        return phi_add, mat_gdd, mat_tod

    def _parse_range(self):
        fmin = self.freq_min_edit.text().strip()
        fmax = self.freq_max_edit.text().strip()
        try:
            vmin = float(fmin) if fmin else None
            vmax = float(fmax) if fmax else None
        except ValueError:
            self._notify("Invalid input: frequency range must be numeric.", error=True)
            return None, None
        if vmin is not None and vmax is not None and vmin >= vmax:
            self._notify("Invalid range: freq min must be smaller than freq max.", error=True)
            return None, None
        return vmin, vmax

    def _parse_fit_order(self):
        txt = self.fit_order_edit.text().strip()
        try:
            fit_order = int(txt) if txt else 3
        except ValueError:
            self._notify("Invalid input: fit order must be an integer.", error=True)
            return None
        if fit_order < 3:
            self._notify("Invalid input: fit order must be at least 3.", error=True)
            return None
        return fit_order

    def _parse_dispersion_add(self):
        gdd_text = self.add_gdd_edit.text().strip()
        tod_text = self.add_tod_edit.text().strip()
        try:
            add_gdd = float(gdd_text) if gdd_text else 0.0
            add_tod = float(tod_text) if tod_text else 0.0
        except ValueError:
            self._notify("Invalid input: GDD/TOD values must be numeric.", error=True)
            return 0.0, 0.0
        return add_gdd, add_tod

    def save_phase_analysis(self):
        if self.result_obj is None:
            self._notify("No data: please load a result first.", error=True)
            return
        try:
            retrieval_dir, save_prefix = _build_retrieval_save_prefix(self.result_obj.prefix)
            out_path = f"{save_prefix}_phase_analysis.png"
            self.fig.savefig(out_path, dpi=200, bbox_inches="tight")
            self._notify(f"Saved to: {retrieval_dir} | Phase analysis: {out_path}", error=False)
        except Exception as e:
            self._notify(f"Save error: {e}", error=True)

    def redraw(self):
        self.freq_ax.clear()
        self.intensity_ax.clear()
        self.time_ax.clear()
        self.freq_ax.set_xlabel("Frequency (PHz)")
        self.freq_ax.set_ylabel("Phase (rad)")
        self.intensity_ax.set_ylabel("Intensity")
        self.intensity_ax.yaxis.set_label_position("right")
        self.intensity_ax.yaxis.tick_right()
        self.intensity_ax.spines["right"].set_visible(True)
        self.intensity_ax.spines["left"].set_visible(False)
        self.intensity_ax.set_ylim(0.0, 1.4)
        self.angular_freq_ax.set_xlabel("Angular frequency (rad/fs)")
        self.time_ax.set_xlabel("Time (fs)")
        self.time_ax.set_ylabel("Intensity")
        self.time_ax.set_ylim(0.0, 1.4)
        self.freq_ax.grid(True, alpha=0.25)
        self.time_ax.grid(True, alpha=0.25)

        if self.result_obj is None:
            self.freq_ax.set_title("Phase analysis (load a result first)")
            self.time_ax.set_title("Dispersion compensation")
            self.fig.tight_layout()
            self.canvas.draw_idle()
            return

        try:
            fmin, fmax = self._parse_range()
            fit_order = self._parse_fit_order()
            if fit_order is None:
                return
            add_gdd, add_tod = self._parse_dispersion_add()
            material, thickness_mm = self._parse_material_addition()
            res = self.result_obj.get_GDD_TOD(freq_min=fmin, freq_max=fmax, fit_order=fit_order)

            freq = np.asarray(self.result_obj.freq, dtype=float)
            phase = np.asarray(self.result_obj.freq_phase, dtype=float)
            intensity = np.asarray(self.result_obj.freq_intensity, dtype=float)
            angular_freq_fit = np.asarray(res["angular_freq"], dtype=float)
            angular_freq0 = float(res["angular_freq0"])
            phase_fit = np.asarray(res["phase_fit"], dtype=float)
            freq_fit = angular_freq_fit / (2 * np.pi)
            freq0 = angular_freq0 / (2 * np.pi)
            angular_freq = freq * 2 * np.pi
            add_phase = add_gdd / 2 * (angular_freq - angular_freq0) ** 2 + add_tod / 6 * (angular_freq - angular_freq0) ** 3
            add_phase_material, mat_gdd, mat_tod = self._material_phase_add(freq, angular_freq0, material, thickness_mm)
            phase_comp = phase + add_phase + add_phase_material
            # Set compensated phase reference: phase(angular_freq0) = 0.
            phase_comp = phase_comp - self._phase_at_frequency(freq, phase_comp, freq0)
            if material is not None and abs(thickness_mm) > 0:
                gvd = mat_gdd / thickness_mm
                tod_per_mm = mat_tod / thickness_mm
                self.material_disp_label.setText(
                    f"{material} {thickness_mm:.4g} mm -> GDD={mat_gdd:.2f} fs^2, TOD={mat_tod:.2f} fs^3"
                    f" (GVD={gvd:.2f} fs^2/mm, TOD={tod_per_mm:.2f} fs^3/mm)"
                )
            else:
                self.material_disp_label.setText("Material GDD/TOD: --")

            # Left axis: phase and polynomial fit.
            phase_line = self.freq_ax.plot(
                freq,
                phase,
                color="tab:red",
                linewidth=1.8,
                marker="o",
                markersize=3,
                label="Retrieved phase",
            )[0]
            compensated_line = self.freq_ax.plot(
                freq,
                phase_comp,
                color="0.5",
                linewidth=2.0,
                label="Compensated phase",
            )[0]
            fit_line = self.freq_ax.plot(
                freq_fit,
                phase_fit,
                color="tab:blue",
                linewidth=2.2,
                linestyle="--",
                label="Polynomial fit",
            )[0]

            # Right axis: intensity.
            intensity_line = self.intensity_ax.plot(
                freq,
                intensity,
                color="0.15",
                linewidth=1.6,
                marker="o",
                markersize=3,
                alpha=0.75,
                label="Intensity",
            )[0]
            center_line = self.freq_ax.axvline(
                freq0,
                color="0.4",
                linewidth=1.4,
                linestyle="--",
                label=f"ω₀={angular_freq0:.3f} rad/fs",
            )

            if freq_fit.size > 1:
                fit_min = float(np.min(freq_fit))
                fit_max = float(np.max(freq_fit))
                fit_center = 0.5 * (fit_min + fit_max)
                fit_half = 0.5 * (fit_max - fit_min)
                self.freq_ax.set_xlim(fit_center - 3 * fit_half, fit_center + 3 * fit_half)
            else:
                self.freq_ax.set_xlim(float(np.min(freq)), float(np.max(freq)))

            self.freq_ax.set_ylim(-np.pi, np.pi)
            self.intensity_ax.set_ylim(0.0, 1.4)
            self.freq_ax.set_title(
                f"GDD={res['GDD']:.0f} fs², TOD={res['TOD']:.0f} fs³",
                color="tab:blue", fontsize=10,
            )
            self.freq_ax.legend(
                [phase_line, compensated_line, fit_line, center_line, intensity_line],
                ["Retrieved phase", "Compensated phase", "Polynomial fit", center_line.get_label(), "Intensity"],
                loc="best",
                fontsize=10,
            )

            # Time-domain view after adding extra dispersion terms.
            new_time, new_intensity = self.result_obj.get_time_profile_from_frequency(freq, intensity, phase_comp)
            raw_time = np.asarray(self.result_obj.time, dtype=float)
            raw_intensity = np.asarray(self.result_obj.time_intensity, dtype=float)
            ftl_time = np.asarray(self.result_obj.time_FTL, dtype=float)
            ftl_intensity = np.asarray(self.result_obj.time_FTL_intensity, dtype=float)
            new_time = np.asarray(new_time, dtype=float)
            new_intensity = np.asarray(new_intensity, dtype=float)
            _, _, fwhm_raw,_ = find_peak_fwhm(raw_time, raw_intensity)
            _, _, fwhm_new,_ = find_peak_fwhm(new_time, new_intensity)
            _, _, fwhm_ftl,_ = find_peak_fwhm(ftl_time, ftl_intensity)

            self.time_ax.plot(
                raw_time,
                raw_intensity,
                color="0.5",
                linewidth=2.0,
                label=f"Original FWHM={fwhm_raw:.1f} fs",
            )
            self.time_ax.plot(
                new_time,
                new_intensity,
                color="tab:red",
                linewidth=2.4,
                label=f"Compensated FWHM={fwhm_new:.1f} fs",
            )
            self.time_ax.plot(
                ftl_time,
                ftl_intensity,
                color="tab:blue",
                linewidth=2.2,
                linestyle="--",
                label=f"FTL FWHM={fwhm_ftl:.1f} fs",
            )
            pulse0 = float(self.result_obj.pulse_duration)
            time_min = None
            time_max = None
            if callable(self._time_limits_provider):
                time_min, time_max = self._time_limits_provider()
            if (
                time_min is not None
                and time_max is not None
                and time_min < time_max
            ):
                self.time_ax.set_xlim(time_min, time_max)
            else:
                self.time_ax.set_xlim(-3.5 * pulse0, 2.5 * pulse0)
            self.time_ax.set_ylim(0.0, 1.4)
            self.time_ax.legend(fontsize=10)
            mat_text = f", material={material} {thickness_mm:.4g} mm" if material is not None and abs(thickness_mm) > 0 else ""
            self.time_ax.set_title(
                f"add_GDD={add_gdd:.0f} fs², add_TOD={add_tod:.0f} fs³{mat_text}",fontsize=10
            )
        except Exception as e:
            self.material_disp_label.setText("Material GDD/TOD: error")
            self.freq_ax.set_title(f"Frequency analysis error: {e}")
            self.time_ax.set_title(f"Compensation plot error: {e}")
            self._notify(f"Frequency analysis error: {e}", error=True)
        self.fig.tight_layout()
        self.canvas.draw_idle()


class FrogResultGUI(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("FROG Result GUI")
        self.resize(1050, 750)

        self.last_dir = os.getcwd()
        self.result_obj = None
        self.measured_wave = None
        self.measured_intensity = None
        self._cbars = {}
        self.phase_analysis_widget = None

        self._build_ui()
        self._build_status_bar()

    def _build_status_bar(self):
        self.status_bar = QtWidgets.QStatusBar(self)
        self.setStatusBar(self.status_bar)
        self._show_status("Ready.", error=False, timeout_ms=4000)

    def _show_status(self, text: str, error: bool = False, timeout_ms: int = 10000):
        color = "#b00020" if error else "#202020"
        self.status_bar.setStyleSheet(f"QStatusBar{{color:{color};}}")
        self.status_bar.showMessage(str(text), timeout_ms)

    def _build_ui(self):
        central = QtWidgets.QWidget()
        self.setCentralWidget(central)
        root = QtWidgets.QHBoxLayout(central)
        root.setContentsMargins(2, 2, 2, 2)
        root.setSpacing(2)

        split = QtWidgets.QSplitter(QtCore.Qt.Orientation.Horizontal)
        split.setHandleWidth(2)
        split.setChildrenCollapsible(False)
        root.addWidget(split, 1)

        left_panel = QtWidgets.QWidget()
        left_panel.setFixedWidth(200)
        left_layout = QtWidgets.QVBoxLayout(left_panel)
        left_layout.setContentsMargins(2, 2, 2, 2)
        left_layout.setSpacing(2)
        split.addWidget(left_panel)

        params_box = QtWidgets.QGroupBox("Parameters")
        params_form = QtWidgets.QFormLayout(params_box)
        left_layout.addWidget(params_box)

        self.delay_min = QtWidgets.QLineEdit("")
        params_form.addRow("delay_min", self.delay_min)
        self.delay_max = QtWidgets.QLineEdit("")
        params_form.addRow("delay_max", self.delay_max)
        self.freq_min = QtWidgets.QLineEdit("")
        params_form.addRow("freq_min", self.freq_min)
        self.freq_max = QtWidgets.QLineEdit("")
        params_form.addRow("freq_max", self.freq_max)
        self.time_min = QtWidgets.QLineEdit("")
        params_form.addRow("time_min", self.time_min)
        self.time_max = QtWidgets.QLineEdit("")
        params_form.addRow("time_max", self.time_max)
        self.wave_min = QtWidgets.QLineEdit("")
        params_form.addRow("wavelength_min", self.wave_min)
        self.wave_max = QtWidgets.QLineEdit("")
        params_form.addRow("wavelength_max", self.wave_max)
        self.raw_shift = QtWidgets.QLineEdit("")
        params_form.addRow("raw_shift", self.raw_shift)
        self.trace_log = QtWidgets.QCheckBox("Trace log scale")
        self.trace_log.setChecked(False)
        params_form.addRow(self.trace_log)
        self.time_reverse_btn = QtWidgets.QCheckBox("Time Reverse")
        self.time_reverse_btn.setChecked(False)
        params_form.addRow(self.time_reverse_btn)
        left_layout.addStretch(1)

        right_panel = QtWidgets.QWidget()
        right_layout = QtWidgets.QVBoxLayout(right_panel)
        right_layout.setContentsMargins(2, 2, 2, 2)
        right_layout.setSpacing(2)
        split.addWidget(right_panel)
        split.setStretchFactor(0, 0)
        split.setStretchFactor(1, 1)
        QtCore.QTimer.singleShot(0, lambda: split.setSizes([220, 10_000]))

        row1 = QtWidgets.QHBoxLayout()
        row1.setContentsMargins(0, 0, 0, 0)
        row1.setSpacing(6)
        right_layout.addLayout(row1)
        row1.addWidget(QtWidgets.QLabel("prefix"))
        self.prefix_edit = QtWidgets.QLineEdit()
        row1.addWidget(self.prefix_edit, 1)
        self.open_btn = QtWidgets.QPushButton("Open")
        self.load_btn = QtWidgets.QPushButton("Load")
        self.save_btn = QtWidgets.QPushButton("Save")
        row1.addWidget(self.open_btn)
        row1.addWidget(self.load_btn)
        row1.addWidget(self.save_btn)

        row2 = QtWidgets.QHBoxLayout()
        row2.setContentsMargins(0, 0, 0, 0)
        row2.setSpacing(6)
        right_layout.addLayout(row2)
        row2.addWidget(QtWidgets.QLabel("Fundamental_spectra_path"))
        self.raw_spectra_edit = QtWidgets.QLineEdit()
        row2.addWidget(self.raw_spectra_edit, 1)
        self.open_spectra_btn = QtWidgets.QPushButton("Open Spectra")
        row2.addWidget(self.open_spectra_btn)

        self.fig = Figure(figsize=(12, 9), dpi=80)
        self.ax_exp = self.fig.add_subplot(221)
        self.ax_rec = self.fig.add_subplot(222)
        self.ax_time = self.fig.add_subplot(223)
        self.ax_phase = self.ax_time.twinx()
        self.ax_spec = self.fig.add_subplot(224)
        self.canvas = FigureCanvasQTAgg(self.fig)

        self.result_tabs = QtWidgets.QTabWidget()
        right_layout.addWidget(self.result_tabs, 1)
        result_tab = QtWidgets.QWidget()
        result_tab_layout = QtWidgets.QVBoxLayout(result_tab)
        result_tab_layout.setContentsMargins(0, 0, 0, 0)
        result_tab_layout.addWidget(self.canvas, 1)
        self.result_tabs.addTab(result_tab, "FROG Result")

        self.phase_analysis_widget = FrequencyAnalysisWidget(
            self.result_obj,
            self,
            status_sink=self._show_status,
            time_limits_provider=self._phase_analysis_time_limits,
        )
        self.result_tabs.addTab(self.phase_analysis_widget, "Phase Analysis")

        self.open_btn.clicked.connect(self.open_result_file)
        self.load_btn.clicked.connect(self.load_result)
        self.save_btn.clicked.connect(self.save_figure)
        self.open_spectra_btn.clicked.connect(self.open_raw_spectra)
        self.trace_log.toggled.connect(self.redraw)
        self.time_reverse_btn.toggled.connect(self._on_time_reverse_toggled)
        self.result_tabs.currentChanged.connect(self._on_result_tab_changed)
        self.time_min.editingFinished.connect(self._on_time_limits_changed)
        self.time_max.editingFinished.connect(self._on_time_limits_changed)
        self.wave_min.editingFinished.connect(self.redraw)
        self.wave_max.editingFinished.connect(self.redraw)
        self.delay_min.editingFinished.connect(self.redraw)
        self.delay_max.editingFinished.connect(self.redraw)
        self.freq_min.editingFinished.connect(self.redraw)
        self.freq_max.editingFinished.connect(self.redraw)
        self.raw_shift.editingFinished.connect(self.redraw)

    def _extract_prefix(self, file_path):
        path = str(file_path)
        suffixes = [".A.dat",".Arecon.dat",".Ek.dat",".Ew.dat",".Speck.dat"]
        lower = path.lower()
        for s in suffixes:
            if lower.endswith(s.lower()):
                return path[: -len(s)]
        m = re.search(r"\.[^.\\/]+\.dat$", path, re.IGNORECASE)
        if m:
            return path[: m.start()]
        if lower.endswith(".dat"):
            return path[:-4]
        return path

    def _update_last_dir(self, path):
        folder = os.path.dirname(os.path.abspath(path))
        if os.path.isdir(folder):
            self.last_dir = folder

    @staticmethod
    def _has_array_data(obj, name: str) -> bool:
        if obj is None or not hasattr(obj, name):
            return False
        try:
            return np.asarray(getattr(obj, name)).size > 0
        except Exception:
            return False

    @staticmethod
    def _load_5col_dat(path: str):
        arr = np.loadtxt(path, dtype=float)
        arr = np.asarray(arr, dtype=float)
        if arr.ndim == 1:
            if arr.size < 5:
                raise ValueError(f"{os.path.basename(path)} has fewer than 5 columns.")
            arr = arr.reshape(1, -1)
        if arr.ndim != 2 or arr.shape[1] < 5:
            raise ValueError(f"{os.path.basename(path)} must have at least 5 columns.")
        return arr[:, 0], arr[:, 1], arr[:, 2], arr[:, 3], arr[:, 4]

    def _load_partial_result(self, prefix: str):
        obj = FROG_result.__new__(FROG_result)
        obj.prefix = prefix
        obj._time_reversed = False
        obj._partial = True
        obj._missing_files = []
        obj._load_errors = []

        # Default empty fields so methods like set_time_reversed can still run.
        obj.frg_delay = np.array([], dtype=float)
        obj.frg_freq = np.array([], dtype=float)
        obj.frg_trace = np.empty((0, 0), dtype=float)
        obj.frg_trace_reconstructed = np.empty((0, 0), dtype=float)
        obj.time = np.array([], dtype=float)
        obj.time_intensity = np.array([], dtype=float)
        obj.time_phase = np.array([], dtype=float)
        obj.freq = np.array([], dtype=float)
        obj.freq_intensity = np.array([], dtype=float)
        obj.freq_phase = np.array([], dtype=float)
        obj.wavelength = np.array([], dtype=float)
        obj.wavelength_intensity = np.array([], dtype=float)
        obj.wavelength_phase = np.array([], dtype=float)
        obj.time_FTL = np.array([], dtype=float)
        obj.time_FTL_intensity = np.array([], dtype=float)
        obj.pulse_duration = np.nan
        obj.pulse_duration_FTL = np.nan

        files = {
            ".A.dat": f"{prefix}.A.dat",
            ".Arecon.dat": f"{prefix}.Arecon.dat",
            ".Ek.dat": f"{prefix}.Ek.dat",
            ".Ew.dat": f"{prefix}.Ew.dat",
            ".Speck.dat": f"{prefix}.Speck.dat",
        }

        for suffix, path in files.items():
            if not os.path.isfile(path):
                obj._missing_files.append(suffix)

        if ".A.dat" not in obj._missing_files:
            try:
                obj.frg_delay, obj.frg_freq, obj.frg_trace = obj.load_frog_trace(files[".A.dat"])
            except Exception as e:
                obj._load_errors.append(f".A.dat: {e}")

        if ".Arecon.dat" not in obj._missing_files:
            try:
                d_rec, f_rec, z_rec = obj.load_frog_trace(files[".Arecon.dat"])
                obj.frg_trace_reconstructed = z_rec
                if obj.frg_delay.size == 0:
                    obj.frg_delay = d_rec
                    obj.frg_freq = f_rec
            except Exception as e:
                obj._load_errors.append(f".Arecon.dat: {e}")

        if ".Ek.dat" not in obj._missing_files:
            try:
                t, i, p, _, _ = self._load_5col_dat(files[".Ek.dat"])
                obj.time = np.asarray(t, dtype=float)
                obj.time_intensity = np.asarray(i, dtype=float)
                obj.time_phase = np.asarray(p, dtype=float)
            except Exception as e:
                obj._load_errors.append(f".Ek.dat: {e}")

        if ".Ew.dat" not in obj._missing_files:
            try:
                f, i, p, _, _ = self._load_5col_dat(files[".Ew.dat"])
                obj.freq = np.asarray(f, dtype=float)
                obj.freq_intensity = np.asarray(i, dtype=float)
                obj.freq_phase = np.asarray(p, dtype=float)
            except Exception as e:
                obj._load_errors.append(f".Ew.dat: {e}")

        if ".Speck.dat" not in obj._missing_files:
            try:
                w, i, p, _, _ = self._load_5col_dat(files[".Speck.dat"])
                obj.wavelength = np.asarray(w, dtype=float)
                obj.wavelength_intensity = np.asarray(i, dtype=float)
                obj.wavelength_phase = np.asarray(p, dtype=float)
            except Exception as e:
                obj._load_errors.append(f".Speck.dat: {e}")

        if obj.time.size >= 3 and obj.time_intensity.size == obj.time.size:
            try:
                _, _, obj.pulse_duration, _ = find_peak_fwhm(obj.time, obj.time_intensity)
            except Exception:
                obj.pulse_duration = np.nan
        if obj.freq.size >= 2 and obj.freq_intensity.size == obj.freq.size:
            try:
                obj.time_FTL, obj.time_FTL_intensity = obj.get_time_profile_from_frequency(
                    obj.freq, obj.freq_intensity, 0
                )
                _, _, obj.pulse_duration_FTL, _ = find_peak_fwhm(obj.time_FTL, obj.time_FTL_intensity)
            except Exception:
                obj.time_FTL = np.array([], dtype=float)
                obj.time_FTL_intensity = np.array([], dtype=float)
                obj.pulse_duration_FTL = np.nan

        obj._base_state = {
            "time": np.asarray(obj.time, dtype=float).copy(),
            "time_intensity": np.asarray(obj.time_intensity, dtype=float).copy(),
            "time_phase": np.asarray(obj.time_phase, dtype=float).copy(),
            "freq": np.asarray(obj.freq, dtype=float).copy(),
            "freq_intensity": np.asarray(obj.freq_intensity, dtype=float).copy(),
            "freq_phase": np.asarray(obj.freq_phase, dtype=float).copy(),
            "wavelength": np.asarray(obj.wavelength, dtype=float).copy(),
            "wavelength_intensity": np.asarray(obj.wavelength_intensity, dtype=float).copy(),
            "wavelength_phase": np.asarray(obj.wavelength_phase, dtype=float).copy(),
            "time_FTL": np.asarray(obj.time_FTL, dtype=float).copy(),
            "time_FTL_intensity": np.asarray(obj.time_FTL_intensity, dtype=float).copy(),
        }

        has_any_data = any(
            self._has_array_data(obj, name)
            for name in ("frg_trace", "frg_trace_reconstructed", "time_intensity", "freq_intensity", "wavelength_intensity")
        )
        if not has_any_data:
            details = "; ".join(obj._load_errors) if obj._load_errors else "no valid data files found."
            raise FileNotFoundError(f"Cannot load result from prefix '{prefix}': {details}")
        return obj

    def load_result_payload(self, payload):
        """Load result directly from an in-memory payload dict."""
        if not isinstance(payload, dict):
            raise ValueError("Result payload must be a dict.")
        prefix = str(payload.get("prefix", "in_memory_result")).strip() or "in_memory_result"
        obj = FROG_result.__new__(FROG_result)
        obj.prefix = prefix
        obj._time_reversed = False
        obj._partial = False
        obj._missing_files = []
        obj._load_errors = []

        def _arr(key, default=None):
            base = np.array([], dtype=float) if default is None else default
            return np.asarray(payload.get(key, base), dtype=float).copy()

        obj.frg_delay = _arr("frg_delay")
        obj.frg_freq = _arr("frg_freq")
        obj.frg_trace = _arr("frg_trace", np.empty((0, 0), dtype=float))
        obj.frg_trace_reconstructed = _arr("frg_trace_reconstructed", np.empty((0, 0), dtype=float))
        obj.time = _arr("time")
        obj.time_intensity = _arr("time_intensity")
        obj.time_phase = _arr("time_phase")
        obj.freq = _arr("freq")
        obj.freq_intensity = _arr("freq_intensity")
        obj.freq_phase = _arr("freq_phase")
        obj.wavelength = _arr("wavelength")
        obj.wavelength_intensity = _arr("wavelength_intensity")
        obj.wavelength_phase = _arr("wavelength_phase")
        obj.time_FTL = _arr("time_FTL")
        obj.time_FTL_intensity = _arr("time_FTL_intensity")

        obj.frg_trace = np.nan_to_num(obj.frg_trace, nan=0.0, posinf=0.0, neginf=0.0)
        obj.frg_trace = np.clip(obj.frg_trace, 0.0, None)
        obj.frg_trace_reconstructed = np.nan_to_num(obj.frg_trace_reconstructed, nan=0.0, posinf=0.0, neginf=0.0)
        obj.frg_trace_reconstructed = np.clip(obj.frg_trace_reconstructed, 0.0, None)
        obj.time_intensity = np.clip(np.nan_to_num(obj.time_intensity, nan=0.0, posinf=0.0, neginf=0.0), 0.0, None)
        obj.freq_intensity = np.clip(np.nan_to_num(obj.freq_intensity, nan=0.0, posinf=0.0, neginf=0.0), 0.0, None)
        obj.wavelength_intensity = np.clip(np.nan_to_num(obj.wavelength_intensity, nan=0.0, posinf=0.0, neginf=0.0), 0.0, None)

        pulse_duration = payload.get("pulse_duration", np.nan)
        try:
            obj.pulse_duration = float(pulse_duration)
        except Exception:
            obj.pulse_duration = np.nan
        if (not np.isfinite(obj.pulse_duration)) and obj.time.size >= 3 and obj.time_intensity.size == obj.time.size:
            try:
                _, _, obj.pulse_duration, _ = find_peak_fwhm(obj.time, obj.time_intensity)
            except Exception:
                obj.pulse_duration = np.nan

        pulse_duration_ftl = payload.get("pulse_duration_FTL", np.nan)
        try:
            obj.pulse_duration_FTL = float(pulse_duration_ftl)
        except Exception:
            obj.pulse_duration_FTL = np.nan
        if obj.time_FTL.size < 2 and obj.freq.size >= 2 and obj.freq_intensity.size == obj.freq.size:
            try:
                obj.time_FTL, obj.time_FTL_intensity = obj.get_time_profile_from_frequency(obj.freq, obj.freq_intensity, 0)
            except Exception:
                obj.time_FTL = np.array([], dtype=float)
                obj.time_FTL_intensity = np.array([], dtype=float)
        if (not np.isfinite(obj.pulse_duration_FTL)) and obj.time_FTL.size >= 3 and obj.time_FTL_intensity.size == obj.time_FTL.size:
            try:
                _, _, obj.pulse_duration_FTL, _ = find_peak_fwhm(obj.time_FTL, obj.time_FTL_intensity)
            except Exception:
                obj.pulse_duration_FTL = np.nan

        obj._base_state = {
            "time": np.asarray(obj.time, dtype=float).copy(),
            "time_intensity": np.asarray(obj.time_intensity, dtype=float).copy(),
            "time_phase": np.asarray(obj.time_phase, dtype=float).copy(),
            "freq": np.asarray(obj.freq, dtype=float).copy(),
            "freq_intensity": np.asarray(obj.freq_intensity, dtype=float).copy(),
            "freq_phase": np.asarray(obj.freq_phase, dtype=float).copy(),
            "wavelength": np.asarray(obj.wavelength, dtype=float).copy(),
            "wavelength_intensity": np.asarray(obj.wavelength_intensity, dtype=float).copy(),
            "wavelength_phase": np.asarray(obj.wavelength_phase, dtype=float).copy(),
            "time_FTL": np.asarray(obj.time_FTL, dtype=float).copy(),
            "time_FTL_intensity": np.asarray(obj.time_FTL_intensity, dtype=float).copy(),
        }

        has_any_data = any(
            self._has_array_data(obj, name)
            for name in ("frg_trace", "frg_trace_reconstructed", "time_intensity", "freq_intensity", "wavelength_intensity")
        )
        if not has_any_data:
            raise ValueError("In-memory result payload has no plottable data.")

        wave = payload.get("measured_wave", None)
        inten = payload.get("measured_intensity", None)
        if wave is not None and inten is not None:
            self.measured_wave = np.asarray(wave, dtype=float).copy()
            self.measured_intensity = np.asarray(inten, dtype=float).copy()

        self.result_obj = obj
        self.prefix_edit.setText(prefix)
        self.result_obj.set_time_reversed(self.time_reverse_btn.isChecked())
        self.redraw()
        self._sync_phase_analysis_tab(redraw=True)
        self._show_status(f"Loaded in-memory result: {prefix}", error=False)

    def open_result_file(self):
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self,
            "Open FROG result file",
            self.last_dir,
            "Data files (*.dat);;All files (*)",
        )
        if not path:
            return
        self._update_last_dir(path)
        self.prefix_edit.setText(self._extract_prefix(path))
        self.load_result()

    def load_result(self):
        prefix = self.prefix_edit.text().strip()
        if not prefix:
            self._show_status("Missing prefix: please input or open a result file first.", error=True)
            return
        try:
            try:
                self.result_obj = FROG_result(prefix)
                self.result_obj._partial = False
                self.result_obj._missing_files = []
                self.result_obj._load_errors = []
            except Exception:
                self.result_obj = self._load_partial_result(prefix)
            self.result_obj.set_time_reversed(self.time_reverse_btn.isChecked())
            self.redraw()
            self._sync_phase_analysis_tab(redraw=True)
            missing = getattr(self.result_obj, "_missing_files", [])
            load_errors = getattr(self.result_obj, "_load_errors", [])
            if missing or load_errors:
                warn_parts = []
                if missing:
                    warn_parts.append(f"missing: {', '.join(missing)}")
                if load_errors:
                    warn_parts.append(f"load errors: {'; '.join(load_errors)}")
                self._show_status(f"Loaded partial result: {prefix} ({' | '.join(warn_parts)})", error=True)
            else:
                self._show_status(f"Loaded result: {prefix}", error=False)
        except Exception as e:
            self._show_status(f"Load error: {e}", error=True)

    def _on_time_reverse_toggled(self, checked: bool):
        if self.result_obj is not None:
            try:
                self.result_obj.set_time_reversed(bool(checked))
            except Exception as e:
                self._show_status(f"Time reverse error: {e}", error=True)
        self.redraw()
        self._sync_phase_analysis_tab(redraw=True)

    def _phase_analysis_time_limits(self):
        time_min_text = self.time_min.text().strip()
        time_max_text = self.time_max.text().strip()
        time_min = float(time_min_text) if time_min_text else None
        time_max = float(time_max_text) if time_max_text else None
        return time_min, time_max

    def _on_time_limits_changed(self):
        self.redraw()
        self._sync_phase_analysis_tab(redraw=True)

    def _sync_phase_analysis_tab(self, redraw: bool = False):
        if self.phase_analysis_widget is None:
            return
        self.phase_analysis_widget.result_obj = self.result_obj
        if redraw and self.result_tabs.currentWidget() is self.phase_analysis_widget:
            self.phase_analysis_widget.redraw()

    def _on_result_tab_changed(self, index: int):
        if self.result_tabs.widget(index) is self.phase_analysis_widget:
            self._sync_phase_analysis_tab(redraw=True)

    def open_raw_spectra(self):
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self,
            "Open raw spectra file",
            _open_dialog_dir(self.last_dir),
            "All files (*)",
        )
        if not path:
            return
        try:
            x, ys = import_data_1d(path)
            if len(ys) < 1 or len(ys[0]) < 1:
                raise ValueError("Raw spectra file must contain at least 2 columns.")
            self.measured_wave = np.asarray(x, dtype=float)
            self.measured_intensity = np.asarray(ys[0], dtype=float)
            self._update_last_dir(path)
            self.raw_spectra_edit.setText(path)
            self.redraw()
            self._show_status(f"Loaded raw spectra: {path}", error=False)
        except Exception as e:
            self.measured_wave = None
            self.measured_intensity = None
            self.redraw()
            self._show_status(f"Open spectra error: {e}", error=True)

    def save_figure(self):
        if self.result_obj is None:
            self._show_status("No data: please load a result first.", error=True)
            return
        prefix = self.prefix_edit.text().strip()
        if not prefix:
            self._show_status("Missing prefix: please input prefix first.", error=True)
            return
        try:
            retrieval_dir, save_prefix = _build_retrieval_save_prefix(prefix)
            out_path = f"{save_prefix}_retrieval_result.png"
            self.fig.savefig(out_path, dpi=200, bbox_inches="tight")
            self._show_status(f"Saved to: {retrieval_dir} | Retrieval figure: {out_path}", error=False)
        except Exception as e:
            self._show_status(f"Save error: {e}", error=True)

    def _safe_remove_cbar(self, ax):
        cbar = self._cbars.pop(ax, None)
        if cbar is None:
            return
        try:
            cbar.remove()
        except Exception:
            try:
                if cbar.ax in self.fig.axes:
                    cbar.ax.remove()
            except Exception:
                pass

    def _plot_trace(self, ax, x, y, z, title, norm):
        x = np.asarray(x, dtype=float).ravel()
        y = np.asarray(y, dtype=float).ravel()
        z = np.asarray(z, dtype=float)

        if z.shape == (len(x), len(y)):
            zz = z.T
        elif z.shape == (len(y), len(x)):
            zz = z
        else:
            raise ValueError(f"Trace shape mismatch: z={z.shape}, x={len(x)}, y={len(y)}")

        if isinstance(norm, LogNorm):
            zz = np.ma.masked_less_equal(zz, 0.0)
        im = ax.pcolormesh(x, y, zz, shading="auto", cmap=custom_cmap, norm=norm)
        self._cbars[ax] = self.fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
        ax.set_title(title)
        ax.set_xlabel("Delay (fs)")
        ax.set_ylabel("Frequency (PHz)")

    def _manual_xlim(self):
        tmin = self.time_min.text().strip()
        tmax = self.time_max.text().strip()
        wmin = self.wave_min.text().strip()
        wmax = self.wave_max.text().strip()

        t1 = float(tmin) if tmin else None
        t2 = float(tmax) if tmax else None
        w1 = float(wmin) if wmin else None
        w2 = float(wmax) if wmax else None
        return t1, t2, w1, w2

    def _manual_trace_limits(self):
        dmin = self.delay_min.text().strip()
        dmax = self.delay_max.text().strip()
        fmin = self.freq_min.text().strip()
        fmax = self.freq_max.text().strip()

        d1 = float(dmin) if dmin else None
        d2 = float(dmax) if dmax else None
        f1 = float(fmin) if fmin else None
        f2 = float(fmax) if fmax else None
        return d1, d2, f1, f2

    def redraw(self):
        for ax in (self.ax_exp, self.ax_rec):
            self._safe_remove_cbar(ax)

        # Always clear all plotting axes before redrawing.
        self.ax_exp.clear()
        self.ax_rec.clear()
        self.ax_time.clear()
        self.ax_phase.clear()
        self.ax_spec.clear()

        if self.result_obj is None:
            self.ax_exp.set_title("Experiment")
            self.ax_rec.set_title("Reconstructed")
            self.ax_time.set_title("Time profile")
            self.ax_spec.set_title("Spectra")
            self.ax_spec.set_xlabel("Wavelength (nm)")
            self.ax_spec.set_ylabel("Intensity")
            self.ax_spec.set_ylim(0, 1.4)
            self.ax_spec.grid(True, alpha=0.3)
            try:
                if self.measured_wave is not None and self.measured_intensity is not None:
                    if self.raw_shift.text().strip():
                        try:
                            shift_val = float(self.raw_shift.text().strip())
                            wave = self.measured_wave + shift_val
                        except ValueError:
                            wave = np.asarray(self.measured_wave, dtype=float)
                    else:
                        wave = np.asarray(self.measured_wave, dtype=float)
                    measured = np.asarray(self.measured_intensity, dtype=float)
                    vmax = np.max(measured) if measured.size > 0 else 0.0
                    if vmax > 0:
                        measured = measured / vmax
                    self.ax_spec.plot(wave, measured, "--", lw=2, label="Measured", color="tab:green")
                    self.ax_spec.legend()
                    self.ax_spec.autoscale(enable=True, axis="x")
            except Exception as e:
                self._show_status(f"Measured spectra plot error: {e}", error=True)
            self.fig.tight_layout()
            self.canvas.draw_idle()
            return

        r = self.result_obj
        display_errors = []
        trace_norm = LogNorm(vmin=2e-4, vmax=1.0) if self.trace_log.isChecked() else Normalize(vmin=0.0, vmax=1.0)

        try:
            dmin, dmax, fmin, fmax = self._manual_trace_limits()
        except Exception:
            dmin, dmax, fmin, fmax = None, None, None, None
            display_errors.append("Invalid trace range input.")
        try:
            tmin, tmax, wmin, wmax = self._manual_xlim()
        except Exception:
            tmin, tmax, wmin, wmax = None, None, None, None
            display_errors.append("Invalid time/wavelength range input.")

        exp_ok = False
        rec_ok = False
        if self._has_array_data(r, "frg_delay") and self._has_array_data(r, "frg_freq") and self._has_array_data(r, "frg_trace"):
            try:
                self._plot_trace(self.ax_exp, r.frg_delay, r.frg_freq, r.frg_trace, "Experiment", trace_norm)
                exp_ok = True
            except Exception as e:
                self.ax_exp.set_title(f"Experiment (plot error: {e})")
                display_errors.append(f"Experiment plot error: {e}")
        else:
            self.ax_exp.set_title("Experiment (missing .A.dat)")
            self.ax_exp.set_xlabel("Delay (fs)")
            self.ax_exp.set_ylabel("Frequency (PHz)")

        if self._has_array_data(r, "frg_delay") and self._has_array_data(r, "frg_freq") and self._has_array_data(r, "frg_trace_reconstructed"):
            try:
                self._plot_trace(self.ax_rec, r.frg_delay, r.frg_freq, r.frg_trace_reconstructed, "Reconstructed", trace_norm)
                rec_ok = True
            except Exception as e:
                self.ax_rec.set_title(f"Reconstructed (plot error: {e})")
                display_errors.append(f"Reconstructed plot error: {e}")
        else:
            self.ax_rec.set_title("Reconstructed (missing .Arecon.dat)")
            self.ax_rec.set_xlabel("Delay (fs)")
            self.ax_rec.set_ylabel("Frequency (PHz)")

        for ax, ok in ((self.ax_exp, exp_ok), (self.ax_rec, rec_ok)):
            if not ok:
                continue
            if dmin is not None or dmax is not None:
                xmin, xmax = ax.get_xlim()
                left = dmin if dmin is not None else xmin
                right = dmax if dmax is not None else xmax
                if left < right:
                    ax.set_xlim(left, right)
            if fmin is not None or fmax is not None:
                ymin, ymax = ax.get_ylim()
                bottom = fmin if fmin is not None else ymin
                top = fmax if fmax is not None else ymax
                if bottom < top:
                    ax.set_ylim(bottom, top)

        # Time-domain panel
        self.ax_time.set_title("Time profile")
        self.ax_time.set_xlabel("Time (fs)")
        self.ax_time.set_ylabel("Intensity")
        self.ax_time.set_ylim(0, 1.4)
        self.ax_time.grid(True, alpha=0.3)
        self.ax_phase.yaxis.set_label_position("right")
        self.ax_phase.yaxis.tick_right()
        self.ax_phase.set_ylabel("")
        self.ax_phase.set_yticks([])
        time_handles = []
        time_labels = []
        has_time_trace = False

        if self._has_array_data(r, "time") and self._has_array_data(r, "time_intensity"):
            try:
                pulse = float(getattr(r, "pulse_duration", np.nan))
                lbl = f"Retrieved {pulse:.1f} fs" if np.isfinite(pulse) else "Retrieved"
                h = self.ax_time.plot(r.time, r.time_intensity, "-", lw=1.5, color="black")[0]
                time_handles.append(h)
                time_labels.append(lbl)
                has_time_trace = True
            except Exception as e:
                display_errors.append(f"Retrieved time plot error: {e}")
        if self._has_array_data(r, "time_FTL") and self._has_array_data(r, "time_FTL_intensity"):
            try:
                pulse_ftl = float(getattr(r, "pulse_duration_FTL", np.nan))
                lbl = f"FTL {pulse_ftl:.1f} fs" if np.isfinite(pulse_ftl) else "FTL"
                h = self.ax_time.plot(r.time_FTL, r.time_FTL_intensity, "--", lw=1.5, color="tab:blue")[0]
                time_handles.append(h)
                time_labels.append(lbl)
                has_time_trace = True
            except Exception as e:
                display_errors.append(f"FTL time plot error: {e}")
        if self._has_array_data(r, "time") and self._has_array_data(r, "time_phase"):
            try:
                h = self.ax_phase.plot(r.time, r.time_phase, ":", lw=1.5, color="tab:red")[0]
                time_handles.append(h)
                time_labels.append("Phase")
                self.ax_phase.set_ylabel("Phase")
                self.ax_phase.set_ylim(-np.pi, np.pi)
            except Exception as e:
                display_errors.append(f"Time phase plot error: {e}")
                self.ax_phase.set_ylabel("")
                self.ax_phase.set_yticks([])
        else:
            self.ax_phase.set_ylabel("")
            self.ax_phase.set_yticks([])

        if not has_time_trace:
            self.ax_time.set_title("Time profile (missing .Ek.dat/.Ew.dat)")

        if time_handles:
            self.ax_time.legend(time_handles, time_labels)

        if tmin is not None and tmax is not None and tmin < tmax:
            self.ax_time.set_xlim(tmin, tmax)
        elif has_time_trace:
            base_fwhm = float(getattr(r, "pulse_duration", np.nan))
            if not np.isfinite(base_fwhm) or base_fwhm <= 0:
                base_fwhm = float(getattr(r, "pulse_duration_FTL", np.nan))
            if np.isfinite(base_fwhm) and base_fwhm > 0:
                self.ax_time.set_xlim(-3 * base_fwhm, 3 * base_fwhm)
            else:
                self.ax_time.autoscale(enable=True, axis="x")

        # Spectra panel
        self.ax_spec.set_title("Spectra")
        self.ax_spec.set_xlabel("Wavelength (nm)")
        self.ax_spec.set_ylabel("Intensity")
        self.ax_spec.set_ylim(0, 1.4)
        self.ax_spec.grid(True, alpha=0.3)
        has_spec_curve = False
        try:
            if self._has_array_data(r, "wavelength") and self._has_array_data(r, "wavelength_intensity"):
                self.ax_spec.plot(r.wavelength, r.wavelength_intensity, "-", lw=1.5, label="Retrieved", color="black")
                has_spec_curve = True
        except Exception as e:
            display_errors.append(f"Retrieved spectra plot error: {e}")

        try:
            if self.measured_wave is not None and self.measured_intensity is not None:
                if self.raw_shift.text().strip():
                    try:
                        shift_val = float(self.raw_shift.text().strip())
                        wave = self.measured_wave + shift_val
                    except ValueError:
                        wave = np.asarray(self.measured_wave, dtype=float)
                else:
                    wave = np.asarray(self.measured_wave, dtype=float)
                measured = np.asarray(self.measured_intensity, dtype=float)
                vmax = 0.0
                if measured.size > 0:
                    if wmin is not None and wmax is not None and wmin != wmax:
                        lo, hi = min(wmin, wmax), max(wmin, wmax)
                        mask = (wave >= lo) & (wave <= hi)
                        if np.any(mask):
                            vmax = np.max(measured[mask])
                    if vmax <= 0:
                        vmax = np.max(measured)
                if vmax > 0:
                    measured = measured / vmax
                self.ax_spec.plot(wave, measured, "--", lw=2, label="Measured", color="tab:green")
                has_spec_curve = True
        except Exception as e:
            display_errors.append(f"Measured spectra plot error: {e}")

        if not has_spec_curve:
            self.ax_spec.set_title("Spectra (missing .Speck.dat and raw spectra)")
        elif wmin is not None and wmax is not None and wmin < wmax:
            self.ax_spec.set_xlim(wmin, wmax)
        elif self._has_array_data(r, "wavelength") and self._has_array_data(r, "wavelength_intensity"):
            try:
                _, _, w_fwhm, w_center = find_peak_fwhm(
                    np.asarray(r.wavelength, dtype=float),
                    np.asarray(r.wavelength_intensity, dtype=float),
                )
                #print(f"Auto wavelength range: center={w_center:.2f} nm, FWHM={w_fwhm:.2f} nm")
                self.ax_spec.set_xlim(w_center - 3 * w_fwhm, w_center + 3 * w_fwhm)
            except Exception:
                self.ax_spec.autoscale(enable=True, axis="x")
        else:
            self.ax_spec.autoscale(enable=True, axis="x")

        handles, labels = self.ax_spec.get_legend_handles_labels()
        if handles:
            self.ax_spec.legend()

        if display_errors:
            self._show_status("Display warning: " + " | ".join(display_errors), error=True)

        self.fig.tight_layout()
        self.canvas.draw_idle()


def main():
    app = QtWidgets.QApplication(sys.argv)
    w = FrogResultGUI()
    w.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()

"""FROG Retrieval GUI — unified pipeline for Simulate / Convert / Retrieval / Result.

A single-window PySide6 application with four tabs:

  0) Simulate  — generate simulated FROG traces from user-defined pulses.
  1) Convert   — load raw experimental FROG data, preprocess and bin into .frg format.
  2) Retrieval — run the RANA pulse-retrieval algorithm on a .frg trace.
  3) Result    — visualise retrieved pulse (time/frequency), phase analysis,
                 and dispersion compensation with material databases.

Data flows between tabs via in-memory payloads:
  Simulate -> Convert -> Retrieval -> Result

Usage:
    python frog_gui.py

Dependencies:
    numpy, scipy, matplotlib, pyqtgraph, PySide6
    (optional) Cython extensions for accelerated retrieval — see setup_cython.py

Reference:
    R. Jafari and R. Trebino, IEEE J. Quant. Electr. 56, 1-8 (2020).
    https://frog.gatech.edu/code.html
"""
from __future__ import annotations
import sys
from typing import Optional
from PySide6 import QtCore, QtWidgets
from retrieval_class.frog_convert_gui import FROGConvertGUI
from retrieval_class.frog_retrieval_gui import FrogRetrievalGUI
from retrieval_class.frog_result_gui import FrogResultGUI
from retrieval_class.simulate_frog_gui import SimulateFrogGUI

class FrogPipelineGUI(QtWidgets.QMainWindow):
    """Single-window host that wires data flow across existing GUIs."""

    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("FROG Unified GUI")
        self._initial_width = 1200
        self._initial_height = 800
        self._initial_size_applied = False
        self.resize(self._initial_width, self._initial_height)

        self._latest_convert_payload: Optional[dict] = None
        self._latest_result_payload: Optional[dict] = None
        self._pipeline_action: Optional[str] = None
        self._pipeline_convert_payload_received = False
        self._status_seq = 0

        self.simulate_gui = SimulateFrogGUI()
        self.convert_gui = FROGConvertGUI()
        self.retrieval_gui = FrogRetrievalGUI()
        self.result_gui = FrogResultGUI()
        for _w in (self.simulate_gui, self.convert_gui, self.retrieval_gui, self.result_gui):
            _w.setMinimumSize(0, 0)
            _w.setSizePolicy(
                QtWidgets.QSizePolicy.Policy.Ignored,
                QtWidgets.QSizePolicy.Policy.Ignored,
            )

        self._build_ui()
        self._wire_signals()
        self._set_inline_status("Ready.", timeout_ms=3000)

    def showEvent(self, event) -> None:
        super().showEvent(event)
        if self._initial_size_applied:
            return
        self._initial_size_applied = True
        QtCore.QTimer.singleShot(0, lambda: self.resize(self._initial_width, self._initial_height))

    def _build_ui(self) -> None:
        root = QtWidgets.QWidget()
        self.setCentralWidget(root)
        vbox = QtWidgets.QVBoxLayout(root)
        vbox.setContentsMargins(2, 2, 2, 2)
        vbox.setSpacing(2)
        vbox.setSizeConstraint(QtWidgets.QLayout.SizeConstraint.SetNoConstraint)

        row = QtWidgets.QHBoxLayout()
        row.setContentsMargins(2, 2, 2, 2)
        row.setSpacing(4)
        vbox.addLayout(row)

        self.btn_send_simulate = QtWidgets.QPushButton("Simulate -> Convert")
        self.btn_send_convert = QtWidgets.QPushButton("Convert -> Retrieval")
        self.btn_send_result = QtWidgets.QPushButton("Retrieval -> Result")
        self.btn_convert_retrieval = QtWidgets.QPushButton("Convert + Retrieval")
        self.btn_save_all = QtWidgets.QPushButton("Save All")
        self.btn_run_all = QtWidgets.QPushButton("Run All")
        self.btn_convert_retrieval.setToolTip("Run Convert, then start Retrieval.")
        self.btn_run_all.setToolTip("Run Convert and Retrieval, then Save All.")
        self.chk_auto_convert = QtWidgets.QCheckBox("Auto Convert->Retrieval")
        self.chk_auto_result = QtWidgets.QCheckBox("Auto Retrieval->Result")
        self.chk_auto_convert.setChecked(True)
        self.chk_auto_result.setChecked(True)

        row.addWidget(self.btn_send_simulate)
        row.addWidget(self.btn_send_convert)
        row.addWidget(self.btn_send_result)
        row.addWidget(self.chk_auto_convert)
        row.addWidget(self.chk_auto_result)
        row.addSpacing(8)
        self.pipeline_status_label = QtWidgets.QLabel("Ready.")
        self.pipeline_status_label.setStyleSheet("color:#404040;")
        self.pipeline_status_label.setMinimumWidth(320)
        self.pipeline_status_label.setAlignment(QtCore.Qt.AlignmentFlag.AlignLeft | QtCore.Qt.AlignmentFlag.AlignVCenter)
        row.addWidget(self.pipeline_status_label)
        row.addStretch(1)
        row.addWidget(self.btn_convert_retrieval)
        row.addWidget(self.btn_save_all)
        row.addWidget(self.btn_run_all)
        self.pipeline_controls_layout = row

        self.tabs = QtWidgets.QTabWidget()
        self.tabs.setMinimumSize(0, 0)
        self.tabs.setSizePolicy(
            QtWidgets.QSizePolicy.Policy.Ignored,
            QtWidgets.QSizePolicy.Policy.Ignored,
        )
        self.tabs.tabBar().setExpanding(False)
        self.tabs.setStyleSheet("QTabWidget::tab-bar { alignment: center; }")
        self.tabs.addTab(self.simulate_gui, "0) Simulate")
        self.tabs.addTab(self.convert_gui, "1) Convert")
        self.tabs.addTab(self.retrieval_gui, "2) Retrieval")
        self.tabs.addTab(self.result_gui, "3) Result")
        self.tabs.setCurrentIndex(1)
        vbox.addWidget(self.tabs, 1)

    def _wire_signals(self) -> None:
        self.btn_send_simulate.clicked.connect(self.send_simulate_to_convert)
        self.btn_send_convert.clicked.connect(self.send_convert_to_retrieval)
        self.btn_send_result.clicked.connect(self.send_retrieval_to_result)
        self.btn_convert_retrieval.clicked.connect(self.run_convert_retrieval)
        self.btn_save_all.clicked.connect(self.save_all)
        self.btn_run_all.clicked.connect(self.run_all)
        self.convert_gui.retrieval_data_ready.connect(self._on_convert_payload_ready)
        self.retrieval_gui.result_data_ready.connect(self._on_retrieval_payload_ready)
        self.retrieval_gui.retrieval_failed.connect(self._on_pipeline_retrieval_failed)

    def save_all(self) -> None:
        """Trigger all save actions in the pipeline."""
        self.convert_gui.save_param_btn.click()
        self.convert_gui.save_results()
        self.retrieval_gui.btn_save.click()
        self.result_gui.save_btn.click()
        try:
            if self.result_gui.phase_analysis_widget is not None:
                self.result_gui.phase_analysis_widget.save_phase_analysis()
        except Exception:
            pass

    def run_convert_retrieval(self) -> None:
        self._start_pipeline_action("convert_retrieval")

    def run_all(self) -> None:
        self._start_pipeline_action("run_all")

    def _set_pipeline_action_buttons_enabled(self, enabled: bool) -> None:
        self.btn_convert_retrieval.setEnabled(enabled)
        self.btn_run_all.setEnabled(enabled)

    def _start_pipeline_action(self, action: str) -> None:
        if self._pipeline_action is not None:
            self._set_inline_status("A pipeline run is already active.", timeout_ms=5000, error=True)
            return
        thread = self.retrieval_gui._thread
        if thread is not None and thread.isRunning():
            self._set_inline_status("Retrieval is already running.", timeout_ms=5000, error=True)
            return

        self._pipeline_action = str(action)
        self._pipeline_convert_payload_received = False
        self._set_pipeline_action_buttons_enabled(False)
        self.tabs.setCurrentIndex(1)
        label = "Convert + Retrieval" if action == "convert_retrieval" else "Run All"
        self._set_inline_status(f"{label}: running Convert...")
        self.convert_gui.process_btn.click()

        if self._pipeline_action is not None and not self._pipeline_convert_payload_received:
            self._finish_pipeline_action(
                "Convert did not produce retrieval data.",
                error=True,
            )

    def _finish_pipeline_action(self, text: str, error: bool = False) -> None:
        self._pipeline_action = None
        self._pipeline_convert_payload_received = False
        self._set_pipeline_action_buttons_enabled(True)
        self._set_inline_status(text, timeout_ms=8000, error=error)

    def _save_all_after_pipeline(self) -> None:
        if self._pipeline_action != "run_all":
            return
        try:
            self.save_all()
            self._finish_pipeline_action("Run All completed: Convert, Retrieval, and Save All.")
        except Exception as exc:
            self._finish_pipeline_action(f"Run All save failed: {exc}", error=True)

    def _set_inline_status(self, text: str, timeout_ms: int = 0, error: bool = False) -> None:
        self._status_seq += 1
        seq = self._status_seq
        self.pipeline_status_label.setText(str(text))
        self.pipeline_status_label.setStyleSheet("color:#b00020;" if error else "color:#404040;")
        if timeout_ms and timeout_ms > 0:
            QtCore.QTimer.singleShot(
                int(timeout_ms),
                lambda: self._clear_inline_status_if_current(seq),
            )

    def _clear_inline_status_if_current(self, seq: int) -> None:
        if seq != self._status_seq:
            return
        self.pipeline_status_label.setText("")

    def _on_convert_payload_ready(self, payload_obj: object) -> None:
        if not isinstance(payload_obj, dict):
            return
        self._latest_convert_payload = payload_obj
        if self._pipeline_action is not None:
            self._pipeline_convert_payload_received = True
            if not self.send_convert_to_retrieval(payload_obj):
                self._finish_pipeline_action(
                    "Convert completed, but transfer to Retrieval failed.",
                    error=True,
                )
                return
            self.tabs.setCurrentIndex(2)
            self._set_inline_status("Convert completed. Retrieval is running...")
            self.retrieval_gui.btn_run.click()
            if self.retrieval_gui._thread is None:
                self._finish_pipeline_action(
                    "Retrieval could not be started.",
                    error=True,
                )
            return
        if self.chk_auto_convert.isChecked():
            self.send_convert_to_retrieval(payload_obj)

    def _on_retrieval_payload_ready(self, payload_obj: object) -> None:
        if not isinstance(payload_obj, dict):
            return
        self._latest_result_payload = payload_obj
        if self._pipeline_action == "run_all":
            if not self.send_retrieval_to_result(payload_obj):
                self._finish_pipeline_action(
                    "Retrieval completed, but transfer to Result failed.",
                    error=True,
                )
                return
            self.tabs.setCurrentIndex(3)
            QtCore.QTimer.singleShot(0, self._save_all_after_pipeline)
            return
        if self._pipeline_action == "convert_retrieval":
            if self.chk_auto_result.isChecked():
                self.send_retrieval_to_result(payload_obj)
            self._finish_pipeline_action("Convert + Retrieval completed.")
            return
        if self.chk_auto_result.isChecked():
            self.send_retrieval_to_result(payload_obj)

    def _on_pipeline_retrieval_failed(self, error_text: str) -> None:
        if self._pipeline_action is None:
            return
        summary = str(error_text).strip().splitlines()
        detail = summary[-1] if summary else "unknown error"
        self._finish_pipeline_action(f"Retrieval failed: {detail}", error=True)

    def send_simulate_to_convert(self) -> None:
        payload = self.simulate_gui.export_convert_payload()
        if payload is None:
            QtWidgets.QMessageBox.warning(self, "No Simulation Data", "Please run Simulate first.")
            return
        try:
            self.convert_gui.load_simulate_payload(payload)
            self.tabs.setCurrentIndex(1)
            self._set_inline_status("Sent simulated trace to Convert.", timeout_ms=5000)
        except Exception as e:
            self._set_inline_status(f"Simulate -> Convert failed: {e}", timeout_ms=8000, error=True)
            QtWidgets.QMessageBox.critical(self, "Transfer Error", f"Simulate -> Convert failed:\n{e}")

    def send_convert_to_retrieval(self, payload_obj: Optional[object] = None) -> bool:
        payload = payload_obj if isinstance(payload_obj, dict) else self.convert_gui.export_retrieval_payload()
        if payload is None:
            payload = self._latest_convert_payload
        if payload is None:
            QtWidgets.QMessageBox.warning(self, "No Convert Data", "Please run Process in Convert tab first.")
            return False
        try:
            self.retrieval_gui.load_frg_payload(payload, source_label="from Convert")
            self._set_inline_status("Sent converted trace to Retrieval.", timeout_ms=5000)
            return True
        except Exception as e:
            self._set_inline_status(f"Convert -> Retrieval failed: {e}", timeout_ms=8000, error=True)
            QtWidgets.QMessageBox.critical(self, "Transfer Error", f"Convert -> Retrieval failed:\n{e}")
            return False

    def send_retrieval_to_result(self, payload_obj: Optional[object] = None) -> bool:
        payload = payload_obj if isinstance(payload_obj, dict) else self.retrieval_gui.export_result_payload()
        if payload is None:
            payload = self._latest_result_payload
        if payload is None:
            QtWidgets.QMessageBox.warning(
                self,
                "No Retrieval Result",
                "Please run retrieval in Retrieval tab first.",
            )
            return False
        try:
            self.result_gui.load_result_payload(payload)
            self._set_inline_status("Sent retrieval result to Result viewer.", timeout_ms=5000)
            return True
        except Exception as e:
            self._set_inline_status(f"Retrieval -> Result failed: {e}", timeout_ms=8000, error=True)
            QtWidgets.QMessageBox.critical(self, "Transfer Error", f"Retrieval -> Result failed:\n{e}")
            return False


def main() -> None:
    app = QtWidgets.QApplication(sys.argv)
    w = FrogPipelineGUI()
    w.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()

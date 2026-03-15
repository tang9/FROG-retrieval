# FROG Retrieval

A Python GUI application for Frequency-Resolved Optical Gating (FROG) pulse retrieval, built with PySide6 and pyqtgraph.

Supports **SHG**, **PG**, **THG**, and **SD** FROG geometries.

## Quick start

```bash
pip install numpy scipy matplotlib pyqtgraph PySide6 Cython
python setup_cython.py build_ext --inplace
python frog_gui.py
```

## Pipeline overview

The GUI provides a four-tab workflow. Data flows between tabs via in-memory payloads or can be loaded/saved independently at each stage.

```
Simulate  ->  Convert  ->  Retrieval  ->  Result
```

| Tab | Purpose |
|-----|---------|
| **0) Simulate** | Generate simulated FROG traces from user-defined pulses (Gaussian spectrum or loaded from file). |
| **1) Convert** | Load raw experimental FROG data, subtract background, apply noise filters, and bin into an N×N frequency-domain `.frg` trace. |
| **2) Retrieval** | Run the RANA pulse-retrieval algorithm on a `.frg` trace. Monitor convergence in real time. |
| **3) Result** | Visualise retrieved pulse in time/frequency/wavelength domains. Phase analysis with GDD/TOD fitting and dispersion compensation using material databases. |

Auto-transfer checkboxes (`Convert->Retrieval`, `Retrieval->Result`) are enabled by default for a seamless one-click workflow.

## Retrieval algorithm

The retrieval engine implements the **RANA** (Robust Algorithm for N-dimensional Analysis) method:

> R. Jafari and R. Trebino, "Extremely Robust Pulse Retrieval From Even Noisy Second-Harmonic-Generation Frequency-Resolved Optical Gating Traces," *IEEE J. Quant. Electr.* **56**, 1–8 (2020).

Key features:
- Multi-grid coarse-to-fine retrieval with automatic seed generation
- Geometry-specific gradient descent with exact polynomial line search
- Dual convergence criteria (G error and G' error)
- Weighted zero-pixel handling for sparse/thresholded traces
- Optional Cython acceleration

## Optional Cython acceleration

To build Cython extensions for faster retrieval:

```bash
pip install Cython
python setup_cython.py build_ext --inplace
```

This builds:
- `retrieval_class/retrievers/rana_cython` — accelerated RANA core
- `retrieval_class/field2trace_cython` — accelerated forward model

If not built, pure Python/NumPy fallbacks are used automatically.

## Project structure

```
frog_gui.py                          # Entry point — unified pipeline GUI
retrieval_class/
    __init__.py                      # Package exports
    api.py                           # Top-level retrieve_pulse() API
    types.py                         # Data types (FrogGrid, FrogTrace, RetrievalConfig, ...)
    common.py                        # Shared utilities (plotting, peak fitting, FFT, ...)
    field2trace.py                   # Forward model: E(t) -> FROG trace
    frog_convert.py                  # Raw FROG data processing and binning
    frog_result.py                   # Retrieval result file I/O
    factory.py                       # Retriever factory
    simulate_frog_gui.py             # Tab 0: Simulate GUI
    frog_convert_gui.py              # Tab 1: Convert GUI
    frog_retrieval_gui.py            # Tab 2: Retrieval GUI
    frog_result_gui.py               # Tab 3: Result GUI
    retrievers/
        base.py                      # Base retriever class
        rana.py                       # RANA retriever implementation
        rana_cython.pyx              # Cython-accelerated RANA kernels
refractive_index/                    # Material dispersion data (Sellmeier coefficients)
    FS_dis.py, CaF2_dis.py, ...      # One module per material
setup_cython.py                      # Build script for Cython extensions
```

## Output files

After retrieval, export writes the following files (given a prefix like `trace_binned256`):

| File | Content |
|------|---------|
| `prefix.A.dat` | Measured FROG trace |
| `prefix.Arecon.dat` | Reconstructed FROG trace |
| `prefix.Ek.dat` | Retrieved time-domain field (time, intensity, phase) |
| `prefix.Ew.dat` | Retrieved frequency-domain field (frequency, intensity, phase) |
| `prefix.Speck.dat` | Wavelength-domain spectrum and phase |

## Input file format

### Raw FROG trace (for Convert tab)

Plain text file:
```
delay_1  delay_2  delay_3  ...        # Line 1: delay values
wl_1     wl_2     wl_3     ...        # Line 2: wavelength values (nm)
z_11     z_12     z_13     ...        # Lines 3+: trace matrix (one row per delay)
z_21     z_22     z_23     ...
...
```

### `.frg` file (for Retrieval tab)

```
N_delay  N_freq  dt  dv  v0           # Line 1: grid parameters
z_11     z_12    ...                  # Lines 2+: trace matrix [freq x delay]
...
```

## Requirements

- Python >= 3.10
- numpy, scipy, matplotlib
- PySide6
- pyqtgraph
- Cython (optional, for acceleration)

## License

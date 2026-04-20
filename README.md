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
| **2) Retrieval** | Run the RANA pulse-retrieval algorithm on a `.frg` trace (or .dat file from Trebino export, saved from retrieval). Monitor convergence in real time. |
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

## File formats

### Raw text trace for the Convert tab

The Convert tab reads plain-text experimental traces with this layout:

```text
delay_1  delay_2  delay_3  ...        # first numeric row: delay or motor positions
wl_1     wl_2     wl_3     ...        # second numeric row: wavelength in nm
z_11     z_12     z_13     ...        # remaining rows: one spectrum per delay
z_21     z_22     z_23     ...
...
```

Notes:

- Any title/comment lines before the first numeric row are ignored.
- The first numeric row is interpreted using the selected delay-step unit (`nm`, `um`, `mm`, `m`, `fs`, `ps`, `as`).
- The trace matrix is stored as `N_delay x N_wavelength`.
- Multiple raw files can be loaded and averaged in the Convert tab before binning.

### `.frg` trace for the Retrieval tab

The Retrieval tab reads the `.frg` files written by the converter. The first line always contains five numbers:

```text
N_delay  N_freq  dt  dv  center
```

followed by `N_freq` rows of trace data, each row containing `N_delay` intensity samples.

The active `.frg` file written by the converter is `prefix_binnedN.frg`, a square frequency-domain trace used for retrieval. Here `dt` is in fs/pixel, `dv` is in PHz/pixel, and `center` is the trace-center frequency in PHz.

The matrix in the file is written as fixed-frequency rows and delay columns.

### `.dat` trace for the Retrieval tab

The Retrieval tab can also open the trace format written by the retrieval exporter (`.A.dat` / `.Arecon.dat`):

```text
N_delay  N_freq
z_min    z_max
wl_1
wl_2
...
wl_Nfreq
delay_1
delay_2
...
delay_Ndelay
z_11
z_12
...
```

After the two-line header, the file contains:

- `N_freq` wavelength values in nm
- `N_delay` delay values in fs
- `N_delay * N_freq` trace samples flattened in delay-major order

### Retrieved field profile files

The retrieval exporter writes `Ek.dat`, `Ew.dat`, and `Speck.dat` as five-column text files with no header:

```text
axis    intensity    phase    real(field)    imag(field)
```

where the axis is:

- time in fs for `Ek.dat`
- frequency in PHz for `Ew.dat`
- wavelength in nm for `Speck.dat`

Intensity is normalized to a peak value of 1, and phase is the unwrapped retrieved phase referenced to the peak of the profile.

## Output files

### Convert tab outputs

`Save Convert` writes these files next to the chosen prefix:

| File | Content |
|------|---------|
| `prefix_binnedN.frg` | Square, normalized retrieval input trace (`N x N`) in frequency-delay coordinates |
| `prefix_processed_N{N}.png` | Snapshot of the processed Convert-tab figure |
| `prefix_processed_N{N}.txt` | JSON parameter dump for the current Convert-tab settings |
| `prefix_autocorrelation_N{N}.txt` | Delay-axis autocorrelation export with normalized trace and Gaussian-fit columns |

The converter also writes this auxiliary text export:

| File | Content |
|------|---------|
| `prefix_processed.txtSpecScan` | Processed text trace with delay row, wavelength row, then `N_delay x N_wavelength` matrix |

### Retrieval tab outputs

After retrieval, export writes the following files for a prefix such as `trace_binned256`:

| File | Content |
|------|---------|
| `prefix.A.dat` | Measured FROG trace in retrieval-export `.dat` format |
| `prefix.Arecon.dat` | Reconstructed FROG trace in the same `.dat` format |
| `prefix.Ek.dat` | Retrieved time-domain field: time, normalized intensity, phase, real(E), imag(E) |
| `prefix.Ew.dat` | Retrieved frequency-domain field: frequency, normalized intensity, phase, real(E), imag(E) |
| `prefix.Speck.dat` | Retrieved wavelength-domain spectrum: wavelength, normalized intensity, phase, real(E), imag(E) |

## Requirements

- Python >= 3.10
- numpy, scipy, matplotlib
- PySide6
- pyqtgraph
- Cython (optional, for acceleration)

## License

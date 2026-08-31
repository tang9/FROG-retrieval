# Version Log

## Unreleased — 2026-08-15

### Unified GUI

- Added `Convert + Retrieval` to the left of `Save All`, which runs data conversion followed by pulse retrieval.
- Added `Run All` to the right of `Save All`, which runs Convert, Retrieval, Result data hand-off and Save All in sequence.
- The chained workflow waits for the asynchronous Retrieval to finish before continuing; buttons are disabled during the run to prevent re-entry, and are restored with an error message on failure.
- `Save All` now handles the embedded Phase Analysis page.

### Convert Panel

- Moved `Load Param` and `Save Param` above `Save Default Param`.
- Merged `time_min` and `time_max` onto the same row as `delay range`, and removed the duplicated labels.
- Adjusted the proportions and spacing of the Parameters panel, labels and input area for a more compact input layout.
- Unified the width of the bottom action buttons; the buttons keep a fixed width when the window is stretched, with the extra space placed between `Trace log scale` and `Show 1D`.
- Replaced the `cutoff ratio` field with separate `LP delay scale (fs)` and `LP wavelength scale (nm)` fields, so the low-pass scale can be controlled independently along the delay and wavelength axes.
- When loading legacy parameter files, `cutoff`, `cutoff_ratio` or `cutoff ratio` is automatically converted to the new delay/wavelength low-pass scales; the conversion is deferred until data is loaded, preserving backward compatibility.

### Retrieval

- Added an optional `delay_smearing (fs)` parameter; the Gaussian instrument response along the delay axis is applied only when the box is checked.
- Both the Python and Cython compute paths support delay smearing, and stay consistent in the error calculation and in the final reconstructed trace.
- Optimized the error calculation, array handling and polynomial term accumulation in the Python path, reducing redundant computation and temporary arrays.
- Applied the corresponding optimizations to the Cython path, including reusing the error context, merging the G/G′ calculation, and performing the delay-axis convolution in compiled code.

### Result and Phase Analysis

- Moved Phase Analysis into the second tab of the FROG Result window, removing the standalone button and pop-up dialog.
- Phase Analysis plotting now uses Matplotlib and supports saving the analysis figure directly.
- Changing `time_min` or `time_max` in Parameters now refreshes the Phase Analysis time range accordingly.
- Fixed the overlapping dual y-axis labels in the left plot.
- Unified the y-axis range of the left-plot spectral intensity and the right-plot time-domain intensity profile to `0–1.4`.

### Output Paths

- Unified the save-path generation logic across Convert, Retrieval and Result.
- Results already located inside a `retrieval_result` directory reuse the existing directory, avoiding repeatedly nested `retrieval_result/retrieval_result` folders.

### Compatibility and Validation

- Legacy low-pass `cutoff ratio` parameter files can still be loaded and are migrated automatically.
- The Cython-generated C file has been regenerated accordingly.
- Automated tests cover low-pass parameter migration, delay smearing, Python/Cython numerical consistency, output paths, the Phase Analysis tab and the unified run workflow.
- Current test results: 22 of 22 passing.

# hbmlens roadmap

## Direction

hbmlens aims to be an open reference toolkit for HBM validation that runs ahead
of physical hardware: test definitions, a virtual device to run them on, the same
tests on real GPUs, and analysis that reads failures like an inspection image.
Everything is built from public information (JEDEC-level descriptions, published
papers, open-source simulators and datasets).

## Status (v0.1-dev, 2026-09-29)

Working:

- HBM organization presets (tiny, small, illustrative HBM3 16 GiB / HBM4 32 GiB)
  following the Channel > PseudoChannel > Sid > BankGroup > Bank > Row > Column hierarchy
- Pluggable address mappers (linear, GPU-like interleave with bank XOR hashing)
- Pattern spec + library: MATS+, March C-, walking ones, checkerboard,
  moving inversions, random data, retention
- Fault model with ground truth: stuck cell, row, column, DQ lane, temperature-dependent retention
- Virtual backend (NumPy) and CUDA backend (CuPy kernels) with exact fail counts
  and full fail records (no fixed record cap)
- Smoke results: CUDA and virtual backends report identical failing words for the
  same injected faults; a 1 GiB March C- pass on a laptop GPU takes about 0.05 s
- Analyzer: failing (word, bit) cells classified into DQ lane, row, column, bank and
  isolated cell signatures; recovers every injected fault in 48 randomized placements
  (faults hidden inside a larger same-bit fault are reported as masked)
- Fail bitmaps per signature (only that signature's bits, so a column inside a failing
  DQ lane stays visible), markdown report, signatures.json
- CLI: `hbmlens demo`, `hbmlens run --backend virtual|cuda [--inject]`, `hbmlens analyze`
- 35 tests, including CUDA/virtual parity on four patterns

## P0: 3D visualization of the virtual device (required)

The device under test does not exist physically, so seeing it is mandatory.
The three.js viewer (`viewer.html` written by `demo` and `analyze`) shows stacked core
dies over a base die with TSV pillars, bank tiles, failing words coloured by fault kind
(ground truth, or analyzer signatures for real logs), analyzer outlines, hover
coordinates, click-to-zoom full-resolution bank bitmaps, a replay slider over pattern
steps and a die spacing slider. Still to build:

- Temperature and pattern comparisons side by side
- Animated replay of address order inside an element (not only first-failing step)
- Per-bit view for words with several failing bits

## P1

- Coverage matrix: which pattern catches which fault model
- Unit tests for faults and both backends (CUDA tests marked `cuda`)

## P2

- Fail-map classifier in TensorFlow (public WM-811K wafer maps + synthetic HBM fail maps)
- Telemetry scenarios with DCGM NVML injection (ECC, row remap, Xid) and anomaly detection
- LLM agent that drafts a failure analysis report from logs and fail maps
- Coupling faults and read disturbance (Hammer) in the virtual device, calibrated
  with public HBM2 characterization data
- Coalesced CUDA access pattern (current per-thread chunking leaves bandwidth unused)
- Runs on rented HBM GPUs (GH200 / B200) with ECC and row-remap telemetry
- Next-generation (HBM4 and beyond) test definitions from public specifications

## Principles

- Public information only; no affiliation with any memory or GPU vendor
- Injected faults and measured results are always labeled separately
- Real GPU address maps are unknown; every analysis states the mapper it assumed

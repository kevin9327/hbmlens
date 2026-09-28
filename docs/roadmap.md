# hbmlens roadmap

## Direction

hbmlens aims to be an open reference toolkit for HBM validation that runs ahead
of physical hardware: test definitions, a virtual device to run them on, the same
tests on real GPUs, and analysis that reads failures like an inspection image.
Everything is built from public information (JEDEC-level descriptions, published
papers, open-source simulators and datasets).

## Status (v0.1-dev, 2026-09-28)

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

## P0: 3D visualization of the virtual device (required)

The device under test does not exist physically, so seeing it is mandatory.
Build a browser viewer (three.js / WebGL):

- HBM cube as stacked dies over a base die; channels and pseudo channels as
  columns through the stack; banks as tiles on each die
- Fail cells and signatures (row, column, DQ lane, bank) highlighted in place
- Replay of a pattern run over time (which elements hit which banks)
- Input: exported FailLog (Parquet/JSON) plus geometry and mapper description
- Temperature and pattern comparisons side by side

## P1

- Analyzer: decode fail records to coordinates, classify signatures, fail bitmaps, report
- CLI: `hbmlens demo`, `run`, `analyze`
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

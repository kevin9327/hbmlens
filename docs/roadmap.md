# hbmlens roadmap

## Direction

hbmlens aims to be an open reference toolkit for HBM validation that runs ahead
of physical hardware: test definitions, a virtual device to run them on, the same
tests on real GPUs, and analysis that reads failures like an inspection image.
Everything is built from public information (JEDEC-level descriptions, published
papers, open-source simulators and datasets).

## Status (v0.1-dev, 2026-09-29)

Working:

- HBM organization presets (tiny, small, medium 4 GiB for GPU runs, illustrative HBM3
  16 GiB / HBM4 32 GiB) following the Channel > PseudoChannel > Sid > BankGroup > Bank >
  Row > Column hierarchy
- Pluggable address mappers (linear, GPU-like interleave with bank XOR hashing)
- Pattern spec + library: MATS+, March C-, March SS, word-oriented March C- (6 data
  backgrounds), intra-word coupling test, walking ones, checkerboard, moving inversions,
  random data, retention, and cuda_memtest-style re-implementations for comparison
- Fault model with ground truth: stuck cell, row, column, DQ lane, temperature-dependent retention
- Virtual backend (NumPy) and CUDA backend (CuPy kernels) with exact fail counts
  and full fail records up to a configurable capacity (default 1,048,576); overflowing
  logs are flagged in the CLI and the report
- DRAM-faithful GPU execution: elements with repeated accesses to a word run op-major
  over chunks 8x the L2 size, and L2 is emptied when the sweep direction turns, so
  every op reaches DRAM (a single pass would exceed the bus peak, i.e. hit the cache)
- CUDA and virtual backends report identical fail records for the same injected faults
  (all patterns, all access modes, chunked and tail regions)
- Coverage measurement (`hbmlens coverage`, docs/coverage.md): 17 static fault models,
  bit-exact, guaranteed detection over every initial state and address order, suites
  run back to back; hbmlens suite (March SS + intra-word + retention) 100% with
  51 ops/word vs a cuda_memtest-style suite 82.5% mean with 289 ops/word; exhaustive
  search for the cheapest complete suite (docs/vs-free-tools.md)
- Analyzer: failing (word, bit) cells classified into DQ lane, row, column, bank and
  isolated cell signatures; recovers every injected fault in 48 randomized placements
  (faults hidden inside a larger same-bit fault are reported as masked)
- Fail bitmaps per signature (only that signature's bits, so a column inside a failing
  DQ lane stays visible), markdown report, signatures.json
- CLI: `hbmlens demo`, `hbmlens run --backend virtual|cuda [--inject] [--max-records]`,
  `hbmlens analyze`, `hbmlens coverage`
- Benchmarks: `benchmarks/gpu_bandwidth.py`, `benchmarks/suite_time.py`
- 77 tests, including CUDA/virtual record parity and textbook coverage results

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

- Overflow-proof capture: per-row and per-column fail masks built on the GPU (mapper
  decoded in the kernel), so analysis stays exact when millions of words fail (a dead
  DQ lane on 4 GiB fails 32 Mi words; today the analyzer sees only the recorded part)
- Dynamic faults (March RAW), linked faults and neighbourhood pattern-sensitive faults
  in the coverage simulator
- DRAM traffic confirmed with hardware counters (Nsight Compute dram bytes), not only
  timing against the bus peak
- Same comparison on real hardware: the other tools' own binaries on the same GPU,
  wall-clock per GiB

## P2

- Fail-map classifier in TensorFlow (public WM-811K wafer maps + synthetic HBM fail maps)
- Telemetry scenarios with DCGM NVML injection (ECC, row remap, Xid) and anomaly detection
- LLM agent that drafts a failure analysis report from logs and fail maps
- Coupling faults and read disturbance (Hammer) in the virtual device, calibrated
  with public HBM2 characterization data
- Runs on rented HBM GPUs (GH200 / B200) with ECC and row-remap telemetry
- Next-generation (HBM4 and beyond) test definitions from public specifications

## Principles

- Public information only; no affiliation with any memory or GPU vendor
- Injected faults and measured results are always labeled separately
- Real GPU address maps are unknown; every analysis states the mapper it assumed

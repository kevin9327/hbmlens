# Design: core axes and quality bar

hbmlens is built along four axes. Each has a quality bar that every change must keep.

## 1. Fault science (the core)

Faults are fault primitives `<S/F/R>` in the notation of the memory test literature:
static and dynamic, single- and two-cell, with the DRAM-specific attributes partial,
dirty, soft and transient. `hbmlens.fpsim` proves which faults a march test detects,
exhaustively over placements, initial values and address orders
([fault-primitives.md](fault-primitives.md)).

Bar: every published result hbmlens cites is reproduced by a test in `tests/test_fp.py`
(so far: VTS 2002 Table 4, 64 of 64 values; DATE 2006 Table 6, 12 of 12; the
completeness claims of March SS, RAW1 and RAW). Where hbmlens disagrees with a
publication, the difference is traced to a stated semantic, shown with a trace, and a
fix is proven (March H1C+/T1C+, the even hammer count for H2C).

## 2. Device model

A virtual HBM device: organization (stack, channel, pseudo channel, bank group, bank,
row, column), address maps, injected faults with ground truth, retention that depends on
temperature.

Bar: parameters come from public sources and say so; injected faults are always
labelled synthetic; the address map in use is stated with every result.

## 3. Execution

The same test runs on the virtual device and on GPUs (CUDA).

Bar: GPU and virtual device agree record for record on the same injected faults; every
memory operation reaches DRAM (elements that would hit the cache run op-major, checked
against the bus peak); failure counts are exact and an overflowing log is never silent.

## 4. Reading failures

Fail signatures (DQ lane, row, column, bank, cell), fail bitmaps, a report and a
three.js view of the device.

Bar: recovery is measured against injected ground truth; results that depend on the
address map say which map they assumed.

## Engineering

Lint (`ruff`) and the test suite run on every push; numbers in `docs/` are regenerated
by the command named next to them (`hbmlens coverage`, `hbmlens fp-coverage`,
`benchmarks/*.py`), never typed in.

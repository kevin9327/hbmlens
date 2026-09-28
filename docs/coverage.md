# Fault coverage of memory test patterns

Measured with `hbmlens coverage --trials 2000 --seed 0`: every fault model is placed at 2000 random locations in a 1024-word memory and simulated bit-exactly. A fault counts as detected only if every run order allowed by the pattern catches it. `ops/word` is the number of memory reads and writes per word (the cost of the test).

Fault models: SAF, TF, CFin, CFid, CFst, CFin-intra, CFid-intra, AF, DRF (see `hbmlens/coverage.py`). The cuda_memtest-style patterns are re-implemented from that project's public test list; they are not its original code.

| pattern | ops/word | SAF | TF | CFin | CFid | CFst | CFin-intra | CFid-intra | AF | DRF |
|---|---|---|---|---|---|---|---|---|---|---|
| mats-plus | 5 | 100% | 50.9% | 74.4% | 38.6% | 74.4% | 51.7% | 25.9% | 100% | 0% |
| march-c-minus | 10 | 100% | 100% | 100% | 100% | 100% | 100% | 50.0% | 100% | 0% |
| walking-ones | 64 | 100% | 98.6% | 0% | 0% | 71.9% | 98.7% | 47.8% | 0% | 0% |
| checkerboard | 4 | 100% | 74.5% | 0% | 0% | 50.7% | 77.5% | 38.0% | 48.9% | 0% |
| moving-inversions | 12 | 100% | 100% | 100% | 52.6% | 76.6% | 100% | 49.5% | 100% | 0% |
| random | 4 | 100% | 75.7% | 0% | 0% | 50.7% | 76.3% | 39.0% | 100% | 0% |
| retention | 4 | 100% | 100% | 0% | 0% | 48.9% | 100% | 50.0% | 0% | 100% |
| march-c-minus-wom | 60 | 100% | 100% | 100% | 100% | 100% | 100% | 100% | 100% | 0% |
| own-address | 2 | 48.3% | 7.7% | 0% | 0% | 24.7% | 7.4% | 3.4% | 100% | 0% |
| mi-ones-zeros | 10 | 100% | 100% | 100% | 76.4% | 100% | 100% | 50.0% | 100% | 0% |
| mi-8bit | 40 | 100% | 100% | 100% | 85.9% | 100% | 100% | 89.9% | 100% | 0% |
| mi-random | 5 | 100% | 75.9% | 74.6% | 43.2% | 81.6% | 75.5% | 39.0% | 100% | 0% |
| mi-32bit | 160 | 100% | 100% | 100% | 87.1% | 100% | 100% | 98.4% | 100% | 0% |
| bit-fade | 4 | 100% | 100% | 0% | 0% | 48.9% | 100% | 50.0% | 0% | 100% |
| **suite: cuda_memtest-style (8 tests)** | 289 | **100%** | **100%** | **100%** | **99.8%** | **100%** | **100%** | **100%** | **100%** | **100%** |
| **suite: hbmlens full: march-c-minus-wom + retention** | 64 | **100%** | **100%** | **100%** | **100%** | **100%** | **100%** | **100%** | **100%** | **100%** |
| **suite: hbmlens quick: march-c-minus + retention** | 14 | **100%** | **100%** | **100%** | **100%** | **100%** | **100%** | **50.0%** | **100%** | **100%** |

## Reading the table

- The hbmlens full suite detects every fault in every model with 64 operations per word; the cuda_memtest-style suite needs 289 and still misses some idempotent coupling faults (CFid).
- The missed class: an aggressor cell at a lower address rises 0->1 and forces the same bit of a higher word to 0 (or falls 1->0 and forces it to 1). With the same data in every word, moving inversions never read the victim after it was flipped against its current value; March C-'s down(r0,w1) element does. Reproduce: `tests/test_coverage.py::test_moving_inversions_suite_misses_this_idempotent_coupling`.
- Plain March C- catches all bit-level faults but only half of the intra-word idempotent coupling faults; running it over the six word-oriented data backgrounds closes that gap.
- Retention faults need a pause; the retention and bit-fade patterns are the only ones with one.

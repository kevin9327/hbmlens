# Fault coverage of memory test patterns

Measured with `hbmlens coverage --trials 2000 --seed 0`. Each fault model is placed at 2000 random locations in a 1024-word memory of 32-bit words and every pattern is simulated bit-exactly (`hbmlens/coverage.py`). A fault counts as detected only if detection is guaranteed: for every initial value of the bits it involves (memory content before a test is unknown) and for both address orders wherever the pattern allows either. A suite is simulated as one program, its patterns back to back in the listed order, the way a test tool runs it.

Fault models (the static fault taxonomy of the memory test literature):

- `SAF` stuck-at: a cell always reads 0 (or 1)
- `TF` transition: a cell cannot go 0->1 (or 1->0)
- `WDF` write destructive: writing the value a cell already holds flips it
- `RDF` read destructive: a read flips the cell and returns the flipped value
- `DRDF` deceptive read destructive: a read returns the right value but flips the cell
- `IRF` incorrect read: a read returns the wrong value, the cell keeps its value
- `CFin` inversion coupling: a transition in another word inverts a victim bit
- `CFid` idempotent coupling: a transition in another word forces a victim bit to 0/1
- `CFst` state coupling: while a bit of another word holds a value, a victim bit is forced
- `CFds` disturb coupling: reading another word, or rewriting its value, flips a victim bit
- `CFwd` write destructive coupling: WDF on the victim while an aggressor bit holds a value
- `CFdrd` deceptive read destructive coupling: DRDF on the victim while an aggressor bit holds a value
- `CFin-intra` inversion coupling between two bits of the same word
- `CFid-intra` idempotent coupling between two bits of the same word
- `CFst-intra` state coupling between two bits of the same word
- `AF` address decoder: two addresses reach the same word
- `DRF` data retention: a cell leaks to its discharged value during a pause (>= 30 s here)

The cuda_memtest-style patterns are re-implemented from that project's public test list; they are not its original code.

## Suites

| | cuda_memtest-style (8 tests) | March C- + retention | word-oriented March C- + retention | hbmlens: march-ss + intra-word + retention |
|---|---|---|---|---|
| memory ops per word | 289 | 14 | 64 | 51 |
| pause time | 3 h | 128 s | 128 s | 128 s |
| SAF | 100% | 100% | 100% | 100% |
| TF | 100% | 100% | 100% | 100% |
| WDF | **79.5%** | **0%** | **76.5%** | 100% |
| RDF | 100% | 100% | 100% | 100% |
| DRDF | **0%** | **0%** | **0%** | 100% |
| IRF | 100% | 100% | 100% | 100% |
| CFin | 100% | 100% | 100% | 100% |
| CFid | **99.6%** | 100% | 100% | 100% |
| CFst | 100% | 100% | 100% | 100% |
| CFds | **63.1%** | **49.5%** | **49.5%** | 100% |
| CFwd | **60.6%** | **0%** | **40.2%** | 100% |
| CFdrd | **0%** | **0%** | **0%** | 100% |
| CFin-intra | 100% | 100% | 100% | 100% |
| CFid-intra | 100% | **48.0%** | 100% | 100% |
| CFst-intra | 100% | **49.6%** | 100% | 100% |
| AF | 100% | 100% | 100% | 100% |
| DRF | 100% | 100% | 100% | 100% |
| **mean over models** | **82.5%** | **67.5%** | **80.4%** | **100%** |

Cheapest combination of library patterns that detects every simulated fault, each pattern on its own and in any order (exhaustive search, `hbmlens.coverage.cheapest_suite`): **retention + march-ss + intra-word**, 51 memory operations per word.

## Reading the table

- Deceptive read destructive faults (DRDF, CFdrd) return the right value and flip the cell, so only a second read of the same cell sees them. March C- and moving inversions always write right after a read, which hides the flip. March SS reads twice in a row.
- Write destructive faults (WDF, CFwd) need a write of the value a cell already holds. March C- never does that on purpose; suites that change data between tests or backgrounds do it for some bits by accident, which is why they land between 0% and 100%.
- March SS covers every single-cell and two-cell static fault in a bit-oriented memory. In a 32-bit word, two bits that always hold the same value are never tested against each other; the 25N intra-word test adds five data backgrounds for that.
- Retention faults need a pause. The DRF model here leaks after 30 s, so a 64 s pause and a 90 min pause both catch it; cells that leak between the two are only caught by the longer pause (`retention(pause_s)` is configurable).
- Idempotent coupling and moving inversions: an aggressor at a lower address that rises and forces the same bit of a higher word to 0 (or falls and forces 1) is missed with solid data (`tests/test_coverage.py::test_moving_inversions_suite_misses_this_idempotent_coupling`).

## Every pattern

| pattern | ops/word | SAF | TF | WDF | RDF | DRDF | IRF | CFin | CFid | CFst | CFds | CFwd | CFdrd | CFin-intra | CFid-intra | CFst-intra | AF | DRF |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mats-plus | 5 | 100% | 51.0% | 0% | 100% | 0% | 100% | 74.4% | 37.4% | 74.1% | 17.9% | 0% | 0% | 48.6% | 24.4% | 49.6% | 100% | 0% |
| march-c-minus | 10 | 100% | 100% | 0% | 100% | 0% | 100% | 100% | 100% | 100% | 49.5% | 0% | 0% | 100% | 48.0% | 49.6% | 100% | 0% |
| walking-ones | 64 | 100% | 96.9% | 49.1% | 100% | 0% | 100% | 0% | 0% | 71.3% | 0% | 46.1% | 0% | 97.4% | 46.7% | 75.9% | 0% | 0% |
| checkerboard | 4 | 100% | 49.0% | 0% | 100% | 0% | 100% | 0% | 0% | 50.1% | 0% | 0% | 0% | 50.5% | 24.8% | 51.0% | 52.2% | 0% |
| moving-inversions | 12 | 100% | 100% | 51.1% | 100% | 0% | 100% | 100% | 49.1% | 74.9% | 32.5% | 24.9% | 0% | 100% | 50.1% | 49.1% | 100% | 0% |
| random | 4 | 100% | 49.4% | 0% | 100% | 0% | 100% | 0% | 0% | 48.6% | 0% | 0% | 0% | 50.2% | 25.8% | 50.2% | 100% | 0% |
| retention | 4 | 100% | 48.9% | 0% | 100% | 0% | 100% | 0% | 0% | 48.8% | 0% | 0% | 0% | 51.3% | 23.6% | 49.6% | 0% | 100% |
| march-c-minus-wom | 60 | 100% | 100% | 68.6% | 100% | 0% | 100% | 100% | 100% | 100% | 49.5% | 33.9% | 0% | 100% | 100% | 100% | 100% | 0% |
| march-ss | 22 | 100% | 100% | 100% | 100% | 100% | 100% | 100% | 100% | 100% | 100% | 100% | 100% | 100% | 48.0% | 49.6% | 100% | 0% |
| intra-word | 25 | 100% | 100% | 0% | 100% | 0% | 100% | 0% | 0% | 94.0% | 0% | 0% | 0% | 100% | 98.1% | 98.3% | 0% | 0% |
| own-address | 2 | 50.1% | 0% | 0% | 50.2% | 0% | 51.1% | 0% | 0% | 25.2% | 0% | 0% | 0% | 0% | 0% | 25.6% | 100% | 0% |
| mi-ones-zeros | 10 | 100% | 100% | 0% | 100% | 0% | 100% | 100% | 75.0% | 100% | 36.4% | 0% | 0% | 100% | 48.0% | 49.6% | 100% | 0% |
| mi-8bit | 40 | 100% | 100% | 49.1% | 100% | 0% | 100% | 100% | 86.6% | 100% | 52.4% | 42.3% | 0% | 100% | 90.1% | 95.4% | 100% | 0% |
| mi-random | 5 | 100% | 50.4% | 0% | 100% | 0% | 100% | 73.5% | 39.9% | 74.5% | 18.6% | 0% | 0% | 49.8% | 23.8% | 50.0% | 100% | 0% |
| mi-32bit | 160 | 100% | 100% | 49.1% | 100% | 0% | 100% | 100% | 87.2% | 100% | 54.6% | 47.9% | 0% | 100% | 98.6% | 100% | 100% | 0% |
| bit-fade | 4 | 100% | 48.9% | 0% | 100% | 0% | 100% | 0% | 0% | 48.8% | 0% | 0% | 0% | 51.3% | 23.6% | 49.6% | 0% | 100% |

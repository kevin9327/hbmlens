# hbmlens compared with free GPU memory test tools

Free tools in this space include the DCGM diagnostic memory plugins, cuda_memtest,
memtest_vulkan and gpu-burn. They answer "does this GPU pass?". hbmlens is built to
also answer "how good is the test, did every access really test DRAM, what exactly
failed, and why". Every number below was measured with the command shown or read
from a tool's public source; nothing is estimated.

## 1. More faults found with under a fifth of the memory operations

`hbmlens coverage --trials 2000` places 17 fault models from the memory test
literature (stuck-at, transition, read and write destructive, incorrect read,
inversion / idempotent / state / disturb coupling between and inside words,
address decoder, data retention) at random locations and simulates every pattern
bit-exactly. A fault only counts if it is caught for every initial memory content
and every allowed address order; a suite runs as one program, its patterns back
to back ([full tables](coverage.md)).

| suite | memory ops per word | pauses | mean detection over 17 models | worst models |
|---|---|---|---|---|
| hbmlens: March SS + intra-word + retention | 51 | 128 s | **100%** | all 100% |
| cuda_memtest-style: 8 of its tests, re-implemented from the public test list | 289 | 3 h | 82.5% | deceptive read destructive 0% (single cell and coupled), disturb coupling 63%, write destructive coupling 61%, write destructive 80% |

The zeros are structural: a deceptive read destructive cell returns the right
value and flips afterwards, so only a second read of the same word sees it, and
tests that always write right after reading never do that. The hbmlens suite is
also the cheapest combination of library patterns that detects every simulated
fault (exhaustive search, `hbmlens.coverage.cheapest_suite`).

## 2. Every memory operation reaches DRAM

On a GPU, a kernel that reads a word twice, or reads it back right after writing
it, in the same pass gets the second access from registers or the L2 cache, and
L2 cannot be bypassed for device memory. Such a test checks the cache, not DRAM.
It shows in the timing: the run finishes faster than its accesses could cross the
memory bus.

`python benchmarks/suite_time.py 4` on a laptop RTX 5080 (48 MiB L2, bus peak
~896 GB/s), 4 GiB region; "ops as GB/s" is all reads and writes of the pattern
divided by the run time:

| pattern (elements with repeated accesses) | one pass per element | hbmlens |
|---|---|---|
| March SS (r,r,w,r,w) | 0.059 s = 1610 GB/s, above the bus peak | 0.128 s = 739 GB/s |
| intra-word (w,w,r,w,r) | 0.043 s = 2495 GB/s, above the bus peak | 0.147 s = 730 GB/s |
| walking ones (w,r per word) | 0.197 s = 1395 GB/s, above the bus peak | 0.424 s = 649 GB/s |

hbmlens runs such elements op-major: chunks 8x the size of L2, visited in the
element's direction, one full pass per op, and L2 is emptied whenever the sweep
direction turns. Each cell still sees its ops in order, which is the virtual
device's semantics; a test checks the GPU against the virtual device record for
record. Elements shaped (r), (w) or (r, w) already send every access to DRAM and
keep the single fast pass. For regions smaller than 4x L2, `hbmlens run` says
that some accesses may be served from cache.

## 3. Every failure is kept

DCGM's memtest plugin counts all errors but keeps address and data for only the
last 10 (`RECORD_ERR` writes slot `count % MAX_ERR_RECORD_COUNT`, defined as 10,
in `plugin_src/memtest/tests.cu` and `misc.h`). hbmlens keeps the exact failure
count and a full record (address, expected, actual, element, op, iteration) for
every failure up to a configurable capacity (default 1,048,576), and says when it
overflowed.

## 4. Time per suite on the same GPU

Same benchmark, both suites on hbmlens kernels, 4 GiB:

| suite | memory passes | pauses |
|---|---|---|
| cuda_memtest-style (8 tests) | 1.87 s | 10,800 s (bit fade, 2 x 90 min) |
| hbmlens | 0.30 s | 128 s (retention, 2 x 64 s) |

The DRF model leaks after 30 s, so both pauses catch it; cells that leak between
64 s and 90 min are only caught by the longer pause, which hbmlens can run too
(`retention(pause_s)`).

Throughput of single patterns (`python benchmarks/gpu_bandwidth.py 4`, best of 5
after 1 s of warm-up): March C- 715 GB/s, write + read 671 GB/s, moving
inversions 638 GB/s, random data 597 GB/s. Short runs on this laptop vary by up
to 20% between invocations. The other tools' own throughput was not measured.

## 5. Failures are read, not just counted

- Failing bits are classified into DQ lane, row, column, bank and isolated cell
  signatures; on injected faults the classifier recovered every fault in 48
  randomized populations (12 of them run in the test suite), reported with
  recall and precision.
- Fail bitmaps per signature, a markdown report, and a three.js 3D view of the
  device with click-to-zoom bank maps and a step-by-step replay.
- The same fault set runs on the virtual device and on a real GPU (as a read
  overlay), and both report identical fail records, so the whole chain can be
  validated against ground truth.

## Limits

- The fault models are the standard static abstractions; dynamic (multi-operation)
  faults, linked faults and physical effects such as read disturbance in real HBM
  are not modelled yet.
- That every op reaches DRAM is shown by timing against the bus peak, not yet by
  hardware DRAM counters.
- The GPU address map is not public; results that need physical coordinates state
  the mapper they assumed.
- Patterns labelled cuda_memtest-style follow that project's published test list;
  they are not its original code, and 3 of its 11 tests are not included.

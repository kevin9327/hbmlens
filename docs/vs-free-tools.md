# hbmlens compared with free GPU memory test tools

Free tools in this space: the DCGM diagnostic memory plugins (NVVS), cuda_memtest,
memtest_vulkan and gpu-burn. They answer "does this GPU pass?". hbmlens is built to
also answer "how good is the test, what exactly failed, and why". Every number below
was measured with the commands shown or read from the tools' public source; nothing
is estimated.

## 1. Same coverage with 4.5x less memory traffic

`hbmlens coverage --trials 2000` places nine classic fault models (stuck-at,
transition, inversion / idempotent / state coupling between and inside words,
address decoder, data retention) at random locations and simulates each pattern
bit-exactly ([full table](coverage.md)).

| suite | memory ops per word | faults detected |
|---|---|---|
| hbmlens full (word-oriented March C- + retention) | 64 | 100% in all nine models |
| cuda_memtest-style (its 8 test descriptions, re-implemented) | 289 | 99.8% of idempotent coupling, 100% elsewhere |

The missed faults form one class (a rising aggressor at a lower address forcing the
same bit of a higher word to 0, or the opposite polarity), which solid-data moving
inversions cannot expose. A regression test reproduces it.

## 2. Every failure is kept

DCGM's memtest plugin counts all errors but keeps address and data for only the
last 10 (`MAX_ERR_RECORD_COUNT 10`, a ring buffer in
`nvvs/plugin_src/memtest/tests.cu`). hbmlens keeps the exact failure count and a
full record (address, expected, actual, element, op, iteration) for every failure
up to a configurable capacity (default 1,048,576), and says when it overflowed.

## 3. Fast kernels

On a laptop RTX 5080 (`python benchmarks/gpu_bandwidth.py 4`, 4 GiB region):

| pattern | hbmlens GB/s |
|---|---|
| write + read | 828 |
| March C- | 726 |
| random data | 640 |
| moving inversions | 609 |

For reference, a CuPy device-to-device copy on the same GPU measured 592 GB/s.
The other tools' throughput was not measured on this machine.

## 4. Failures are read, not just counted

- Failing bits are classified into DQ lane, row, column, bank and isolated cell
  signatures; on injected faults the classifier recovers every fault across 48
  randomized placements (reported with recall and precision).
- Fail bitmaps per signature, a markdown report, and a three.js 3D view of the
  device with click-to-zoom bank maps and a step-by-step replay.
- The same fault set runs on the virtual device and on a real GPU (as a read
  overlay), and both report identical failing words, so the whole chain can be
  validated against ground truth.

## Limits

- The fault models are the standard abstractions; physical effects such as read
  disturbance in real HBM are only partly modelled so far.
- The GPU address map is not public; results that need physical coordinates state
  the mapper they assumed.
- Patterns labelled cuda_memtest-style follow that project's published test list;
  they are not its original code.

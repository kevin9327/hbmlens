"""Wall-clock time of whole test suites on the same GPU, and a check that every op reaches DRAM.

For each pattern, "fused" applies all ops of an element to a word in one pass; the
default ("vector") runs elements such as (r, r, w, r, w) op-major so each op is a
DRAM access. If a run finishes faster than all its ops could cross the memory bus,
some ops were served from cache: the "ops/s as bytes" column then exceeds the
device's peak bandwidth.

Pauses (retention, bit fade) are not slept here; they are reported separately
because they cost the same on any device.

Run: python benchmarks/suite_time.py [GiB]
"""
import sys
import time
import warnings

warnings.simplefilter("ignore")

import cupy as cp  # noqa: E402

from hbmlens.backends.cuda import CudaBackend  # noqa: E402
from hbmlens.patterns.base import Pattern, Pause  # noqa: E402
from hbmlens.patterns.library import CUDA_MEMTEST_STYLE, HBMLENS_SUITE, get_pattern  # noqa: E402

gib = float(sys.argv[1]) if len(sys.argv) > 1 else 4.0
words = int(gib * 2**30) // 4
SUITES = {
    "cuda_memtest-style (8 tests)": CUDA_MEMTEST_STYLE,
    "hbmlens: " + " + ".join(HBMLENS_SUITE): HBMLENS_SUITE,
}

props = cp.cuda.runtime.getDeviceProperties(0)
peak = props["memoryBusWidth"] / 8 * props["memoryClockRate"] * 1e3 * 2 / 1e9
print(f"device: {props['name'].decode()}, L2 {props['l2CacheSize'] / 2**20:g} MiB, "
      f"peak ~{peak:.0f} GB/s (bus width x clock x 2), region {gib:g} GiB")
devs = {a: CudaBackend(words, "bench", access=a) for a in ("fused", "vector")}


def best_time(dev, pat):
    t_end = time.perf_counter() + 1.0  # let a laptop GPU leave its idle clocks
    while time.perf_counter() < t_end:
        dev.run(pat)
    return min(dev.run(pat).meta.elapsed_s for _ in range(5))


print(f"{'pattern':16s} {'ops/word':>8s} | {'fused s':>8s} {'ops as GB/s':>11s} | "
      f"{'hbmlens s':>9s} {'ops as GB/s':>11s}")
for suite, names in SUITES.items():
    total, pause, ops_total = 0.0, 0.0, 0
    for name in names:
        p = get_pattern(name)
        pause += p.iterations * sum(s.seconds for s in p.steps if isinstance(s, Pause))
        ops = p.ops_per_word() * p.iterations
        ops_total += ops
        mem = Pattern(p.name, tuple(s for s in p.steps if not isinstance(s, Pause)), p.iterations)
        tf, tv = best_time(devs["fused"], mem), best_time(devs["vector"], mem)
        total += tv
        gbs = ops * words * 4 / 1e9  # all reads and writes of the pattern, in GB
        print(f"{name:16s} {ops:8d} | {tf:8.3f} {gbs / tf:11.0f} | {tv:9.3f} {gbs / tv:11.0f}")
    print(f"== {suite}: {ops_total} ops/word, memory passes {total:.2f} s (+{pause:g} s pauses)\n")

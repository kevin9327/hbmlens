"""GPU memory-test throughput: hbmlens kernels vs a plain device copy on the same GPU.

Each figure is the best of 5 runs after 1 s of warm-up.

Run: python benchmarks/gpu_bandwidth.py [GiB]
"""
import sys
import time
import warnings

warnings.simplefilter("ignore")

import cupy as cp  # noqa: E402

from hbmlens.backends.cuda import CudaBackend  # noqa: E402
from hbmlens.patterns.base import ZERO, Element, Pattern, Read, Write  # noqa: E402
from hbmlens.patterns.library import get_pattern  # noqa: E402

gib = float(sys.argv[1]) if len(sys.argv) > 1 else 4.0
words = int(gib * 2**30) // 4


def copy_ceiling() -> float:
    a = cp.zeros(words, cp.uint32)
    b = cp.empty_like(a)
    t_end = time.perf_counter() + 1.0
    while time.perf_counter() < t_end:
        cp.copyto(b, a)
        cp.cuda.Device().synchronize()
    dt = float("inf")
    for _ in range(5):
        t = time.perf_counter()
        cp.copyto(b, a)
        cp.cuda.Device().synchronize()
        dt = min(dt, time.perf_counter() - t)
    del a, b
    cp.get_default_memory_pool().free_all_blocks()
    return 2 * words * 4 / dt / 1e9


print(f"device: {cp.cuda.runtime.getDeviceProperties(0)['name'].decode()}, region {gib:g} GiB")
print(f"reference: device-to-device copy (read+write) {copy_ceiling():.0f} GB/s")
write_read = Pattern("write+read", (Element("any", (Write(ZERO),)), Element("any", (Read(ZERO),))))
patterns = [("write+read", write_read), ("march-c-minus", get_pattern("march-c-minus")),
            ("random", get_pattern("random")), ("moving-inversions", get_pattern("moving-inversions"))]
for access in ("ordered", "vector"):
    dev = CudaBackend(words, "bench", access=access)
    for name, pat in patterns:
        t_end = time.perf_counter() + 1.0  # let a laptop GPU leave its idle clocks
        while time.perf_counter() < t_end:
            dev.run(pat)
        log = min((dev.run(pat) for _ in range(5)), key=lambda lg: lg.meta.elapsed_s)
        print(f"{access:8s} {name:18s} {log.meta.elapsed_s:7.3f}s {log.meta.bandwidth_gbps:6.0f} GB/s"
              f"  fails={log.meta.total_fails}")
    del dev
    cp.get_default_memory_pool().free_all_blocks()

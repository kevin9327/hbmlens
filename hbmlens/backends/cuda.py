"""Run patterns on a real GPU buffer with CUDA kernels (via CuPy).

Each thread walks a contiguous chunk of words in the element's direction and
applies the element's ops at every word before moving on, so march order is
preserved inside each chunk; chunks themselves run in parallel. Every failing
read increments an exact 64-bit counter and, while capacity remains, stores a
full record (index, expected, actual, element, op, iteration).

Mapping GPU buffer indices to HBM coordinates is an assumption (see
hbmlens.mapping); real GPU address maps are not public.
"""
from __future__ import annotations

import time
import uuid

import numpy as np

from ..patterns.base import Element, Hammer, Pattern, Pause, Read
from ..records import FailLog, RunMeta, empty_records
from .base import ReadOverlay, UnsupportedStep

KERNEL_SRC = r"""
typedef unsigned int u32;
typedef unsigned long long u64;

__device__ __forceinline__ u32 hash32(u64 i, u32 seed) {
    u32 x = (u32)i ^ ((u32)(i >> 32) * 0x85EBCA6Bu) ^ seed;
    x ^= x >> 16; x *= 0x7FEB352Du; x ^= x >> 15; x *= 0x846CA68Bu; x ^= x >> 16;
    return x;
}

// kind: 0 solid, 1 checker, 2 addr, 3 random
__device__ __forceinline__ u32 background(int kind, u32 value, u32 seed, int invert, u64 i) {
    u32 v;
    if (kind == 0)      v = value;
    else if (kind == 1) v = (i & 1ull) ? ~value : value;
    else if (kind == 2) v = (u32)i;
    else                v = hash32(i, seed);
    return invert ? ~v : v;
}

__device__ __forceinline__ u32 overlay(u32 v, u64 i, const u64* oidx, const u32* oand,
                                       const u32* oor, long long n) {
    long long lo = 0, hi = n - 1;
    while (lo <= hi) {
        long long mid = (lo + hi) >> 1;
        u64 m = oidx[mid];
        if (m == i) return (v & oand[mid]) | oor[mid];
        if (m < i) lo = mid + 1; else hi = mid - 1;
    }
    return v;
}

extern "C" __global__ void run_element(
    u32* mem, u64 start, u64 n, u64 chunk, int down,
    int nops, const int* op_read, const int* op_kind, const u32* op_value,
    const u32* op_seed, const int* op_invert,
    const u64* oidx, const u32* oand, const u32* oor, long long on,
    u64* rec_index, u32* rec_exp, u32* rec_act, unsigned short* rec_elem,
    unsigned short* rec_op, u32* rec_iter, u64* count, u64 cap,
    int element, u32 iteration)
{
    u64 tid = (u64)blockIdx.x * blockDim.x + threadIdx.x;
    u64 c0 = tid * chunk;
    if (c0 >= n) return;
    u64 c1 = c0 + chunk; if (c1 > n) c1 = n;
    for (u64 k = 0; k < c1 - c0; ++k) {
        u64 off = down ? (c1 - 1 - k) : (c0 + k);
        u64 i = start + off;
        for (int o = 0; o < nops; ++o) {
            u32 want = background(op_kind[o], op_value[o], op_seed[o], op_invert[o], i);
            if (op_read[o]) {
                u32 got = mem[off];
                if (on > 0) got = overlay(got, i, oidx, oand, oor, on);
                if (got != want) {
                    u64 slot = atomicAdd(count, 1ull);
                    if (slot < cap) {
                        rec_index[slot] = i; rec_exp[slot] = want; rec_act[slot] = got;
                        rec_elem[slot] = (unsigned short)element; rec_op[slot] = (unsigned short)o;
                        rec_iter[slot] = iteration;
                    }
                }
            } else {
                mem[off] = want;
            }
        }
    }
}

#define RECORD(I, WANT, GOT, O) do { \
    u64 slot_ = atomicAdd(count, 1ull); \
    if (slot_ < cap) { rec_index[slot_] = (I); rec_exp[slot_] = (WANT); rec_act[slot_] = (GOT); \
        rec_elem[slot_] = (unsigned short)element; rec_op[slot_] = (unsigned short)(O); rec_iter[slot_] = iteration; } \
} while (0)

// Coalesced variant: consecutive threads own consecutive 16-byte groups (4 words) and
// move through the region with a grid stride, so every warp issues full 128-bit
// transactions. Inside a group, each op is applied to the 4 words before the next op.
extern "C" __global__ void run_element_vec(
    u32* mem, u64 start, u64 n, int down,
    int nops, const int* op_read, const int* op_kind, const u32* op_value,
    const u32* op_seed, const int* op_invert,
    const u64* oidx, const u32* oand, const u32* oor, long long on,
    u64* rec_index, u32* rec_exp, u32* rec_act, unsigned short* rec_elem,
    unsigned short* rec_op, u32* rec_iter, u64* count, u64 cap,
    int element, u32 iteration)
{
    const u64 stride = (u64)gridDim.x * blockDim.x;
    const u64 tid = (u64)blockIdx.x * blockDim.x + threadIdx.x;
    const u64 ng = n >> 2;
    uint4* vmem = reinterpret_cast<uint4*>(mem);
    for (u64 k = tid; k < ng; k += stride) {
        const u64 gi = down ? (ng - 1 - k) : k;
        const u64 base = start + (gi << 2);
        for (int o = 0; o < nops; ++o) {
            const int kind = op_kind[o], inv = op_invert[o];
            const u32 val = op_value[o], seed = op_seed[o];
            u32 w[4];
            #pragma unroll
            for (int j = 0; j < 4; ++j) w[j] = background(kind, val, seed, inv, base + j);
            if (op_read[o]) {
                const uint4 v = vmem[gi];
                u32 g[4] = {v.x, v.y, v.z, v.w};
                #pragma unroll
                for (int j = 0; j < 4; ++j) {
                    u32 got = g[j];
                    if (on > 0) got = overlay(got, base + j, oidx, oand, oor, on);
                    if (got != w[j]) RECORD(base + j, w[j], got, o);
                }
            } else {
                vmem[gi] = make_uint4(w[0], w[1], w[2], w[3]);
            }
        }
    }
    const u64 t0 = ng << 2;  // tail words when n is not a multiple of 4
    if (tid < n - t0) {
        const u64 off = t0 + tid, i = start + off;
        for (int o = 0; o < nops; ++o) {
            const u32 want = background(op_kind[o], op_value[o], op_seed[o], op_invert[o], i);
            if (op_read[o]) {
                u32 got = mem[off];
                if (on > 0) got = overlay(got, i, oidx, oand, oor, on);
                if (got != want) RECORD(i, want, got, o);
            } else {
                mem[off] = want;
            }
        }
    }
}
"""

_KIND = {"solid": 0, "checker": 1, "addr": 2, "random": 3}


def available() -> bool:
    try:
        import cupy as cp  # noqa: F401

        return cp.cuda.runtime.getDeviceCount() > 0
    except Exception:
        return False


class CudaBackend:
    """``words`` 32-bit words are allocated on the GPU and treated as the device under test."""

    name = "cuda"

    def __init__(self, words: int, geometry_name: str = "custom", overlay: ReadOverlay | None = None,
                 access: str = "vector", chunk: int = 256, threads_per_block: int = 256, device: int = 0):
        """``access="vector"`` (default): coalesced 128-bit accesses with a grid stride, fastest.
        ``access="ordered"``: each thread walks a contiguous chunk strictly in march order."""
        import cupy as cp

        if access not in ("vector", "ordered"):
            raise ValueError("access must be 'vector' or 'ordered'")
        self.cp = cp
        self.words = int(words)
        self.geometry_name = geometry_name
        self.access = access
        self.chunk = chunk
        self.tpb = threads_per_block
        cp.cuda.Device(device).use()
        free, _ = cp.cuda.runtime.memGetInfo()
        if self.words * 4 > free * 0.8:
            raise MemoryError(f"{self.words * 4 / 2**30:.2f} GiB requested, {free / 2**30:.2f} GiB free")
        self.mem = cp.zeros(self.words, dtype=cp.uint32)
        module = cp.RawModule(code=KERNEL_SRC)
        self._k_ordered = module.get_function("run_element")  # compiled here, not in the first timed run
        self._k_vector = module.get_function("run_element_vec")
        ov = overlay or ReadOverlay.empty()
        self._ov = (cp.asarray(ov.indices, dtype=cp.uint64), cp.asarray(ov.and_mask, dtype=cp.uint32),
                    cp.asarray(ov.or_mask, dtype=cp.uint32), len(ov.indices))
        props = cp.cuda.runtime.getDeviceProperties(device)
        self._device_name = props["name"].decode()
        self._sms = int(props["multiProcessorCount"])

    def info(self) -> dict:
        return {"backend": self.name, "device": self._device_name, "words": self.words,
                "bytes": self.words * 4, "overlay_words": self._ov[3], "access": self.access}

    def run(self, pattern: Pattern, *, region: tuple[int, int] | None = None,
            max_records: int = 1 << 20, run_id: str | None = None) -> FailLog:
        cp = self.cp
        start, stop = region or (0, self.words)
        n = stop - start
        cap = int(max_records)
        rec = {
            "index": cp.zeros(max(cap, 1), cp.uint64), "expected": cp.zeros(max(cap, 1), cp.uint32),
            "actual": cp.zeros(max(cap, 1), cp.uint32), "element": cp.zeros(max(cap, 1), cp.uint16),
            "op": cp.zeros(max(cap, 1), cp.uint16), "iteration": cp.zeros(max(cap, 1), cp.uint32),
        }
        count = cp.zeros(1, cp.uint64)
        view = self.mem[start:stop]
        vector = self.access == "vector" and start % 4 == 0  # 16-byte aligned view
        if vector:
            blocks = max(1, min((n // 4 + self.tpb - 1) // self.tpb, self._sms * 32))
        else:
            blocks = (n + self.chunk * self.tpb - 1) // (self.chunk * self.tpb)
        # upload the op tables once, outside the timed loop
        arr = lambda xs, dt: cp.asarray(np.array(xs, dtype=dt))  # noqa: E731
        tables = {}
        for si, step in enumerate(pattern.steps):
            if isinstance(step, Hammer):
                raise UnsupportedStep("Hammer needs a known physical address map; not supported on GPUs")
            if isinstance(step, Element):
                ops = step.ops
                tables[si] = (np.int32(len(ops)), arr([isinstance(o, Read) for o in ops], np.int32),
                              arr([_KIND[o.background.kind] for o in ops], np.int32),
                              arr([o.background.value & 0xFFFFFFFF for o in ops], np.uint32),
                              arr([o.background.seed & 0xFFFFFFFF for o in ops], np.uint32),
                              arr([o.invert for o in ops], np.int32))
        tail = (self._ov[0], self._ov[1], self._ov[2], np.int64(self._ov[3]),
                rec["index"], rec["expected"], rec["actual"], rec["element"], rec["op"], rec["iteration"],
                count, np.uint64(cap))
        bytes_moved = 0
        cp.cuda.Device().synchronize()
        t0 = time.perf_counter()
        for it in range(pattern.iterations):
            for si, step in enumerate(pattern.steps):
                if isinstance(step, Pause):
                    cp.cuda.Device().synchronize()
                    time.sleep(step.seconds)
                    continue
                down = np.int32(step.order == "down")
                ids = (np.int32(si), np.uint32(it))
                if vector:
                    self._k_vector((blocks,), (self.tpb,),
                                   (view, np.uint64(start), np.uint64(n), down, *tables[si], *tail, *ids))
                else:
                    self._k_ordered((blocks,), (self.tpb,),
                                    (view, np.uint64(start), np.uint64(n), np.uint64(self.chunk), down,
                                     *tables[si], *tail, *ids))
                bytes_moved += n * 4 * len(step.ops)
        cp.cuda.Device().synchronize()
        elapsed = time.perf_counter() - t0
        total = int(count.get()[0])
        kept = min(total, cap)
        records = empty_records(kept)
        for name in records.dtype.names:
            records[name] = rec[name][:kept].get()
        order = np.argsort(records["index"], kind="stable")
        meta = RunMeta(run_id=run_id or uuid.uuid4().hex[:8], pattern=pattern.name, backend=self.name,
                       geometry=self.geometry_name, words_tested=n, total_fails=total, capacity=cap,
                       overflow=total > cap, elapsed_s=elapsed,
                       bandwidth_gbps=bytes_moved / elapsed / 1e9 if elapsed > 0 else None,
                       notes={"device": self._device_name, "access": "vector" if vector else "ordered"})
        return FailLog(meta, records[order])

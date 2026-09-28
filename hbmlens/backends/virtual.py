"""NumPy virtual HBM device.

v0.1 executes each march element op-by-op over the whole region (vectorized).
That is equivalent to the per-address march order as long as faults do not
couple different cells, which holds for every fault type in :mod:`hbmlens.faults`.
"""
from __future__ import annotations

import time
import uuid

import numpy as np

from ..faults import FaultSet
from ..geometry import HBMGeometry, get_geometry
from ..patterns.base import Element, Hammer, Pattern, Pause, Read, background_values
from ..records import FailLog, RunMeta, empty_records
from .base import UnsupportedStep

CHUNK = 1 << 22  # words per vectorized chunk


class VirtualBackend:
    name = "virtual"

    def __init__(self, geometry: "str | HBMGeometry", faults: FaultSet | None = None,
                 temperature_c: float = 25.0):
        self.geometry = get_geometry(geometry)
        self.faults = faults or FaultSet()
        self.temperature_c = temperature_c
        self.memory = np.zeros(self.geometry.total_words, dtype=np.uint32)
        self.overlay = self.faults.to_overlay()
        ret = self.faults.retention
        self._ret_idx = ret.indices
        self._ret_bit = ret.bits.astype(np.uint32)
        self._ret_leak = ret.leak_value.astype(np.uint32)
        self._ret_limit = ret.retention_s(temperature_c)
        self._ret_written = np.zeros(len(ret.indices))  # simulated time of last write
        self.now = 0.0  # simulated seconds

    def info(self) -> dict:
        return {"backend": self.name, "geometry": self.geometry.describe(),
                "temperature_c": self.temperature_c, "faults": self.faults.describe()}

    # ---- device model ----------------------------------------------------
    def _write(self, idx: np.ndarray, values: np.ndarray) -> None:
        self.memory[idx] = values
        if len(self._ret_idx):
            hit = np.isin(self._ret_idx, idx)
            self._ret_written[hit] = self.now

    def _read(self, idx: np.ndarray) -> np.ndarray:
        values = self.overlay.apply(idx.astype(np.uint64), self.memory[idx])
        if len(self._ret_idx):
            leaked = (self.now - self._ret_written) > self._ret_limit
            if leaked.any():
                pos = np.searchsorted(idx, self._ret_idx[leaked].astype(idx.dtype))
                inside = pos < len(idx)
                pos, cells = pos[inside], np.flatnonzero(leaked)[inside]
                match = idx[pos] == self._ret_idx[cells].astype(idx.dtype)
                pos, cells = pos[match], cells[match]
                mask = np.uint32(1) << self._ret_bit[cells]
                values[pos] = (values[pos] & ~mask) | (self._ret_leak[cells] << self._ret_bit[cells])
                self.memory[idx[pos]] = values[pos]  # a leaked cell stays leaked
        return values

    # ---- execution -------------------------------------------------------
    def run(self, pattern: Pattern, *, region: tuple[int, int] | None = None,
            max_records: int = 1 << 20, run_id: str | None = None) -> FailLog:
        start, stop = region or (0, self.geometry.total_words)
        rec_parts: list[np.ndarray] = []
        total = 0
        t0 = time.perf_counter()
        for it in range(pattern.iterations):
            for si, step in enumerate(pattern.steps):
                if isinstance(step, Pause):
                    self.now += step.seconds
                    continue
                if isinstance(step, Hammer):
                    raise UnsupportedStep("Hammer is not implemented in hbmlens v0.1")
                assert isinstance(step, Element)
                for oi, op in enumerate(step.ops):
                    for c0 in range(start, stop, CHUNK):
                        idx = np.arange(c0, min(c0 + CHUNK, stop), dtype=np.int64)
                        want = background_values(op.background, idx.astype(np.uint64), op.invert)
                        if isinstance(op, Read):
                            got = self._read(idx)
                            bad = np.flatnonzero(got != want)
                            if bad.size:
                                total += bad.size
                                room = max_records - sum(len(p) for p in rec_parts)
                                keep = bad[:max(room, 0)]
                                r = empty_records(len(keep))
                                r["index"] = idx[keep]
                                r["expected"], r["actual"] = want[keep], got[keep]
                                r["element"], r["op"], r["iteration"] = si, oi, it
                                rec_parts.append(r)
                        else:
                            self._write(idx, want)
        elapsed = time.perf_counter() - t0
        records = np.concatenate(rec_parts) if rec_parts else empty_records()
        meta = RunMeta(run_id=run_id or uuid.uuid4().hex[:8], pattern=pattern.name, backend=self.name,
                       geometry=self.geometry.name, words_tested=stop - start,
                       temperature_c=self.temperature_c, total_fails=total, capacity=max_records,
                       overflow=total > len(records), elapsed_s=elapsed)
        return FailLog(meta, records)

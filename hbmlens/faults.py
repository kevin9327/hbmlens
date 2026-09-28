"""Synthetic HBM faults with ground-truth labels.

v0.1 models cell-independent faults only (no coupling between cells):

* ``cell``      one stuck bit in one word
* ``row``       one bit stuck in every word of a row (e.g. a broken local wordline segment)
* ``column``    one bit stuck in one column of a bank across all rows (bitline / sense amp)
* ``dq-lane``   one data bit stuck in every word of a pseudo channel (TSV / IO lane)
* ``retention`` cells that lose their charge after a pause; retention time halves
  every 10 degC above 25 degC (a common DRAM rule of thumb)

Static faults (everything except retention) compile into a :class:`ReadOverlay`,
so the same fault set can be laid over a real GPU buffer.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .backends.base import ReadOverlay
from .geometry import LEVELS
from .mapping import LinearMapper

FULL = 0xFFFFFFFF


@dataclass
class Fault:
    kind: str
    indices: np.ndarray  # uint64 word indices
    bit: int
    stuck_value: int  # 0 or 1
    location: dict  # ground truth, e.g. {"bank_key": 3, "row": 17}

    def label(self) -> str:
        loc = ", ".join(f"{k}={v}" for k, v in self.location.items())
        return f"{self.kind}[{loc}] bit{self.bit} stuck-at-{self.stuck_value} ({len(self.indices)} words)"


@dataclass
class RetentionCells:
    indices: np.ndarray  # uint64
    bits: np.ndarray  # uint8
    t25_s: np.ndarray  # retention time at 25 degC, seconds
    leak_value: np.ndarray  # value the cell decays to (0 or 1)

    @classmethod
    def empty(cls) -> "RetentionCells":
        return cls(np.zeros(0, np.uint64), np.zeros(0, np.uint8), np.zeros(0), np.zeros(0, np.uint8))

    def retention_s(self, temperature_c: float) -> np.ndarray:
        return self.t25_s * 2.0 ** (-(temperature_c - 25.0) / 10.0)


@dataclass
class FaultSet:
    faults: list[Fault] = field(default_factory=list)
    retention: RetentionCells = field(default_factory=RetentionCells.empty)

    def to_overlay(self) -> ReadOverlay:
        if not self.faults:
            return ReadOverlay.empty()
        idx = np.concatenate([f.indices for f in self.faults]).astype(np.uint64)
        and_m = np.concatenate(
            [np.full(len(f.indices), FULL & ~(1 << f.bit) if f.stuck_value == 0 else FULL, np.uint32) for f in self.faults]
        )
        or_m = np.concatenate(
            [np.full(len(f.indices), (1 << f.bit) if f.stuck_value else 0, np.uint32) for f in self.faults]
        )
        order = np.argsort(idx, kind="stable")
        idx, and_m, or_m = idx[order], and_m[order], or_m[order]
        # merge several faults on the same word
        uniq, start = np.unique(idx, return_index=True)
        and_merged = np.bitwise_and.reduceat(and_m, start)
        or_merged = np.bitwise_or.reduceat(or_m, start)
        return ReadOverlay(uniq, and_merged, or_merged)

    def describe(self) -> list[str]:
        out = [f.label() for f in self.faults]
        if len(self.retention.indices):
            out.append(f"retention: {len(self.retention.indices)} weak cells")
        return out


class FaultFactory:
    """Places faults at random physical locations using an address mapper."""

    def __init__(self, mapper: LinearMapper, seed: int = 0):
        self.mapper = mapper
        self.g = mapper.geometry
        self.rng = np.random.default_rng(seed)
        self.sizes = self.g.level_sizes()

    def _rand_fields(self) -> dict:
        return {lvl: int(self.rng.integers(self.sizes[lvl])) for lvl in LEVELS}

    def _indices(self, fixed: dict, free: tuple[str, ...]) -> np.ndarray:
        grids = np.meshgrid(*[np.arange(self.sizes[l]) for l in free], indexing="ij") if free else []
        n = grids[0].size if free else 1
        fields = {l: np.full(n, fixed[l]) for l in LEVELS if l not in free}
        for l, gr in zip(free, grids):
            fields[l] = gr.ravel()
        return np.sort(self.mapper.encode(fields).astype(np.uint64))

    def _bit_value(self) -> tuple[int, int]:
        return int(self.rng.integers(32)), int(self.rng.integers(2))

    def cell(self) -> Fault:
        f = self._rand_fields()
        bit, val = self._bit_value()
        return Fault("cell", self._indices(f, ()), bit, val, {"index": int(self.mapper.encode(f))})

    def row(self) -> Fault:
        f = self._rand_fields()
        bit, val = self._bit_value()
        loc = {l: f[l] for l in LEVELS[:-3]} | {"row": f["row"]}
        return Fault("row", self._indices(f, ("column", "word")), bit, val, loc)

    def column(self) -> Fault:
        f = self._rand_fields()
        bit, val = self._bit_value()
        loc = {l: f[l] for l in LEVELS[:-3]} | {"column": f["column"]}
        return Fault("column", self._indices(f, ("row", "word")), bit, val, loc)

    def dq_lane(self) -> Fault:
        f = self._rand_fields()
        bit, val = self._bit_value()
        loc = {"stack": f["stack"], "channel": f["channel"], "pseudo_channel": f["pseudo_channel"]}
        free = ("sid", "bank_group", "bank", "row", "column", "word")
        return Fault("dq-lane", self._indices(f, free), bit, val, loc)

    def retention_cells(self, n: int, t25_s: tuple[float, float] = (200.0, 2000.0)) -> RetentionCells:
        idx = np.unique(self.rng.integers(0, self.g.total_words, size=n).astype(np.uint64))
        k = len(idx)
        lo, hi = np.log(t25_s[0]), np.log(t25_s[1])
        return RetentionCells(
            idx,
            self.rng.integers(0, 32, size=k).astype(np.uint8),
            np.exp(self.rng.uniform(lo, hi, size=k)),
            self.rng.integers(0, 2, size=k).astype(np.uint8),
        )


def demo_faults(mapper: LinearMapper, seed: int = 7) -> FaultSet:
    """A mixed fault population used by ``hbmlens demo`` and the tests."""
    fac = FaultFactory(mapper, seed)
    faults = [fac.dq_lane()] + [fac.row() for _ in range(2)] + [fac.column() for _ in range(2)]
    faults += [fac.cell() for _ in range(12)]
    return FaultSet(faults, fac.retention_cells(40))

"""Contract shared by all execution backends."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np

from ..patterns.base import Pattern
from ..records import FailLog


class UnsupportedStep(RuntimeError):
    """Raised when a backend cannot execute a pattern step (e.g. Hammer on real GPUs)."""


@dataclass(frozen=True)
class ReadOverlay:
    """Synthetic defects laid over real memory reads.

    Used to validate the pipeline on healthy hardware: for every read of
    ``indices[i]`` the backend returns ``(value & and_mask[i]) | or_mask[i]``.
    A stuck-at-0 bit b is ``and_mask = ~(1<<b), or_mask = 0``; stuck-at-1 is
    ``and_mask = 0xFFFFFFFF, or_mask = 1<<b``. ``indices`` must be sorted and unique.
    """

    indices: np.ndarray  # uint64, sorted, unique
    and_mask: np.ndarray  # uint32
    or_mask: np.ndarray  # uint32

    def __post_init__(self) -> None:
        idx = np.asarray(self.indices, dtype=np.uint64)
        if idx.size and np.any(np.diff(idx.astype(np.int64)) <= 0):
            raise ValueError("overlay indices must be sorted and unique")
        object.__setattr__(self, "indices", idx)
        object.__setattr__(self, "and_mask", np.asarray(self.and_mask, dtype=np.uint32))
        object.__setattr__(self, "or_mask", np.asarray(self.or_mask, dtype=np.uint32))
        if not (len(self.indices) == len(self.and_mask) == len(self.or_mask)):
            raise ValueError("overlay arrays must have equal length")

    @classmethod
    def empty(cls) -> "ReadOverlay":
        return cls(np.zeros(0, np.uint64), np.zeros(0, np.uint32), np.zeros(0, np.uint32))

    def apply(self, index: np.ndarray, values: np.ndarray) -> np.ndarray:
        """Reference implementation (NumPy)."""
        if not len(self.indices):
            return values
        pos = np.searchsorted(self.indices, np.asarray(index, dtype=np.uint64))
        pos_c = np.minimum(pos, len(self.indices) - 1)
        hit = self.indices[pos_c] == np.asarray(index, dtype=np.uint64)
        out = values.copy()
        out[hit] = (out[hit] & self.and_mask[pos_c[hit]]) | self.or_mask[pos_c[hit]]
        return out


class Backend(Protocol):
    name: str

    def run(
        self,
        pattern: Pattern,
        *,
        region: tuple[int, int] | None = None,
        max_records: int = 1 << 20,
        run_id: str | None = None,
    ) -> FailLog:
        """Execute ``pattern`` over ``region`` = [start, stop) word indices (default: all)."""

    def info(self) -> dict:
        """Device / configuration description for reports."""

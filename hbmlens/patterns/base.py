"""Backend-independent description of memory test patterns.

A pattern is a list of steps. The main step is a *march element*: visit every
word of the region in a given order and, at each word, apply a short list of
read/write operations before moving to the next word. This is the notation used
by the memory-test literature, e.g. March C-:

    {any(w0); up(r0,w1); up(r1,w0); down(r0,w1); down(r1,w0); any(r0)}

Data written or expected is produced by a :class:`Background`, a pure function
of the flat word index. Every backend (NumPy virtual device, CUDA kernels) must
implement :func:`background_values` bit-exactly, which is what makes results
comparable across backends.

Semantics every backend must follow:

* ``up`` visits indices in ascending order, ``down`` descending, ``any`` in an
  implementation-defined order (parallel backends may run chunks concurrently;
  see the backend docs for how strictly order is preserved).
* At each index, ops run in list order. A read compares the value read with the
  expected background value; a mismatch produces one fail record with
  ``element`` = step index and ``op`` = op index. The memory keeps whatever value
  was read (no correction).
* :class:`Pause` waits (virtual devices advance simulated time; real devices sleep),
  which exposes retention failures.
* :class:`Hammer` activates chosen rows many times to provoke read disturbance.
  It needs a physical address map and is only supported by the virtual backend.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal, Union

import numpy as np

MASK32 = np.uint32(0xFFFFFFFF)

BackgroundKind = Literal["solid", "checker", "addr", "random"]


@dataclass(frozen=True)
class Background:
    """Data background.

    * ``solid``   every word = ``value``
    * ``checker`` even words = ``value``, odd words = ``~value``
    * ``addr``    word = low 32 bits of its own index (address-in-address test)
    * ``random``  word = ``hash32(index, seed)``; reproducible pseudo random data
    """

    kind: BackgroundKind = "solid"
    value: int = 0
    seed: int = 0

    def label(self) -> str:
        if self.kind == "solid":
            return f"0x{self.value:08X}"
        if self.kind == "checker":
            return f"checker(0x{self.value:08X})"
        if self.kind == "random":
            return f"random(seed={self.seed})"
        return "addr"


ZERO = Background("solid", 0x00000000)
ONES = Background("solid", 0xFFFFFFFF)


def hash32(index: np.ndarray, seed: int) -> np.ndarray:
    """Reference 32-bit hash (Wellons' lowbias32 on the index folded to 32 bits).

    CUDA implementation must match::

        uint32_t x = (uint32_t)i ^ ((uint32_t)(i >> 32) * 0x85EBCA6Bu) ^ seed;
        x ^= x >> 16; x *= 0x7FEB352Du; x ^= x >> 15; x *= 0x846CA68Bu; x ^= x >> 16;
    """
    i = np.asarray(index, dtype=np.uint64)
    lo = (i & np.uint64(0xFFFFFFFF)).astype(np.uint32)
    hi = (i >> np.uint64(32)).astype(np.uint32)
    with np.errstate(over="ignore"):
        x = lo ^ (hi * np.uint32(0x85EBCA6B)) ^ np.uint32(seed & 0xFFFFFFFF)
        x ^= x >> np.uint32(16)
        x *= np.uint32(0x7FEB352D)
        x ^= x >> np.uint32(15)
        x *= np.uint32(0x846CA68B)
        x ^= x >> np.uint32(16)
    return x.astype(np.uint32)


def background_values(bg: Background, index: np.ndarray, invert: bool = False) -> np.ndarray:
    """Values (uint32) of ``bg`` at the given flat indices, optionally inverted."""
    idx = np.asarray(index, dtype=np.uint64)
    if bg.kind == "solid":
        out = np.full(idx.shape, bg.value & 0xFFFFFFFF, dtype=np.uint32)
    elif bg.kind == "checker":
        base = np.uint32(bg.value & 0xFFFFFFFF)
        out = np.where((idx & np.uint64(1)) == 0, base, ~base).astype(np.uint32)
    elif bg.kind == "addr":
        out = (idx & np.uint64(0xFFFFFFFF)).astype(np.uint32)
    elif bg.kind == "random":
        out = hash32(idx, bg.seed)
    else:  # pragma: no cover - guarded by the Literal type
        raise ValueError(f"unknown background kind {bg.kind!r}")
    return ~out if invert else out


@dataclass(frozen=True)
class Write:
    background: Background = ZERO
    invert: bool = False

    def label(self) -> str:
        return f"w{'~' if self.invert else ''}{self.background.label()}"


@dataclass(frozen=True)
class Read:
    background: Background = ZERO
    invert: bool = False

    def label(self) -> str:
        return f"r{'~' if self.invert else ''}{self.background.label()}"


Op = Union[Write, Read]
Order = Literal["up", "down", "any"]


@dataclass(frozen=True)
class Element:
    order: Order
    ops: tuple[Op, ...]

    def label(self) -> str:
        arrow = {"up": "up", "down": "down", "any": "any"}[self.order]
        return f"{arrow}({','.join(op.label() for op in self.ops)})"


@dataclass(frozen=True)
class Pause:
    seconds: float

    def label(self) -> str:
        return f"pause({self.seconds:g}s)"


@dataclass(frozen=True)
class Hammer:
    """Activate aggressor rows ``activations`` times in every selected bank.

    ``rows`` are aggressor row numbers; neighbours (row +/- 1) are the victims.
    ``banks`` limits the banks (bank_key values, see mapping.bank_key); ``None``
    means every bank. Victim data is checked by a following read element.
    """

    rows: tuple[int, ...]
    activations: int
    banks: tuple[int, ...] | None = None

    def label(self) -> str:
        return f"hammer(rows={list(self.rows)}, x{self.activations})"


Step = Union[Element, Pause, Hammer]


@dataclass(frozen=True)
class Pattern:
    name: str
    steps: tuple[Step, ...]
    iterations: int = 1
    description: str = ""
    tags: tuple[str, ...] = field(default_factory=tuple)

    def label(self) -> str:
        return "{" + "; ".join(step.label() for step in self.steps) + "}"

    def reads_per_word(self) -> int:
        return sum(sum(isinstance(op, Read) for op in s.ops) for s in self.steps if isinstance(s, Element))

    def ops_per_word(self) -> int:
        return sum(len(s.ops) for s in self.steps if isinstance(s, Element))


def concat(name: str, patterns: list[Pattern]) -> Pattern:
    """Patterns run back to back in the given order (iterations unrolled), as a test tool runs a suite."""
    steps: list = []
    for p in patterns:
        steps += list(p.steps) * p.iterations
    return Pattern(name, tuple(steps), description=" + ".join(p.name for p in patterns))

"""Mapping between flat word indices and physical HBM coordinates.

A test program only sees a flat array of 32-bit words. Where each word lives
inside the device (stack, channel, bank, row, column ...) is decided by the
memory controller. Real GPU address maps are not public, so hbmlens treats the
map as a *pluggable, explicit assumption*: every analysis result states which
mapper produced it.

All functions are vectorized over NumPy arrays of word indices.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol, Sequence

import numpy as np

from .geometry import LEVELS, HBMGeometry, get_geometry

Fields = dict[str, np.ndarray]


class AddressMapper(Protocol):
    geometry: HBMGeometry

    def decode(self, index: np.ndarray) -> Fields:
        """Flat word index -> ``{level: coordinate}`` for every name in ``LEVELS``."""

    def encode(self, fields: Fields) -> np.ndarray:
        """Inverse of :meth:`decode`."""

    def describe(self) -> str:
        """Human readable description, stored next to every analysis result."""


@dataclass(frozen=True)
class XorRule:
    """``target ^= source & (2**bits - 1)`` applied after the mixed-radix split.

    Models the bank/channel hashing that memory controllers use to spread
    consecutive rows across banks. ``target`` size must be a power of two and
    ``2**bits`` must not exceed it.
    """

    target: str
    source: str
    bits: int


@dataclass
class LinearMapper:
    """Mixed-radix mapper with an explicit level order plus optional XOR hashing.

    ``order`` lists the levels from most to least significant. The default puts
    the whole bank contiguously (easy to reason about); ``interleaved`` spreads
    consecutive 32 B bursts across channels like a GPU memory controller would.
    """

    geometry: HBMGeometry
    order: Sequence[str] = LEVELS
    xor_rules: Sequence[XorRule] = field(default_factory=tuple)
    label: str = "linear"

    def __post_init__(self) -> None:
        self.geometry = get_geometry(self.geometry)
        if sorted(self.order) != sorted(LEVELS):
            raise ValueError(f"order must be a permutation of {LEVELS}, got {self.order}")
        self._sizes = self.geometry.level_sizes()
        # stride of each level in the flat index (least significant level has stride 1)
        stride = 1
        self._stride: dict[str, int] = {}
        for level in reversed(tuple(self.order)):
            self._stride[level] = stride
            stride *= self._sizes[level]
        for rule in self.xor_rules:
            size = self._sizes[rule.target]
            if size & (size - 1):
                raise ValueError(f"XOR target {rule.target!r} size {size} is not a power of two")
            if (1 << rule.bits) > size:
                raise ValueError(f"XOR rule {rule} uses more bits than {rule.target} has")

    def decode(self, index: np.ndarray) -> Fields:
        idx = np.asarray(index, dtype=np.int64)
        if idx.size and (idx.min() < 0 or idx.max() >= self.geometry.total_words):
            raise IndexError("word index outside the device")
        out: Fields = {}
        for level in LEVELS:
            out[level] = (idx // self._stride[level]) % self._sizes[level]
        for rule in self.xor_rules:
            out[rule.target] = out[rule.target] ^ (out[rule.source] & ((1 << rule.bits) - 1))
        return out

    def encode(self, fields: Fields) -> np.ndarray:
        f = {k: np.asarray(v, dtype=np.int64) for k, v in fields.items()}
        for rule in reversed(tuple(self.xor_rules)):
            f[rule.target] = f[rule.target] ^ (f[rule.source] & ((1 << rule.bits) - 1))
        index = np.zeros(np.broadcast(*f.values()).shape, dtype=np.int64)
        for level in LEVELS:
            index += f[level] * self._stride[level]
        return index

    def describe(self) -> str:
        rules = "".join(f" xor({r.target}^={r.source}[{r.bits}b])" for r in self.xor_rules)
        return f"{self.label}: {'>'.join(self.order)}{rules} on {self.geometry.name}"


def linear(geometry: "str | HBMGeometry") -> LinearMapper:
    """Each bank is one contiguous block; rows then columns inside it."""
    return LinearMapper(get_geometry(geometry), LEVELS, (), "linear")


def interleaved(geometry: "str | HBMGeometry") -> LinearMapper:
    """GPU-like interleave: consecutive bursts rotate over channels and pseudo channels,
    then stacks; banks are hashed with low row bits. Illustrative, not a real GPU map."""
    g = get_geometry(geometry)
    order = ("row", "bank", "bank_group", "sid", "column", "stack", "pseudo_channel", "channel", "word")
    rules = []
    sizes = g.level_sizes()
    for target in ("bank", "bank_group"):
        size = sizes[target]
        if size > 1 and not size & (size - 1):
            rules.append(XorRule(target, "row", size.bit_length() - 1))
    return LinearMapper(g, order, tuple(rules), "interleaved")


MAPPERS = {"linear": linear, "interleaved": interleaved}


def get_mapper(name: str, geometry: "str | HBMGeometry") -> LinearMapper:
    try:
        return MAPPERS[name](geometry)
    except KeyError:
        raise KeyError(f"unknown mapper {name!r}; choose from {sorted(MAPPERS)}") from None


def bank_key(fields: Fields, geometry: HBMGeometry) -> np.ndarray:
    """Single integer that identifies a bank across the whole device."""
    g = geometry
    key = fields["stack"]
    key = key * g.channels + fields["channel"]
    key = key * g.pseudo_channels + fields["pseudo_channel"]
    key = key * g.sids + fields["sid"]
    key = key * g.bank_groups + fields["bank_group"]
    key = key * g.banks_per_group + fields["bank"]
    return key

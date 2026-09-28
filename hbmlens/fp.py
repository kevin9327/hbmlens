"""Fault primitives: the formal notation of memory faults.

A fault primitive (FP) ``<S/F/R>`` names the sensitizing operation sequence S,
the value F of the faulty (victim) cell afterwards, and the output R of the last
operation of S if it is a read (``-`` otherwise). Two-cell FPs are written
``<Sa; Sv/F/R>`` with the aggressor part first. A functional fault model (FFM),
such as the transition fault TF, is a set of FPs.

This module holds the FP catalogues exactly as published, a parser for the
notation, and the DRAM-specific fault attributes:

* static simple faults (at most one operation), single-cell (12 FPs) and
  two-cell (36 FPs): Al-Ars, van de Goor, Hamdioui, "Space of DRAM Fault Models
  and Corresponding Testing", DATE 2006, Tables 1 and 2 (after van de Goor and
  Al-Ars, "Functional Fault Models: A Formal Notation and Taxonomy", VTS 2000);
* dynamic faults sensitized by a write immediately followed by a read, single-cell
  (12 FPs) and two-cell (32 FPs): Hamdioui, Al-Ars, van de Goor, "Testing Static
  and Dynamic Faults in Random Access Memories", VTS 2002, Tables 1 and 2;
* DRAM-specific attributes (DATE 2006, Section 3): partial (an operation must be
  repeated ``h`` times: ``pi`` on the initialization, ``pa`` on the activation),
  dirty (detection needs an operation with opposite data on another cell of the
  same bit line first), and the timing attributes hard, soft (the effect appears
  only after a delay) and transient (the effect disappears again).

Faults are pure data; :mod:`hbmlens.fpsim` gives them behaviour.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, replace
from typing import Iterable

ARROW = {"↑": 1, "↓": 0, "^": 1, "v": 0}


@dataclass(frozen=True)
class Op:
    """One memory operation inside an FP: ``kind`` 'w' or 'r' with data ``value``."""

    kind: str
    value: int

    def __str__(self) -> str:
        return f"{self.kind}{self.value}"


@dataclass(frozen=True)
class FaultPrimitive:
    ffm: str  # functional fault model, e.g. "TF", "CFds", "dRDF"
    notation: str  # as published, e.g. "<0w1/0/->"
    cells: int  # 1 or 2
    v_state: int  # victim state before S
    v_ops: tuple[Op, ...]  # operations of S applied to the victim
    F: int  # victim value after S
    R: int | None  # output of the last victim op if it is a read
    a_state: int | None = None  # aggressor state (two-cell FPs)
    a_ops: tuple[Op, ...] = ()  # operations of S applied to the aggressor
    # DRAM-specific attributes (DATE 2006): "", "pi", "pa" or "pia"; dirty; "h", "s" or "t"
    partial: str = ""
    dirty: bool = False
    timing: str = "h"

    @property
    def dynamic(self) -> bool:
        return len(self.v_ops) + len(self.a_ops) > 1

    @property
    def name(self) -> str:
        attrs = self.partial + ("d" if self.dirty else "") + (self.timing if self.timing != "h" else "")
        return (f"{attrs} " if attrs else "") + f"{self.ffm}{self.notation}"

    def with_attributes(self, partial: str = "", dirty: bool = False, timing: str = "h") -> "FaultPrimitive":
        check_realistic(self, partial, dirty, timing)
        return replace(self, partial=partial, dirty=dirty, timing=timing)


_OPS = re.compile(r"([wr])([01])")


def _ops(text: str) -> tuple[int | None, tuple[Op, ...]]:
    """'0w1r1' -> (0, (w1, r1)); 'w1' -> (None, (w1,)); '1' -> (1, ())."""
    text = text.strip()
    m = re.fullmatch(r"([01]?)((?:[wr][01])*)", text)
    if not m:
        raise ValueError(f"cannot parse operation sequence {text!r}")
    state = int(m.group(1)) if m.group(1) else None
    ops = tuple(Op(k, int(v)) for k, v in _OPS.findall(m.group(2)))
    return state, ops


def parse(notation: str, ffm: str) -> FaultPrimitive:
    """Parse ``<S/F/R>`` or ``<Sa; Sv/F/R>``; F may be 0, 1 or an arrow (↑ = 1, ↓ = 0)."""
    body = notation.strip()
    if not (body.startswith("<") and body.endswith(">")):
        raise ValueError(f"FP must be enclosed in <>: {notation!r}")
    s, f, r = (part.strip() for part in body[1:-1].rsplit("/", 2))
    F = ARROW[f] if f in ARROW else int(f)
    R = None if r in ("-", "−") else int(r)
    if ";" in s:
        sa, sv = (x.strip() for x in s.split(";"))
        a_state, a_ops = _ops(sa)
        v_state, v_ops = _ops(sv)
        if a_state is None or v_state is None:
            raise ValueError(f"two-cell FP needs initial states for both cells: {notation!r}")
        return FaultPrimitive(ffm, notation, 2, v_state, v_ops, F, R, a_state, a_ops)
    v_state, v_ops = _ops(s)
    if v_state is None:
        raise ValueError(f"FP needs the victim's initial state: {notation!r}")
    return FaultPrimitive(ffm, notation, 1, v_state, v_ops, F, R)


def _catalogue(rows: Iterable[tuple[str, str]]) -> list[FaultPrimitive]:
    return [parse(n, ffm) for ffm, notations in rows for n in notations.split(", ")]


# DATE 2006, Table 1
STATIC_SINGLE = _catalogue([
    ("SF", "<0/1/->, <1/0/->"),
    ("TF", "<0w1/0/->, <1w0/1/->"),
    ("WDF", "<0w0/1/->, <1w1/0/->"),
    ("RDF", "<0r0/1/1>, <1r1/0/0>"),
    ("IRF", "<0r0/0/1>, <1r1/1/0>"),
    ("DRDF", "<0r0/1/0>, <1r1/0/1>"),
])

# DATE 2006, Table 2 (x, y in {0, 1} expanded)
STATIC_TWO = _catalogue([
    ("CFst", "<0; 0/1/->, <0; 1/0/->, <1; 1/0/->, <1; 0/1/->"),
    ("CFds", ", ".join(f"<{x}w{y}; {v}/{1 - v}/->" for x in (0, 1) for y in (0, 1) for v in (0, 1))
     + ", " + ", ".join(f"<{x}r{x}; {v}/{1 - v}/->" for x in (0, 1) for v in (0, 1))),
    ("CFtr", "<0; 0w1/0/->, <0; 1w0/1/->, <1; 0w1/0/->, <1; 1w0/1/->"),
    ("CFwd", "<0; 0w0/1/->, <0; 1w1/0/->, <1; 0w0/1/->, <1; 1w1/0/->"),
    ("CFrd", "<0; 0r0/1/1>, <0; 1r1/0/0>, <1; 0r0/1/1>, <1; 1r1/0/0>"),
    ("CFir", "<0; 0r0/0/1>, <0; 1r1/1/0>, <1; 0r0/0/1>, <1; 1r1/1/0>"),
    ("CFdrd", "<0; 0r0/1/0>, <0; 1r1/0/1>, <1; 0r0/1/0>, <1; 1r1/0/1>"),
])

# VTS 2002, Table 1
DYNAMIC_SINGLE = _catalogue([
    ("dRDF", "<0w0r0/↑/1>, <0w1r1/↓/0>, <1w0r0/↑/1>, <1w1r1/↓/0>"),
    ("dDRDF", "<0w0r0/↑/0>, <0w1r1/↓/1>, <1w0r0/↑/0>, <1w1r1/↓/1>"),
    ("dIRF", "<0w0r0/0/1>, <0w1r1/1/0>, <1w0r0/0/1>, <1w1r1/1/0>"),
])

# VTS 2002, Table 2 (x in {0, 1} expanded)
DYNAMIC_TWO = _catalogue([
    ("dCFds", "<0w0r0; 0/↑/->, <0w1r1; 0/↑/->, <1w0r0; 0/↑/->, <1w1r1; 0/↑/->, "
              "<0w0r0; 1/↓/->, <0w1r1; 1/↓/->, <1w0r0; 1/↓/->, <1w1r1; 1/↓/->"),
    ("dCFrd", ", ".join(f"<{x}; 0w0r0/↑/1>, <{x}; 0w1r1/↓/0>, <{x}; 1w0r0/↑/1>, <{x}; 1w1r1/↓/0>" for x in (0, 1))),
    ("dCFdrd", ", ".join(f"<{x}; 0w0r0/↑/0>, <{x}; 0w1r1/↓/1>, <{x}; 1w0r0/↑/0>, <{x}; 1w1r1/↓/1>" for x in (0, 1))),
    ("dCFir", ", ".join(f"<{x}; 0w0r0/0/1>, <{x}; 0w1r1/1/0>, <{x}; 1w0r0/0/1>, <{x}; 1w1r1/1/0>" for x in (0, 1))),
])


def check_realistic(fp: FaultPrimitive, partial: str, dirty: bool, timing: str) -> None:
    """The realistic space of DRAM-specific faults (DATE 2006, Section 3.3):
    single-cell faults may be pi, d or pid (state faults not partial); two-cell faults
    may only be partial, pi on the aggressor and pa only when S operates on the aggressor
    (state coupling faults not partial). Soft and transient timing are modelled for
    static faults only."""
    if partial not in ("", "pi", "pa", "pia") or timing not in ("h", "s", "t"):
        raise ValueError(f"unknown attributes {partial!r} {timing!r}")
    if fp.dynamic and (partial or dirty or timing != "h"):
        raise ValueError("DRAM attributes are defined here for static faults only")
    if fp.cells == 1:
        if "pa" in partial or partial == "pia":
            raise ValueError("single-cell faults can only be partial in the initialization (pi)")
        if partial and fp.ffm == "SF":
            raise ValueError("state faults may not be partial")
    else:
        if dirty:
            raise ValueError("coupling faults may not be dirty")
        if partial and fp.ffm == "CFst":
            raise ValueError("state coupling faults may not be partial")
        if "a" in partial.replace("pi", "") and not fp.a_ops:
            raise ValueError("activation partial faults need S to operate on the aggressor")
    if timing == "s" and any(op.kind == "r" for op in fp.v_ops + fp.a_ops):
        raise ValueError("soft faults are modelled for state and write faults only")


def realistic_single_cell(timing: str = "h") -> list[FaultPrimitive]:
    """All single-cell static FPs with every realistic DRAM attribute: {-, pi, d, pid} (DATE 2006, Expr. 3)."""
    out = []
    for fp in STATIC_SINGLE:
        for partial in ("", "pi"):
            for dirty in (False, True):
                try:
                    out.append(fp.with_attributes(partial, dirty, timing))
                except ValueError:
                    pass
    return out


def realistic_two_cell(timing: str = "h") -> list[FaultPrimitive]:
    """All two-cell static FPs with every realistic partial attribute: {-, p} (DATE 2006, Expr. 4)."""
    out = []
    for fp in STATIC_TWO:
        for partial in ("", "pi", "pa", "pia"):
            try:
                out.append(fp.with_attributes(partial, False, timing))
            except ValueError:
                pass
    return out

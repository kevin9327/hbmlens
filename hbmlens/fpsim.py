"""Exact simulation of fault primitives under march tests, and guaranteed-detection proofs.

A small bit-oriented memory array (default 4 rows x 2 columns, row-major
addresses, one bit line per column) runs the test operation by operation. One
fault primitive is placed in it; every other cell is fault free. A fault
counts as *detected* only if some read returns a value different from the
fault-free expectation for **every** placement of the faulty cells, every
initial value of the cells involved (and of the rest: all 0 or all 1), and
every choice of direction for the test's ``⇕`` elements. Two-cell faults are
reported separately for a victim above and below the aggressor, as in the
literature.

Behaviour of a fault primitive <S/F/R> (see :mod:`hbmlens.fp`):

* When the operations of S happen, the victim takes value F; if the last
  operation of S is a read of the victim, that read returns R. A read in S
  matches when the cell holds the value it names. "Immediately" (dynamic
  faults) means no other operation anywhere in between.
* State faults (SF, CFst) act whenever their state condition holds.
* ``pi``: the state an FP starts from only counts once the cell has been
  written with that value ``h`` times in a row (reads do not break the run);
  for a write destructive fault the run includes the destroying write. For
  two-cell faults this applies to the aggressor.
* ``pa``: the aggressor's sensitizing operation has to be repeated ``h`` times
  in a row.
* dirty: a read of the faulty cell returns the fault-free value unless, since
  the last operation on that cell, another cell on its bit line was accessed
  with the opposite data.
* soft: the fault effect appears only after a delay element (``del``) without
  the cell being rewritten; transient: it disappears again after ``L``
  operations, so detection needs back-to-back operations.
"""
from __future__ import annotations

import itertools
from dataclasses import dataclass, field

import numpy as np

from .fp import FaultPrimitive, Op
from .patterns.base import Element, Hammer, Pattern, Pause, Read, background_values

PAUSE_UNITS = 10**9  # a delay element outlasts any soft-fault delay
SOFT_DELAY = 10**6  # soft faults need a delay element (longer than any run of operations)
TRANSIENT_LIFE = 3  # operations a transient fault effect survives
FAULT_H = 4  # repetitions partial faults need; the literature tests here hammer 5 times


@dataclass(frozen=True)
class Array:
    rows: int = 4
    cols: int = 2

    @property
    def cells(self) -> int:
        return self.rows * self.cols

    def partner(self, cell: int) -> int:
        """The other cell a ``b`` operation reaches: same column (bit line), neighbouring row."""
        r, c = divmod(cell, self.cols)
        return (r ^ 1) % self.rows * self.cols + c

    def bitline_mates(self, cell: int) -> tuple[int, ...]:
        c = cell % self.cols
        return tuple(r * self.cols + c for r in range(self.rows) if r * self.cols + c != cell)


DEFAULT_ARRAY = Array()

# stream entry: (time, cell, is_write, value, element, op); cell -1 is a delay element
Entry = tuple[int, int, bool, int, int, int]


def _any_elements(pattern: Pattern) -> list[int]:
    return [i for i, s in enumerate(pattern.steps) if isinstance(s, Element) and s.order == "any"]


def order_choices(pattern: Pattern) -> list[dict[int, str]]:
    idx = _any_elements(pattern)
    return [dict(zip(idx, combo, strict=True)) for combo in itertools.product(("up", "down"), repeat=len(idx))]


def op_stream(pattern: Pattern, array: Array, orders: dict[int, str], bit: int = 0) -> list[Entry]:
    """Flatten a test into timed operations on the array's cells (bit ``bit`` of each word's data)."""
    out: list[Entry] = []
    t = 0
    values: dict = {}

    def data(op, cell: int) -> int:
        key = (op.background, op.invert, cell)
        if key not in values:
            word = int(background_values(op.background, np.array([cell], dtype=np.uint64), op.invert)[0])
            values[key] = word >> bit & 1
        return values[key]

    for _ in range(pattern.iterations):
        for si, step in enumerate(pattern.steps):
            if isinstance(step, Pause):
                t += PAUSE_UNITS
                out.append((t, -1, False, 0, si, 0))
                continue
            if isinstance(step, Hammer):
                raise ValueError("Hammer steps are not modelled by the fault primitive simulator")
            order = orders.get(si, step.order) if step.order == "any" else step.order
            addrs = range(array.cells) if order == "up" else range(array.cells - 1, -1, -1)
            for cell in addrs:
                for oi, op in enumerate(step.ops):
                    target = array.partner(cell) if op.on == "bl" else cell
                    x = data(op, target)
                    for _ in range(op.repeat):
                        t += 1
                        out.append((t, target, not isinstance(op, Read), x, si, oi))
    return out


class _Sim:
    """One fault primitive at one placement; see the module docstring for the semantics."""

    def __init__(self, fp: FaultPrimitive, v: int, a: int | None, mates: tuple[int, ...],
                 init: dict[int, int], h: int, immediate: str = "element"):
        self.fp, self.v, self.a, self.mates, self.h = fp, v, a, frozenset(mates), h
        self.same_element = immediate == "element"
        self.val = dict(init)
        self.gold: dict[int, int] = {}
        self.run: dict[int, tuple[int, int]] = {}  # cell -> (value, consecutive writes of it)
        self.arun: tuple[Op | None, int, bool] = (None, 0, False)  # aggressor op run (pa)
        self.bl_clean = False
        self.pending: tuple[int, int] | None = None  # soft fault: (due time, value)
        self.revert: tuple[int, int] | None = None  # transient fault: (time, value to restore)
        self.prev: tuple | None = None  # previous op: (time, cell, op, pre-state, element)
        self._state_faults(0, entered=True)

    def _written(self, cell: int, x: int, run: tuple[int, int] | None = None) -> bool:
        """Partial initialization: ``cell`` was last written ``x`` at least h times in a row."""
        value, count = run if run is not None else self.run.get(cell, (-1, 0))
        return value == x and count >= self.h

    def _aggressor_holds(self) -> bool:
        fp = self.fp
        if fp.cells == 1:
            return True
        if self.val.get(self.a, 0) != fp.a_state:
            return False
        return "pi" not in fp.partial or self._written(self.a, fp.a_state)

    def _effect(self, t: int) -> None:
        """The victim takes F: now (hard), after a delay element (soft) or for a while (transient)."""
        fp = self.fp
        if fp.timing == "s":
            self.pending = (t + SOFT_DELAY, fp.F)
            return
        if fp.timing == "t":
            self.revert = (t + TRANSIENT_LIFE, self.gold.get(self.v, self.val.get(self.v, 0)))
        self.val[self.v] = fp.F

    def _state_faults(self, t: int, entered: bool) -> None:
        fp = self.fp
        if fp.ffm not in ("SF", "CFst") or (fp.timing != "h" and not entered):
            return
        if self.val.get(self.v, 0) == fp.v_state and self._aggressor_holds():
            self._effect(t)

    def _tick(self, t: int) -> None:
        if self.pending and t >= self.pending[0]:
            self.val[self.v] = self.pending[1]
            self.pending = None
        if self.revert and t >= self.revert[0]:
            if self.val.get(self.v) == self.fp.F:
                self.val[self.v] = self.revert[1]
            self.revert = None

    def step(self, t: int, cell: int, is_write: bool, x: int, element: int = -1) -> int | None:
        """Apply one operation; return what a read of the victim outputs (else None)."""
        self._tick(t)
        if cell == -1:  # delay element
            return None
        fp, v = self.fp, self.v
        pre = self.val.get(cell, 0)
        op = Op("w", x) if is_write else Op("r", pre)
        run_before = self.run.get(cell, (-1, 0))
        if is_write:
            self.run[cell] = (x, run_before[1] + 1) if run_before[0] == x else (x, 1)
        if fp.dirty and cell in self.mates and v in self.gold and (x if is_write else pre) != self.gold[v]:
            self.bl_clean = True  # opposite data on the victim's bit line

        self._element = element
        out = None
        if cell == v:
            out = self._victim_op(t, op, pre, is_write, x, run_before)
        else:
            if is_write:
                self.val[cell] = x
                self.gold[cell] = x
            if cell == self.a and fp.a_ops:
                self._aggressor_op(t, op, pre, run_before)
        self._state_faults(t, entered=is_write and cell in (v, self.a))
        self.prev = (t, cell, op, pre, element)
        return out

    def _right_after(self, cell: int, t: int, op: Op, state: int) -> bool:
        """The previous operation was ``op`` on ``cell`` holding ``state``, immediately before
        (in the same march element, or also across elements when immediate="time")."""
        p = self.prev
        return (p is not None and p[0] == t - 1 and p[1] == cell and p[2] == op and p[3] == state
                and (not self.same_element or p[4] == self._element))

    def _victim_op(self, t: int, op: Op, pre: int, is_write: bool, x: int, run_before) -> int | None:
        fp, v = self.fp, self.v
        sens = False
        if len(fp.v_ops) == 1 and op == fp.v_ops[0]:
            if fp.cells == 1 and "pi" in fp.partial:
                # DATE 2006 writes these <w_x^h ...>: the initial state is replaced by h writes of x,
                # which include the sensitizing write itself when that write is a w_x
                same_value_write = is_write and x == fp.v_state
                sens = self._written(v, fp.v_state, None if same_value_write else run_before)
                sens = sens and (same_value_write or pre == fp.v_state)
            else:
                sens = pre == fp.v_state
            sens = sens and self._aggressor_holds()
        elif len(fp.v_ops) == 2 and op == fp.v_ops[1]:
            sens = self._right_after(v, t, fp.v_ops[0], fp.v_state) and self._aggressor_holds()
        out = None
        if is_write:
            self.val[v] = x
            self.gold[v] = x
            self.pending = None
            self.revert = None
        else:
            out = pre
        if sens:
            self._effect(t)
            if not is_write and fp.R is not None:
                out = fp.R
        if fp.dirty:
            if not is_write and not self.bl_clean and v in self.gold:
                out = self.gold[v]
            self.bl_clean = False
        return out

    def _aggressor_op(self, t: int, op: Op, pre: int, run_before) -> None:
        fp = self.fp
        if "pi" in fp.partial:  # as for single-cell faults, a sensitizing w_x counts in the run
            same_value_write = op == Op("w", fp.a_state)
            initialised = self._written(self.a, fp.a_state, None if same_value_write else run_before)
            initialised = initialised and (same_value_write or pre == fp.a_state)
        else:
            initialised = pre == fp.a_state
        # pa: count the same operation repeated in a row, from the moment the aggressor holds a_state
        last, count, started_initialised = self.arun
        if last == op and count:
            count += 1
        elif pre == fp.a_state:
            count, started_initialised = 1, initialised
        else:
            count = 0
        self.arun = (op, count, started_initialised)
        if len(fp.a_ops) == 1:
            if op != fp.a_ops[0]:
                return
            if fp.partial in ("pa", "pia"):
                if "pi" not in fp.partial:
                    init_ok = True
                elif op == Op("w", fp.a_state):  # rewriting x both initializes and activates
                    init_ok = self._written(self.a, fp.a_state)
                else:
                    init_ok = started_initialised
                hit = count == self.h and init_ok
            else:
                hit = initialised
        else:  # dynamic: a write immediately followed by a read on the aggressor
            hit = op == fp.a_ops[1] and self._right_after(self.a, t, fp.a_ops[0], fp.a_state)
        if hit and self.val.get(self.v, 0) == fp.v_state:
            self._effect(t)


@dataclass
class CaseResult:
    fp: FaultPrimitive
    position: str  # "" for single-cell faults, "v>a" or "v<a"
    detected: bool
    first: set = field(default_factory=set)  # (element, op) of the first detecting read, per run


def _placements(fp: FaultPrimitive, array: Array):
    n = array.cells
    if fp.cells == 1:
        yield "", [(v, None) for v in range(n)]
    else:
        pairs = [(v, a) for v in range(n) for a in range(n) if v != a]
        yield "v>a", [(v, a) for v, a in pairs if v > a]
        yield "v<a", [(v, a) for v, a in pairs if v < a]


def run_case(fp: FaultPrimitive, streams: list[list[Entry]], placements, array: Array, h: int,
             immediate: str = "element") -> tuple[bool, set]:
    firsts: set = set()
    for v, a in placements:
        involved = [v] if a is None else [v, a]
        mates = array.bitline_mates(v) if fp.dirty else ()
        relevant = set(involved) | set(mates) | {-1}
        others = [c for c in range(array.cells) if c not in involved]
        for stream in streams:
            sub = [e for e in stream if e[1] in relevant]
            for bits in itertools.product((0, 1), repeat=len(involved)):
                for rest in (0, 1):
                    init = {c: rest for c in others}
                    init.update(zip(involved, bits, strict=True))
                    sim = _Sim(fp, v, a, mates, init, h, immediate)
                    hit = None
                    for t, cell, is_write, x, el, oi in sub:
                        out = sim.step(t, cell, is_write, x, el)
                        # a tester compares with the data the read expects; reads of a cell
                        # before it was written are not comparisons a test can rely on
                        if out is not None and out != x and v in sim.gold:
                            hit = (el, oi)
                            break
                    if hit is None:
                        return False, firsts
                    firsts.add(hit)
    return True, firsts


def evaluate(pattern: Pattern, fps: list[FaultPrimitive], array: Array = DEFAULT_ARRAY, h: int = FAULT_H,
             bit: int = 0, immediate: str = "element") -> list[CaseResult]:
    """Guaranteed detection of every FP (and position) in ``fps`` by ``pattern``; ``h`` is the
    number of repetitions a partial fault needs (the test's own hammer count is in the pattern)."""
    streams = [op_stream(pattern, array, o, bit) for o in order_choices(pattern)]
    results = []
    for fp in fps:
        for position, placements in _placements(fp, array):
            ok, first = run_case(fp, streams, placements, array, h, immediate)
            results.append(CaseResult(fp, position, ok, first))
    return results


def coverage_by_ffm(results: list[CaseResult]) -> dict[str, tuple[int, int]]:
    """FFM -> (detected cases, cases); a two-cell FP counts once per position."""
    out: dict[str, list[int]] = {}
    for r in results:
        key = r.fp.name.split("<")[0] if r.fp.partial or r.fp.dirty or r.fp.timing != "h" else r.fp.ffm
        d = out.setdefault(key, [0, 0])
        d[0] += r.detected
        d[1] += 1
    return {k: (v[0], v[1]) for k, v in out.items()}

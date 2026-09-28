"""Test-quality measurement: which pattern detects which fault model.

Classic memory fault models (van de Goor's and Hamdioui's static fault taxonomy)
are simulated exactly at the bit level. Only the words a fault touches can ever
misbehave, so a pattern is executed on just those words, in the element's address
order. That keeps the exact march semantics (order matters for coupling faults)
and makes thousands of trials cheap.

A fault counts as detected only when detection is *guaranteed*:

* elements with order ``any`` may be run in either direction by a real backend,
  so the fault must be caught in both directions;
* memory content before the test is unknown, so the fault must be caught for
  every initial value of the bits it involves. (Assuming all-zero memory would
  credit a test for write destructive faults it only hits by luck.)

Fault models:

* ``SAF``     stuck-at 0/1
* ``TF``      transition fault: a cell cannot make one transition (0->1 or 1->0)
* ``WDF``     write destructive: writing the value a cell already holds flips it
* ``RDF``     read destructive: a read flips the cell and returns the flipped value
* ``DRDF``    deceptive read destructive: a read returns the right value but flips the cell
* ``IRF``     incorrect read: a read returns the wrong value; the cell keeps its value
* ``CFin``    inversion coupling: a transition in the aggressor cell inverts the victim
* ``CFid``    idempotent coupling: a transition in the aggressor forces the victim to 0/1
* ``CFst``    state coupling: while the aggressor holds a value, the victim is forced
* ``CFds``    disturb coupling: reading the aggressor, or rewriting the value it holds,
  flips the victim
* ``CFwd``    WDF in the victim, only while the aggressor holds a given value
* ``CFdrd``   DRDF in the victim, only while the aggressor holds a given value
* ``*-intra`` the same coupling with aggressor and victim in one 32-bit word
* ``AF``      address decoder fault: two addresses reach the same word
* ``DRF``     data retention: a cell leaks to its discharged value during a pause
"""
from __future__ import annotations

import itertools
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Callable

import numpy as np

from .patterns.base import Background, Element, Hammer, Pattern, Pause, Read, background_values

WORDS = 1024  # address space of a coverage trial
RETENTION_S = 30.0  # DRF leak time; patterns need a pause at least this long
FULL = 0xFFFFFFFF


@dataclass
class FaultSpec:
    model: str
    words: tuple[int, ...]  # words the fault touches (aggressor first for coupling)
    params: dict = field(default_factory=dict)


FAULT_INFO = {
    "SAF": "stuck-at: a cell always reads 0 (or 1)",
    "TF": "transition: a cell cannot go 0->1 (or 1->0)",
    "WDF": "write destructive: writing the value a cell already holds flips it",
    "RDF": "read destructive: a read flips the cell and returns the flipped value",
    "DRDF": "deceptive read destructive: a read returns the right value but flips the cell",
    "IRF": "incorrect read: a read returns the wrong value, the cell keeps its value",
    "CFin": "inversion coupling: a transition in another word inverts a victim bit",
    "CFid": "idempotent coupling: a transition in another word forces a victim bit to 0/1",
    "CFst": "state coupling: while a bit of another word holds a value, a victim bit is forced",
    "CFds": "disturb coupling: reading another word, or rewriting its value, flips a victim bit",
    "CFwd": "write destructive coupling: WDF on the victim while an aggressor bit holds a value",
    "CFdrd": "deceptive read destructive coupling: DRDF on the victim while an aggressor bit holds a value",
    "CFin-intra": "inversion coupling between two bits of the same word",
    "CFid-intra": "idempotent coupling between two bits of the same word",
    "CFst-intra": "state coupling between two bits of the same word",
    "AF": "address decoder: two addresses reach the same word",
    "DRF": f"data retention: a cell leaks to its discharged value during a pause (>= {RETENTION_S:g} s here)",
}

_SINGLE_CELL = {"SAF", "TF", "WDF", "RDF", "DRDF", "IRF", "DRF"}


class SparseSim:
    """Bit-exact memory holding only the words a fault touches (others behave ideally)."""

    def __init__(self, spec: FaultSpec, init: dict[int, int] | None = None):
        self.f = spec
        self.p = spec.params
        self.model = spec.model
        self.mem: dict[int, int] = dict(init or {})
        self.alias = {spec.words[1]: spec.words[0]} if spec.model == "AF" else {}
        self._apply_state()

    def _bit(self, word: int, bit: int) -> int:
        return self.mem.get(word, 0) >> bit & 1

    def _aggressor_holds(self) -> bool:
        return self._bit(self.f.words[0], self.p["abit"]) == self.p["astate"]

    def write(self, addr: int, value: int) -> None:
        p, model = self.p, self.model
        phys = self.alias.get(addr, addr)
        old = self.mem.get(phys, 0)
        new = value & FULL
        agg = self.f.words[0]
        if phys == agg and model in ("TF", "WDF"):
            b = p["bit"]
            was, now = old >> b & 1, new >> b & 1
            if model == "TF":
                if p["rising"] and not was and now:
                    new &= ~(1 << b)  # cannot go 0 -> 1
                if not p["rising"] and was and not now:
                    new |= 1 << b  # cannot go 1 -> 0
            elif was == now == p["state"]:
                new ^= 1 << b
        if model == "CFwd" and phys == self.f.words[1] and self._aggressor_holds():
            b = p["vbit"]
            if (old >> b & 1) == (new >> b & 1) == p["state"]:
                new ^= 1 << b
        self.mem[phys] = new
        if phys == agg and (model.startswith("CFin") or model.startswith("CFid")):
            a = p["abit"]
            went_up = not old >> a & 1 and new >> a & 1
            went_down = old >> a & 1 and not new >> a & 1
            if went_up if p["rising"] else went_down:
                vw, vb = self.f.words[1], 1 << p["vbit"]
                cur = self.mem.get(vw, 0)
                if model.startswith("CFin"):
                    self.mem[vw] = cur ^ vb
                else:
                    self.mem[vw] = (cur | vb) if p["value"] else (cur & ~vb)
        if model == "CFds" and phys == agg and p["op"] == "w":
            a = p["abit"]
            if (old >> a & 1) == (new >> a & 1) == p["astate"]:  # non-transition write
                self._disturb_victim()
        self._apply_state()

    def _disturb_victim(self) -> None:
        vw, vb = self.f.words[1], self.p["vbit"]
        if self._bit(vw, vb) == self.p["vstate"]:
            self.mem[vw] = self.mem.get(vw, 0) ^ (1 << vb)

    def _apply_state(self) -> None:
        if self.model.startswith("CFst") and self._aggressor_holds():
            vw, vb = self.f.words[1], 1 << self.p["vbit"]
            cur = self.mem.get(vw, 0)
            self.mem[vw] = (cur | vb) if self.p["value"] else (cur & ~vb)

    def read(self, addr: int) -> int:
        p, model = self.p, self.model
        phys = self.alias.get(addr, addr)
        v = self.mem.get(phys, 0)
        if phys == self.f.words[0]:
            if model == "SAF":
                b = 1 << p["bit"]
                v = (v | b) if p["value"] else (v & ~b)
            elif model in ("RDF", "DRDF", "IRF") and (v >> p["bit"] & 1) == p["state"]:
                flipped = v ^ (1 << p["bit"])
                if model != "IRF":
                    self.mem[phys] = flipped
                if model != "DRDF":
                    v = flipped
            elif model == "CFds" and p["op"] == "r" and (v >> p["abit"] & 1) == p["astate"]:
                self._disturb_victim()
        if model == "CFdrd" and phys == self.f.words[1] and self._aggressor_holds():
            if (v >> p["vbit"] & 1) == p["state"]:
                self.mem[phys] = v ^ (1 << p["vbit"])  # returns the old, correct value
        return v

    def pause(self, seconds: float) -> None:
        if self.model == "DRF" and seconds >= self.p["retention_s"]:
            w, b = self.f.words[0], 1 << self.p["bit"]
            cur = self.mem.get(w, 0)
            self.mem[w] = (cur | b) if self.p["leak"] else (cur & ~b)


@lru_cache(maxsize=None)
def _value(bg: Background, invert: bool, addr: int) -> int:
    return int(background_values(bg, np.array([addr], dtype=np.uint64), invert)[0])


@dataclass(frozen=True)
class _Program:
    """A pattern flattened for the simulator: ('pause', s) or ('el', order, ((is_read, bg, inv), ...))."""
    steps: tuple
    iterations: int
    has_any: bool

    @classmethod
    def of(cls, pattern: Pattern) -> "_Program":
        steps = []
        for s in pattern.steps:
            if isinstance(s, Pause):
                steps.append(("pause", s.seconds))
            elif isinstance(s, Element):
                steps.append(("el", s.order, tuple((isinstance(op, Read), op.background, op.invert)
                                                  for op in s.ops)))
            elif not isinstance(s, Hammer):  # pragma: no cover
                raise TypeError(s)
        return cls(tuple(steps), pattern.iterations, any(s[0] == "el" and s[1] == "any" for s in steps))


def _run(prog: _Program, spec: FaultSpec, init: dict[int, int], any_order: str) -> bool:
    sim = SparseSim(spec, init)
    addrs = sorted(set(spec.words))
    rev = addrs[::-1]
    for _ in range(prog.iterations):
        for step in prog.steps:
            if step[0] == "pause":
                sim.pause(step[1])
                continue
            order = any_order if step[1] == "any" else step[1]
            for a in (addrs if order == "up" else rev):
                for is_read, bg, inv in step[2]:
                    val = _value(bg, inv, a)
                    if is_read:
                        if sim.read(a) != val:
                            return True
                    else:
                        sim.write(a, val)
    return False


def initial_states(spec: FaultSpec) -> list[dict[int, int]]:
    """Every combination of initial values of the bits the fault involves (others 0)."""
    p = spec.params
    if spec.model == "AF":
        cells = [(spec.words[0], None)]
    elif spec.model in _SINGLE_CELL:
        cells = [(spec.words[0], p["bit"])]
    else:
        cells = [(spec.words[0], p["abit"]), (spec.words[1], p["vbit"])]
    states = []
    for combo in itertools.product((0, 1), repeat=len(cells)):
        init: dict[int, int] = {}
        for (w, b), on in zip(cells, combo, strict=True):
            if on:
                init[w] = init.get(w, 0) | (FULL if b is None else 1 << b)
        states.append(init)
    return states


def _detected(prog: _Program, spec: FaultSpec) -> bool:
    orders = ("up", "down") if prog.has_any and len(set(spec.words)) > 1 else ("up",)
    return all(_run(prog, spec, init, o) for init in initial_states(spec) for o in orders)


def detected(pattern: Pattern, spec: FaultSpec) -> bool:
    """Guaranteed detection: caught for every initial state and every allowed address order."""
    return _detected(_Program.of(pattern), spec)


def _two_words(rng) -> tuple[int, int]:
    a, b = rng.choice(WORDS, size=2, replace=False)
    return int(a), int(b)


def _coupling(model: str, intra: bool) -> Callable:
    def gen(rng) -> FaultSpec:
        if intra:
            w = int(rng.integers(WORDS))
            abit, vbit = (int(x) for x in rng.choice(32, size=2, replace=False))
            words = (w, w)
        else:
            words = _two_words(rng)
            abit, vbit = int(rng.integers(32)), int(rng.integers(32))
        params = {"abit": abit, "vbit": vbit, "rising": bool(rng.integers(2)),
                  "value": int(rng.integers(2)), "astate": int(rng.integers(2)),
                  "op": "rw"[int(rng.integers(2))], "state": int(rng.integers(2)),
                  "vstate": int(rng.integers(2))}
        return FaultSpec(model + ("-intra" if intra else ""), words, params)
    return gen


def _cell(model: str, **extra: Callable) -> Callable:
    def gen(rng) -> FaultSpec:
        params = {"bit": int(rng.integers(32))}
        params.update({k: f(rng) for k, f in extra.items()})
        return FaultSpec(model, (int(rng.integers(WORDS)),), params)
    return gen


def _coin(rng) -> int:
    return int(rng.integers(2))


FAULT_MODELS: dict[str, Callable] = {
    "SAF": _cell("SAF", value=_coin),
    "TF": _cell("TF", rising=lambda rng: bool(rng.integers(2))),
    "WDF": _cell("WDF", state=_coin),
    "RDF": _cell("RDF", state=_coin),
    "DRDF": _cell("DRDF", state=_coin),
    "IRF": _cell("IRF", state=_coin),
    "CFin": _coupling("CFin", False),
    "CFid": _coupling("CFid", False),
    "CFst": _coupling("CFst", False),
    "CFds": _coupling("CFds", False),
    "CFwd": _coupling("CFwd", False),
    "CFdrd": _coupling("CFdrd", False),
    "CFin-intra": _coupling("CFin", True),
    "CFid-intra": _coupling("CFid", True),
    "CFst-intra": _coupling("CFst", True),
    "AF": lambda rng: FaultSpec("AF", _two_words(rng)),
    "DRF": _cell("DRF", leak=_coin, retention_s=lambda rng: RETENTION_S),
}


@dataclass
class CoverageResult:
    models: list[str]
    patterns: list[str]
    rate: np.ndarray  # [pattern, model] detection rate
    hits: np.ndarray  # [pattern, model, trial] bool
    ops_per_word: dict[str, int]
    pause_s: dict[str, float]

    def suite_rate(self, names: list[str]) -> np.ndarray:
        idx = [self.patterns.index(n) for n in names]
        return self.hits[idx].any(axis=0).mean(axis=1)

    def suite_cost(self, names: list[str]) -> tuple[int, float]:
        return sum(self.ops_per_word[n] for n in names), sum(self.pause_s[n] for n in names)


def _pause_s(p: Pattern) -> float:
    return p.iterations * sum(s.seconds for s in p.steps if isinstance(s, Pause))


def measure(patterns: dict[str, Pattern], trials: int = 200, seed: int = 0,
            models: list[str] | None = None) -> CoverageResult:
    models = models or list(FAULT_MODELS)
    rng = np.random.default_rng(seed)
    specs = {m: [FAULT_MODELS[m](rng) for _ in range(trials)] for m in models}
    names = list(patterns)
    hits = np.zeros((len(names), len(models), trials), dtype=bool)
    for i, name in enumerate(names):
        prog = _Program.of(patterns[name])
        for j, m in enumerate(models):
            for t, spec in enumerate(specs[m]):
                hits[i, j, t] = _detected(prog, spec)
    ops = {n: p.ops_per_word() * p.iterations for n, p in patterns.items()}
    pause = {n: _pause_s(p) for n, p in patterns.items()}
    return CoverageResult(models, names, hits.mean(axis=2), hits, ops, pause)


def cheapest_suite(res: CoverageResult, candidates: list[str] | None = None,
                   max_size: int = 5) -> list[str] | None:
    """Smallest-cost set of patterns that detects every simulated fault instance.

    Cost is memory operations per word, then total pause time. Exhaustive over
    subsets of up to ``max_size`` candidates, so the answer is exact within them.
    """
    names = candidates or res.patterns
    cover = {}
    for n in names:
        bits = np.packbits(res.hits[res.patterns.index(n)].ravel(), bitorder="little")
        cover[n] = int.from_bytes(bits.tobytes(), "little")
    full = int.from_bytes(np.packbits(np.ones(res.hits[0].size, dtype=bool), bitorder="little").tobytes(),
                          "little")
    best, best_cost = None, None
    for k in range(1, max_size + 1):
        for combo in itertools.combinations(names, k):
            cost = res.suite_cost(list(combo))
            if best_cost is not None and cost >= best_cost:
                continue
            acc = 0
            for n in combo:
                acc |= cover[n]
            if acc == full:
                best, best_cost = list(combo), cost
    return best


def _pct(r: float) -> str:
    """Never round a miss up to 100% (or a hit down to 0%)."""
    if r >= 1.0:
        return "100%"
    if r <= 0.0:
        return "0%"
    return f"{min(max(r * 100, 0.1), 99.9):.1f}%"


def _duration(s: float) -> str:
    if s <= 0:
        return "-"
    if s >= 3600:
        return f"{s / 3600:g} h"
    if s >= 600:
        return f"{s / 60:.0f} min"
    return f"{s:g} s"


def suite_table(res: CoverageResult) -> str:
    """Fault models as rows, the measured patterns (typically whole suites, see
    :func:`hbmlens.patterns.base.concat`) as columns."""
    names = res.patterns
    lines = ["| | " + " | ".join(names) + " |", "|---|" + "---|" * len(names)]
    lines.append("| memory ops per word | " + " | ".join(str(res.ops_per_word[n]) for n in names) + " |")
    lines.append("| pause time | " + " | ".join(_duration(res.pause_s[n]) for n in names) + " |")
    rates = list(res.rate)
    for j, m in enumerate(res.models):
        cells = []
        for r in rates:
            cells.append(f"**{_pct(r[j])}**" if r[j] < 1.0 else _pct(r[j]))
        lines.append(f"| {m} | " + " | ".join(cells) + " |")
    lines.append("| **mean over models** | " + " | ".join(
        f"**{_pct(float(r.mean()))}**" for r in rates) + " |")
    return "\n".join(lines) + "\n"


def markdown_table(res: CoverageResult, suites: dict[str, list[str]] | None = None) -> str:
    """Patterns as rows, fault models as columns."""
    head = "| pattern | ops/word | " + " | ".join(res.models) + " |"
    lines = [head, "|" + "---|" * (len(res.models) + 2)]
    for i, name in enumerate(res.patterns):
        cells = " | ".join(_pct(r) for r in res.rate[i])
        lines.append(f"| {name} | {res.ops_per_word[name]} | {cells} |")
    for sname, members in (suites or {}).items():
        rate = res.suite_rate(members)
        ops = sum(res.ops_per_word[m] for m in members)
        cells = " | ".join(f"**{_pct(r)}**" for r in rate)
        lines.append(f"| **suite: {sname}** | {ops} | {cells} |")
    return "\n".join(lines) + "\n"

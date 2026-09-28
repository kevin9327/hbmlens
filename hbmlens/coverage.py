"""Test-quality measurement: which pattern detects which fault model.

Classic memory fault models (van de Goor's taxonomy) are simulated exactly at the
bit level. Only the words a fault touches can ever misbehave, so a pattern is
executed on just those words, in the element's address order. That keeps the
exact march semantics (order matters for coupling faults) and makes thousands of
trials cheap.

Elements with order ``any`` may be run in either direction by a real backend,
so a fault only counts as detected when it is caught in *both* directions
(guaranteed detection).

Fault models:

* ``SAF``     stuck-at 0/1
* ``TF``      transition fault: a cell cannot make one transition (0->1 or 1->0)
* ``CFin``    inversion coupling: a transition in the aggressor cell inverts the victim
* ``CFid``    idempotent coupling: a transition in the aggressor forces the victim to 0/1
* ``CFst``    state coupling: while the aggressor holds a value, the victim is forced
* ``*-intra`` the same coupling with aggressor and victim in one 32-bit word
* ``AF``      address decoder fault: two addresses reach the same word
* ``DRF``     data retention: a cell leaks to its discharged value during a pause
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import numpy as np

from .patterns.base import Element, Hammer, Pattern, Pause, Read, background_values

WORDS = 1024  # address space of a coverage trial


@dataclass
class FaultSpec:
    model: str
    words: tuple[int, ...]  # words the fault touches (aggressor first for coupling)
    params: dict = field(default_factory=dict)


class SparseSim:
    """Bit-exact memory holding only the words a fault touches (others behave ideally)."""

    def __init__(self, spec: FaultSpec):
        self.f = spec
        self.p = spec.params
        self.mem: dict[int, int] = {}
        if spec.model == "AF":  # both addresses reach one physical word
            self.alias = {spec.words[1]: spec.words[0]}
        else:
            self.alias = {}

    def _phys(self, addr: int) -> int:
        return self.alias.get(addr, addr)

    def _get(self, addr: int) -> int:
        return self.mem.get(self._phys(addr), 0)

    def write(self, addr: int, value: int) -> None:
        p, model = self.p, self.f.model
        phys = self._phys(addr)
        old = self.mem.get(phys, 0)
        new = value & 0xFFFFFFFF
        if model == "TF" and phys == self.f.words[0]:
            b = 1 << p["bit"]
            rising = p["rising"]
            if rising and not old & b and new & b:
                new &= ~b  # cannot go 0 -> 1
            if not rising and old & b and not new & b:
                new |= b  # cannot go 1 -> 0
        self.mem[phys] = new
        if (model.startswith("CFin") or model.startswith("CFid")) and phys == self.f.words[0]:
            ab = 1 << p["abit"]
            went_up = not old & ab and new & ab
            went_down = old & ab and not new & ab
            triggered = went_up if p["rising"] else went_down
            if triggered:
                vw = self.f.words[1]
                vb = 1 << p["vbit"]
                cur = self.mem.get(vw, 0)
                if model.startswith("CFin"):
                    self.mem[vw] = cur ^ vb
                else:  # CFid
                    self.mem[vw] = (cur | vb) if p["value"] else (cur & ~vb)
        self._apply_state()

    def _apply_state(self) -> None:
        if self.f.model.startswith("CFst"):
            p = self.p
            aw, vw = self.f.words[0], self.f.words[1]
            if bool(self.mem.get(aw, 0) >> p["abit"] & 1) == bool(p["astate"]):
                vb = 1 << p["vbit"]
                cur = self.mem.get(vw, 0)
                self.mem[vw] = (cur | vb) if p["value"] else (cur & ~vb)

    def read(self, addr: int) -> int:
        v = self._get(addr)
        if self.f.model == "SAF" and self._phys(addr) == self.f.words[0]:
            b = 1 << self.p["bit"]
            v = (v | b) if self.p["value"] else (v & ~b)
        return v

    def pause(self, seconds: float) -> None:
        if self.f.model == "DRF" and seconds >= self.p["retention_s"]:
            w, b = self.f.words[0], 1 << self.p["bit"]
            cur = self.mem.get(w, 0)
            self.mem[w] = (cur | b) if self.p["leak"] else (cur & ~b)


def detects(pattern: Pattern, spec: FaultSpec, any_order: str) -> bool:
    sim = SparseSim(spec)
    addrs = sorted(set(spec.words))
    for _ in range(pattern.iterations):
        for step in pattern.steps:
            if isinstance(step, Pause):
                sim.pause(step.seconds)
                continue
            if isinstance(step, Hammer):
                continue
            assert isinstance(step, Element)
            order = any_order if step.order == "any" else step.order
            for a in (addrs if order == "up" else addrs[::-1]):
                idx = np.array([a], dtype=np.uint64)
                for op in step.ops:
                    val = int(background_values(op.background, idx, op.invert)[0])
                    if isinstance(op, Read):
                        if sim.read(a) != val:
                            return True
                    else:
                        sim.write(a, val)
    return False


def detected(pattern: Pattern, spec: FaultSpec) -> bool:
    """Guaranteed detection: caught whichever direction ``any`` elements run in."""
    return detects(pattern, spec, "up") and detects(pattern, spec, "down")


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
                  "value": int(rng.integers(2)), "astate": int(rng.integers(2))}
        return FaultSpec(model + ("-intra" if intra else ""), words, params)
    return gen


FAULT_MODELS: dict[str, Callable] = {
    "SAF": lambda rng: FaultSpec("SAF", (int(rng.integers(WORDS)),),
                                 {"bit": int(rng.integers(32)), "value": int(rng.integers(2))}),
    "TF": lambda rng: FaultSpec("TF", (int(rng.integers(WORDS)),),
                                {"bit": int(rng.integers(32)), "rising": bool(rng.integers(2))}),
    "CFin": _coupling("CFin", False),
    "CFid": _coupling("CFid", False),
    "CFst": _coupling("CFst", False),
    "CFin-intra": _coupling("CFin", True),
    "CFid-intra": _coupling("CFid", True),
    "AF": lambda rng: FaultSpec("AF", _two_words(rng)),
    "DRF": lambda rng: FaultSpec("DRF", (int(rng.integers(WORDS)),),
                                 {"bit": int(rng.integers(32)), "leak": int(rng.integers(2)),
                                  "retention_s": 30.0}),
}


@dataclass
class CoverageResult:
    models: list[str]
    patterns: list[str]
    rate: np.ndarray  # [pattern, model] detection rate
    hits: np.ndarray  # [pattern, model, trial] bool
    ops_per_word: dict[str, int]

    def suite_rate(self, names: list[str]) -> np.ndarray:
        idx = [self.patterns.index(n) for n in names]
        return self.hits[idx].any(axis=0).mean(axis=1)


def measure(patterns: dict[str, Pattern], trials: int = 200, seed: int = 0,
            models: list[str] | None = None) -> CoverageResult:
    models = models or list(FAULT_MODELS)
    rng = np.random.default_rng(seed)
    specs = {m: [FAULT_MODELS[m](rng) for _ in range(trials)] for m in models}
    names = list(patterns)
    hits = np.zeros((len(names), len(models), trials), dtype=bool)
    for i, name in enumerate(names):
        for j, m in enumerate(models):
            for t, spec in enumerate(specs[m]):
                hits[i, j, t] = detected(patterns[name], spec)
    ops = {n: p.ops_per_word() * p.iterations for n, p in patterns.items()}
    return CoverageResult(models, names, hits.mean(axis=2), hits, ops)


def _pct(r: float) -> str:
    """Never round a miss up to 100% (or a hit down to 0%)."""
    if r >= 1.0:
        return "100%"
    if r <= 0.0:
        return "0%"
    return f"{min(max(r * 100, 0.1), 99.9):.1f}%"


def markdown_table(res: CoverageResult, suites: dict[str, list[str]] | None = None) -> str:
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

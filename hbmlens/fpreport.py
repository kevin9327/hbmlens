"""Generate docs/fault-primitives.md: validation against publications, findings, coverage matrix."""
from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor

from .fp import (DYNAMIC_SINGLE, DYNAMIC_TWO, STATIC_SINGLE, STATIC_TWO, realistic_single_cell,
                 realistic_two_cell)
from .fpsim import FAULT_H, SOFT_DELAY, TRANSIENT_LIFE, Array, evaluate
from .patterns.library import retention
from .patterns.march import ALL_TESTS, MARCH_TESTS, PROPOSED_TESTS, march_test

GROUPS = {
    "static 1-cell": lambda: STATIC_SINGLE,
    "static 2-cell": lambda: STATIC_TWO,
    "dynamic 1-cell": lambda: DYNAMIC_SINGLE,
    "dynamic 2-cell": lambda: DYNAMIC_TWO,
    "DRAM 1-cell hard": lambda: realistic_single_cell("h"),
    "DRAM 2-cell hard": lambda: realistic_two_cell("h"),
    "DRAM 1-cell transient": lambda: realistic_single_cell("t"),
    "DRAM 1-cell soft": lambda: realistic_single_cell("s"),
}
H_TEST = 5  # hammer count of the tests; partial faults need FAULT_H = H_TEST - 1 repetitions

# ---- published results the engine is checked against ------------------------------------

# Hamdioui, Al-Ars, van de Goor, VTS 2002, Table 4: coverage (%) of dynamic FFMs
TABLE4_TESTS = ["mats-plus", "march-c-minus", "march-b", "pmovi", "march-u", "march-sr", "march-la", "march-lr"]
TABLE4 = {
    "dRDF": [0, 0, 50, 50, 50, 50, 50, 50], "dDRDF": [0, 0, 0, 50, 0, 0, 50, 0],
    "dIRF": [0, 0, 50, 50, 50, 50, 50, 50], "dCFds (transition write)": [0, 0, 50, 87.5, 50, 50, 100, 50],
    "dCFds (non-transition write)": [0, 0, 0, 0, 0, 0, 0, 0], "dCFrd": [0, 0, 25, 50, 25, 25, 50, 25],
    "dCFdrd": [0, 0, 0, 37.5, 0, 0, 50, 0], "dCFir": [0, 0, 25, 50, 25, 25, 50, 25],
}

# Al-Ars, van de Goor, Hamdioui, DATE 2006, Tables 5 and 6: general single-cell DRAM faults (dh SF,
# pidh others) and the (march element, operation) of March H1C that first detects each
TABLE6 = [("<0/1/->", 0, 4), ("<1/0/->", 1, 4), ("<0w0/1/->", 0, 4), ("<1w1/0/->", 1, 4),
          ("<0w1/0/->", 2, 4), ("<1w0/1/->", 3, 4), ("<0r0/0/1>", 0, 4), ("<1r1/1/0>", 1, 4),
          ("<0r0/1/0>", 0, 4), ("<1r1/0/1>", 1, 4), ("<0r0/1/1>", 0, 4), ("<1r1/0/0>", 1, 4)]


def table4_group(fp) -> str:
    if fp.ffm != "dCFds":
        return fp.ffm
    return "dCFds (transition write)" if fp.a_ops[0].value != fp.a_state else "dCFds (non-transition write)"


def _table4_column(col: int) -> dict:
    res = evaluate(march_test(TABLE4_TESTS[col]), DYNAMIC_SINGLE + DYNAMIC_TWO)
    groups: dict = {}
    for r in res:
        g = groups.setdefault(table4_group(r.fp), [0, 0])
        g[0] += r.detected
        g[1] += 1
    return {ffm: 100 * d / n for ffm, (d, n) in groups.items()}


def table4_check() -> list[tuple[str, str, float, float]]:
    """(test, FFM, published %, hbmlens %) for all 64 entries."""
    out = []
    for col, test in enumerate(TABLE4_TESTS):
        ours = _table4_column(col)
        out += [(test, ffm, row[col], ours[ffm]) for ffm, row in TABLE4.items()]
    return out


def table6_check() -> list[tuple[str, tuple[int, int], set]]:
    """(fault, published (element, op), hbmlens first detecting (element, op) over all runs)."""
    fps = []
    for notation, _, _ in TABLE6:
        fp = next(f for f in STATIC_SINGLE if f.notation == notation)
        fps.append(fp.with_attributes("" if fp.ffm == "SF" else "pi", True))
    res = evaluate(march_test("march-h1c", H_TEST), fps)
    return [(r.fp.name, (el, op), {(e, o + 1) for e, o in r.first} if r.detected else set())
            for r, (_, el, op) in zip(res, TABLE6, strict=True)]


def _pattern(row: str):
    if row == "retention (hbmlens)":
        return retention()
    if row == "march-h2c (h=4)":
        return march_test("march-h2c", 4)
    return march_test(row, H_TEST)


def _row(args):
    row, immediate = args
    pattern = _pattern(row)
    fault_h = 3 if row == "march-h2c (h=4)" else FAULT_H
    out = {}
    for group, fps in GROUPS.items():
        res = evaluate(pattern, fps(), h=fault_h, immediate=immediate)
        out[group] = (sum(r.detected for r in res), len(res),
                      sorted({f"{r.fp.name}{' ' + r.position if r.position else ''}" for r in res if not r.detected}))
    return row, pattern.ops_per_word(), out


ROWS = list(MARCH_TESTS) + ["march-h2c (h=4)"] + list(PROPOSED_TESTS) + ["retention (hbmlens)"]


def compute(workers: int | None = None, immediate: str = "element") -> list:
    with ProcessPoolExecutor(max_workers=workers) as pool:
        return list(pool.map(_row, [(r, immediate) for r in ROWS]))


def _cell(d: int, n: int) -> str:
    return f"{d}/{n}" if d < n else f"**{d}/{n}**"


def markdown(rows: list, rows_time: list) -> str:
    groups = list(GROUPS)
    by_name = {name: out for name, _, out in rows}
    t4 = table4_check()
    t4_ok = sum(p == o for _, _, p, o in t4)
    t6 = table6_check()
    t6_ok = sum(first == {pub} for _, pub, first in t6)
    lines = [
        "# Fault primitive coverage of march tests",
        "",
        "Generated by `hbmlens fp-coverage` (`hbmlens/fp.py`, `hbmlens/fpsim.py`). Every number is an exact, "
        "exhaustive result of the simulation described below, not a sample.",
        "",
        "## What is measured",
        "",
        "Faults are fault primitives `<S/F/R>` from the memory test literature: the operation sequence S that "
        "sensitizes the fault, the faulty cell's value F afterwards and the read output R. A fault counts as "
        "**detected** by a test only if some read returns data different from what the test expects for "
        f"every placement of the faulty cells in a {Array().rows}x{Array().cols} array (row-major addresses, one "
        "bit line per column), every initial value of the cells involved (the others all 0 or all 1) and "
        "every direction of the test's `⇕` elements. Two-cell faults count once with the victim above and once "
        "below the aggressor, as in the publications.",
        "",
        "DRAM-specific attributes follow Al-Ars et al. (DATE 2006): *partial* faults need an operation repeated "
        f"(the tests hammer {H_TEST} times, the faults need {FAULT_H}); *dirty* faults stay invisible until "
        "another cell on the same bit line is accessed with opposite data; *transient* faults vanish after "
        f"{TRANSIENT_LIFE} operations; *soft* faults appear only after a delay element (longer than "
        f"{SOFT_DELAY:,} operations). Dynamic faults need their two operations back to back in one march "
        "element, as in the publications (a stricter timing reading is shown further down).",
        "",
        "## Validation against the publications",
        "",
        "| published result | source | hbmlens |",
        "|---|---|---|",
        "| Coverage of 8 dynamic fault models by MATS+, March C-, B, PMOVI, U, SR, LA, LR | "
        f"VTS 2002, Table 4 | {t4_ok} of {len(t4)} values reproduced |",
        f"| March SS detects all static simple faults | MTDT 2002 | "
        f"{_frac(by_name['march-ss'], 'static 1-cell', 'static 2-cell')} |",
        f"| March RAW1 detects all single-cell dynamic faults | VTS 2002 | "
        f"{_frac(by_name['march-raw1'], 'dynamic 1-cell')} |",
        f"| March RAW detects all dynamic faults | VTS 2002 | "
        f"{_frac(by_name['march-raw'], 'dynamic 1-cell', 'dynamic 2-cell')} |",
        "| March H1C: first detecting operation of each general single-cell DRAM fault | DATE 2006, Table 6 | "
        f"{t6_ok} of {len(t6)} reproduced |",
        f"| March H1C detects all single-cell hard DRAM faults | DATE 2006 | "
        f"{_frac(by_name['march-h1c'], 'DRAM 1-cell hard')}, see finding 1 |",
        f"| March H2C detects all two-cell hard DRAM faults | DATE 2006 | "
        f"{_frac(by_name['march-h2c'], 'DRAM 2-cell hard')} with h = {H_TEST}, "
        f"{_frac(by_name['march-h2c (h=4)'], 'DRAM 2-cell hard')} with h = 4, see finding 2 |",
        f"| March T1C detects all single-cell transient DRAM faults | IDT 2006 | "
        f"{_frac(by_name['march-t1c'], 'DRAM 1-cell transient')}, see finding 1 |",
        "",
        "`tests/test_fp.py` checks each of these rows, and that every fault primitive does exactly what its "
        "notation says when its sequence S is applied.",
        "",
        "## Findings",
        "",
        "1. **Hammered writes toggle a write-destructive cell.** A plain WDF `<0w0/1/->` flips on every write "
        "of the value it holds, so `w0` repeated h times leaves it faulty or not depending on h and the value "
        "it started from (in H1C also on whether the neighbour's `w1b` ran first). March H1C and T1C therefore "
        "miss the plain and dirty WDFs; the partial WDFs they were derived for are detected. Appending one "
        "same-value write, a completing bit-line write and a read to the first two elements fixes it: "
        f"**March H1C+** ({_frac(by_name['march-h1c-plus'], 'DRAM 1-cell hard')}, "
        f"{ALL_TESTS['march-h1c-plus'][1][0]}n + {ALL_TESTS['march-h1c-plus'][1][1]}hn) and **March T1C+** "
        f"({_frac(by_name['march-t1c-plus'], 'DRAM 1-cell transient')}, "
        f"{ALL_TESTS['march-t1c-plus'][1][0]}n + {ALL_TESTS['march-t1c-plus'][1][1]}hn), proposed here.",
        "2. **March H2C needs an even hammer count.** Its victim writes `w0h` start from a cell holding 1, so "
        "with odd h a write-destructive coupling fault (CFwd) toggles back before the next read: "
        f"{_frac(by_name['march-h2c'], 'DRAM 2-cell hard')} with h = {H_TEST}, "
        f"{_frac(by_name['march-h2c (h=4)'], 'DRAM 2-cell hard')} with h = 4 (and 6).",
        "3. **Element boundaries under strict timing.** If \"back to back\" means consecutive in time rather "
        "than inside one march element, a `⇕(w0)` run downward ends on the cell where the next `⇑` element "
        "starts, so that cell sees `w0` and `r0` back to back across the boundary. "
        + _time_diff(rows, rows_time) + " Running the init element upward removes it.",
        "",
        "## Coverage matrix",
        "",
        "Detected / cases per group; bold = complete. `ops/word` is the test length for h = "
        f"{H_TEST} (March H2C (h=4): h = 4).",
        "",
        "| test | ops/word | " + " | ".join(groups) + " |",
        "|---|---|" + "---|" * len(groups),
    ]
    for name, ops, out in rows:
        label = f"{name} (proposed)" if name in PROPOSED_TESTS else name
        lines.append(f"| {label} | {ops} | " + " | ".join(_cell(*out[g][:2]) for g in groups) + " |")
    lines += ["", "## References", ""]
    refs = sorted({ref for _, _, ref in MARCH_TESTS.values()})
    lines += [f"- {r}" for r in refs]
    lines += [
        "- Fault primitive notation and static catalogues: A.J. van de Goor and Z. Al-Ars, VTS 2000; "
        "Z. Al-Ars, A.J. van de Goor, S. Hamdioui, DATE 2006, Tables 1-2",
        "- Dynamic fault catalogues: S. Hamdioui, Z. Al-Ars, A.J. van de Goor, VTS 2002, Tables 1-2",
        "- DRAM-specific fault space: Z. Al-Ars, A.J. van de Goor, S. Hamdioui, DATE 2006, Section 3",
        "",
    ]
    return "\n".join(lines)


def _frac(out: dict, *groups: str) -> str:
    d = sum(out[g][0] for g in groups)
    n = sum(out[g][1] for g in groups)
    return f"{d}/{n}"


def _time_diff(rows: list, rows_time: list) -> str:
    changes = []
    for (name, _, a), (_, _, b) in zip(rows, rows_time, strict=True):
        for g in ("dynamic 1-cell", "dynamic 2-cell"):
            if a[g][0] != b[g][0]:
                lost = sorted(set(b[g][2]) - set(a[g][2]))
                changes.append(f"{name} {g}: {a[g][0]} -> {b[g][0]} of {a[g][1]} ({', '.join(lost)})")
    if not changes:
        return "No test changes under the strict reading."
    return "Under the strict reading: " + "; ".join(changes) + "."

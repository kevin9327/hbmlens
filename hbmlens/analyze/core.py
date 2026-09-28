"""Fail log analysis: decode failing cells, classify failure signatures, draw fail bitmaps.

A *cell* is one failing bit of one word, ``(index, bit)``. Signatures are found
in priority order, and each cell is assigned to at most one signature:

1. ``dq-lane``  one data bit fails in many banks and rows of a pseudo channel
               (typical of a broken TSV or IO lane)
2. ``row``      many positions of one row of one bank fail
3. ``column``   one column (burst position) of one bank fails in many rows
4. ``bank``     failures spread over a large share of a bank's rows and columns
5. ``cell``     whatever is left: isolated failing bits

Thresholds are fractions of the geometry, so the same rules work from the tiny
test device up to full-size presets.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from ..faults import FaultSet
from ..geometry import LEVELS, HBMGeometry
from ..mapping import LinearMapper, bank_key
from ..records import FailLog

BANK_LEVELS = ("stack", "channel", "pseudo_channel", "sid", "bank_group", "bank")
PC_LEVELS = ("stack", "channel", "pseudo_channel")


@dataclass
class Signature:
    kind: str
    location: dict
    bits: list[int]
    cells: int
    words: int
    first_element: int = 0

    def label(self) -> str:
        loc = ", ".join(f"{k}={v}" for k, v in self.location.items())
        bits = ",".join(map(str, self.bits[:6])) + ("..." if len(self.bits) > 6 else "")
        return f"{self.kind}[{loc}] bits={bits} ({self.words} words, {self.cells} cells)"


@dataclass
class Rules:
    lane_bank_frac: float = 0.5  # of the banks in a pseudo channel
    lane_row_frac: float = 0.25
    row_pos_frac: float = 0.25  # of the word positions in a row
    col_row_frac: float = 0.25  # of the rows in a bank
    bank_frac: float = 0.25  # rows and columns of a bank
    min_count: int = 4


def fail_cells(log: FailLog, mapper: LinearMapper) -> pd.DataFrame:
    """One row per unique failing (index, bit) with decoded coordinates."""
    rec = log.records
    cols = {"index": [], "bit": [], "element": []}
    xor = rec["expected"] ^ rec["actual"]
    for b in range(32):
        hit = ((xor >> np.uint32(b)) & np.uint32(1)).astype(bool)
        if hit.any():
            cols["index"].append(rec["index"][hit])
            cols["bit"].append(np.full(int(hit.sum()), b, np.int16))
            cols["element"].append(rec["element"][hit])
    if not cols["index"]:
        return pd.DataFrame(columns=["index", "bit", "element", *LEVELS, "bank_key", "pos"])
    df = pd.DataFrame({k: np.concatenate(v) for k, v in cols.items()})
    df = df.groupby(["index", "bit"], as_index=False)["element"].min()
    fields = mapper.decode(df["index"].to_numpy(np.int64))
    for level in LEVELS:
        df[level] = fields[level]
    g = mapper.geometry
    df["bank_key"] = bank_key(fields, g)
    df["pos"] = df["column"] * g.words_per_column + df["word"]
    return df


def _sig(kind: str, grp: pd.DataFrame, keys: tuple[str, ...]) -> Signature:
    first = grp.iloc[0]
    return Signature(
        kind=kind,
        location={k: int(first[k]) for k in keys},
        bits=sorted(int(b) for b in grp["bit"].unique()),
        cells=len(grp),
        words=int(grp["index"].nunique()),
        first_element=int(grp["element"].min()),
    )


def classify(cells: pd.DataFrame, geometry: HBMGeometry, rules: Rules | None = None) -> list[Signature]:
    r = rules or Rules()
    g = geometry
    banks_per_pc = g.sids * g.bank_groups * g.banks_per_group
    left = cells.copy()
    sigs: list[Signature] = []

    def take(mask: pd.Series) -> pd.DataFrame:
        nonlocal left
        chosen = left[mask]
        left = left[~mask]
        return chosen

    # 1. DQ lane: one bit fails in most banks of a pseudo channel, each over many rows
    if len(left):
        keys = (*PC_LEVELS, "bit")
        per_bank = left.groupby([*keys, "bank_key"])["row"].nunique()
        strong = per_bank[per_bank >= max(r.min_count, r.lane_row_frac * g.rows)]
        banks = strong.groupby(level=list(range(len(keys)))).size()
        lanes = banks[banks >= max(2, r.lane_bank_frac * banks_per_pc)]
        if len(lanes):
            key = pd.MultiIndex.from_frame(left[list(keys)])
            chosen = take(pd.Series(key.isin(lanes.index), index=left.index))
            for _, grp in chosen.groupby(list(keys)):
                sigs.append(_sig("dq-lane", grp, PC_LEVELS))

    # 2. row, 3. column: judged per bit (a bitline or wordline segment feeds one DQ),
    # then bits at the same location are merged into one signature
    for kind, level, count_of, need in (
        ("row", "row", "pos", max(r.min_count, r.row_pos_frac * g.words_per_row)),
        ("column", "column", "row", max(r.min_count, r.col_row_frac * g.rows)),
    ):
        if not len(left):
            break
        keys = ("bank_key", level, "bit")
        stats = left.groupby(list(keys))[count_of].nunique()
        hits = stats[stats >= need]
        if len(hits):
            key = pd.MultiIndex.from_frame(left[list(keys)])
            chosen = take(pd.Series(key.isin(hits.index), index=left.index))
            for _, grp in chosen.groupby(["bank_key", level]):
                sigs.append(_sig(kind, grp, (*BANK_LEVELS, level)))

    # 4. bank: spread over many rows and columns of one bank
    if len(left):
        stats = left.groupby("bank_key").agg(rows=("row", "nunique"), cols=("column", "nunique"))
        hits = stats[(stats["rows"] >= max(r.min_count, r.bank_frac * g.rows))
                     & (stats["cols"] >= max(2, r.bank_frac * g.columns))]
        if len(hits):
            chosen = take(left["bank_key"].isin(hits.index))
            for _, grp in chosen.groupby("bank_key"):
                sigs.append(_sig("bank", grp, BANK_LEVELS))

    # 5. isolated cells
    for _, grp in left.groupby(["index", "bit"]):
        s = _sig("cell", grp, ())
        s.location = {"index": int(grp["index"].iloc[0])}
        sigs.append(s)
    return sigs


@dataclass
class Recovery:
    matched: list[tuple[str, str]] = field(default_factory=list)  # (fault label, signature label)
    missed: list[str] = field(default_factory=list)
    extra: list[str] = field(default_factory=list)
    masked: list[str] = field(default_factory=list)  # hidden inside another injected fault, same bit

    @property
    def recall(self) -> float:
        total = len(self.matched) + len(self.missed)
        return len(self.matched) / total if total else 1.0

    @property
    def precision(self) -> float:
        total = len(self.matched) + len(self.extra)
        return len(self.matched) / total if total else 1.0


def compare_with_truth(signatures: list[Signature], faults: FaultSet) -> Recovery:
    """Match injected static faults to signatures of the same kind, location and bit."""
    rec = Recovery()
    used: set[int] = set()
    for f in faults.faults:
        # a fault whose every word already fails on the same bit because of a larger
        # injected fault cannot be told apart from it; count it separately
        covering = [o for o in faults.faults if o is not f and o.bit == f.bit and len(o.indices) > len(f.indices)]
        if any(np.isin(f.indices, o.indices).all() for o in covering):
            rec.masked.append(f.label())
            continue
        found = None
        for i, s in enumerate(signatures):
            if i in used or s.kind != f.kind or f.bit not in s.bits:
                continue
            if all(s.location.get(k) == v for k, v in f.location.items()):
                found = i
                break
        if found is None:
            rec.missed.append(f.label())
        else:
            used.add(found)
            rec.matched.append((f.label(), signatures[found].label()))
    rec.extra = [s.label() for i, s in enumerate(signatures) if i not in used]
    return rec


def bank_bitmaps(cells: pd.DataFrame, geometry: HBMGeometry, max_banks: int = 64) -> dict[int, np.ndarray]:
    """bank_key -> bool array [rows, words_per_row] of failing words (banks with most fails first)."""
    out: dict[int, np.ndarray] = {}
    if not len(cells):
        return out
    words = cells.drop_duplicates("index")
    order = words["bank_key"].value_counts().index[:max_banks]
    for bk in order:
        sub = words[words["bank_key"] == bk]
        bm = np.zeros((geometry.rows, geometry.words_per_row), dtype=bool)
        bm[sub["row"].to_numpy(), sub["pos"].to_numpy()] = True
        out[int(bk)] = bm
    return out


def signature_banks(cells: pd.DataFrame, sig: Signature) -> list[int]:
    sel = cells["bit"].isin(sig.bits)
    for k, v in sig.location.items():
        sel &= cells[k] == v
    return [int(b) for b in cells.loc[sel, "bank_key"].unique()]


def representative_banks(cells: pd.DataFrame, signatures: list[Signature], limit: int = 16,
                         per_lane: int = 2) -> list[int]:
    """Banks to show first: one per row/column/bank signature, a couple per DQ lane, then cells."""
    order: list[int] = []
    rank = {"row": 0, "column": 1, "bank": 2, "dq-lane": 3, "cell": 4}
    for s in sorted(signatures, key=lambda s: rank.get(s.kind, 9)):
        banks = signature_banks(cells, s)
        for b in banks[: per_lane if s.kind == "dq-lane" else 1]:
            if b not in order:
                order.append(b)
        if len(order) >= limit:
            break
    return order[:limit]


def signature_panels(cells: pd.DataFrame, signatures: list[Signature], geometry: HBMGeometry,
                     limit: int = 16, per_lane: int = 2) -> list[tuple[str, np.ndarray]]:
    """(title, bitmap) panels, one per signature bank, drawn with only that signature's bits,
    so a column hidden inside a failing DQ lane is still visible."""
    panels: list[tuple[str, np.ndarray]] = []
    rank = {"row": 0, "column": 1, "bank": 2, "dq-lane": 3, "cell": 4}
    for s in sorted(signatures, key=lambda s: rank.get(s.kind, 9)):
        sel = cells["bit"].isin(s.bits)
        for k, v in s.location.items():
            sel &= cells[k] == v
        sub = cells[sel].drop_duplicates("index")
        for bk in list(sub["bank_key"].unique())[: per_lane if s.kind == "dq-lane" else 1]:
            part = sub[sub["bank_key"] == bk]
            bm = np.zeros((geometry.rows, geometry.words_per_row), dtype=bool)
            bm[part["row"].to_numpy(), part["pos"].to_numpy()] = True
            c = part.iloc[0]
            title = (f"{s.kind} bit {','.join(map(str, s.bits[:4]))} | s{int(c.stack)} ch{int(c.channel)} "
                     f"pc{int(c.pseudo_channel)} bg{int(c.bank_group)} b{int(c.bank)}\n{int(bm.sum()):,} failing words")
            panels.append((title, bm))
        if len(panels) >= limit:
            break
    return panels[:limit]


def save_panels(panels: list[tuple[str, np.ndarray]], path, cols: int = 4):
    """Grid of fail bitmaps (like wafer maps)."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    n = max(len(panels), 1)
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(3.2 * cols, 3.0 * rows), squeeze=False)
    for ax in axes.ravel():
        ax.axis("off")
    for ax, (title, bm) in zip(axes.ravel(), panels):
        ax.axis("on")
        ax.imshow(bm, cmap="magma", vmin=0, vmax=1, interpolation="nearest", aspect="auto")
        if 0 < bm.sum() <= 16:
            ys, xs = np.nonzero(bm)
            ax.scatter(xs, ys, s=90, facecolors="none", edgecolors="#7CFF6B", linewidths=1.2)
        ax.set_title(title, fontsize=7.5)
        ax.set_xlabel("word position in row", fontsize=7)
        ax.set_ylabel("row", fontsize=7)
        ax.tick_params(labelsize=6)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)
    return path


def save_bitmaps(bitmaps: dict[int, np.ndarray], cells: pd.DataFrame, path, cols: int = 4):
    """Grid of fail bitmaps (like wafer maps), one panel per bank."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    n = max(len(bitmaps), 1)
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(3.2 * cols, 3.0 * rows), squeeze=False)
    names = cells.drop_duplicates("bank_key").set_index("bank_key") if len(cells) else None
    for ax in axes.ravel():
        ax.axis("off")
    for ax, (bk, bm) in zip(axes.ravel(), bitmaps.items()):
        ax.imshow(bm, cmap="magma", vmin=0, vmax=1, interpolation="nearest", aspect="auto")
        n_fail = int(bm.sum())
        if 0 < n_fail <= 16:  # isolated cells are single pixels: circle them
            ys, xs = np.nonzero(bm)
            ax.scatter(xs, ys, s=90, facecolors="none", edgecolors="#7CFF6B", linewidths=1.2)
        if names is not None:
            c = names.loc[bk]
            ax.set_title(f"s{int(c.stack)} ch{int(c.channel)} pc{int(c.pseudo_channel)} "
                         f"bg{int(c.bank_group)} b{int(c.bank)}\n{n_fail:,} failing words", fontsize=8)
        ax.set_xlabel("word position in row", fontsize=7)
        ax.set_ylabel("row", fontsize=7)
        ax.axis("on")
        ax.tick_params(labelsize=6)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)
    return path


def report_markdown(log: FailLog, mapper: LinearMapper, signatures: list[Signature],
                    recovery: Recovery | None = None, max_cells: int = 20) -> str:
    m = log.meta
    lines = [
        f"# hbmlens report: {m.pattern} on {m.backend}",
        "",
        f"- device: {mapper.geometry.describe()}",
        f"- address map (assumed): {mapper.describe()}",
        f"- temperature: {m.temperature_c} degC" if m.temperature_c is not None else "- temperature: n/a",
        f"- failing reads: {m.total_fails:,} (recorded {m.recorded:,}{', overflow' if m.overflow else ''})",
        f"- failing words: {len(log.failing_indices()):,}",
        "",
        "## Signatures",
        "",
        "| kind | location | bits | words | cells | first element |",
        "|---|---|---|---|---|---|",
    ]
    cell_sigs = [s for s in signatures if s.kind == "cell"]
    for s in [s for s in signatures if s.kind != "cell"] + cell_sigs[:max_cells]:
        loc = ", ".join(f"{k}={v}" for k, v in s.location.items())
        lines.append(f"| {s.kind} | {loc} | {','.join(map(str, s.bits[:8]))} | {s.words} | {s.cells} | {s.first_element} |")
    if len(cell_sigs) > max_cells:
        lines.append(f"| cell | ... {len(cell_sigs) - max_cells} more isolated cells | | | | |")
    if recovery is not None:
        lines += [
            "",
            "## Against injected ground truth",
            "",
            f"- recall {recovery.recall:.0%}, precision {recovery.precision:.0%}",
            f"- matched {len(recovery.matched)}, missed {len(recovery.missed)}, extra {len(recovery.extra)}",
        ]
        lines += [f"- missed: {x}" for x in recovery.missed]
        lines += [f"- extra: {x}" for x in recovery.extra[:10]]
    lines += ["", "Synthetic faults are labeled as such; the address map is an assumption, not a vendor map."]
    return "\n".join(lines) + "\n"

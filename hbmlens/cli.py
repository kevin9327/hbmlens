"""Command line entry point: ``hbmlens demo | run | analyze``."""
from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

import click

from .analyze.core import (Recovery, bank_bitmaps, classify, compare_with_truth, fail_cells,
                           report_markdown, save_panels, signature_panels)
from .backends.virtual import VirtualBackend
from .faults import FaultSet, demo_faults
from .geometry import PRESETS, get_geometry
from .mapping import MAPPERS, LinearMapper, get_mapper
from .patterns.base import Pattern
from .patterns.library import PATTERNS, get_pattern
from .records import FailLog
from .viz import viewer_data, write_viewer

GEOMETRY = click.option("--geometry", default="small", type=click.Choice(sorted(PRESETS)), show_default=True)
MAPPER = click.option("--mapper", "mapper_name", default="interleaved", type=click.Choice(sorted(MAPPERS)),
                      show_default=True, help="assumed address map")


def analyze_to(out_dir: Path, log: FailLog, mapper: LinearMapper, faults: FaultSet | None = None,
               pattern: Pattern | None = None) -> tuple[list, Recovery | None]:
    """Decode, classify, write report.md, bitmaps.png, signatures.json and viewer.html."""
    out_dir.mkdir(parents=True, exist_ok=True)
    cells = fail_cells(log, mapper)
    sigs = classify(cells, mapper.geometry)
    rec = compare_with_truth(sigs, faults) if faults is not None and faults.faults else None
    bitmaps = bank_bitmaps(cells, mapper.geometry)
    (out_dir / "report.md").write_text(report_markdown(log, mapper, sigs, rec), encoding="utf-8")
    (out_dir / "signatures.json").write_text(json.dumps([asdict(s) for s in sigs], indent=1), encoding="utf-8")
    panels = signature_panels(cells, sigs, mapper.geometry)
    if panels:
        save_panels(panels, out_dir / "bitmaps.png")
    data = viewer_data(log, mapper, cells, faults=faults, signatures=sigs, bitmaps=bitmaps, pattern=pattern)
    write_viewer(out_dir / "viewer.html", data)
    return sigs, rec


def _print_summary(log: FailLog, sigs: list, rec: Recovery | None, out_dir: Path) -> None:
    kinds: dict[str, int] = {}
    for s in sigs:
        kinds[s.kind] = kinds.get(s.kind, 0) + 1
    t = f" in {log.meta.elapsed_s:.2f}s" if log.meta.elapsed_s else ""
    click.echo(f"{log.meta.total_fails:,} failing reads on {len(log.failing_indices()):,} words{t}")
    click.echo("signatures: " + (", ".join(f"{k} x{v}" for k, v in kinds.items()) or "none"))
    if log.meta.overflow:
        click.echo(f"warning: incomplete fail log ({log.meta.recorded:,} of {log.meta.total_fails:,} failing "
                   "reads recorded); signatures describe the recorded part only, see the report")
    if rec is not None:
        click.echo(f"vs injected: recall {rec.recall:.0%}, precision {rec.precision:.0%}"
                   + (f", {len(rec.masked)} masked" if rec.masked else ""))
    click.echo(f"report: {(out_dir / 'report.md').resolve()}")
    click.echo(f"viewer: {(out_dir / 'viewer.html').resolve()}")


@click.group()
def main() -> None:
    """hbmlens: read HBM failures like an inspection image."""


@main.command()
@GEOMETRY
@MAPPER
@click.option("--pattern", default="march-c-minus", type=click.Choice(sorted(PATTERNS)), show_default=True)
@click.option("--temperature", default=25.0, show_default=True, help="device temperature in degC")
@click.option("--seed", default=7, show_default=True)
@click.option("--out", default="out/demo", show_default=True, type=click.Path(file_okay=False))
def demo(geometry: str, mapper_name: str, pattern: str, temperature: float, seed: int, out: str) -> None:
    """Inject a mixed fault population into a virtual HBM, test it, analyze it, draw it."""
    mapper = get_mapper(mapper_name, geometry)
    faults = demo_faults(mapper, seed)
    pat = get_pattern(pattern)
    log = VirtualBackend(geometry, faults, temperature).run(pat, run_id=f"demo-{seed}")
    out_dir = Path(out)
    out_dir.mkdir(parents=True, exist_ok=True)
    log.save(out_dir / "fails.parquet")
    click.echo(mapper.geometry.describe())
    for line in faults.describe():
        click.echo(f"  injected {line}")
    sigs, rec = analyze_to(out_dir, log, mapper, faults, pat)
    _print_summary(log, sigs, rec, out_dir)


@main.command()
@click.option("--backend", default="virtual", type=click.Choice(["virtual", "cuda"]), show_default=True)
@GEOMETRY
@MAPPER
@click.option("--pattern", default="march-c-minus", type=click.Choice(sorted(PATTERNS)), show_default=True)
@click.option("--inject/--no-inject", default=False, show_default=True,
              help="lay the demo fault population over the device (virtual faults, or a read overlay on GPU)")
@click.option("--temperature", default=25.0, show_default=True, help="virtual backend only")
@click.option("--seed", default=7, show_default=True)
@click.option("--max-records", default=1 << 20, show_default=True,
              help="detailed fail records to keep (the failure count is always exact)")
@click.option("--out", default="out/run.parquet", show_default=True, type=click.Path(dir_okay=False))
def run(backend: str, geometry: str, mapper_name: str, pattern: str, inject: bool, temperature: float,
        seed: int, max_records: int, out: str) -> None:
    """Run one pattern and save the fail log (Parquet)."""
    g = get_geometry(geometry)
    faults = demo_faults(get_mapper(mapper_name, g), seed) if inject else FaultSet()
    pat = get_pattern(pattern)
    if backend == "virtual":
        log = VirtualBackend(g, faults, temperature).run(pat, max_records=max_records)
    else:
        from .backends.cuda import CudaBackend

        try:
            dev = CudaBackend(g.total_words, g.name, overlay=faults.to_overlay())
        except MemoryError as e:
            raise click.ClickException(f"{e}; choose a smaller --geometry") from None
        click.echo(f"GPU: {dev.info()['device']}, {g.total_bytes / 2**20:.1f} MiB under test")
        log = dev.run(pat, max_records=max_records)
        if not log.meta.notes.get("dram_faithful"):
            click.echo(f"note: {g.total_bytes / 2**20:.0f} MiB is under 4x the GPU's L2 cache "
                       f"({dev.l2_bytes / 2**20:.0f} MiB); some accesses may be served from cache, not DRAM")
    log.meta.notes["mapper"] = mapper_name
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    log.save(out)
    bw = f", {log.meta.bandwidth_gbps:.0f} GB/s" if log.meta.bandwidth_gbps else ""
    click.echo(f"{pattern} on {backend}: {log.meta.total_fails:,} failing reads, "
               f"{log.meta.elapsed_s:.3f}s{bw} -> {out}")
    if log.meta.overflow:
        click.echo(f"warning: only {log.meta.recorded:,} failing reads were recorded in detail "
                   "(raise --max-records or test a smaller region)")


@main.command()
@click.argument("log_path", type=click.Path(exists=True, dir_okay=False))
@GEOMETRY
@MAPPER
@click.option("--out", default="out/analysis", show_default=True, type=click.Path(file_okay=False))
def analyze(log_path: str, geometry: str, mapper_name: str, out: str) -> None:
    """Analyze a saved fail log: signatures, fail bitmaps, report and 3D viewer."""
    log = FailLog.load(log_path)
    mapper = get_mapper(log.meta.notes.get("mapper", mapper_name), log.meta.geometry
                        if log.meta.geometry in PRESETS else geometry)
    pat = get_pattern(log.meta.pattern) if log.meta.pattern in PATTERNS else None
    sigs, rec = analyze_to(Path(out), log, mapper, None, pat)
    _print_summary(log, sigs, rec, Path(out))


@main.command()
@click.option("--trials", default=500, show_default=True, help="random fault placements per model")
@click.option("--seed", default=0, show_default=True)
@click.option("--out", default="docs/coverage.md", show_default=True, type=click.Path(dir_okay=False))
def coverage(trials: int, seed: int, out: str) -> None:
    """Measure which pattern detects which classic fault model; compare test suites."""
    from .coverage import FAULT_INFO, RETENTION_S, cheapest_suite, markdown_table, measure, suite_table
    from .patterns.base import concat
    from .patterns.library import CUDA_MEMTEST_STYLE, HBMLENS_SUITE

    pats = {name: PATTERNS[name]() for name in PATTERNS}
    res = measure(pats, trials=trials, seed=seed)
    suites = {
        "cuda_memtest-style (8 tests)": CUDA_MEMTEST_STYLE,
        "March C- + retention": ["march-c-minus", "retention"],
        "word-oriented March C- + retention": ["march-c-minus-wom", "retention"],
        "hbmlens: " + " + ".join(HBMLENS_SUITE): HBMLENS_SUITE,
    }
    # same seed -> the same fault instances as the per-pattern measurement
    res_suites = measure({s: concat(s, [pats[n] for n in names]) for s, names in suites.items()},
                         trials=trials, seed=seed)
    best = cheapest_suite(res)
    ops, _ = res.suite_cost(best) if best else (0, 0.0)
    models = "\n".join(f"- `{m}` {FAULT_INFO[m]}" for m in res.models)
    text = (
        "# Fault coverage of memory test patterns\n\n"
        f"Measured with `hbmlens coverage --trials {trials} --seed {seed}`. Each fault model is placed at "
        f"{trials} random locations in a 1024-word memory of 32-bit words and every pattern is simulated "
        "bit-exactly (`hbmlens/coverage.py`). A fault counts as detected only if detection is guaranteed: "
        "for every initial value of the bits it involves (memory content before a test is unknown) and for "
        "both address orders wherever the pattern allows either. A suite is simulated as one program, its "
        "patterns back to back in the listed order, the way a test tool runs it.\n\n"
        "Fault models (the static fault taxonomy of the memory test literature):\n\n" + models + "\n\n"
        "The cuda_memtest-style patterns are re-implemented from that project's public test list; they are "
        "not its original code.\n\n"
        "## Suites\n\n" + suite_table(res_suites) + "\n"
        + (f"Cheapest combination of library patterns that detects every simulated fault, each pattern on its "
           f"own and in any order (exhaustive search, `hbmlens.coverage.cheapest_suite`): "
           f"**{' + '.join(best)}**, {ops} memory operations per word.\n\n"
           if best else "")
        + "## Reading the table\n\n"
        "- Deceptive read destructive faults (DRDF, CFdrd) return the right value and flip the cell, so only a "
        "second read of the same cell sees them. March C- and moving inversions always write right after a "
        "read, which hides the flip. March SS reads twice in a row.\n"
        "- Write destructive faults (WDF, CFwd) need a write of the value a cell already holds. March C- never "
        "does that on purpose; suites that change data between tests or backgrounds do it for some bits by "
        "accident, which is why they land between 0% and 100%.\n"
        "- March SS covers every single-cell and two-cell static fault in a bit-oriented memory. In a 32-bit "
        "word, two bits that always hold the same value are never tested against each other; the 25N "
        "intra-word test adds five data backgrounds for that.\n"
        f"- Retention faults need a pause. The DRF model here leaks after {RETENTION_S:g} s, so a 64 s pause "
        "and a 90 min pause both catch it; cells that leak between the two are only caught by the longer "
        "pause (`retention(pause_s)` is configurable).\n"
        "- Idempotent coupling and moving inversions: an aggressor at a lower address that rises and forces "
        "the same bit of a higher word to 0 (or falls and forces 1) is missed with solid data "
        "(`tests/test_coverage.py::test_moving_inversions_suite_misses_this_idempotent_coupling`).\n\n"
        "## Every pattern\n\n" + markdown_table(res)
    )
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    Path(out).write_text(text, encoding="utf-8")
    click.echo(suite_table(res_suites))
    if best:
        click.echo(f"cheapest complete suite: {' + '.join(best)} ({ops} ops/word)")
    click.echo(f"written {out}")


if __name__ == "__main__":
    main()

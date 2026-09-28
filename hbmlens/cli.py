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
@click.option("--out", default="out/run.parquet", show_default=True, type=click.Path(dir_okay=False))
def run(backend: str, geometry: str, mapper_name: str, pattern: str, inject: bool, temperature: float,
        seed: int, out: str) -> None:
    """Run one pattern and save the fail log (Parquet)."""
    g = get_geometry(geometry)
    faults = demo_faults(get_mapper(mapper_name, g), seed) if inject else FaultSet()
    pat = get_pattern(pattern)
    if backend == "virtual":
        log = VirtualBackend(g, faults, temperature).run(pat)
    else:
        from .backends.cuda import CudaBackend

        dev = CudaBackend(g.total_words, g.name, overlay=faults.to_overlay())
        click.echo(f"GPU: {dev.info()['device']}, {g.total_bytes / 2**20:.1f} MiB under test")
        log = dev.run(pat)
    log.meta.notes["mapper"] = mapper_name
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    log.save(out)
    bw = f", {log.meta.bandwidth_gbps:.0f} GB/s" if log.meta.bandwidth_gbps else ""
    click.echo(f"{pattern} on {backend}: {log.meta.total_fails:,} failing reads, "
               f"{log.meta.elapsed_s:.3f}s{bw} -> {out}")


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


if __name__ == "__main__":
    main()

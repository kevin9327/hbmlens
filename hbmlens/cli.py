"""Command line entry point."""
from __future__ import annotations

from pathlib import Path

import click

from .backends.virtual import VirtualBackend
from .faults import demo_faults
from .geometry import PRESETS
from .mapping import MAPPERS, get_mapper
from .patterns.library import PATTERNS, get_pattern
from .viz import write_viewer


@click.group()
def main() -> None:
    """hbmlens: read HBM failures like an inspection image."""


@main.command()
@click.option("--geometry", default="small", type=click.Choice(sorted(PRESETS)), show_default=True)
@click.option("--mapper", "mapper_name", default="interleaved", type=click.Choice(sorted(MAPPERS)), show_default=True)
@click.option("--pattern", default="march-c-minus", type=click.Choice(sorted(PATTERNS)), show_default=True)
@click.option("--temperature", default=25.0, show_default=True, help="device temperature in degC")
@click.option("--seed", default=7, show_default=True)
@click.option("--out", default="out/demo", show_default=True, type=click.Path(file_okay=False))
def demo(geometry: str, mapper_name: str, pattern: str, temperature: float, seed: int, out: str) -> None:
    """Inject a mixed fault population into a virtual HBM, test it and write a 3D viewer."""
    mapper = get_mapper(mapper_name, geometry)
    faults = demo_faults(mapper, seed)
    log = VirtualBackend(geometry, faults, temperature).run(get_pattern(pattern), run_id=f"demo-{seed}")
    out_dir = Path(out)
    out_dir.mkdir(parents=True, exist_ok=True)
    log.save(out_dir / "fails.parquet")
    viewer = write_viewer(out_dir / "viewer.html", log, mapper, faults)
    click.echo(mapper.geometry.describe())
    for line in faults.describe():
        click.echo(f"  injected {line}")
    click.echo(f"{log.meta.total_fails} failing reads on {len(log.failing_indices())} words "
               f"in {log.meta.elapsed_s:.2f}s")
    click.echo(f"viewer: {viewer.resolve()}")


if __name__ == "__main__":
    main()

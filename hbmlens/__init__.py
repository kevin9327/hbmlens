"""hbmlens: read HBM failures like an inspection image.

Fault injection on a virtual HBM, memory test patterns that run on a virtual
device or on a real GPU (CUDA), exact fail logging and fail-bitmap analysis.
Everything is based on public information; see docs/disclaimer.md.
"""
from .geometry import PRESETS, HBMGeometry, get_geometry
from .mapping import LinearMapper, get_mapper, interleaved, linear

__version__ = "0.1.0.dev0"
__all__ = ["HBMGeometry", "PRESETS", "get_geometry", "LinearMapper", "get_mapper", "linear", "interleaved"]

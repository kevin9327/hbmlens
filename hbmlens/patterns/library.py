"""Standard memory test patterns expressed with :mod:`hbmlens.patterns.base`."""
from __future__ import annotations

from .base import ONES, ZERO, Background, Element, Pattern, Pause, Read, Write


def _w(bg: Background, inv: bool = False) -> Write:
    return Write(bg, inv)


def _r(bg: Background, inv: bool = False) -> Read:
    return Read(bg, inv)


def mats_plus() -> Pattern:
    """MATS+ {any(w0); up(r0,w1); down(r1,w0)} - stuck-at and address decoder faults."""
    return Pattern(
        "mats-plus",
        (
            Element("any", (_w(ZERO),)),
            Element("up", (_r(ZERO), _w(ONES))),
            Element("down", (_r(ONES), _w(ZERO))),
        ),
        description="MATS+: 5N, stuck-at and address decoder faults",
    )


def march_c_minus(bg: Background = ZERO) -> Pattern:
    """March C- {any(w0); up(r0,w1); up(r1,w0); down(r0,w1); down(r1,w0); any(r0)} (10N)."""
    return Pattern(
        "march-c-minus",
        (
            Element("any", (_w(bg),)),
            Element("up", (_r(bg), _w(bg, True))),
            Element("up", (_r(bg, True), _w(bg))),
            Element("down", (_r(bg), _w(bg, True))),
            Element("down", (_r(bg, True), _w(bg))),
            Element("any", (_r(bg),)),
        ),
        description="March C-: 10N, stuck-at, transition and most coupling faults",
    )


def walking_ones() -> Pattern:
    """Write and read back a single 1 moving across the 32 data bits (DQ lane test)."""
    steps = []
    for bit in range(32):
        b = Background("solid", 1 << bit)
        steps.append(Element("any", (_w(b), _r(b))))
    return Pattern("walking-ones", tuple(steps), description="walking 1 over 32 data bits")


def checkerboard() -> Pattern:
    b = Background("checker", 0x55555555)
    return Pattern(
        "checkerboard",
        (Element("any", (_w(b),)), Element("any", (_r(b),)), Element("any", (_w(b, True),)), Element("any", (_r(b, True),))),
        description="alternating 0101/1010 words and the inverse",
    )


def moving_inversions(iterations: int = 2) -> Pattern:
    """memtest-style moving inversions with address-in-address data."""
    b = Background("addr")
    return Pattern(
        "moving-inversions",
        (
            Element("up", (_w(b),)),
            Element("up", (_r(b), _w(b, True))),
            Element("down", (_r(b, True), _w(b))),
            Element("any", (_r(b),)),
        ),
        iterations=iterations,
        description="moving inversions over address-in-address data (also catches aliasing)",
    )


def random_data(seed: int = 1) -> Pattern:
    b = Background("random", seed=seed)
    return Pattern(
        "random",
        (Element("any", (_w(b),)), Element("any", (_r(b),)), Element("any", (_w(b, True),)), Element("any", (_r(b, True),))),
        description="reproducible pseudo random data and its inverse",
    )


def retention(pause_s: float = 64.0) -> Pattern:
    """Write ones, wait, read; then zeros. Weak cells leak during the pause (worse when hot)."""
    return Pattern(
        "retention",
        (
            Element("any", (_w(ONES),)),
            Pause(pause_s),
            Element("any", (_r(ONES),)),
            Element("any", (_w(ZERO),)),
            Pause(pause_s),
            Element("any", (_r(ZERO),)),
        ),
        description=f"data retention with {pause_s:g}s pauses",
    )


PATTERNS = {
    "mats-plus": mats_plus,
    "march-c-minus": march_c_minus,
    "walking-ones": walking_ones,
    "checkerboard": checkerboard,
    "moving-inversions": moving_inversions,
    "random": random_data,
    "retention": retention,
}


def get_pattern(name: str) -> Pattern:
    try:
        return PATTERNS[name]()
    except KeyError:
        raise KeyError(f"unknown pattern {name!r}; choose from {sorted(PATTERNS)}") from None

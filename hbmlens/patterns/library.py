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
        (Element("any", (_w(b),)), Element("any", (_r(b),)),
         Element("any", (_w(b, True),)), Element("any", (_r(b, True),))),
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
        (Element("any", (_w(b),)), Element("any", (_r(b),)),
         Element("any", (_w(b, True),)), Element("any", (_r(b, True),))),
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


def march_ss(bg: Background = ZERO) -> Pattern:
    """March SS (22N), Hamdioui et al. 2002: every static simple fault of a bit-oriented memory.

    {any(w0); up(r0,r0,w0,r0,w1); up(r1,r1,w1,r1,w0); down(r0,r0,w0,r0,w1); down(r1,r1,w1,r1,w0); any(r0)}

    The back-to-back reads expose deceptive read destructive faults and the
    same-value writes expose write destructive faults; March C- has neither.
    """
    def body(inv: bool) -> tuple:
        return (_r(bg, inv), _r(bg, inv), _w(bg, inv), _r(bg, inv), _w(bg, not inv))

    return Pattern(
        "march-ss",
        (
            Element("any", (_w(bg),)),
            Element("up", body(False)),
            Element("up", body(True)),
            Element("down", body(False)),
            Element("down", body(True)),
            Element("any", (_r(bg),)),
        ),
        description="March SS: 22N, all static simple faults incl. read/write destructive",
    )


WOM_BACKGROUNDS = (0x00000000, 0x55555555, 0x33333333, 0x0F0F0F0F, 0x00FF00FF, 0x0000FFFF)


def march_c_minus_wom() -> Pattern:
    """Word-oriented March C-: March C- over the log2(32)+1 standard data backgrounds, so
    every pair of bits inside a 32-bit word sees both equal and opposite values
    (needed for intra-word coupling faults). 60N."""
    steps: list = []
    for v in WOM_BACKGROUNDS:
        steps += list(march_c_minus(Background("solid", v)).steps)
    return Pattern("march-c-minus-wom", tuple(steps),
                   description="March C- over 6 word-oriented data backgrounds (60N)")


def intra_word() -> Pattern:
    """Intra-word coupling test (25N): for each non-solid standard background,
    any(w~bg, wbg, rbg, w~bg, r~bg). Both transitions happen on every word and are
    read back, so each pair of bits that differs in the background is tested in both
    directions; pairs with equal bits are covered by any solid-background march."""
    steps = []
    for v in WOM_BACKGROUNDS[1:]:
        b = Background("solid", v)
        steps.append(Element("any", (_w(b, True), _w(b), _r(b), _w(b, True), _r(b, True))))
    return Pattern("intra-word", tuple(steps),
                   description="intra-word coupling over 5 data backgrounds (25N)")


# ---- re-implementations of the cuda_memtest test list (from its public README) ----
# These follow the published test descriptions; they are not the original code.

def _moving_inversions(bg: Background) -> tuple:
    return (Element("up", (_w(bg),)), Element("up", (_r(bg), _w(bg, True))),
            Element("down", (_r(bg, True), _w(bg))))


def own_address() -> Pattern:
    b = Background("addr")
    return Pattern("own-address", (Element("any", (_w(b),)), Element("any", (_r(b),))),
                   description="cuda_memtest-style test 1: each word holds its own address")


def mi_ones_zeros() -> Pattern:
    return Pattern("mi-ones-zeros", _moving_inversions(ZERO) + _moving_inversions(ONES),
                   description="cuda_memtest-style test 2: moving inversions, ones and zeros")


def mi_8bit() -> Pattern:
    steps: tuple = ()
    for k in range(8):
        steps += _moving_inversions(Background("solid", 0x01010101 << k))
    return Pattern("mi-8bit", steps, description="cuda_memtest-style test 3: moving inversions, 8-bit walking pattern")


def mi_random(seed: int = 3) -> Pattern:
    return Pattern("mi-random", _moving_inversions(Background("random", seed=seed)),
                   description="cuda_memtest-style test 4: moving inversions, random pattern")


def mi_32bit() -> Pattern:
    steps: tuple = ()
    for k in range(32):
        steps += _moving_inversions(Background("solid", 1 << k))
    return Pattern("mi-32bit", steps,
                   description="cuda_memtest-style test 6: moving inversions, 32-bit walking pattern")


def bit_fade(pause_s: float = 5400.0) -> Pattern:
    p = retention(pause_s)
    return Pattern("bit-fade", p.steps, description="cuda_memtest-style test 9: bit fade (90 min pauses)")


HBMLENS_SUITE = ["march-ss", "intra-word", "retention"]

CUDA_MEMTEST_STYLE = ["walking-ones", "own-address", "mi-ones-zeros", "mi-8bit", "mi-random", "mi-32bit",
                      "random", "bit-fade"]

PATTERNS = {
    "mats-plus": mats_plus,
    "march-c-minus": march_c_minus,
    "walking-ones": walking_ones,
    "checkerboard": checkerboard,
    "moving-inversions": moving_inversions,
    "random": random_data,
    "retention": retention,
    "march-c-minus-wom": march_c_minus_wom,
    "march-ss": march_ss,
    "intra-word": intra_word,
    "own-address": own_address,
    "mi-ones-zeros": mi_ones_zeros,
    "mi-8bit": mi_8bit,
    "mi-random": mi_random,
    "mi-32bit": mi_32bit,
    "bit-fade": bit_fade,
}


def get_pattern(name: str) -> Pattern:
    try:
        return PATTERNS[name]()
    except KeyError:
        raise KeyError(f"unknown pattern {name!r}; choose from {sorted(PATTERNS)}") from None

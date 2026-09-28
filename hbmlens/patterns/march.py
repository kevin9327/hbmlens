"""March tests written in the notation of the memory test literature.

``parse_march("{⇕(w0); ⇑(r0,w1); ⇓(r1,w0)}")`` builds a :class:`Pattern`. Address
orders: ``⇑``/``up``, ``⇓``/``down``, ``⇕``/``any``. Operations: ``r0 r1 w0 w1``;
suffix ``h`` repeats the operation (a hammer, ``h`` times), suffix ``b`` applies it to
another cell on the same bit line (DRAM-specific tests). ``del`` is a delay element.
In a word-oriented memory ``0``/``1`` stand for the data background and its inverse.

Every test below is copied from the cited publication; the unit tests check each
one's length against the figure the publication gives (e.g. March SS is 22n).
"""
from __future__ import annotations

import re

from .base import ZERO, Background, Element, Pattern, Pause, Read, Write

_ORDER = {"⇕": "any", "↕": "any", "any": "any", "⇑": "up", "↑": "up", "up": "up",
          "⇓": "down", "↓": "down", "down": "down"}
_ELEMENT = re.compile(r"^\s*(⇕|↕|⇑|↑|⇓|↓|any|up|down)\s*\((.*)\)\s*$")
_OP = re.compile(r"^([rw])([01])(h?)(b?)$")


def parse_march(text: str, name: str, *, h: int = 5, delay_s: float = 64.0,
                bg: Background = ZERO, description: str = "") -> Pattern:
    body = text.strip()
    if not (body.startswith("{") and body.endswith("}")):
        raise ValueError("a march test is enclosed in { }")
    steps: list = []
    for part in body[1:-1].split(";"):
        part = part.strip()
        if part.lower() in ("del", "delay"):
            steps.append(Pause(delay_s))
            continue
        m = _ELEMENT.match(part)
        if not m:
            raise ValueError(f"cannot parse march element {part!r}")
        ops = []
        for tok in (t.strip() for t in m.group(2).split(",")):
            o = _OP.match(tok)
            if not o:
                raise ValueError(f"cannot parse operation {tok!r} in {part!r}")
            kind, value, hammer, bitline = o.groups()
            cls = Write if kind == "w" else Read
            ops.append(cls(bg, value == "1", repeat=h if hammer else 1, on="bl" if bitline else "self"))
        steps.append(Element(_ORDER[m.group(1)], tuple(ops)))
    return Pattern(name, tuple(steps), description=description)


# name -> (notation, length as published (terms in n and h*n), reference)
MARCH_TESTS: dict[str, tuple[str, tuple[int, int], str]] = {
    "mats-plus": ("{⇕(w0); ⇑(r0,w1); ⇓(r1,w0)}", (5, 0),
                  "R. Nair, IEEE Trans. Computers C-28(3), 1979"),
    "march-c-minus": ("{⇕(w0); ⇑(r0,w1); ⇑(r1,w0); ⇓(r0,w1); ⇓(r1,w0); ⇕(r0)}", (10, 0),
                      "M. Marinescu, ITC 1982; A.J. van de Goor, Testing Semiconductor Memories, 1998"),
    "march-b": ("{⇕(w0); ⇑(r0,w1,r1,w0,r0,w1); ⇑(r1,w0,w1); ⇓(r1,w0,w1,w0); ⇓(r0,w1,w0)}", (17, 0),
                "D.S. Suk and S.M. Reddy, IEEE Trans. Computers C-30(12), 1981"),
    "pmovi": ("{⇓(w0); ⇑(r0,w1,r1); ⇑(r1,w0,r0); ⇓(r0,w1,r1); ⇓(r1,w0,r0)}", (13, 0),
              "J.H. De Jonge and A.J. Smeulders, Computer Design, 1976"),
    "march-u": ("{⇕(w0); ⇑(r0,w1,r1,w0); ⇑(r0,w1); ⇓(r1,w0,r0,w1); ⇓(r1,w0)}", (13, 0),
                "A.J. van de Goor and G.N. Gaydadjiev, IEE Proc. Circuits Devices Syst. 144(3), 1997"),
    "march-sr": ("{⇓(w0); ⇑(r0,w1,r1,w0); ⇑(r0,r0); ⇑(w1); ⇓(r1,w0,r0,w1); ⇓(r1,r1)}", (14, 0),
                 "S. Hamdioui and A.J. van de Goor, ATS 2000"),
    "march-la": ("{⇕(w0); ⇑(r0,w1,w0,w1,r1); ⇑(r1,w0,w1,w0,r0); ⇓(r0,w1,w0,w1,r1); "
                 "⇓(r1,w0,w1,w0,r0); ⇓(r0)}", (22, 0),
                 "A.J. van de Goor et al., European Design and Test Conf. 1999"),
    "march-lr": ("{⇕(w0); ⇓(r0,w1); ⇑(r1,w0,r0,w1); ⇑(r1,w0); ⇑(r0,w1,r1,w0); ⇑(r0)}", (14, 0),
                 "A.J. van de Goor and G. Gaydadjiev, VTS 1996"),
    "march-ss": ("{⇕(w0); ⇑(r0,r0,w0,r0,w1); ⇑(r1,r1,w1,r1,w0); ⇓(r0,r0,w0,r0,w1); "
                 "⇓(r1,r1,w1,r1,w0); ⇕(r0)}", (22, 0),
                 "S. Hamdioui, A.J. van de Goor, M. Rodgers, MTDT 2002"),
    "march-raw1": ("{⇕(w0); ⇕(w0,r0); ⇕(r0); ⇕(w1,r1); ⇕(r1); ⇕(w1,r1); ⇕(r1); ⇕(w0,r0); ⇕(r0)}", (13, 0),
                   "S. Hamdioui, Z. Al-Ars, A.J. van de Goor, VTS 2002, Fig. 3"),
    "march-raw": ("{⇕(w0); ⇑(r0,w0,r0,r0,w1,r1); ⇑(r1,w1,r1,r1,w0,r0); ⇓(r0,w0,r0,r0,w1,r1); "
                  "⇓(r1,w1,r1,r1,w0,r0); ⇕(r0)}", (26, 0),
                  "S. Hamdioui, Z. Al-Ars, A.J. van de Goor, VTS 2002, Fig. 4"),
    "march-h1c": ("{⇕(w0h,r0,w1b,r0); ⇕(w1h,r1,w0b,r1); ⇕(w0h,w1,w0b,r1); ⇕(w1h,w0,w1b,r0)}", (12, 4),
                  "Z. Al-Ars, A.J. van de Goor, S. Hamdioui, DATE 2006"),
    "march-h2c": ("{⇕(w0h); ⇑(r0h,w1h); ⇑(r1h,w0h); ⇓(r0h,w1h); ⇓(r1h,w0h); ⇕(r0)}", (1, 9),
                  "Z. Al-Ars, A.J. van de Goor, S. Hamdioui, DATE 2006"),
    "march-t1c": ("{⇕(w0h,w1b,r0); ⇕(w1h,w0b,r1); ⇕(w0h,w1,w0b,r1); ⇕(w1h,w0,w1b,r0); "
                  "⇕(w0h,r0,w1b,r0); ⇕(w1h,r1,w0b,r1)}", (16, 6),
                  "Z. Al-Ars, S. Hamdioui, G. Gaydadjiev, IDT 2006"),
}

# Lengths are (a, b) for a*n + b*h*n, e.g. March H2C's published n + 9hn is (1, 9).

# Proposed by hbmlens, not published: with the literal fault primitive semantics a write-destructive
# cell flips on every write of the value it holds, so the hammered writes of H1C / T1C leave a plain
# (non-partial) WDF faulty or not depending on the hammer count and the value the cell started from.
# One more same-value write, a bit-line completing write and a read after the first detection catch
# the other parity. hbmlens.fpsim proves both complete for every realistic single-cell fault.
PROPOSED_TESTS: dict[str, tuple[str, tuple[int, int], str]] = {
    "march-h1c-plus": ("{⇕(w0h,r0,w1b,r0,w0,w1b,r0); ⇕(w1h,r1,w0b,r1,w1,w0b,r1); ⇕(w0h,w1,w0b,r1); "
                       "⇕(w1h,w0,w1b,r0)}", (18, 4), "hbmlens: March H1C + (w0,w1b,r0) in ME0, (w1,w0b,r1) in ME1"),
    "march-t1c-plus": ("{⇕(w0h,w1b,r0,w0,w1b,r0); ⇕(w1h,w0b,r1,w1,w0b,r1); ⇕(w0h,w1,w0b,r1); "
                       "⇕(w1h,w0,w1b,r0); ⇕(w0h,r0,w1b,r0); ⇕(w1h,r1,w0b,r1)}", (22, 6),
                       "hbmlens: March T1C + (w0,w1b,r0) in ME0, (w1,w0b,r1) in ME1"),
}
ALL_TESTS = {**MARCH_TESTS, **PROPOSED_TESTS}


def march_test(name: str, h: int = 5) -> Pattern:
    notation, _, ref = ALL_TESTS[name]
    return parse_march(notation, name, h=h, description=f"{notation} ({ref})")


def published_length(name: str, h: int) -> int:
    a, b = ALL_TESTS[name][1]
    return a + b * h


__all__ = ["parse_march", "MARCH_TESTS", "PROPOSED_TESTS", "ALL_TESTS", "march_test", "published_length"]

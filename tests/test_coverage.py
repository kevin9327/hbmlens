"""The coverage simulator must reproduce textbook results before we trust its tables."""
import numpy as np
import pytest

from hbmlens.coverage import FAULT_MODELS, FaultSpec, detected, measure
from hbmlens.patterns.library import CUDA_MEMTEST_STYLE, PATTERNS


@pytest.fixture(scope="module")
def res():
    return measure({n: PATTERNS[n]() for n in PATTERNS}, trials=150, seed=1)


def rate(res, pattern, model):
    return res.rate[res.patterns.index(pattern), res.models.index(model)]


def test_mats_plus_detects_all_stuck_at_and_address_faults(res):
    assert rate(res, "mats-plus", "SAF") == 1.0
    assert rate(res, "mats-plus", "AF") == 1.0
    assert rate(res, "mats-plus", "TF") < 1.0  # MATS+ misses half of the transition faults


@pytest.mark.parametrize("model", ["SAF", "TF", "CFin", "CFid", "CFst", "AF"])
def test_march_c_minus_detects_all_unlinked_bit_faults(res, model):
    assert rate(res, "march-c-minus", model) == 1.0


def test_word_oriented_backgrounds_needed_for_intra_word_coupling(res):
    assert rate(res, "march-c-minus", "CFid-intra") < 1.0
    assert rate(res, "march-c-minus-wom", "CFid-intra") == 1.0


def test_only_patterns_with_pauses_detect_retention(res):
    for name in res.patterns:
        expected = 1.0 if name in ("retention", "bit-fade") else 0.0
        assert rate(res, name, "DRF") == expected, name


def test_hbmlens_suite_complete_at_a_quarter_of_the_cost(res):
    full = res.suite_rate(["march-c-minus-wom", "retention"])
    assert np.all(full == 1.0)
    cost_full = res.ops_per_word["march-c-minus-wom"] + res.ops_per_word["retention"]
    cost_cm = sum(res.ops_per_word[n] for n in CUDA_MEMTEST_STYLE)
    assert cost_full * 4 < cost_cm


def test_moving_inversions_suite_misses_this_idempotent_coupling():
    # rising aggressor at a lower address forces the same bit of a higher word to 0:
    # solid-data moving inversions never expose it, March C-'s down(r0,w1) element does
    spec = FaultSpec("CFid", (3, 588), {"abit": 13, "vbit": 13, "rising": True, "value": 0})
    assert not any(detected(PATTERNS[n](), spec) for n in CUDA_MEMTEST_STYLE)
    assert detected(PATTERNS["march-c-minus"](), spec)


def test_coupling_direction_matters():
    # inversion coupling with aggressor above the victim, rising transition: a pure 'up'
    # march element that writes 0->1 reaches the aggressor after the victim was read
    from hbmlens.patterns.base import ONES, ZERO, Element, Pattern, Read, Write

    spec = FaultSpec("CFin", (10, 3), {"abit": 0, "vbit": 0, "rising": True})
    up_only = Pattern("up", (Element("up", (Write(ZERO),)), Element("up", (Read(ZERO), Write(ONES))),
                             Element("up", (Read(ONES),))))
    down_only = Pattern("down", (Element("up", (Write(ZERO),)), Element("down", (Read(ZERO), Write(ONES))),
                                 Element("up", (Read(ONES),))))
    assert detected(down_only, spec)
    assert detected(up_only, spec)  # the final read catches the inverted victim
    spec2 = FaultSpec("CFin", (10, 3), {"abit": 0, "vbit": 0, "rising": True})
    no_final = Pattern("nf", (Element("up", (Write(ZERO),)), Element("up", (Read(ZERO), Write(ONES)))))
    assert not detected(no_final, spec2)


def test_every_model_generates_valid_specs():
    rng = np.random.default_rng(0)
    for name, gen in FAULT_MODELS.items():
        spec = gen(rng)
        assert spec.words and all(0 <= w < 1024 for w in spec.words), name

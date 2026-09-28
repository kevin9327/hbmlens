"""The coverage simulator must reproduce textbook results before we trust its tables."""
import numpy as np
import pytest

from hbmlens.coverage import FAULT_MODELS, FaultSpec, _Program, _run, cheapest_suite, detected, measure
from hbmlens.patterns.base import ZERO, Element, Pattern, Read, Write
from hbmlens.patterns.library import CUDA_MEMTEST_STYLE, HBMLENS_SUITE, PATTERNS


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


@pytest.mark.parametrize("model", ["SAF", "TF", "WDF", "RDF", "DRDF", "IRF", "CFin", "CFid", "CFst",
                                   "CFds", "CFwd", "CFdrd", "AF"])
def test_march_ss_detects_every_static_bit_fault(res, model):
    assert rate(res, "march-ss", model) == 1.0


@pytest.mark.parametrize("model", ["WDF", "DRDF", "CFwd", "CFdrd"])
def test_march_c_minus_misses_destructive_reads_and_writes(res, model):
    assert rate(res, "march-c-minus", model) == 0.0


def test_intra_word_test_covers_what_solid_backgrounds_cannot(res):
    assert rate(res, "march-ss", "CFid-intra") < 1.0
    assert res.suite_rate(["march-ss", "intra-word"])[res.models.index("CFid-intra")] == 1.0


def test_hbmlens_suite_complete_at_under_a_fifth_of_the_cost(res):
    assert np.all(res.suite_rate(HBMLENS_SUITE) == 1.0)
    cost_cm = sum(res.ops_per_word[n] for n in CUDA_MEMTEST_STYLE)
    assert res.suite_cost(HBMLENS_SUITE)[0] * 5 < cost_cm


def test_cuda_memtest_style_never_sees_deceptive_reads(res):
    for model in ("DRDF", "CFdrd"):
        assert res.suite_rate(CUDA_MEMTEST_STYLE)[res.models.index(model)] == 0.0
    assert res.suite_rate(CUDA_MEMTEST_STYLE)[res.models.index("WDF")] < 1.0


def test_suites_run_back_to_back():
    # run as one program, carried-over data may add detections, but never deceptive reads
    from hbmlens.patterns.base import concat

    pats = {n: PATTERNS[n]() for n in set(CUDA_MEMTEST_STYLE) | set(HBMLENS_SUITE)}
    seq = measure({"cm": concat("cm", [pats[n] for n in CUDA_MEMTEST_STYLE]),
                   "hb": concat("hb", [pats[n] for n in HBMLENS_SUITE])}, trials=100, seed=2)
    cm, hb = seq.rate
    for model in ("DRDF", "CFdrd"):
        assert cm[seq.models.index(model)] == 0.0
    assert np.all(hb == 1.0)


def test_cheapest_suite_is_complete_and_no_dearer_than_hbmlens(res):
    best = cheapest_suite(res)
    assert best is not None and np.all(res.suite_rate(best) == 1.0)
    assert res.suite_cost(best) <= res.suite_cost(HBMLENS_SUITE)


def test_detection_must_hold_for_every_initial_state():
    # a same-value write only destroys the cell if it already held that value
    spec = FaultSpec("WDF", (7,), {"bit": 5, "state": 0})
    lucky = Pattern("lucky", (Element("any", (Write(ZERO),)), Element("any", (Read(ZERO),))))
    assert _run(_Program.of(lucky), spec, {}, "up")  # all-zero memory: caught
    assert not _run(_Program.of(lucky), spec, {7: 1 << 5}, "up")  # bit was 1: plain transition
    assert not detected(lucky, spec)


def test_deceptive_read_needs_two_reads():
    spec = FaultSpec("DRDF", (7,), {"bit": 5, "state": 0})
    once = Pattern("once", (Element("any", (Write(ZERO),)), Element("any", (Read(ZERO), Write(ZERO)))))
    twice = Pattern("twice", (Element("any", (Write(ZERO),)), Element("any", (Read(ZERO), Read(ZERO)))))
    assert not detected(once, spec)
    assert detected(twice, spec)


def test_moving_inversions_suite_misses_this_idempotent_coupling():
    # rising aggressor at a lower address forces the same bit of a higher word to 0:
    # solid-data moving inversions never expose it, March C-'s down(r0,w1) element does
    spec = FaultSpec("CFid", (3, 588), {"abit": 13, "vbit": 13, "rising": True, "value": 0})
    assert not any(detected(PATTERNS[n](), spec) for n in CUDA_MEMTEST_STYLE)
    assert detected(PATTERNS["march-c-minus"](), spec)


def test_coupling_direction_matters():
    # inversion coupling with aggressor above the victim, rising transition: a pure 'up'
    # march element that writes 0->1 reaches the aggressor after the victim was read
    from hbmlens.patterns.base import ONES

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

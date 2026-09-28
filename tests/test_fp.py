"""The fault primitive engine must reproduce the published results before its tables are trusted.

Sources (see hbmlens/fp.py and hbmlens/patterns/march.py):
VTS 2002 = Hamdioui, Al-Ars, van de Goor, "Testing Static and Dynamic Faults in RAMs"
DATE 2006 = Al-Ars, van de Goor, Hamdioui, "Space of DRAM Fault Models and Corresponding Testing"
IDT 2006 = Al-Ars, Hamdioui, Gaydadjiev, "Using Linear Tests for Transient Faults in DRAMs"
"""
import pytest

from hbmlens.fp import (DYNAMIC_SINGLE, DYNAMIC_TWO, STATIC_SINGLE, STATIC_TWO, parse,
                        realistic_single_cell, realistic_two_cell)
from hbmlens.fpreport import table4_check, table6_check
from hbmlens.fpsim import Array, _Sim, evaluate, op_stream, order_choices
from hbmlens.patterns.march import MARCH_TESTS, PROPOSED_TESTS, march_test, parse_march, published_length


def missed(results):
    return {(r.fp.name, r.position) for r in results if not r.detected}


def test_catalogue_sizes():
    assert (len(STATIC_SINGLE), len(STATIC_TWO), len(DYNAMIC_SINGLE), len(DYNAMIC_TWO)) == (12, 36, 12, 32)
    # DATE 2006 Expr. 3 and 4: {-, pi, d, pid} single-cell (state faults not partial), {-, p} two-cell
    assert len(realistic_single_cell()) == 44
    assert len(realistic_two_cell()) == 92


def test_notation_parser():
    fp = parse("<0; 1w1/0/->", "CFwd")
    assert (fp.cells, fp.a_state, fp.a_ops, fp.v_state, fp.F, fp.R) == (2, 0, (), 1, 0, None)
    fp = parse("<1w0r0/↑/1>", "dRDF")
    assert [str(o) for o in fp.v_ops] == ["w0", "r0"] and fp.F == 1 and fp.R == 1
    with pytest.raises(ValueError):
        parse("<w1/0/->", "TF")  # no initial state
    with pytest.raises(ValueError):
        STATIC_SINGLE[0].with_attributes(partial="pi")  # state faults may not be partial


@pytest.mark.parametrize("fp", STATIC_SINGLE + STATIC_TWO + DYNAMIC_SINGLE + DYNAMIC_TWO, ids=lambda f: f.name)
def test_every_fp_does_what_its_notation_says(fp):
    v, a = 1, (0 if fp.cells == 2 else None)
    init = {v: fp.v_state, **({a: fp.a_state} if a is not None else {})}
    sim = _Sim(fp, v, a, (), init, 1)
    sim.gold.update(init)
    t, out = 10, None
    for cell, ops in ((a, fp.a_ops), (v, fp.v_ops)):
        for op in ops:
            t += 1
            out = sim.step(t, cell, op.kind == "w", op.value)
    assert sim.val[v] == fp.F
    if fp.R is not None:
        assert out == fp.R


@pytest.mark.parametrize("name", list(MARCH_TESTS) + list(PROPOSED_TESTS))
def test_test_lengths_match_publications(name):
    for h in (4, 5):
        assert march_test(name, h).ops_per_word() == published_length(name, h)


@pytest.mark.parametrize("name", list(MARCH_TESTS) + list(PROPOSED_TESTS))
def test_fault_free_memory_never_fails(name):
    # every read expects the value the test last wrote there (b operations included)
    pattern = march_test(name)
    for orders in order_choices(pattern):
        memory: dict = {}
        for _, cell, is_write, x, _, _ in op_stream(pattern, Array(), orders):
            if is_write:
                memory[cell] = x
            elif cell >= 0:
                assert memory.get(cell, x) == x, (name, orders)


def test_vts2002_table4_dynamic_coverage():
    rows = table4_check()
    assert len(rows) == 64
    assert [r for r in rows if r[2] != r[3]] == []


def test_march_ss_detects_all_static_simple_faults():  # MTDT 2002
    assert not missed(evaluate(march_test("march-ss"), STATIC_SINGLE + STATIC_TWO))


def test_march_raw1_and_raw_detect_all_dynamic_faults():  # VTS 2002, Sections 6.1 and 6.2
    assert not missed(evaluate(march_test("march-raw1"), DYNAMIC_SINGLE))
    assert not missed(evaluate(march_test("march-raw"), DYNAMIC_SINGLE + DYNAMIC_TWO))


def test_date2006_table6_first_detecting_operation():
    rows = table6_check()
    assert len(rows) == 12
    assert [r for r in rows if r[2] != {r[1]}] == []


# Findings: repeated writes of the value a write-destructive cell already holds toggle it
# (<0w0/1/-> flips on every non-transition write), so a hammered write leaves it faulty or not
# depending on the count and the value it started from.

WDF_FORMS = {("WDF<0w0/1/->", ""), ("WDF<1w1/0/->", ""), ("d WDF<0w0/1/->", ""), ("d WDF<1w1/0/->", "")}


def test_h1c_detects_every_single_cell_hard_fault_but_plain_write_destructive_ones():
    for h in (4, 5):
        assert missed(evaluate(march_test("march-h1c", h), realistic_single_cell("h"), h=h - 1)) == WDF_FORMS


def test_t1c_detects_every_single_cell_transient_fault_but_plain_write_destructive_ones():
    expected = {("t WDF<0w0/1/->", ""), ("t WDF<1w1/0/->", ""), ("dt WDF<0w0/1/->", ""), ("dt WDF<1w1/0/->", "")}
    assert missed(evaluate(march_test("march-t1c"), realistic_single_cell("t"))) == expected


@pytest.mark.parametrize("name,faults", [("march-h1c-plus", "h"), ("march-t1c-plus", "t")])
def test_proposed_fixes_detect_all_realistic_single_cell_faults(name, faults):
    for h in (4, 5, 6):
        assert not missed(evaluate(march_test(name, h), realistic_single_cell(faults), h=h - 1))


def test_h2c_needs_an_even_hammer_count_for_write_destructive_coupling():
    cfwd = [fp for fp in realistic_two_cell() if fp.ffm == "CFwd"]
    assert len(missed(evaluate(march_test("march-h2c", 5), cfwd, h=4))) == 16
    assert not missed(evaluate(march_test("march-h2c", 4), cfwd, h=3))
    assert not missed(evaluate(march_test("march-h2c", 4), realistic_two_cell(), h=3))


def test_strict_timing_exposes_an_element_boundary_in_march_raw():
    # when the init element runs downward, its last write and M1's first read hit the lowest cell
    # back to back; the dynamic fault fires early and turns M1's 0w0r0 into 1w0r0
    fp = next(f for f in DYNAMIC_TWO if f.ffm == "dCFdrd" and f.notation.startswith("<0; 0w0r0"))
    assert not missed(evaluate(march_test("march-raw"), [fp]))
    assert missed(evaluate(march_test("march-raw"), [fp], immediate="time")) == {(fp.name, "v<a")}
    up_init = parse_march(MARCH_TESTS["march-raw"][0].replace("⇕(w0)", "⇑(w0)", 1), "raw-up")
    assert not missed(evaluate(up_init, [fp], immediate="time"))

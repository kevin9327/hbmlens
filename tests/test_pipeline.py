import base64

import numpy as np
import pytest
from click.testing import CliRunner

from hbmlens.analyze.core import classify, compare_with_truth, fail_cells, signature_panels
from hbmlens.backends.cuda import available as cuda_available
from hbmlens.backends.virtual import VirtualBackend
from hbmlens.cli import main
from hbmlens.faults import Fault, FaultFactory, FaultSet, demo_faults
from hbmlens.mapping import get_mapper
from hbmlens.patterns.library import PATTERNS, get_pattern
from hbmlens.viz import viewer_data


def _run(geo, mapper_name, faults, pattern="march-c-minus", temperature=25.0):
    return VirtualBackend(geo, faults, temperature).run(get_pattern(pattern))


@pytest.mark.parametrize("seed", range(12))
def test_demo_faults_are_recovered(seed):
    m = get_mapper("interleaved" if seed % 2 else "linear", "tiny")
    faults = demo_faults(m, seed)
    log = _run("tiny", m, faults)
    rec = compare_with_truth(classify(fail_cells(log, m), m.geometry), faults)
    assert not rec.missed and not rec.extra, (rec.missed, rec.extra)


def test_every_static_fault_word_fails_march_c_minus():
    m = get_mapper("linear", "tiny")
    faults = demo_faults(m, 3)
    log = _run("tiny", m, faults)
    assert np.array_equal(log.failing_indices(), faults.to_overlay().indices)


def test_same_bit_column_and_cell_are_not_a_dq_lane():
    # regression: a column plus a same-bit cell in another bank of the pseudo channel
    m = get_mapper("linear", "tiny")
    fac = FaultFactory(m, 0)
    col = fac.column()
    loc = {k: col.location[k] for k in ("stack", "channel", "pseudo_channel")}
    other_bank = 1 - col.location["bank"]
    fields = {**{k: np.array([v]) for k, v in col.location.items() if k not in ("column",)},
              "bank": np.array([other_bank]), "row": np.array([5]), "column": np.array([2]), "word": np.array([0])}
    idx = m.encode(fields).astype(np.uint64)
    cell = Fault("cell", idx, col.bit, col.stuck_value, {"index": int(idx[0])})
    faults = FaultSet([col, cell])
    sigs = classify(fail_cells(_run("tiny", m, faults), m), m.geometry)
    assert sorted(s.kind for s in sigs) == ["cell", "column"], [s.label() for s in sigs]
    assert all(s.location != loc for s in sigs)


def test_different_bit_cell_inside_column_word_stays_separate():
    m = get_mapper("linear", "tiny")
    fac = FaultFactory(m, 1)
    col = fac.column()
    word = col.indices[:1]
    cell = Fault("cell", word, (col.bit + 7) % 32, 1 - col.stuck_value, {"index": int(word[0])})
    faults = FaultSet([col, cell])
    rec = compare_with_truth(classify(fail_cells(_run("tiny", m, faults), m), m.geometry), faults)
    assert not rec.missed and not rec.extra


def test_masked_fault_is_reported_separately():
    m = get_mapper("linear", "tiny")
    fac = FaultFactory(m, 2)
    lane = fac.dq_lane()
    cell = Fault("cell", lane.indices[10:11], lane.bit, lane.stuck_value, {"index": int(lane.indices[10])})
    rec = compare_with_truth(classify(fail_cells(_run("tiny", m, FaultSet([lane, cell])), m), m.geometry),
                             FaultSet([lane, cell]))
    assert len(rec.masked) == 1 and not rec.missed


def test_retention_depends_on_temperature():
    m = get_mapper("linear", "tiny")
    fac = FaultFactory(m, 4)
    faults = FaultSet([], fac.retention_cells(60))
    cold = _run("tiny", m, faults, "retention", 25.0).meta.total_fails
    hot = _run("tiny", m, faults, "retention", 85.0).meta.total_fails
    assert cold == 0 and hot > 0


def test_all_library_patterns_run():
    m = get_mapper("linear", "tiny")
    faults = demo_faults(m, 5)
    for name in PATTERNS:
        log = _run("tiny", m, faults, name)
        assert log.meta.total_fails > 0, name


def test_signature_panels_show_only_signature_bits():
    m = get_mapper("linear", "small")
    faults = demo_faults(m, 7)
    cells = fail_cells(_run("small", m, faults), m)
    sigs = classify(cells, m.geometry)
    panels = signature_panels(cells, sigs, m.geometry)
    col_panels = [bm for title, bm in panels if title.startswith("column")]
    assert col_panels and all(bm.sum() == m.geometry.rows * m.geometry.words_per_column for bm in col_panels)


def test_viewer_data_bitmaps_roundtrip():
    m = get_mapper("interleaved", "tiny")
    faults = demo_faults(m, 0)
    log = _run("tiny", m, faults)
    cells = fail_cells(log, m)
    from hbmlens.analyze.core import bank_bitmaps

    bms = bank_bitmaps(cells, m.geometry)
    data = viewer_data(log, m, cells, faults=faults, signatures=classify(cells, m.geometry), bitmaps=bms,
                       pattern=get_pattern("march-c-minus"))
    bk, bm = next(iter(bms.items()))
    raw = np.frombuffer(base64.b64decode(data["bitmaps"][str(bk)]["b64"]), dtype=np.uint8)
    back = np.unpackbits(raw, bitorder="little")[: bm.size].reshape(bm.shape).astype(bool)
    assert np.array_equal(back, bm)
    assert len(data["points"]) == data["shown_words"] and len(data["elements"]) == 6


def test_cli_demo(tmp_path):
    out = tmp_path / "demo"
    res = CliRunner().invoke(main, ["demo", "--geometry", "tiny", "--out", str(out)])
    assert res.exit_code == 0, res.output
    assert "recall 100%" in res.output
    for name in ("fails.parquet", "report.md", "bitmaps.png", "viewer.html", "signatures.json"):
        assert (out / name).exists(), name


@pytest.mark.cuda
@pytest.mark.skipif(not cuda_available(), reason="no CUDA device")
def test_cuda_matches_virtual():
    from hbmlens.backends.cuda import CudaBackend

    m = get_mapper("interleaved", "tiny")
    faults = demo_faults(m, 9)
    for access in ("vector", "ordered"):
        for name in ("march-c-minus", "walking-ones", "random", "moving-inversions", "checkerboard"):
            v = _run("tiny", m, faults, name)
            c = CudaBackend(m.geometry.total_words, "tiny", overlay=faults.to_overlay(),
                            access=access).run(get_pattern(name))
            assert c.meta.total_fails == v.meta.total_fails, (access, name)
            assert np.array_equal(c.failing_indices(), v.failing_indices()), (access, name)


@pytest.mark.cuda
@pytest.mark.skipif(not cuda_available(), reason="no CUDA device")
@pytest.mark.parametrize("region", [(0, 32768 - 3), (4, 32768 - 1), (3, 32000)])
def test_cuda_regions_with_tail_or_misalignment(region):
    from hbmlens.backends.cuda import CudaBackend

    m = get_mapper("linear", "tiny")
    faults = demo_faults(m, 12)
    pat = get_pattern("march-c-minus")
    v = VirtualBackend("tiny", faults).run(pat, region=region)
    c = CudaBackend(m.geometry.total_words, "tiny", overlay=faults.to_overlay()).run(pat, region=region)
    assert np.array_equal(c.failing_indices(), v.failing_indices())
    assert c.meta.notes["access"] == ("vector" if region[0] % 4 == 0 else "ordered")

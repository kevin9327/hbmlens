import numpy as np
import pytest

from hbmlens.backends.base import ReadOverlay
from hbmlens.geometry import LEVELS, PRESETS, get_geometry
from hbmlens.mapping import XorRule, LinearMapper, bank_key, get_mapper
from hbmlens.patterns.base import Background, background_values, hash32
from hbmlens.records import FailLog, RunMeta, empty_records


@pytest.mark.parametrize("mapper_name", ["linear", "interleaved"])
@pytest.mark.parametrize("geo", ["tiny", "small"])
def test_mapping_roundtrip_is_bijective(mapper_name, geo):
    m = get_mapper(mapper_name, geo)
    idx = np.arange(m.geometry.total_words, dtype=np.int64)
    fields = m.decode(idx)
    back = m.encode(fields)
    assert np.array_equal(back, idx)
    # every coordinate tuple appears exactly once
    key = np.zeros_like(idx)
    for level in LEVELS:
        key = key * m.geometry.level_sizes()[level] + fields[level]
    assert len(np.unique(key)) == len(idx)


def test_large_presets_decode_sample():
    for name in ("hbm3-16g", "hbm4-32g"):
        m = get_mapper("interleaved", name)
        rng = np.random.default_rng(0)
        idx = rng.integers(0, m.geometry.total_words, size=10_000, dtype=np.int64)
        assert np.array_equal(m.encode(m.decode(idx)), idx)


def test_linear_mapper_bank_is_contiguous():
    m = get_mapper("linear", "tiny")
    g = m.geometry
    per_bank = g.rows * g.words_per_row
    f = m.decode(np.arange(per_bank))
    assert len(np.unique(bank_key(f, g))) == 1
    assert f["row"].max() == g.rows - 1


def test_xor_rule_validation():
    g = get_geometry("tiny")
    LinearMapper(g, xor_rules=(XorRule("row", "column", 3),))  # 64 rows, 8 <= 64: valid
    with pytest.raises(ValueError):
        LinearMapper(g.scaled(rows=48), xor_rules=(XorRule("row", "column", 3),))  # not a power of two
    with pytest.raises(ValueError):
        LinearMapper(g, xor_rules=(XorRule("bank", "row", 3),))  # 2 banks < 2**3


def test_decode_rejects_out_of_range():
    m = get_mapper("linear", "tiny")
    with pytest.raises(IndexError):
        m.decode(np.array([m.geometry.total_words]))


def test_hash32_reference_values():
    # frozen reference values: CUDA backend must reproduce these bit-exactly
    idx = np.array([0, 1, 2, 0xFFFFFFFF, 0x100000000, 123456789012], dtype=np.uint64)
    got = hash32(idx, 0x1234).tolist()
    # cross-checked against a pure-Python lowbias32 implementation
    assert got == [0xD8614160, 0x06AEAF0B, 0x51C0D36E, 0x2D707013, 0xF376E5DB, 0xF36EAE7E]
    assert hash32(np.array([5], np.uint64), 1)[0] != hash32(np.array([5], np.uint64), 2)[0]


def test_backgrounds():
    idx = np.arange(6, dtype=np.uint64)
    assert background_values(Background("solid", 0xA5A5A5A5), idx).tolist() == [0xA5A5A5A5] * 6
    chk = background_values(Background("checker", 0x55555555), idx)
    assert chk.tolist() == [0x55555555, 0xAAAAAAAA] * 3
    assert background_values(Background("addr"), idx).tolist() == list(range(6))
    inv = background_values(Background("addr"), idx, invert=True)
    assert inv.tolist() == [(~i) & 0xFFFFFFFF for i in range(6)]


def test_read_overlay_forces_bits():
    ov = ReadOverlay(np.array([3, 10], np.uint64), np.array([0xFFFFFFFE, 0xFFFFFFFF], np.uint32),
                     np.array([0, 0x80000000], np.uint32))
    vals = np.full(12, 0xFFFFFFFF, np.uint32)
    vals[10] = 0
    out = ov.apply(np.arange(12, dtype=np.uint64), vals)
    assert out[3] == 0xFFFFFFFE and out[10] == 0x80000000
    assert (np.delete(out, [3, 10]) == np.delete(vals, [3, 10])).all()
    with pytest.raises(ValueError):
        ReadOverlay(np.array([5, 5], np.uint64), np.zeros(2, np.uint32), np.zeros(2, np.uint32))


def test_faillog_roundtrip(tmp_path):
    rec = empty_records(3)
    rec["index"] = [5, 9, 9]
    rec["expected"] = [0, 0xFFFFFFFF, 0]
    rec["actual"] = [1, 0xFFFFFFFE, 0]
    meta = RunMeta(run_id="r1", pattern="march-c-minus", backend="virtual", geometry="tiny",
                   words_tested=100, temperature_c=85.0, total_fails=3)
    log = FailLog(meta, rec)
    path = log.save(tmp_path / "r.parquet")
    back = FailLog.load(path)
    assert back.meta == log.meta
    assert np.array_equal(back.records, log.records)
    assert back.xor.tolist() == [1, 1, 0]
    df = back.to_frame()
    assert set(["xor", "run_id", "temperature_c"]).issubset(df.columns)


def test_presets_capacity():
    assert PRESETS["hbm3-16g"].total_bytes == 16 * 2**30
    assert PRESETS["hbm4-32g"].total_bytes == 32 * 2**30
    assert PRESETS["tiny"].total_words == 32 * 1024
    assert PRESETS["medium"].total_bytes == 4 * 2**30

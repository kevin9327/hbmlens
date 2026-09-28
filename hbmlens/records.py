"""Fail records produced by every backend.

Unlike tools that keep only the last few failing addresses, hbmlens keeps the
exact total fail count and a detailed record for every failure up to a
configurable capacity. When the capacity is exceeded, ``meta.overflow`` is set
and ``meta.total_fails`` still holds the exact count.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

#: One row per failing read. ``element``/``op`` point into ``Pattern.steps``.
FAIL_DTYPE = np.dtype(
    [
        ("index", "<u8"),  # flat word index
        ("expected", "<u4"),
        ("actual", "<u4"),
        ("element", "<u2"),
        ("op", "<u2"),
        ("iteration", "<u4"),
    ]
)


def empty_records(n: int = 0) -> np.ndarray:
    return np.zeros(n, dtype=FAIL_DTYPE)


@dataclass
class RunMeta:
    run_id: str
    pattern: str
    backend: str
    geometry: str
    words_tested: int
    temperature_c: float | None = None
    seed: int | None = None
    total_fails: int = 0  # exact number of failing reads, even past capacity
    recorded: int = 0  # number of detailed records kept
    capacity: int = 0
    overflow: bool = False
    elapsed_s: float | None = None
    bandwidth_gbps: float | None = None
    notes: dict = field(default_factory=dict)


@dataclass
class FailLog:
    meta: RunMeta
    records: np.ndarray = field(default_factory=empty_records)

    def __post_init__(self) -> None:
        if self.records.dtype != FAIL_DTYPE:
            self.records = np.asarray(self.records).astype(FAIL_DTYPE)
        self.meta.recorded = int(len(self.records))

    def __len__(self) -> int:
        return len(self.records)

    @property
    def xor(self) -> np.ndarray:
        return self.records["expected"] ^ self.records["actual"]

    def to_frame(self) -> pd.DataFrame:
        df = pd.DataFrame({name: self.records[name] for name in FAIL_DTYPE.names})
        df["xor"] = self.xor
        df["run_id"] = self.meta.run_id
        df["pattern"] = self.meta.pattern
        df["temperature_c"] = self.meta.temperature_c
        return df

    def failing_indices(self) -> np.ndarray:
        return np.unique(self.records["index"])

    # ---- persistence ---------------------------------------------------
    def save(self, path: "str | Path") -> Path:
        path = Path(path)
        table = pa.Table.from_pandas(
            pd.DataFrame({name: self.records[name] for name in FAIL_DTYPE.names}), preserve_index=False
        )
        table = table.replace_schema_metadata({b"hbmlens.meta": json.dumps(asdict(self.meta)).encode()})
        pq.write_table(table, path)
        return path

    @classmethod
    def load(cls, path: "str | Path") -> "FailLog":
        table = pq.read_table(path)
        meta = RunMeta(**json.loads(table.schema.metadata[b"hbmlens.meta"]))
        df = table.to_pandas()
        rec = empty_records(len(df))
        for name in FAIL_DTYPE.names:
            rec[name] = df[name].to_numpy()
        return cls(meta, rec)

    @staticmethod
    def concat_frames(logs: Iterable["FailLog"]) -> pd.DataFrame:
        frames = [log.to_frame() for log in logs]
        return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()

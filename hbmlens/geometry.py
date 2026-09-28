"""Organization of a (virtual) HBM device.

The hierarchy follows the levels used by public DRAM simulators for HBM3/HBM4
(e.g. Ramulator 2: Channel > PseudoChannel > Sid > BankGroup > Bank > Row > Column),
with an extra ``stack`` level for multiple HBM cubes and a ``word`` level that
splits one column access (burst) into 32-bit words.

Presets other than ``tiny``/``small`` are *illustrative*: their sizes are chosen
to be capacity-consistent with public descriptions of the standards, not copied
from any vendor datasheet.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, replace

#: Field order from the largest to the smallest unit. Mappers and analyzers use
#: these names as dictionary keys.
LEVELS: tuple[str, ...] = (
    "stack",
    "channel",
    "pseudo_channel",
    "sid",
    "bank_group",
    "bank",
    "row",
    "column",
    "word",
)

WORD_BITS = 32
WORD_BYTES = 4


@dataclass(frozen=True)
class HBMGeometry:
    """Sizes of every level. ``bank`` counts banks *per bank group*."""

    name: str
    stacks: int
    channels: int  # per stack
    pseudo_channels: int  # per channel
    sids: int  # stack-ID (3D-stacked ranks) per pseudo channel
    bank_groups: int  # per sid
    banks_per_group: int
    rows: int  # per bank
    columns: int  # column (burst) addresses per row
    words_per_column: int  # 32-bit words delivered by one column access

    def __post_init__(self) -> None:
        for level, size in self.level_sizes().items():
            if size < 1:
                raise ValueError(f"{level} size must be >= 1, got {size}")

    # ---- derived sizes -------------------------------------------------
    def level_sizes(self) -> dict[str, int]:
        return {
            "stack": self.stacks,
            "channel": self.channels,
            "pseudo_channel": self.pseudo_channels,
            "sid": self.sids,
            "bank_group": self.bank_groups,
            "bank": self.banks_per_group,
            "row": self.rows,
            "column": self.columns,
            "word": self.words_per_column,
        }

    @property
    def banks_total(self) -> int:
        """Number of independent banks in the whole device."""
        return (
            self.stacks
            * self.channels
            * self.pseudo_channels
            * self.sids
            * self.bank_groups
            * self.banks_per_group
        )

    @property
    def words_per_row(self) -> int:
        return self.columns * self.words_per_column

    @property
    def row_bytes(self) -> int:
        return self.words_per_row * WORD_BYTES

    @property
    def total_words(self) -> int:
        return self.banks_total * self.rows * self.words_per_row

    @property
    def total_bytes(self) -> int:
        return self.total_words * WORD_BYTES

    def scaled(self, **sizes: int) -> "HBMGeometry":
        """Return a copy with some level sizes replaced, e.g. ``g.scaled(rows=256)``."""
        return replace(self, **sizes)

    def as_dict(self) -> dict:
        return asdict(self)

    def describe(self) -> str:
        gib = self.total_bytes / 2**30
        size = f"{gib:.2f} GiB" if gib >= 1 else f"{self.total_bytes / 2**20:.2f} MiB"
        return (
            f"{self.name}: {self.stacks} stack x {self.channels} ch x {self.pseudo_channels} pc x "
            f"{self.sids} sid x {self.bank_groups} bg x {self.banks_per_group} bank, "
            f"{self.rows} rows x {self.row_bytes} B/row = {size}"
        )


PRESETS: dict[str, HBMGeometry] = {
    # Unit-test sized device: 32 Ki words (128 KiB).
    "tiny": HBMGeometry(
        name="tiny", stacks=1, channels=2, pseudo_channels=2, sids=1,
        bank_groups=2, banks_per_group=2, rows=64, columns=16, words_per_column=2,
    ),
    # Demo sized device: 8 Mi words (32 MiB); every level except sid is > 1 so all
    # failure signatures (row, column, bank, channel, stack, DQ lane) are visible.
    "small": HBMGeometry(
        name="small", stacks=2, channels=4, pseudo_channels=2, sids=1,
        bank_groups=4, banks_per_group=2, rows=256, columns=32, words_per_column=8,
    ),
    # GPU test sized device: the HBM3 organization below with 8192 rows per bank,
    # 1 Gi words (4 GiB). Much larger than any GPU L2, so every access reaches DRAM.
    "medium": HBMGeometry(
        name="medium", stacks=1, channels=16, pseudo_channels=2, sids=1,
        bank_groups=4, banks_per_group=4, rows=8192, columns=32, words_per_column=8,
    ),
    # Illustrative HBM3 cube: 16 channels x 2 pseudo channels, 16 banks per pseudo
    # channel, 1 KiB page, 32 B access -> 16 GiB.
    "hbm3-16g": HBMGeometry(
        name="hbm3-16g", stacks=1, channels=16, pseudo_channels=2, sids=1,
        bank_groups=4, banks_per_group=4, rows=32768, columns=32, words_per_column=8,
    ),
    # Illustrative HBM4 cube following the Ramulator 2 HBM4 level list
    # (Channel/PseudoChannel/Sid/BankGroup/Bank/Row/Column), 32 channels -> 32 GiB.
    "hbm4-32g": HBMGeometry(
        name="hbm4-32g", stacks=1, channels=32, pseudo_channels=2, sids=2,
        bank_groups=4, banks_per_group=4, rows=16384, columns=32, words_per_column=8,
    ),
}


def get_geometry(name_or_geometry: "str | HBMGeometry") -> HBMGeometry:
    if isinstance(name_or_geometry, HBMGeometry):
        return name_or_geometry
    try:
        return PRESETS[name_or_geometry]
    except KeyError:
        raise KeyError(f"unknown geometry {name_or_geometry!r}; presets: {sorted(PRESETS)}") from None

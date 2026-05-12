"""Utilities for the club's 2026 rating-system change.

The historical CSV already contains the important neutral result columns:
``position`` (finish place), ``tournament_type`` and ``bounties``.  These
helpers recalculate rating points from those neutral results so the model can
train on the new rating economy instead of learning the old top-heavy one.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import numpy as np
import pandas as pd


NEW_RATING_LADDER: tuple[tuple[int, int, int], ...] = (
    (1, 1, 2000),
    (2, 2, 1500),
    (3, 3, 1000),
    (4, 4, 850),
    (5, 5, 775),
    (6, 6, 700),
    (7, 7, 625),
    (8, 8, 550),
    (9, 9, 475),
    (10, 13, 350),
    (14, 16, 325),
    (17, 20, 300),
    (21, 23, 250),
    (24, 27, 200),
    (28, 30, 150),
)

POINT_BUBBLE_RANK = 30
BOUNTY_POINTS_PER_KO = 20


@dataclass(frozen=True)
class TournamentMetadata:
    """Stable structure/format metadata for one weekly tournament type."""

    day_name: str
    starting_stack: int
    rebuy_limit: float | int | None
    addon_at_freezeout: bool = True
    freezeout_addon_format: bool = True
    rating_multiplier: float = 1.0
    bounty_points_per_ko: int = 0

    @property
    def unlimited_rebuys(self) -> bool:
        return self.rebuy_limit is None or self.rebuy_limit == float("inf")


TOURNAMENT_ALIASES: Mapping[str, str] = {
    "tuesday stack": "high roller",
    "tuesday": "high roller",
    "high roller": "high roller",
    "double rating points": "double rating points",
    "wednesday": "double rating points",
    "phoenix": "phoenix",
    "deep classic": "deep stack",
    "deep stack": "deep stack",
    "friday": "deep stack",
    "bounty": "bounty",
    "bounty day": "bounty",
    "triple shot": "triple shot",
    "sunday": "triple shot",
}

TOURNAMENT_METADATA: Mapping[str, TournamentMetadata] = {
    "high roller": TournamentMetadata(
        day_name="Tuesday",
        starting_stack=15000,
        rebuy_limit=None,
    ),
    "double rating points": TournamentMetadata(
        day_name="Wednesday",
        starting_stack=25000,
        rebuy_limit=None,
        rating_multiplier=2.0,
    ),
    "phoenix": TournamentMetadata(
        day_name="Thursday",
        starting_stack=60000,
        rebuy_limit=1,
    ),
    "deep stack": TournamentMetadata(
        day_name="Friday",
        starting_stack=50000,
        rebuy_limit=None,
    ),
    "bounty": TournamentMetadata(
        day_name="Saturday",
        starting_stack=25000,
        rebuy_limit=None,
        bounty_points_per_ko=BOUNTY_POINTS_PER_KO,
    ),
    "triple shot": TournamentMetadata(
        day_name="Sunday",
        starting_stack=25000,
        rebuy_limit=3,
    ),
}


def normalize_tournament_type(tournament_type: object) -> str:
    """Return the canonical tournament key used by metadata and schedule code."""

    key = str(tournament_type).strip().lower()
    return TOURNAMENT_ALIASES.get(key, key)


def points_for_position(position: object, tournament_type: object) -> float:
    """Calculate placement rating points under the new top-30 ladder."""

    if pd.isna(position):
        return 0.0

    pos = int(position)
    base = 0
    for lo, hi, value in NEW_RATING_LADDER:
        if lo <= pos <= hi:
            base = value
            break

    canonical_type = normalize_tournament_type(tournament_type)
    metadata = TOURNAMENT_METADATA.get(canonical_type)
    multiplier = metadata.rating_multiplier if metadata else 1.0
    return float(base * multiplier)


def bounty_points(bounties: object, tournament_type: object) -> float:
    """Calculate bounty bonus points for formats that award rating KOs."""

    canonical_type = normalize_tournament_type(tournament_type)
    metadata = TOURNAMENT_METADATA.get(canonical_type)
    per_ko = metadata.bounty_points_per_ko if metadata else 0
    if per_ko <= 0 or pd.isna(bounties):
        return 0.0
    return float(max(0, int(bounties)) * per_ko)


def total_new_rating_points(position: object, tournament_type: object, bounties: object = 0) -> float:
    """Calculate total daily rating points under the new rules."""

    return points_for_position(position, tournament_type) + bounty_points(bounties, tournament_type)


def convert_legacy_points(df: pd.DataFrame, keep_old_points: bool = True) -> pd.DataFrame:
    """Return a copy with ``points`` recalculated from position/type/bounties.

    If ``keep_old_points`` is true, the old CSV's points are preserved in
    ``old_points`` for auditability.  The model and reports still read the
    standard ``points`` column, so callers can adopt the new format by passing
    raw data through this function once at load time.
    """

    required = {"position", "tournament_type"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"Cannot convert rating points; missing columns: {sorted(missing)}")

    out = df.copy()
    if keep_old_points and "points" in out.columns and "old_points" not in out.columns:
        out["old_points"] = out["points"]

    if "bounties" not in out.columns:
        out["bounties"] = 0

    out["tournament_type"] = out["tournament_type"].map(normalize_tournament_type)
    out["placement_points"] = [
        points_for_position(pos, t_type)
        for pos, t_type in zip(out["position"], out["tournament_type"])
    ]
    out["bounty_points"] = [
        bounty_points(kos, t_type)
        for kos, t_type in zip(out["bounties"], out["tournament_type"])
    ]
    out["points"] = out["placement_points"] + out["bounty_points"]
    out["gets_rating_points"] = out["points"] > 0
    out["points_bubble_distance"] = out["position"].astype(float) - POINT_BUBBLE_RANK
    out["made_top_30"] = out["position"].astype(float) <= POINT_BUBBLE_RANK
    return out


def add_tournament_metadata_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Add stack/rebuy/addon/multiplier columns usable by the ML features."""

    out = df.copy()
    canonical_types = out["tournament_type"].map(normalize_tournament_type)
    out["tournament_type"] = canonical_types

    def meta_value(t_type: str, attr: str, default: float = 0.0) -> float:
        metadata = TOURNAMENT_METADATA.get(t_type)
        if metadata is None:
            return default
        value = getattr(metadata, attr)
        if value is None or value == float("inf"):
            return default
        return float(value)

    out["starting_stack"] = [meta_value(t, "starting_stack") for t in canonical_types]
    out["rating_multiplier"] = [meta_value(t, "rating_multiplier", 1.0) for t in canonical_types]
    out["bounty_points_per_ko"] = [meta_value(t, "bounty_points_per_ko") for t in canonical_types]
    out["addon_at_freezeout"] = [int(TOURNAMENT_METADATA.get(t, TournamentMetadata("", 0, 0)).addon_at_freezeout) for t in canonical_types]
    out["freezeout_addon_format"] = [int(TOURNAMENT_METADATA.get(t, TournamentMetadata("", 0, 0)).freezeout_addon_format) for t in canonical_types]
    out["unlimited_rebuys"] = [int(TOURNAMENT_METADATA.get(t, TournamentMetadata("", 0, 0)).unlimited_rebuys) for t in canonical_types]
    out["rebuy_limit"] = [meta_value(t, "rebuy_limit", 99.0) for t in canonical_types]
    return out


def score_ranked_players(ranked_players: list[str], tournament_type: object) -> dict[str, float]:
    """Assign new placement points to players ordered by simulated finish."""

    return {
        player: points_for_position(rank, tournament_type)
        for rank, player in enumerate(ranked_players, start=1)
    }


def build_bounty_expectations(df: pd.DataFrame, prior_kos: float = 1.0, prior_events: float = 2.0) -> dict[str, float]:
    """Estimate expected KOs on bounty day for each player with light shrinkage."""

    if "bounties" not in df.columns:
        return {}
    work = df.copy()
    work["tournament_type"] = work["tournament_type"].map(normalize_tournament_type)
    bounty_rows = work[work["tournament_type"] == "bounty"]
    if bounty_rows.empty:
        return {}

    player_totals = bounty_rows.groupby("player_id")["bounties"].sum()
    player_games = bounty_rows.groupby("player_id").size()
    field_mean = float(bounty_rows["bounties"].mean()) if len(bounty_rows) else prior_kos

    expectations = {}
    for player in work["player_id"].unique():
        kos = float(player_totals.get(player, 0.0))
        games = float(player_games.get(player, 0.0))
        expectations[player] = (kos + prior_events * field_mean) / (games + prior_events)
    return expectations


def sample_bounty_bonus(player: str, tournament_type: object, bounty_expectations: Mapping[str, float] | None) -> float:
    """Sample bounty rating points for one player in a simulated event."""

    canonical_type = normalize_tournament_type(tournament_type)
    metadata = TOURNAMENT_METADATA.get(canonical_type)
    if metadata is None or metadata.bounty_points_per_ko <= 0:
        return 0.0
    expected_kos = float((bounty_expectations or {}).get(player, 0.0))
    sampled_kos = np.random.poisson(max(0.0, expected_kos))
    return float(sampled_kos * metadata.bounty_points_per_ko)


def convert_file(input_csv: str, output_csv: str, keep_old_points: bool = True) -> int:
    """Convert one CSV file on disk and return the number of rows written.

    This function is intentionally not wired to argparse here because
    ``conversion.py`` is imported by the application entrypoint. Keeping this
    module import-only prevents accidental CLI parsing when running
    ``python main.py``.
    """

    df = pd.read_csv(input_csv)
    converted = convert_legacy_points(df, keep_old_points=keep_old_points)
    converted.to_csv(output_csv, index=False)
    return len(converted)
#!/usr/bin/env python3
"""Pitch geometry helpers shared by the local xG+ pipeline."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime
from typing import Any


DEFAULT_PITCH_LENGTH = 105.0
DEFAULT_PITCH_WIDTH = 68.0

GOAL_WIDTH = 7.32
GOAL_HALF_WIDTH = GOAL_WIDTH / 2.0
PENALTY_AREA_DEPTH = 16.5
PENALTY_AREA_WIDTH = 40.32
GOAL_AREA_DEPTH = 5.5
GOAL_AREA_WIDTH = 18.32
PENALTY_SPOT_DISTANCE = 11.0
CENTER_CIRCLE_RADIUS = 9.15


@dataclass(frozen=True)
class PitchDimensions:
    length: float = DEFAULT_PITCH_LENGTH
    width: float = DEFAULT_PITCH_WIDTH
    pitch_id: str | None = None

    @property
    def x_min(self) -> float:
        return -self.length / 2.0

    @property
    def x_max(self) -> float:
        return self.length / 2.0

    @property
    def y_min(self) -> float:
        return -self.width / 2.0

    @property
    def y_max(self) -> float:
        return self.width / 2.0


def _parse_date(value: Any) -> date | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    try:
        return datetime.fromisoformat(text.replace("Z", "+00:00")).date()
    except ValueError:
        try:
            return date.fromisoformat(text[:10])
        except ValueError:
            return None


def active_pitch_record(metadata: dict[str, Any]) -> dict[str, Any] | None:
    """Return the stadium pitch record active on the match date."""
    stadium = metadata.get("stadium") or {}
    pitches = stadium.get("pitches") or []
    if not pitches:
        return None

    match_date = _parse_date(metadata.get("date"))
    if match_date is None:
        return pitches[-1]

    dated: list[tuple[date, dict[str, Any]]] = []
    for pitch in pitches:
        start = _parse_date(pitch.get("startDate")) or date.min
        end = _parse_date(pitch.get("endDate")) or date.max
        dated.append((start, pitch))
        if start <= match_date <= end:
            return pitch

    before_match = [(start, pitch) for start, pitch in dated if start <= match_date]
    if before_match:
        return max(before_match, key=lambda item: item[0])[1]
    return min(dated, key=lambda item: item[0])[1]


def active_pitch_dimensions(metadata: dict[str, Any] | None) -> PitchDimensions:
    pitch = active_pitch_record(metadata or {})
    if pitch is None:
        return PitchDimensions()
    return PitchDimensions(
        length=float(pitch.get("length") or DEFAULT_PITCH_LENGTH),
        width=float(pitch.get("width") or DEFAULT_PITCH_WIDTH),
        pitch_id=None if pitch.get("id") is None else str(pitch.get("id")),
    )

"""Thermal START triage: heat + motion + posture + context -> category.

| Category   | Rule                                                         | U   |
|------------|--------------------------------------------------------------|-----|
| Immediate  | warm, immobile > 10 s, lying; or in water without flotation  | 1.0 |
| Delayed    | warm, lying or sitting, moving                               | 0.7 |
| Minor      | warm, standing or waving                                     | 0.3 |
| Expectant  | near ambient temperature, immobile                           | 0.1 |

Thermal cannot prove death, hence "Expectant / unresponsive".
Proximity to a mine raises the category by one step.
Attributes may come from the vision/thermal nodes (temp_k, posture, moving,
flotation); motion is otherwise estimated from the track history.
"""
from __future__ import annotations

from typing import Optional

from ..domain import URGENCY, URGENCY_STEP_UP, Start


def classify(
    temp_k: Optional[float],
    posture: Optional[str],
    moving: Optional[bool],
    observed_s: float,
    in_water: bool,
    flotation: bool,
    near_mine: bool,
    ambient_k: float,
    warm_delta_k: float,
    immobile_s: float,
) -> tuple[Start, str]:
    warm = None if temp_k is None else (temp_k >= ambient_k + warm_delta_k)
    posture = (posture or "").lower()
    why = []
    if warm is False and moving is not True:
        cat = Start.EXPECTANT
        why.append(f"temp {temp_k:.1f}K near ambient, immobile")
    elif in_water and not flotation:
        cat = Start.IMMEDIATE
        why.append("in water without flotation")
    elif warm is None:
        cat = Start.UNKNOWN
        why.append("no thermal reading yet")
    elif posture in ("standing", "waving") or (posture == "" and moving):
        cat = Start.MINOR
        why.append(f"warm, {posture or 'moving'}")
    elif moving is False and observed_s >= immobile_s and posture in ("lying", ""):
        cat = Start.IMMEDIATE
        why.append(f"warm, immobile {observed_s:.0f}s, {posture or 'lying?'}")
    elif moving:
        cat = Start.DELAYED
        why.append(f"warm, {posture or 'down'}, moving")
    else:
        cat = Start.DELAYED
        why.append("warm, observing (immobility not yet established)")
    if near_mine:
        up = URGENCY_STEP_UP[cat]
        if up != cat:
            why.append(f"near mine: {cat.value}->{up.value}")
        cat = up
    return cat, "; ".join(why)


def urgency_of(cat: Start) -> float:
    return URGENCY[cat]

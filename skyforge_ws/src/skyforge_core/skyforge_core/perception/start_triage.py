"""FLIR START triage: heat + breathing/motion + posture + context -> category.

| Category              | Rule                                                              | U    |
|-----------------------|-------------------------------------------------------------------|------|
| Immediate             | in water without flotation (always, even if skin is cold);       | 1.0  |
|                       | or warm, immobile > 10 s, lying                                   |      |
| Delayed               | warm, lying or sitting, moving                                    | 0.7  |
| Minor                 | warm, standing or waving                                          | 0.3  |
| Expectant/unresponsive| skin cool (below the warm threshold), immobile                    | 0.1  |
| Likely deceased (DCD) | on land, skin within `deceased_delta_k` of ambient, immobile, AND | 0.0  |
|                       | no breathing micro-motion observed continuously >= deceased_obs_s |      |

Why these rules (and not a cooling rate): post-mortem cooling is ~1 K per hour,
unmeasurable in a few minutes of flight, and a living hypothermic person in
cold water also shows cold skin. So DCD needs positive close-range evidence
(no breathing over several seconds, seen at low altitude during the assessment
hover) and is never assigned in water. The drone never "declares" death: DCD
casualties get no kit (kits are kept for the living) and are reported to the
station for recovery and human confirmation.

Injured are always served before EXPECTANT, and DCD never consumes a kit.
Proximity to a mine raises the category by one step (not for DCD).
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
    no_breath_s: float = 0.0,
    deceased_delta_k: float = 2.0,
    deceased_obs_s: float = 8.0,
) -> tuple[Start, str]:
    warm = None if temp_k is None else (temp_k >= ambient_k + warm_delta_k)
    posture = (posture or "").lower()
    why = []
    if in_water and not flotation:
        cat = Start.IMMEDIATE
        why.append("in water without flotation" + (" (cold skin: possible hypothermia)" if warm is False else ""))
    elif warm is None:
        cat = Start.UNKNOWN
        why.append("no thermal reading yet")
    elif warm is False and moving is not True:
        delta = temp_k - ambient_k
        if delta < deceased_delta_k and no_breath_s >= deceased_obs_s:
            cat = Start.DECEASED
            why.append(f"skin {temp_k:.1f}K ~ ambient (+{delta:.1f}K), immobile, no breathing for {no_breath_s:.0f}s")
        else:
            cat = Start.EXPECTANT
            why.append(f"skin {temp_k:.1f}K below warm threshold, immobile" + (", breathing seen" if no_breath_s == 0 else ""))
    elif warm is False:
        cat = Start.DELAYED  # cold but moving: alive, needs help
        why.append(f"skin {temp_k:.1f}K cool but moving")
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
    if near_mine and cat != Start.DECEASED:
        up = URGENCY_STEP_UP[cat]
        if up != cat:
            why.append(f"near mine: {cat.value}->{up.value}")
        cat = up
    return cat, "; ".join(why)


def urgency_of(cat: Start) -> float:
    return URGENCY[cat]

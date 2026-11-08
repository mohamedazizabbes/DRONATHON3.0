"""Shared value types. Frozen in hour 1: change only with the whole team."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any, Optional


class Phase(str, Enum):
    INIT = "INIT"
    EXPLORE = "EXPLORE"
    DETECT = "DETECT"
    TRIAGE = "TRIAGE"
    NAVIGATE = "NAVIGATE"
    DELIVER = "DELIVER"
    RETURN_BASE = "RETURN_BASE"
    LANDED = "LANDED"
    MAYDAY = "MAYDAY"


# Labels from the jury's specification (section 7), shown on the dashboard.
PHASE_FR = {
    Phase.INIT: "INIT",
    Phase.EXPLORE: "EXPLORER",
    Phase.DETECT: "DÉTECTER",
    Phase.TRIAGE: "TRIAGE",
    Phase.NAVIGATE: "NAVIGUER",
    Phase.DELIVER: "LIVRER",
    Phase.RETURN_BASE: "RETOUR BASE",
    Phase.LANDED: "POSÉ",
    Phase.MAYDAY: "MAYDAY",
}


class Cls:
    DEBRIS = "debris"
    MINE = "mine"
    WEAPON = "weapon"
    INJURED = "injured"
    HOSTILE = "hostile_unit"


DANGER_CLASSES = {Cls.MINE, Cls.WEAPON, Cls.HOSTILE}
ALL_CLASSES = {Cls.DEBRIS, Cls.MINE, Cls.WEAPON, Cls.INJURED, Cls.HOSTILE}


class Start(str, Enum):
    """START triage categories (thermal + motion)."""

    IMMEDIATE = "IMM"
    DELAYED = "DEL"
    MINOR = "MIN"
    EXPECTANT = "EXP"
    DECEASED = "DCD"  # likely deceased (FLIR + no breathing at close range); never declared by the drone
    UNKNOWN = "UNK"


URGENCY = {
    Start.IMMEDIATE: 1.0,
    Start.DELAYED: 0.7,
    Start.MINOR: 0.3,
    Start.EXPECTANT: 0.1,
    Start.DECEASED: 0.0,
    # Unknown (no thermal yet) is treated as DELAYED: over-prioritise rather than ignore.
    Start.UNKNOWN: 0.7,
}

# One step up when close to a mine (design doc section 6).
URGENCY_STEP_UP = {
    Start.EXPECTANT: Start.MINOR,
    Start.MINOR: Start.DELAYED,
    Start.DELAYED: Start.IMMEDIATE,
    Start.IMMEDIATE: Start.IMMEDIATE,
    Start.UNKNOWN: Start.IMMEDIATE,
    Start.DECEASED: Start.DECEASED,
}


@dataclass
class Detection:
    """One 2D detection, the common output of the jury and team vision modules."""

    cls: str
    conf: float
    bbox: tuple[float, float, float, float]  # x1, y1, x2, y2 in pixels
    stamp: float = 0.0
    source: str = "team"  # "jury" | "team"
    attrs: dict[str, Any] = field(default_factory=dict)  # temp_k, contrast, ...

    @property
    def bottom_center(self) -> tuple[float, float]:
        x1, y1, x2, y2 = self.bbox
        return (0.5 * (x1 + x2), y2)

    @property
    def center(self) -> tuple[float, float]:
        x1, y1, x2, y2 = self.bbox
        return (0.5 * (x1 + x2), 0.5 * (y1 + y2))


@dataclass
class Zone:
    id: str
    cls: str
    center: tuple[float, float]
    radius: float  # hard exclusion radius
    vel: tuple[float, float] = (0.0, 0.0)
    soft_radius: float = 0.0  # hostile line-of-sight ring (soft cost)
    source: str = "perception"  # perception | station | params

    def to_dict(self) -> dict:
        d = asdict(self)
        d["center"] = [round(self.center[0], 2), round(self.center[1], 2)]
        return d


@dataclass
class Track:
    id: str
    cls: str
    pos: tuple[float, float]
    sigma: float
    vel: tuple[float, float] = (0.0, 0.0)
    conf: float = 0.0
    hits: int = 0
    confirmed: bool = False
    first_seen: float = 0.0
    last_seen: float = 0.0
    floating: bool = False
    category: Start = Start.UNKNOWN
    urgency: float = URGENCY[Start.UNKNOWN]
    attrs: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "cls": self.cls,
            "pos": [round(self.pos[0], 2), round(self.pos[1], 2)],
            "sigma": round(self.sigma, 2),
            "vel": [round(self.vel[0], 3), round(self.vel[1], 3)],
            "conf": round(self.conf, 3),
            "confirmed": self.confirmed,
            "floating": self.floating,
            "category": self.category.value if isinstance(self.category, Start) else self.category,
            "urgency": self.urgency,
            "first_seen": round(self.first_seen, 2),
            "last_seen": round(self.last_seen, 2),
        }


@dataclass
class Obstacle:
    id: str
    p: tuple[float, float]
    r: float
    v: tuple[float, float] = (0.0, 0.0)  # moving obstacles (boat)
    h: float = 100.0  # height; obstacles below flight altitude are ignored by A*


def track_from_dict(d: dict) -> Track:
    cat = d.get("category", "UNK")
    try:
        cat = Start(cat)
    except ValueError:
        cat = Start.UNKNOWN
    return Track(
        id=d["id"],
        cls=d["cls"],
        pos=tuple(d["pos"]),
        sigma=float(d.get("sigma", 1.0)),
        vel=tuple(d.get("vel", (0.0, 0.0))),
        conf=float(d.get("conf", 0.5)),
        confirmed=bool(d.get("confirmed", True)),
        floating=bool(d.get("floating", False)),
        category=cat,
        urgency=float(d.get("urgency", URGENCY[cat])),
        first_seen=float(d.get("first_seen", 0.0)),
        last_seen=float(d.get("last_seen", 0.0)),
        hits=int(d.get("hits", 3)),
    )


def zone_from_dict(d: dict) -> Zone:
    return Zone(
        id=d["id"],
        cls=d.get("cls", "manual"),
        center=tuple(d["center"]),
        radius=float(d["radius"]),
        vel=tuple(d.get("vel", (0.0, 0.0))),
        soft_radius=float(d.get("soft_radius", 0.0)),
        source=d.get("source", "station"),
    )


def opt_float(x: Optional[float], nd: int = 2) -> Optional[float]:
    return None if x is None else round(float(x), nd)

"""Headless 2D+altitude world for end-to-end tests of the whole brain.

Not a replacement for Gazebo: it exists so the orchestrator, agents, reflex
and comm stack can be run, measured and regression-tested on any laptop
(the "every node runs alone with a fake publisher" rule, end to end).
Ground truth is kept for every object so metrics are computed vs reality.
"""
from __future__ import annotations

import copy
import math
import random
from dataclasses import dataclass, field
from typing import Optional

from ..domain import Cls, Detection
from ..geometry import in_any_polygon
from ..perception.projection import Camera, world_to_pixel

DEFAULT_SCENARIO = {
    "name": "coastal_demo",
    "seed": 7,
    "duration_s": 900.0,
    # parameter overrides that belong to this scenario (jury-defined world)
    "params": {
        "comm": {"shadow_polygons": [[[120, 55], [170, 55], [170, 95], [120, 95]]]},
        "energy": {"capacity_wh": 60.0},
        "mission": {"kits": 6},
    },
    "wind": [2.0, 1.0],
    "visibility": 1.0,
    "current": [0.20, 0.02],  # truth; the drone's forecast (drift.current) is 0.15/0.05: slightly wrong
    # tide change: the current rotates and strengthens mid-mission (estimated online by the drone)
    "current_schedule": [{"t": 200, "current": [0.32, -0.06]}],  # alongshore current strengthens and turns
    "waves": {"amp": 0.6, "period": 7.0, "dir": [0.0, 1.0]},
    "k_leeway": 0.03,
    "objects": [
        # casualties (truth START category in "cat")
        {"id": "T-IMM-land", "cls": "injured", "p": [60, 40], "size": 1.8, "temp_k": 309.5, "posture": "lying", "moving": False, "cat": "IMM"},
        {"id": "T-DEL", "cls": "injured", "p": [150, 70], "size": 1.2, "temp_k": 308.5, "posture": "sitting", "moving": True, "cat": "DEL"},
        {"id": "T-MIN", "cls": "injured", "p": [195, -5], "size": 0.8, "temp_k": 309.0, "posture": "standing", "moving": True, "cat": "MIN"},
        {"id": "T-IMM-water", "cls": "injured", "p": [110, -45], "size": 1.8, "temp_k": 307.0, "posture": "lying", "moving": False, "cat": "IMM", "floating": True},
        {"id": "T-EXP", "cls": "injured", "p": [205, 35], "size": 1.8, "temp_k": 299.5, "posture": "lying", "moving": False, "cat": "EXP", "breathing": True},
        {"id": "T-DCD", "cls": "injured", "p": [25, 55], "size": 1.8, "temp_k": 295.4, "posture": "lying", "moving": False, "cat": "DCD", "breathing": False},
        # threats
        {"id": "T-mine-beach", "cls": "mine", "p": [92, 18], "size": 0.6},
        {"id": "T-mine-float", "cls": "mine", "p": [150, -40], "size": 0.8, "floating": True},
        {"id": "T-weapon", "cls": "weapon", "p": [40, 75], "size": 1.0},
        # hidden mines: only the magnetometer (shallow / buried) or the dipping sonar (deep) can find them
        {"id": "T-mine-shallow", "cls": "mine", "p": [60, -55], "size": 0.8, "depth": 3.0},
        {"id": "T-mine-deep", "cls": "mine", "p": [175, -60], "size": 0.8, "depth": 12.0},
        {"id": "T-mine-buried", "cls": "mine", "p": [140, 30], "size": 0.5, "depth": 0.3},
        {"id": "T-hostile", "cls": "hostile_unit", "p": [205, 82], "size": 2.0, "detection_radius": 20.0},
        # debris
        {"id": "T-debris1", "cls": "debris", "p": [70, -30], "size": 1.0, "floating": True},
        {"id": "T-debris2", "cls": "debris", "p": [125, 8], "size": 0.8},
    ],
    "obstacles": [
        {"id": "lighthouse", "p": [100, 62], "r": 5.0, "h": 40.0},
        {"id": "rock", "p": [165, 2], "r": 4.0, "h": 12.0},
        {"id": "pole", "p": [30, 10], "r": 1.0, "h": 30.0},
    ],
    "moving_obstacles": [
        {"id": "boat", "r": 4.0, "h": 9.0, "speed": 3.0, "loop": [[20, -55], [200, -55], [200, -62], [20, -62]]},
    ],
    "faults": [
        {"t": 70, "type": "jam", "dur": 20},
        {"t": 100, "type": "goal", "goal": {"type": "zone_add", "zone": {"id": "NOFLY1", "center": [120, -15], "radius": 8}}},
        {"t": 115, "type": "spoof", "offset": [40, 0], "ramp": 8, "dur": 25},
        {"t": 135, "type": "forged_goal", "goal": {"type": "rtb"}},
        {"t": 137, "type": "replay_goal"},
        {"t": 160, "type": "goal", "goal": {"type": "goto", "p": [70, 85]}},
        {"t": 185, "type": "gust", "wind": [5.0, 3.0], "dur": 30},
    ],
    "detector": {
        "p_detect": {"injured": 0.85, "mine": 0.8, "weapon": 0.8, "hostile_unit": 0.85, "debris": 0.75},
        "false_pos_rate": 0.03,
        "pixel_noise": 2.0,
        "max_alt": 45.0,
    },
}


@dataclass
class TruthObject:
    id: str
    cls: str
    p: list
    size: float = 1.0
    floating: bool = False
    attrs: dict = field(default_factory=dict)


class World:
    def __init__(self, params: dict, scenario: Optional[dict] = None, seed: Optional[int] = None):
        self.sc = copy.deepcopy(scenario or DEFAULT_SCENARIO)
        self.params = params
        self.rng = random.Random(seed if seed is not None else self.sc.get("seed", 0))
        self.t = 0.0
        self.base_wind = list(self.sc.get("wind", [0, 0]))
        self.wind = list(self.base_wind)
        self.visibility = float(self.sc.get("visibility", 1.0))
        self.current = list(self.sc.get("current", params["drift"]["current"]))
        self.current_schedule = sorted(self.sc.get("current_schedule", []), key=lambda c: c["t"])
        self.wave = self.sc.get("waves", {"amp": 0.0, "period": 7.0, "dir": [0.0, 1.0]})
        self.k_leeway = self.sc.get("k_leeway", params["drift"]["k_leeway"])
        self.water = params["water"]["polygons"]
        self.objects: list[TruthObject] = []
        for o in self.sc["objects"]:
            attrs = {k: v for k, v in o.items() if k not in ("id", "cls", "p", "size", "floating")}
            # submerged / buried objects (depth > 0) are anchored: moored or bottom mines do not drift
            fl = bool(o.get("floating", in_any_polygon(o["p"], self.water) and float(o.get("depth", 0.0)) <= 0.0))
            self.objects.append(TruthObject(o["id"], o["cls"], list(o["p"]), o.get("size", 1.0), fl, attrs))
        self.obstacles = [dict(o) for o in self.sc.get("obstacles", [])]
        self.movers = []
        for m in self.sc.get("moving_obstacles", []):
            self.movers.append({**m, "p": list(m["loop"][0]), "v": [0.0, 0.0], "seg": 0})
        self.gust_until = -1.0

    # ------------------------------------------------------------ dynamics
    def step(self, dt: float) -> None:
        self.t += dt
        if self.gust_until >= 0 and self.t > self.gust_until:
            self.wind = list(self.base_wind)
            self.gust_until = -1.0
        while self.current_schedule and self.current_schedule[0]["t"] <= self.t:
            self.current = list(self.current_schedule.pop(0)["current"])  # tide / current change
        drift = (self.current[0] + self.k_leeway * self.wind[0], self.current[1] + self.k_leeway * self.wind[1])
        # wave orbital motion: zero-mean oscillation on top of the drift (derivative of A*sin(wt+phi))
        A, T = float(self.wave.get("amp", 0.0)), float(self.wave.get("period", 7.0))
        wd = self.wave.get("dir", [0.0, 1.0])
        for o in self.objects:
            if o.floating:
                # p_mean drifts with the current; the wave orbital motion is a zero-mean
                # oscillation around it shared by everything floating nearby (kit included)
                pm = o.attrs.setdefault("p_mean", list(o.p))
                phi = o.attrs.setdefault("_wave_phase", self.rng.uniform(0, 2 * math.pi))
                pm[0] += drift[0] * dt
                pm[1] += drift[1] * dt
                osc = A * math.sin(2 * math.pi * self.t / T + phi) if A > 0 else 0.0
                o.p[0], o.p[1] = pm[0] + osc * wd[0], pm[1] + osc * wd[1]
                if not in_any_polygon(o.p, self.water):
                    o.floating = False  # washed ashore: stops drifting
        for m in self.movers:
            loop = m["loop"]
            tgt = loop[(m["seg"] + 1) % len(loop)]
            dx, dy = tgt[0] - m["p"][0], tgt[1] - m["p"][1]
            d = math.hypot(dx, dy)
            if d < m["speed"] * dt:
                m["seg"] = (m["seg"] + 1) % len(loop)
                continue
            m["v"] = [dx / d * m["speed"], dy / d * m["speed"]]
            m["p"][0] += m["v"][0] * dt
            m["p"][1] += m["v"][1] * dt

    def set_gust(self, wind, dur: float) -> None:
        self.wind = list(wind)
        self.gust_until = self.t + dur

    def all_obstacles(self):
        """(center, r, h, v) of static + moving obstacles (truth)."""
        out = [(tuple(o["p"]), o["r"], o.get("h", 100.0), (0.0, 0.0)) for o in self.obstacles]
        out += [(tuple(m["p"]), m["r"], m.get("h", 10.0), tuple(m["v"])) for m in self.movers]
        return out

    # -------------------------------------------------------------- camera
    def camera_frame(self, pos, rpy, cam: Camera) -> tuple[dict, dict]:
        """Simulated jury + team detector outputs, as JSON-like payloads.

        Jury payload uses French keys (classe / confiance / boite) and team
        payload the canonical format, to exercise the adapter both ways.
        """
        det = self.sc["detector"]
        jury, team = [], []
        alt = pos[2]
        if alt < 1.0 or alt > det["max_alt"]:
            return self._payloads(jury, team, cam)
        vis = self.visibility
        for o in self.objects:
            if float(o.attrs.get("depth", 0.0)) > 0.0:
                continue  # submerged / buried: invisible to the camera (layers 2-3)
            px = world_to_pixel((o.p[0], o.p[1], 0.0), pos, rpy, cam)
            if px is None or not (0 <= px[0] < cam.width and 0 <= px[1] < cam.height):
                continue
            p = det["p_detect"].get(o.cls, 0.7) * (0.5 + 0.5 * vis)
            if self.rng.random() > p:
                continue
            s_px = max(4.0, o.size * cam.fx / max(alt, 1.0))
            u = px[0] + self.rng.gauss(0, det["pixel_noise"])
            v = px[1] + self.rng.gauss(0, det["pixel_noise"])
            conf = self.rng.uniform(0.45, 0.95)
            if o.cls == Cls.INJURED:
                bbox = [u - s_px / 2, v - s_px / 2, u + s_px / 2, v + s_px / 2]  # centre = person
                attrs = {"temp_k": round(o.attrs.get("temp_k", 309.0) + self.rng.gauss(0, 0.4), 1),
                         "posture": o.attrs.get("posture"), "moving": bool(o.attrs.get("moving", False))}
                if o.attrs.get("marked"):  # delivered kit floating next to them, IR strobe visible in FLIR
                    attrs["marked"] = True
                if alt < 12.0:  # breathing micro-motion is only resolvable at close range
                    attrs["breathing"] = bool(o.attrs.get("breathing", True))
                team.append({"class": "injured", "confidence": round(conf, 3), "bbox": bbox, "attrs": attrs})
            elif o.cls == Cls.HOSTILE:
                bbox = [u - s_px / 2, v - s_px, u + s_px / 2, v]  # bottom centre = ground contact
                team.append({"class": "hostile_unit", "confidence": round(conf, 3), "bbox": bbox, "attrs": {}})
            else:
                bbox = [u - s_px / 2, v - s_px, u + s_px / 2, v]
                jury.append({"classe": {"mine": "mine", "weapon": "arme", "debris": "dechet"}[o.cls],
                             "confiance": round(conf, 3), "boite": bbox})
        # false positives: random class, random pixel, low confidence
        if self.rng.random() < det["false_pos_rate"]:
            cls = self.rng.choice(["mine", "arme", "dechet"])
            u, v = self.rng.uniform(20, cam.width - 20), self.rng.uniform(20, cam.height - 20)
            jury.append({"classe": cls, "confiance": round(self.rng.uniform(0.2, 0.5), 3), "boite": [u - 5, v - 10, u + 5, v]})
        if self.rng.random() < det["false_pos_rate"]:
            u, v = self.rng.uniform(20, cam.width - 20), self.rng.uniform(20, cam.height - 20)
            team.append({"class": "injured", "confidence": round(self.rng.uniform(0.2, 0.5), 3), "bbox": [u - 6, v - 6, u + 6, v + 6],
                         "attrs": {"temp_k": 300.0, "posture": "lying", "moving": False}})
        return self._payloads(jury, team, cam)

    def _payloads(self, jury, team, cam):
        return (
            {"source": "jury", "image": [cam.width, cam.height], "objets": jury},
            {"source": "team", "image": [cam.width, cam.height], "detections": team},
        )

    # ---------------------------------------------------- mine layers 2 and 3
    MAG_K = {"mine": 4.0e4, "weapon": 3.0e3}  # dipole strength, nT * m^3

    def mag_sample(self, pos) -> float:
        """Total magnetic field (nT) at the sensor: Earth field + slow drift + ferrous objects + noise."""
        b = 45000.0 + 0.5 * math.sin(2 * math.pi * self.t / 600.0)
        for o in self.objects:
            k = self.MAG_K.get(o.cls)
            if k:
                r2 = (o.p[0] - pos[0]) ** 2 + (o.p[1] - pos[1]) ** 2 + (max(pos[2], 0.5) + float(o.attrs.get("depth", 0.0))) ** 2
                b += k / r2 ** 1.5
        return b + self.rng.gauss(0.0, float(self.sc.get("mag_noise_nt", 0.15)))

    def sonar_ping(self, pos, range_m: float = 60.0) -> list[dict]:
        """Dipping sonar: works only hovering low over water; sees submerged objects within range."""
        if pos[2] > 4.5 or not in_any_polygon(pos, self.water):
            return []
        out = []
        for o in self.objects:
            if o.cls == "mine" and float(o.attrs.get("depth", 0.0)) > 0.5 and in_any_polygon(o.p, self.water):
                r = math.hypot(o.p[0] - pos[0], o.p[1] - pos[1])
                if r <= range_m:
                    out.append({"range": max(0.5, r + self.rng.gauss(0, 1.0)),
                                "bearing": math.atan2(o.p[1] - pos[1], o.p[0] - pos[0]) + self.rng.gauss(0, 0.03),
                                "depth": float(o.attrs["depth"]) + self.rng.gauss(0, 0.5),
                                "strength": max(0.55, 0.95 - 0.4 * r / range_m)})
        if self.rng.random() < 0.1:  # rock / clutter: weak contact (never confirms alone)
            out.append({"range": self.rng.uniform(10, range_m), "bearing": self.rng.uniform(-math.pi, math.pi),
                        "depth": self.rng.uniform(2, 15), "strength": 0.3})
        return out

    def truth_of(self, cls: str) -> list[TruthObject]:
        return [o for o in self.objects if o.cls == cls]

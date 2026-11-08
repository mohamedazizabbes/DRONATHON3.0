"""Reflex layer (50 Hz, target < 5 ms per tick), classical control only.

1. Waypoint following (pure pursuit with switching radius) of the path from
   the orchestrator.
2. Potential-field repulsion from obstacles; moving obstacles (boat) are
   predicted at constant velocity over `boat_horizon_s`.
3. Hard geofence veto: the commanded velocity is projected off any exclusion
   zone boundary (no inward component inside the buffer band) and off the
   operating-area boundary.
4. Wind feed-forward: integral of (desired - achieved) horizontal velocity.
5. Autonomous safety: critical battery -> go home and land on its own;
   stale plan (brain silent) -> hold position.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Sequence

from ..geometry import dist2


@dataclass
class ReflexOutput:
    vel: tuple[float, float, float]
    yaw: float
    mode: str  # follow | hold | critical_rth
    vetoed: bool = False
    info: dict = field(default_factory=dict)


class ReflexController:
    def __init__(self, params: dict):
        self.rp = params["reflex"]
        self.mp = params["mission"]
        self.ep = params["energy"]
        self.area = params["mission"]["area"]
        self.home = tuple(self.mp["home"])
        self.wind_ff = [0.0, 0.0]
        self.last_des = (0.0, 0.0)
        self.yaw = 0.0
        self._path_key = None
        self._idx = 0

    def _target(self, pos, wps) -> tuple[tuple[float, float, float], bool]:
        key = tuple(tuple(round(c, 2) for c in w) for w in wps)
        if key != self._path_key:
            self._path_key = key
            self._idx = 0
        L = max(self.rp["lookahead_m"], 2.0)
        while self._idx < len(wps) - 1 and dist2(pos, wps[self._idx]) < L:
            self._idx += 1
        w = wps[self._idx]
        return (w[0], w[1], w[2]), self._idx == len(wps) - 1

    def step(
        self,
        t: float,
        pos: Sequence[float],
        vel: Sequence[float],
        waypoints: Sequence[Sequence[float]],
        plan_t: float,
        zones: Sequence[tuple[tuple[float, float], float]],
        obstacles: Sequence[tuple[tuple[float, float], float, tuple[float, float]]] = (),
        battery_pct: float = 100.0,
        speed_scale: float = 1.0,
        provisional: Sequence[tuple[tuple[float, float], float]] = (),
    ) -> ReflexOutput:
        """`zones`: confirmed hard zones (veto + escape if inside).
        `provisional`: unconfirmed danger detections: inward veto only when
        outside, never an escape manoeuvre (a false positive must not move us)."""
        rp = self.rp
        # --- wind feed-forward: integrate the tracking error of the last tick
        a = rp["wind_ff_alpha"]
        for i in (0, 1):
            self.wind_ff[i] += a * (self.last_des[i] - vel[i])
            self.wind_ff[i] = max(-rp["wind_ff_max"], min(rp["wind_ff_max"], self.wind_ff[i]))

        mode = "follow"
        wps = [tuple(w) for w in waypoints]
        if battery_pct <= self.ep["critical_pct"]:
            mode = "critical_rth"
            far = dist2(pos, self.home) > 2.0
            wps = [(self.home[0], self.home[1], self.mp["cruise_alt"] if far else 0.0)]
        elif t - plan_t > rp["stale_plan_s"] or not wps:
            mode = "hold"
            wps = [(pos[0], pos[1], pos[2])]

        target, final = self._target(pos, wps)
        vmax = min(rp["max_speed"], self.mp["cruise_speed"]) * speed_scale
        if mode == "critical_rth":
            vmax = rp["max_speed"]
        dx, dy = target[0] - pos[0], target[1] - pos[1]
        d = math.hypot(dx, dy)
        sp = min(vmax, rp["kp_pos"] * d) if final else vmax
        vx, vy = (dx / d * sp, dy / d * sp) if d > 1e-6 else (0.0, 0.0)

        # --- repulsion from obstacles (static and moving)
        d0 = rp["rep_influence_m"]
        for c, r, v in obstacles:
            cx = c[0] + v[0] * rp["boat_horizon_s"]
            cy = c[1] + v[1] * rp["boat_horizon_s"]
            ox, oy = pos[0] - cx, pos[1] - cy
            n = math.hypot(ox, oy) or 1e-6
            dd = max(0.1, n - r)
            if dd < d0:
                mag = min(rp["rep_gain"] * (1.0 / dd - 1.0 / d0) / (dd * dd), rp["max_speed"])
                vx += mag * ox / n
                vy += mag * oy / n

        self.last_des = (vx, vy)
        vx += self.wind_ff[0]
        vy += self.wind_ff[1]

        # --- hard veto: exclusion zones and area geofence
        vetoed = False
        buf = rp["zone_buffer_m"]
        for c, r in zones:
            ox, oy = pos[0] - c[0], pos[1] - c[1]
            dc = math.hypot(ox, oy) or 1e-6
            nx, ny = ox / dc, oy / dc  # outward normal
            if dc < r + buf:
                inward = -(vx * nx + vy * ny)
                if inward > 0:
                    vx += inward * nx
                    vy += inward * ny
                    vetoed = True
                if dc < r:  # a zone appeared on top of us: leave radially
                    vx += nx * 2.0
                    vy += ny * 2.0
                    vetoed = True
        for c, r in provisional:
            ox, oy = pos[0] - c[0], pos[1] - c[1]
            dc = math.hypot(ox, oy) or 1e-6
            if r <= dc < r + buf:
                nx, ny = ox / dc, oy / dc
                inward = -(vx * nx + vy * ny)
                if inward > 0:
                    vx += inward * nx
                    vy += inward * ny
                    vetoed = True
        ar, m = self.area, 25.0
        if (pos[0] < ar["xmin"] - m and vx < 0) or (pos[0] > ar["xmax"] + m and vx > 0):
            vx, vetoed = 0.0, True
        if (pos[1] < ar["ymin"] - m and vy < 0) or (pos[1] > ar["ymax"] + m and vy > 0):
            vy, vetoed = 0.0, True

        h = math.hypot(vx, vy)
        cap = rp["max_speed"] + rp["wind_ff_max"]
        if h > cap:
            vx, vy = vx / h * cap, vy / h * cap
        vz = max(-rp["max_vz"], min(rp["max_vz"], rp["kp_alt"] * (target[2] - pos[2])))
        # climbing out after a delivery: climb before transit, so the camera
        # footprint is large enough to see danger zones before entering them
        if pos[2] < rp["climb_first_frac"] * target[2] and target[2] - pos[2] > 3.0:
            vx, vy = 0.0, 0.0
        # descend last (mirror rule): keep cruise altitude, hence a large camera
        # footprint, until above a lower target (delivery, sonar dip, landing)
        if target[2] < pos[2] - 3.0 and math.hypot(dx, dy) > rp["descend_radius_m"]:
            vz = max(vz, 0.0)
        dvx, dvy = self.last_des
        if math.hypot(dvx, dvy) > 0.5:
            self.yaw = math.atan2(dvy, dvx)
        return ReflexOutput((vx, vy, vz), self.yaw, mode, vetoed, {"target": target, "final": final})

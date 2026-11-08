"""PlannerAgent (spec section 3): prioritises casualties, computes wind-corrected
avoidance trajectories, manages objectives (coverage lanes, intercept points)
and decides the return to base. Deterministic, called by the orchestrator nodes.

    agent = PlannerAgent(params)
    agent.triage(state)                -> TriageResult (F6 score + energy gate)
    agent.intercept(state, track)      -> drift intercept point
    agent.plan_path(state, goal)       -> PathResult (tiered margins, evasive exit)
    agent.rth(state)                   -> (threshold_wh, e_rth_wh) (F11)
    agent.calibrate_energy(state)      -> online power-model calibration

Evasive manoeuvres (threat-aware planning, never targeting): if the drone is
inside a hostile unit's line-of-sight ring, the path first exits the ring by
the shortest way (radially away from the unit), then A* continues with the
x10 line-of-sight cost; waypoints still inside a ring are flown at the lower
`planner.evasive_alt` to reduce exposure.
"""
from __future__ import annotations

import math
from typing import Optional

from ..domain import Track, Zone
from ..energy import EnergyModel
from ..geometry import dist2
from .astar import GridPlanner, PathResult
from .coverage import CoverageGrid, lawnmower
from .intercept import intercept_point
from .triage import TriageResult, score_casualties


class PlannerAgent:
    def __init__(self, params: dict):
        self.p = params
        self.mp = params["mission"]
        self.pp = params["planner"]
        self.energy = EnergyModel.from_params(params)
        self.home = tuple(self.mp["home"])
        self.static_obstacles = [(tuple(o["p"]), float(o["r"]), float(o.get("h", 100.0))) for o in params.get("obstacles", [])]
        self.build_area(self.mp["area"])
        self._rth_path, self._rth_path_t = None, -1e9

    # ------------------------------------------------------------ objectives
    def build_area(self, area: dict) -> None:
        self.area = dict(area)
        cam = self.p["camera"]
        self.grid = GridPlanner(area, self.pp["grid_res"], margin=40.0, max_expansions=self.pp["max_expansions"])
        self.coverage = CoverageGrid(area, 2.0)
        self.lanes = lawnmower(area, self.mp["cruise_alt"], cam["hfov_deg"], cam["lane_overlap"], self.home)
        self.sonar_stations = self._sonar_stations(area)

    def _sonar_stations(self, area: dict) -> list[tuple[float, float]]:
        """Dipping-sonar survey stations over water, ~station_spacing apart (layer 3)."""
        sp = self.p.get("sonar", {})
        if not sp.get("enabled") or not sp.get("survey"):
            return []
        from ..geometry import in_any_polygon

        step = sp["station_spacing_m"] / 2.0
        water = self.p["water"]["polygons"]
        cands = []
        y = area["ymin"] + step / 2
        while y <= area["ymax"]:
            x = area["xmin"] + step / 2
            while x <= area["xmax"]:
                if in_any_polygon((x, y), water):
                    cands.append((x, y))
                x += step
            y += step
        out = []
        for c in sorted(cands, key=lambda p: (p[0], p[1])):
            if all(dist2(c, o) >= sp["station_spacing_m"] * 0.9 for o in out):
                out.append(c)
        return out

    def standoff(self, s, target, extra: float = 0.0):
        """Closest point to `target` that is outside every hard zone (sonar dips, inspections)."""
        if not self.in_zone(s, target):
            return tuple(target)
        pos = s["pos"]
        best = None
        for c, r in self.hard_circles(s, extra):
            if dist2(target, c) < r:
                d = dist2(pos, c) or 1e-6
                ux, uy = (pos[0] - c[0]) / d, (pos[1] - c[1]) / d  # on the drone's side
                cand = (c[0] + ux * (r + 0.5), c[1] + uy * (r + 0.5))
                if not self.in_zone(s, cand) and (best is None or dist2(cand, target) < dist2(best, target)):
                    best = cand
        return best

    # --------------------------------------------------------------- safety
    def margin(self, s) -> float:
        zp = self.p["zones"]
        wind = math.hypot(*s.get("wind", (0.0, 0.0)))
        m = self.pp["drone_radius"] + self.pp["safety_margin"] + self.pp["wind_margin_k"] * wind
        if s.get("visibility", 1.0) < 0.5:
            m += self.pp["low_vis_extra"]
        if not s.get("gnss_ok", True):
            m += zp["gnss_lost_extra"]
        return m

    @staticmethod
    def all_zones(s) -> list[Zone]:
        return list(s.get("zones", [])) + list(s.get("manual_zones", []))

    def hard_circles(self, s, extra: float = 0.0) -> list:
        m = self.margin(s) + extra
        circles = [(z.center, z.radius + m) for z in self.all_zones(s)]
        alt = self.mp["cruise_alt"]
        circles += [(c, r + m) for c, r, h in self.static_obstacles if h >= alt - 5.0]
        return circles

    def soft_circles(self, s) -> list:
        k = self.p["zones"]["los_cost"]
        return [(z.center, z.soft_radius, k) for z in self.all_zones(s) if z.soft_radius > z.radius]

    def in_zone(self, s, p) -> bool:
        return any(dist2(p, c) < r for c, r in self.hard_circles(s))

    def in_los(self, s, p) -> Optional[tuple]:
        for c, r, _ in self.soft_circles(s):
            if dist2(p, c) < r:
                return (c, r)
        return None

    # ----------------------------------------------------------- casualties
    @staticmethod
    def pending(s) -> list[Track]:
        done = set(s.get("served_ids", [])) | set(s.get("failed_ids", []))
        later = {k for k, v in s.get("retry_after", {}).items() if v > s.get("t", 0.0)}  # paused deliveries
        return [t for t in s.get("casualties", []) if t.id not in done and t.id not in later]

    def vg(self, s, toward) -> float:
        pos = s["pos"]
        d = dist2(pos, toward)
        if d < 1e-3:
            return self.mp["cruise_speed"]
        u = ((toward[0] - pos[0]) / d, (toward[1] - pos[1]) / d)
        return self.energy.ground_speed(u, self.mp["cruise_speed"], s.get("wind", (0, 0)))

    def intercept(self, s, t: Track):
        if not t.floating:
            return t.pos
        return intercept_point(s["pos"], t.pos, t.vel, self.vg(s, t.pos))

    def duplicate_of(self, s, t: Track) -> Optional[str]:
        """Served casualty this track probably is (re-detection after drift).
        A casualty seen WITH a delivered kit (IR strobe on the kit pod, visible
        in FLIR) is served whatever its drift history."""
        if t.attrs.get("marked"):
            return "kit-marked"
        served = set(s.get("served_ids", []))
        for o in s.get("casualties", []):
            if o.id in served and o.id != t.id and o.floating == t.floating:
                if o.floating:
                    r = min(self.mp["duplicate_radius_m"] + 3.0 * max(o.sigma, t.sigma), self.mp["duplicate_max_floating_m"])
                else:
                    r = min(self.mp["duplicate_radius_m"] + 2.0 * max(o.sigma, t.sigma), self.mp["duplicate_max_m"])
                if dist2(o.pos, t.pos) <= r:
                    return o.id
        return None

    def triage(self, s, only_urgent: bool = False) -> TriageResult:
        a = self.area
        cands = [t for t in self.pending(s) if self.duplicate_of(s, t) is None]
        if self.mp.get("serve_deceased", False) is False:
            cands = [t for t in cands if not t.attrs.get("likely_deceased", False)]
        if only_urgent:
            cands = [t for t in cands if t.urgency >= self.mp["preempt_urgency"]]
        d_max = math.hypot(a["xmax"] - a["xmin"], a["ymax"] - a["ymin"])
        return score_casualties(
            cands, s["pos"], self.home, self.energy, s["battery_wh"], s.get("wind", (0, 0)),
            self.mp["cruise_speed"], d_max, s.get("gnss_ok", True),
            intercept_fn=lambda t: self.intercept(s, t),
            blocked_fn=lambda p: self.in_zone(s, p),
        )

    # ------------------------------------------------------------- energy
    def rth(self, s) -> tuple[float, float]:
        """E_RTH along the PLANNED path home (A* around zones), refreshed every 2 s,
        never less than the straight line: detours around exclusion zones cost energy."""
        w = s.get("wind", (0, 0))
        v = self.mp["cruise_speed"]
        t = s.get("t", 0.0)
        if t - self._rth_path_t >= 2.0 or self._rth_path is None:
            self._rth_path_t = t
            res = self.plan_path(s, self.home) if dist2(s["pos"], self.home) > 5.0 else None
            self._rth_path = res.path if res is not None and res.ok else None
        straight = self.energy.leg_wh(s["pos"], self.home, v, w)
        leg = straight
        if self._rth_path:
            pts = [tuple(s["pos"][:2])] + [tuple(p) for p in self._rth_path[1:]]
            leg = max(straight, self.energy.path_wh(pts, v, w))
        # vertical: climbing back to cruise altitude (after a delivery / sonar dip) costs hover power
        climb = max(0.0, self.mp["cruise_alt"] - s["pos"][2]) / max(self.p["reflex"]["max_vz"], 0.5)
        e_climb = self.energy.power_w(0.0) * climb / 3600.0
        e_rth = leg + e_climb + self.energy.land_wh + self.energy.reserve_wh
        return self.energy.factor(s.get("gnss_ok", True)) * e_rth, e_rth

    def calibrate_energy(self, s) -> Optional[float]:
        airborne = s["pos"][2] > 2.0 and s.get("armed", False)
        return self.energy.calibrate(s["t"], s["battery_wh"], s.get("vel", (0, 0, 0)), s.get("wind", (0, 0)), airborne)

    # --------------------------------------------------------------- paths
    def plan_path(self, s, goal_xy) -> PathResult:
        """Tiered margins (hysteresis -> nominal -> minimum safe) with evasive exit."""
        start = s["pos"][:2]
        prefix = []
        los = self.in_los(s, start)
        goal_in_same_los = los is not None and dist2(goal_xy, los[0]) < los[1]
        if los is not None and not goal_in_same_los:
            c, r = los
            d = dist2(start, c) or 1e-6
            ux, uy = (start[0] - c[0]) / d, (start[1] - c[1]) / d
            exit_pt = (c[0] + ux * (r + 3.0), c[1] + uy * (r + 3.0))
            if not self.in_zone(s, exit_pt):
                prefix = [exit_pt]
        tiers = [(self.pp["plan_extra_m"], "hyst"), (0.0, "nominal"), (self.pp["drone_radius"] + 1.0 - self.margin(s), "min")]
        res = None
        for extra, tier in tiers:
            origin = prefix[-1] if prefix else start
            res = self.grid.plan(origin, goal_xy, self.hard_circles(s, extra), self.soft_circles(s))
            if res.ok:
                if prefix:
                    res.path = [start] + prefix + res.path[1:]  # res.path[0] is the exit point
                    res.reason = "evasive+" + res.reason
                if tier != "hyst":
                    res.reason += f"/{tier}-margin"
                return res
        return res

    def waypoint_alt(self, s, p, default_alt: float) -> float:
        """Inside a hostile line-of-sight ring: fly at the lower evasive altitude."""
        if self.in_los(s, p) is not None:
            return min(default_alt, self.pp["evasive_alt"])
        return default_alt

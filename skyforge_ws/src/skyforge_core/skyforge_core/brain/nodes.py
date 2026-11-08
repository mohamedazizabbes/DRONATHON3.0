"""Deterministic orchestrator nodes (pure Python, no LLM, no LangGraph import).

Every node takes the DroneState and returns a partial update. The same
functions run under LangGraph or under the plain dict-dispatch FSM.

Graph per tick (10 Hz):
    ingest -> router -> <phase handler> [-> <next phase handler> ...] -> planner
Instant transitions (e.g. EXPLORE -> DETECT -> TRIAGE -> NAVIGATE) are chained
inside the same tick (max 4 hops), so a goal_update or a new casualty is acted
on in the tick where it arrives.
"""
from __future__ import annotations

import hashlib
import math
from typing import Any, Optional

from ..geometry import dist2, segment_clear
from ..mission_log import event
from ..planning.agent import PlannerAgent
from ..planning.intercept import expanding_square, search_step_from_sigma
from ..domain import Cls, Phase, PHASE_FR, Start, Track, Zone, zone_from_dict
from .state import DroneState, snapshot

MAX_HOPS = 4


class Brain:
    """State machine + interrupt router. Planning decisions are delegated to the
    PlannerAgent (prioritisation, intercept, paths, energy/RTH)."""

    def __init__(self, params: dict):
        self.p = params
        self.mp = params["mission"]
        self.home = tuple(self.mp["home"])
        self.agent = PlannerAgent(params)
        cam = params["camera"]
        self.hfov = cam["hfov_deg"]
        self.vfov = math.degrees(2 * math.atan(math.tan(math.radians(self.hfov) / 2) * cam["height"] / cam["width"]))
        self.last_triage = None

    # views on the agent (kept for readability of the handlers)
    energy = property(lambda self: self.agent.energy)
    coverage = property(lambda self: self.agent.coverage)
    lanes = property(lambda self: self.agent.lanes)
    area = property(lambda self: self.agent.area)

    # ================================================================ helpers
    def _ev(self, s: DroneState, etype: str, msg: str, **kw) -> dict:
        ph = s.get("phase", Phase.INIT)
        return event(s.get("t", 0.0), etype, ph.value if hasattr(ph, "value") else str(ph), msg, **kw)

    def _transition(self, s: DroneState, new: Phase, why: str, ev: list) -> dict:
        old = s.get("phase")
        if old == new:
            return {}
        ev.append(
            event(
                s["t"], "phase", new.value, f"{PHASE_FR.get(old, old)} -> {PHASE_FR[new]}: {why}",
                frm=old.value if hasattr(old, "value") else old, to=new.value, why=why,
            )
        )
        s["phase"] = new  # visible to the rest of this node
        return {"phase": new, "prev_phase": old, "phase_since": s["t"], "replan": True}

    def margin(self, s):
        return self.agent.margin(s)

    def all_zones(self, s):
        return self.agent.all_zones(s)

    def hard_circles(self, s, extra: float = 0.0):
        return self.agent.hard_circles(s, extra)

    def soft_circles(self, s):
        return self.agent.soft_circles(s)

    def in_zone(self, s, p) -> bool:
        return self.agent.in_zone(s, p)

    def pending(self, s):
        return self.agent.pending(s)

    def intercept(self, s, t):
        return self.agent.intercept(s, t)

    def duplicate_of(self, s, t):
        return self.agent.duplicate_of(s, t)

    def triage(self, s, only_urgent: bool = False):
        return self.agent.triage(s, only_urgent)

    def _triage_sig(self, s: DroneState) -> str:
        items = sorted((t.id, t.category.value if hasattr(t.category, "value") else str(t.category)) for t in self.pending(s))
        return str(items)

    def _goal(self, kind: str, p, alt: float, **kw) -> dict:
        return {"kind": kind, "p": (float(p[0]), float(p[1])), "alt": float(alt), **kw}

    # ================================================================ nodes
    def ingest(self, s: DroneState) -> dict:
        pos = s["pos"]
        if pos[2] > 0.6 * self.mp["cruise_alt"]:
            # footprint centre is ahead of the drone when the gimbal is tilted
            ahead = pos[2] * math.tan(math.radians(self.p["camera"].get("tilt_deg", 0.0)))
            yaw = s.get("rpy", (0.0, 0.0, 0.0))[2]
            c = (pos[0] + ahead * math.cos(yaw), pos[1] + ahead * math.sin(yaw))
            self.coverage.mark_seen(c, pos[2], self.hfov, self.vfov)
        sig = self._zones_sig(s)
        if sig != s.get("zones_sig"):
            self.coverage.set_blocked([(c, r) for c, r in self.hard_circles(s)])
        ev = []
        scale = self.agent.calibrate_energy(s)
        if scale is not None and abs(scale - s.get("energy_scale", 1.0)) > 0.03:
            ev.append(self._ev(s, "energy", f"power model recalibrated from battery drain: x{scale:.2f}", scale=round(scale, 3)))
        thr, e_rth = self.agent.rth(s)
        return {
            "events": ev,
            "energy_scale": self.energy.scale,
            "coverage": self.coverage.fraction(),
            "rth_threshold_wh": thr,
            "e_rth_wh": e_rth,
            "interrupts": [],
            "actions": [],
            "hops": 0,
        }

    def _zones_sig(self, s: DroneState) -> str:
        q = 2.0  # metres: small drifts/growth do not count as a change
        z = sorted((zz.id, round(zz.center[0] / q), round(zz.center[1] / q), round(zz.radius / q)) for zz in self.all_zones(s))
        return hashlib.md5(str(z).encode()).hexdigest()[:12]

    def router(self, s: DroneState) -> dict:
        """Interrupts in priority order (design doc section 7.1)."""
        ev: list[dict] = []
        u: dict[str, Any] = {}
        ints: list[str] = []
        phase = s["phase"]
        terminal = phase in (Phase.LANDED,)

        # 1. MAYDAY
        if s.get("health_fault") and phase not in (Phase.MAYDAY, Phase.LANDED):
            ints.append("mayday")
            u.update(self._transition(s, Phase.MAYDAY, f"health fault: {s['health_fault']}", ev))
            phase = Phase.MAYDAY

        # 2. Battery below the dynamic RTH threshold
        if phase not in (Phase.RETURN_BASE, Phase.LANDED, Phase.MAYDAY, Phase.INIT):
            if s["battery_wh"] <= s["rth_threshold_wh"]:
                ints.append("battery_rth")
                why = (
                    f"E_avail {s['battery_wh']:.1f}Wh <= {self.energy.factor(s.get('gnss_ok', True)):.1f} x "
                    f"E_RTH {s['e_rth_wh']:.1f}Wh (wind {math.hypot(*s.get('wind', (0, 0))):.1f}m/s, "
                    f"d_home {dist2(s['pos'], self.home):.0f}m)"
                )
                ev.append(self._ev(s, "dec", f"RTH forced: {why}", decision="rth", e_avail=round(s["battery_wh"], 2),
                                   threshold=round(s["rth_threshold_wh"], 2)))
                u.update(self._transition(s, Phase.RETURN_BASE, "battery below RTH threshold", ev))
                u["rth_reason"] = "battery"
                phase = Phase.RETURN_BASE

        # 3. GNSS lost / spoofed
        gnss_ok = s.get("gnss_ok", True)
        if gnss_ok != s.get("gnss_prev_ok", True):
            ints.append("gnss")
            if not gnss_ok:
                ev.append(self._ev(s, "nav", f"GNSS rejected -> nav source {s.get('nav_source')}, RTH factor "
                                   f"{self.energy.rth_factor_gnss_lost}, margins +{self.p['zones']['gnss_lost_extra']}m"))
            else:
                ev.append(self._ev(s, "nav", "GNSS restored"))
            u["gnss_prev_ok"] = gnss_ok
            u["replan"] = True

        # 4. Link
        link_up = s.get("link_up", True)
        if link_up == s.get("autonomous", False):  # state changed
            ints.append("link")
            ev.append(self._ev(s, "link", "LINK LOST -> autonomous mode, messages buffered" if not link_up else "link restored"))
            u["autonomous"] = not link_up

        # 5. goal_update from the station (already HMAC/seq verified by comm)
        if s.get("goal_updates") and not terminal:
            ints.append("goal_update")
            u.update(self._apply_goal_updates(s, ev))
            u["goal_updates"] = []

        # 6. New / moved threat zone crossing the current path
        sig = self._zones_sig(s)
        if sig != s.get("zones_sig"):
            u["zones_sig"] = sig
            wps = s.get("waypoints") or []
            pts = [s["pos"][:2]] + [w[:2] for w in wps]
            circles = self.hard_circles(s)
            if any(not segment_clear(pts[i], pts[i + 1], circles) for i in range(len(pts) - 1)):
                ints.append("zone_replan")
                u["replan"] = True
                ev.append(self._ev(s, "dec", "threat zone crosses current path -> replan", decision="replan"))
        u["interrupts"] = ints
        u["events"] = ev
        return u

    def _apply_goal_updates(self, s: DroneState, ev: list) -> dict:
        u: dict[str, Any] = {"replan": True}
        manual = list(s.get("manual_zones", []))
        locked = s["phase"] == Phase.MAYDAY or (s["phase"] == Phase.RETURN_BASE and s.get("rth_reason") == "battery")
        for g in s["goal_updates"]:
            typ = g.get("type")
            recv = g.get("_recv_t", s["t"])
            if locked and typ not in ("zone_add", "zone_clear", "rtb"):
                ev.append(self._ev(s, "goal", f"goal_update {typ} refused: safety lock ({s['phase'].value}, {s.get('rth_reason')})",
                                   goal=typ, refused=True, seq=g.get("_seq")))
                continue
            if typ == "goto":
                p = tuple(g["p"][:2])
                u["station_goal"] = p
                s["station_goal"] = p
                u.update(self._transition(s, Phase.EXPLORE, f"goal_update goto {p}", ev))
                u["target"] = None
            elif typ == "area":
                self.agent.build_area(g["area"])
                u["explore_idx"] = 0
                u.update(self._transition(s, Phase.EXPLORE, "goal_update new area", ev))
            elif typ == "rtb":
                u.update(self._transition(s, Phase.RETURN_BASE, "goal_update rtb", ev))
                u["rth_reason"] = "station"
            elif typ == "resume":
                u["station_goal"] = None
                s["station_goal"] = None
            elif typ == "zone_add":
                z = zone_from_dict({**g["zone"], "source": "station"})
                manual = [m for m in manual if m.id != z.id] + [z]
            elif typ == "zone_clear":
                manual = [m for m in manual if m.id != g.get("id")]
            elif typ == "priority":
                u["priority_id"] = g.get("id")
                s["priority_id"] = g.get("id")
                if s["phase"] in (Phase.EXPLORE, Phase.NAVIGATE, Phase.TRIAGE):
                    u.update(self._transition(s, Phase.TRIAGE, f"goal_update priority {g.get('id')}", ev))
            else:
                ev.append(self._ev(s, "goal", f"goal_update ignored: unknown type {typ!r}"))
                continue
            ev.append(self._ev(s, "goal", f"goal_update {typ} accepted", goal=typ,
                               reaction_ms=round((s["t"] - recv) * 1000.0, 1), seq=g.get("_seq")))
        u["manual_zones"] = manual
        s["manual_zones"] = manual
        return u

    # ---------------------------------------------------------------- phases
    def h_init(self, s: DroneState) -> dict:
        ev: list = []
        u: dict = {"current_goal": self._goal("takeoff", self.home, self.mp["cruise_alt"])}
        if s.get("armed") and s["pos"][2] >= self.mp["cruise_alt"] - self.mp["takeoff_tol"]:
            if s.get("handover") and self.pending(s):
                # MAYDAY handover: skip explored areas, go straight to TRIAGE
                u.update(self._transition(s, Phase.TRIAGE, "handover snapshot: casualties pending", ev))
            else:
                u.update(self._transition(s, Phase.EXPLORE, "armed, offboard, at cruise altitude", ev))
        u["events"] = ev
        return u

    def _policy(self, s: DroneState) -> str:
        """Effective triage policy. A MAYDAY handover drone serves the casualties it
        inherited immediately (they already waited), whatever the configured policy."""
        return "immediate" if s.get("handover") else self.mp["triage_policy"]

    def _urgent_only(self, s: DroneState) -> bool:
        return self._exploring_left(s) and self._policy(s) == "hybrid"

    def _preempt_needed(self, s: DroneState) -> bool:
        policy = self._policy(s)
        pend = self.pending(s)
        if not pend:
            return False
        if policy == "immediate":
            return True
        if policy == "hybrid":
            return any(t.urgency >= self.mp["preempt_urgency"] for t in pend)
        return False

    def h_explore(self, s: DroneState) -> dict:
        ev: list = []
        u: dict = {}
        if s.get("new_tracks"):
            u.update(self._transition(s, Phase.DETECT, f"{len(s['new_tracks'])} new confirmed track(s)", ev))
            u["events"] = ev
            return u
        # Re-enter triage when a pending casualty (newly) qualifies, e.g. its
        # category was upgraded to IMMEDIATE after 10 s immobile.
        sig = self._triage_sig(s)
        if self._preempt_needed(s) and (sig != s.get("triage_sig") or s["t"] - s.get("triage_t", -1e9) > 10.0):
            # quiet feasibility check first: only leave exploration for a casualty we can serve
            if self.triage(s, only_urgent=self._urgent_only(s)).best is not None:
                u.update(self._transition(s, Phase.TRIAGE, "casualty pending (policy %s)" % self._policy(s), ev))
                u["events"] = ev
                return u
            u["triage_sig"], u["triage_t"] = sig, s["t"]

        sg = s.get("station_goal")
        if sg is not None:
            if dist2(s["pos"], sg) <= self.mp["waypoint_tol"]:
                ev.append(self._ev(s, "goal", f"station goal {tuple(round(v, 1) for v in sg)} reached, resuming coverage"))
                u["station_goal"] = None
            else:
                u["current_goal"] = self._goal("station_goto", sg, self.mp["cruise_alt"])
                u["events"] = ev
                return u

        idx = s.get("explore_idx", 0)
        circles = self.hard_circles(s)
        while idx < len(self.lanes):
            wp = self.lanes[idx]
            if dist2(s["pos"], wp) <= self.mp["waypoint_tol"]:
                idx += 1
            elif any(dist2(wp, c) < r for c, r in circles):
                ev.append(self._ev(s, "dec", f"lane waypoint {idx} inside exclusion zone, skipped", decision="skip_wp"))
                idx += 1
            else:
                break
        u["explore_idx"] = idx
        if idx >= len(self.lanes) or s.get("coverage", 0.0) >= self.mp["coverage_target"]:
            why = f"coverage {100 * s.get('coverage', 0):.0f}% complete"
            if self.pending(s):
                u.update(self._transition(s, Phase.TRIAGE, why + ", casualties pending", ev))
            else:
                u.update(self._transition(s, Phase.RETURN_BASE, why + ", nobody left to serve", ev))
                u["rth_reason"] = "mission_complete"
        else:
            u["current_goal"] = self._goal("explore", self.lanes[idx], self.mp["cruise_alt"], idx=idx)
        u["events"] = ev
        return u

    def h_detect(self, s: DroneState) -> dict:
        ev: list = []
        for t in s.get("new_tracks", []):
            extra = f" START={t.category.value}" if t.cls == Cls.INJURED else ""
            ev.append(self._ev(s, "det", f"confirmed {t.cls} {t.id} at ({t.pos[0]:.1f},{t.pos[1]:.1f}) conf {t.conf:.2f}{extra}",
                               track=t.to_dict()))
        u: dict = {"new_tracks": []}
        s["new_tracks"] = []
        if self._preempt_needed(s):
            u.update(self._transition(s, Phase.TRIAGE, "casualty confirmed", ev))
        else:
            u.update(self._transition(s, Phase.EXPLORE, "no casualty to serve now", ev))
        u["events"] = ev
        return u

    def h_triage(self, s: DroneState) -> dict:
        ev: list = []
        u: dict = {"triage_sig": self._triage_sig(s), "triage_t": s["t"]}
        dups = [(t.id, self.duplicate_of(s, t)) for t in self.pending(s)]
        dups = [(a, b) for a, b in dups if b]
        if dups:
            for a, b in dups:
                why = "seen with a delivered kit (IR strobe)" if b == "kit-marked" else f"re-detection of served {b}"
                ev.append(self._ev(s, "dec", f"{a}: {why}, no second kit", decision="duplicate"))
            u["served_ids"] = list(s.get("served_ids", [])) + [a for a, _ in dups]
            s["served_ids"] = u["served_ids"]
        if s.get("kits", 0) <= 0:
            u.update(self._transition(s, Phase.RETURN_BASE, "no medical kit left", ev))
            u["rth_reason"] = "payload_empty"
            u["events"] = ev
            return u
        res = self.triage(s, only_urgent=self._urgent_only(s))
        self.last_triage = res
        best = res.best
        pid = s.get("priority_id")
        if pid:
            forced = next((c for c in res.ranked if c.id == pid and c.feasible), None)
            if forced is not None:
                best = forced
                forced.reason = "station priority override"
            u["priority_id"] = None
        if res.ranked:
            ev.append(self._ev(
                s, "dec",
                (f"TRIAGE -> {best.id} ({best.category}, S={best.score:.2f}: 0.7*U{best.urgency}*c{best.conf:.2f} + 0.3*P{best.proximity:.2f})"
                 if best else "TRIAGE -> no feasible casualty"),
                decision="triage", chosen=best.id if best else None, candidates=[c.to_log() for c in res.ranked],
            ))
        if best is not None:
            u["target"] = {"id": best.id, "score": best.score, "point": best.target, "retries": 0,
                           "search": None, "search_t": None, "low_t": None,
                           "urgent_only": self._urgent_only(s)}
            u.update(self._transition(s, Phase.NAVIGATE, f"best feasible target {best.id}", ev))
        else:
            u["target"] = None
            if s.get("explore_idx", 0) < len(self.lanes) and s.get("coverage", 0) < self.mp["coverage_target"]:
                why = "none feasible" if res.ranked else ("none eligible yet (policy hybrid)" if self.pending(s) else "no casualty pending")
                u.update(self._transition(s, Phase.EXPLORE, f"{why}, area left", ev))
            else:
                why = "none feasible" if res.ranked else "no casualty pending"
                u.update(self._transition(s, Phase.RETURN_BASE, f"{why}, coverage complete", ev))
                u["rth_reason"] = "mission_complete"
        u["events"] = ev
        return u

    def _exploring_left(self, s: DroneState) -> bool:
        return s.get("explore_idx", 0) < len(self.lanes) and s.get("coverage", 0.0) < self.mp["coverage_target"]

    def _target_track(self, s: DroneState) -> Optional[Track]:
        tgt = s.get("target")
        if not tgt:
            return None
        return next((t for t in s.get("casualties", []) if t.id == tgt["id"]), None)

    def h_navigate(self, s: DroneState) -> dict:
        ev: list = []
        u: dict = {}
        tgt = dict(s.get("target") or {})
        tr = self._target_track(s)
        if not tgt or tr is None:
            u.update(self._transition(s, Phase.TRIAGE, "target lost from track list", ev))
            u["events"] = ev
            return u

        # Pre-emption with hysteresis: a much better casualty appeared.
        res = self.triage(s, only_urgent=self._urgent_only(s)
                          and tgt.get("urgent_only", False))
        if res.best and res.best.id != tgt["id"] and res.best.score > tgt["score"] * self.mp["switch_hysteresis"]:
            ev.append(self._ev(s, "dec", f"switch target {tgt['id']} -> {res.best.id} (S {tgt['score']:.2f} -> {res.best.score:.2f})",
                               decision="retarget", candidates=[c.to_log() for c in res.ranked]))
            tgt = {"id": res.best.id, "score": res.best.score, "point": res.best.target, "retries": 0,
                   "search": None, "search_t": None, "low_t": None}
            tr = next(t for t in s["casualties"] if t.id == tgt["id"])
        else:
            cur = next((c for c in res.ranked if c.id == tgt["id"]), None)
            if cur is not None and not cur.feasible:
                ev.append(self._ev(s, "dec", f"target {tgt['id']} no longer feasible: {cur.reason}", decision="drop_target"))
                u["target"] = None
                u.update(self._transition(s, Phase.TRIAGE, "target infeasible", ev))
                u["events"] = ev
                return u

        point = self.intercept(s, tr)
        tgt["point"] = point
        seen_age = s["t"] - tr.last_seen
        near = dist2(s["pos"], point)

        # Lost target -> expanding square search around the prediction.
        if tgt.get("search"):
            sq = tgt["search"]
            if seen_age < 1.0:
                ev.append(self._ev(s, "dec", f"{tr.id} re-acquired during search", decision="reacquired"))
                tgt["search"] = None
            elif s["t"] - tgt["search_t"] > self.mp["lost_target_search_s"] or not sq:
                ev.append(self._ev(s, "dec", f"{tr.id} not found after search, marked lost", decision="lost"))
                u["failed_ids"] = list(s.get("failed_ids", [])) + [tr.id]
                u["target"] = None
                u.update(self._transition(s, Phase.TRIAGE, "target lost", ev))
                u["events"] = ev
                return u
            else:
                if dist2(s["pos"], sq[0]) <= self.mp["waypoint_tol"]:
                    sq = sq[1:]
                    tgt["search"] = sq
                if sq:
                    u["current_goal"] = self._goal("search", sq[0], self.mp["cruise_alt"] * 0.6)
                    u["target"] = tgt
                    u["events"] = ev
                    return u
        elif near < 4.0 and seen_age > 4.0:
            step = search_step_from_sigma(tr.sigma, self.p["planner"]["search_step"])
            tgt["search"] = expanding_square(point, step, self.p["planner"]["search_legs"])[1:]
            tgt["search_t"] = s["t"]
            ev.append(self._ev(s, "dec", f"{tr.id} not at predicted point, expanding-square search (step {step:.0f}m)",
                               decision="search"))

        if near <= self.mp["delivery_radius"] or (near <= self.mp["waypoint_tol"] and seen_age < 1.0):
            u.update(self._transition(s, Phase.DELIVER, f"within {near:.1f}m of intercept point", ev))
        u["target"] = tgt
        u["current_goal"] = self._goal("casualty", point, self.mp["cruise_alt"], id=tr.id)
        u["events"] = ev
        return u

    def h_deliver(self, s: DroneState) -> dict:
        ev: list = []
        u: dict = {}
        tgt = dict(s.get("target") or {})
        tr = self._target_track(s)
        if not tgt or tr is None:
            u.update(self._transition(s, Phase.TRIAGE, "target lost during delivery", ev))
            u["events"] = ev
            return u
        alt = self.mp["delivery_alt"]
        lead = 1.0 / max(self.p["reflex"]["kp_pos"], 0.1)
        point = (tr.pos[0] + tr.vel[0] * lead, tr.pos[1] + tr.vel[1] * lead)
        u["current_goal"] = self._goal("deliver", point, alt, id=tr.id)
        z = s["pos"][2]
        if tr.attrs.get("likely_deceased"):
            ev.append(self._ev(s, "dec", f"{tr.id} assessed LIKELY DECEASED ({tr.attrs.get('category_why', 'FLIR')}): "
                                         f"kit kept for the living, position reported for recovery",
                               decision="deceased", id=tr.id, pos=[round(tr.pos[0], 1), round(tr.pos[1], 1)]))
            u["failed_ids"] = list(s.get("failed_ids", [])) + [tr.id]
            u["target"] = None
            u.update(self._transition(s, Phase.TRIAGE, "likely deceased, no kit", ev))
            u["events"] = ev
            return u
        if z <= alt + 0.8:
            if tgt.get("low_t") is None:
                tgt["low_t"] = s["t"]
            assessing = tr.category == Start.EXPECTANT and s["t"] - tgt["low_t"] < self.mp["assess_hover_s"]
            err = dist2(s["pos"], tr.pos)
            fresh = (s["t"] - tr.last_seen) <= 1.5
            v = s.get("vel", (0.0, 0.0, 0.0))
            steady = math.hypot(v[0] - tr.vel[0], v[1] - tr.vel[1]) <= self.mp["release_max_speed"]
            # release only well inside the spec radius: estimated offset plus half the
            # track uncertainty must fit in release_offset_m, and never on dead reckoning
            nav_ok = s.get("gnss_ok", True)
            cond = err + 0.5 * tr.sigma <= self.mp["release_offset_m"] and fresh and steady and nav_ok
            tgt["ok_ticks"] = tgt.get("ok_ticks", 0) + 1 if cond else 0
            if cond and not assessing and tgt["ok_ticks"] >= self.mp["release_confirm_ticks"]:
                kits = s.get("kits", 0) - 1
                ev.append(self._ev(
                    s, "dec", f"DELIVERY to {tr.id} ({tr.category.value}): offset {err:.2f}m <= {self.mp['delivery_radius']}m, kits left {kits}",
                    decision="deliver", id=tr.id, offset=round(err, 2), drop=[round(s['pos'][0], 2), round(s['pos'][1], 2)],
                ))
                u["actions"] = list(s.get("actions", [])) + [{"type": "release", "id": tr.id, "p": s["pos"][:2]}]
                u["kits"] = kits
                u["served_ids"] = list(s.get("served_ids", [])) + [tr.id]
                u["target"] = None
                u.update(self._transition(s, Phase.TRIAGE, "kit released", ev))
                u["events"] = ev
                return u
            if s["t"] - tgt["low_t"] > 6.0 + (self.mp["assess_hover_s"] if tr.category == Start.EXPECTANT else 0.0):
                tgt["retries"] = tgt.get("retries", 0) + 1
                tgt["low_t"] = None
                why = (f"track not seen for {s['t'] - tr.last_seen:.1f}s" if not fresh
                       else "GNSS rejected, no blind release" if not nav_ok
                       else f"offset {err:.1f}m (+0.5 sigma {0.5 * tr.sigma:.1f})" if err + 0.5 * tr.sigma > self.mp["release_offset_m"]
                       else "not steady")
                if tgt["retries"] >= self.mp["delivery_max_retries"]:
                    rounds = dict(s.get("abort_rounds", {}))
                    rounds[tr.id] = rounds.get(tr.id, 0) + 1
                    u["abort_rounds"] = rounds
                    if rounds[tr.id] >= self.mp["delivery_rounds"]:
                        ev.append(self._ev(s, "dec", f"delivery to {tr.id} given up after {rounds[tr.id]} rounds ({why})", decision="abort"))
                        u["failed_ids"] = list(s.get("failed_ids", [])) + [tr.id]
                    else:
                        retry = dict(s.get("retry_after", {}))
                        retry[tr.id] = s["t"] + self.mp["delivery_cooldown_s"]
                        u["retry_after"] = retry
                        ev.append(self._ev(s, "dec", f"delivery to {tr.id} paused ({why}); retry in {self.mp['delivery_cooldown_s']:.0f}s "
                                                     f"(round {rounds[tr.id]}/{self.mp['delivery_rounds']})", decision="abort"))
                    u["target"] = None
                    u.update(self._transition(s, Phase.TRIAGE, "delivery not confirmed", ev))
                    u["events"] = ev
                    return u
                ev.append(self._ev(s, "dec", f"release blocked ({why}), re-acquire try {tgt['retries']}", decision="retry"))
        u["target"] = tgt
        u["events"] = ev
        return u

    def h_return(self, s: DroneState) -> dict:
        ev: list = []
        u: dict = {}
        d = dist2(s["pos"], self.home)
        if d <= self.mp["waypoint_tol"]:
            u["current_goal"] = self._goal("land", self.home, 0.0)
            if s["pos"][2] < 0.3:
                u.update(self._transition(s, Phase.LANDED, "on the pad", ev))
                u["actions"] = list(s.get("actions", [])) + [{"type": "disarm"}]
        else:
            u["current_goal"] = self._goal("home", self.home, self.mp["cruise_alt"])
        u["events"] = ev
        return u

    def h_landed(self, s: DroneState) -> dict:
        return {"current_goal": None, "waypoints": [], "events": []}

    def h_mayday(self, s: DroneState) -> dict:
        ev: list = []
        u: dict = {}
        if not s.get("mayday_sent"):
            snap = snapshot(s, self.coverage.to_rle(), self.coverage.digest())
            n_open = sum(1 for c in snap["cas"] if not c["s"])
            ev.append(self._ev(s, "may", f"MAYDAY: {s.get('health_fault')} -> handover snapshot ({n_open} casualties to serve, "
                                         f"{len(snap['zones'])} zones)", fault=s.get("health_fault"), snapshot=snap))
            u["actions"] = list(s.get("actions", [])) + [{"type": "mayday", "snapshot": snap}]
            u["mayday_sent"] = True
            spot = self._safe_landing_spot(s)
            u["current_goal"] = self._goal("mayday_land", spot, 0.0)
        else:
            u["current_goal"] = s.get("current_goal")
        if s["pos"][2] < 0.3:
            u.update(self._transition(s, Phase.LANDED, "controlled emergency landing", ev))
        u["events"] = ev
        return u

    def _safe_landing_spot(self, s: DroneState):
        pos = s["pos"][:2]
        cas = [t.pos for t in s.get("casualties", [])]
        circles = self.hard_circles(s)

        def ok(p):
            return all(dist2(p, c) >= 10.0 for c in cas) and all(dist2(p, c) >= r for c, r in circles)

        if ok(pos):
            return pos
        for r in (10.0, 15.0, 20.0, 30.0, 40.0):
            for k in range(16):
                a = 2 * math.pi * k / 16
                p = (pos[0] + r * math.cos(a), pos[1] + r * math.sin(a))
                if ok(p):
                    return p
        return pos

    # ---------------------------------------------------------------- planner
    def planner_node(self, s: DroneState) -> dict:
        ev: list = []
        u: dict = {}
        goal = s.get("current_goal")
        if not goal:
            return {"waypoints": [], "events": []}
        gp = (goal["p"][0], goal["p"][1], goal["alt"])
        prev = s.get("path_goal")
        moved = prev is None or dist2(prev, gp) > self.p["planner"]["replan_move_m"] or abs(prev[2] - gp[2]) > 0.5
        if not (moved or s.get("replan") or not s.get("waypoints")):
            return {"events": []}
        if goal["kind"] == "deliver" and dist2(s["pos"], gp) < 15.0:
            # final approach over a moving casualty: waypoint follows it every tick
            return {"waypoints": [(gp[0], gp[1], gp[2])], "path_goal": gp, "replan": False, "events": []}
        if goal["kind"] in ("land", "mayday_land", "takeoff"):
            # vertical manoeuvre (land pad / takeoff point): no horizontal planning
            wps = [(gp[0], gp[1], gp[2])]
            return {"waypoints": wps, "path_goal": gp, "replan": False, "events": []}
        res = self.agent.plan_path(s, goal["p"])
        if not res.ok:
            ev.append(self._ev(s, "dec", f"no path to {goal['kind']} goal: {res.reason}", decision="no_path", reason=res.reason))
            if goal["kind"] == "explore":
                u["explore_idx"] = s.get("explore_idx", 0) + 1
            elif goal["kind"] in ("casualty", "deliver") and s.get("target"):
                u["failed_ids"] = list(s.get("failed_ids", [])) + [s["target"]["id"]]
                u["target"] = None
                u.update(self._transition(s, Phase.TRIAGE, "target unreachable", ev))
            elif goal["kind"] == "station_goto":
                u["station_goal"] = None
            # hold position meanwhile (the reflex keeps us safe)
            u["waypoints"] = [(s["pos"][0], s["pos"][1], max(s["pos"][2], 2.0))]
            u["path_goal"] = None
            u["replan"] = True
            u["events"] = ev
            return u
        cruise = self.mp["cruise_alt"]
        pts = res.path[1:]
        wps = [(p[0], p[1], self.agent.waypoint_alt(s, p, cruise)) for p in pts[:-1]] + [(pts[-1][0], pts[-1][1], gp[2])]
        if res.reason.startswith("evasive"):
            ev.append(self._ev(s, "dec", f"EVASIVE: inside hostile line of sight, exiting radially then {res.reason}",
                               decision="evasive"))
        elif res.reason != "direct":
            ev.append(self._ev(s, "plan", f"A* path to {goal['kind']} ({len(wps)} wps, {res.reason})"))
        u.update({"waypoints": wps, "path_goal": gp, "replan": False, "events": ev})
        return u

    # ------------------------------------------------------------- dispatch
    def handlers(self) -> dict:
        return {
            Phase.INIT: self.h_init,
            Phase.EXPLORE: self.h_explore,
            Phase.DETECT: self.h_detect,
            Phase.TRIAGE: self.h_triage,
            Phase.NAVIGATE: self.h_navigate,
            Phase.DELIVER: self.h_deliver,
            Phase.RETURN_BASE: self.h_return,
            Phase.LANDED: self.h_landed,
            Phase.MAYDAY: self.h_mayday,
        }

    def wrap_handler(self, phase: Phase, fn):
        def node(s: DroneState) -> dict:
            out = fn(s)
            out["hops"] = s.get("hops", 0) + 1
            out["ran"] = phase.value
            return out

        node.__name__ = f"h_{phase.value.lower()}"
        return node

    @staticmethod
    def next_after_handler(s: DroneState) -> str:
        ph = s["phase"]
        phv = ph.value if hasattr(ph, "value") else ph
        if phv != s.get("ran") and s.get("hops", 0) < MAX_HOPS:
            return phv
        return "planner"

    @staticmethod
    def route_phase(s: DroneState) -> str:
        ph = s["phase"]
        return ph.value if hasattr(ph, "value") else ph

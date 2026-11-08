"""End-to-end headless mission: world + PerceptionAgent + LangGraph orchestrator
+ reflex + nav integrity + health + comm (ring buffer, HMAC) + ground station.

    python -m skyforge_core.sim.run                    # default coastal demo
    python -m skyforge_core.sim.run --mayday-at 260    # motor failure -> drone 2
    python -m skyforge_core.sim.run --engine fsm --seeds 10   # seeded batch

Writes runs/<name>/: mission_log.jsonl, metrics.json, trajectory.png
"""
from __future__ import annotations

import argparse
import copy
import json
import math
import os
import random
import time
from collections import deque
from typing import Optional

from ..brain.orchestrator import Orchestrator
from ..brain.state import snapshot as make_snapshot
from ..comm.drone_comm import DroneComm
from ..comm.link import LinkSimulator
from ..comm.messages import event_to_msg
from ..comm.station import GroundStation
from ..domain import Cls, Phase, Start, URGENCY, Zone, track_from_dict
from ..energy import EnergyModel
from ..geometry import dist2, kendall_tau
from ..mission_log import MissionLog, event
from ..monitor.health import HealthMonitor
from ..monitor.metrics import LatencyTracker
from ..monitor.nav_integrity import NavIntegrity
from ..params import deep_merge, load_params
from ..perception.adapter import parse_detections
from ..perception.agent import PerceptionAgent
from ..perception.projection import Camera
from ..reflex.controller import ReflexController
from .world import DEFAULT_SCENARIO, World

SAFETY_TRUTH = {"mine": 15.0, "weapon": 10.0}


class DroneUnit:
    """One drone: physics + its full onboard stack."""

    def __init__(self, name: str, params: dict, world: World, log: MissionLog, engine: str, rng: random.Random,
                 station_q: deque, t0: float):
        self.name = name
        self.p = params
        self.world = world
        self.rng = rng
        home = params["mission"]["home"]
        self.pos = [home[0], home[1], 0.0]
        self.vel = [0.0, 0.0, 0.0]
        self.yaw = 0.0
        self.energy = EnergyModel.from_params(params)
        self.battery_wh = self.energy.capacity_wh
        self.cam = Camera.from_params(params)
        self.agent = PerceptionAgent(params, self.cam)
        self.orch = Orchestrator(params, log, engine)
        self.orch.state["t"] = t0
        self.orch.state["phase_since"] = t0
        self.reflex = ReflexController(params)
        self.nav = NavIntegrity(params, home)
        self.health = HealthMonitor(params)
        self.log = log
        self.station_q = station_q
        self.link = None  # set by sim
        self.comm = DroneComm(params, self._tx, snapshot_fn=lambda: make_snapshot(self.orch.state), boot_id=rng.randint(1, 10**6))
        self.armed = True
        self.motor_failed = False
        self.plan_t = t0
        self.goals: list[dict] = []
        self.spoof = None  # (t0, offset, ramp, dur)
        self.trajectory: list[tuple] = []
        self.done = False
        self.mayday_snapshot = None
        self.lat_reflex = LatencyTracker("reflex")
        self.lat_percep = LatencyTracker("perception")
        self.lat_total = LatencyTracker("total")
        self._last_percep_ms = 0.0
        self.e_at_rth = None
        self.pred_rth = None
        self.distance = 0.0

    def _tx(self, data: bytes) -> bool:
        if self.link.deliver(self.pos):
            self.station_q.append(data)
            return True
        return False

    # ----------------------------------------------------------- sensors
    def nav_position(self, t: float):
        gx = self.pos[0] + self.rng.gauss(0, 0.6)
        gy = self.pos[1] + self.rng.gauss(0, 0.6)
        if self.spoof:
            t0, off, ramp, dur = self.spoof
            if t <= t0 + dur:
                k = min(1.0, (t - t0) / max(ramp, 1e-3))
                gx += off[0] * k
                gy += off[1] * k
            else:
                self.spoof = None
        self.nav.propagate(t, (self.vel[0] + self.rng.gauss(0, 0.05), self.vel[1] + self.rng.gauss(0, 0.05)))
        self.nav.gnss_fix(t, (gx, gy))
        self.nav.check_timeout(t)
        x, y = self.nav.position((gx, gy))
        return (x, y, self.pos[2])

    # ----------------------------------------------------------- physics
    def physics(self, dt: float, cmd) -> None:
        tau, cd = 0.4, 0.15
        w = self.world.wind
        vmax = 3.0 if self.motor_failed else 99.0
        for i in (0, 1):
            a = (cmd[i] - self.vel[i]) / tau + cd * (w[i] - self.vel[i])
            self.vel[i] += a * dt
        h = math.hypot(self.vel[0], self.vel[1])
        if h > vmax:
            self.vel[0] *= vmax / h
            self.vel[1] *= vmax / h
        self.vel[2] += (cmd[2] - self.vel[2]) / tau * dt
        old = list(self.pos)
        for i in range(3):
            self.pos[i] += self.vel[i] * dt
        if self.pos[2] <= 0.0:
            self.pos[2] = 0.0
            self.vel[2] = max(0.0, self.vel[2])
            if self.orch.state["phase"] in (Phase.INIT,):
                pass
            else:
                self.vel[0] *= 0.5
                self.vel[1] *= 0.5
        self.distance += math.hypot(self.pos[0] - old[0], self.pos[1] - old[1])
        if self.armed and (self.pos[2] > 0.05 or self.orch.state["phase"] == Phase.INIT):
            v_air = math.hypot(self.vel[0] - w[0], self.vel[1] - w[1])
            p = self.energy.power_w(v_air) * 1.04 + 25.0 * max(0.0, self.vel[2])
            self.battery_wh -= p * dt / 3600.0

    def motor_outputs(self):
        if self.motor_failed:
            return [1.0, 0.62, 0.6, 0.61]
        return [0.55 + 0.05 * self.rng.random() for _ in range(4)]


class MissionSim:
    def __init__(self, params: dict, scenario: Optional[dict] = None, out_dir: str = "runs/demo", engine: str = "langgraph",
                 mayday_at: Optional[float] = None, handover: bool = True, seed: Optional[int] = None, quiet: bool = False):
        self.sc = copy.deepcopy(scenario or DEFAULT_SCENARIO)
        if seed is not None:
            self.sc["seed"] = seed
        # obstacles known to the brain come from the scenario (jury-defined)
        params = deep_merge(params, self.sc.get("params", {}))
        params = deep_merge(params, {"obstacles": self.sc.get("obstacles", [])})
        self.p = params
        self.out = out_dir
        os.makedirs(out_dir, exist_ok=True)
        self.quiet = quiet
        self.rng = random.Random(self.sc.get("seed", 0))
        self.world = World(params, self.sc)
        self.log = MissionLog(os.path.join(out_dir, "mission_log.jsonl"))
        self.engine = engine
        self.link = LinkSimulator(params, seed=self.sc.get("seed", 0))
        self.to_station: deque = deque()
        self.to_drone: deque = deque()
        self.station = GroundStation(params, self._station_tx, self._on_mayday, boot_id=99)
        self.mayday_at = mayday_at
        self.handover = handover
        self.drones: list[DroneUnit] = [self._new_drone("drone1", 0.0)]
        self.active = self.drones[0]
        self.faults = sorted(self.sc.get("faults", []), key=lambda f: f["t"])
        self.fault_i = 0
        self.jam_until = -1.0
        self.deliveries: list[dict] = []
        self.collisions: list[dict] = []
        self.zone_entries: list[dict] = []
        self.min_clear: dict[str, float] = {}
        self._in_contact: set = set()
        self.goal_reactions: list[float] = []
        self.pending_handover: Optional[dict] = None

    def _new_drone(self, name: str, t0: float) -> DroneUnit:
        d = DroneUnit(name, self.p, self.world, self.log, self.engine, self.rng, self.to_station, t0)
        d.link = self.link
        return d

    def _station_tx(self, data: bytes) -> bool:
        if self.link.deliver(self.active.pos):
            self.to_drone.append(data)
            return True
        return False

    def _on_mayday(self, msg: dict) -> None:
        self.pending_handover = msg

    def say(self, s: str) -> None:
        if not self.quiet:
            print(s, flush=True)

    # ---------------------------------------------------------------- faults
    def _faults(self, t: float) -> None:
        while self.fault_i < len(self.faults) and self.faults[self.fault_i]["t"] <= t:
            f = self.faults[self.fault_i]
            self.fault_i += 1
            ft = f["type"]
            d = self.active
            if ft == "jam":
                self.link.set_jammed(True)
                self.jam_until = t + f.get("dur", 20)
            elif ft == "spoof":
                d.spoof = (t, f["offset"], f.get("ramp", 5), f.get("dur", 20))
            elif ft == "gust":
                self.world.set_gust(f["wind"], f.get("dur", 20))
            elif ft == "goal":
                self.station.send_goal(f["goal"], t)
            elif ft == "forged_goal":
                self.station.send_forged_goal(f["goal"], t)
            elif ft == "replay_goal":
                self.station.replay_last_goal()
            elif ft == "motor":
                d.motor_failed = True
            self.log.add(event(t, "fault", d.orch.state["phase"].value, f"FAULT INJECTED: {ft} {json.dumps({k: v for k, v in f.items() if k not in ('t', 'type')})}"))
        if self.jam_until >= 0 and t > self.jam_until:
            self.link.set_jammed(False)
            self.jam_until = -1.0
        if self.mayday_at is not None and t >= self.mayday_at and not self.active.motor_failed and self.active.name == "drone1":
            self.active.motor_failed = True
            self.log.add(event(t, "fault", self.active.orch.state["phase"].value, "FAULT INJECTED: motor failure"))

    # -------------------------------------------------------------- main loop
    def run(self) -> dict:
        dt = 0.02
        dur = float(self.sc.get("duration_s", 900))
        steps = int(dur / dt)
        wall0 = time.time()
        for k in range(steps):
            t = round((k + 1) * dt, 4)
            self.world.step(dt)
            self._faults(t)
            d = self.active
            # --- comm queues (one hop per step)
            while self.to_station:
                self.station.on_receive(self.to_station.popleft(), t)
            while self.to_drone:
                d.goals += d.comm.on_receive(self.to_drone.popleft(), t)
            # --- decision 10 Hz
            if k % 5 == 0:
                self._brain_tick(d, t)
            # --- reflex 50 Hz
            t0 = time.perf_counter_ns()
            st = d.orch.state
            zones = [(z.center, z.radius + self.p["planner"]["drone_radius"] + 1.0) for z in list(st.get("zones", [])) + list(st.get("manual_zones", []))]
            obst = [(c, r + self.p["planner"]["drone_radius"], v) for c, r, h, v in self.world.all_obstacles() if h > d.pos[2] - 2.0]
            # navigation estimate from the last decision tick, propagated with velocity
            age = t - st["t"]
            pos_est = (st["pos"][0] + d.vel[0] * age, st["pos"][1] + d.vel[1] * age, d.pos[2])
            out = d.reflex.step(t, pos_est, d.vel, st.get("waypoints", []), d.plan_t, zones, obst,
                                battery_pct=100.0 * d.battery_wh / d.energy.capacity_wh,
                                speed_scale=0.5 if d.agent.slow_down(pos_est) else 1.0,
                                provisional=d.agent.provisional_zones())
            rms = (time.perf_counter_ns() - t0) / 1e6
            d.lat_reflex.add(rms)
            if st["phase"] == Phase.LANDED:
                out.vel = (0.0, 0.0, -0.5)
            d.physics(dt, out.vel)
            self._truth_checks(d, t)
            if k % 25 == 0:
                d.trajectory.append((t, d.pos[0], d.pos[1], d.pos[2], st["phase"].value))
            # --- end conditions
            if st["phase"] == Phase.LANDED and d.pos[2] <= 0.05:
                if d.mayday_snapshot is not None and self.handover and d.name == "drone1":
                    if self.pending_handover is not None or t - st["phase_since"] > 30:
                        self._spawn_drone2(t)
                        continue
                else:
                    if not d.done:
                        d.done = True
                        self.say(f"[{t:7.1f}s] {d.name} landed, mission over")
                    break
        self.wall_s = time.time() - wall0
        return self._finish()

    def _brain_tick(self, d: DroneUnit, t: float) -> None:
        st = d.orch.state
        # sensors -> nav integrity
        pos = d.nav_position(t)
        # perception 5 Hz (every other decision tick), on its own clock
        tp0 = time.perf_counter_ns()
        if int(round(t / 0.1)) % 2 == 0:
            jury, team = self.world.camera_frame(d.pos, (0.0 + self.rng.gauss(0, 0.01), self.rng.gauss(0, 0.01), d.reflex.yaw), d.cam)
            dets = parse_detections(json.dumps(jury), d.cam.width, d.cam.height, "jury", t) + parse_detections(team, d.cam.width, d.cam.height, "team", t)
            d.agent.process(dets, pos, (0.0, 0.0, d.reflex.yaw), t, self.world.wind)
            d._last_percep_ms = (time.perf_counter_ns() - tp0) / 1e6
            d.lat_percep.add(d._last_percep_ms)
        else:
            d.agent.predict(t, self.world.wind)
        # health
        fault = d.health.update(t, d.motor_outputs(), t, t, 3.9, 45.0)
        link_up = self.link.is_up(d.pos)
        d.comm.set_link(link_up, t)
        new_tracks = d.agent.pop_new_confirmed()
        goals, d.goals = d.goals, []
        inputs = {
            "t": t, "pos": pos, "vel": tuple(d.vel), "rpy": (0.0, 0.0, d.reflex.yaw), "armed": d.armed,
            "battery_wh": d.battery_wh, "battery_pct": 100.0 * d.battery_wh / d.energy.capacity_wh,
            "link_up": link_up, "link_quality": self.link.quality(d.pos), "gnss_ok": d.nav.gnss_ok, "nav_source": d.nav.source,
            "wind": tuple(self.world.wind), "visibility": self.world.visibility,
            "casualties": d.agent.casualties(), "zones": d.agent.zones(), "new_tracks": new_tracks, "health_fault": fault,
        }
        old_wps = st.get("waypoints")
        prev_phase = st["phase"]
        s = d.orch.tick(inputs, goals)
        if s.get("waypoints") != old_wps:
            d.plan_t = t
        else:
            d.plan_t = t  # brain alive -> plan fresh
        d.lat_total.add(d.orch.latency.buf[-1] + d._last_percep_ms + (d.lat_reflex.buf[-1] if d.lat_reflex.buf else 0.0))
        if prev_phase != Phase.RETURN_BASE and s["phase"] == Phase.RETURN_BASE:
            d.e_at_rth = d.battery_wh
            d.pred_rth = s.get("e_rth_wh")
        # events -> console + link
        for ev in s.get("events", []):
            if ev["type"] == "goal" and "reaction_ms" in ev:
                self.goal_reactions.append(ev["reaction_ms"] + s["decision_ms"])
            if ev["type"] in ("phase", "dec", "det", "goal", "nav", "may", "link"):
                self.say(f"[{t:7.1f}s] {d.name} {ev['phase']:<11} {ev['msg']}")
            m = event_to_msg(ev)
            if m:
                d.comm.send(m[0], m[1], t)
        for ev in d.comm.pop_events():
            ev.setdefault("phase", s["phase"].value)
            self.log.add(ev)
            self.say(f"[{t:7.1f}s] {d.name} comm        {ev['msg']}")
        if int(round(t * 10)) % 10 == 0:
            d.comm.send("tel", d.orch.mission_state_msg(), t)
        self.station.tick(t)
        for a in s.get("actions", []):
            if a["type"] == "release":
                self._record_delivery(d, a, t)
            elif a["type"] == "mayday":
                d.mayday_snapshot = a["snapshot"]
            elif a["type"] == "disarm":
                d.armed = False

    def _record_delivery(self, d: DroneUnit, a: dict, t: float) -> None:
        cas = self.world.truth_of(Cls.INJURED)
        nearest = min(cas, key=lambda o: dist2(o.p, d.pos))
        err = dist2(nearest.p, d.pos)
        served_truth = {x["truth"] for x in self.deliveries}
        rec = {"t": t, "track": a["id"], "truth": nearest.id, "cat": nearest.attrs.get("cat"), "err_m": round(err, 2),
               "on_target": err <= self.p["mission"]["delivery_radius"] and nearest.id not in served_truth, "drone": d.name}
        self.deliveries.append(rec)
        self.say(f"[{t:7.1f}s] {d.name} DROP        kit -> {nearest.id} (truth error {err:.2f} m)")

    def _truth_checks(self, d: DroneUnit, t: float) -> None:
        z = d.pos[2]
        for c, r, h, v in self.world.all_obstacles():
            dd = dist2(c, d.pos) - r - 0.4
            key = ("obs", c if v == (0.0, 0.0) else "mover", r)
            if z < h:
                self.min_clear["obstacle"] = min(self.min_clear.get("obstacle", 1e9), dd)
                if dd < 0:
                    if key not in self._in_contact:
                        self._in_contact.add(key)
                        self.collisions.append({"t": t, "pos": [round(x, 1) for x in d.pos], "r": r})
                    continue
            self._in_contact.discard(key)
        if z < 0.5:
            return
        for o in self.world.objects:
            if o.cls in SAFETY_TRUTH:
                R = SAFETY_TRUTH[o.cls]
            elif o.cls == Cls.HOSTILE:
                R = float(o.attrs.get("detection_radius", 20.0))
            else:
                continue
            dd = dist2(o.p, d.pos) - R
            self.min_clear[o.cls] = min(self.min_clear.get(o.cls, 1e9), dd)
            key = ("zone", o.id)
            if dd < 0:
                if key not in self._in_contact:
                    self._in_contact.add(key)
                    self.zone_entries.append({"t": t, "obj": o.id, "depth_m": round(-dd, 2)})
            else:
                self._in_contact.discard(key)

    # ----------------------------------------------------------- handover
    def _spawn_drone2(self, t: float) -> None:
        msg = self.pending_handover or {"s": self.active.mayday_snapshot}
        snap = msg.get("s", msg)
        via = "station (reassembled from fragments)" if self.pending_handover else "local copy (station never received it)"
        self.say(f"[{t:7.1f}s] station     MAYDAY handover -> drone2 takes off with snapshot via {via}")
        d2 = self._new_drone("drone2", t)
        st = d2.orch.state
        st["served_ids"] = sorted(set(snap.get("served", [])) | {c["id"] for c in snap.get("cas", []) if c.get("s")})
        st["explore_idx"] = int(snap.get("explore_idx", 0))
        st["handover"] = True
        if snap.get("cov_rle"):
            d2.orch.brain.coverage.load_rle(snap["cov_rle"])
        st["manual_zones"] = [Zone(z["id"], z["cls"], tuple(z["center"]), float(z["radius"]), tuple(z.get("vel", (0, 0))), 0.0, "handover")
                              for z in snap.get("zones", [])]
        d2.agent.seed_tracks([track_from_dict({**c, "cls": Cls.INJURED, "confirmed": True}) for c in snap.get("cas", [])], t)
        self.log.add(event(t, "handover", "INIT", f"drone2 resumes: {sum(1 for c in snap.get('cas', []) if not c.get('s'))} casualties to serve, "
                                                    f"{len(snap.get('zones', []))} zones, coverage {snap.get('cov')}"))
        self.drones.append(d2)
        self.active = d2
        self.pending_handover = None

    # ------------------------------------------------------------ metrics
    def _finish(self) -> dict:
        self.log.close()
        cas_truth = self.world.truth_of(Cls.INJURED)
        served_order = [x["truth"] for x in self.deliveries if x["on_target"]]
        oracle = sorted([o.id for o in cas_truth], key=lambda i: -URGENCY[Start(next(o for o in cas_truth if o.id == i).attrs["cat"])])
        loc_err = []
        for d in self.drones:
            for tr in d.agent.confirmed():
                same = [o for o in self.world.objects if o.cls == tr.cls]
                if same:
                    loc_err.append(min(dist2(o.p, tr.pos) for o in same))
        false_tracks = 0
        for d in self.drones:
            for tr in d.agent.confirmed():
                if not any(o.cls == tr.cls and dist2(o.p, tr.pos) < 8.0 for o in self.world.objects):
                    false_tracks += 1
        detected_truth = {}
        for o in self.world.objects:
            detected_truth[o.id] = any(tr.cls == o.cls and dist2(o.p, tr.pos) < 8.0 for d in self.drones for tr in d.agent.confirmed())
        d1 = self.drones[0]
        last = self.drones[-1]
        rth = None
        if last.e_at_rth is not None:
            used = last.e_at_rth - last.battery_wh
            pred = (last.pred_rth or 0) - last.energy.reserve_wh
            rth = {"predicted_wh": round(pred, 2), "actual_wh": round(used, 2),
                   "error_pct": round(100 * (pred - used) / max(used, 1e-6), 1),
                   "battery_at_landing_pct": round(100 * last.battery_wh / last.energy.capacity_wh, 1),
                   "reserve_pct": round(100 * last.energy.reserve_frac, 1)}
        comm = d1.comm.summary()
        m = {
            "scenario": self.sc.get("name"), "seed": self.sc.get("seed"), "engine": self.engine,
            "sim_time_s": round(self.world.t, 1), "wall_time_s": round(self.wall_s, 1),
            "final_phase": [d.orch.state["phase"].value for d in self.drones],
            "latency": {
                "decision": d1.orch.latency.summary(), "reflex": d1.lat_reflex.summary(),
                "perception": d1.lat_percep.summary(), "total_chain": d1.lat_total.summary(),
            },
            "safety": {"collisions": len(self.collisions), "zone_entries": len(self.zone_entries),
                       "min_clearance_m": {k: round(v, 2) for k, v in self.min_clear.items()},
                       "collision_list": self.collisions, "zone_entry_list": self.zone_entries},
            "coverage": round(max(d.orch.state.get("coverage", 0) for d in self.drones), 3),
            "detection": {"truth_objects": len(self.world.objects), "found": sum(detected_truth.values()),
                          "per_object": detected_truth, "false_confirmed_tracks": false_tracks,
                          "loc_err_mean_m": round(sum(loc_err) / max(1, len(loc_err)), 2), "loc_err_max_m": round(max(loc_err or [0]), 2)},
            "triage": {"deliveries": self.deliveries, "served_on_target": len(served_order), "casualties": len(cas_truth),
                       "off_target": sum(1 for x in self.deliveries if not x["on_target"]),
                       "served_order": served_order, "oracle_urgency_order": oracle,
                       "kendall_tau_vs_urgency": round(kendall_tau(served_order, oracle), 3),
                       "first_delivery_s": self.deliveries[0]["t"] if self.deliveries else None},
            "rth": rth,
            "comms": {**comm, "station_resync_s": [round(x, 2) for x in self.station.resync_times],
                      "station_rx": self.station.stats, "goal_reaction_ms": self.goal_reactions},
            "security": {"forged_rejected": comm["forged_rejected"], "replay_rejected": comm["replay_rejected"],
                         "spoof_detect_s": round(d1.nav.spoof_detect_t - self._spoof_start(), 2) if d1.nav.spoof_detect_t and self._spoof_start() else None},
            "efficiency": {"distance_m": round(sum(d.distance for d in self.drones), 1),
                           "energy_wh": round(sum(d.energy.capacity_wh - d.battery_wh for d in self.drones), 2),
                           "wh_per_casualty": round(sum(d.energy.capacity_wh - d.battery_wh for d in self.drones) / max(1, len(served_order)), 2)},
            "mayday": {"happened": d1.mayday_snapshot is not None, "drone2": len(self.drones) > 1},
        }
        with open(os.path.join(self.out, "metrics.json"), "w", encoding="utf-8") as f:
            json.dump(m, f, indent=1, default=str)
        try:
            self._plot()
        except Exception as e:  # noqa: BLE001 (plot is best effort)
            self.say(f"plot skipped: {e}")
        return m

    def _spoof_start(self):
        f = next((f for f in self.faults if f["type"] == "spoof"), None)
        return f["t"] if f else None

    def _plot(self) -> None:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(11, 8))
        a = self.p["mission"]["area"]
        ax.add_patch(plt.Rectangle((a["xmin"], a["ymin"]), a["xmax"] - a["xmin"], a["ymax"] - a["ymin"], fill=False, ls="--", color="gray"))
        for poly in self.p["water"]["polygons"]:
            ax.fill([p[0] for p in poly], [p[1] for p in poly], color="#cfe8ff", zorder=0)
        for poly in self.p["comm"].get("shadow_polygons", []):
            ax.fill([p[0] for p in poly], [p[1] for p in poly], color="#ddd", alpha=0.6, zorder=0, label="radio shadow")
        colors = {"IMM": "red", "DEL": "orange", "MIN": "green", "EXP": "black"}
        for o in self.world.objects:
            if o.cls == Cls.INJURED:
                ax.plot(*o.p, "P", color=colors.get(o.attrs.get("cat"), "m"), ms=12)
                ax.annotate(o.id, o.p, fontsize=7)
            elif o.cls in SAFETY_TRUTH or o.cls == Cls.HOSTILE:
                R = SAFETY_TRUTH.get(o.cls, o.attrs.get("detection_radius", 20))
                ax.add_patch(plt.Circle(o.p, R, color="red", alpha=0.15))
                ax.plot(*o.p, "x", color="red")
                ax.annotate(o.id, o.p, fontsize=7)
            else:
                ax.plot(*o.p, ".", color="brown")
        for c, r, h, v in self.world.all_obstacles():
            ax.add_patch(plt.Circle(c, r, color="gray", alpha=0.7))
        styles = ["b-", "m-"]
        for i, d in enumerate(self.drones):
            xs = [p[1] for p in d.trajectory]
            ys = [p[2] for p in d.trajectory]
            ax.plot(xs, ys, styles[i % 2], lw=1, label=d.name)
        for x in self.deliveries:
            dd = next(d for d in self.drones if d.name == x["drone"])
            ax.plot(*[p for p in dd.trajectory if p[0] <= x["t"]][-1][1:3], "*", color="gold", ms=14, mec="k")
        ax.set_aspect("equal")
        m = 30.0  # view = mission area + margin (water polygons are much larger)
        ax.set_xlim(a["xmin"] - m, a["xmax"] + m)
        ax.set_ylim(a["ymin"] - m, a["ymax"] + m)
        ax.set_xlabel("x east (m)")
        ax.set_ylabel("y north (m)")
        ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1.0), fontsize=8)
        ax.set_title(f"SKYFORGE sim - {self.sc.get('name')} seed {self.sc.get('seed')} (drifting objects at end-of-mission positions; stars = kit drops)", fontsize=10)
        fig.tight_layout()
        fig.savefig(os.path.join(self.out, "trajectory.png"), dpi=110)
        plt.close(fig)


def summarize(m: dict) -> str:
    L = m["latency"]
    lines = [
        f"phase(s): {m['final_phase']}   sim {m['sim_time_s']}s (wall {m['wall_time_s']}s)",
        f"latency p95 ms: decision {L['decision']['p95_ms']} | reflex {L['reflex']['p95_ms']} | perception {L['perception']['p95_ms']} | chain {L['total_chain']['p95_ms']}",
        f"safety: collisions {m['safety']['collisions']}, zone entries {m['safety']['zone_entries']}, min clearance {m['safety']['min_clearance_m']}",
        f"coverage {100 * m['coverage']:.1f}%   detections {m['detection']['found']}/{m['detection']['truth_objects']} "
        f"(false confirmed {m['detection']['false_confirmed_tracks']}, loc err mean {m['detection']['loc_err_mean_m']} m)",
        f"triage: served {m['triage']['served_on_target']}/{m['triage']['casualties']} off-target {m['triage']['off_target']} "
        f"order {m['triage']['served_order']} tau {m['triage']['kendall_tau_vs_urgency']}",
        f"rth: {m['rth']}",
        f"comms: max {m['comms']['max_bytes']} B, mean {m['comms']['mean_bytes']} B, replayed {m['comms']['replayed']}, syncs {m['comms']['syncs']}, "
        f"resync {m['comms']['station_resync_s']} s, goal reaction ms {m['comms']['goal_reaction_ms']}",
        f"security: {m['security']}   mayday: {m['mayday']}",
    ]
    return "\n".join(lines)


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description="SKYFORGE headless mission simulator")
    ap.add_argument("--config", default=None, help="skyforge.yaml")
    ap.add_argument("--scenario", default=None, help="scenario yaml/json (default: built-in coastal demo)")
    ap.add_argument("--engine", default="langgraph", choices=["langgraph", "fsm"])
    ap.add_argument("--out", default="runs/demo")
    ap.add_argument("--mayday-at", type=float, default=None)
    ap.add_argument("--no-handover", action="store_true")
    ap.add_argument("--seeds", type=int, default=0, help="run N seeded runs and aggregate")
    ap.add_argument("--quiet", action="store_true")
    a = ap.parse_args(argv)
    try:  # French phase labels on Windows consoles
        import sys

        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:  # noqa: BLE001
        pass
    params = load_params(a.config)
    scenario = None
    if a.scenario:
        import yaml

        with open(a.scenario, encoding="utf-8") as f:
            scenario = deep_merge(DEFAULT_SCENARIO, yaml.safe_load(f))
    if a.seeds:
        rows = []
        for s in range(a.seeds):
            m = MissionSim(params, scenario, os.path.join(a.out, f"seed{s}"), a.engine, a.mayday_at, not a.no_handover, seed=s, quiet=True).run()
            rows.append(m)
            print(f"seed {s}: served {m['triage']['served_on_target']}/{m['triage']['casualties']} coll {m['safety']['collisions']} "
                  f"zone {m['safety']['zone_entries']} cov {m['coverage']:.2f} dec p95 {m['latency']['decision']['p95_ms']}ms "
                  f"phase {m['final_phase']}", flush=True)
        agg = {
            "runs": len(rows),
            "collisions_total": sum(r["safety"]["collisions"] for r in rows),
            "zone_entries_total": sum(r["safety"]["zone_entries"] for r in rows),
            "off_target_total": sum(r["triage"]["off_target"] for r in rows),
            "served_mean": sum(r["triage"]["served_on_target"] for r in rows) / len(rows),
            "coverage_mean": sum(r["coverage"] for r in rows) / len(rows),
            "decision_p95_max_ms": max(r["latency"]["decision"]["p95_ms"] for r in rows),
            "reflex_p95_max_ms": max(r["latency"]["reflex"]["p95_ms"] for r in rows),
            "landed_all": all(all(p == "LANDED" for p in r["final_phase"]) for r in rows),
        }
        with open(os.path.join(a.out, "aggregate.json"), "w", encoding="utf-8") as f:
            json.dump(agg, f, indent=1)
        print(json.dumps(agg, indent=1))
        return
    m = MissionSim(params, scenario, a.out, a.engine, a.mayday_at, not a.no_handover, quiet=a.quiet).run()
    print("\n" + summarize(m))
    print(f"\noutputs in {os.path.abspath(a.out)}")


if __name__ == "__main__":
    main()

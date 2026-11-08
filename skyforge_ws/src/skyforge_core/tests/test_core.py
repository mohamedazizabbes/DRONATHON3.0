"""Unit tests for the SKYFORGE core. Run:  pytest skyforge_ws/src/skyforge_core/tests -q"""
import json
import math
import os

import pytest

from skyforge_core.brain.orchestrator import Orchestrator
from skyforge_core.comm.drone_comm import DroneComm
from skyforge_core.comm.protocol import Codec, ForgedMessage, MessageTooLarge, ReplayGuard
from skyforge_core.comm.ring_buffer import RingBuffer
from skyforge_core.comm.station import GroundStation
from skyforge_core.domain import Cls, Detection, Phase, Start, Track, Zone
from skyforge_core.energy import EnergyModel
from skyforge_core.monitor.nav_integrity import NavIntegrity
from skyforge_core.params import load_params
from skyforge_core.perception.adapter import parse_detections
from skyforge_core.perception.agent import PerceptionAgent
from skyforge_core.perception.drift import DriftKF
from skyforge_core.perception.projection import Camera, pixel_to_ground, world_to_pixel
from skyforge_core.perception.start_triage import classify
from skyforge_core.planning.astar import GridPlanner
from skyforge_core.planning.coverage import lawnmower
from skyforge_core.planning.intercept import intercept_point
from skyforge_core.planning.triage import score_casualties
from skyforge_core.reflex.controller import ReflexController


@pytest.fixture
def P():
    return load_params()


# ------------------------------------------------------------------ comm
def test_codec_roundtrip_size_and_hmac(P):
    c = Codec.from_params(P, boot_id=1)
    data = c.encode("det", {"id": "C3", "c": "injured", "u": "IMM", "p": [41.2, -12.8, 0], "cf": 0.87}, 1832, 1762560000.12)
    assert len(data) < 1024
    msg = c.decode(data)
    assert msg["seq"] == 1832 and msg["id"] == "C3"
    tampered = data.replace(b'"IMM"', b'"MIN"')
    with pytest.raises(ForgedMessage):
        c.decode(tampered)
    evil = Codec(b"wrong", 1, 1024, 1)
    with pytest.raises(ForgedMessage):
        c.decode(evil.encode("goal", {"g": {"type": "rtb"}}, 5, 1.0))


def test_codec_rejects_oversize_and_fragments(P):
    c = Codec.from_params(P, 1)
    big = {"s": {"cas": [{"id": f"C{i}", "pos": [i * 1.1, -i * 2.2]} for i in range(120)]}}
    with pytest.raises(MessageTooLarge):
        c.encode("may", big, 1, 0.0)
    frags = c.fragment("may", big, fid=7)
    assert len(frags) > 1
    for i, f in enumerate(frags):
        assert len(c.encode("frag", f, 100 + i, 0.0)) < 1024


def test_replay_guard():
    g = ReplayGuard()
    assert g.check({"b": 1, "seq": 1, "ts": 1.0})
    assert g.check({"b": 1, "seq": 2, "ts": 2.0})
    assert not g.check({"b": 1, "seq": 2, "ts": 2.0})  # replay
    assert not g.check({"b": 9, "seq": 50, "ts": 0.5})  # old session
    assert g.check({"b": 9, "seq": 1, "ts": 3.0})  # new session, newer in time


def test_ring_buffer_evicts_telemetry_first():
    rb = RingBuffer(4)
    rb.push(1, 2, b"goal", False)
    rb.push(2, 0, b"tel", False)
    rb.push(3, 1, b"det", False)
    rb.push(4, 0, b"tel", False)
    rb.push(5, 1, b"dec", False)  # evicts seq 2 (oldest tel)
    assert 2 not in rb.items and 1 in rb.items
    assert rb.lost_after(0) == {0: 1}
    assert [s for s, _ in rb.to_replay(0, 5)] == [1, 3, 4, 5]
    rb.ack(3)
    assert list(rb.items) == [4, 5]


def test_link_outage_replay_and_resync(P):
    """Drone buffers during an outage; on reconnect the station gets everything in order."""
    link = {"up": True}
    to_station, to_drone = [], []
    drone = DroneComm(P, lambda d: link["up"] and (to_station.append(d) or True), boot_id=3)
    gs = GroundStation(P, lambda d: link["up"] and (to_drone.append(d) or True), boot_id=4)

    def pump(t):
        while to_station or to_drone:
            while to_station:
                gs.on_receive(to_station.pop(0), t)
            while to_drone:
                drone.on_receive(to_drone.pop(0), t)

    for i in range(10):
        drone.send("tel", {"i": i}, float(i))
        pump(float(i))
    link["up"] = False
    drone.set_link(False, 10.0)
    for i in range(10, 30):
        drone.send("det" if i % 5 == 0 else "tel", {"i": i}, float(i))
    link["up"] = True
    drone.set_link(True, 30.0)
    drone.send("tel", {"i": 30}, 30.0)
    for k in range(5):
        pump(30.0 + k)
        gs.tick(31.0 + k * 3)
        pump(31.0 + k * 3)
    assert gs.contig == drone.seq  # nothing lost, in order
    assert drone.stats["replayed"] >= 20
    assert drone.stats["replayed"] < 60  # no replay storm


def test_forged_goal_is_rejected_real_goal_accepted(P):
    to_drone = []
    drone = DroneComm(P, lambda d: True, boot_id=3)
    gs = GroundStation(P, lambda d: to_drone.append(d) or True, boot_id=4)
    gs.send_forged_goal({"type": "rtb"}, 1.0)
    gs.send_goal({"type": "goto", "p": [10, 20]}, 2.0)
    gs.replay_last_goal()
    goals = []
    for d in to_drone:
        goals += drone.on_receive(d, 3.0)
    assert [g["type"] for g in goals] == ["goto"]
    assert drone.stats["forged_rejected"] == 1 and drone.stats["replay_rejected"] == 1


# ------------------------------------------------------------- energy/F11
def test_rth_threshold_grows_with_headwind_and_distance(P):
    e = EnergyModel.from_params(P)
    calm = e.rth_threshold((200, 0), (0, 0), 6.0, (0, 0))
    head = e.rth_threshold((200, 0), (0, 0), 6.0, (4, 0))  # wind blowing away from home
    tail = e.rth_threshold((200, 0), (0, 0), 6.0, (-4, 0))
    near = e.rth_threshold((20, 0), (0, 0), 6.0, (0, 0))
    assert head > calm > tail and calm > near
    assert e.rth_threshold((200, 0), (0, 0), 6.0, (0, 0), gnss_ok=False) > calm


# ----------------------------------------------------------------- triage
def _cas(tid, pos, cat, conf=0.9, t0=0.0):
    from skyforge_core.domain import URGENCY

    return Track(tid, Cls.INJURED, pos, 1.0, conf=conf, confirmed=True, category=cat, urgency=URGENCY[cat], first_seen=t0)


def test_triage_urgency_dominates_and_energy_gate(P):
    e = EnergyModel.from_params(P)
    cas = [_cas("near_minor", (10, 0), Start.MINOR), _cas("far_imm", (150, 0), Start.IMMEDIATE), _cas("too_far", (5000, 0), Start.IMMEDIATE)]
    res = score_casualties(cas, (0, 0), (0, 0), e, 60.0, (0, 0), 6.0, 300.0)
    assert res.best.id == "far_imm"
    too_far = next(c for c in res.ranked if c.id == "too_far")
    assert too_far.score == 0.0 and not too_far.feasible and "energy" in too_far.reason


# ------------------------------------------------------------------- A*
def test_astar_avoids_zone_and_escapes(P):
    area = {"xmin": 0, "xmax": 100, "ymin": -50, "ymax": 50}
    g = GridPlanner(area, 2.0)
    zone = [((50.0, 0.0), 15.0)]
    r = g.plan((0, 0), (100, 0), zone)
    assert r.ok and r.reason != "direct"
    for i in range(len(r.path) - 1):
        from skyforge_core.geometry import seg_point_dist

        assert seg_point_dist(r.path[i], r.path[i + 1], (50, 0)) >= 15.0 - 1e-6
    r2 = g.plan((50, 5), (100, 0), zone)  # start inside a zone that just appeared
    assert r2.ok and "escape" in r2.reason
    r3 = g.plan((0, 0), (50, 0), zone)
    assert not r3.ok and r3.reason == "goal_in_zone"


def test_lawnmower_covers_area(P):
    a = P["mission"]["area"]
    wps = lawnmower(a, 25.0, 80.0, 0.8)
    ys = sorted({round(w[1], 1) for w in wps})
    assert len(wps) % 2 == 0 and len(ys) >= 4
    assert all(a["xmin"] <= w[0] <= a["xmax"] for w in wps)


def test_intercept_leads_drifting_target():
    p = intercept_point((0, 0), (100, 0), (0.0, 1.0), ground_speed=5.0)
    assert p[1] > 15.0  # ~20 s of flight -> ~20 m of drift


# ------------------------------------------------------------ perception
def test_projection_roundtrip_with_tilt_and_attitude(P):
    for tilt in (0.0, 20.0):
        cam = Camera.from_params(P)
        cam.tilt_deg = tilt
        pos, rpy = (10.0, -5.0, 25.0), (0.03, -0.02, 0.7)
        for pt in [(12.0, -3.0), (25.0, 4.0), (0.0, -15.0)]:
            px = world_to_pixel(pt, pos, rpy, cam)
            if px is None or not (0 <= px[0] <= cam.width and 0 <= px[1] <= cam.height):
                continue
            g = pixel_to_ground(px[0], px[1], pos, rpy, cam)
            assert math.hypot(g[0] - pt[0], g[1] - pt[1]) < 1e-6


def test_image_top_is_forward(P):
    cam = Camera.from_params(P)
    cam.tilt_deg = 0.0
    g_top = pixel_to_ground(cam.cx, 0, (0, 0, 20), (0, 0, 0), cam)  # yaw 0 = facing +x (east)
    assert g_top[0] > 5 and abs(g_top[1]) < 1e-6


def test_adapter_accepts_jury_and_team_formats():
    jury = {"objets": [{"classe": "arme", "confiance": 87, "boite": [10, 20, 30, 60]}]}
    team = {"detections": [{"class": "injured", "confidence": 0.8, "bbox": {"xc": 0.5, "yc": 0.5, "w": 0.1, "h": 0.1}}]}
    yolo_list = [{"label": "landmine", "score": 0.4, "box": {"x": 5, "y": 5, "w": 10, "h": 10}}, {"label": "cat", "score": 0.9, "box": [0, 0, 1, 1]}]
    d1 = parse_detections(json.dumps(jury))
    d2 = parse_detections(team, 640, 480)
    d3 = parse_detections(yolo_list)
    assert d1[0].cls == Cls.WEAPON and abs(d1[0].conf - 0.87) < 1e-9
    assert d2[0].cls == Cls.INJURED and d2[0].bbox == pytest.approx((288, 216, 352, 264))
    assert len(d3) == 1 and d3[0].cls == Cls.MINE and d3[0].bbox == (5, 5, 15, 15)


def test_confirmation_filters_single_false_positive(P):
    ag = PerceptionAgent(P)
    cam = ag.cam
    pos, rpy = (0.0, 0.0, 25.0), (0.0, 0.0, 0.0)
    target = (5.0, 3.0)
    px = world_to_pixel(target, pos, rpy, cam)
    det = lambda: Detection(Cls.MINE, 0.8, (px[0] - 5, px[1] - 10, px[0] + 5, px[1]))
    ag.process([det(), Detection(Cls.INJURED, 0.6, (100, 100, 110, 110))], pos, rpy, 0.2)
    assert not ag.confirmed()
    ag.process([det()], pos, rpy, 0.4)
    ag.process([det()], pos, rpy, 0.6)
    conf = ag.confirmed()
    assert [t.cls for t in conf] == [Cls.MINE]
    assert math.hypot(conf[0].pos[0] - target[0], conf[0].pos[1] - target[1]) < 1.0
    z = ag.zones()
    assert len(z) == 1 and z[0].radius >= P["zones"]["safety"]["mine"]
    for k in range(4):  # the lone false casualty dies after misses in view
        ag.process([det()], pos, rpy, 0.8 + 0.2 * k)
    assert not any(t.cls == Cls.INJURED for t in ag.tracks())


def test_drift_kf_predicts_and_grows():
    kf = DriftKF((0, 0), 1.0, True, (0.2, 0.1), vel_sigma0=0.1)
    s0 = kf.sigma
    kf.predict(30.0)
    assert kf.pos == pytest.approx((6.0, 3.0)) and kf.sigma > s0


def test_start_categories(P):
    pp = P["perception"]
    kw = dict(ambient_k=pp["ambient_k"], warm_delta_k=pp["warm_delta_k"], immobile_s=pp["immobile_s"])
    assert classify(309, "lying", False, 15, False, False, False, **kw)[0] == Start.IMMEDIATE
    assert classify(309, "sitting", True, 15, False, False, False, **kw)[0] == Start.DELAYED
    assert classify(309, "standing", True, 15, False, False, False, **kw)[0] == Start.MINOR
    assert classify(296, "lying", False, 15, False, False, False, **kw)[0] == Start.EXPECTANT
    assert classify(307, "lying", False, 2, True, False, False, **kw)[0] == Start.IMMEDIATE  # in water
    assert classify(309, "standing", True, 15, False, False, True, **kw)[0] == Start.DELAYED  # near mine: +1


# -------------------------------------------------------------- reflex
def test_reflex_vetoes_zone_entry(P):
    rc = ReflexController(P)
    out = rc.step(0.0, (0, 0, 25), (0, 0, 0), [(100, 0, 25)], 0.0, [((17.0, 0.0), 16.0)])
    assert out.vetoed and out.vel[0] <= 1e-6  # inward (+x) component removed


def test_reflex_holds_on_stale_plan_and_rth_on_critical(P):
    rc = ReflexController(P)
    out = rc.step(10.0, (50, 50, 25), (0, 0, 0), [(100, 0, 25)], 0.0, [])
    assert out.mode == "hold"
    out = rc.step(10.0, (50, 50, 25), (0, 0, 0), [(100, 0, 25)], 10.0, [], battery_pct=5.0)
    assert out.mode == "critical_rth" and out.vel[0] < 0 and out.vel[1] < 0


# --------------------------------------------------------------- nav
def test_spoofing_detected_quickly(P):
    nav = NavIntegrity(P, (0, 0))
    t, x = 0.0, 0.0
    for k in range(100):  # 10 s steady flight at 5 m/s
        t += 0.1
        x += 0.5
        nav.propagate(t, (5.0, 0.0))
        nav.gnss_fix(t, (x, 0.0))
    assert nav.gnss_ok
    t_sp = t
    for k in range(40):  # ramp spoof 5 m/s
        t += 0.1
        x += 0.5
        nav.propagate(t, (5.0, 0.0))
        nav.gnss_fix(t, (x + 5.0 * (t - t_sp), 0.0))
        if not nav.gnss_ok:
            break
    assert not nav.gnss_ok and t - t_sp < 2.0
    assert abs(nav.position()[0] - x) < 3.0  # still knows where it is


# ---------------------------------------------------------- orchestrator
def _tick(o, t, **kw):
    base = {"t": t, "armed": True, "pos": (0.0, 0.0, 25.0), "battery_wh": 55.0}
    base.update(kw)
    return o.tick(base, kw.pop("goals", None))


@pytest.mark.parametrize("engine", ["langgraph", "fsm"])
def test_orchestrator_flow_and_interrupts(P, engine):
    o = Orchestrator(P, engine=engine)
    s = _tick(o, 0.1)
    assert s["phase"] == Phase.EXPLORE and s["waypoints"]
    cas = _cas("C1", (30.0, 10.0), Start.IMMEDIATE)
    s = _tick(o, 0.2, casualties=[cas], new_tracks=[cas])
    assert s["phase"] == Phase.NAVIGATE and s["target"]["id"] == "C1"  # same tick: DETECT->TRIAGE->NAVIGATE
    # goal_update reroutes immediately (in the tick it arrives)
    s = o.tick({"t": 0.3, "casualties": [cas]}, [{"type": "goto", "p": [100, 50], "_recv_t": 0.3}])
    assert s["phase"] == Phase.EXPLORE and s["current_goal"]["kind"] == "station_goto"
    assert s["waypoints"][-1][:2] == pytest.approx((100, 50))
    # battery under the dynamic threshold -> RETURN_BASE, and goal_update can't override it
    s = _tick(o, 0.4, battery_wh=1.0, pos=(150.0, 0.0, 25.0))
    assert s["phase"] == Phase.RETURN_BASE
    s = o.tick({"t": 0.5, "battery_wh": 1.0}, [{"type": "goto", "p": [10, 10], "_recv_t": 0.5}])
    assert s["phase"] == Phase.RETURN_BASE
    # health fault -> MAYDAY with snapshot action
    s = _tick(o, 0.6, health_fault="motor saturation", battery_wh=1.0)
    assert s["phase"] == Phase.MAYDAY and any(a["type"] == "mayday" for a in s["actions"])


def test_engines_make_identical_decisions(P):
    a, b = Orchestrator(P, engine="langgraph"), Orchestrator(P, engine="fsm")
    cas = [_cas("C1", (60.0, 20.0), Start.DELAYED), _cas("C2", (30.0, -10.0), Start.IMMEDIATE)]
    for k in range(30):
        inp = {"t": 0.1 * k, "armed": True, "pos": (2.0 * k, 0.0, 25.0), "battery_wh": 50.0,
               "casualties": cas if k > 5 else [], "new_tracks": cas if k == 6 else []}
        sa, sb = a.tick(dict(inp)), b.tick(dict(inp))
        assert sa["phase"] == sb["phase"] and sa["waypoints"] == sb["waypoints"]


def test_decision_latency_budget(P):
    o = Orchestrator(P)
    zones = [Zone(f"Z{i}", Cls.MINE, (20.0 + 15 * i, 5.0 * (-1) ** i), 17.0) for i in range(8)]
    cas = [_cas(f"C{i}", (30.0 + 20 * i, 40.0 - 10 * i), Start.DELAYED) for i in range(6)]
    for k in range(100):
        o.tick({"t": 0.1 * k, "armed": True, "pos": (1.0 * k, 0.0, 25.0), "battery_wh": 50.0,
                "zones": zones, "casualties": cas})
    assert o.latency.pct(95) < 100.0  # spec: decision <= 100 ms (p95)


def test_short_sim_smoke(tmp_path, P):
    from skyforge_core.sim.run import MissionSim

    m = MissionSim(P, out_dir=str(tmp_path), quiet=True).run()
    assert m["safety"]["collisions"] == 0 and m["safety"]["zone_entries"] == 0
    assert m["triage"]["off_target"] == 0 and m["triage"]["served_on_target"] >= 3
    assert all(p == "LANDED" for p in m["final_phase"])
    assert m["comms"]["max_bytes"] < 1024


# ------------------------------------------------- energy model v2 / agent
def test_headwind_stronger_than_cruise_does_not_explode(P):
    e = EnergyModel.from_params(P)
    calm = e.e_rth((200, 0), (0, 0), 6.0, (0, 0))
    gale = e.e_rth((200, 0), (0, 0), 6.0, (7.0, 0))  # 7 m/s * 1.2 gust > 6 m/s cruise, blowing away from home
    assert calm < gale < 4 * calm  # airspeed rises to ~14 m/s, ground speed held: no 1 m/s collapse
    storm = e.e_rth((200, 0), (0, 0), 6.0, (15.0, 0))  # beyond the airframe: ground speed drops, energy grows a lot
    assert storm > gale


def test_energy_calibration_converges(P):
    e = EnergyModel.from_params(P)
    true_scale, wh, t = 1.25, 80.0, 0.0
    vel, wind = (6.0, 0.0, 0.0), (0.0, 0.0)
    for _ in range(600):  # 60 s at 10 Hz, steady cruise
        t += 0.1
        wh -= true_scale * e.p_hover_w * (1 + e.k_speed * 36.0) * 0.1 / 3600.0
        e.calibrate(t, wh, vel, wind, airborne=True)
    assert abs(e.scale - true_scale) < 0.05


def test_evasive_exit_from_hostile_line_of_sight(P):
    from skyforge_core.planning.agent import PlannerAgent

    ag = PlannerAgent(P)
    hostile = Zone("ZH1", Cls.HOSTILE, (100.0, 0.0), 25.0, soft_radius=40.0)
    s = {"pos": (90.0, 20.0, 25.0), "zones": [hostile], "manual_zones": [], "wind": (0, 0), "gnss_ok": True}
    res = ag.plan_path(s, (0.0, 60.0))
    assert res.ok and res.reason.startswith("evasive")
    exit_pt = res.path[1]
    assert math.hypot(exit_pt[0] - 100.0, exit_pt[1]) >= 40.0  # first leg leaves the ring
    assert ag.waypoint_alt(s, (95.0, 30.0), 25.0) == P["planner"]["evasive_alt"]
    assert ag.waypoint_alt(s, (0.0, 60.0), 25.0) == 25.0


def test_flir_deceased_rules(P):
    pp = P["perception"]
    kw = dict(ambient_k=pp["ambient_k"], warm_delta_k=pp["warm_delta_k"], immobile_s=pp["immobile_s"],
              deceased_delta_k=pp["deceased_delta_k"], deceased_obs_s=pp["deceased_obs_s"])
    # ambient skin + no breathing for long enough, on land -> likely deceased
    assert classify(295.5, "lying", False, 30, False, False, False, no_breath_s=10, **kw)[0] == Start.DECEASED
    # same but breathing seen recently -> still EXPECTANT (alive until proven otherwise)
    assert classify(295.5, "lying", False, 30, False, False, False, no_breath_s=2, **kw)[0] == Start.EXPECTANT
    # cold skin IN WATER is never "deceased": possible hypothermia -> IMMEDIATE
    assert classify(295.5, "lying", False, 30, True, False, False, no_breath_s=60, **kw)[0] == Start.IMMEDIATE
    # cool but moving -> alive, DELAYED
    assert classify(298.0, "sitting", True, 30, False, False, False, **kw)[0] == Start.DELAYED
    # proximity to a mine never upgrades a DCD
    assert classify(295.5, "lying", False, 30, False, False, True, no_breath_s=10, **kw)[0] == Start.DECEASED


def test_deceased_never_gets_a_kit(P):
    from skyforge_core.planning.agent import PlannerAgent

    ag = PlannerAgent(P)
    dcd = _cas("C9", (20.0, 0.0), Start.DECEASED)
    dcd.attrs["likely_deceased"] = True
    exp = _cas("C10", (150.0, 0.0), Start.EXPECTANT)
    s = {"pos": (0.0, 0.0, 25.0), "casualties": [dcd, exp], "served_ids": [], "failed_ids": [], "battery_wh": 80.0,
         "wind": (0, 0), "gnss_ok": True, "zones": [], "manual_zones": []}
    res = ag.triage(s)
    assert res.best.id == "C10" and all(c.id != "C9" for c in res.ranked)


def test_sea_state_tracks_a_current_change():
    from skyforge_core.perception.sea_state import SeaStateEstimator

    sea = SeaStateEstimator((0.15, 0.05), k_leeway=0.03, tau_s=30.0)
    wind = (2.0, 0.0)
    t = 0.0
    for _ in range(1200):  # 120 s observing floating objects that drift with a NEW current (0.32, -0.06)
        t += 0.1
        v = (0.32 + 0.03 * wind[0], -0.06)
        sea.update(t, [(v[0], v[1], 0.01), (v[0] + 0.05, v[1] - 0.05, 0.04)], wind)
    assert abs(sea.current[0] - 0.32) < 0.03 and abs(sea.current[1] + 0.06) < 0.03


def test_long_gap_reid_learns_the_drift(P):
    """A floating casualty seen, lost for 120 s, then re-seen far away keeps its id
    and its displacement/time teaches the sea-state estimator the real current."""
    ag = PerceptionAgent(P)
    cam = ag.cam
    rpy = (0.0, 0.0, 0.0)
    true_v = (0.35, -0.05)  # real drift, quite different from the forecast prior
    p0 = (60.0, -50.0)

    def frame(t, drone_xy, obj):
        pos = (drone_xy[0], drone_xy[1], 25.0)
        px = world_to_pixel(obj, pos, rpy, cam)
        d = Detection(Cls.INJURED, 0.8, (px[0] - 6, px[1] - 6, px[0] + 6, px[1] + 6), t, "team",
                      {"temp_k": 308.0, "posture": "lying", "moving": False})
        ag.process([d], pos, rpy, t)

    for k in range(6):  # seen for ~1 s
        frame(0.2 * k, p0, p0)
    tid = ag.confirmed()[0].id
    t = 120.0
    obj = (p0[0] + true_v[0] * t, p0[1] + true_v[1] * t)
    for k in range(3):
        frame(t + 0.2 * k, obj, (obj[0] + true_v[0] * 0.2 * k, obj[1] + true_v[1] * 0.2 * k))
    floats = [x for x in ag.confirmed() if x.cls == Cls.INJURED]
    assert [x.id for x in floats] == [tid]  # same person, no new identity
    assert math.hypot(ag.sea.current[0] - (true_v[0] - 0.0), ag.sea.current[1] - true_v[1]) < \
        math.hypot(P["drift"]["current"][0] - true_v[0], P["drift"]["current"][1] - true_v[1])  # moved toward the truth


def test_mayday_handover_resumes_in_triage(P, tmp_path):
    from skyforge_core.brain.handover import load_snapshot, maybe_resume
    from skyforge_core.brain.state import snapshot

    o1 = Orchestrator(P, engine="fsm")
    served, todo = _cas("C1", (40.0, 10.0), Start.IMMEDIATE), _cas("C2", (80.0, -20.0), Start.DELAYED)
    o1.state.update(casualties=[served, todo], served_ids=["C1"], explore_idx=3)
    path = tmp_path / "mayday_snapshot.json"
    path.write_text(json.dumps({"t": "may", "s": snapshot(o1.state)}), encoding="utf-8")
    assert load_snapshot(str(path))["explore_idx"] == 3

    o2, ag2 = Orchestrator(P, engine="fsm"), PerceptionAgent(P)
    res = maybe_resume(str(path), o2, ag2, 0.0)
    assert res["casualties_to_serve"] == 1 and o2.state["served_ids"] == ["C1"]
    tracks = {t.id: t for t in ag2.confirmed()}
    assert tracks["C1"].attrs.get("marked") and not tracks["C2"].attrs.get("marked")
    s = o2.tick({"t": 0.1, "armed": True, "pos": (0.0, 0.0, 25.0), "battery_wh": 60.0, "casualties": ag2.casualties()})
    assert s["phase"] == Phase.NAVIGATE and s["target"]["id"] == "C2"  # INIT -> TRIAGE -> NAVIGATE, skips the served one


# ----------------------------------------------------- mine layers 2 and 3
def _mag_passes(P, mine=(0.0, 10.0), depth=3.0, lanes=(0.0, 34.0), k=4e4, seed=1):
    import random

    from skyforge_core.perception.magnetic import MagAnomalyDetector

    rng = random.Random(seed)
    det, t, out = MagAnomalyDetector(P), 0.0, []
    for lane_y in lanes:
        for i in range(600):
            t += 0.1
            x = -180 + 0.6 * i
            r2 = (x - mine[0]) ** 2 + (lane_y - mine[1]) ** 2 + (25 + depth) ** 2
            c = det.add_sample(t, (x, lane_y, 25.0), 45000 + k / r2 ** 1.5 + rng.gauss(0, 0.15))
            if c:
                out.append(c)
    return out


def test_magnetometer_two_pass_inversion_and_single_pass_flag(P):
    cs = _mag_passes(P)
    assert cs[0].single_pass and cs[0].sigma > 5.0  # one lane: side/depth ambiguity is declared
    best = cs[-1]
    assert not best.single_pass and math.hypot(best.pos[0], best.pos[1] - 10.0) < 3.0  # two lanes: resolved


def test_mag_contact_on_weapon_is_attributed_not_a_mine(P):
    ag = PerceptionAgent(P)
    w = Track("W1", Cls.WEAPON, (40.0, 75.0), 1.0, confirmed=True)
    from skyforge_core.perception.agent import _TrackState

    ag._tracks["W1"] = _TrackState(w, DriftKF((40.0, 75.0), 1.0, False), 5)
    assert ag.add_mine_contact("mag", (42.0, 76.0), 0.0, 3.0, 0.9, 1.0) is None
    assert not [t for t in ag.tracks() if t.cls == Cls.MINE] and "mag" in w.attrs["layers"]


def test_non_visual_mine_contact_survives_camera_and_confirms_on_repeat(P):
    ag = PerceptionAgent(P)
    tr = ag.add_mine_contact("mag", (5.0, 3.0), 0.3, 4.0, 0.3, 1.0)  # weak, tentative
    assert tr is not None and not tr.confirmed
    for k in range(10):  # the camera looks right at it and sees nothing: must NOT delete it
        ag.process([], (5.0, 3.0, 25.0), (0.0, 0.0, 0.0), 2.0 + 0.2 * k)
    assert any(t.id == tr.id for t in ag.tracks())


def test_sonar_ping_geometry_and_confirmation(P):
    ag = PerceptionAgent(P)
    out = ag.add_sonar_ping((100.0, -45.0), [{"range": 30.0, "bearing": math.pi, "depth": 12.0, "strength": 0.8},
                                             {"range": 20.0, "bearing": 0.0, "depth": 5.0, "strength": 0.3}], 1.0)
    strong, weak = out
    assert strong.pos == pytest.approx((70.0, -45.0), abs=1e-6) and strong.confirmed and strong.attrs["submerged"]
    assert not weak.confirmed  # rock-like weak return never confirms alone
    z = [z for z in ag.zones() if z.id == f"Z{strong.id}"][0]
    assert z.radius >= P["zones"]["safety_submerged_mine"]


def test_jury_station_compatibility_mode(P):
    from skyforge_core.params import deep_merge

    jury_cmd = json.dumps({"type": "goal_update", "goal": {"cmd": "waypoint", "position": [120, 40]}}).encode()
    strict = DroneComm(P, lambda d: True, boot_id=3)
    assert strict.on_receive(jury_cmd, 1.0) == [] and strict.stats["forged_rejected"] == 1  # default: HMAC required
    compat = DroneComm(deep_merge(P, {"comm": {"station_auth": "none"}}), lambda d: True, boot_id=3)
    g = compat.on_receive(jury_cmd, 1.0)
    assert len(g) == 1 and g[0]["type"] == "goto" and g[0]["p"] == [120, 40]
    assert any(e.get("event") == "unsigned_mode" for e in compat.pop_events())  # logged loudly
    w = compat.on_receive(json.dumps({"t": "goal", "g": {"type": "météo", "wind": [3, 1], "visibility": 0.4}}).encode(), 2.0)
    assert w[0]["type"] == "weather"


def test_mission_map_export(P, tmp_path):
    from skyforge_core.mission_map import export_mission_map, track_row
    from skyforge_core.planning.coverage import CoverageGrid

    cov = CoverageGrid(P["mission"]["area"])
    cov.mark_seen((50, 0), 25.0, 80.0, 60.0)
    m = _cas("C1", (40.0, 10.0), Start.IMMEDIATE)
    mine = Track("M1", Cls.MINE, (60.0, -40.0), 2.0, confirmed=True, attrs={"layers": ["sonar"], "depth": 12.0})
    out = export_mission_map(str(tmp_path), P["mission"]["area"], [track_row(m, ["C1"]), track_row(mine)],
                             [{"id": "ZM1", "cls": "mine", "center": [60, -40], "radius": 15.0}], cov, [(0, 0), (50, 0)], ["C1"],
                             P["water"]["polygons"], "map")
    doc = json.loads(open(out["json"], encoding="utf-8").read())
    assert doc["objects"][0]["served"] and doc["objects"][1]["layers"] == ["sonar"] and 0 < doc["coverage"] < 1
    assert out["png"] and os.path.exists(out["png"])

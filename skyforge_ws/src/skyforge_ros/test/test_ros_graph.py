"""Closed-loop test of the REAL skyforge_ros nodes on a fake rclpy bus.

world (2D sim) -> /odom, /gnss/fix, /vision/* JSON -> all nodes -> /cmd_vel -> world
plus a real UDP ground station on localhost talking to comm_node.
Run without ROS 2:  pytest skyforge_ws/src/skyforge_ros/test -q
"""
import json
import math
import os
import socket
import sys

import pytest

HERE = os.path.dirname(__file__)
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.abspath(os.path.join(HERE, "..")))
sys.path.insert(0, os.path.abspath(os.path.join(HERE, "..", "..", "skyforge_core")))

import fake_ros  # noqa: E402

fake_ros.install()

from fake_ros import Auto, Bus, Clock, Node, run_timers  # noqa: E402
from skyforge_core.comm.station import GroundStation  # noqa: E402
from skyforge_core.params import deep_merge, load_params  # noqa: E402
from skyforge_core.sim.world import DEFAULT_SCENARIO, World  # noqa: E402


def _free_port():
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    s.bind(("127.0.0.1", 0))
    p = s.getsockname()[1]
    s.close()
    return p


@pytest.fixture
def cfg(tmp_path):
    import yaml

    p = load_params()
    p = deep_merge(p, DEFAULT_SCENARIO["params"])
    p["obstacles"] = DEFAULT_SCENARIO["obstacles"]
    p["comm"]["udp"].update({"drone_port": _free_port(), "station_port": _free_port()})
    p["logging"]["dir"] = str(tmp_path / "logs")
    path = tmp_path / "skyforge.yaml"
    path.write_text(yaml.safe_dump({"skyforge": p}), encoding="utf-8")
    os.environ["SKYFORGE_CONFIG"] = str(path)
    return p


def test_full_graph_closed_loop(cfg):
    Bus.reset()
    Clock.t = 0.0
    Node.overrides = {"reflex_node": {"backend": "twist"}, "fault_injector": {"gnss_source": "navsatfix"},
                      "comm_node": {"mode": "local"}}
    from skyforge_ros import comm_node, fault_injector, monitor_nodes, orchestrator_node, perception_agent_node, reflex_node, viz_node

    nodes = [reflex_node.ReflexNode(), orchestrator_node.OrchestratorNode(), perception_agent_node.PerceptionAgentNode(),
             comm_node.CommNode(), monitor_nodes.HealthNode(), monitor_nodes.NavIntegrityNode(), monitor_nodes.MetricsNode(),
             fault_injector.FaultInjector(), viz_node.VizNode()]

    world = World(cfg, DEFAULT_SCENARIO)
    from skyforge_core.perception.projection import Camera

    cam = Camera.from_params(cfg)
    st = {"pos": [0.0, 0.0, 0.0], "vel": [0.0, 0.0, 0.0], "cmd": (0.0, 0.0, 0.0), "yaw": 0.0}
    Bus.subs["/cmd_vel"].append(lambda m: st.__setitem__("cmd", (m.linear.x, m.linear.y, m.linear.z)))
    releases, events = [], []
    Bus.subs["/payload/release"].append(lambda m: releases.append(json.loads(m.data)))
    Bus.subs["/mission/events"].append(lambda m: events.append((m.type, m.msg)))

    # real UDP ground station
    udp = cfg["comm"]["udp"]
    ssock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    ssock.bind(("127.0.0.1", udp["station_port"]))
    ssock.setblocking(False)
    gs = GroundStation(cfg, lambda d: ssock.sendto(d, ("127.0.0.1", udp["drone_port"])) or True, boot_id=77)
    sent = {"goal": False, "forge": False, "jam": False, "spoof": False}

    def hook(t):
        # physics 50 Hz
        dt = 0.02
        world.step(dt)
        for i in range(3):
            st["vel"][i] += (st["cmd"][i] - st["vel"][i]) / 0.3 * dt
            st["pos"][i] += st["vel"][i] * dt
        st["pos"][2] = max(0.0, st["pos"][2])
        if math.hypot(st["vel"][0], st["vel"][1]) > 0.5:
            st["yaw"] = math.atan2(st["vel"][1], st["vel"][0])
        o = Auto()
        o.header.stamp = Auto(sec=int(t), nanosec=int((t % 1) * 1e9))
        o.pose.pose.position = Auto(x=st["pos"][0], y=st["pos"][1], z=st["pos"][2])
        o.pose.pose.orientation = Auto(x=0.0, y=0.0, z=math.sin(st["yaw"] / 2), w=math.cos(st["yaw"] / 2))
        o.twist.twist.linear = Auto(x=st["vel"][0], y=st["vel"][1], z=st["vel"][2])
        o.twist.twist.angular = Auto(x=0.0, y=0.0, z=0.0)
        for cb in Bus.subs["/odom"]:
            cb(o)
        k = int(round(t / dt))
        if k % 10 == 0:  # GNSS 5 Hz + camera 5 Hz
            lat0, lon0 = 36.8065, 10.1815
            fix = Auto(latitude=lat0 + math.degrees(st["pos"][1] / 6378137.0),
                       longitude=lon0 + math.degrees(st["pos"][0] / (6378137.0 * math.cos(math.radians(lat0)))),
                       altitude=st["pos"][2], status=Auto(status=0))
            for cb in Bus.subs["/gnss/fix"]:
                cb(fix)
            jury, team = world.camera_frame(st["pos"], (0.0, 0.0, st["yaw"]), cam)
            jury["stamp"] = team["stamp"] = t
            for topic, payload in (("/vision/jury", jury), ("/vision/team", team)):
                for cb in Bus.subs[topic]:
                    cb(Auto(data=json.dumps(payload)))
        # station side
        while True:
            try:
                data, _ = ssock.recvfrom(65535)
            except (BlockingIOError, OSError):
                break
            gs.on_receive(data, t)
        gs.tick(t)
        cmd = lambda d: [cb(Auto(data=json.dumps(d))) for cb in Bus.subs["/skyforge/fault_cmd"]]
        if t >= 60 and not sent["jam"]:
            sent["jam"] = True
            cmd({"type": "jam", "on": True, "dur_s": 15})
        if t >= 100 and not sent["goal"]:
            sent["goal"] = True
            gs.send_goal({"type": "zone_add", "zone": {"id": "NOFLY1", "center": [120, -15], "radius": 8}}, t)
        if t >= 105 and not sent["forge"]:
            sent["forge"] = True
            gs.send_forged_goal({"type": "rtb"}, t)
        if t >= 115 and not sent["spoof"]:
            sent["spoof"] = True
            cmd({"type": "spoof", "dx": 40, "dy": 0, "ramp_s": 8, "dur_s": 20})

    nav_states = []
    Bus.subs["/skyforge/nav"].append(lambda m: nav_states.append(json.loads(m.data)["gnss_ok"]))
    run_timers(260.0, 0.02, hook)
    ssock.close()

    orch = nodes[1].orch
    phases = [m for t_, m in events if t_ == "phase"]
    msgs = [m for _, m in events]
    assert any("EXPLORER" in m for m in phases), phases[:5]
    assert releases, "no kit released through /payload/release"
    assert any("zone_add accepted" in m for m in msgs), "goal_update did not reach the orchestrator"
    assert any("forged_rejected" in m for m in msgs), "forged command not rejected"
    assert False in nav_states, "spoofing never detected by nav_integrity_node"
    assert gs.stats["rx"] > 50 and gs.detections, "station got no telemetry/detections over UDP"
    assert gs.resync_times, "no resync after the jam"
    assert orch.latency.pct(95) < 100.0
    mk = Bus.last.get("/viz/markers")
    assert mk is not None and any(m.ns == "tracks" for m in mk.markers) and any(m.ns == "zones" for m in mk.markers)
    lat = Bus.last.get("/metrics")
    assert lat is not None
    print(f"\nphase {orch.phase.value}, releases {len(releases)}, station rx {gs.stats['rx']}, "
          f"resync {gs.resync_times}, decision p95 {orch.latency.pct(95):.2f} ms")

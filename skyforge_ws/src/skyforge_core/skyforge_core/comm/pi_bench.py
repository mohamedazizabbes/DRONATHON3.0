"""F12 bonus benchmark: run ON the Raspberry Pi to measure the comm module.

Runs the real CommBridge with a brain emulator (10 Hz telemetry + detections
and decisions) and a ground station over real UDP sockets, forces a radio
outage, and reports what the spec asks for (F12: memory, latency,
synchronisation):

  RAM (RSS), CPU %, per-message processing latency p50/p95/p99,
  round-trip time, resync time after the outage, messages buffered/replayed,
  max message size, nothing lost.

  python3 -m skyforge_core.comm.pi_bench --duration 60 --outage-at 10 --outage 30
"""
from __future__ import annotations

import argparse
import json
import platform
import socket
import time

from ..params import load_params
from .bridge import CommBridge
from .station import GroundStation


def pct(xs, q):
    if not xs:
        return 0.0
    s = sorted(xs)
    return s[min(len(s) - 1, int(q / 100.0 * len(s)))]


def run(duration: float = 60.0, outage_at: float = 10.0, outage: float = 30.0, rate_hz: float = 10.0, config=None) -> dict:
    params = load_params(config)
    a = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)  # bridge <-> station link
    b = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    a.bind(("127.0.0.1", 0))
    b.bind(("127.0.0.1", 0))
    a.setblocking(False)
    b.setblocking(False)
    addr_a, addr_b = a.getsockname(), b.getsockname()
    goals = []
    bridge = CommBridge(params, lambda d: a.sendto(d, addr_b) or True,
                        lambda o: goals.append(o) if o.get("op") == "goal" else None)
    bridge.link.shadows = []
    bridge.link.max_range = 1e9
    station = GroundStation(params, lambda d: b.sendto(d, addr_a) or True, boot_id=5)
    t0 = time.time()
    k = 0
    period = 1.0 / rate_hz
    outage_on = False
    restored_t = None
    resync_s = None
    rss_max = 0.0
    cpu = []
    next_t = t0
    while True:
        now = time.time()
        el = now - t0
        if el >= duration:
            break
        # --- outage control
        if not outage_on and outage_at <= el < outage_at + outage:
            outage_on = True
            bridge.on_op({"op": "jam", "on": True})
        elif outage_on and el >= outage_at + outage:
            outage_on = False
            bridge.on_op({"op": "jam", "on": False})
            restored_t = now
        # --- brain traffic
        if now >= next_t:
            next_t += period
            k += 1
            bridge.on_op({"op": "send", "t": "tel", "body": {"p": [k * 0.1, 2.0, 25.0], "ph": "EXPLORE", "bat": 80.0, "k": k}})
            if k % 20 == 0:
                bridge.on_op({"op": "send", "t": "det", "body": {"id": f"C{k}", "c": "injured", "u": "IMM", "p": [k, 3, 0], "cf": 0.8}})
            if k % 25 == 0:
                bridge.on_op({"op": "send", "t": "dec", "body": {"d": "triage", "m": f"TRIAGE -> C{k}"}})
            if k % 50 == 0:
                bridge.on_op({"op": "snapshot", "body": {"pos": [k, 0, 25], "cas": []}})
                station.send_goal({"type": "goto", "p": [10, k % 100]}, now)
        # --- pump sockets
        for sock, fn in ((b, lambda d: station.on_receive(d, time.time())), (a, bridge.on_station)):
            while True:
                try:
                    d, _ = sock.recvfrom(65535)
                except (BlockingIOError, OSError):
                    break
                fn(d)
        station.tick(now)
        bridge._update_link(now)
        if restored_t is not None and resync_s is None and station.contig >= bridge.comm.seq:
            resync_s = time.time() - restored_t
        if k % 10 == 0:
            rss_max = max(rss_max, bridge.meter.rss_mb())
            cpu.append(bridge.meter.cpu_pct())
        time.sleep(0.002)
    m = bridge.metrics()
    out = {
        "host": platform.node(), "machine": platform.machine(), "python": platform.python_version(),
        "duration_s": duration, "outage_s": outage, "msgs_sent": m["sent"],
        "rss_mb_max": round(rss_max, 2), "cpu_pct_mean": round(sum(cpu) / max(1, len(cpu)), 1),
        "proc_p50_ms": round(pct(bridge.proc_ms, 50), 3), "proc_p95_ms": round(pct(bridge.proc_ms, 95), 3),
        "proc_p99_ms": round(pct(bridge.proc_ms, 99), 3),
        "rtt_p50_ms": round(pct(bridge.rtt_ms, 50), 2), "rtt_p95_ms": round(pct(bridge.rtt_ms, 95), 2),
        "resync_s": round(resync_s, 3) if resync_s is not None else None,
        "buffered_during_outage": m["buffered_while_down"], "replayed": m["replayed"], "syncs": m["syncs"],
        "max_msg_bytes": m["max_bytes"], "mean_msg_bytes": m["mean_bytes"],
        "ring_capacity": m["buffer_capacity"], "ring_max_occupancy": m["ring_max_occupancy"],
        "station_contiguous_seq": station.contig, "drone_last_seq": bridge.comm.seq,
        "lost": bridge.comm.seq - station.contig, "goals_delivered": len(goals),
    }
    a.close()
    b.close()
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description="SKYFORGE comm bridge benchmark (F12)")
    ap.add_argument("--duration", type=float, default=60.0)
    ap.add_argument("--outage-at", type=float, default=10.0)
    ap.add_argument("--outage", type=float, default=30.0)
    ap.add_argument("--rate", type=float, default=10.0)
    ap.add_argument("--config", default=None)
    ap.add_argument("--out", default="pi_bench.json")
    a = ap.parse_args(argv)
    r = run(a.duration, a.outage_at, a.outage, a.rate, a.config)
    with open(a.out, "w", encoding="utf-8") as f:
        json.dump(r, f, indent=1)
    print("| metric | value |\n|---|---|")
    for k, v in r.items():
        print(f"| {k} | {v} |")


if __name__ == "__main__":
    main()

"""Comm bridge: the Communication module as a standalone process (F12 bonus).

Runs on the Raspberry Pi (or in-process inside the ROS comm_node). It sits
between the brain and the ground station:

    brain --(internal UDP JSON ops)--> [CommBridge on Pi] --(signed JSON)--> station
    brain <--(verified goal_updates)-- [CommBridge on Pi] <--(signed JSON)-- station

Internal ops (brain -> bridge), one JSON object per datagram:
    {"op":"send","t":"tel|det|dec|may","body":{...}}
    {"op":"snapshot","body":{...}}         latest state snapshot for sync
    {"op":"pos","p":[x,y]}                 drone position for the link simulator
    {"op":"jam","on":true}                 fault injector
Bridge -> brain:
    {"op":"goal","g":{...}}                verified, fresh goal_update
    {"op":"event","e":{...}}               security / link events for mission_logs
    {"op":"stats","s":{...}}               metrics, once per second

Only the Python standard library is required (psutil is used when present).
Run:  python3 -m skyforge_core.comm.bridge --config skyforge.yaml
"""
from __future__ import annotations

import argparse
import json
import os
import socket
import time
from typing import Callable, Optional

from ..params import load_params
from .drone_comm import DroneComm
from .link import LinkSimulator


class ResourceMeter:
    """RAM (RSS) and CPU % of this process: psutil if available, else /proc."""

    def __init__(self):
        try:
            import psutil  # type: ignore

            self._ps = psutil.Process(os.getpid())
            self._ps.cpu_percent(None)
        except Exception:  # noqa: BLE001
            self._ps = None
        self._t0 = time.monotonic()
        self._c0 = sum(os.times()[:2])

    def rss_mb(self) -> float:
        if self._ps is not None:
            return self._ps.memory_info().rss / 1e6
        try:
            with open("/proc/self/status", encoding="ascii") as f:
                for line in f:
                    if line.startswith("VmRSS:"):
                        return int(line.split()[1]) / 1e3
        except OSError:
            pass
        try:
            import resource  # type: ignore

            return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e3
        except Exception:  # noqa: BLE001
            return -1.0

    def cpu_pct(self) -> float:
        if self._ps is not None:
            return self._ps.cpu_percent(None)
        t, c = time.monotonic(), sum(os.times()[:2])
        pct = 100.0 * (c - self._c0) / max(1e-6, t - self._t0)
        self._t0, self._c0 = t, c
        return pct


class CommBridge:
    """Transport-agnostic core: feed it ops and station datagrams."""

    def __init__(self, params: dict, to_station: Callable[[bytes], bool], to_brain: Callable[[dict], None],
                 link: Optional[LinkSimulator] = None, clock: Callable[[], float] = time.time):
        self.params = params
        self.clock = clock
        self.to_brain = to_brain
        self.link = link or LinkSimulator(params)
        self.pos = tuple(params["mission"]["home"])
        self._snapshot: dict = {}
        self._to_station = to_station
        self.comm = DroneComm(params, self._tx, snapshot_fn=lambda: self._snapshot)
        self.meter = ResourceMeter()
        self.proc_ms: list[float] = []
        self.sent_at: dict[int, float] = {}
        self.rtt_ms: list[float] = []
        self._resync_start: Optional[float] = None
        self.resync_s: list[float] = []

    def _tx(self, data: bytes) -> bool:
        if not self.link.deliver(self.pos):
            return False
        return bool(self._to_station(data))

    # ------------------------------------------------------------ brain side
    def on_op(self, op: dict) -> None:
        t0 = time.perf_counter_ns()
        now = self.clock()
        kind = op.get("op")
        if kind == "send":
            seqs = self.comm.send(op["t"], op.get("body", {}), now)
            for s in seqs:
                self.sent_at[s] = now
            if len(self.sent_at) > 4096:
                for k in sorted(self.sent_at)[:2048]:
                    del self.sent_at[k]
        elif kind == "snapshot":
            self._snapshot = op.get("body", {})
        elif kind == "pos":
            self.pos = tuple(op["p"][:2])
        elif kind == "jam":
            self.link.set_jammed(bool(op.get("on")))
        self._update_link(now)
        self.proc_ms.append((time.perf_counter_ns() - t0) / 1e6)
        self.proc_ms = self.proc_ms[-2000:]

    def _update_link(self, now: float) -> None:
        up = self.link.is_up(self.pos)
        if up != self.comm.link_up:
            self.comm.set_link(up, now)
            if up:
                self._resync_start = now
                # heartbeat so the station notices us and asks for the gap
                self.comm.send("tel", {"hb": 1, "p": list(self.pos)}, now)
        for e in self.comm.pop_events():
            self.to_brain({"op": "event", "e": e})

    # ---------------------------------------------------------- station side
    def on_station(self, data: bytes) -> None:
        t0 = time.perf_counter_ns()
        now = self.clock()
        if not self.link.deliver(self.pos):
            return  # the radio did not hear it
        before = len(self.comm.buffer)
        goals = self.comm.on_receive(data, now)
        for g in goals:
            self.to_brain({"op": "goal", "g": g})
        try:
            msg = json.loads(data)
            if msg.get("t") == "ack" and int(msg.get("last", 0)) in self.sent_at:
                self.rtt_ms.append((now - self.sent_at[int(msg["last"])]) * 1000.0)
                self.rtt_ms = self.rtt_ms[-500:]
        except (ValueError, TypeError):
            pass
        if self._resync_start is not None and before and not any(
            not v[2] for v in self.comm.buffer.items.values()
        ):
            self.resync_s.append(now - self._resync_start)
            self._resync_start = None
        for e in self.comm.pop_events():
            self.to_brain({"op": "event", "e": e})
        self.proc_ms.append((time.perf_counter_ns() - t0) / 1e6)

    # --------------------------------------------------------------- metrics
    def metrics(self) -> dict:
        def pct(xs, q):
            if not xs:
                return 0.0
            s = sorted(xs)
            return s[min(len(s) - 1, int(q / 100.0 * len(s)))]

        return {
            "rss_mb": round(self.meter.rss_mb(), 2),
            "cpu_pct": round(self.meter.cpu_pct(), 1),
            "proc_p50_ms": round(pct(self.proc_ms, 50), 3),
            "proc_p95_ms": round(pct(self.proc_ms, 95), 3),
            "rtt_p50_ms": round(pct(self.rtt_ms, 50), 1),
            "rtt_p95_ms": round(pct(self.rtt_ms, 95), 1),
            "resync_s_last": round(self.resync_s[-1], 2) if self.resync_s else None,
            "link_up": self.comm.link_up,
            **self.comm.summary(),
        }


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description="SKYFORGE comm bridge (Raspberry Pi / standalone)")
    ap.add_argument("--config", default=None, help="skyforge.yaml")
    ap.add_argument("--brain-listen", default="0.0.0.0:14700", help="ops from the brain")
    ap.add_argument("--brain-reply", default="127.0.0.1:14701", help="where to send goals/events/stats")
    ap.add_argument("--station-listen", default=None, help="host:port for station datagrams (default comm.udp.drone_port)")
    ap.add_argument("--station", default=None, help="station host:port (default from config)")
    ap.add_argument("--metrics-file", default=None, help="append metrics JSON lines here")
    a = ap.parse_args(argv)

    params = load_params(a.config)
    udp = params["comm"]["udp"]
    st_addr = _hp(a.station or f"{udp['station_host']}:{udp['station_port']}")
    br_reply = _hp(a.brain_reply)

    s_station = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    s_station.bind(_hp(a.station_listen or f"0.0.0.0:{udp['drone_port']}"))
    s_brain = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    s_brain.bind(_hp(a.brain_listen))
    for s in (s_station, s_brain):
        s.setblocking(False)

    def to_station(data: bytes) -> bool:
        try:
            s_station.sendto(data, st_addr)
            return True
        except OSError:
            return False

    def to_brain(obj: dict) -> None:
        try:
            s_brain.sendto(json.dumps(obj, separators=(",", ":")).encode(), br_reply)
        except OSError:
            pass

    bridge = CommBridge(params, to_station, to_brain)
    import selectors

    sel = selectors.DefaultSelector()
    sel.register(s_station, selectors.EVENT_READ, "station")
    sel.register(s_brain, selectors.EVENT_READ, "brain")
    last_stats = 0.0
    mf = open(a.metrics_file, "a", encoding="utf-8") if a.metrics_file else None
    print(f"[skyforge-bridge] station={st_addr} brain_reply={br_reply} ring={bridge.comm.buffer.capacity}", flush=True)
    while True:
        for key, _ in sel.select(timeout=0.1):
            sock = key.fileobj
            while True:
                try:
                    data, _addr = sock.recvfrom(65535)
                except BlockingIOError:
                    break
                if key.data == "station":
                    bridge.on_station(data)
                else:
                    try:
                        bridge.on_op(json.loads(data))
                    except (ValueError, KeyError) as e:
                        print(f"[skyforge-bridge] bad op: {e}", flush=True)
        now = time.time()
        if now - last_stats >= 1.0:
            last_stats = now
            m = bridge.metrics()
            to_brain({"op": "stats", "s": m})
            if mf:
                mf.write(json.dumps({"ts": now, **m}) + "\n")
                mf.flush()


def _hp(s: str):
    h, p = s.rsplit(":", 1)
    return (h, int(p))


if __name__ == "__main__":
    main()

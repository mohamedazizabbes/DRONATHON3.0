"""Drone-side communication module (no agent): JSON codec, HMAC, ring buffer,
autonomous buffering during outages and resynchronisation on reconnect.

Transport-agnostic: `transmit(bytes)` is injected (UDP socket on the ROS
node / Raspberry Pi, a function call in the simulator).
"""
from __future__ import annotations

import random
import time
from typing import Callable, Optional

from .protocol import GOAL_TYPES, PRIORITY, Codec, ForgedMessage, Reassembler, ReplayGuard, normalize_goal
from .ring_buffer import RingBuffer


class DroneComm:
    def __init__(self, params: dict, transmit: Callable[[bytes], bool], snapshot_fn: Optional[Callable[[], dict]] = None,
                 boot_id: Optional[int] = None):
        self.p = params["comm"]
        self.codec = Codec.from_params(params, boot_id if boot_id is not None else random.randint(1, 2**31 - 1))
        self.buffer = RingBuffer(int(self.p["ring_capacity"]))
        self.transmit = transmit
        self.snapshot_fn = snapshot_fn
        self.seq = 0
        self.link_up = True
        self.guard = ReplayGuard()
        self.reasm = Reassembler()
        self._fid = 0
        self.events: list[dict] = []  # comm events for mission_logs
        self.stats = {
            "sent": 0, "tx_ok": 0, "tx_fail": 0, "buffered_while_down": 0, "replayed": 0, "syncs": 0,
            "forged_rejected": 0, "replay_rejected": 0, "max_bytes": 0, "sum_bytes": 0, "goals_accepted": 0,
            "outages": 0, "lost_tel": 0,
        }
        self.outage_start: Optional[float] = None
        self.last_outage_s = 0.0

    # ------------------------------------------------------------- outgoing
    def _next_seq(self) -> int:
        self.seq += 1
        return self.seq

    def send(self, t: str, body: dict, now: float) -> list[int]:
        prio = PRIORITY.get(t, 1)
        bodies = [(t, body)]
        if not self.codec.fits(t, body):
            self._fid += 1
            bodies = [("frag", fb) for fb in self.codec.fragment(t, body, self._fid)]
        seqs = []
        for tt, b in bodies:
            seq = self._next_seq()
            data = self.codec.encode(tt, b, seq, now)
            self.stats["sent"] += 1
            self.stats["max_bytes"] = max(self.stats["max_bytes"], len(data))
            self.stats["sum_bytes"] += len(data)
            ok = self._tx(data) if self.link_up else False
            if not self.link_up:
                self.stats["buffered_while_down"] += 1
            self.buffer.push(seq, prio, data, ok)
            seqs.append(seq)
        return seqs

    def _tx(self, data: bytes) -> bool:
        try:
            ok = bool(self.transmit(data))
        except OSError:
            ok = False
        self.stats["tx_ok" if ok else "tx_fail"] += 1
        return ok

    def set_link(self, up: bool, now: float) -> None:
        if up == self.link_up:
            return
        self.link_up = up
        if not up:
            self.outage_start = now
            self.stats["outages"] += 1
            self.events.append({"ts": now, "type": "link", "msg": "comm: link down, buffering"})
        else:
            self.last_outage_s = now - (self.outage_start or now)
            # proactive replay of everything that never left (in order); a station that does
            # not implement our ack protocol (jury station) still gets the outage's messages
            for seq, (prio, data, tx) in list(self.buffer.items.items()):
                if not tx and self._tx(data):
                    self.buffer.mark_transmitted(seq)
                    self.buffer.mark_replayed(seq, now)
                    self.stats["replayed"] += 1
            self.events.append({"ts": now, "type": "link", "msg": f"comm: link up after {self.last_outage_s:.1f}s, "
                                f"{len(self.buffer)} msgs buffered"})

    # ------------------------------------------------------------- incoming
    def on_receive(self, data: bytes, now: float) -> list[dict]:
        """Returns verified goal_update bodies for the orchestrator."""
        compat = self.p.get("station_auth", "hmac") == "none"
        try:
            msg = self.codec.decode(data)
        except ForgedMessage as e:
            if not compat:
                self.stats["forged_rejected"] += 1
                self.events.append({"ts": now, "type": "security", "msg": f"forged_rejected: {e}", "event": "forged_rejected"})
                return []
            try:  # jury station without our key: accept unsigned (forgery protection OFF)
                msg = self.codec.decode_unverified(data)
            except ForgedMessage:
                return []
            self.stats["unsigned_accepted"] = self.stats.get("unsigned_accepted", 0) + 1
            if self.stats["unsigned_accepted"] == 1:
                self.events.append({"ts": now, "type": "security", "event": "unsigned_mode",
                                    "msg": "WARNING: unsigned station message accepted (station_auth=none, forgery protection OFF)"})
        if "seq" in msg and "b" in msg and not self.guard.check(msg):
            self.stats["replay_rejected"] += 1
            self.events.append({"ts": now, "type": "security", "msg": f"replay_rejected seq {msg.get('seq')}",
                                "event": "replay_rejected"})
            return []
        t = msg.get("t")
        if t == "ack":
            self._on_ack(int(msg["last"]), int(msg.get("max", msg["last"])), now)
            return []
        if t in GOAL_TYPES or (compat and t is None and ("type" in msg or "goal" in msg)):
            g = normalize_goal(msg)
            g["_recv_t"] = now
            g["_seq"] = msg.get("seq")
            self.stats["goals_accepted"] += 1
            return [g]
        return []

    def _on_ack(self, last: int, station_max: int, now: float) -> None:
        self.buffer.ack(last)
        if not self.link_up:
            return
        lost = self.buffer.lost_after(last)
        if lost:
            # Gap overwritten: send a state snapshot plus the count of lost messages.
            self.stats["syncs"] += 1
            self.stats["lost_tel"] += lost.get(0, 0)
            body = {"lost": {str(k): v for k, v in lost.items()}, "s": self.snapshot_fn() if self.snapshot_fn else {}}
            self.buffer.forget_dropped_before(self.seq)
            self.send("sync", body, now)
        for seq, data in self.buffer.to_replay(last, station_max, now):
            self.buffer.mark_replayed(seq, now)
            if self._tx(data):
                self.buffer.mark_transmitted(seq)
                self.stats["replayed"] += 1

    def pop_events(self) -> list[dict]:
        out, self.events = self.events, []
        return out

    def summary(self) -> dict:
        s = dict(self.stats)
        s["mean_bytes"] = round(s["sum_bytes"] / max(1, s["sent"]), 1)
        s["buffer_occupancy"] = len(self.buffer)
        s["buffer_capacity"] = self.buffer.capacity
        s.update({f"ring_{k}": v for k, v in self.buffer.stats.items()})
        return s


def now_s() -> float:
    return time.time()

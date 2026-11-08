"""Ground station logic: verifies, de-duplicates and re-orders drone messages,
acks (last contiguous seq + max seen), reassembles fragments, keeps the
mirrored mission_logs, sends signed goal_updates and dispatches drone 2 on
MAYDAY. Also offers forged / replayed command injection for the demo.
"""
from __future__ import annotations

import random
from typing import Callable, Optional

from .protocol import Codec, ForgedMessage, Reassembler, ReplayGuard


class GroundStation:
    def __init__(self, params: dict, transmit: Callable[[bytes], bool], on_mayday: Optional[Callable[[dict], None]] = None,
                 boot_id: Optional[int] = None):
        self.p = params["comm"]
        self.codec = Codec.from_params(params, boot_id if boot_id is not None else random.randint(1, 2**31 - 1))
        self.transmit = transmit
        self.on_mayday = on_mayday
        self.seq = 0
        self.drone_b: Optional[int] = None
        self.contig = 0
        self.max_seen = 0
        self.received: set[int] = set()
        self.reasm = Reassembler()
        self.guard_ts = ReplayGuard()
        self.last_tel: dict = {}
        self.detections: dict[str, dict] = {}
        self.decisions: list[dict] = []
        self.logs: list[dict] = []
        self.mayday: Optional[dict] = None
        self.last_sync: Optional[dict] = None
        self._last_ack_t = -1e9
        self._gap_since: Optional[float] = None
        self.resync_times: list[float] = []
        self.last_goal_bytes: Optional[bytes] = None
        self.stats = {"rx": 0, "dup": 0, "forged": 0, "gaps": 0, "frags": 0}

    # ---------------------------------------------------------------- rx
    def on_receive(self, data: bytes, now: float) -> None:
        try:
            msg = self.codec.decode(data)
        except ForgedMessage:
            self.stats["forged"] += 1
            return
        self.stats["rx"] += 1
        b, seq = msg["b"], int(msg["seq"])
        if b != self.drone_b:
            self.drone_b, self.contig, self.max_seen, self.received = b, seq - 1, seq - 1, set()
        if seq <= self.contig or seq in self.received:
            self.stats["dup"] += 1
            return
        self.received.add(seq)
        self.max_seen = max(self.max_seen, seq)
        while self.contig + 1 in self.received:
            self.contig += 1
            self.received.discard(self.contig)
        if self.max_seen > self.contig:
            if self._gap_since is None:
                self._gap_since = now
                self.stats["gaps"] += 1
                self.send_ack(now)  # ask for the gap right away
            elif now - self._last_ack_t >= 0.5:
                self.send_ack(now)  # still missing: ask again, rate-limited
        elif self._gap_since is not None:
            self.resync_times.append(now - self._gap_since)
            self.logs.append({"ts": now, "type": "resync", "msg": f"resynced in {now - self._gap_since:.2f}s"})
            self._gap_since = None
        self._handle(msg, now)

    def _handle(self, msg: dict, now: float) -> None:
        t = msg["t"]
        if t == "frag":
            self.stats["frags"] += 1
            done = self.reasm.add(msg["b"], msg)
            if done is None:
                return
            t, body = done
            msg = {**body, "t": t, "seq": msg["seq"], "ts": msg["ts"]}
        if t == "tel":
            self.last_tel = msg
        elif t == "det":
            self.detections[msg.get("id", str(msg["seq"]))] = msg
        elif t == "dec":
            self.decisions.append(msg)
        elif t == "sync":
            self.last_sync = msg
            self.logs.append({"ts": now, "type": "sync", "msg": f"sync snapshot, lost {msg.get('lost')}"})
        elif t == "may":
            self.mayday = msg
            self.logs.append({"ts": now, "type": "may", "msg": "MAYDAY received -> dispatching drone 2"})
            if self.on_mayday:
                self.on_mayday(msg)
        self.logs.append({"ts": msg.get("ts"), "seq": msg.get("seq"), "type": t, "msg": msg.get("m", "")})

    # ---------------------------------------------------------------- tx
    def _next(self) -> int:
        self.seq += 1
        return self.seq

    def send_ack(self, now: float) -> None:
        data = self.codec.encode("ack", {"last": self.contig, "max": self.max_seen}, self._next(), now)
        self.transmit(data)
        self._last_ack_t = now

    def tick(self, now: float) -> None:
        if now - self._last_ack_t >= float(self.p["ack_period_s"]):
            self.send_ack(now)

    def send_goal(self, goal: dict, now: float) -> bytes:
        data = self.codec.encode("goal", {"g": goal}, self._next(), now)
        self.last_goal_bytes = data
        self.transmit(data)
        return data

    # ------------------------------------------------- demo attack helpers
    def send_forged_goal(self, goal: dict, now: float) -> bytes:
        evil = Codec(b"not-the-shared-key", self.codec.version, self.codec.max_bytes, self.codec.boot_id)
        data = evil.encode("goal", {"g": goal}, self._next(), now)
        self.transmit(data)
        return data

    def replay_last_goal(self) -> Optional[bytes]:
        if self.last_goal_bytes:
            self.transmit(self.last_goal_bytes)
        return self.last_goal_bytes

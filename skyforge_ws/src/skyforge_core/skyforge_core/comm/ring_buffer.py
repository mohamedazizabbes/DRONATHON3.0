"""Priority ring buffer, declared capacity 512 messages (< 512 KB worst case).

Every outgoing message is stored until the station acknowledges it. When
full, eviction order is: oldest telemetry first, then oldest det/dec, and
critical messages (goal, ack, sync, may) only as a last resort. Evicted
sequence numbers are remembered so a reconnect can tell "replay the gap"
from "gap overwritten -> send a sync snapshot".
"""
from __future__ import annotations

from collections import OrderedDict


class RingBuffer:
    def __init__(self, capacity: int = 512):
        self.capacity = capacity
        self.items: "OrderedDict[int, list]" = OrderedDict()  # seq -> [prio, data, transmitted]
        self.dropped: dict[int, int] = {}  # seq -> prio
        self.replayed_at: dict[int, float] = {}
        self.stats = {"pushed": 0, "evicted_tel": 0, "evicted_high": 0, "evicted_critical": 0, "max_occupancy": 0}

    def __len__(self) -> int:
        return len(self.items)

    @property
    def nbytes(self) -> int:
        return sum(len(v[1]) for v in self.items.values())

    def push(self, seq: int, prio: int, data: bytes, transmitted: bool) -> None:
        while len(self.items) >= self.capacity:
            self._evict()
        self.items[seq] = [prio, data, transmitted]
        self.stats["pushed"] += 1
        self.stats["max_occupancy"] = max(self.stats["max_occupancy"], len(self.items))

    def _evict(self) -> None:
        for p, name in ((0, "evicted_tel"), (1, "evicted_high"), (2, "evicted_critical")):
            for seq, (prio, _, _) in self.items.items():
                if prio == p:
                    del self.items[seq]
                    self.dropped[seq] = prio
                    self.stats[name] += 1
                    return

    def ack(self, last_seq: int) -> int:
        n = 0
        for seq in list(self.items.keys()):
            if seq <= last_seq:
                del self.items[seq]
                n += 1
            else:
                break
        for seq in [s for s in self.dropped if s <= last_seq]:
            del self.dropped[seq]
        return n

    def lost_after(self, last_seq: int) -> dict[int, int]:
        """Messages after `last_seq` that were overwritten (count per priority)."""
        out: dict[int, int] = {}
        for seq, prio in self.dropped.items():
            if seq > last_seq:
                out[prio] = out.get(prio, 0) + 1
        return out

    def to_replay(self, last_seq: int, station_max: int, now: float = 0.0, debounce_s: float = 1.0) -> list[tuple[int, bytes]]:
        """Messages the station is missing: in the gap, or never transmitted.
        A message replayed less than `debounce_s` ago is not replayed again
        (prevents replay storms when several acks cross in flight)."""
        out = []
        for seq, (prio, data, tx) in self.items.items():
            if seq > last_seq and (seq <= station_max or not tx):
                if now - self.replayed_at.get(seq, -1e18) >= debounce_s:
                    out.append((seq, data))
        return out

    def mark_replayed(self, seq: int, now: float) -> None:
        self.replayed_at[seq] = now
        if len(self.replayed_at) > 4 * self.capacity:
            for s in [s for s in self.replayed_at if s not in self.items]:
                del self.replayed_at[s]

    def mark_transmitted(self, seq: int) -> None:
        if seq in self.items:
            self.items[seq][2] = True

    def forget_dropped_before(self, seq: int) -> None:
        for s in [s for s in self.dropped if s <= seq]:
            del self.dropped[s]

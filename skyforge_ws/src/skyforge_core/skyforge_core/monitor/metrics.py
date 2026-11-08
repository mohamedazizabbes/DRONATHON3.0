"""Latency tracking (p50/p95/p99 with perf_counter_ns) and KPI counters."""
from __future__ import annotations

from collections import deque

import numpy as np


class LatencyTracker:
    def __init__(self, name: str, maxlen: int = 2000):
        self.name = name
        self.buf: deque[float] = deque(maxlen=maxlen)
        self.all_max = 0.0
        self.n = 0

    def add(self, ms: float) -> None:
        self.buf.append(ms)
        self.all_max = max(self.all_max, ms)
        self.n += 1

    def pct(self, q: float) -> float:
        if not self.buf:
            return 0.0
        return float(np.percentile(np.fromiter(self.buf, float), q))

    def summary(self) -> dict:
        return {
            "name": self.name,
            "n": self.n,
            "p50_ms": round(self.pct(50), 3),
            "p95_ms": round(self.pct(95), 3),
            "p99_ms": round(self.pct(99), 3),
            "max_ms": round(self.all_max, 3),
        }

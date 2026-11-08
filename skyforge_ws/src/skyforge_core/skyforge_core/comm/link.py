"""Radio link simulator: up unless in a shadow polygon, out of range, or jammed.

Jamming = packet loss + raised noise floor, driven by the fault injector.
"""
from __future__ import annotations

import math
import random
from typing import Optional

from ..geometry import in_any_polygon


class LinkSimulator:
    def __init__(self, params: dict, seed: int = 0):
        c = params["comm"]
        self.station = tuple(c["station"])
        self.max_range = float(c["max_range_m"])
        self.shadows = c.get("shadow_polygons", [])
        self.base_loss = float(c.get("base_loss", 0.0))
        self.jam_loss = float(c.get("jam_loss", 1.0))
        self.jammed = False
        self.rng = random.Random(seed)
        self.reason = ""

    def is_up(self, pos) -> bool:
        if self.jammed and self.jam_loss >= 1.0:
            self.reason = "jammed"
            return False
        if math.hypot(pos[0] - self.station[0], pos[1] - self.station[1]) > self.max_range:
            self.reason = "out of range"
            return False
        if in_any_polygon(pos, self.shadows):
            self.reason = "radio shadow"
            return False
        self.reason = ""
        return True

    def quality(self, pos) -> float:
        if not self.is_up(pos):
            return 0.0
        d = math.hypot(pos[0] - self.station[0], pos[1] - self.station[1])
        q = max(0.0, 1.0 - d / self.max_range)
        if self.jammed:
            q *= 1.0 - self.jam_loss
        return q

    def deliver(self, pos) -> bool:
        """Does one packet get through right now?"""
        if not self.is_up(pos):
            return False
        loss = self.base_loss + (self.jam_loss if self.jammed else 0.0)
        return self.rng.random() >= loss

    def set_jammed(self, on: bool, loss: Optional[float] = None) -> None:
        self.jammed = on
        if loss is not None:
            self.jam_loss = loss

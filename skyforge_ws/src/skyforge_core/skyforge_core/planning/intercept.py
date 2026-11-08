"""Intercept point of a drifting target and expanding-square search."""
from __future__ import annotations

import math

from ..geometry import dist2


def intercept_point(drone_pos, target_pos, target_vel, ground_speed: float, now: float = 0.0, t_obs: float = 0.0, iters: int = 3):
    """Fixed-point solve p* = p(t_obs) + v * (now - t_obs + |p* - drone| / vg)."""
    vg = max(ground_speed, 0.5)
    age = max(0.0, now - t_obs)
    p = (target_pos[0] + target_vel[0] * age, target_pos[1] + target_vel[1] * age)
    for _ in range(iters):
        t = dist2(drone_pos, p) / vg
        p = (
            target_pos[0] + target_vel[0] * (age + t),
            target_pos[1] + target_vel[1] * (age + t),
        )
    return p


def expanding_square(center, step: float, legs: int) -> list[tuple[float, float]]:
    """IAMSAR expanding square: legs of length step, step, 2step, 2step, 3step..."""
    x, y = center
    pts = [(x, y)]
    dirs = [(0, 1), (1, 0), (0, -1), (-1, 0)]
    for k in range(legs):
        L = step * (k // 2 + 1)
        dx, dy = dirs[k % 4]
        x, y = x + dx * L, y + dy * L
        pts.append((x, y))
    return pts


def search_step_from_sigma(sigma: float, base: float) -> float:
    return max(base, min(3.0 * sigma, 20.0)) if math.isfinite(sigma) else base

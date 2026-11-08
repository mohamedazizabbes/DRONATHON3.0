"""Frames and small geometry helpers.

Convention everywhere in SKYFORGE: world frame ENU (x east, y north, z up),
body frame FLU. PX4 talks NED/FRD; ned_to_enu / enu_to_ned are the ONLY
conversion points (used by the ROS reflex/brain wrappers).
"""
from __future__ import annotations

import math
from typing import Iterable, Sequence


def ned_to_enu(v: Sequence[float]) -> tuple[float, float, float]:
    return (float(v[1]), float(v[0]), -float(v[2]))


def enu_to_ned(v: Sequence[float]) -> tuple[float, float, float]:
    return (float(v[1]), float(v[0]), -float(v[2]))


def yaw_ned_to_enu(yaw_ned: float) -> float:
    return wrap_pi(math.pi / 2.0 - yaw_ned)


def yaw_enu_to_ned(yaw_enu: float) -> float:
    return wrap_pi(math.pi / 2.0 - yaw_enu)


def quat_ned_frd_to_enu_flu(q_wxyz: Sequence[float]) -> tuple[float, float, float]:
    """PX4 attitude quaternion (FRD body in NED) -> (roll, pitch, yaw) of FLU body in ENU."""
    w, x, y, z = q_wxyz
    # Euler (NED/FRD)
    roll = math.atan2(2 * (w * x + y * z), 1 - 2 * (x * x + y * y))
    pitch = math.asin(max(-1.0, min(1.0, 2 * (w * y - z * x))))
    yaw = math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
    # FRD->FLU flips pitch sign; NED yaw -> ENU yaw.
    return (roll, -pitch, yaw_ned_to_enu(yaw))


def wrap_pi(a: float) -> float:
    return (a + math.pi) % (2 * math.pi) - math.pi


def dist2(a: Sequence[float], b: Sequence[float]) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])


def rot_world_body(roll: float, pitch: float, yaw: float) -> np.ndarray:
    """Rotation matrix body(FLU) -> world(ENU), ZYX convention."""
    import numpy as np  # lazy: keeps the comm path numpy-free for the Raspberry Pi

    cr, sr = math.cos(roll), math.sin(roll)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cy, sy = math.cos(yaw), math.sin(yaw)
    return np.array(
        [
            [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
            [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
            [-sp, cp * sr, cp * cr],
        ]
    )


def point_in_polygon(p: Sequence[float], poly: Sequence[Sequence[float]]) -> bool:
    x, y = p[0], p[1]
    inside = False
    n = len(poly)
    j = n - 1
    for i in range(n):
        xi, yi = poly[i][0], poly[i][1]
        xj, yj = poly[j][0], poly[j][1]
        if (yi > y) != (yj > y):
            xint = (xj - xi) * (y - yi) / (yj - yi + 1e-12) + xi
            if x < xint:
                inside = not inside
        j = i
    return inside


def in_any_polygon(p: Sequence[float], polys: Iterable[Sequence[Sequence[float]]]) -> bool:
    return any(point_in_polygon(p, poly) for poly in polys)


def seg_point_dist(a: Sequence[float], b: Sequence[float], p: Sequence[float]) -> float:
    ax, ay = a[0], a[1]
    bx, by = b[0], b[1]
    px, py = p[0], p[1]
    dx, dy = bx - ax, by - ay
    L2 = dx * dx + dy * dy
    if L2 < 1e-12:
        return math.hypot(px - ax, py - ay)
    t = max(0.0, min(1.0, ((px - ax) * dx + (py - ay) * dy) / L2))
    return math.hypot(px - (ax + t * dx), py - (ay + t * dy))


def segment_clear(a, b, circles: Iterable[tuple[tuple[float, float], float]]) -> bool:
    """True if segment a-b does not intersect any (center, radius) circle."""
    for c, r in circles:
        if seg_point_dist(a, b, c) < r:
            return False
    return True


def path_length(pts: Sequence[Sequence[float]]) -> float:
    return sum(dist2(pts[i], pts[i + 1]) for i in range(len(pts) - 1))


def kendall_tau(order_a: Sequence[str], order_b: Sequence[str]) -> float:
    """Kendall tau between two rankings over their common items."""
    common = [x for x in order_a if x in set(order_b)]
    n = len(common)
    if n < 2:
        return 1.0
    pos_b = {x: i for i, x in enumerate(order_b)}
    conc = disc = 0
    for i in range(n):
        for j in range(i + 1, n):
            if pos_b[common[i]] < pos_b[common[j]]:
                conc += 1
            else:
                disc += 1
    return (conc - disc) / (conc + disc)

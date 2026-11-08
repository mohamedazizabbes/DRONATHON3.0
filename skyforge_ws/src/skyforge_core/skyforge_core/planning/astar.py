"""A* on a 2D grid at cruise altitude, with inflated hard zones and a soft
threat-aware cost (hostile line of sight costs `los_cost` times more).

Fast path: if the straight segment is clear of every hard circle and of every
soft ring, the straight line is returned without building a grid.
Paths are smoothed by string pulling against the exact circles.
"""
from __future__ import annotations

import heapq
import math
from typing import Optional, Sequence

import numpy as np

from ..geometry import dist2, segment_clear

Circle = tuple[tuple[float, float], float]


class PathResult:
    def __init__(self, path, ok: bool, reason: str = "", expansions: int = 0):
        self.path: list[tuple[float, float]] = path
        self.ok = ok
        self.reason = reason
        self.expansions = expansions

    def __repr__(self):
        return f"PathResult(ok={self.ok}, n={len(self.path)}, reason={self.reason!r})"


class GridPlanner:
    def __init__(self, area: dict, res: float = 2.0, margin: float = 30.0, max_expansions: int = 60000):
        self.res = res
        self.x0 = area["xmin"] - margin
        self.y0 = area["ymin"] - margin
        self.nx = int(math.ceil((area["xmax"] - area["xmin"] + 2 * margin) / res))
        self.ny = int(math.ceil((area["ymax"] - area["ymin"] + 2 * margin) / res))
        xs = self.x0 + (np.arange(self.nx) + 0.5) * res
        ys = self.y0 + (np.arange(self.ny) + 0.5) * res
        self.cx, self.cy = np.meshgrid(xs, ys, indexing="ij")
        self.max_expansions = max_expansions

    # -- grid helpers -------------------------------------------------------
    def to_cell(self, p) -> tuple[int, int]:
        i = int((p[0] - self.x0) / self.res)
        j = int((p[1] - self.y0) / self.res)
        return (min(max(i, 0), self.nx - 1), min(max(j, 0), self.ny - 1))

    def to_world(self, c) -> tuple[float, float]:
        return (self.x0 + (c[0] + 0.5) * self.res, self.y0 + (c[1] + 0.5) * self.res)

    def _maps(self, hard: list[Circle], soft: list[tuple[tuple[float, float], float, float]]):
        blocked = np.zeros((self.nx, self.ny), dtype=bool)
        for c, r in hard:
            blocked |= (self.cx - c[0]) ** 2 + (self.cy - c[1]) ** 2 <= r * r
        cost = np.ones((self.nx, self.ny), dtype=np.float32)
        for c, r, k in soft:
            m = (self.cx - c[0]) ** 2 + (self.cy - c[1]) ** 2 <= r * r
            cost[m] = np.maximum(cost[m], k)
        return blocked, cost

    def _nearest_free(self, cell, blocked) -> Optional[tuple[int, int]]:
        free = np.argwhere(~blocked)
        if len(free) == 0:
            return None
        d = (free[:, 0] - cell[0]) ** 2 + (free[:, 1] - cell[1]) ** 2
        k = int(np.argmin(d))
        return (int(free[k, 0]), int(free[k, 1]))

    # -- public -------------------------------------------------------------
    def plan(
        self,
        start: Sequence[float],
        goal: Sequence[float],
        hard: list[Circle],
        soft: Optional[list[tuple[tuple[float, float], float, float]]] = None,
    ) -> PathResult:
        soft = soft or []
        start = (float(start[0]), float(start[1]))
        goal = (float(goal[0]), float(goal[1]))

        goal_blocked = any(dist2(goal, c) < r for c, r in hard)
        start_blocked = any(dist2(start, c) < r for c, r in hard)
        if not goal_blocked and not start_blocked:
            if segment_clear(start, goal, hard) and segment_clear(start, goal, [(c, r) for c, r, _ in soft]):
                return PathResult([start, goal], True, "direct")

        blocked, cost = self._maps(hard, soft)
        s = self.to_cell(start)
        g = self.to_cell(goal)
        prefix: list[tuple[float, float]] = [start]
        reason = "astar"
        if blocked[s]:
            # We are inside an inflated zone (zone just appeared): escape to
            # the nearest free cell first, by the shortest way out.
            s2 = self._nearest_free(s, blocked)
            if s2 is None:
                return PathResult([start], False, "no_free_cell")
            prefix.append(self.to_world(s2))
            s = s2
            reason = "escape+astar"
        if blocked[g]:
            return PathResult([start], False, "goal_in_zone")

        cells = self._astar(s, g, blocked, cost)
        if cells is None:
            return PathResult([start], False, "no_path")
        pts = prefix + [self.to_world(c) for c in cells[1:-1]] + [goal]
        pts = self._smooth(pts, hard, soft)
        return PathResult(pts, True, reason)

    def _astar(self, s, g, blocked, cost):
        nx, ny = self.nx, self.ny
        sq2 = math.sqrt(2.0)
        nbrs = [(1, 0, 1.0), (-1, 0, 1.0), (0, 1, 1.0), (0, -1, 1.0), (1, 1, sq2), (1, -1, sq2), (-1, 1, sq2), (-1, -1, sq2)]
        gx, gy = g

        def h(c):
            dx, dy = abs(c[0] - gx), abs(c[1] - gy)
            return (dx + dy) + (sq2 - 2) * min(dx, dy)

        g_cost = {s: 0.0}
        parent = {s: None}
        openq = [(h(s), 0.0, s)]
        closed = set()
        exp = 0
        blocked_l = blocked
        while openq:
            f, gc, c = heapq.heappop(openq)
            if c in closed:
                continue
            if c == g:
                path = [c]
                while parent[path[-1]] is not None:
                    path.append(parent[path[-1]])
                return path[::-1]
            closed.add(c)
            exp += 1
            if exp > self.max_expansions:
                return None
            cx, cy = c
            for dx, dy, step in nbrs:
                nx_, ny_ = cx + dx, cy + dy
                if nx_ < 0 or ny_ < 0 or nx_ >= nx or ny_ >= ny:
                    continue
                if blocked_l[nx_, ny_]:
                    continue
                if dx and dy and (blocked_l[cx + dx, cy] or blocked_l[cx, cy + dy]):
                    continue  # no corner cutting
                n = (nx_, ny_)
                ng = gc + step * float(cost[nx_, ny_])
                if ng < g_cost.get(n, 1e18):
                    g_cost[n] = ng
                    parent[n] = c
                    heapq.heappush(openq, (ng + h(n), ng, n))
        return None

    def _smooth(self, pts, hard, soft):
        if len(pts) <= 2:
            return pts
        soft_c = [(c, r) for c, r, _ in soft]
        out = [pts[0]]
        i = 0
        while i < len(pts) - 1:
            j = len(pts) - 1
            while j > i + 1:
                a, b = pts[i], pts[j]
                # never shortcut through a hard zone; only shortcut through a
                # soft ring if both ends are already inside it
                if segment_clear(a, b, hard) and (
                    segment_clear(a, b, soft_c)
                    or all(dist2(a, c) < r and dist2(b, c) < r for c, r in soft_c if not segment_clear(a, b, [(c, r)]))
                ):
                    break
                j -= 1
            out.append(pts[j])
            i = j
        return out

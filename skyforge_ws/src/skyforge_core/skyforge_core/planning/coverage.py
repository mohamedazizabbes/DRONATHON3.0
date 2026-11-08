"""Boustrophedon (lawnmower) exploration + 2 m coverage grid (F1/F2)."""
from __future__ import annotations

import math

import numpy as np


def footprint_half_width(alt: float, hfov_deg: float) -> float:
    return alt * math.tan(math.radians(hfov_deg) / 2.0)


def lawnmower(area: dict, alt: float, hfov_deg: float, overlap: float, start=(0.0, 0.0)) -> list[tuple[float, float]]:
    """Lanes along x, stepping in y. Starts from the corner closest to `start`."""
    half_w = footprint_half_width(alt, hfov_deg)
    spacing = max(2.0, 2.0 * half_w * overlap)
    xmin, xmax = area["xmin"], area["xmax"]
    ymin, ymax = area["ymin"], area["ymax"]
    # Keep lanes half a footprint inside the area so the borders get seen.
    y0, y1 = ymin + min(half_w, (ymax - ymin) / 2), ymax - min(half_w, (ymax - ymin) / 2)
    n = max(1, int(math.ceil((y1 - y0) / spacing)) + 1)
    ys = [y0 + i * (y1 - y0) / max(1, n - 1) for i in range(n)] if n > 1 else [0.5 * (y0 + y1)]
    if abs(start[1] - ys[-1]) < abs(start[1] - ys[0]):
        ys = ys[::-1]
    xa, xb = xmin + 2.0, xmax - 2.0
    if abs(start[0] - xb) < abs(start[0] - xa):
        xa, xb = xb, xa
    wps = []
    for i, y in enumerate(ys):
        if i % 2 == 0:
            wps += [(xa, y), (xb, y)]
        else:
            wps += [(xb, y), (xa, y)]
    return wps


class CoverageGrid:
    def __init__(self, area: dict, res: float = 2.0):
        self.res = res
        self.x0, self.y0 = area["xmin"], area["ymin"]
        self.nx = max(1, int(math.ceil((area["xmax"] - area["xmin"]) / res)))
        self.ny = max(1, int(math.ceil((area["ymax"] - area["ymin"]) / res)))
        self.seen = np.zeros((self.nx, self.ny), dtype=bool)
        self.blocked = np.zeros((self.nx, self.ny), dtype=bool)
        xs = self.x0 + (np.arange(self.nx) + 0.5) * res
        ys = self.y0 + (np.arange(self.ny) + 0.5) * res
        self.cx, self.cy = np.meshgrid(xs, ys, indexing="ij")

    def mark_seen(self, pos, alt: float, hfov_deg: float, vfov_deg: float) -> None:
        if alt < 2.0:
            return
        hw = alt * math.tan(math.radians(hfov_deg) / 2)
        hh = alt * math.tan(math.radians(vfov_deg) / 2)
        r = min(hw, hh)  # conservative: circle inscribed in the footprint
        i0 = max(0, int((pos[0] - r - self.x0) / self.res))
        i1 = min(self.nx, int((pos[0] + r - self.x0) / self.res) + 1)
        j0 = max(0, int((pos[1] - r - self.y0) / self.res))
        j1 = min(self.ny, int((pos[1] + r - self.y0) / self.res) + 1)
        if i0 >= i1 or j0 >= j1:
            return
        sub_x = self.cx[i0:i1, j0:j1]
        sub_y = self.cy[i0:i1, j0:j1]
        m = (sub_x - pos[0]) ** 2 + (sub_y - pos[1]) ** 2 <= r * r
        self.seen[i0:i1, j0:j1] |= m

    def set_blocked(self, circles) -> None:
        self.blocked[:] = False
        for (c, r) in circles:
            self.blocked |= (self.cx - c[0]) ** 2 + (self.cy - c[1]) ** 2 <= r * r

    def fraction(self) -> float:
        reachable = ~self.blocked
        n = int(reachable.sum())
        if n == 0:
            return 1.0
        return float((self.seen & reachable).sum()) / n

    def is_seen(self, p) -> bool:
        i = int((p[0] - self.x0) / self.res)
        j = int((p[1] - self.y0) / self.res)
        if 0 <= i < self.nx and 0 <= j < self.ny:
            return bool(self.seen[i, j])
        return True

    def digest(self) -> str:
        import hashlib

        return hashlib.sha256(np.packbits(self.seen).tobytes()).hexdigest()[:16]

    def to_rle(self) -> list[int]:
        """Run-length encoding of the seen mask (for MAYDAY snapshots)."""
        flat = self.seen.ravel().astype(np.int8)
        runs, cur, n = [], 0, 0
        for v in flat:
            if v == cur:
                n += 1
            else:
                runs.append(n)
                cur, n = v, 1
        runs.append(n)
        return runs

    def load_rle(self, runs: list[int]) -> None:
        flat = np.zeros(self.nx * self.ny, dtype=bool)
        idx, val = 0, False
        for n in runs:
            flat[idx : idx + n] = val
            idx += n
            val = not val
        self.seen = flat.reshape(self.nx, self.ny)

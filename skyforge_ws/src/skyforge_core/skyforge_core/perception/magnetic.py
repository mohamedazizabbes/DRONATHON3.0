"""Mine detection layer 2: magnetic anomaly detection (MAD).

A scalar/total-field magnetometer (boom- or tow-mounted, away from the motors)
samples at ~10 Hz. A ferrous mine produces a local anomaly that decays as 1/r^3
with the distance r between sensor and mine (dipole far field):

    a(x, y) ~ A / r^3,   r = sqrt((x - xm)^2 + (y - ym)^2 + (h + d)^2)

with h the sensor height above the surface and d the mine depth. Pipeline:

1. background: slow EMA of the total field (diurnal drift, platform offset),
   frozen while an anomaly event is open;
2. event detection: |a| > k_sigma * noise opens an event, the peak is
   tracked; the event closes once |a| stayed below half the threshold for
   `close_s` (the drone flew past);
3. inversion: grid search over (x, y) around the peak and depth, amplitude A
   solved in closed form by least squares for each candidate, using EVERY
   stored sample within `fit_radius_m` (adjacent lawnmower lanes give
   cross-track geometry). Returns position, depth, amplitude, R^2 and SNR.

The deeper the mine, the weaker the anomaly (1/r^3): beyond ~5-6 m depth at
cruise altitude it drops under the threshold, which is where the sonar layer
takes over.
"""
from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass
from typing import Optional

import numpy as np


@dataclass
class MagContact:
    pos: tuple[float, float]
    depth: float
    amplitude: float
    r2: float
    snr: float
    sigma: float
    conf: float
    t: float
    single_pass: bool = False  # one straight pass: left/right and lateral-vs-depth ambiguity


class MagAnomalyDetector:
    def __init__(self, params: dict):
        mp = params["magnetic"]
        self.noise = float(mp["noise_nt"])
        self.k_sigma = float(mp["k_sigma"])
        self.tau_bg = float(mp["bg_tau_s"])
        self.close_s = float(mp["close_s"])
        self.fit_r = float(mp["fit_radius_m"])
        self.w_sigma = float(mp.get("fit_weight_sigma_m", 20.0))
        self.grid_half = float(mp["grid_half_m"])
        self.grid_step = float(mp["grid_step_m"])
        self.depths = np.arange(0.0, float(mp["max_depth_m"]) + 1e-6, float(mp["depth_step_m"]))
        self.min_r2 = float(mp["min_r2"])
        self.bg: Optional[float] = None
        self.samples: deque = deque(maxlen=int(mp["max_samples"]))  # (t, x, y, h, a)
        self.event: Optional[dict] = None
        self._t: Optional[float] = None
        self.contacts: list[MagContact] = []
        self.known: dict[str, tuple[float, float, float, float]] = {}  # key -> (x, y, depth, A): sources already explained

    def register(self, key: str, pos, depth: float, amplitude: float) -> None:
        """Remember an accepted source so its field is peeled off before fitting new anomalies."""
        self.known[key] = (float(pos[0]), float(pos[1]), float(depth), float(amplitude))

    @property
    def threshold(self) -> float:
        return self.k_sigma * self.noise

    def add_sample(self, t: float, pos, total_field_nt: float) -> Optional[MagContact]:
        """pos = (x, y, h) with h the sensor height above the surface. Returns a contact when an event closes."""
        if self.bg is None:
            self.bg, self._t = total_field_nt, t
            return None
        dt = max(0.0, t - (self._t or t))
        self._t = t
        a = total_field_nt - self.bg
        if self.event is None:  # background follows slow drift only outside anomalies
            k = 1.0 - math.exp(-dt / self.tau_bg)
            self.bg += k * a
            a = total_field_nt - self.bg
        self.samples.append((t, float(pos[0]), float(pos[1]), max(float(pos[2]), 0.5), a))
        thr = self.threshold
        if abs(a) >= thr:
            if self.event is None:
                self.event = {"peak": abs(a), "peak_xy": (pos[0], pos[1]), "below_since": None}
            elif abs(a) > self.event["peak"]:
                self.event.update(peak=abs(a), peak_xy=(pos[0], pos[1]))
            self.event["below_since"] = None
            return None
        if self.event is not None:
            if abs(a) < 0.5 * thr:
                if self.event["below_since"] is None:
                    self.event["below_since"] = t
                elif t - self.event["below_since"] >= self.close_s:
                    ev, self.event = self.event, None
                    c = self.invert(ev["peak_xy"], ev["peak"], t)
                    if c is not None:
                        self.contacts.append(c)
                    return c
            else:
                self.event["below_since"] = None
        return None

    def invert(self, peak_xy, peak: float, t: float) -> Optional[MagContact]:
        S = np.array([s for s in self.samples if math.hypot(s[1] - peak_xy[0], s[2] - peak_xy[1]) <= self.fit_r])
        if len(S) < 8:
            return None
        xs, ys, hs, a = S[:, 1], S[:, 2], S[:, 3], S[:, 4].copy()
        # peel: subtract the modelled field of sources already explained elsewhere
        # (a nearby second mine or a weapon would otherwise bias this single-dipole fit)
        for kx, ky, kd, kA in self.known.values():
            if math.hypot(kx - peak_xy[0], ky - peak_xy[1]) > 8.0:
                a -= kA / ((xs - kx) ** 2 + (ys - ky) ** 2 + (hs + kd) ** 2) ** 1.5
        # proximity weights: samples far from the peak may carry a neighbour's field
        w = np.exp(-((xs - peak_xy[0]) ** 2 + (ys - peak_xy[1]) ** 2) / (2.0 * self.w_sigma ** 2))
        g = np.arange(-self.grid_half, self.grid_half + 1e-6, self.grid_step)
        cx = peak_xy[0] + g
        cy = peak_xy[1] + g
        best = (np.inf, None)
        # vectorised over (x, y) candidates for each depth: shape (nx, ny, n_samples)
        dx2 = (cx[:, None, None] - xs[None, None, :]) ** 2
        dy2 = (cy[None, :, None] - ys[None, None, :]) ** 2
        for d in self.depths:
            r3 = (dx2 + dy2 + ((hs + d) ** 2)[None, None, :]) ** 1.5
            f = 1.0 / r3
            A = (w * f * a).sum(-1) / (w * f * f).sum(-1)  # closed-form weighted-LS amplitude per candidate
            res = (w * (a - A[..., None] * f) ** 2).sum(-1)
            i = np.unravel_index(int(np.argmin(res)), res.shape)
            if res[i] < best[0]:
                best = (float(res[i]), (float(cx[i[0]]), float(cy[i[1]]), float(d), float(A[i])))
        res, (bx, by, bd, bA) = best
        ss = float((w * a * a).sum())
        r2 = 1.0 - res / ss if ss > 0 else 0.0
        if r2 < self.min_r2 or bA * np.sign(a[np.argmax(np.abs(a))]) <= 0:
            return None
        conf = float(min(1.0, (peak / self.noise / (4 * self.k_sigma)) * r2))
        snr = peak / self.noise
        # localisation precision ~ distance / SNR (honest: weak, far anomalies are imprecise)
        r_min = float(np.mean(hs)) + bd
        sigma = max(self.grid_step, 1.5 * r_min / max(snr, 1.0)) * (1.5 if r2 < 0.8 else 1.0)
        # geometry check: samples along a single line cannot resolve the side of the
        # track nor separate lateral offset from depth -> large sigma, low confidence
        P = np.cov(np.vstack([xs, ys]))
        single = math.sqrt(max(float(np.linalg.eigvalsh(P)[0]), 0.0)) < 4.0
        if single:
            lateral = math.hypot(bx - peak_xy[0], by - peak_xy[1])  # offset from the track: the ambiguous part
            sigma = max(sigma, lateral + 5.0)
            conf *= 0.5
        return MagContact((bx, by), bd, bA, r2, snr, sigma, conf, t, single)

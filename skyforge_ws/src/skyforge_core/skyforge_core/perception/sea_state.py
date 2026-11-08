"""Online sea-state (surface current) estimator.

Every floating object drifts with  v = current + k_leeway * wind  (+ wave
orbital motion, zero-mean over a wave period). Each floating track that is
currently observed gives a sample of the current:

    c_i = v_track_i - k_leeway * wind,   weight_i = 1 / var(v_track_i)

The estimate is an inverse-variance weighted EMA, starting from the forecast
in params (drift.current). It feeds the drift prior of new floating tracks and
re-centres the prediction of floating tracks that are not seen any more, so a
tide change or a shifted current moves the predicted intercept points of
everything at sea - not only of what the camera is looking at.
"""
from __future__ import annotations

import math


class SeaStateEstimator:
    def __init__(self, current0, k_leeway: float, tau_s: float = 30.0, min_obs: int = 5):
        self.current = [float(current0[0]), float(current0[1])]
        self.k_leeway = k_leeway
        self.tau_s = tau_s
        self.min_obs = min_obs
        self.n_updates = 0
        self._t = None

    def prior(self, wind) -> tuple[float, float]:
        return (self.current[0] + self.k_leeway * wind[0], self.current[1] + self.k_leeway * wind[1])

    def update_baseline(self, v, var: float, wind, gain_max: float = 0.6) -> None:
        """Long-baseline drift sample (re-acquired object): strong, gain from its variance."""
        c = (v[0] - self.k_leeway * wind[0], v[1] - self.k_leeway * wind[1])
        k = gain_max * (0.05**2) / (0.05**2 + var)  # 0.05 m/s: typical current knowledge
        self.current[0] += k * (c[0] - self.current[0])
        self.current[1] += k * (c[1] - self.current[1])
        self.n_updates += 1

    def update(self, t: float, samples, wind) -> bool:
        """samples: iterable of (vx, vy, var_v) from floating tracks seen right now."""
        dt = 0.0 if self._t is None else max(0.0, t - self._t)
        self._t = t
        num_x = num_y = den = 0.0
        for vx, vy, var in samples:
            w = 1.0 / max(var, 1e-4)
            num_x += w * (vx - self.k_leeway * wind[0])
            num_y += w * (vy - self.k_leeway * wind[1])
            den += w
        if den <= 0.0 or dt <= 0.0:
            return False
        a = 1.0 - math.exp(-dt / self.tau_s)
        self.current[0] += a * (num_x / den - self.current[0])
        self.current[1] += a * (num_y / den - self.current[1])
        self.n_updates += 1
        return True

"""Navigation integrity: GNSS loss and spoofing detection.

Two complementary filters run on the same velocity (IMU / optical flow /
odometry) and GNSS inputs:

* `est` (output): velocity-propagated position corrected towards GNSS with a
  moderate gain -> smooths GNSS noise for the brain and the perception.
* `ref` (integrity reference): same propagation but a LOW gain, so it cannot
  be dragged along by a spoofer ramping the position. The innovation test

      d2 = |gnss - ref|^2 / (sigma_gnss^2 + sigma_ref^2)  > chi2 gate (2 dof)

  failing `spoof_consecutive` times in a row rejects GNSS (nav source ->
  odometry, i.e. dead reckoning). A jump spoof fails at once; a ramp spoof
  builds a lag error v_ramp * dt / gain and fails within ~1 s at 5 m/s.
  Recovery needs `recover_consecutive` consistent fixes.
"""
from __future__ import annotations

import math
from typing import Optional


class NavIntegrity:
    def __init__(self, params: dict, pos0=(0.0, 0.0), out_gain: float = 0.3, ref_gain: float = 0.02):
        self.np = params["nav"]
        self.est = [float(pos0[0]), float(pos0[1])]
        self.ref = [float(pos0[0]), float(pos0[1])]
        self.out_gain = out_gain
        self.ref_gain = ref_gain
        self.dr_sigma = 0.0  # growth while GNSS is rejected
        self.trusted = True
        self.source = "gnss"
        self.bad = 0
        self.good = 0
        self.last_fix_t: Optional[float] = None
        self.reason = ""
        self.spoof_detect_t: Optional[float] = None
        self.last_d = 0.0
        self._t: Optional[float] = None

    def propagate(self, t: float, vel_xy) -> None:
        if self._t is not None:
            dt = max(0.0, t - self._t)
            for p in (self.est, self.ref):
                p[0] += vel_xy[0] * dt
                p[1] += vel_xy[1] * dt
            if not self.trusted:
                self.dr_sigma += self.np["dr_sigma_growth"] * dt
        self._t = t

    def gnss_fix(self, t: float, pos_xy, sats: int = 12) -> None:
        if sats < self.np["min_sats"]:
            return  # treated as no fix -> timeout handles it
        self.last_fix_t = t
        sg = self.np["gnss_sigma"]
        s2 = sg**2 + (0.5 * sg) ** 2 + self.dr_sigma**2
        dx, dy = pos_xy[0] - self.ref[0], pos_xy[1] - self.ref[1]
        d2 = (dx * dx + dy * dy) / s2
        self.last_d = math.sqrt(dx * dx + dy * dy)
        if d2 > self.np["gate_chi2"]:
            self.bad += 1
            self.good = 0
            if self.trusted and self.bad >= self.np["spoof_consecutive"]:
                self.trusted = False
                self.source = "odometry"
                self.reason = f"spoofing: GNSS {self.last_d:.1f} m from inertial reference ({math.sqrt(d2):.1f} sigma)"
                self.spoof_detect_t = t
        else:
            self.good += 1
            self.bad = 0
            if not self.trusted and self.good >= self.np["recover_consecutive"]:
                self.trusted = True
                self.source = "gnss"
                self.reason = "GNSS consistent again"
                self.dr_sigma = 0.0
        if self.trusted:
            for p, k in ((self.est, self.out_gain), (self.ref, self.ref_gain)):
                p[0] += k * (pos_xy[0] - p[0])
                p[1] += k * (pos_xy[1] - p[1])

    def check_timeout(self, t: float) -> None:
        if self.last_fix_t is not None and t - self.last_fix_t > self.np["fix_timeout_s"] and self.trusted:
            self.trusted = False
            self.source = "odometry"
            self.reason = "GNSS fix lost"

    @property
    def gnss_ok(self) -> bool:
        return self.trusted

    def position(self, _gnss_xy=None):
        return (self.est[0], self.est[1])

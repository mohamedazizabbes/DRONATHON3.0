"""Energy model, wind-aware return-to-home threshold (F11) and energy gate (F6).

Control model (consistent with the reflex layer): the drone is commanded a
GROUND velocity v_g along the leg; the autopilot holds it by flying the
airspeed given by the wind triangle, up to the airframe limit v_air_max:

    v_air = | v_g * u - w |              (w = air-mass velocity, ENU, gust-inflated)
    if v_air > v_air_max: v_g is reduced until v_air = v_air_max (strong headwind)
    P(v_air) = scale * p_hover * (1 + k_speed * v_air^2)
    E_leg    = P(v_air) * d / v_g

    E_RTH = E_leg(pos -> home) + E_land + E_res
    RTH forced when E_avail <= factor * E_RTH   (factor 1.3, 1.5 without GNSS)

`scale` is calibrated ONLINE from the measured battery drain (EMA, bounded),
so a wrong power budget is corrected during the flight and the predicted vs
actual return energy converges. The triage energy gate uses the same factor
as RTH so choosing a casualty never trips RTH right after the delivery.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional, Sequence


@dataclass
class EnergyModel:
    capacity_wh: float
    p_hover_w: float
    k_speed: float
    v_min: float
    reserve_frac: float
    rth_factor: float
    rth_factor_gnss_lost: float
    land_wh: float
    deliver_wh: float
    gust_factor: float
    critical_pct: float
    v_air_max: float = 16.0
    v_cmd: float = 6.0  # commanded cruise ground speed (mission.cruise_speed)
    scale: float = 1.0  # online calibration factor on P(v)
    calib_alpha: float = 0.15
    calib_bounds: tuple = (0.6, 1.8)
    _calib: dict = field(default_factory=dict, repr=False)

    @classmethod
    def from_params(cls, p: dict) -> "EnergyModel":
        e = p["energy"]
        kw = {k: float(e[k]) for k in cls.__dataclass_fields__ if k in e and not k.startswith("_") and k != "calib_bounds"}
        m = cls(**kw)
        m.v_cmd = float(p.get("mission", {}).get("cruise_speed", m.v_cmd))
        return m

    @property
    def reserve_wh(self) -> float:
        return self.reserve_frac * self.capacity_wh

    def power_w(self, v_air: float) -> float:
        return self.scale * self.p_hover_w * (1.0 + self.k_speed * v_air * v_air)

    def _wind(self, wind) -> tuple[float, float]:
        return (wind[0] * self.gust_factor, wind[1] * self.gust_factor)

    def leg_speeds(self, u: Sequence[float], v_g: float, wind) -> tuple[float, float]:
        """(achieved ground speed, airspeed) for a commanded ground speed along u."""
        wx, wy = self._wind(wind)
        ax, ay = v_g * u[0] - wx, v_g * u[1] - wy
        v_air = math.hypot(ax, ay)
        if v_air <= self.v_air_max:
            return v_g, v_air
        # Strong headwind: the airframe saturates; best ground speed with |v_air| = v_air_max
        w_par = wx * u[0] + wy * u[1]
        w_perp2 = (wx * wx + wy * wy) - w_par * w_par
        if w_perp2 >= self.v_air_max**2:
            return self.v_min, self.v_air_max
        vg = w_par + math.sqrt(self.v_air_max**2 - w_perp2)
        return max(self.v_min, vg), self.v_air_max

    def ground_speed(self, u: Sequence[float], v_cmd: Optional[float], wind) -> float:
        return self.leg_speeds(u, v_cmd or self.v_cmd, wind)[0]

    def leg_wh(self, a, b, v_cmd: Optional[float], wind) -> float:
        dx, dy = b[0] - a[0], b[1] - a[1]
        d = math.hypot(dx, dy)
        if d < 1e-6:
            return 0.0
        vg, v_air = self.leg_speeds((dx / d, dy / d), v_cmd or self.v_cmd, wind)
        return self.power_w(v_air) * (d / vg) / 3600.0

    def path_wh(self, pts, v_cmd, wind) -> float:
        return sum(self.leg_wh(pts[i], pts[i + 1], v_cmd, wind) for i in range(len(pts) - 1))

    def factor(self, gnss_ok: bool) -> float:
        return self.rth_factor if gnss_ok else self.rth_factor_gnss_lost

    def e_rth(self, pos, home, v_cmd, wind) -> float:
        return self.leg_wh(pos, home, v_cmd, wind) + self.land_wh + self.reserve_wh

    def rth_threshold(self, pos, home, v_cmd, wind, gnss_ok: bool = True) -> float:
        return self.factor(gnss_ok) * self.e_rth(pos, home, v_cmd, wind)

    def mission_cost_wh(self, pos, target, home, v_cmd, wind) -> dict:
        return {
            "e_go": self.leg_wh(pos, target, v_cmd, wind),
            "e_deliver": self.deliver_wh,
            "e_back": self.leg_wh(target, home, v_cmd, wind),
            "e_land": self.land_wh,
            "e_res": self.reserve_wh,
        }

    def gate_ok(self, cost: dict, e_avail: float, gnss_ok: bool = True) -> bool:
        total = cost["e_go"] + cost["e_deliver"] + cost["e_back"] + cost["e_land"] + cost["e_res"]
        return self.factor(gnss_ok) * total <= e_avail

    # ----------------------------------------------------------- calibration
    def calibrate(self, t: float, battery_wh: float, vel, wind, airborne: bool, window_s: float = 5.0) -> Optional[float]:
        """Feed (time, remaining Wh, ground velocity, wind) every tick in flight.
        Every `window_s` compares measured mean power with the model and updates
        `scale` (EMA, bounded). Returns the new scale when updated."""
        c = self._calib
        if not airborne:
            c.clear()
            return None
        v_air = math.hypot(vel[0] - wind[0], vel[1] - wind[1])
        if "t0" not in c:
            c.update(t0=t, e0=battery_wh, pred_wh=0.0, last_t=t)
            return None
        dt = t - c["last_t"]
        c["last_t"] = t
        c["pred_wh"] += self.p_hover_w * (1.0 + self.k_speed * v_air * v_air) * max(dt, 0.0) / 3600.0  # unscaled
        if t - c["t0"] < window_s:
            return None
        used = c["e0"] - battery_wh
        pred = c["pred_wh"]
        c.update(t0=t, e0=battery_wh, pred_wh=0.0)
        if pred <= 1e-6 or used <= 0.0:
            return None  # e.g. battery reading quantised / not updated in this window
        ratio = min(max(used / pred, self.calib_bounds[0]), self.calib_bounds[1])
        self.scale = (1 - self.calib_alpha) * self.scale + self.calib_alpha * ratio
        return self.scale

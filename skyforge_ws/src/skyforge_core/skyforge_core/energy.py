"""Energy model, wind-aware return-to-home threshold (F11) and energy gate (F6).

Wind convention: `wind` is the velocity of the air mass in ENU (the direction
the wind blows TOWARDS). Ground speed along a leg is solved with the full wind
triangle, so cross-wind is accounted for, not only head/tail wind.

    P(v_air)  = p_hover * (1 + k_speed * v_air^2)          (calibrated in sim)
    v_g       = w.u + sqrt(v_air^2 - |w_perp|^2)
    E_leg     = P(v_air) * d / max(v_g, v_min)
    E_RTH     = E_leg(pos -> home) + E_land + E_res
    RTH forced when E_avail <= factor * E_RTH   (factor 1.3, 1.5 without GNSS)

The triage energy gate uses the same factor so that choosing a casualty can
never immediately trip the RTH threshold after the delivery (no oscillation).
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence


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

    @classmethod
    def from_params(cls, p: dict) -> "EnergyModel":
        e = p["energy"]
        return cls(**{k: float(e[k]) for k in cls.__dataclass_fields__})

    @property
    def reserve_wh(self) -> float:
        return self.reserve_frac * self.capacity_wh

    def power_w(self, v_air: float) -> float:
        return self.p_hover_w * (1.0 + self.k_speed * v_air * v_air)

    def ground_speed(self, u: Sequence[float], v_air: float, wind: Sequence[float]) -> float:
        wx, wy = wind[0] * self.gust_factor, wind[1] * self.gust_factor
        w_par = wx * u[0] + wy * u[1]
        w_perp2 = (wx * wx + wy * wy) - w_par * w_par
        if w_perp2 >= v_air * v_air:
            return self.v_min
        return max(self.v_min, w_par + math.sqrt(v_air * v_air - w_perp2))

    def leg_wh(self, a: Sequence[float], b: Sequence[float], v_air: float, wind: Sequence[float]) -> float:
        dx, dy = b[0] - a[0], b[1] - a[1]
        d = math.hypot(dx, dy)
        if d < 1e-6:
            return 0.0
        u = (dx / d, dy / d)
        vg = self.ground_speed(u, v_air, wind)
        return self.power_w(v_air) * (d / vg) / 3600.0

    def path_wh(self, pts: Sequence[Sequence[float]], v_air: float, wind) -> float:
        return sum(self.leg_wh(pts[i], pts[i + 1], v_air, wind) for i in range(len(pts) - 1))

    def factor(self, gnss_ok: bool) -> float:
        return self.rth_factor if gnss_ok else self.rth_factor_gnss_lost

    def e_rth(self, pos, home, v_air, wind) -> float:
        return self.leg_wh(pos, home, v_air, wind) + self.land_wh + self.reserve_wh

    def rth_threshold(self, pos, home, v_air, wind, gnss_ok: bool = True) -> float:
        return self.factor(gnss_ok) * self.e_rth(pos, home, v_air, wind)

    def mission_cost_wh(self, pos, target, home, v_air, wind) -> dict:
        """Energy components to serve a target and come back home."""
        e_go = self.leg_wh(pos, target, v_air, wind)
        e_back = self.leg_wh(target, home, v_air, wind)
        return {
            "e_go": e_go,
            "e_deliver": self.deliver_wh,
            "e_back": e_back,
            "e_land": self.land_wh,
            "e_res": self.reserve_wh,
        }

    def gate_ok(self, cost: dict, e_avail: float, gnss_ok: bool = True) -> bool:
        total = cost["e_go"] + cost["e_deliver"] + cost["e_back"] + cost["e_land"] + cost["e_res"]
        return self.factor(gnss_ok) * total <= e_avail

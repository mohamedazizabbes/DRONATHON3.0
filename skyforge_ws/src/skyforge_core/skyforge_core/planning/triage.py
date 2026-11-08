"""Casualty prioritisation (F6).

    S_i = 0.7 * U_i * c_i + 0.3 * P_i   if the energy gate passes, else 0
    P_i = 1 - d_i / d_max

Urgency U (START category) is the main criterion, c the detection confidence,
P proximity. Ties are broken by detection time (earliest first). Every call
returns all candidates with their components and why the losers lost, which
goes straight into mission_logs.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from ..energy import EnergyModel
from ..geometry import dist2
from ..domain import Track


@dataclass
class Candidate:
    id: str
    score: float
    urgency: float
    conf: float
    proximity: float
    dist: float
    category: str
    feasible: bool
    energy_wh: float
    reason: str = ""
    target: tuple[float, float] = (0.0, 0.0)
    first_seen: float = 0.0

    def to_log(self) -> dict:
        return {
            "id": self.id,
            "S": round(self.score, 3),
            "U": self.urgency,
            "c": round(self.conf, 2),
            "P": round(self.proximity, 3),
            "d": round(self.dist, 1),
            "cat": self.category,
            "ok": self.feasible,
            "E": round(self.energy_wh, 2),
            "why": self.reason,
        }


@dataclass
class TriageResult:
    best: Optional[Candidate]
    ranked: list[Candidate] = field(default_factory=list)

    def order(self) -> list[str]:
        return [c.id for c in self.ranked if c.feasible]


def score_casualties(
    casualties: list[Track],
    pos,
    home,
    energy: EnergyModel,
    e_avail: float,
    wind,
    v_air: float,
    d_max: float,
    gnss_ok: bool = True,
    w_urg: float = 0.7,
    w_prox: float = 0.3,
    intercept_fn=None,
    blocked_fn=None,
) -> TriageResult:
    cands: list[Candidate] = []
    for t in casualties:
        target = intercept_fn(t) if intercept_fn else t.pos
        d = dist2(pos, target)
        prox = max(0.0, 1.0 - d / max(d_max, 1e-6))
        cost = energy.mission_cost_wh(pos, target, home, v_air, wind)
        e_total = sum(cost.values())
        feasible = energy.gate_ok(cost, e_avail, gnss_ok)
        reason = ""
        if not feasible:
            reason = f"energy {energy.factor(gnss_ok) * e_total:.1f}Wh > avail {e_avail:.1f}Wh"
        elif blocked_fn is not None and blocked_fn(target):
            feasible = False
            reason = "target inside exclusion zone"
        s = (w_urg * t.urgency * t.conf + w_prox * prox) if feasible else 0.0
        cat = t.category.value if hasattr(t.category, "value") else str(t.category)
        cands.append(
            Candidate(t.id, s, t.urgency, t.conf, prox, d, cat, feasible, e_total, reason, target, t.first_seen)
        )
    cands.sort(key=lambda c: (-c.score, c.first_seen))
    best = cands[0] if cands and cands[0].feasible and cands[0].score > 0 else None
    for c in cands:
        if best is not None and c is not best and c.feasible:
            c.reason = f"lower score ({c.score:.3f} < {best.score:.3f})"
    if best is not None:
        best.reason = "best feasible"
    return TriageResult(best, cands)

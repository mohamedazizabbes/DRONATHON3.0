"""MAYDAY handover: drone 2 resumes the mission from drone 1's snapshot.

The failing drone sends `may` with a snapshot (fragmented < 1 KB by the comm
module, reassembled by the station, saved as mayday_snapshot.json). Drone 2
starts from base with it:
  - casualties (served ones flagged) -> confirmed tracks in its PerceptionAgent,
    so it goes straight to TRIAGE and never re-serves anyone,
  - exclusion zones -> manual zones until its own perception sees them,
  - coverage grid + lane index -> explored areas are skipped.
"""
from __future__ import annotations

import json
from typing import Any, Optional

from ..domain import Cls, Zone, track_from_dict


def load_snapshot(path: str) -> dict:
    with open(path, encoding="utf-8") as f:
        msg = json.load(f)
    return msg.get("s", msg)  # station saves the whole 'may' message; accept the bare snapshot too


def apply_snapshot(orch, agent, snap: dict, now: float, log=None) -> dict[str, Any]:
    """Seed an Orchestrator (and optionally a PerceptionAgent) with a MAYDAY snapshot."""
    st = orch.state
    served = set(snap.get("served", [])) | {c["id"] for c in snap.get("cas", []) if c.get("s")}
    st["served_ids"] = sorted(served)
    st["explore_idx"] = int(snap.get("explore_idx", 0))
    st["handover"] = True
    if snap.get("cov_rle"):
        orch.brain.coverage.load_rle(snap["cov_rle"])
    st["manual_zones"] = [Zone(z["id"], z["cls"], tuple(z["center"]), float(z["radius"]), tuple(z.get("vel", (0, 0))), 0.0, "handover")
                          for z in snap.get("zones", [])]
    if agent is not None:
        tracks = [track_from_dict({**c, "cls": Cls.INJURED, "confirmed": True}) for c in snap.get("cas", [])]
        for tr, c in zip(tracks, snap.get("cas", [])):
            if c.get("s"):
                tr.attrs["marked"] = True  # already has a kit
        agent.seed_tracks(tracks, now)
    open_cas = sum(1 for c in snap.get("cas", []) if not c.get("s"))
    summary = {"casualties_to_serve": open_cas, "served": len(served), "zones": len(snap.get("zones", [])),
               "coverage": snap.get("cov")}
    if log is not None:
        from ..mission_log import event

        log.add(event(now, "handover", "INIT", f"resuming from MAYDAY snapshot: {open_cas} casualties to serve, "
                                               f"{len(snap.get('zones', []))} zones, coverage {snap.get('cov')}", **summary))
    return summary


def maybe_resume(path: Optional[str], orch, agent, now: float, log=None) -> Optional[dict]:
    if not path or path in ("none", "None"):
        return None
    return apply_snapshot(orch, agent, load_snapshot(path), now, log)

"""DroneState: the single shared state of the orchestrator (LangGraph TypedDict)."""
from __future__ import annotations

import operator
from typing import Annotated, Any, Optional, TypedDict

from ..domain import Phase, Track, Zone


class DroneState(TypedDict, total=False):
    # --- inputs (written by the ROS wrapper / simulator before each tick)
    t: float
    pos: tuple[float, float, float]
    vel: tuple[float, float, float]
    rpy: tuple[float, float, float]
    armed: bool
    battery_wh: float
    battery_pct: float
    link_up: bool
    link_quality: float
    gnss_ok: bool
    nav_source: str  # "gnss" | "odometry"
    wind: tuple[float, float]
    visibility: float  # 0..1
    casualties: list[Track]  # confirmed injured tracks from PerceptionAgent
    zones: list[Zone]  # perception exclusion zones
    new_tracks: list[Track]  # newly confirmed this tick (any class)
    health_fault: Optional[str]
    goal_updates: list[dict]  # verified goal_update bodies from the comm node

    # --- mission memory
    phase: Phase
    prev_phase: Phase
    phase_since: float
    manual_zones: list[Zone]
    served_ids: list[str]
    failed_ids: list[str]
    target: Optional[dict]
    priority_id: Optional[str]
    station_goal: Optional[tuple[float, float]]
    current_goal: Optional[dict]
    waypoints: list[tuple[float, float, float]]
    path_goal: Optional[tuple[float, float, float]]
    zones_sig: str
    explore_idx: int
    coverage: float
    kits: int
    rth_threshold_wh: float
    e_rth_wh: float
    rth_reason: Optional[str]
    autonomous: bool
    gnss_prev_ok: bool
    mayday_sent: bool
    handover: bool  # drone 2 started from a MAYDAY snapshot
    replan: bool
    triage_sig: str
    triage_t: float
    hops: int
    ran: str

    # --- per-tick outputs
    interrupts: list[str]
    actions: list[dict]
    events: Annotated[list[dict], operator.add]
    decision_ms: float


def initial_state(params: dict, t: float = 0.0) -> DroneState:
    home = params["mission"]["home"]
    return DroneState(
        t=t,
        pos=(home[0], home[1], 0.0),
        vel=(0.0, 0.0, 0.0),
        rpy=(0.0, 0.0, 0.0),
        armed=False,
        battery_wh=params["energy"]["capacity_wh"],
        battery_pct=100.0,
        link_up=True,
        link_quality=1.0,
        gnss_ok=True,
        nav_source="gnss",
        wind=(0.0, 0.0),
        visibility=1.0,
        casualties=[],
        zones=[],
        new_tracks=[],
        health_fault=None,
        goal_updates=[],
        phase=Phase.INIT,
        prev_phase=Phase.INIT,
        phase_since=t,
        manual_zones=[],
        served_ids=[],
        failed_ids=[],
        target=None,
        priority_id=None,
        station_goal=None,
        current_goal=None,
        waypoints=[],
        path_goal=None,
        zones_sig="",
        explore_idx=0,
        coverage=0.0,
        kits=int(params["mission"]["kits"]),
        rth_threshold_wh=0.0,
        e_rth_wh=0.0,
        rth_reason=None,
        autonomous=False,
        gnss_prev_ok=True,
        mayday_sent=False,
        handover=False,
        replan=True,
        triage_sig="",
        triage_t=-1e9,
        hops=0,
        ran="",
        interrupts=[],
        actions=[],
        events=[],
        decision_ms=0.0,
    )


def snapshot(state: DroneState, coverage_rle: Optional[list[int]] = None, coverage_hash: str = "") -> dict[str, Any]:
    """MAYDAY / sync snapshot: enough for drone 2 to resume the mission."""
    served = set(state.get("served_ids", []))
    return {
        "pos": [round(v, 1) for v in state["pos"]],
        "phase": state["phase"].value if hasattr(state["phase"], "value") else state["phase"],
        "bat": round(state.get("battery_pct", 0.0), 1),
        "kits": state.get("kits", 0),
        "served": sorted(served),
        # all casualties, served ones flagged ("s":1) so drone 2 can recognise
        # re-detections of people who already received a kit
        "cas": [
            {**{k: v for k, v in t.to_dict().items() if k in ("id", "pos", "vel", "category", "sigma", "floating", "conf", "last_seen")},
             "s": int(t.id in served)}
            for t in state.get("casualties", [])
        ],
        "zones": [
            {"id": z.id, "cls": z.cls, "center": [round(z.center[0], 1), round(z.center[1], 1)], "radius": round(z.radius, 1), "vel": [round(z.vel[0], 2), round(z.vel[1], 2)]}
            for z in list(state.get("zones", [])) + list(state.get("manual_zones", []))
        ],
        "cov": round(state.get("coverage", 0.0), 3),
        "cov_hash": coverage_hash,
        "cov_rle": coverage_rle or [],
        "explore_idx": state.get("explore_idx", 0),
    }

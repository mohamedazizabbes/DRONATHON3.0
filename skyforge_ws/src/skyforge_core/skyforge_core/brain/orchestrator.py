"""Orchestrator: LangGraph StateGraph of deterministic nodes, ticking at 10 Hz.

No language model anywhere in the loop. Node functions live in `nodes.Brain`
and don't depend on LangGraph, so `engine="fsm"` runs the exact same functions
through a plain dict dispatch (fallback if LangGraph overhead shows in p95).
"""
from __future__ import annotations

import time
from typing import Any, Optional

from ..mission_log import MissionLog
from ..monitor.metrics import LatencyTracker
from ..domain import Phase
from .nodes import MAX_HOPS, Brain
from .state import DroneState, initial_state


def build_graph(brain: Brain):
    from langgraph.graph import END, START, StateGraph

    g = StateGraph(DroneState)
    g.add_node("ingest", brain.ingest)
    g.add_node("router", brain.router)
    g.add_node("planner", brain.planner_node)
    handlers = brain.handlers()
    for ph, fn in handlers.items():
        g.add_node(ph.value, brain.wrap_handler(ph, fn))
    g.add_edge(START, "ingest")
    g.add_edge("ingest", "router")
    phase_map = {ph.value: ph.value for ph in handlers}
    g.add_conditional_edges("router", Brain.route_phase, phase_map)
    after = {**phase_map, "planner": "planner"}
    for ph in handlers:
        g.add_conditional_edges(ph.value, Brain.next_after_handler, after)
    g.add_edge("planner", END)
    return g.compile()


class Orchestrator:
    INPUT_KEYS = (
        "t", "pos", "vel", "rpy", "armed", "battery_wh", "battery_pct", "link_up", "link_quality",
        "gnss_ok", "nav_source", "wind", "visibility", "casualties", "zones", "new_tracks", "health_fault",
    )

    def __init__(self, params: dict, log: Optional[MissionLog] = None, engine: Optional[str] = None):
        self.p = params
        self.brain = Brain(params)
        self.engine = engine or params["orchestrator"]["engine"]
        self.log = log or MissionLog()
        self.latency = LatencyTracker("decision", 2000)
        self.graph = build_graph(self.brain) if self.engine == "langgraph" else None
        self.state: DroneState = initial_state(params)

    # ------------------------------------------------------------------
    def tick(self, inputs: dict[str, Any], goal_updates: Optional[list[dict]] = None) -> DroneState:
        s: DroneState = dict(self.state)  # type: ignore[assignment]
        for k in self.INPUT_KEYS:
            if k in inputs:
                s[k] = inputs[k]
        if inputs.get("new_tracks"):
            s["new_tracks"] = list(self.state.get("new_tracks", [])) + list(inputs["new_tracks"])
        if goal_updates:
            s["goal_updates"] = list(s.get("goal_updates", [])) + list(goal_updates)
        s["events"] = []
        t0 = time.perf_counter_ns()
        if self.graph is not None:
            out = self.graph.invoke(s, {"recursion_limit": 4 + 2 * MAX_HOPS})
        else:
            out = self._run_fsm(s)
        ms = (time.perf_counter_ns() - t0) / 1e6
        self.latency.add(ms)
        out["decision_ms"] = ms
        self.log.extend(out.get("events", []))
        self.state = out
        return out

    def _run_fsm(self, s: DroneState) -> DroneState:
        b = self.brain
        events: list = []

        def apply(upd: dict) -> None:
            for k, v in upd.items():
                if k == "events":
                    events.extend(v)
                else:
                    s[k] = v

        apply(b.ingest(dict(s)))
        apply(b.router(dict(s)))
        handlers = {ph.value: b.wrap_handler(ph, fn) for ph, fn in b.handlers().items()}
        nxt = Brain.route_phase(s)
        while nxt != "planner":
            apply(handlers[nxt](dict(s)))
            nxt = Brain.next_after_handler(s)
        apply(b.planner_node(dict(s)))
        s["events"] = events
        return s

    # ------------------------------------------------------------------
    @property
    def phase(self) -> Phase:
        return self.state["phase"]

    def mission_state_msg(self) -> dict:
        """Compact telemetry body (comm 'tel' message)."""
        s = self.state
        return {
            "p": [round(v, 1) for v in s["pos"]],
            "ph": s["phase"].value,
            "bat": round(s.get("battery_pct", 0.0), 1),
            "e": round(s.get("battery_wh", 0.0), 1),
            "rth": round(s.get("rth_threshold_wh", 0.0), 1),
            "cov": round(s.get("coverage", 0.0), 3),
            "nav": s.get("nav_source", "gnss"),
            "kit": s.get("kits", 0),
            "tg": (s.get("target") or {}).get("id"),
        }

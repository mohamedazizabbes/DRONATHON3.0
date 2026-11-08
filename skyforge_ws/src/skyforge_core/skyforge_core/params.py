"""Parameters: defaults here, overrides from a plain YAML file (skyforge.yaml).

Rule from the design doc: nothing is hard-coded in nodes. Every tunable lives
in this tree and can be overridden without a rebuild.
"""
from __future__ import annotations

import copy
from typing import Any

DEFAULTS: dict[str, Any] = {
    "mission": {
        "home": [0.0, 0.0],
        # Exploration area, ENU metres, relative to the home pad.
        "area": {"xmin": -10.0, "xmax": 230.0, "ymin": -70.0, "ymax": 90.0},
        "cruise_alt": 25.0,
        "cruise_speed": 6.0,
        "delivery_alt": 5.0,
        "delivery_radius": 1.5,  # F7: track must be within this of the drop point
        "delivery_max_retries": 3,
        "delivery_rounds": 3,  # paused deliveries are retried up to this many rounds before giving up
        "delivery_cooldown_s": 30.0,
        "release_offset_m": 1.0,  # estimated offset required to release (spec radius 1.5)
        "release_max_speed": 0.6,  # m/s relative to the casualty
        "takeoff_tol": 1.0,
        "waypoint_tol": 2.5,
        # immediate: go to every casualty as soon as it is confirmed
        # after_coverage: explore everything first, then rescue in score order
        # hybrid: only casualties with urgency >= preempt_urgency interrupt exploration
        "triage_policy": "hybrid",
        "preempt_urgency": 0.95,
        "kits": 4,  # medical kits on board
        "serve_deceased": False,  # never spend a kit on a casualty flagged likely deceased (reported to the station)
        "coverage_target": 0.97,
        "switch_hysteresis": 1.25,  # new target must beat current score by this factor
        "duplicate_radius_m": 6.0,  # + 2 sigma: candidate on a served casualty = same person
        "duplicate_max_m": 15.0,  # land
        "duplicate_max_floating_m": 45.0,  # drifting: identity gate grows with uncertainty
        "release_confirm_ticks": 3,  # release condition must hold this many decision ticks
        "lost_target_search_s": 25.0,
        "assess_hover_s": 10.0,  # EXPECTANT target: hover at delivery altitude to assess breathing before release
    },
    "camera": {
        "width": 640,
        "height": 480,
        "hfov_deg": 80.0,
        "mount_offset": [0.0, 0.0, -0.10],  # body frame (FLU), metres
        # forward tilt of the gimbal from nadir. 20 deg gives ~31 m look-ahead at
        # 25 m altitude (> 15 m mine radius); a pure nadir camera only sees ~16 m ahead.
        "tilt_deg": 20.0,
        "lane_overlap": 0.8,  # lane spacing = footprint width * this
    },
    "perception": {
        "gate_m": 3.0,
        "gate_sigma_k": 2.0,
        "gate_max_m": 8.0,  # association gate never exceeds this (no track swallows a neighbour)
        # re-identification of STALE confirmed tracks (not seen for reid_stale_s):
        # gate = min(reid_max_m, 3 sigma + reid_base_m). Drifting casualties keep their id.
        "reid_stale_s": 5.0,
        "reid_base_m": 6.0,
        "reid_max_m": 25.0,
        "reid_max_floating_m": 90.0,  # drifting objects: gate grows with uncertainty up to this
        "confirm_hits": 3,
        "confirm_window": 5,
        "delete_after_misses": 4,  # unconfirmed tracks only
        "conf_threshold": {
            "default": 0.25,
            "mine": 0.15,
            "weapon": 0.15,
            "injured": 0.20,
            "hostile_unit": 0.20,
        },
        "pixel_sigma": 3.0,
        "attitude_sigma_rad": 0.02,
        "ambient_k": 295.0,
        "warm_delta_k": 6.0,
        "immobile_s": 10.0,
        "motion_window_s": 15.0,
        "motion_thresh_m": 0.8,
        "mine_proximity_m": 20.0,
        "deceased_delta_k": 2.0,  # skin within this of ambient ...
        "deceased_obs_s": 8.0,  # ... and no breathing seen for this long (close range) -> likely deceased
        "slow_radius_m": 25.0,
    },
    "drift": {
        "current": [0.15, 0.05],  # sea current, m/s (ENU)
        "k_leeway": 0.03,  # fraction of wind speed, downwind
        "q_pos": 0.02,
        "q_vel": 0.0001,
        "max_speed_dev": 0.3,
        "vel_sigma_floor": 0.08,  # honest drift-velocity uncertainty: sigma grows with time unseen
        "meas_sigma_floor": 0.8,  # floating targets: per-detection position noise floor, m  # |v - drift prior| clamp for floating tracks, m/s
        "r_meas": 1.0,
        "max_sigma": 30.0,
        "sea_tau_s": 30.0,  # time constant of the online current estimate
        "vel_sigma0": 0.1,  # prior drift velocity uncertainty, m/s
    },
    "zones": {
        "safety": {"mine": 15.0, "weapon": 10.0, "hostile_unit": 25.0},
        "safety_submerged_mine": 12.0,  # mines found by magnetometer/sonar (below the surface)
        "sigma_k": 2.0,
        "wind_k": 0.5,  # metres of margin per m/s of wind
        "los_factor": 1.6,  # hostile line-of-sight soft ring = radius * this
        "los_cost": 10.0,
        "gnss_lost_extra": 5.0,
        "max_sigma_term_m": 12.0,  # cap on sigma_k * sigma in the zone radius
    },
    "water": {
        # Polygons (ENU) considered water: tracks inside drift with the current.
        "polygons": [[[-1000.0, -1000.0], [1000.0, -1000.0], [1000.0, -20.0], [-1000.0, -20.0]]],
    },
    "obstacles": [],  # static: [{"id": "rock1", "p": [x, y], "r": 4.0, "h": 30.0}]
    "planner": {
        "grid_res": 2.0,
        "drone_radius": 0.6,
        "safety_margin": 3.0,
        "wind_margin_k": 0.3,
        "low_vis_extra": 2.0,
        "replan_move_m": 3.0,
        "plan_extra_m": 2.0,  # planning hysteresis on top of the safety margin
        "evasive_alt": 15.0,  # altitude for waypoints inside a hostile line-of-sight ring
        "search_step": 6.0,
        "search_legs": 8,
        "max_expansions": 60000,
    },
    "energy": {
        "capacity_wh": 90.0,
        "p_hover_w": 180.0,
        "k_speed": 0.012,  # P(v_air) = p_hover * (1 + k_speed * v_air^2)
        "v_min": 1.0,
        "reserve_frac": 0.15,
        "rth_factor": 1.3,
        "rth_factor_gnss_lost": 1.5,
        "land_wh": 0.6,
        "deliver_wh": 0.8,
        "gust_factor": 1.2,
        "critical_pct": 8.0,
        "v_air_max": 16.0,  # airframe airspeed limit (strong headwind -> ground speed drops)
    },
    "reflex": {
        "rate_hz": 50.0,
        "max_speed": 8.0,
        "max_vz": 2.5,
        "lookahead_m": 4.0,
        "kp_pos": 0.8,
        "kp_alt": 0.9,
        "rep_influence_m": 10.0,
        "rep_gain": 30.0,
        "zone_buffer_m": 2.0,
        "wind_ff_alpha": 0.05,
        "wind_ff_max": 4.0,
        "stale_plan_s": 1.5,
        "climb_first_frac": 0.8,
        "descend_radius_m": 3.0,  # only descend to a lower target within this horizontal distance  # below this fraction of the target altitude: climb before transit
        "boat_horizon_s": 2.0,
    },
    "comm": {
        "version": 1,
        "key_hex": "736b79666f7267652d64726f6e617468",  # demo key; override in deployment
        "ring_capacity": 512,
        "max_msg_bytes": 1024,
        "tel_period_s": 1.0,
        "ack_period_s": 2.0,
        "station": [0.0, 0.0],
        "max_range_m": 450.0,
        "shadow_polygons": [],
        "base_loss": 0.0,
        "jam_loss": 1.0,
        "udp": {"drone_port": 14600, "station_port": 14601, "station_host": "127.0.0.1", "drone_host": "127.0.0.1"},
        "http_port": 8080,
    },
    "health": {
        "motor_sat_level": 0.95,
        "motor_sat_s": 1.0,
        "odom_timeout_s": 0.5,
        "cam_timeout_s": 3.0,
        "v_min_cell": 3.3,
        "temp_max_c": 85.0,
    },
    "nav": {
        "gnss_sigma": 1.5,
        "dr_sigma_growth": 0.3,  # m per second of dead reckoning
        "gate_chi2": 13.8,  # 2 dof, p = 0.999
        "spoof_consecutive": 5,
        "recover_consecutive": 20,
        "fix_timeout_s": 1.0,
        "min_sats": 6,
    },
    # Mine detection layer 2: magnetometer (MAD)
    "magnetic": {
        "enabled": True,
        "noise_nt": 0.15,  # platform noise (fluxgate on a boom), 1 sigma
        "k_sigma": 6.0,  # event threshold = k_sigma * noise
        "bg_tau_s": 60.0,
        "close_s": 1.0,
        "fit_radius_m": 45.0,
        "grid_half_m": 30.0,
        "grid_step_m": 2.0,
        "max_depth_m": 20.0,
        "depth_step_m": 2.0,
        "min_r2": 0.5,
        "max_samples": 6000,
        "confirm_conf": 0.6,  # stronger contacts confirm a mine on their own, weaker ones ask for a sonar dip
        "weapon_attribution_m": 5.0,  # a ferrous anomaly on a known weapon is the weapon, not a mine
        "confirm_events": 2,  # ... or this many independent, consistent magnetic events
        "min_event_conf": 0.25,
    },
    # Mine detection layer 3: dipping sonar (winched transducer, drone hovering low over water)
    "sonar": {
        "enabled": True,
        "survey": True,  # sonar stations over water (deep mines invisible to camera and magnetometer)
        "range_m": 60.0,
        "station_spacing_m": 95.0,
        "dip_alt": 3.0,
        "dwell_s": 4.0,
        "task_radius_m": 70.0,  # do a pending dip when it is this close to the drone
        "task_timeout_s": 60.0,
        "standoff_extra_m": 3.0,
        "confirm_conf": 0.5,
        "range_sigma_m": 1.0,
    },
    "orchestrator": {"engine": "langgraph", "rate_hz": 10.0},
    "logging": {"dir": "runs"},
}


def deep_merge(base: dict, override: dict) -> dict:
    out = copy.deepcopy(base)
    for k, v in (override or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = deep_merge(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


def load_params(path: str | None = None, overrides: dict | None = None) -> dict:
    p = copy.deepcopy(DEFAULTS)
    if path:
        import yaml  # lazy: the Raspberry Pi bridge runs without PyYAML when no file is given

        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
        # Accept either a bare tree or a {"skyforge": {...}} wrapper.
        p = deep_merge(p, data.get("skyforge", data))
    if overrides:
        p = deep_merge(p, overrides)
    return p

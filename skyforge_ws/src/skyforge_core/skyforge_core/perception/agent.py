"""PerceptionAgent.

Fuses the jury and team vision modules, converts 2D detections to 3D,
filters false detections (3 hits in 5 in-view frames), runs a drift Kalman
filter per track, assigns START categories and publishes exclusion zones.

Never blocks: the orchestrator just reads `.casualties()` / `.zones()`.
"""
from __future__ import annotations

import math
from collections import deque
from typing import Optional

from ..geometry import dist2, in_any_polygon
from ..domain import DANGER_CLASSES, Cls, Detection, Start, Track, URGENCY, Zone
from .drift import DriftKF, drift_prior
from .projection import Camera, ground_sigma, in_view, pixel_to_ground
from .start_triage import classify


class _TrackState:
    __slots__ = ("track", "kf", "hist", "miss_streak", "obs", "attrs_hist", "last_t")

    def __init__(self, track: Track, kf: DriftKF, window: int):
        self.track = track
        self.kf = kf
        self.hist: deque[int] = deque([1], maxlen=window)
        self.miss_streak = 0
        self.obs: deque[tuple[float, float, float]] = deque(maxlen=200)  # (t, x, y)
        self.attrs_hist: dict = {}
        self.last_t = track.last_seen


class PerceptionAgent:
    def __init__(self, params: dict, camera: Optional[Camera] = None):
        self.p = params
        self.pp = params["perception"]
        self.dp = params["drift"]
        self.zp = params["zones"]
        self.cam = camera or Camera.from_params(params)
        self.water = params["water"]["polygons"]
        self._tracks: dict[str, _TrackState] = {}
        self._next = {c: 1 for c in (Cls.DEBRIS, Cls.MINE, Cls.WEAPON, Cls.INJURED, Cls.HOSTILE)}
        self._new_confirmed: list[Track] = []
        self._t = 0.0
        self.wind = (0.0, 0.0)
        self.stats = {"frames": 0, "dets_in": 0, "dets_low_conf": 0, "dets_no_ground": 0, "tracks_created": 0, "tracks_deleted": 0}

    # ------------------------------------------------------------------ io
    def set_camera(self, cam: Camera) -> None:
        self.cam = cam

    def tracks(self) -> list[Track]:
        return [s.track for s in self._tracks.values()]

    def confirmed(self) -> list[Track]:
        return [s.track for s in self._tracks.values() if s.track.confirmed]

    def casualties(self) -> list[Track]:
        return [t for t in self.confirmed() if t.cls == Cls.INJURED]

    def pop_new_confirmed(self) -> list[Track]:
        out, self._new_confirmed = self._new_confirmed, []
        return out

    def track(self, tid: str) -> Optional[Track]:
        s = self._tracks.get(tid)
        return s.track if s else None

    def kf(self, tid: str) -> Optional[DriftKF]:
        s = self._tracks.get(tid)
        return s.kf if s else None

    # ------------------------------------------------------------ pipeline
    def predict(self, now: float, wind=None) -> None:
        if wind is not None:
            self.wind = tuple(wind)
        dt = now - self._t
        if dt <= 0:
            return
        for s in self._tracks.values():
            s.kf.predict(dt)
            self._sync(s)
        self._t = now

    def process(self, detections: list[Detection], pos, rpy, stamp: float, wind=None) -> None:
        """One camera frame worth of detections (jury + team, already merged)."""
        self.predict(stamp, wind)
        self.stats["frames"] += 1
        alt = pos[2]
        meas: list[tuple[Detection, tuple[float, float], float]] = []
        thr = self.pp["conf_threshold"]
        for d in detections:
            self.stats["dets_in"] += 1
            if d.conf < thr.get(d.cls, thr["default"]):
                self.stats["dets_low_conf"] += 1
                continue
            u, v = d.bottom_center if d.cls != Cls.INJURED else d.center
            g = pixel_to_ground(u, v, pos, rpy, self.cam)
            if g is None:
                self.stats["dets_no_ground"] += 1
                continue
            sig = ground_sigma(u, v, alt, self.cam, self.pp["pixel_sigma"], self.pp["attitude_sigma_rad"])
            meas.append((d, g, sig))

        # --- association: greedy nearest neighbour per class, gated
        pairs = []
        for mi, (d, g, sig) in enumerate(meas):
            for tid, s in self._tracks.items():
                if s.track.cls != d.cls:
                    continue
                dd = dist2(g, s.kf.pos)
                gate = min(self.pp["gate_max_m"], max(self.pp["gate_m"], self.pp["gate_sigma_k"] * math.hypot(s.kf.sigma, sig)))
                if dd <= gate:
                    pairs.append((dd, mi, tid))
        pairs.sort()
        used_m, used_t = set(), set()
        for dd, mi, tid in pairs:
            if mi in used_m or tid in used_t:
                continue
            used_m.add(mi)
            used_t.add(tid)
            self._update(self._tracks[tid], *meas[mi], stamp)

        # --- misses: only count frames where the track should have been visible
        for tid, s in list(self._tracks.items()):
            if tid in used_t:
                continue
            if alt > 2.0 and in_view((*s.kf.pos, 0.0), pos, rpy, self.cam, margin_px=10):
                s.hist.append(0)
                s.miss_streak += 1
                s.track.attrs["missed_in_view"] = s.track.attrs.get("missed_in_view", 0) + 1
                if not s.track.confirmed and s.miss_streak >= self.pp["delete_after_misses"]:
                    del self._tracks[tid]
                    self.stats["tracks_deleted"] += 1

        # --- re-identification: unmatched detections vs STALE confirmed tracks
        # (a drifting casualty re-seen after a long gap keeps its identity)
        pp = self.pp
        pairs = []
        for mi, (d, g, sig) in enumerate(meas):
            if mi in used_m:
                continue
            for tid, s in self._tracks.items():
                tr = s.track
                if tid in used_t or tr.cls != d.cls or not tr.confirmed or stamp - tr.last_seen < pp["reid_stale_s"]:
                    continue
                if tr.floating != in_any_polygon(g, self.water):
                    continue  # water/land status must match
                gate = min(pp["reid_max_m"], 3.0 * s.kf.sigma + pp["reid_base_m"])
                dd = dist2(g, s.kf.pos)
                if dd <= gate:
                    pairs.append((dd, mi, tid))
        pairs.sort()
        for dd, mi, tid in pairs:
            if mi in used_m or tid in used_t:
                continue
            used_m.add(mi)
            used_t.add(tid)
            self.stats["reid"] = self.stats.get("reid", 0) + 1
            self._update(self._tracks[tid], *meas[mi], stamp)

        # --- births
        for mi, (d, g, sig) in enumerate(meas):
            if mi not in used_m:
                self._birth(d, g, sig, stamp)

        self._refresh_categories(stamp)

    # ------------------------------------------------------------ internals
    def _birth(self, d: Detection, g, sig: float, stamp: float) -> None:
        floating = in_any_polygon(g, self.water)
        prior = drift_prior(self.dp["current"], self.wind, self.dp["k_leeway"]) if floating else (0.0, 0.0)
        kf = DriftKF(g, sig, floating, prior, self.dp["q_pos"], self.dp["q_vel"], self.dp["max_sigma"], self.dp["vel_sigma0"],
                     self.dp["max_speed_dev"], self.dp["vel_sigma_floor"], self.dp["meas_sigma_floor"])
        prefix = {Cls.INJURED: "C", Cls.MINE: "M", Cls.WEAPON: "W", Cls.HOSTILE: "H", Cls.DEBRIS: "D"}[d.cls]
        tid = f"{prefix}{self._next[d.cls]}"
        self._next[d.cls] += 1
        tr = Track(tid, d.cls, g, sig, kf.vel, d.conf, 1, False, stamp, stamp, floating)
        st = _TrackState(tr, kf, self.pp["confirm_window"])
        st.obs.append((stamp, g[0], g[1]))
        self._merge_attrs(st, d)
        self._tracks[tid] = st
        self.stats["tracks_created"] += 1
        self._check_confirm(st)

    def _update(self, s: _TrackState, d: Detection, g, sig: float, stamp: float) -> None:
        s.kf.update(g, sig)
        s.hist.append(1)
        s.miss_streak = 0
        tr = s.track
        tr.hits += 1
        tr.conf = 0.7 * tr.conf + 0.3 * d.conf if tr.hits > 1 else d.conf
        tr.last_seen = stamp
        tr.attrs["missed_in_view"] = 0
        s.obs.append((stamp, g[0], g[1]))
        self._merge_attrs(s, d)
        self._sync(s)
        self._check_confirm(s)

    def _merge_attrs(self, s: _TrackState, d: Detection) -> None:
        for k, v in d.attrs.items():
            s.track.attrs[k] = v
        # breathing micro-motion (thermal/RGB at close range): track how long it has been ABSENT
        if "breathing" in d.attrs:
            if d.attrs["breathing"]:
                s.track.attrs.pop("no_breath_since", None)
            else:
                s.track.attrs.setdefault("no_breath_since", d.stamp)
        if "detection_radius" in d.attrs:
            s.track.attrs["detection_radius"] = float(d.attrs["detection_radius"])

    def _check_confirm(self, s: _TrackState) -> None:
        if not s.track.confirmed and sum(s.hist) >= self.pp["confirm_hits"]:
            s.track.confirmed = True
            self._new_confirmed.append(s.track)

    def _sync(self, s: _TrackState) -> None:
        s.track.pos = s.kf.pos
        s.track.vel = s.kf.vel
        s.track.sigma = s.kf.sigma

    def _moving(self, s: _TrackState, now: float) -> Optional[bool]:
        if "moving" in s.track.attrs:
            return bool(s.track.attrs["moving"])
        win = self.pp["motion_window_s"]
        pts = [(x, y) for (t, x, y) in s.obs if now - t <= win]
        if len(pts) < 4:
            return None
        if s.track.floating:
            # remove the drift component before judging motion
            vx, vy = s.kf.vel
            pts = [(x - vx * (t - now), y - vy * (t - now)) for (t, x, y) in s.obs if now - t <= win]
        mx = sum(p[0] for p in pts) / len(pts)
        my = sum(p[1] for p in pts) / len(pts)
        spread = max(math.hypot(p[0] - mx, p[1] - my) for p in pts)
        return spread > self.pp["motion_thresh_m"] + 2.0 * s.kf.sigma

    def _refresh_categories(self, now: float) -> None:
        mines = [s.kf.pos for s in self._tracks.values() if s.track.cls == Cls.MINE and s.track.confirmed]
        for s in self._tracks.values():
            tr = s.track
            if tr.cls != Cls.INJURED:
                continue
            near_mine = any(dist2(tr.pos, m) <= self.pp["mine_proximity_m"] for m in mines)
            cat, why = classify(
                tr.attrs.get("temp_k"),
                tr.attrs.get("posture"),
                self._moving(s, now),
                now - tr.first_seen,
                tr.floating,
                bool(tr.attrs.get("flotation", False)),
                near_mine,
                self.pp["ambient_k"],
                self.pp["warm_delta_k"],
                self.pp["immobile_s"],
                no_breath_s=(now - tr.attrs["no_breath_since"]) if "no_breath_since" in tr.attrs else 0.0,
                deceased_delta_k=self.pp["deceased_delta_k"],
                deceased_obs_s=self.pp["deceased_obs_s"],
            )
            tr.attrs["likely_deceased"] = cat == Start.DECEASED
            if cat != tr.category:
                tr.attrs["category_why"] = why
            tr.category = cat
            tr.urgency = URGENCY[cat]

    # --------------------------------------------------------------- zones
    def zones(self, extra_margin: float = 0.0) -> list[Zone]:
        wind = math.hypot(*self.wind)
        out = []
        for s in self._tracks.values():
            tr = s.track
            if tr.cls not in DANGER_CLASSES or not tr.confirmed:
                continue
            base = tr.attrs.get("detection_radius", self.zp["safety"][tr.cls]) if tr.cls == Cls.HOSTILE else self.zp["safety"][tr.cls]
            sig_term = min(self.zp["sigma_k"] * s.kf.sigma, self.zp["max_sigma_term_m"])
            r = base + sig_term + self.zp["wind_k"] * wind + extra_margin
            soft = r * self.zp["los_factor"] if tr.cls == Cls.HOSTILE else 0.0
            out.append(Zone(f"Z{tr.id}", tr.cls, s.kf.pos, r, s.kf.vel, soft))
        return out

    def provisional_zones(self) -> list[tuple[tuple[float, float], float]]:
        """Unconfirmed danger detections as (center, safety radius) for the REFLEX
        veto only (recall first for mines/weapons). Never used for decisions."""
        out = []
        for s in self._tracks.values():
            tr = s.track
            if tr.cls in DANGER_CLASSES and not tr.confirmed:
                base = self.zp["safety"][tr.cls]
                out.append((s.kf.pos, base + min(self.zp["sigma_k"] * s.kf.sigma, 4.0)))
        return out

    def slow_down(self, pos) -> bool:
        """Unconfirmed danger detection nearby: slow down to look again."""
        R = self.pp["slow_radius_m"]
        return any(
            (not s.track.confirmed) and s.track.cls in DANGER_CLASSES and dist2(pos, s.kf.pos) < R
            for s in self._tracks.values()
        )

    def seed_tracks(self, tracks: list[Track], now: float) -> None:
        """MAYDAY handover: start with the failing drone's confirmed tracks."""
        for tr in tracks:
            kf = DriftKF(tr.pos, max(tr.sigma, 1.0), tr.floating, tr.vel, self.dp["q_pos"], self.dp["q_vel"], self.dp["max_sigma"])
            age = max(0.0, now - tr.last_seen) if tr.last_seen else 0.0
            kf.predict(age)
            tr.confirmed = True
            st = _TrackState(tr, kf, self.pp["confirm_window"])
            st.hist.extend([1] * self.pp["confirm_window"])
            self._tracks[tr.id] = st
            self._sync(st)
            pre = tr.id[0]
            cls = {"C": Cls.INJURED, "M": Cls.MINE, "W": Cls.WEAPON, "H": Cls.HOSTILE, "D": Cls.DEBRIS}.get(pre)
            if cls and tr.id[1:].isdigit():
                self._next[cls] = max(self._next[cls], int(tr.id[1:]) + 1)
        self._t = now

    def snapshot(self) -> dict:
        return {"tracks": [t.to_dict() for t in self.confirmed()], "zones": [z.to_dict() for z in self.zones()]}

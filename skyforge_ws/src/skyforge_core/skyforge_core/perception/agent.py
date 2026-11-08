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
from .drift import DriftKF
from .magnetic import MagAnomalyDetector
from .sea_state import SeaStateEstimator
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
        self.sea = SeaStateEstimator(self.dp["current"], self.dp["k_leeway"], self.dp.get("sea_tau_s", 30.0))
        self.mag = MagAnomalyDetector(params) if params["magnetic"]["enabled"] else None
        self.rejected: list[dict] = []  # phantom mine cues rejected by inspection / sonar
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
        # re-centre floating tracks on the current sea-state estimate: unseen ones
        # follow a changing current instead of keeping a stale velocity
        prior = self.sea.prior(self.wind)
        a = 1.0 - math.exp(-dt / max(self.dp.get("sea_tau_s", 30.0), 1e-3))
        for s in self._tracks.values():
            if s.kf.floating:
                s.kf.prior = prior
                if now - s.track.last_seen > self.pp.get("reid_stale_s", 5.0):
                    s.kf.x[2] += a * (prior[0] - s.kf.x[2])
                    s.kf.x[3] += a * (prior[1] - s.kf.x[3])
            s.kf.predict(dt)
            self._sync(s)
        self._t = now

    def _update_sea_state(self, now: float) -> None:
        samples = []
        for s in self._tracks.values():
            tr = s.track
            if tr.floating and tr.confirmed and now - tr.last_seen < 1.0 and tr.hits >= self.sea.min_obs:
                samples.append((s.kf.x[2], s.kf.x[3], max(s.kf.P[2, 2], s.kf.P[3, 3])))
        self.sea.update(now, samples, self.wind)

    @property
    def sea_current(self) -> tuple[float, float]:
        return (self.sea.current[0], self.sea.current[1])

    def process(self, detections: list[Detection], pos, rpy, stamp: float, wind=None) -> None:
        """One camera frame worth of detections (jury + team, already merged)."""
        self.predict(stamp, wind)
        self.stats["frames"] += 1
        alt = pos[2]
        if alt < self.pp.get("min_detect_alt", 0.0):
            self.stats["dets_low_alt"] = self.stats.get("dets_low_alt", 0) + len(detections)
            return
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

        # --- association: greedy nearest neighbour per class, gated. The minimum gate
        # scales with altitude (ground sample distance): at 3 m the whole image covers a
        # few metres, a fixed 3 m gate would merge every random false positive into one track
        gate_floor = min(self.pp["gate_m"], max(self.pp["gate_min_m"], self.pp["gate_per_alt"] * alt))
        pairs = []
        for mi, (d, g, sig) in enumerate(meas):
            for tid, s in self._tracks.items():
                if s.track.cls != d.cls:
                    continue
                dd = dist2(g, s.kf.pos)
                # floating targets move with the waves: never gate tighter than their motion noise
                sig_d = max(sig, self.dp["meas_sigma_floor"]) if s.kf.floating else sig
                gate = min(self.pp["gate_max_m"], max(gate_floor, self.pp["gate_sigma_k"] * math.hypot(s.kf.sigma, sig_d)))
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
            layers = s.track.attrs.get("layers")
            if layers and "vision" not in layers:
                continue  # magnetometer / sonar contact: the camera cannot see it, a camera miss is no evidence
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
                if tr.floating != in_any_polygon(g, self.water) and dist2(g, s.kf.pos) > 10.0:
                    continue  # water/land status must match (except right at the shoreline: washed ashore)
                gmax = pp["reid_max_floating_m"] if tr.floating else pp["reid_max_m"]
                gate = min(gmax, 3.0 * s.kf.sigma + pp["reid_base_m"])
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
            st = self._tracks[tid]
            g = meas[mi][1]
            long_v = None
            if st.track.floating and st.obs:
                # long-baseline drift measurement: displacement since last seen / time unseen
                t_last, x_last, y_last = st.obs[-1]
                gap = stamp - t_last
                if gap >= self.pp["reid_stale_s"]:
                    long_v = ((g[0] - x_last) / gap, (g[1] - y_last) / gap)
                    var = (2.0 * self.dp["meas_sigma_floor"]) ** 2 / gap**2 + 0.02**2
                    self.sea.update_baseline(long_v, var, self.wind)
            self._update(st, *meas[mi], stamp)
            if long_v is not None:
                st.kf.x[2], st.kf.x[3] = long_v  # best estimate of this object's drift
                st.kf.prior = long_v

        # --- births
        for mi, (d, g, sig) in enumerate(meas):
            if mi not in used_m:
                self._birth(d, g, sig, stamp)

        self._refresh_categories(stamp)
        self._update_sea_state(stamp)
        self._expire_cues(stamp)

    def _expire_cues(self, now: float) -> None:
        ttl = self.p["magnetic"]["cue_ttl_s"]
        for tid, s in list(self._tracks.items()):
            tr = s.track
            layers = tr.attrs.get("layers")
            if not tr.confirmed and layers and "vision" not in layers and now - tr.last_seen > ttl:
                self._reject_mine_cue(tid, now, f"unconfirmed for {ttl:.0f}s")

    # ------------------------------------------------------------ internals
    def _birth(self, d: Detection, g, sig: float, stamp: float) -> None:
        # in water if the coastline map says so OR the detector's posture class does (injured_water)
        floating = in_any_polygon(g, self.water) or bool(d.attrs.get("in_water", False))
        prior = self.sea.prior(self.wind) if floating else (0.0, 0.0)
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
        min_conf = self.pp["confirm_min_conf"].get(s.track.cls, 0.0)
        if not s.track.confirmed and sum(s.hist) >= self.pp["confirm_hits"] and s.track.conf >= min_conf:
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
            if tr.cls == Cls.HOSTILE:
                base = tr.attrs.get("detection_radius", self.zp["safety"][tr.cls])
            elif tr.cls == Cls.MINE and tr.attrs.get("submerged"):
                base = self.zp["safety_submerged_mine"]
            else:
                base = self.zp["safety"][tr.cls]
            sig_term = min(self.zp["sigma_k"] * s.kf.sigma, self.zp["max_sigma_term_m"])
            r = base + sig_term + self.zp["wind_k"] * wind + extra_margin
            soft = r * self.zp["los_factor"] if tr.cls == Cls.HOSTILE else 0.0
            out.append(Zone(f"Z{tr.id}", tr.cls, s.kf.pos, r, s.kf.vel, soft))
        return out

    # ------------------------------------------------- mine layers 2 and 3
    def add_mine_contact(self, layer: str, pos, depth: float, sigma: float, conf: float, stamp: float) -> Optional[Track]:
        """Fuse a magnetometer ("mag") or sonar ("sonar") contact into the mine tracks.

        - a magnetic anomaly on a known weapon is the weapon (ferrous), not a mine;
        - otherwise associate to an existing mine track (any layer) or create one;
        - strong contacts confirm on their own, weak magnetic ones stay tentative
          and are turned into sonar-dip requests by the brain."""
        mp, sp = self.p["magnetic"], self.p["sonar"]
        if layer == "mag":
            for s in self._tracks.values():
                if s.track.cls == Cls.WEAPON and s.track.confirmed and dist2(s.kf.pos, pos) <= mp["weapon_attribution_m"]:
                    s.track.attrs.setdefault("layers", [])
                    if "mag" not in s.track.attrs["layers"]:
                        s.track.attrs["layers"].append("mag")
                    return None
        best, best_d = None, 1e9
        for s in self._tracks.values():
            if s.track.cls != Cls.MINE:
                continue
            d = dist2(s.kf.pos, pos)
            if d <= max(4.0, 3.0 * math.hypot(s.kf.sigma, sigma)) and d < best_d:
                best, best_d = s, d
        need = mp["confirm_conf"] if layer == "mag" else sp["confirm_conf"]
        if best is None:
            in_water = in_any_polygon(pos, self.water)
            kf = DriftKF(pos, max(sigma, 0.5), False)  # bottom / moored mine: static
            tid = f"M{self._next[Cls.MINE]}"
            self._next[Cls.MINE] += 1
            tr = Track(tid, Cls.MINE, tuple(pos), sigma, (0.0, 0.0), conf, 1, False, stamp, stamp, False)
            tr.attrs.update(layers=[layer], depth=round(depth, 1), submerged=depth > 0.5, in_water=in_water)
            best = _TrackState(tr, kf, self.pp["confirm_window"])
            self._tracks[tid] = best
            self.stats["tracks_created"] += 1
        else:
            best.kf.update(pos, max(sigma, 0.5))
            tr = best.track
            tr.hits += 1
            tr.conf = max(tr.conf, conf)
            tr.last_seen = stamp
            layers = tr.attrs.setdefault("layers", ["vision"])
            if layer not in layers:
                layers.append(layer)
            if "depth" not in tr.attrs or layer == "sonar":  # sonar gives the best depth
                tr.attrs["depth"] = round(depth, 1)
                tr.attrs["submerged"] = depth > 0.5
            self._sync(best)
        if not tr.confirmed and (conf >= need or len(tr.attrs.get("layers", [])) >= 2):
            tr.confirmed = True
            self._new_confirmed.append(tr)
        return tr

    def add_mag_sample(self, t: float, sensor_pos, total_field_nt: float) -> Optional[Track]:
        """Layer 2 input: one magnetometer sample (sensor_pos = x, y, height above surface)."""
        if self.mag is None:
            return None
        c = self.mag.add_sample(t, sensor_pos, total_field_nt)
        if c is None:
            return None
        return self._apply_mag_contact(c, t)

    def _apply_mag_contact(self, c, t: float) -> Optional[Track]:
        # a new inversion uses every sample around the anomaly (superset of the previous
        # passes): it SUPERSEDES an earlier tentative magnetic estimate nearby
        for s in self._tracks.values():
            tr = s.track
            if (tr.cls == Cls.MINE and tr.attrs.get("layers") == ["mag"]
                    and dist2(s.kf.pos, c.pos) <= self.p["magnetic"]["fit_radius_m"] * 0.8
                    and dist2(s.kf.pos, c.pos) <= max(8.0, 3.0 * math.hypot(s.kf.sigma, c.sigma))):  # same source
                # independent magnetic events that agree count as repeated evidence
                agree = dist2(s.kf.pos, c.pos) <= max(6.0, 2.0 * math.hypot(s.kf.sigma, c.sigma))
                tr.attrs["mag_events"] = tr.attrs.get("mag_events", 1) + (1 if agree else 0)
                s.kf = DriftKF(c.pos, max(c.sigma, 0.5), False)
                tr.attrs.update(depth=round(c.depth, 1), single_pass=c.single_pass)
                tr.conf, tr.last_seen, tr.hits = max(c.conf, tr.conf), t, tr.hits + 1
                self._sync(s)
                mp = self.p["magnetic"]
                repeated = tr.attrs["mag_events"] >= mp["confirm_events"] and c.conf >= mp["min_event_conf"]
                if not tr.confirmed and (c.conf >= mp["confirm_conf"] or repeated):
                    tr.confirmed = True
                    self._new_confirmed.append(tr)
                self.mag.register(tr.id, c.pos, c.depth, c.amplitude)
                return tr
        tr = self.add_mine_contact("mag", c.pos, c.depth, c.sigma, c.conf, t)
        # the source is explained (new mine, existing mine or a weapon): peel it from future fits
        self.mag.register(tr.id if tr is not None else f"ferrous@{c.pos[0]:.0f},{c.pos[1]:.0f}", c.pos, c.depth, c.amplitude)
        if tr is not None:
            tr.attrs["single_pass"] = c.single_pass
        return tr

    def add_sonar_ping(self, dip_pos, contacts, stamp: float) -> list[Track]:
        """Layer 3 input: contacts of one sonar dip, each {range, bearing (rad, ENU), depth, strength}."""
        out = []
        sig = self.p["sonar"]["range_sigma_m"]
        for c in contacts:
            r, b = float(c["range"]), float(c["bearing"])
            pos = (dip_pos[0] + r * math.cos(b), dip_pos[1] + r * math.sin(b))
            if not in_any_polygon(pos, self.water):
                continue  # a sonar cannot see under land: clutter / multipath
            tr = self.add_mine_contact("sonar", pos, float(c.get("depth", 5.0)), sig + 0.02 * r, float(c.get("strength", 0.8)), stamp)
            if tr is not None:
                out.append(tr)
        # negative evidence: a tentative magnetic cue in water, well inside sonar range, that the
        # ping did not confirm is a phantom
        rng = self.p["sonar"]["range_m"]
        hit = {tr.id for tr in out}
        for tid, s in list(self._tracks.items()):
            tr = s.track
            if (tr.cls == Cls.MINE and not tr.confirmed and tr.attrs.get("layers") == ["mag"] and tid not in hit
                    and in_any_polygon(s.kf.pos, self.water) and dist2(s.kf.pos, dip_pos) <= 0.8 * rng):
                self._reject_mine_cue(tid, stamp, "sonar sees nothing there")
        return out

    def mag_requests(self) -> list[tuple[str, tuple[float, float]]]:
        """Tentative magnetic contacts on LAND (sonar impossible): ask for a low magnetic inspection."""
        if not self.p["magnetic"].get("inspect", True):
            return []
        return [(s.track.id, s.kf.pos) for s in self._tracks.values()
                if s.track.cls == Cls.MINE and not s.track.confirmed and s.track.attrs.get("layers") == ["mag"]
                and not in_any_polygon(s.kf.pos, self.water)]

    def mag_reinspect(self, tid: str, t: float) -> Optional[Track]:
        """After an inspection hover: re-invert with every sample around the contact (incl. close range)."""
        st = self._tracks.get(tid)
        if self.mag is None or st is None:
            return None
        near = [abs(x[4]) for x in self.mag.samples if dist2((x[1], x[2]), st.kf.pos) <= self.mag.fit_r]
        if not near:
            return None
        c = self.mag.invert(st.kf.pos, max(near), t)
        if c is None or c.conf < self.p["magnetic"]["confirm_conf"] or dist2(c.pos, st.kf.pos) > 3.0 * max(st.kf.sigma, 4.0):
            # close range shows no strong source there: the weak cue was a phantom -> reject it
            self._reject_mine_cue(tid, t, "no strong magnetic source at close range")
            return None
        return self._apply_mag_contact(c, t)

    def _reject_mine_cue(self, tid: str, t: float, why: str) -> None:
        st = self._tracks.pop(tid, None)
        if st is None:
            return
        if self.mag is not None:
            self.mag.known.pop(tid, None)  # stop peeling a source that does not exist
        self.rejected.append({"t": round(t, 2), "id": tid, "pos": [round(v, 1) for v in st.kf.pos], "why": why})
        self.stats["mine_cues_rejected"] = self.stats.get("mine_cues_rejected", 0) + 1

    def sonar_requests(self) -> list[tuple[str, tuple[float, float]]]:
        """Tentative magnetic mine contacts that need a sonar dip to confirm/deny."""
        return [(s.track.id, s.kf.pos) for s in self._tracks.values()
                if s.track.cls == Cls.MINE and not s.track.confirmed and "mag" in s.track.attrs.get("layers", [])
                and in_any_polygon(s.kf.pos, self.water)]  # sonar only works in water

    def provisional_zones(self) -> list[tuple[tuple[float, float], float]]:
        """Unconfirmed danger detections as (center, safety radius) for the REFLEX
        veto only (recall first for mines/weapons). Never used for decisions."""
        out = []
        for s in self._tracks.values():
            tr = s.track
            layers = tr.attrs.get("layers")
            if layers and "vision" not in layers:
                continue  # buried / submerged cue: no hazard to the aircraft, not a registered zone
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

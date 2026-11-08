# SKYFORGE architecture: brain, agents, communication

The deliberative logic is written once, in **pure Python** (`skyforge_core`, no `rclpy`). It is used three
ways:

1. by the ROS 2 Jazzy nodes (`skyforge_ros`) in Gazebo / PX4,
2. by the headless 2D mission simulator (`skyforge_core.sim`), which is also the backup demo and the KPI bench,
3. by the unit tests and by a closed-loop test that runs the *real* ROS nodes on a fake rclpy bus.

Every node therefore "runs alone with a fake publisher" (team rule 2), and every decision can be replayed
and measured on any laptop.

## Layers and timing

| Layer | Rate | Code | Budget (spec) | Measured (sim, p95) |
|---|---|---|---|---|
| Reflex | 50 Hz | `reflex/controller.py`, `skyforge_ros/reflex_node.py` | < 50 ms | ~0.1 ms |
| Decision (LangGraph, no LLM) | 10 Hz + immediate tick on `goal_update` | `brain/` | ≤ 100 ms | 3–7 ms |
| Perception agent | async, 5–10 Hz | `perception/` | not blocking | ~1–2 ms per frame (excl. CNN) |
| Total chain (frame → decision → command) | | | ≤ 150 ms | < 10 ms + CNN inference |

## Decision layer: the LangGraph StateGraph

```
START → ingest → router ─(phase)→ <phase handler> ─(phase changed? ≤4 hops)→ <next handler> … → planner → END
```

- **ingest** updates coverage (the camera footprint centre, shifted forward by the gimbal tilt) and the
  dynamic RTH threshold.
- **router** checks interrupts in priority order:
  1. MAYDAY (health fault)
  2. battery ≤ RTH threshold → RETURN_BASE
  3. GNSS lost / spoofed → margins + 5 m, RTH factor 1.3 → 1.5
  4. link lost → autonomous flag
  5. `goal_update` → applied in this tick
  6. a new or moved threat zone crossing the current path → replan
- **Phase handlers** are INIT, EXPLORE, DETECT, TRIAGE, NAVIGATE, DELIVER, RETURN_BASE, LANDED and MAYDAY.
  Instant transitions are chained inside one tick (EXPLORE → DETECT → TRIAGE → NAVIGATE), so a confirmed
  casualty is acted on in the tick where it is confirmed.
- **planner** handles A* with tiered margins: hysteresis (+2 m), then nominal, then minimum (drone radius + 1 m,
  still a hard constraint). It replans only when the goal moves more than 3 m, the zones change, or a
  replan is requested.
- The node functions do not import LangGraph. `engine: fsm` runs the same functions through a dict dispatch,
  and a test asserts both engines make identical decisions.
- `DroneState` (`brain/state.py`) holds position, velocity, attitude, battery, link, GNSS/nav source, wind,
  visibility, casualties, zones, served/failed ids, target, current goal, waypoints, phase, interrupts,
  RTH threshold, kits and events.

### Formulas (every input and output is logged in mission_logs)

**Triage, F6** (`planning/triage.py`):

```
S_i = 0.7·U_i·c_i + 0.3·P_i   if the energy gate passes, else 0
P_i = 1 − d_i/d_max
```

- U is the START urgency: IMMEDIATE 1.0, DELAYED 0.7, MINOR 0.3, EXPECTANT 0.1; unknown is treated as 0.7.
- c is the detection confidence, d the distance to the drift intercept point.
- Ties go to the earliest detection.
- **Energy gate:** `factor·(E_go + E_deliver + E_back + E_land + E_res) ≤ E_avail`. It uses the same factor as
  RTH, so choosing a casualty can never immediately trip RTH after the delivery (no oscillation).

**RTH, F11** (`energy.py`), recomputed every tick:

- `E_RTH = P(v_air)·d_home / v_g + E_land + E_res`, with the ground speed `v_g` taken from the full wind
  triangle (cross-wind included) using gust-inflated wind.
- RETURN_BASE is forced when `E_avail ≤ 1.3·E_RTH`, or `1.5·E_RTH` without GNSS.
- The reflex also triggers its own RTH at the critical battery floor.

**Delivery, F7:**

- Fly to the drift intercept point (3 fixed-point iterations), then descend to 5 m.
- Release only when **estimated offset + 0.5σ_track ≤ 1.0 m** (spec radius 1.5 m), the track was seen
  within 1.5 s, the relative speed is ≤ 0.6 m/s, GNSS is trusted, and all of this holds for 3 consecutive
  ticks.
- Otherwise retry, then abort after 3 attempts. An off-target release is impossible by construction.

## PerceptionAgent (`perception/agent.py`)

Pipeline: detector JSON (adapter), then the bbox point (bottom-centre, or centre for people), then a ray through
the intrinsics, the gimbal (with forward tilt) and the attitude, intersected with z = 0. After that:

- **Association** is greedy nearest-neighbour per class. The gate is `min(8 m, max(3 m, 2·√(σ_track² + σ_det²)))`,
  so an uncertain track can never swallow a neighbour.
- **Confirmation** needs 3 hits in 5 frames where the track *should* be visible (a miss only counts when the
  predicted position is inside the image). Unconfirmed tracks are deleted after 4 misses in view.
- **Re-identification:** a detection that matches nothing is compared with *stale* confirmed tracks (unseen for
  ≥ 5 s) of the same class and the same water/land status, using a gate of `min(25 m, 3σ + 6 m)`. A drifting
  casualty keeps its id.
- **Drift KF** per track, state `[x, y, vx, vy]`. Floating tracks start at the prior
  `v = current + k_leeway·wind`. The velocity is bounded to ±0.15 m/s around the prior, and its variance has a
  floor so σ grows honestly while the track is unseen. Land tracks are static.
- **START triage** uses heat, posture and motion, with one step up when near a mine.
- **Zones:**
  - radius = class safety distance (mine 15 m, weapon 10 m, hostile = declared radius, default 25 m)
    + min(2σ, 12 m) + 0.5 m per m/s of wind;
  - hostile units also get a line-of-sight ring (1.6 × radius) where A* cost is ×10;
  - **provisional zones** come from unconfirmed mine/weapon/hostile detections and go to the reflex only, as an
    inward veto, never as an escape manoeuvre. This puts recall first without letting false positives push the
    drone around.

## Communication (no agent): `comm/`

- Signed compact JSON, HMAC-SHA256 truncated to 8 bytes, with `seq` and boot id. Every message is < 1 KB,
  asserted at encode time. Larger bodies (MAYDAY / sync snapshots) are fragmented automatically.
- The priority ring buffer holds 512 messages and evicts telemetry first. Unacked messages are kept, and the
  station acks `last`/`max`. On reconnect the drone replays the gap in order, or sends `sync` with a snapshot
  and the lost-message count.
- RTT uses an NTP-style echo in the acks.
- The comm module runs in-process in `comm_node` (`mode:=local`), or on the Raspberry Pi as a stdlib-only UDP
  bridge (`mode:=pi`, see `deploy/raspberry_pi`).

## Resilience

| Threat | Detection | Response | Measured in sim |
|---|---|---|---|
| Radio jamming / shadow zone | link simulator (polygons, range, jam) | autonomous flag, everything buffered, replay on reconnect | 20 s outage → resync 0.02 s, 0 lost |
| GNSS spoofing | low-gain inertial reference not dragged by a ramp; χ² innovation gate, 5 consecutive | reject GNSS, roll the estimate back to the reference, RTH ×1.5, margins +5 m, no blind release | 40 m ramp detected in 1.9 s |
| Forged / replayed command | HMAC + per-session seq | rejected, `forged_rejected` / `replay_rejected` logged | 100 % rejected |
| Motor / sensor failure | motor saturation > 1 s, odometry timeout, voltage sag, overheating | MAYDAY: snapshot (fragmented < 1 KB), controlled landing ≥ 10 m from casualties and outside zones | drone 2 resumes from the station's snapshot, skips served casualties |
| Wind gust | wind estimate | RTH threshold and margins grow, reflex wind feed-forward | threshold jumps on the gauge |

## What we added to the SKYFORGE plan (and why)

Each item came out of a failure found in the 2D simulator, not out of theory:

1. **Gimbal tilted 20° forward (`camera.tilt_deg`).** A nadir camera at 25 m sees only about 16 m ahead,
   barely more than the 15 m mine radius. The sim showed zone entries before the mine was even detected.
   With the tilt we see about 31 m ahead. **Conception: please put this on the 2-axis gimbal.**
2. **Climb before transit** after a delivery. At 5–10 m the footprint is too small to see danger zones.
3. **Provisional zones** (recall first) for unconfirmed danger detections, used in the reflex only.
4. **Re-identification and duplicate suppression.** A drifting casualty re-detected minutes later was getting a
   second kit (off-target penalty). Fixed in both perception and triage, and carried through the MAYDAY
   snapshot (served flag).
5. **Bounded association gate and an honest drift-velocity uncertainty.** Without them, uncertain tracks
   swallowed neighbours and ghost zones grew to 75 m.
6. **Medical kit count** (`mission.kits`). The plan assumed unlimited kits. Now RETURN_BASE happens when the
   payload is empty.
7. **Hybrid triage policy.** Only IMMEDIATE casualties interrupt exploration; the others are served after
   coverage, in score order. `immediate` and `after_coverage` are also available.
8. **Safety lock.** A station `goto` cannot pull the drone out of MAYDAY or out of a battery-forced RTH.
9. **Fragmentation, boot id and replay debounce** in the protocol, which the MAYDAY snapshot and reboots
   made necessary.
10. **Ramp-resistant spoof detection** with estimate rollback. The single-step innovation test in the plan
    missed a 5 m/s ramp completely.
11. **Tiered planner margins.** RTH must never be left without a path because of a hysteresis band.

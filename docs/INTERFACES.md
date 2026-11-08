# SKYFORGE interfaces (frozen: change only with the whole team)

## 1. Vision modules → PerceptionAgent (the contract for the detection team)

Each detector publishes **one JSON string per processed frame** on a `std_msgs/String` topic:

| Module | Topic | Classes |
|---|---|---|
| Jury module (debris, mines, weapons) | `/vision/jury` | `debris`, `mine`, `weapon` |
| Team module F5 (casualties, hostile units) | `/vision/team` | `injured`, `hostile_unit` |

```json
{"stamp": 1762560000.12, "source": "team", "image": [640, 480],
 "detections": [
   {"class": "injured", "confidence": 0.81, "bbox": [312.0, 200.5, 340.2, 251.0],
    "attrs": {"temp_k": 309.5, "posture": "lying", "moving": false}}
 ]}
```

Rules:
- `stamp` is the **image** timestamp in seconds, not the publish time. The agent uses it to look up the drone
  pose at that instant.
- `bbox` is `[x1, y1, x2, y2]` in pixels of the published image. These also work, but convert to the canonical
  form when you can: normalised YOLO boxes (all values ≤ 1), `{"x","y","w","h"}`, `{"xc","yc","w","h"}`.
- Keep the confidence threshold **low** (0.15–0.25). The agent applies per-class thresholds
  (`perception.conf_threshold`) and needs **3 hits in 5 frames** before anything triggers a decision.
  Recall matters most for mines and weapons.
- Raw class names are mapped by `skyforge_core/perception/adapter.py` (`CLASS_ALIASES`). For example
  `gun`, `pistol`, `rifle`, `arme` → `weapon`; `landmine`, `sea_mine` → `mine`; litter classes → `debris`;
  `soldier`, `tank`, `warship` → `hostile_unit`; `swimmer`, `floater`, `blessé` → `injured`.
  Add your dataset's labels there.
- The jury's own output format is unknown in advance, so the adapter also accepts French keys
  (`classe`, `confiance`, `boite`) and the lists `objets`, `objects`, `results` and `predictions`.
- Optional `attrs` for `injured` feed the START triage:

  | Attribute | Meaning |
  |---|---|
  | `temp_k` | mean body temperature (thermal box) |
  | `posture` | `lying` / `sitting` / `standing` / `waving` |
  | `moving` | bool |
  | `flotation` | bool |

  Missing attributes are estimated from the track: motion over 15 s, drift. A casualty with no thermal
  reading is treated as DELAYED (urgency 0.7). Over-prioritising is safer than ignoring someone.
- For `hostile_unit`, `attrs.detection_radius` (m) overrides the default exclusion radius of 25 m.

`skyforge_ros/vision_node.py` is a ready wrapper: give it a `.pt` or `.onnx` path
(`team_model:=/models/team.onnx`) and it subscribes to `/cam/rgb` and publishes this JSON. It drops frames
when busy, so it never queues stale images. Test any detector offline with:
```python
from skyforge_core.perception.adapter import parse_detections
parse_detections(open("sample.json").read())     # -> list[Detection]
```

## 2. ROS 2 topics

| Topic | Type | Rate | From → To |
|---|---|---|---|
| `/vision/jury`, `/vision/team` | `std_msgs/String` (JSON above) | 5–10 Hz | vision nodes → perception agent |
| `/cam/rgb`, `/cam/thermal`, `/cam/rgb/camera_info` | `sensor_msgs` | 10–15 Hz | Gazebo bridge → vision, perception |
| `/skyforge/odom` | `nav_msgs/Odometry` (ENU, `map`) | 50–100 Hz | reflex_node (vehicle interface) → all |
| `/skyforge/battery` | `sensor_msgs/BatteryState` (`charge`, `capacity` in **Wh**) | 50 Hz | reflex_node → brain |
| `/skyforge/armed`, `/skyforge/link` | `std_msgs/Bool` | | reflex / comm → brain |
| `/skyforge/nav` | `String` JSON `{gnss_ok, source, x, y, reason}` | 10 Hz | nav_integrity → brain |
| `/skyforge/gnss` | `sensor_msgs/NavSatFix` | 5–10 Hz | fault_injector relay (spoofing injection) → nav_integrity |
| `/perception/tracks` | `TrackArray` | 10 Hz | perception agent → orchestrator |
| `/perception/zones` | `ZoneArray` (confirmed danger) | 10 Hz | perception agent → orchestrator, reflex |
| `/perception/provisional` | `ZoneArray` (unconfirmed danger) | 10 Hz | perception agent → reflex (inward veto only) |
| `/plan/waypoints` | `nav_msgs/Path` | 10 Hz (heartbeat) | orchestrator → reflex |
| `/mission/state` | `MissionState` | 10 Hz | orchestrator → comm, dashboard, metrics |
| `/mission/events` | `MissionEvent` (feeds mission_logs) | on event | all → comm, dashboard |
| `/mission/snapshot` | `String` JSON (latched) | on MAYDAY / 0.2 Hz | orchestrator → comm (sync) |
| `/payload/release` | `String` JSON `{id, p}` | on event | orchestrator → payload servo / Gazebo detach |
| `/station/goal_update` | `String` JSON (already verified) | on event | comm → orchestrator |
| `/health/fault` | `String` (`""` = healthy) | 10 Hz | health_node → orchestrator |
| `/skyforge/fault_cmd`, `/skyforge/fault` | `String` JSON | on demand | operator → fault_injector → nodes |
| `/skyforge/obstacles` | `String` JSON `[{"p":[x,y],"r":4,"v":[vx,vy],"h":9}]` | 10 Hz | moving obstacle tracker / sim → reflex |
| `/fmu/in/trajectory_setpoint`, `/fmu/in/offboard_control_mode`, `/fmu/in/vehicle_command` | `px4_msgs` | 50 Hz | reflex_node → PX4 |
| `/metrics` | `Metric` | 1 Hz | metrics_node → dashboard |

PX4 rules:
- `reflex_node` is the **only** node that touches `px4_msgs`, and the only place NED↔ENU conversion
  happens (`skyforge_core/geometry.py`).
- `/fmu/out/*` uses BEST_EFFORT QoS. The status topic is `vehicle_status` on PX4 1.15 and
  `vehicle_status_v1` on 1.16+ (`status_topic` parameter). The `px4_msgs` branch must match the firmware
  (Docker build arg `PX4_MSGS_BRANCH`).
- Velocity offboard. Setpoints are streamed for at least 1 s before the node switches to OFFBOARD and arms.

## 3. Ground link (JSON, < 1 KB, asserted in code)

```json
{"v":1,"seq":1832,"ts":1762560000.123,"b":4711,"t":"det","id":"C3","c":"injured","u":"IMM",
 "p":[41.2,-12.8,0],"vd":[0.31,0.12],"cf":0.87,"fl":1,"h":"9f2a1c7e4b0d3a11"}
```

| Field | Meaning |
|---|---|
| `v` | protocol version |
| `seq` | per-sender sequence number |
| `ts` | sender time (1 ms resolution) |
| `b` | sender boot id: `seq` restarts on reboot, so replay protection is per session |
| `t` | message type (table below) |
| `h` | HMAC-SHA256 over the canonical JSON (sorted keys, without `h`), truncated to 8 bytes in hex. The shared key is `comm.key_hex`. |

| `t` | Content | Priority in the ring buffer |
|---|---|---|
| `tel` | `p` position, `ph` phase, `bat`, `rth`, `cov`, `nav`, `kit`, `tg` target | low (overwritten first) |
| `det` | confirmed detection: `id c u p vd cf fl` | high |
| `dec` | decision: `d` kind, `m` sentence, `ch` chosen, `cand` `[[id,S,ok],…]` | high |
| `goal` | `g`: goal_update body (station → drone) | critical |
| `ack` | `last` contiguous seq, `max` seen, `e`/`hd` RTT echo (station → drone) | critical |
| `sync` | `lost` counts per priority + `s` snapshot, sent when the gap was overwritten | critical |
| `may` | MAYDAY: `f` fault + `s` snapshot (casualties incl. served flag, zones, coverage RLE) | critical |
| `frag` | `ft` original type, `fid` group, `fi`/`fn` index/count, `d` JSON slice; used when a body is > 1 KB | as `ft` |

**Ring buffer:** 512 messages (< 512 KB). Unacked messages stay buffered. On reconnect the station acks
`last`/`max`, and the drone replays the gap in order, or sends `sync` if the gap was overwritten. Replays are
debounced to 1 s per message, so there are no replay storms.

**Accepted only if** the HMAC is valid AND (`seq` is newer within the same `b` OR a new `b` with a newer `ts`).
Otherwise the message is dropped and logged as `forged_rejected` / `replay_rejected`.

### goal_update bodies (`"g"`)

| Body | Effect |
|---|---|
| `{"type":"goto","p":[x,y]}` | go and look, then resume coverage |
| `{"type":"area","area":{"xmin":…,"xmax":…,"ymin":…,"ymax":…}}` | new exploration area |
| `{"type":"rtb"}` | return to base |
| `{"type":"resume"}` | cancel the station goto |
| `{"type":"zone_add","zone":{"id":"NOFLY1","center":[x,y],"radius":r}}` | add a no-fly zone |
| `{"type":"zone_clear","id":"NOFLY1"}` | remove a station zone |
| `{"type":"priority","id":"C4"}` | serve this casualty next if feasible (override is logged) |

Every update is applied in the decision tick where it arrives. The node also ticks immediately on receipt,
so reaction time is the measured `reaction_ms`. **Safety lock:** during MAYDAY or a battery-forced RTH, only
`rtb`, `zone_add` and `zone_clear` are accepted.

Ground station operator API (`station_server`, HTTP :8080):
```bash
curl localhost:8080/status
curl -X POST localhost:8080/goal -d '{"type":"goto","p":[120,40]}'
curl -X POST localhost:8080/forge -d '{"type":"rtb"}'     # demo: must be rejected
curl -X POST localhost:8080/replay                        # demo: must be rejected
```

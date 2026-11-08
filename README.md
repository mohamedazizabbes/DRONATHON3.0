# DRONATHON 3.0: Mission SKYFORGE

An autonomous coastal reconnaissance and rescue drone (IEEE AESS/RAS ENSIT, 7–8 November 2026).
A **LangGraph orchestrator with deterministic nodes** (no language model in the real-time loop) and two AI
agents, **PerceptionAgent** and **PlannerAgent**, drive the drone with no human input:

- explore a coastal zone and map mines, weapons and debris;
- find and triage casualties (thermal START);
- avoid hostile units and obstacles;
- drop medical kits on the right person, including someone drifting at sea;
- keep the mission alive through jamming, GNSS spoofing, forged commands and motor failure;
- come home before the battery runs out.

## What the brain does (highlights)

- **Triage with FLIR:** START categories from heat, breathing micro-motion, posture and motion. Injured are
  served first. A **likely deceased** casualty (skin at ambient, no breathing for 8 s at close range, on land)
  never takes a kit and is reported for recovery. Cold skin **in water** stays IMMEDIATE (hypothermia).
- **Drift prediction:** a Kalman filter per floating object, plus an **online sea-current estimator**
  (it follows a tide change) and long-baseline learning when an object is re-found. The **kit's IR strobe**
  marks served casualties, so nobody gets two kits.
- **Return home on time:** a wind-triangle energy model at commanded ground speed, **calibrated online**
  from the battery drain, with a dynamic RTH threshold recomputed every tick.
- **Evasive manoeuvres:** exit a hostile unit's line of sight radially, fly lower inside it, replan.
  Hostile units are always zones to avoid, never targets.
- **MAYDAY:** a motor or sensor fault sends a fragmented snapshot. **Drone 2** resumes from it in the
  simulator or in ROS (`resume_snapshot:=`).
- **Mines in three layers:**
  - **camera** for surface mines;
  - **magnetometer** for shallow and buried mines (dipole inversion, plus a close-range inspection on land);
  - **dipping sonar** for deep mines (survey stations plus confirmation dips).
  
  Always from a standoff, never above a suspected mine.
- **Link resilience:** signed JSON < 1 KB, a 512-message ring buffer, replay and resync, compatibility mode
  for the jury's station, weather from the station or PX4.

## Repository

| Path | What |
|---|---|
| `skyforge_ws/src/skyforge_core` | **All logic, pure Python** (no rclpy): `brain/` LangGraph orchestrator + DroneState, `perception/` PerceptionAgent, `planning/` coverage/A*/triage/intercept, `energy.py` RTH, `reflex/`, `comm/` protocol + ring buffer + station + Pi bridge, `monitor/`, `sim/` 2D mission simulator, `tests/` |
| `skyforge_ws/src/skyforge_interfaces` | ROS 2 messages (frozen) |
| `skyforge_ws/src/skyforge_ros` | ROS 2 Jazzy nodes (thin wrappers), fake-rclpy closed-loop test |
| `skyforge_ws/src/skyforge_bringup` | `mission.launch.py`, `config/skyforge.yaml` (**the only parameter file**), `faults_demo.yaml` |
| `docker/` | Jazzy image + compose (brain, station, foxglove, sim2d) |
| `deploy/raspberry_pi/` | comm module on the Pi (F12 bonus): install script, systemd service, benchmark |
| `docs/ARCHITECTURE.md` | how it decides, formulas, resilience, what we added to the plan |
| `docs/INTERFACES.md` | **contract for the detection team** (JSON), ROS topics, link protocol, goal_update |

## Quick start

**1. No ROS needed: run a full mission in the 2D simulator (any OS, Python ≥ 3.10).**
```bash
pip install numpy pyyaml langgraph matplotlib pytest
cd skyforge_ws/src/skyforge_core
python -m skyforge_core.sim.run --out runs/demo            # decision feed + KPIs + runs/demo/trajectory.png
python -m skyforge_core.sim.run --mayday-at 150            # motor failure -> MAYDAY -> drone 2 takes over
python -m skyforge_core.sim.run --seeds 10 --quiet         # 10 seeded runs -> aggregate.json for the slides
python -m pytest tests -q                                  # 25 unit tests
python -m pytest ../skyforge_ros/test -q                   # real ROS nodes on a fake rclpy bus
```

**2. ROS 2 Jazzy + Gazebo Harmonic + PX4 SITL.**
```bash
cd skyforge_ws && git clone -b release/1.15 https://github.com/PX4/px4_msgs.git src/px4_msgs
pip install --break-system-packages langgraph
colcon build && source install/setup.bash
ros2 launch skyforge_bringup mission.launch.py                     # PX4 offboard backend
ros2 launch skyforge_bringup mission.launch.py backend:=twist      # velocity-controlled Gazebo model fallback
ros2 launch skyforge_bringup mission.launch.py team_model:=/models/team.onnx jury_model:=/models/jury.onnx
python3 -m skyforge_core.comm.station_server --config src/skyforge_bringup/config/skyforge.yaml   # ground station
```
Or with Docker: `docker compose -f docker/compose.yaml up --build brain station foxglove`.

**3. Raspberry Pi comm module (F12).** See [`deploy/raspberry_pi/README.md`](deploy/raspberry_pi/README.md).

## Demo controls
```bash
ros2 topic pub --once /skyforge/fault_cmd std_msgs/String '{data: "{\"type\":\"jam\",\"on\":true,\"dur_s\":20}"}'
ros2 topic pub --once /skyforge/fault_cmd std_msgs/String '{data: "{\"type\":\"spoof\",\"dx\":40,\"dy\":0,\"ramp_s\":8,\"dur_s\":25}"}'
ros2 topic pub --once /skyforge/fault_cmd std_msgs/String '{data: "{\"type\":\"motor\"}"}'
curl -X POST localhost:8080/goal  -d '{"type":"goto","p":[120,40]}'     # accepted, reroute in the same tick
curl -X POST localhost:8080/forge -d '{"type":"rtb"}'                   # rejected (bad HMAC)
curl -X POST localhost:8080/replay                                      # rejected (old seq)
```
`mission_log_*.jsonl` (in `runs/`) holds every decision with its inputs, all candidates and why the losers lost.
`/mission/events` carries the same entries live for the dashboard.

**Dashboard (Foxglove Studio → `ws://localhost:8765`):**

| Panel | Source |
|---|---|
| 3D | `/viz/markers`: zones, line-of-sight rings, casualties coloured by START, drift ellipses, path, trail, phase |
| Plot (RTH gauge) | `/mission/state.battery_wh` vs `/mission/state.rth_threshold_wh` |
| Log (decision feed) | `/rosout` |
| Plot | `/metrics` (p95 per layer) |
| Raw Messages | `/comm/stats` (link, buffer, RTT, resync) and `/skyforge/nav` |

## Results (headless 2D simulator, default coastal scenario)

10 seeded runs. The scenario includes:

- **Casualties:** 5 living (2 IMMEDIATE, one drifting at sea) and 1 likely-deceased body.
- **Mines:** 5 (beach, floating, shallow 3 m, deep 12 m, buried in the beach), plus a weapon and a hostile unit.
- **Environment:** 3 obstacles, a moving boat, a radio shadow zone, waves (±0.6 m) and a tide change at t = 200 s.
- **Faults:** a 20 s jam, a 40 m GNSS spoof ramp, a station zone_add, a forged and a replayed command, a goto and a gust.
- **Payload and energy:** 6 kits; the real airframe draws 15% more power than the paper budget.

| KPI | Result |
|---|---|
| Collisions / danger-zone entries / off-target drops | **0 / 0 / 0** (all 10 runs) |
| Living casualties served | **4.8 / 5** on average (7 runs perfect); the likely-deceased body is assessed and never takes a kit |
| Landed safely / battery at landing | 10/10, lowest **20.4%** (reserve 15%) |
| RTH energy, predicted vs actual | within **−2.8% … +7.8%** (path-aware, online-calibrated power model) |
| Drop error vs truth | max **0.96 m**, mean 0.30 m (spec 1.5 m) |
| Mines | localised within 8 m: 4.3 / 5; inside a registered zone: 4.6 / 5. Deep and shallow mines found by sonar, surface mines by the camera, the buried one by the magnetometer (least reliable) |
| Decision latency p95 | ≤ 6.4 ms (budget 100 ms); reflex p95 < 0.3 ms (budget 50 ms) |
| GNSS spoof detection | 1.9 s (40 m ramp at 5 m/s) |
| Forged / replayed commands | 100% rejected |
| Link outage | buffered messages replayed in order on reconnect, 0 lost; largest message 978 B (< 1 KB) |
| MAYDAY at t = 150 s | drone 1 lands safely; drone 2 resumes from the station's snapshot, skips served casualties; 0 safety events |

These are simulator numbers. Re-measure in Gazebo/PX4 with the real detectors before the presentation.

## Team interfaces
- **Detection team:** publish the JSON in [`docs/INTERFACES.md`](docs/INTERFACES.md#1-vision-modules--perceptionagent-the-contract-for-the-detection-team)
  on `/vision/jury` and `/vision/team`, or just pass the model to `vision_node`. Add your raw class labels to
  `CLASS_ALIASES`.
- **Conception:**
  - The `energy.*` values (`p_hover_w`, `k_speed`, `capacity_wh`) must come from the power budget. Calibrate
    them with a 60 s hover and a 60 s cruise.
  - The gimbal needs a **20° forward tilt**, so the camera sees mines before entering their 15 m zone.
  - Set `mission.kits` to the number of kits the payload pod carries.
  - The **kit pod needs an IR strobe** (FLIR-visible), so served casualties are recognised after drifting.
  - Add a **magnetometer on a boom**, away from the motors (fluxgate, about 0.15 nT noise), publishing
    `/sensors/mag`.
  - Add a **winched dipping sonar** (about 60 m range), commanded on `/payload/sonar_cmd` and publishing
    `/sensors/sonar`.
- **Sim & Control:**
  - Bridge `/cam/rgb`, `/cam/thermal` and `camera_info`, run the XRCE agent, and publish moving obstacles on
    `/skyforge/obstacles`.
  - Set `use_sim_time:=true` only if `/clock` is bridged.
  - Put the jury's scenario (area, obstacles, water and radio-shadow polygons) in `skyforge.yaml`.

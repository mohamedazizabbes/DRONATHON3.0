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

## Results (headless 2D simulator, default coastal scenario)

10 seeded runs. The scenario has 5 casualties (2 IMMEDIATE, one of them drifting at sea), 2 mines (one floating),
a weapon, a hostile unit, 3 obstacles, a moving boat and a radio shadow zone. The faults are a 20 s jam, a 40 m
GNSS spoof ramp, a station zone_add, a forged and a replayed command, a goto and a gust. The drone carries 4 kits.

| KPI | Result |
|---|---|
| Collisions / danger-zone entries / off-target drops | **0 / 0 / 0** (all 10 runs) |
| Casualties served | 4/5 in every run (the maximum with 4 kits; the EXPECTANT one waits) |
| Landed safely | 10/10 |
| Decision latency p95 | 4–11 ms on the dev laptop depending on load (budget 100 ms); reflex p95 < 0.3 ms (budget 50 ms) |
| Drop error vs truth | 0.1 – 1.0 m (spec 1.5 m) |
| GNSS spoof detection | 1.9 s (40 m ramp at 5 m/s) |
| Forged / replayed commands | 100 % rejected |
| Resync after a 20 s outage | 0.02 s, 0 messages lost; max message 978 B (< 1 KB) |
| MAYDAY at t = 150 s | drone 1 lands safely; drone 2 resumes from the station's snapshot, skips the served casualty, total 4/5, 0 safety events |

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
- **Sim & Control:**
  - Bridge `/cam/rgb`, `/cam/thermal` and `camera_info`, run the XRCE agent, and publish moving obstacles on
    `/skyforge/obstacles`.
  - Set `use_sim_time:=true` only if `/clock` is bridged.
  - Put the jury's scenario (area, obstacles, water and radio-shadow polygons) in `skyforge.yaml`.

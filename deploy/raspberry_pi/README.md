# Raspberry Pi deployment: comm module on a board (F12 bonus)

Only the **Communication module** runs on the Pi, as the design says. It is the drone's companion radio
computer: it signs, numbers and buffers every message, keeps working through outages, and resynchronises
with the ground station. The brain (ROS 2 graph) runs on the laptop, and the Pi sits between it and the
station.

```
 laptop (ROS 2 Jazzy)                      Raspberry Pi                         ground station
 comm_node mode:=pi  --UDP ops :14700-->  skyforge comm bridge  --signed JSON-->  station_server :14601
 /station/goal_update <--goals/events---   HMAC, seq, ring 512,  <--acks, goals--  (HTTP :8080 operator)
                       :14701              resync, link sim
```

## 1. Prepare the Pi (5 min)
- Raspberry Pi 3B+/4/5 with Raspberry Pi OS Lite 64-bit (or Ubuntu 24.04). Python >= 3.10 is already there.
- Put it on the same network as the laptop (Ethernet is the most reliable on demo day; Wi-Fi works).
- Note the IPs: `hostname -I` on the Pi, `ip a` on the laptop.

## 2. Install
```bash
scp -r DRONATHON3.0 pi@<PI_IP>:~/
ssh pi@<PI_IP>
cd ~/DRONATHON3.0
sudo ./deploy/raspberry_pi/install.sh --station <LAPTOP_IP> --brain <LAPTOP_IP>
journalctl -u skyforge-comm-bridge -f        # should print: [skyforge-bridge] station=... ring=512
```
No pip and no internet are needed: the bridge uses only the Python standard library.
`python3-psutil` and `python3-yaml` are installed when apt is available, for exact CPU/RAM numbers
and for reading the config file.

## 3. Run the mission through the Pi
On the laptop:
```bash
ros2 launch skyforge_bringup mission.launch.py comm_mode:=pi pi_host:=<PI_IP>
python3 -m skyforge_core.comm.station_server --config skyforge.yaml --drone <PI_IP>:14600
```
Then check the live metrics: `ros2 topic echo /comm/stats`, which shows RSS, CPU, latency, RTT,
buffer occupancy and resync time.

## 4. Measurements for the jury (F12)
```bash
cd /opt/skyforge
python3 -m skyforge_core.comm.pi_bench --duration 60 --outage-at 10 --outage 30 --out pi_bench.json
```
It prints a markdown table with:

| Metric | Meaning |
|---|---|
| `rss_mb_max` | resident memory of the comm process |
| `cpu_pct_mean` | CPU load at 10 Hz telemetry + detections + decisions |
| `proc_p50/p95/p99_ms` | per-message processing time (encode + HMAC + buffer + send) |
| `rtt_p50/p95_ms` | round trip, NTP-style echo in the station acks |
| `resync_s` | time from link restored to station fully in sync after a 30 s forced outage |
| `buffered_during_outage`, `replayed`, `lost` | messages kept during the outage, replayed, and lost (target 0) |
| `max_msg_bytes` | largest message, which must be < 1024 |

Reference run on the dev laptop (x86, 25 s run, 10 s outage):
22.7 MB RSS, 0.53 ms p95 per message, 3 ms RTT p50, resync 0.015 s, 0 lost, max 153 B.
**Replace this with your Pi numbers before the presentation.**

## Troubleshooting
- No `tel` at the station: check the firewall (`sudo ufw allow 14600:14701/udp`) and the IPs in
  `/opt/skyforge/bridge.env`, then `sudo systemctl restart skyforge-comm-bridge`.
- HMAC errors (`forged_rejected` on legitimate traffic): the Pi, laptop and station must share the same
  `comm.key_hex` in `skyforge.yaml`.
- To change parameters, edit `/opt/skyforge/skyforge.yaml` and restart the service. No rebuild is needed.

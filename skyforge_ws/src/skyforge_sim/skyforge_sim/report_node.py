import base64
import glob
import html
import json
import math
import os
import time

import rclpy
from nav_msgs.msg import Odometry
from rclpy.node import Node
from rclpy.qos import HistoryPolicy, QoSProfile, ReliabilityPolicy
from skyforge_interfaces.msg import MissionState, TrackArray, ZoneArray
from std_msgs.msg import String

QOS = QoSProfile(reliability=ReliabilityPolicy.RELIABLE, history=HistoryPolicy.KEEP_LAST, depth=10)
CAT = {"IMM": "Immediate", "DEL": "Delayed", "MIN": "Minor", "EXP": "Expectant", "UNK": "Unknown"}
CAT_COL = {"IMM": "#d62828", "DEL": "#f77f00", "MIN": "#2a9d8f", "EXP": "#6c757d", "UNK": "#adb5bd"}

CSS = """body{font-family:Inter,Segoe UI,Arial,sans-serif;background:#0f141a;color:#e6e9ee;margin:0;padding:28px 36px}
h1{margin:0 0 4px;font-size:26px;letter-spacing:1px}h2{margin:28px 0 10px;font-size:17px;color:#7fd1ff;border-bottom:1px solid #26313d;padding-bottom:6px}
.sub{color:#8b98a6;font-size:13px}.grid{display:grid;grid-template-columns:repeat(4,1fr);gap:12px;margin-top:16px}
.card{background:#17202a;border:1px solid #26313d;border-radius:8px;padding:12px 14px}.card b{display:block;font-size:22px;margin-top:4px}
.card span{color:#8b98a6;font-size:12px;text-transform:uppercase;letter-spacing:.6px}table{border-collapse:collapse;width:100%;font-size:13px}
th,td{padding:7px 9px;border-bottom:1px solid #26313d;text-align:left}th{color:#8b98a6;font-weight:600;text-transform:uppercase;font-size:11px}
.pill{display:inline-block;padding:2px 8px;border-radius:10px;font-size:11px;font-weight:600;color:#fff}.ok{color:#5bd68a}.no{color:#ff7b72}
img{max-width:100%;border:1px solid #26313d;border-radius:8px}.tl{font-family:ui-monospace,Consolas,monospace;font-size:12px;color:#c8d0da}
.tl div{padding:2px 0;border-bottom:1px dotted #1f2a35}"""


class Report(Node):
    def __init__(self):
        super().__init__("mission_report")
        self.declare_parameter("out_dir", "runs")
        self.out_dir = self.get_parameter("out_dir").value
        self.t0 = time.time()
        self.state = None
        self.phase = ""
        self.tracks = {}
        self.zones = []
        self.releases = []
        self.truth = []
        self.dist = 0.0
        self.last = None
        self.max_alt = 0.0
        self.battery0 = None
        self.done = False
        self.create_subscription(MissionState, "/mission/state", self.on_state, 10)
        self.create_subscription(TrackArray, "/perception/tracks", self.on_tracks, 10)
        self.create_subscription(ZoneArray, "/perception/zones", lambda m: setattr(self, "zones", list(m.zones)), 10)
        self.create_subscription(String, "/payload/release", self.on_release, QOS)
        self.create_subscription(String, "/sim/truth", lambda m: setattr(self, "truth", json.loads(m.data)), 10)
        self.create_subscription(Odometry, "/skyforge/odom", self.on_odom, 10)
        self.create_subscription(String, "/report/generate", lambda m: self.write("on demand"), 10)
        self.pending = None
        self.create_timer(1.0, self.tick)

    def on_state(self, m):
        if self.battery0 is None and m.battery_wh > 0:
            self.battery0 = m.battery_wh
        if m.phase != self.phase and m.phase == "LANDED" and self.phase not in ("", "INIT"):
            self.pending = time.time() + 4.0
        self.phase = m.phase
        self.state = m

    def on_tracks(self, m):
        for t in m.tracks:
            if t.confirmed:
                self.tracks[t.id] = t

    def on_release(self, m):
        try:
            a = json.loads(m.data)
            a["t"] = round(time.time() - self.t0, 1)
            self.releases.append(a)
        except ValueError:
            pass

    def on_odom(self, m):
        p = m.pose.pose.position
        if self.last is not None:
            self.dist += math.hypot(p.x - self.last[0], p.y - self.last[1])
        self.last = (p.x, p.y)
        self.max_alt = max(self.max_alt, p.z)

    def tick(self):
        if self.pending and time.time() >= self.pending and not self.done:
            self.write("mission complete")
            self.done = True

    def write(self, why):
        os.makedirs(self.out_dir, exist_ok=True)
        st = self.state
        dur = time.time() - self.t0
        logs = sorted(glob.glob(os.path.join(self.out_dir, "mission_log_*.jsonl")), key=os.path.getmtime)
        events = []
        if logs:
            for line in open(logs[-1]):
                try:
                    e = json.loads(line)
                except ValueError:
                    continue
                if e.get("type") in ("phase", "det", "dec") and "replan" not in e.get("msg", ""):
                    events.append(e)
        maps = sorted(glob.glob(os.path.join(self.out_dir, "mission_map_*.png")), key=os.path.getmtime)
        map_img = ""
        if maps and os.path.getmtime(maps[-1]) > self.t0:
            map_img = base64.b64encode(open(maps[-1], "rb").read()).decode()
        served = {r.get("id"): r for r in self.releases}
        casualties = [t for t in self.tracks.values() if t.cls == "injured"]
        threats = [t for t in self.tracks.values() if t.cls != "injured"]
        truth_tot, truth_found = {}, {}
        for o in self.truth:
            k = o.get("cls", "")
            truth_tot[k] = truth_tot.get(k, 0) + 1
            if any(t.cls == k and math.hypot(t.x - o["p"][0], t.y - o["p"][1]) < 10.0 for t in self.tracks.values()):
                truth_found[k] = truth_found.get(k, 0) + 1
        used = (self.battery0 - st.battery_wh) if (st and self.battery0) else 0.0
        cards = [
            ("Status", html.escape(st.phase_fr if st else "-")), ("Duration", f"{int(dur // 60)} min {int(dur % 60)} s"),
            ("Coverage", f"{100 * st.coverage:.0f} %" if st else "-"), ("Distance flown", f"{self.dist / 1000:.2f} km"),
            ("Casualties found", str(len(casualties))), ("Kits delivered", str(len(self.releases))),
            ("Threats located", str(len(threats))), ("Energy used", f"{used:.1f} Wh"),
        ]
        rows = []
        for t in sorted(casualties, key=lambda t: -t.urgency):
            r = served.get(t.id)
            rows.append(f"<tr><td>{t.id}</td><td><span class='pill' style='background:{CAT_COL.get(t.category, '#adb5bd')}'>{CAT.get(t.category, t.category)}</span></td>"
                        f"<td>{t.urgency:.2f}</td><td>({t.x:.1f}, {t.y:.1f})</td><td>{'water' if t.floating else 'land'}</td><td>{t.conf:.2f}</td>"
                        f"<td>{'<span class=ok>delivered at t=' + str(r['t']) + ' s</span>' if r else '<span class=no>not served</span>'}</td></tr>")
        trows = []
        for t in sorted(threats, key=lambda t: t.cls):
            z = next((z for z in self.zones if math.hypot(z.x - t.x, z.y - t.y) < 3.0), None)
            trows.append(f"<tr><td>{t.id}</td><td>{html.escape(t.cls.replace('_', ' '))}</td><td>({t.x:.1f}, {t.y:.1f})</td><td>{'yes' if t.floating else 'no'}</td>"
                         f"<td>{t.conf:.2f}</td><td>{f'{z.radius:.0f} m' if z else '-'}</td></tr>")
        names = (("injured", "Casualties"), ("mine", "Mines"), ("hostile_unit", "Hostile units"), ("weapon", "Weapons"), ("debris", "Debris"))
        frows = "".join(f"<tr><td>{n}</td><td>{truth_found.get(k, 0)} / {truth_tot[k]}</td></tr>" for k, n in names if k in truth_tot)
        tl = "".join(f"<div>t={e.get('ts', 0):6.1f}s &nbsp; [{html.escape(e.get('phase', ''))}] &nbsp; {html.escape(e.get('msg', ''))}</div>" for e in events[-80:])
        stamp = time.strftime("%Y-%m-%d %H:%M:%S")
        doc = f"""<!doctype html><html><head><meta charset="utf-8"><title>SKYFORGE mission report</title><style>{CSS}</style></head><body>
<h1>SKYFORGE &middot; Mission report</h1><div class="sub">Generated {stamp} ({html.escape(why)}) &middot; autonomous coastal reconnaissance and support</div>
<div class="grid">{''.join(f'<div class="card"><span>{k}</span><b>{v}</b></div>' for k, v in cards)}</div>
<h2>Casualties (triage order)</h2><table><tr><th>ID</th><th>START category</th><th>Urgency</th><th>Position (m)</th><th>Where</th><th>Conf.</th><th>Medical kit</th></tr>{''.join(rows) or '<tr><td colspan=7>none confirmed</td></tr>'}</table>
<h2>Threats and exclusion zones</h2><table><tr><th>ID</th><th>Class</th><th>Position (m)</th><th>Floating</th><th>Conf.</th><th>Exclusion radius</th></tr>{''.join(trows) or '<tr><td colspan=6>none confirmed</td></tr>'}</table>
<h2>Detection versus ground truth (simulation)</h2><table><tr><th>Category</th><th>Found / present</th></tr>{frows or '<tr><td colspan=2>no truth data</td></tr>'}</table>
{f'<h2>Mission map</h2><img src="data:image/png;base64,{map_img}">' if map_img else ''}
<h2>Decision timeline</h2><div class="tl">{tl or 'no log entries'}</div>
</body></html>"""
        path = os.path.join(self.out_dir, f"mission_report_{time.strftime('%Y%m%d_%H%M%S')}.html")
        open(path, "w").write(doc)
        self.get_logger().info(f"mission report written: {path}")


def main():
    rclpy.init()
    n = Report()
    try:
        rclpy.spin(n)
    except KeyboardInterrupt:
        pass
    finally:
        n.destroy_node()
        rclpy.try_shutdown()


if __name__ == "__main__":
    main()

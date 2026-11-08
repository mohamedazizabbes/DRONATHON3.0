import json
import math
import time

import cv2
import numpy as np
import rclpy
from nav_msgs.msg import Odometry, Path
from rclpy.node import Node
from rclpy.qos import HistoryPolicy, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import Image
from skyforge_core.params import DEFAULTS
from skyforge_interfaces.msg import MissionState, TrackArray, ZoneArray
from std_msgs.msg import String

QOS = QoSProfile(reliability=ReliabilityPolicy.BEST_EFFORT, history=HistoryPolicy.KEEP_LAST, depth=2)
GREEN = (80, 255, 80)
CYAN = (255, 220, 0)
RED = (40, 40, 255)
ORANGE = (0, 160, 255)
WHITE = (240, 240, 240)
GREY = (150, 150, 150)
YELLOW = (0, 230, 255)
PHASE_COLOR = {"EXPLORE": CYAN, "DETECT": YELLOW, "TRIAGE": ORANGE, "NAVIGATE": ORANGE, "DELIVER": GREEN,
               "RETURN_BASE": YELLOW, "LANDED": GREY, "MAYDAY": RED, "INIT": GREY}


def to_bgr(m):
    a = np.frombuffer(m.data, np.uint8)
    ch = len(a) // (m.width * m.height)
    a = a.reshape(m.height, m.width, ch)[:, :, :3]
    return np.ascontiguousarray(a[:, :, ::-1]) if m.encoding.startswith("rgb") else np.ascontiguousarray(a)


class Hud(Node):
    def __init__(self):
        super().__init__("hud")
        self.declare_parameter("window", True)
        self.declare_parameter("width", 1280)
        self.declare_parameter("height", 720)
        self.W = int(self.get_parameter("width").value)
        self.H = int(self.get_parameter("height").value)
        self.window = bool(self.get_parameter("window").value)
        self.frame = None
        self.dets = {"jury": ([], (640, 480), 0.0), "team": ([], (640, 480), 0.0)}
        self.state = None
        self.speed = 0.0
        self.alt = 0.0
        self.yaw = 0.0
        self.t0 = time.time()
        self.create_subscription(Image, "/cam/rgb", self.on_img, QOS)
        self.create_subscription(String, "/vision/jury", lambda m: self.on_det(m, "jury"), QOS)
        self.create_subscription(String, "/vision/team", lambda m: self.on_det(m, "team"), QOS)
        self.create_subscription(MissionState, "/mission/state", lambda m: setattr(self, "state", m), 10)
        self.create_subscription(Odometry, "/skyforge/odom", self.on_odom, 10)
        self.create_subscription(TrackArray, "/perception/tracks", lambda m: setattr(self, "tracks", m.tracks), 10)
        self.create_subscription(ZoneArray, "/perception/zones", lambda m: setattr(self, "zones", m.zones), 10)
        self.create_subscription(Path, "/plan/waypoints", lambda m: setattr(self, "path", [(p.pose.position.x, p.pose.position.y) for p in m.poses]), 10)
        self.tracks, self.zones, self.path, self.trail = [], [], [], []
        self.pos = (0.0, 0.0)
        area = DEFAULTS["mission"]["area"]
        self.shore = max(p[1] for poly in DEFAULTS["water"]["polygons"] for p in poly)
        self.ext = (area["xmin"] - 15, area["xmax"] + 15, area["ymin"] - 15, area["ymax"] + 15)
        self.area = area
        self.pub = self.create_publisher(Image, "/hud/image", 2)
        self.create_timer(0.1, self.draw)

    def on_img(self, m):
        self.frame = to_bgr(m)

    def on_det(self, m, src):
        try:
            d = json.loads(m.data)
            self.dets[src] = (d.get("detections", []), tuple(d.get("image", [640, 480])), time.time())
        except ValueError:
            pass

    def on_odom(self, m):
        v = m.twist.twist.linear
        self.speed = math.hypot(v.x, v.y)
        self.alt = m.pose.pose.position.z
        self.pos = (m.pose.pose.position.x, m.pose.pose.position.y)
        if not self.trail or math.hypot(self.pos[0] - self.trail[-1][0], self.pos[1] - self.trail[-1][1]) > 2.0:
            self.trail.append(self.pos)
            self.trail = self.trail[-600:]
        q = m.pose.pose.orientation
        self.yaw = math.degrees(math.atan2(2 * (q.w * q.z + q.x * q.y), 1 - 2 * (q.y * q.y + q.z * q.z)))

    def text(self, img, s, xy, col=WHITE, scale=0.6, th=1):
        cv2.putText(img, s, xy, cv2.FONT_HERSHEY_DUPLEX, scale, (0, 0, 0), th + 2, cv2.LINE_AA)
        cv2.putText(img, s, xy, cv2.FONT_HERSHEY_DUPLEX, scale, col, th, cv2.LINE_AA)

    def map_view(self, w, h):
        img = np.zeros((h, w, 3), np.uint8)
        x0, x1, y0, y1 = self.ext
        sc = min(w / (x1 - x0), h / (y1 - y0))
        ox, oy = (w - sc * (x1 - x0)) / 2, (h - sc * (y1 - y0)) / 2

        def P(x, y):
            return int(ox + (x - x0) * sc), int(h - (oy + (y - y0) * sc))

        sy = P(0, self.shore)[1]
        img[:sy] = (70, 120, 150)
        img[sy:] = (95, 55, 20)
        for gx in range(int(x0) // 25 * 25, int(x1) + 1, 25):
            cv2.line(img, P(gx, y0), P(gx, y1), (60, 60, 60), 1)
        for gy in range(int(y0) // 25 * 25, int(y1) + 1, 25):
            cv2.line(img, P(x0, gy), P(x1, gy), (60, 60, 60), 1)
        a = self.area
        cv2.rectangle(img, P(a["xmin"], a["ymax"]), P(a["xmax"], a["ymin"]), (0, 200, 255), 1)
        over = img.copy()
        for z in self.zones:
            c = P(z.x, z.y)
            cv2.circle(over, c, max(2, int(z.radius * sc)), RED if z.source != "station" else ORANGE, -1)
            if z.soft_radius > 0:
                cv2.circle(img, c, int(z.soft_radius * sc), RED, 1)
        img = cv2.addWeighted(over, 0.35, img, 0.65, 0)
        if len(self.trail) > 1:
            cv2.polylines(img, [np.array([P(*p) for p in self.trail], np.int32)], False, (120, 120, 120), 1)
        if len(self.path) > 1:
            cv2.polylines(img, [np.array([P(self.pos[0], self.pos[1])] + [P(*p) for p in self.path], np.int32)], False, GREEN, 1)
        catcol = {"IMM": RED, "DEL": ORANGE, "MIN": GREEN, "EXP": GREY, "UNK": WHITE}
        for t in self.tracks:
            c = P(t.x, t.y)
            fill = -1 if t.confirmed else 2
            if t.cls == "injured":
                cv2.circle(img, c, 6, catcol.get(t.category, WHITE), fill)
                self.text(img, f"{t.id} {t.category}", (c[0] + 8, c[1] - 6), catcol.get(t.category, WHITE), 0.42)
            elif t.cls == "mine":
                pts = np.array([(c[0], c[1] - 7), (c[0] + 7, c[1]), (c[0], c[1] + 7), (c[0] - 7, c[1])], np.int32)
                cv2.polylines(img, [pts], True, YELLOW, 2) if not t.confirmed else cv2.fillPoly(img, [pts], YELLOW)
            elif t.cls == "hostile_unit":
                cv2.rectangle(img, (c[0] - 6, c[1] - 6), (c[0] + 6, c[1] + 6), RED, fill)
            elif t.cls == "weapon":
                cv2.drawMarker(img, c, (220, 80, 220), cv2.MARKER_TILTED_CROSS, 12, 2)
            else:
                cv2.circle(img, c, 3, (60, 120, 170), fill)
        d = P(*self.pos)
        a_ = math.radians(self.yaw)
        tip = (int(d[0] + 12 * math.cos(a_)), int(d[1] - 12 * math.sin(a_)))
        l = (int(d[0] + 7 * math.cos(a_ + 2.5)), int(d[1] - 7 * math.sin(a_ + 2.5)))
        r = (int(d[0] + 7 * math.cos(a_ - 2.5)), int(d[1] - 7 * math.sin(a_ - 2.5)))
        cv2.fillPoly(img, [np.array([tip, l, r], np.int32)], CYAN)
        hp = P(*DEFAULTS["mission"]["home"])
        cv2.drawMarker(img, hp, GREEN, cv2.MARKER_SQUARE, 10, 2)
        self.text(img, "H", (hp[0] - 4, hp[1] + 20), GREEN, 0.45)
        return img

    def draw(self):
        W, H = self.W, self.H
        canvas = np.full((H, W, 3), 18, np.uint8)
        vw, vh = 640, 480
        if self.frame is not None:
            view = cv2.resize(self.frame, (vw, vh))
        else:
            view = np.zeros((vh, vw, 3), np.uint8)
            self.text(view, "NO VIDEO", (vw // 2 - 80, vh // 2), RED, 1.0, 2)
        locked = None
        for src, col in (("jury", RED), ("team", CYAN)):
            dets, (iw, ih), ts = self.dets[src]
            if time.time() - ts > 1.0:
                continue
            sx, sy = vw / iw, vh / ih
            for d in dets:
                x1, y1, x2, y2 = d["bbox"]
                p1, p2 = (int(x1 * sx), int(y1 * sy)), (int(x2 * sx), int(y2 * sy))
                cv2.rectangle(view, p1, p2, col, 2)
                post = d.get("attrs", {}).get("posture")
                label = f'{d["class"]}{"/" + post if post else ""} {d["confidence"]:.2f}'
                self.text(view, label, (p1[0], max(16, p1[1] - 6)), col, 0.5)
                if d["class"] == "injured" and (locked is None or d["confidence"] > locked[2]):
                    locked = ((p1[0] + p2[0]) // 2, (p1[1] + p2[1]) // 2, d["confidence"])
        cx, cy = vw // 2, vh // 2
        for (a1, b1), (a2, b2) in (((-25, 0), (-8, 0)), ((8, 0), (25, 0)), ((0, -25), (0, -8)), ((0, 8), (0, 25))):
            cv2.line(view, (cx + a1, cy + b1), (cx + a2, cy + b2), GREEN, 1)
        if locked:
            x, y, _ = locked
            cv2.rectangle(view, (x - 28, y - 28), (x + 28, y + 28), YELLOW, 2)
        self.text(view, "CAM  RGB  640x480  tilt 20", (10, vh - 12), GREY, 0.45)
        canvas[56:56 + vh, 10:10 + vw] = view
        cv2.rectangle(canvas, (10, 56), (10 + vw - 1, 56 + vh - 1), (70, 70, 70), 1)
        mw, mh = W - vw - 30, H - 56 - 46
        canvas[56:56 + mh, vw + 20:vw + 20 + mw] = self.map_view(mw, mh)
        cv2.rectangle(canvas, (vw + 20, 56), (vw + 20 + mw - 1, 56 + mh - 1), (70, 70, 70), 1)
        self.text(canvas, "TACTICAL MAP", (vw + 30, 80), GREEN, 0.6, 1)
        st = self.state
        phase = st.phase if st else "INIT"
        mt = int(time.time() - self.t0)
        self.text(canvas, f"MISSION {mt // 60:02d}:{mt % 60:02d}", (14, 36), YELLOW, 0.75, 2)
        self.text(canvas, f"{phase.replace('_', ' ')}", (250, 36), PHASE_COLOR.get(phase, WHITE), 0.8, 2)
        tgt = st.target_id if st and st.target_id else ""
        self.text(canvas, f"TARGET {tgt if tgt else ('LOCKED' if locked else 'SEARCHING')}", (520, 36), GREEN if (tgt or locked) else ORANGE, 0.75, 2)
        nconf = sum(1 for t in self.tracks if t.confirmed)
        self.text(canvas, f"TRACKS {nconf}/{len(self.tracks)}   ZONES {len(self.zones)}", (900, 36), WHITE, 0.6, 1)
        yb = 56 + vh + 26
        items = [("ALT", f"{self.alt:4.1f} m"), ("SPD", f"{self.speed:3.1f} m/s"), ("HDG", f"{self.yaw % 360:3.0f}"),
                 ("COV", f"{100 * st.coverage:3.0f} %" if st else "--"), ("KITS", f"{st.kits}" if st else "--"),
                 ("DEC", f"{st.decision_ms:3.1f} ms" if st else "--")]
        for i, (k, v) in enumerate(items):
            x = 14 + i * 106
            self.text(canvas, k, (x, yb), GREY, 0.5)
            self.text(canvas, v, (x, yb + 26), WHITE, 0.6)
        if st:
            bw = vw - 8
            frac = max(0.0, min(1.0, st.battery_pct / 100.0))
            full = st.battery_wh / max(frac, 1e-3) if st.battery_wh > 0 else 1.0
            thr = min(1.0, st.rth_threshold_wh / max(full, 1e-3))
            yy = yb + 50
            cv2.rectangle(canvas, (14, yy), (14 + bw, yy + 14), (70, 70, 70), -1)
            cv2.rectangle(canvas, (14, yy), (14 + int(bw * frac), yy + 14), GREEN if frac > thr + 0.1 else ORANGE, -1)
            cv2.line(canvas, (14 + int(bw * thr), yy - 4), (14 + int(bw * thr), yy + 18), RED, 2)
            self.text(canvas, f"BATTERY {st.battery_pct:3.0f}%  {st.battery_wh:4.1f} Wh   RTH {st.rth_threshold_wh:4.1f} Wh", (14, yy + 36), WHITE, 0.5)
        link = st.link_up if st else False
        gnss = st.gnss_ok if st else False
        nav = st.nav_source if st else "--"
        bar = f"SYSTEM {'OPERATIONAL' if phase != 'MAYDAY' else 'MAYDAY'}  |  LINK {'UP' if link else 'LOST'}  |  GNSS {'OK' if gnss else 'DENIED'}  |  NAV {nav}"
        self.text(canvas, bar, (vw + 30, H - 16), GREEN if (link and gnss and phase != "MAYDAY") else ORANGE, 0.55, 1)
        out = Image()
        out.header.stamp = self.get_clock().now().to_msg()
        out.height, out.width, out.encoding, out.step = H, W, "bgr8", W * 3
        out.data = canvas.tobytes()
        self.pub.publish(out)
        if self.window:
            try:
                cv2.imshow("SKYFORGE OPERATOR", canvas)
                cv2.waitKey(1)
            except cv2.error:
                self.window = False


def main():
    rclpy.init()
    n = Hud()
    try:
        rclpy.spin(n)
    except KeyboardInterrupt:
        pass
    finally:
        n.destroy_node()
        rclpy.try_shutdown()


if __name__ == "__main__":
    main()

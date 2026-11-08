"""vision_node: template wrapper for a detector (jury module or team F5 model).

The detection team only has to provide a model; this node handles ROS:
  /cam/rgb (sensor_msgs/Image)  ->  model  ->  /vision/<role> (std_msgs/String JSON)

JSON contract (docs/INTERFACES.md):
  {"stamp": <image stamp s>, "source": "team", "image": [w, h],
   "detections": [{"class": "injured", "confidence": 0.81, "bbox": [x1, y1, x2, y2],
                   "attrs": {"temp_k": 309.5, "posture": "lying"}}]}

Parameters:
  role       jury | team                  (output topic /vision/<role>)
  model      path to .pt (ultralytics) or .onnx (onnxruntime / ultralytics)
  conf       global confidence threshold (keep it low; the PerceptionAgent
             applies per-class thresholds and temporal confirmation)
  imgsz      inference size (640)
  classes    optional remap JSON {"0": "injured", "1": "hostile_unit"}
  thermal_topic  optional /cam/thermal: adds attrs.temp_k (mean of the box)
"""
from __future__ import annotations

import json
import time

import numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import Float32, String

from .conv import QOS_SENSOR

POSTURE_CLASSES = {
    "injured_lying": ("injured", {"posture": "lying"}),
    "injured_sitting": ("injured", {"posture": "sitting"}),
    "person_standing": ("injured", {"posture": "standing"}),
    "person_waving": ("injured", {"posture": "waving"}),
    "injured_water": ("injured", {"posture": "lying", "in_water": True}),
}


def image_to_numpy(m: Image) -> np.ndarray:
    ch = {"rgb8": 3, "bgr8": 3, "rgba8": 4, "bgra8": 4, "mono8": 1, "32FC1": 1, "16UC1": 1}.get(m.encoding, 3)
    dtype = np.float32 if m.encoding == "32FC1" else np.uint16 if m.encoding == "16UC1" else np.uint8
    a = np.frombuffer(m.data, dtype=dtype).reshape(m.height, -1)[:, : m.width * ch].reshape(m.height, m.width, ch)
    if m.encoding in ("rgb8", "rgba8"):
        a = a[:, :, 2::-1] if ch >= 3 else a  # -> BGR for ultralytics/opencv
    return a.squeeze() if ch == 1 else np.ascontiguousarray(a[:, :, :3])


class VisionNode(Node):
    def __init__(self):
        super().__init__("vision_node")
        self.declare_parameter("role", "team")
        self.declare_parameter("model", "")
        self.declare_parameter("conf", 0.15)
        self.declare_parameter("imgsz", 640)
        self.declare_parameter("classes", "")
        self.declare_parameter("image_topic", "/cam/rgb")
        self.declare_parameter("thermal_topic", "")
        self.role = self.get_parameter("role").value
        self.conf = float(self.get_parameter("conf").value)
        self.imgsz = int(self.get_parameter("imgsz").value)
        cmap = self.get_parameter("classes").value
        self.remap = json.loads(cmap) if cmap else {}
        path = self.get_parameter("model").value
        path = "" if path in ("none", "None") else path
        self.model = None
        if path:
            from ultralytics import YOLO  # also loads .onnx exports

            self.model = YOLO(path, task="detect")
            self.get_logger().info(f"{self.role} model loaded: {path}")
        else:
            self.get_logger().warn("no model given: publishing empty detections (pipeline test mode)")
        self.thermal = None
        tt = self.get_parameter("thermal_topic").value
        if tt:
            self.create_subscription(Image, tt, lambda m: setattr(self, "thermal", m), QOS_SENSOR)
        self.pub = self.create_publisher(String, f"/vision/{self.role}", 10)
        self.pub_lat = self.create_publisher(Float32, f"/vision/{self.role}/latency_ms", 10)
        self.create_subscription(Image, self.get_parameter("image_topic").value, self.on_image, QOS_SENSOR)
        self.busy = False

    def on_image(self, m: Image):
        if self.busy:
            return  # drop frames instead of queueing: always process the freshest
        self.busy = True
        t0 = time.perf_counter_ns()
        try:
            stamp = m.header.stamp.sec + m.header.stamp.nanosec * 1e-9
            dets = []
            if self.model is not None:
                img = image_to_numpy(m)
                res = self.model.predict(img, conf=self.conf, imgsz=self.imgsz, verbose=False)[0]
                names = res.names
                th = image_to_numpy(self.thermal) if self.thermal is not None else None
                for b in res.boxes:
                    k = int(b.cls)
                    x1, y1, x2, y2 = (float(v) for v in b.xyxy[0])
                    raw = self.remap.get(str(k), names.get(k, str(k)))
                    cls, attrs = POSTURE_CLASSES.get(raw, (raw, {}))
                    d = {"class": cls, "confidence": round(float(b.conf), 3),
                         "bbox": [round(x1, 1), round(y1, 1), round(x2, 1), round(y2, 1)], "attrs": dict(attrs)}
                    if th is not None and th.ndim == 2:
                        sy, sx = th.shape[0] / m.height, th.shape[1] / m.width
                        roi = th[int(y1 * sy): max(int(y2 * sy), int(y1 * sy) + 1), int(x1 * sx): max(int(x2 * sx), int(x1 * sx) + 1)]
                        if roi.size:
                            d["attrs"]["temp_k"] = round(float(np.mean(roi)), 1)  # radiometric image in kelvin
                    dets.append(d)
            out = {"stamp": stamp, "source": self.role, "image": [m.width, m.height], "detections": dets}
            self.pub.publish(String(data=json.dumps(out, separators=(",", ":"))))
        finally:
            self.busy = False
            self.pub_lat.publish(Float32(data=(time.perf_counter_ns() - t0) / 1e6))


def main():
    rclpy.init()
    n = VisionNode()
    try:
        rclpy.spin(n)
    except KeyboardInterrupt:
        pass
    finally:
        n.destroy_node()
        rclpy.try_shutdown()


if __name__ == "__main__":
    main()

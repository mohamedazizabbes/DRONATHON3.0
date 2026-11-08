import json
import time
from pathlib import Path

import numpy as np
from ultralytics import YOLO

SAFETY_CLASSES = {"mine", "weapon", "injured", "hostile_unit"}

POSTURE_CLASSES = {
    "injured_lying": ("injured", {"posture": "lying"}),
    "injured_sitting": ("injured", {"posture": "sitting"}),
    "person_standing": ("injured", {"posture": "standing"}),
    "injured_water": ("injured", {"posture": "lying", "in_water": True}),
}


class Detector:
    def __init__(self, weights, thresholds=None, imgsz=640, device=None, source="jury", class_map=None):
        self.model = YOLO(weights, task="detect")
        self.class_map = class_map
        if class_map:
            self.names = {i: class_map[n] for i, n in self.model.names.items() if n in class_map}
            self.classes = sorted(self.names)
        else:
            self.names = self.model.names
            self.classes = None
        self.imgsz = imgsz
        self.device = device
        self.source = source
        self.thresholds = {n: 0.25 for n in set(self.names.values())}
        for n in self.thresholds:
            if n in SAFETY_CLASSES:
                self.thresholds[n] = 0.2
        if thresholds:
            if isinstance(thresholds, (str, Path)):
                rep = json.loads(Path(thresholds).read_text())
                thresholds = {k: v["conf"] for k, v in rep["operating_thresholds"].items()}
            self.thresholds.update(thresholds)
        self.min_conf = min(self.thresholds.values())

    def __call__(self, frame, frame_id=0, stamp=None):
        t0 = time.perf_counter_ns()
        r = self.model.predict(frame, imgsz=self.imgsz, conf=self.min_conf, device=self.device, classes=self.classes, verbose=False)[0]
        dets = []
        b = r.boxes
        if len(b):
            xyxy = b.xyxy.cpu().numpy()
            cls = b.cls.cpu().numpy().astype(int)
            conf = b.conf.cpu().numpy()
            for (x0, y0, x1, y1), c, s in zip(xyxy, cls, conf):
                name = self.names[c]
                if s < self.thresholds.get(name, self.min_conf):
                    continue
                cls, attrs = POSTURE_CLASSES.get(name, (name, {}))
                dets.append({
                    "class": cls,
                    "confidence": round(float(s), 3),
                    "bbox": [round(float(x0), 1), round(float(y0), 1), round(float(x1), 1), round(float(y1), 1)],
                    "attrs": dict(attrs),
                })
        h, w = r.orig_shape
        return {
            "stamp": stamp if stamp is not None else time.time(),
            "source": self.source,
            "image": [int(w), int(h)],
            "detections": dets,
            "latency_ms": round((time.perf_counter_ns() - t0) / 1e6, 2),
        }


TEAM_V0_MAP = {
    "person": "injured",
    "car": "hostile_unit",
    "truck": "hostile_unit",
    "bus": "hostile_unit",
    "motorcycle": "hostile_unit",
    "boat": "hostile_unit",
}


if __name__ == "__main__":
    import argparse
    import cv2

    ap = argparse.ArgumentParser()
    ap.add_argument("weights")
    ap.add_argument("image")
    ap.add_argument("--thresholds", default=None)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--team-v0", action="store_true")
    a = ap.parse_args()
    det = Detector(a.weights, a.thresholds, a.imgsz, source="team" if a.team_v0 else "jury",
                   class_map=TEAM_V0_MAP if a.team_v0 else None)
    print(json.dumps(det(cv2.imread(a.image)), separators=(",", ":")))

import argparse
import threading
import time
from pathlib import Path

import numpy as np

from gz.msgs10.annotated_axis_aligned_2d_box_v_pb2 import AnnotatedAxisAligned2DBox_V
from gz.msgs10.image_pb2 import Image
from gz.transport13 import Node


def stamp_key(h):
    return h.stamp.sec * 1_000_000_000 + h.stamp.nsec


def to_bgr(msg):
    a = np.frombuffer(msg.data, np.uint8)
    if len(a) == msg.width * msg.height * 3:
        return a.reshape(msg.height, msg.width, 3)[:, :, ::-1].copy()
    if len(a) == msg.width * msg.height * 4:
        return a.reshape(msg.height, msg.width, 4)[:, :, 2::-1].copy()
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image-topic", default="/boxes_image")
    ap.add_argument("--boxes-topic", default="/boxes")
    ap.add_argument("--out", required=True)
    ap.add_argument("--scene", required=True)
    ap.add_argument("--hz", type=float, default=2.0)
    ap.add_argument("--max", type=int, default=500)
    ap.add_argument("--labels", default="1:0,2:1,3:2,4:3,5:4")
    ap.add_argument("--min-px", type=int, default=4)
    ap.add_argument("--keep-empty", type=float, default=0.1)
    args = ap.parse_args()

    import cv2

    lmap = {int(k): int(v) for k, v in (p.split(":") for p in args.labels.split(","))}
    out = Path(args.out)
    (out / "images").mkdir(parents=True, exist_ok=True)
    (out / "labels").mkdir(parents=True, exist_ok=True)

    images = {}
    lock = threading.Lock()
    state = {"n": 0, "last": 0.0, "empty": 0}
    rng = np.random.default_rng(0)

    def on_image(msg):
        with lock:
            images[stamp_key(msg.header)] = msg
            if len(images) > 30:
                for k in sorted(images)[:-30]:
                    images.pop(k)

    def on_boxes(msg):
        now = time.time()
        if now - state["last"] < 1.0 / args.hz or state["n"] >= args.max:
            return
        with lock:
            img = images.get(stamp_key(msg.header))
        if img is None:
            return
        frame = to_bgr(img)
        if frame is None:
            return
        h, w = frame.shape[:2]
        lines = []
        for ab in msg.annotated_box:
            if ab.label not in lmap:
                continue
            x0, y0 = max(0.0, ab.box.min_corner.x), max(0.0, ab.box.min_corner.y)
            x1, y1 = min(w, ab.box.max_corner.x), min(h, ab.box.max_corner.y)
            if x1 - x0 < args.min_px or y1 - y0 < args.min_px:
                continue
            lines.append(f"{lmap[ab.label]} {(x0 + x1) / 2 / w:.6f} {(y0 + y1) / 2 / h:.6f} {(x1 - x0) / w:.6f} {(y1 - y0) / h:.6f}")
        if not lines and rng.random() >= args.keep_empty:
            return
        stem = f"{args.scene}_{state['n']:05d}"
        cv2.imwrite(str(out / "images" / f"{stem}.jpg"), frame, [cv2.IMWRITE_JPEG_QUALITY, 92])
        (out / "labels" / f"{stem}.txt").write_text("\n".join(lines) + ("\n" if lines else ""))
        state["n"] += 1
        state["last"] = now
        if state["n"] % 25 == 0:
            print(f"{state['n']} frames", flush=True)

    node = Node()
    node.subscribe(Image, args.image_topic, on_image)
    node.subscribe(AnnotatedAxisAligned2DBox_V, args.boxes_topic, on_boxes)
    print(f"capturing {args.image_topic} + {args.boxes_topic} -> {out}", flush=True)
    try:
        while state["n"] < args.max:
            time.sleep(0.2)
    except KeyboardInterrupt:
        pass
    print(f"saved {state['n']} frames", flush=True)


if __name__ == "__main__":
    main()

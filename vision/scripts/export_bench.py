import argparse
import json
import time
from pathlib import Path

import numpy as np
from ultralytics import YOLO


def bench(model, imgs, imgsz, device, warmup=10, n=200):
    for i in range(warmup):
        model.predict(imgs[i % len(imgs)], imgsz=imgsz, device=device, verbose=False)
    ts = []
    for i in range(n):
        t0 = time.perf_counter_ns()
        model.predict(imgs[i % len(imgs)], imgsz=imgsz, device=device, verbose=False)
        ts.append((time.perf_counter_ns() - t0) / 1e6)
    a = np.array(ts)
    return {"p50_ms": round(float(np.percentile(a, 50)), 2), "p95_ms": round(float(np.percentile(a, 95)), 2),
            "p99_ms": round(float(np.percentile(a, 99)), 2), "fps": round(1000 / float(a.mean()), 1)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True)
    ap.add_argument("--images", required=True)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--n", type=int, default=200)
    args = ap.parse_args()

    import cv2
    imgs = [cv2.imread(str(p)) for p in sorted(Path(args.images).iterdir())[:50]]
    w = Path(args.weights)
    onnx_path = YOLO(str(w)).export(format="onnx", imgsz=args.imgsz, simplify=True, dynamic=False)

    res = {}
    res["torch_gpu"] = bench(YOLO(str(w)), imgs, args.imgsz, 0, n=args.n)
    res["torch_cpu"] = bench(YOLO(str(w)), imgs, args.imgsz, "cpu", n=args.n)
    res["onnx_cpu"] = bench(YOLO(onnx_path, task="detect"), imgs, args.imgsz, "cpu", n=args.n)
    out = w.parent.parent / "latency.json"
    out.write_text(json.dumps({"onnx": onnx_path, "imgsz": args.imgsz, **res}, indent=2))
    print(json.dumps(res, indent=2))
    print(f"onnx: {onnx_path}\nsaved {out}")


if __name__ == "__main__":
    main()

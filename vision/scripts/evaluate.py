import argparse
import json
from pathlib import Path

import numpy as np
from ultralytics import YOLO


def iou(a, b):
    x0 = np.maximum(a[:, None, 0], b[None, :, 0])
    y0 = np.maximum(a[:, None, 1], b[None, :, 1])
    x1 = np.minimum(a[:, None, 2], b[None, :, 2])
    y1 = np.minimum(a[:, None, 3], b[None, :, 3])
    inter = np.clip(x1 - x0, 0, None) * np.clip(y1 - y0, 0, None)
    aa = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1])
    bb = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    return inter / (aa[:, None] + bb[None, :] - inter + 1e-9)


def load_gt(label_path, w, h):
    rows = []
    if label_path.exists():
        for line in label_path.read_text().splitlines():
            p = line.split()
            if len(p) == 5:
                c, cx, cy, bw, bh = int(p[0]), *map(float, p[1:])
                rows.append([c, (cx - bw / 2) * w, (cy - bh / 2) * h, (cx + bw / 2) * w, (cy + bh / 2) * h])
    return np.array(rows).reshape(-1, 5)


def threshold_sweep(model, data_root, split, names, imgsz, iou_thr=0.5):
    img_dir = Path(data_root) / "images" / split
    lab_dir = Path(data_root) / "labels" / split
    thresholds = np.round(np.arange(0.05, 0.96, 0.05), 2)
    tp = np.zeros((len(names), len(thresholds)))
    fp = np.zeros_like(tp)
    npos = np.zeros(len(names))
    imgs = sorted(p for p in img_dir.iterdir())
    for res in (model.predict(str(p), imgsz=imgsz, conf=0.05, iou=0.6, verbose=False)[0] for p in imgs):
        h, w = res.orig_shape
        gt = load_gt(lab_dir / (Path(res.path).stem + ".txt"), w, h)
        b = res.boxes
        pc = b.cls.cpu().numpy().astype(int)
        ps = b.conf.cpu().numpy()
        pb = b.xyxy.cpu().numpy()
        for c in range(len(names)):
            g = gt[gt[:, 0] == c][:, 1:]
            npos[c] += len(g)
            m = pc == c
            for ti, t in enumerate(thresholds):
                sel = m & (ps >= t)
                boxes = pb[sel]
                scores = ps[sel]
                if len(boxes) == 0:
                    continue
                order = np.argsort(-scores)
                boxes = boxes[order]
                if len(g) == 0:
                    fp[c, ti] += len(boxes)
                    continue
                ious = iou(boxes, g)
                used = np.zeros(len(g), bool)
                for i in range(len(boxes)):
                    j = int(np.argmax(np.where(used, -1, ious[i])))
                    if ious[i, j] >= iou_thr and not used[j]:
                        used[j] = True
                        tp[c, ti] += 1
                    else:
                        fp[c, ti] += 1
    out = {}
    for c, n in enumerate(names):
        rec = tp[c] / max(npos[c], 1)
        prec = tp[c] / np.maximum(tp[c] + fp[c], 1)
        out[n] = {
            "gt": int(npos[c]),
            "sweep": [
                {"conf": float(t), "recall": round(float(r), 4), "precision": round(float(p), 4)}
                for t, r, p in zip(thresholds, rec, prec)
            ],
        }
    return out


def pick_thresholds(sweep, targets):
    picked = {}
    for name, d in sweep.items():
        target = targets.get(name)
        rows = d["sweep"]
        if target is not None:
            ok = [r for r in rows if r["recall"] >= target]
            best = max(ok, key=lambda r: r["conf"]) if ok else rows[0]
        else:
            best = max(rows, key=lambda r: 2 * r["precision"] * r["recall"] / max(r["precision"] + r["recall"], 1e-9))
        picked[name] = best
    return picked


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--split", default="test")
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--recall-target", action="append", default=[], help="class=0.95")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    model = YOLO(args.weights)
    m = model.val(data=args.data, split=args.split, imgsz=args.imgsz, plots=True, verbose=False)
    names = [model.names[i] for i in range(len(model.names))]
    per_class = {}
    for i, c in enumerate(m.box.ap_class_index):
        p, r, ap50, ap = m.box.class_result(i)
        per_class[names[c]] = {"precision": round(p, 4), "recall": round(r, 4), "mAP50": round(ap50, 4), "mAP50-95": round(ap, 4)}

    import yaml
    root = yaml.safe_load(open(args.data))["path"]
    sweep = threshold_sweep(model, root, args.split, names, args.imgsz)
    targets = {k: float(v) for k, v in (t.split("=") for t in args.recall_target)}
    picked = pick_thresholds(sweep, targets)

    report = {
        "weights": args.weights,
        "split": args.split,
        "overall": {"mAP50": round(float(m.box.map50), 4), "mAP50-95": round(float(m.box.map), 4),
                    "precision": round(float(m.box.mp), 4), "recall": round(float(m.box.mr), 4)},
        "per_class": per_class,
        "operating_thresholds": picked,
        "sweep": sweep,
    }
    out = Path(args.out or Path(args.weights).parent.parent / f"eval_{args.split}.json")
    out.write_text(json.dumps(report, indent=2))
    print(json.dumps({k: report[k] for k in ("overall", "per_class", "operating_thresholds")}, indent=2))
    print(f"saved {out}")


if __name__ == "__main__":
    main()

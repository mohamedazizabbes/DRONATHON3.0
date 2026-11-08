import argparse
import json
import os
import random
import re
from collections import Counter, defaultdict
from pathlib import Path

import yaml

IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def find_pairs(root):
    images = {}
    labels = {}
    for p in Path(root).rglob("*"):
        if not p.is_file():
            continue
        if p.suffix.lower() in IMG_EXT:
            images[str(p)] = p
        elif p.suffix == ".txt" and p.name.lower() not in {"classes.txt", "readme.txt", "readme.roboflow.txt"}:
            labels[(str(p.parent).replace("/labels", "/images"), p.stem)] = p
    pairs = []
    for key, img in images.items():
        lab = labels.get((str(img.parent), img.stem))
        if lab is None:
            lab = labels.get((str(img.parent).replace("/labels", "/images"), img.stem))
        pairs.append((img, lab))
    return pairs


def convert_line(parts):
    vals = [float(v) for v in parts[1:]]
    if len(vals) == 4:
        return vals
    if len(vals) >= 6 and len(vals) % 2 == 0:
        xs, ys = vals[0::2], vals[1::2]
        x0, x1 = max(0.0, min(xs)), min(1.0, max(xs))
        y0, y1 = max(0.0, min(ys)), min(1.0, max(ys))
        return [(x0 + x1) / 2, (y0 + y1) / 2, x1 - x0, y1 - y0]
    return None


def group_key(stem, pattern, chunk=0):
    if pattern:
        m = re.match(pattern, stem)
        if m:
            if chunk and m.lastindex and m.lastindex >= 2:
                return f"{m.group(1)}#{int(m.group(2)) // chunk}"
            return m.group(1)
    return re.split(r"\.rf\.|_jpg|_png|_aug", stem)[0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("config")
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.config))
    names = cfg["names"]
    split = cfg.get("split", {"train": 0.7, "val": 0.2, "test": 0.1})
    out = Path(args.out)
    rng = random.Random(args.seed)
    stats = defaultdict(Counter)

    for d in ("train", "val", "test"):
        (out / "images" / d).mkdir(parents=True, exist_ok=True)
        (out / "labels" / d).mkdir(parents=True, exist_ok=True)

    for src in cfg["sources"]:
        if not Path(src["path"]).exists() or not src["class_map"]:
            print(f"skip {src['name']}")
            continue
        cmap = {int(k): int(v) for k, v in src["class_map"].items()}
        keep_bg = src.get("keep_background", 0.0)
        min_wh = src.get("min_wh", 0.0)
        pairs = find_pairs(src["path"])
        groups = defaultdict(list)
        for img, lab in pairs:
            groups[group_key(img.stem, src.get("group_regex"), src.get("chunk", 0))].append((img, lab))
        keys = sorted(groups)
        rng.shuffle(keys)
        n = len(keys)
        cut1 = int(n * split["train"])
        cut2 = cut1 + int(n * split["val"])
        assign = {k: "train" if i < cut1 else "val" if i < cut2 else "test" for i, k in enumerate(keys)}

        for k in keys:
            d = assign[k]
            for img, lab in groups[k]:
                lines = []
                if lab is not None:
                    for raw in open(lab).read().splitlines():
                        parts = raw.split()
                        if not parts:
                            continue
                        c = int(float(parts[0]))
                        if c not in cmap:
                            continue
                        box = convert_line(parts)
                        if box is None or box[2] <= min_wh or box[3] <= min_wh:
                            continue
                        lines.append(f"{cmap[c]} " + " ".join(f"{v:.6f}" for v in box))
                        stats[src["name"]][f"{d}:{names[cmap[c]]}"] += 1
                if not lines and rng.random() >= keep_bg:
                    continue
                stats[src["name"]][f"{d}:images"] += 1
                reps = src.get("train_repeat", 1) if d == "train" else 1
                for r in range(reps):
                    stem = f"{src['name']}__{img.parent.name}__{img.stem}" + (f"__r{r}" if r else "")
                    dst = out / "images" / d / f"{stem}{img.suffix.lower()}"
                    if not dst.exists():
                        os.symlink(img.resolve(), dst)
                    (out / "labels" / d / f"{stem}.txt").write_text("\n".join(lines) + ("\n" if lines else ""))

    data_yaml = {
        "path": str(out.resolve()),
        "train": "images/train",
        "val": "images/val",
        "test": "images/test",
        "names": {i: n for i, n in enumerate(names)},
    }
    (out / "data.yaml").write_text(yaml.safe_dump(data_yaml, sort_keys=False))
    (out / "stats.json").write_text(json.dumps({k: dict(v) for k, v in stats.items()}, indent=2))
    print(json.dumps({k: dict(sorted(v.items())) for k, v in stats.items()}, indent=2))


if __name__ == "__main__":
    main()

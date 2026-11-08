import argparse
import random
from pathlib import Path

import cv2

IMG_EXT = {".jpg", ".jpeg", ".png"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("src")
    ap.add_argument("out")
    ap.add_argument("--tile", type=int, default=640)
    ap.add_argument("--overlap", type=int, default=64)
    ap.add_argument("--keep", default="4")
    ap.add_argument("--empty", type=float, default=0.05)
    ap.add_argument("--min-px", type=float, default=4)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    keep = {int(k) for k in a.keep.split(",")}
    rng = random.Random(a.seed)
    out_i = Path(a.out) / "images"
    out_l = Path(a.out) / "labels"
    out_i.mkdir(parents=True, exist_ok=True)
    out_l.mkdir(parents=True, exist_ok=True)
    n_tiles = n_boxes = 0
    for img_p in sorted(p for p in Path(a.src).rglob("*") if p.suffix.lower() in IMG_EXT):
        lab_p = Path(str(img_p.parent).replace("images", "labels")) / (img_p.stem + ".txt")
        if not lab_p.exists():
            continue
        im = cv2.imread(str(img_p))
        if im is None:
            continue
        H, W = im.shape[:2]
        boxes = []
        for line in lab_p.read_text().splitlines():
            p = line.split()
            if len(p) != 5:
                continue
            c = int(p[0])
            cx, cy, w, h = (float(v) for v in p[1:])
            boxes.append((c, (cx - w / 2) * W, (cy - h / 2) * H, (cx + w / 2) * W, (cy + h / 2) * H))
        step = a.tile - a.overlap
        xs = list(range(0, max(W - a.tile, 0) + 1, step)) or [0]
        ys = list(range(0, max(H - a.tile, 0) + 1, step)) or [0]
        if xs[-1] + a.tile < W:
            xs.append(W - a.tile)
        if ys[-1] + a.tile < H:
            ys.append(H - a.tile)
        base = img_p.stem.split(".rf.")[0]
        for ty in ys:
            for tx in xs:
                lines = []
                for c, x0, y0, x1, y1 in boxes:
                    ix0, iy0 = max(x0, tx), max(y0, ty)
                    ix1, iy1 = min(x1, tx + a.tile), min(y1, ty + a.tile)
                    if ix1 - ix0 < a.min_px or iy1 - iy0 < a.min_px:
                        continue
                    if (ix1 - ix0) * (iy1 - iy0) < 0.5 * (x1 - x0) * (y1 - y0):
                        continue
                    if c not in keep:
                        continue
                    t = a.tile
                    lines.append(f"{c} {((ix0 + ix1) / 2 - tx) / t:.6f} {((iy0 + iy1) / 2 - ty) / t:.6f} {(ix1 - ix0) / t:.6f} {(iy1 - iy0) / t:.6f}")
                if not lines and rng.random() >= a.empty:
                    continue
                stem = f"{base}_t{tx}_{ty}"
                cv2.imwrite(str(out_i / f"{stem}.jpg"), im[ty:ty + a.tile, tx:tx + a.tile], [cv2.IMWRITE_JPEG_QUALITY, 92])
                (out_l / f"{stem}.txt").write_text("\n".join(lines) + ("\n" if lines else ""))
                n_tiles += 1
                n_boxes += len(lines)
    print(f"tiles {n_tiles} boxes {n_boxes}")


if __name__ == "__main__":
    main()

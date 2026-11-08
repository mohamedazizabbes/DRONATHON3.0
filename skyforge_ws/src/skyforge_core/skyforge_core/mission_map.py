"""End-of-mission map (F1/F2 cartography): JSON for the station + PNG for the report.

Contents: operating area, coverage grid (RLE + %), every confirmed object
(class, position, sigma, START category, mine layers/depth, served / likely
deceased flags), exclusion zones, drone trajectory, deliveries.
"""
from __future__ import annotations

import json
import os
import time
from typing import Optional


def export_mission_map(out_dir: str, area: dict, tracks: list[dict], zones: list[dict], coverage=None,
                       trajectory: Optional[list] = None, served: Optional[list] = None, water=None,
                       name: Optional[str] = None) -> dict:
    os.makedirs(out_dir, exist_ok=True)
    stem = name or f"mission_map_{time.strftime('%Y%m%d_%H%M%S')}"
    doc = {
        "area": area,
        "coverage": round(coverage.fraction(), 3) if coverage is not None else None,
        "coverage_rle": coverage.to_rle() if coverage is not None else None,
        "objects": tracks,
        "zones": zones,
        "served": served or [],
        "trajectory": [[round(p[0], 1), round(p[1], 1)] for p in (trajectory or [])],
    }
    jpath = os.path.join(out_dir, stem + ".json")
    with open(jpath, "w", encoding="utf-8") as f:
        json.dump(doc, f, indent=1, default=str)
    ppath = None
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(11, 8))
        for poly in water or []:
            ax.fill([p[0] for p in poly], [p[1] for p in poly], color="#cfe8ff", zorder=0)
        if coverage is not None:
            import numpy as np

            ext = (coverage.x0, coverage.x0 + coverage.nx * coverage.res, coverage.y0, coverage.y0 + coverage.ny * coverage.res)
            ax.imshow(np.where(coverage.seen.T, 1.0, np.nan), origin="lower", extent=ext, cmap="Greens", alpha=0.25, vmin=0, vmax=2)
        ax.add_patch(plt.Rectangle((area["xmin"], area["ymin"]), area["xmax"] - area["xmin"], area["ymax"] - area["ymin"],
                                   fill=False, ls="--", color="gray"))
        for z in zones:
            ax.add_patch(plt.Circle(z["center"], z["radius"], color="red", alpha=0.15))
        colors = {"IMM": "red", "DEL": "orange", "MIN": "green", "EXP": "black", "DCD": "purple", "UNK": "gray"}
        for t in tracks:
            x, y = t["pos"]
            if t["cls"] == "injured":
                ax.plot(x, y, "P", ms=11, color=colors.get(t.get("category"), "gray"),
                        mec="gold" if t.get("served") else "k", mew=2 if t.get("served") else 0.5)
                ax.annotate(f"{t['id']} {t.get('category')}", (x, y), fontsize=7)
            else:
                mk = {"mine": "X", "weapon": "^", "hostile_unit": "s", "debris": "."}.get(t["cls"], "o")
                ax.plot(x, y, mk, color="darkred" if t["cls"] != "debris" else "brown", ms=8)
                lbl = t["id"] + (f" {'/'.join(t['layers'])} d{t.get('depth')}" if t.get("layers") else "")
                ax.annotate(lbl, (x, y), fontsize=6)
        if trajectory:
            ax.plot([p[0] for p in trajectory], [p[1] for p in trajectory], "b-", lw=0.7)
        m = 30.0
        ax.set_xlim(area["xmin"] - m, area["xmax"] + m)
        ax.set_ylim(area["ymin"] - m, area["ymax"] + m)
        ax.set_aspect("equal")
        cov = f"{100 * doc['coverage']:.0f}%" if doc["coverage"] is not None else "n/a"
        ax.set_title(f"SKYFORGE mission map - coverage {cov}, {len(tracks)} objects, {len(zones)} zones", fontsize=10)
        fig.tight_layout()
        ppath = os.path.join(out_dir, stem + ".png")
        fig.savefig(ppath, dpi=110)
        plt.close(fig)
    except Exception:  # noqa: BLE001 (PNG is best effort, JSON is the deliverable)
        ppath = None
    return {"json": jpath, "png": ppath}


def track_row(t, served_ids=()) -> dict:
    d = t.to_dict() if hasattr(t, "to_dict") else dict(t)
    attrs = getattr(t, "attrs", {}) or {}
    for k in ("layers", "depth", "submerged", "likely_deceased", "marked"):
        if k in attrs:
            d[k] = attrs[k]
    d["served"] = d.get("id") in set(served_ids)
    return d

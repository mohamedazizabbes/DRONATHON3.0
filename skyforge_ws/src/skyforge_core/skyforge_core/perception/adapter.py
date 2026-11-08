"""Adapter between the vision modules' JSON and `Detection`.

The jury module outputs "JSON (classe, confiance, boîte englobante)"; the exact
key names are not frozen by the spec, so this parser accepts the usual
variants. The team module (F5) must use the canonical form:

    {"stamp": 12.3, "source": "team", "image": [640, 480],
     "detections": [{"class": "injured", "confidence": 0.81,
                     "bbox": [x1, y1, x2, y2],
                     "attrs": {"temp_k": 309.5, "posture": "lying"}}]}

bbox formats accepted: [x1,y1,x2,y2] pixels (canonical), {"x","y","w","h"}
(top-left + size), {"xc","yc","w","h"} / "xywhn" normalised (YOLO style).
"""
from __future__ import annotations

import json
from typing import Any, Iterable, Union

from ..domain import Cls, Detection

# Raw detector labels -> SKYFORGE classes. Extend freely (lower-case keys).
CLASS_ALIASES: dict[str, str] = {
    # weapons (Kaggle gun-detection + jury)
    "weapon": Cls.WEAPON, "weapons": Cls.WEAPON, "arme": Cls.WEAPON, "armes": Cls.WEAPON,
    "gun": Cls.WEAPON, "pistol": Cls.WEAPON, "handgun": Cls.WEAPON, "rifle": Cls.WEAPON,
    "knife": Cls.WEAPON, "firearm": Cls.WEAPON,
    # mines (Roboflow + jury)
    "mine": Cls.MINE, "mines": Cls.MINE, "landmine": Cls.MINE, "sea_mine": Cls.MINE,
    "naval_mine": Cls.MINE, "uxo": Cls.MINE,
    # marine debris (Zenodo TUN-MarineLitter + jury)
    "debris": Cls.DEBRIS, "dechet": Cls.DEBRIS, "déchet": Cls.DEBRIS, "dechets": Cls.DEBRIS,
    "déchets": Cls.DEBRIS, "litter": Cls.DEBRIS, "trash": Cls.DEBRIS, "plastic": Cls.DEBRIS,
    "bottle": Cls.DEBRIS, "bag": Cls.DEBRIS, "net": Cls.DEBRIS, "can": Cls.DEBRIS,
    "rope": Cls.DEBRIS, "foam": Cls.DEBRIS, "wood": Cls.DEBRIS, "marine_litter": Cls.DEBRIS,
    # casualties (team F5)
    "injured": Cls.INJURED, "blesse": Cls.INJURED, "blessé": Cls.INJURED, "casualty": Cls.INJURED,
    "victim": Cls.INJURED, "swimmer": Cls.INJURED, "floater": Cls.INJURED, "person_lying": Cls.INJURED,
    # hostile units (team F5)
    "hostile_unit": Cls.HOSTILE, "hostile": Cls.HOSTILE, "unite_hostile": Cls.HOSTILE,
    "unité_hostile": Cls.HOSTILE, "soldier": Cls.HOSTILE, "tank": Cls.HOSTILE,
    "warship": Cls.HOSTILE, "patrol_vessel": Cls.HOSTILE, "military_vehicle": Cls.HOSTILE,
}

_CLASS_KEYS = ("class", "classe", "cls", "label", "name", "class_name")
_CONF_KEYS = ("confidence", "confiance", "conf", "score", "probability")
_BOX_KEYS = ("bbox", "box", "boite", "boîte", "boite_englobante", "boîte_englobante", "xyxy", "bounding_box")


def map_class(raw: str) -> str | None:
    if raw is None:
        return None
    k = str(raw).strip().lower().replace(" ", "_").replace("-", "_")
    return CLASS_ALIASES.get(k)


def _first(d: dict, keys) -> Any:
    for k in keys:
        if k in d:
            return d[k]
    return None


def _bbox(raw: Any, img_w: float, img_h: float) -> tuple[float, float, float, float] | None:
    if raw is None:
        return None
    if isinstance(raw, dict):
        if {"x1", "y1", "x2", "y2"} <= raw.keys():
            b = (raw["x1"], raw["y1"], raw["x2"], raw["y2"])
        elif {"xc", "yc", "w", "h"} <= raw.keys():
            xc, yc, w, h = raw["xc"], raw["yc"], raw["w"], raw["h"]
            b = (xc - w / 2, yc - h / 2, xc + w / 2, yc + h / 2)
        elif {"x", "y", "w", "h"} <= raw.keys():
            b = (raw["x"], raw["y"], raw["x"] + raw["w"], raw["y"] + raw["h"])
        elif {"x", "y", "width", "height"} <= raw.keys():
            b = (raw["x"], raw["y"], raw["x"] + raw["width"], raw["y"] + raw["height"])
        else:
            return None
    else:
        b = tuple(float(v) for v in raw)
        if len(b) != 4:
            return None
    b = tuple(float(v) for v in b)
    # Normalised coordinates -> pixels.
    if max(abs(v) for v in b) <= 1.0 + 1e-6:
        b = (b[0] * img_w, b[1] * img_h, b[2] * img_w, b[3] * img_h)
    x1, y1, x2, y2 = b
    if x2 < x1 or y2 < y1:  # [x, y, w, h] given as a list
        x2, y2 = x1 + x2, y1 + y2
    return (x1, y1, x2, y2)


def parse_detections(
    payload: Union[str, bytes, dict, list],
    img_w: float = 640,
    img_h: float = 480,
    default_source: str = "jury",
    default_stamp: float = 0.0,
) -> list[Detection]:
    if isinstance(payload, (bytes, str)):
        payload = json.loads(payload)
    if isinstance(payload, dict):
        img = payload.get("image") or payload.get("image_size")
        if img:
            img_w, img_h = float(img[0]), float(img[1])
        stamp = float(payload.get("stamp", payload.get("ts", default_stamp)))
        source = payload.get("source", default_source)
        items: Iterable = next(
            (payload[k] for k in ("detections", "objects", "objets", "results", "predictions") if k in payload), []
        )
    else:
        stamp, source, items = default_stamp, default_source, payload
    out: list[Detection] = []
    for it in items:
        cls = map_class(_first(it, _CLASS_KEYS))
        if cls is None:
            continue
        conf = _first(it, _CONF_KEYS)
        conf = float(conf) if conf is not None else 0.5
        if conf > 1.0:  # percentages
            conf /= 100.0
        box = _bbox(_first(it, _BOX_KEYS) if any(k in it for k in _BOX_KEYS) else it, img_w, img_h)
        if box is None:
            continue
        out.append(Detection(cls, conf, box, stamp, source, dict(it.get("attrs", {}))))
    return out


def detections_to_json(dets: list[Detection], stamp: float, source: str, img_w: int, img_h: int) -> str:
    return json.dumps(
        {
            "stamp": stamp,
            "source": source,
            "image": [img_w, img_h],
            "detections": [
                {"class": d.cls, "confidence": round(d.conf, 3), "bbox": [round(v, 1) for v in d.bbox], "attrs": d.attrs}
                for d in dets
            ],
        },
        separators=(",", ":"),
    )

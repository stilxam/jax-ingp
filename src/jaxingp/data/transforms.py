import json
import math
import os
from dataclasses import dataclass


@dataclass(frozen=True)
class Frame:
    file_path: str
    transform_matrix: list


@dataclass(frozen=True)
class TransformsData:
    fl_x: float
    fl_y: float
    cx: float
    cy: float
    w: float
    h: float
    k1: float
    k2: float
    p1: float
    p2: float
    aabb_scale: float
    frames: list
    base_dir: str


def load_transforms(path: str) -> TransformsData:
    with open(path) as f:
        d = json.load(f)

    w, h = d["w"], d["h"]
    fl_x = d.get("fl_x") or (0.5 * w / math.tan(d["camera_angle_x"] / 2.0))
    fl_y = d.get("fl_y") or (0.5 * h / math.tan(d.get("camera_angle_y", d["camera_angle_x"]) / 2.0))

    return TransformsData(
        fl_x=fl_x,
        fl_y=fl_y,
        cx=d.get("cx", w / 2),
        cy=d.get("cy", h / 2),
        w=w,
        h=h,
        k1=d.get("k1", 0.0),
        k2=d.get("k2", 0.0),
        p1=d.get("p1", 0.0),
        p2=d.get("p2", 0.0),
        aabb_scale=d.get("aabb_scale", 1),
        frames=[Frame(fr["file_path"], fr["transform_matrix"]) for fr in d["frames"]],
        base_dir=os.path.dirname(os.path.abspath(path)),
    )

"""2D -> 3D: pixel ray intersected with the ground/water plane z = 0.

Downward camera on the body (FLU). Optical frame: x right, y down, z forward
(along the optical axis). With the image top pointing to the drone's nose:
    cam x -> body -y,  cam y -> body -x,  cam z -> body -z
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np

from ..geometry import rot_world_body

R_BODY_CAM = np.array([[0.0, -1.0, 0.0], [-1.0, 0.0, 0.0], [0.0, 0.0, -1.0]])


@dataclass
class Camera:
    width: int
    height: int
    fx: float
    fy: float
    cx: float
    cy: float
    mount: tuple[float, float, float] = (0.0, 0.0, -0.1)
    tilt_deg: float = 0.0  # forward tilt from nadir (about body y)

    @property
    def R_body_cam(self):
        import numpy as np

        th = math.radians(self.tilt_deg)
        c, s = math.cos(th), math.sin(th)
        # rotation about body y by -tilt: optical axis (0,0,-1) -> (sin t, 0, -cos t)
        ry = np.array([[c, 0.0, -s], [0.0, 1.0, 0.0], [s, 0.0, c]])
        return ry @ R_BODY_CAM

    @classmethod
    def from_params(cls, p: dict) -> "Camera":
        c = p["camera"]
        w, h = int(c["width"]), int(c["height"])
        fx = (w / 2.0) / math.tan(math.radians(c["hfov_deg"]) / 2.0)
        return cls(w, h, fx, fx, w / 2.0, h / 2.0, tuple(c.get("mount_offset", (0, 0, -0.1))), float(c.get("tilt_deg", 0.0)))

    def with_K(self, K: Sequence[float], width: int, height: int) -> "Camera":
        """Override with a ROS CameraInfo K (row-major 3x3)."""
        return Camera(width, height, K[0], K[4], K[2], K[5], self.mount, self.tilt_deg)

    @property
    def hfov_deg(self) -> float:
        return math.degrees(2 * math.atan((self.width / 2) / self.fx))

    @property
    def vfov_deg(self) -> float:
        return math.degrees(2 * math.atan((self.height / 2) / self.fy))


def pixel_to_ground(u: float, v: float, pos, rpy, cam: Camera, ground_z: float = 0.0) -> Optional[tuple[float, float]]:
    R = rot_world_body(*rpy) @ cam.R_body_cam
    ray = R @ np.array([(u - cam.cx) / cam.fx, (v - cam.cy) / cam.fy, 1.0])
    origin = np.asarray(pos, dtype=float) + rot_world_body(*rpy) @ np.asarray(cam.mount)
    if ray[2] > -1e-6:
        return None  # ray does not hit the ground
    t = (ground_z - origin[2]) / ray[2]
    if t <= 0:
        return None
    p = origin + t * ray
    return (float(p[0]), float(p[1]))


def world_to_pixel(p_world, pos, rpy, cam: Camera, ground_z: float = 0.0) -> Optional[tuple[float, float]]:
    Rwb = rot_world_body(*rpy)
    origin = np.asarray(pos, dtype=float) + Rwb @ np.asarray(cam.mount)
    pw = np.array([p_world[0], p_world[1], p_world[2] if len(p_world) > 2 else ground_z])
    c = cam.R_body_cam.T @ (Rwb.T @ (pw - origin))
    if c[2] <= 0.1:
        return None
    return (cam.fx * c[0] / c[2] + cam.cx, cam.fy * c[1] / c[2] + cam.cy)


def in_view(p_world, pos, rpy, cam: Camera, margin_px: float = 0.0) -> bool:
    px = world_to_pixel(p_world, pos, rpy, cam)
    if px is None:
        return False
    return margin_px <= px[0] <= cam.width - margin_px and margin_px <= px[1] <= cam.height - margin_px


def ground_sigma(u: float, v: float, alt: float, cam: Camera, pixel_sigma: float, att_sigma: float) -> float:
    """1-sigma horizontal error of a projected point (pixel noise + attitude error)."""
    alt = max(alt, 0.5)
    off = math.hypot((u - cam.cx) / cam.fx, (v - cam.cy) / cam.fy)  # tan(angle off nadir)
    slant = 1.0 + off * off  # 1/cos^2
    gsd = alt / cam.fx
    return math.sqrt((pixel_sigma * gsd * slant) ** 2 + (alt * att_sigma * slant) ** 2) + 0.2

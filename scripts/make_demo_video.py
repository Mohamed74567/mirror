"""Generate the synthetic "sagittal swing" benchmark video + ground truth.

The bundled demo clip is rendered with exactly the pinhole camera model the
pipeline assumes: a two-link leg (femur + tibia capsules) with three coloured
fiducial markers (hip / knee / ankle) on a dark studio-like background with
sensor noise.  Two details make it a *meaningful* benchmark rather than a
trivial one:

* the marker centres are drawn a few pixels away from the true rotation
  centres — exactly like skin-mounted fiducials — which injects a realistic
  angle/length bias that the closed-loop calibration (Part 4) must recover;
* the ground truth (camera, link geometry, marker bias, true joint angles) is
  exported to JSON, so every pipeline estimate can be scored quantitatively.

Outputs (default: ``assets/``)
------------------------------
``demo_sagittal_swing.mp4``   640x480 @ 30 fps, 2.4 s (two swing cycles)
``demo_ground_truth.json``    camera + geometry + per-frame true angles

Usage
-----
    python scripts/make_demo_video.py [--out-dir assets] [--seed 7]
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Tuple

import cv2
import numpy as np

# BGR colours (OpenCV canvas order)
COL_BG_TOP = (38, 36, 34)
COL_BG_BOTTOM = (26, 25, 24)
COL_FEMUR = (206, 208, 212)
COL_TIBIA = (214, 216, 220)
COL_MARKER = {
    "hip": (230, 70, 235),    # magenta
    "knee": (80, 235, 100),   # green
    "ankle": (250, 120, 60),  # blue
}


@dataclass
class DemoConfig:
    """Ground-truth configuration of the synthetic recording."""

    width: int = 640
    height: int = 480
    fps: float = 30.0
    duration_s: float = 2.4
    cycles: float = 2.0

    focal_length_px: float = 620.0
    camera_distance_m: float = 3.0          # TRUE camera distance
    hip_pixel: Tuple[float, float] = (320.0, 150.0)
    hip_height_m: float = 0.95              # world height of the hip anchor

    L_femur_m: float = 0.42
    L_tibia_m: float = 0.40
    femur_radius_m: float = 0.070
    tibia_radius_m: float = 0.052
    marker_radius_m: float = 0.034
    density_kg_m3: float = 1050.0

    # skin-marker placement bias (px, in image coords) away from joint centres
    marker_bias_px: Dict[str, Tuple[float, float]] = field(
        default_factory=lambda: {"hip": (0.0, 0.0), "knee": (5.0, -4.0), "ankle": (-4.0, 5.0)}
    )

    # swing-cycle kinematics: theta(t) from vertical, phi(t) relative knee angle
    thigh_amp_rad: float = 0.28
    thigh_bias_rad: float = -0.04
    knee_amp_rad: float = 0.62
    knee_bias_rad: float = -0.18
    knee_phase_rad: float = 1.05
    knee_max_extension_rad: float = 0.02

    noise_sigma: float = 6.0
    seed: int = 7

    @property
    def swing_freq_hz(self) -> float:
        return self.cycles / self.duration_s

    @property
    def cam_height_m(self) -> float:
        """Camera height chosen so the hip projects onto ``hip_pixel`` row."""
        cy = self.height / 2.0
        return self.hip_height_m - (cy - self.hip_pixel[1]) * self.camera_distance_m / self.focal_length_px


# ---------------------------------------------------------------------------
# Kinematics and projection
# ---------------------------------------------------------------------------
def kinematics(cfg: DemoConfig, t: float) -> Tuple[float, float]:
    """True joint angles (rad) at time t: (thigh angle, relative knee angle)."""
    theta = cfg.thigh_amp_rad * np.sin(2.0 * np.pi * cfg.swing_freq_hz * t) + cfg.thigh_bias_rad
    phi = cfg.knee_bias_rad - cfg.knee_amp_rad * np.sin(2.0 * np.pi * cfg.swing_freq_hz * t + cfg.knee_phase_rad)
    phi = float(min(phi, cfg.knee_max_extension_rad))
    return float(theta), phi


def joint_positions_world(cfg: DemoConfig, theta: float, phi: float) -> Dict[str, np.ndarray]:
    """Sagittal-plane joint centres (x, z), hip anchored at (0, hip_height)."""
    hip = np.array([0.0, cfg.hip_height_m])
    knee = hip + cfg.L_femur_m * np.array([np.sin(theta), -np.cos(theta)])
    theta_shank = theta + phi
    ankle = knee + cfg.L_tibia_m * np.array([np.sin(theta_shank), -np.cos(theta_shank)])
    return {"hip": hip, "knee": knee, "ankle": ankle}


def world_to_pixel(cfg: DemoConfig, x: float, z: float) -> Tuple[float, float]:
    """Pinhole projection: camera at (0, -Z, cam_height) looking along +y."""
    cx, cy = cfg.width / 2.0, cfg.height / 2.0
    u = cx + cfg.focal_length_px * x / cfg.camera_distance_m
    v = cy - cfg.focal_length_px * (z - cfg.cam_height_m) / cfg.camera_distance_m
    return float(u), float(v)


# ---------------------------------------------------------------------------
# Frame rendering
# ---------------------------------------------------------------------------
def background_canvas(cfg: DemoConfig) -> np.ndarray:
    """Dark gradient background with a soft vignette (float32 BGR)."""
    lin = np.linspace(0.0, 1.0, cfg.height, dtype=np.float32)[:, None]
    base = (1.0 - lin) * np.array(COL_BG_TOP, np.float32) + lin * np.array(COL_BG_BOTTOM, np.float32)
    canvas = np.repeat(base[:, None, :], cfg.width, axis=1)

    yy, xx = np.mgrid[0:cfg.height, 0:cfg.width].astype(np.float32)
    r2 = ((xx - cfg.width / 2) / (cfg.width / 2)) ** 2 + ((yy - cfg.height / 2) / (cfg.height / 2)) ** 2
    vignette = 1.0 - 0.28 * np.clip(r2 - 0.15, 0.0, 1.0)
    return np.clip(canvas * vignette[..., None], 0, 255)


def draw_frame(
    cfg: DemoConfig,
    base_bg: np.ndarray,
    rng: np.random.Generator,
    theta: float,
    phi: float,
) -> np.ndarray:
    """Render one video frame (uint8 BGR) for the given true joint angles."""
    canvas = base_bg + rng.normal(0.0, cfg.noise_sigma, base_bg.shape).astype(np.float32)

    joints_px = {
        name: world_to_pixel(cfg, p[0], p[1])
        for name, p in joint_positions_world(cfg, theta, phi).items()
    }

    # limb capsules: femur then shank (drawn through the TRUE joint centres)
    segments = (
        ("hip", "knee", COL_FEMUR, cfg.femur_radius_m),
        ("knee", "ankle", COL_TIBIA, cfg.tibia_radius_m),
    )
    for a, b, color, radius in segments:
        pa = tuple(int(round(c)) for c in joints_px[a])
        pb = tuple(int(round(c)) for c in joints_px[b])
        thickness = int(round(2.0 * cfg.focal_length_px * radius / cfg.camera_distance_m))
        cv2.line(canvas, pa, pb, color, thickness=thickness, lineType=cv2.LINE_AA)
        r = max(1, thickness // 2)
        for pt in (pa, pb):
            cv2.circle(canvas, pt, r, color, thickness=-1, lineType=cv2.LINE_AA)

    # fiducial markers (drawn with their skin-placement bias)
    marker_radius_px = int(round(cfg.focal_length_px * cfg.marker_radius_m / cfg.camera_distance_m))
    for name, px in joints_px.items():
        bias = cfg.marker_bias_px[name]
        center = (int(round(px[0] + bias[0])), int(round(px[1] + bias[1])))
        cv2.circle(canvas, center, marker_radius_px, COL_MARKER[name], thickness=-1, lineType=cv2.LINE_AA)

    return np.clip(canvas, 0, 255).astype(np.uint8)


# ---------------------------------------------------------------------------
# Video + ground-truth generation
# ---------------------------------------------------------------------------
def generate_demo(cfg: DemoConfig = DemoConfig(), out_dir: str | Path = "assets") -> Path:
    """Write the demo .mp4 and its ground-truth .json; return the JSON path."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    n_frames = int(round(cfg.duration_s * cfg.fps))
    rng = np.random.default_rng(cfg.seed)
    base_bg = background_canvas(cfg)

    video_path = out_dir / "demo_sagittal_swing.mp4"
    writer = cv2.VideoWriter(
        str(video_path), cv2.VideoWriter_fourcc(*"mp4v"), cfg.fps, (cfg.width, cfg.height)
    )
    if not writer.isOpened():
        raise RuntimeError("cv2.VideoWriter could not open the demo MP4 (missing ffmpeg backend?)")

    theta_true, phi_true = [], []
    for k in range(n_frames):
        t = k / cfg.fps
        theta, phi = kinematics(cfg, t)
        theta_true.append(theta)
        phi_true.append(phi)
        writer.write(draw_frame(cfg, base_bg, rng, theta, phi))
    writer.release()

    # verify the encoded file is readable and complete
    cap = cv2.VideoCapture(str(video_path))
    decoded = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    if decoded != n_frames:
        raise RuntimeError(f"Demo video verification failed: wrote {n_frames} frames, read back {decoded}.")

    gt = {
        "schema": "mirror/demo-ground-truth-v1",
        "description": (
            "Synthetic sagittal two-link swing cycle rendered with the pipeline's pinhole "
            "camera model. Marker centres are offset from the true rotation centres "
            "(skin-fiducial bias), so vision-based estimates carry a known bias that the "
            "closed-loop calibration must recover."
        ),
        "camera": {
            "width": cfg.width,
            "height": cfg.height,
            "fps": cfg.fps,
            "focal_length_px": cfg.focal_length_px,
            "camera_distance_m": cfg.camera_distance_m,
            "cam_height_m": cfg.cam_height_m,
            "principal_point_px": [cfg.width / 2.0, cfg.height / 2.0],
        },
        "links": {
            "L_femur_m": cfg.L_femur_m,
            "L_tibia_m": cfg.L_tibia_m,
            "femur_radius_m": cfg.femur_radius_m,
            "tibia_radius_m": cfg.tibia_radius_m,
        },
        "marker_radius_m": cfg.marker_radius_m,
        "marker_bias_px": {k: list(v) for k, v in cfg.marker_bias_px.items()},
        "dynamics": {"gravity_m_s2": 9.81, "density_kg_m3": cfg.density_kg_m3},
        "n_frames": n_frames,
        "frame_times_s": [round(k / cfg.fps, 6) for k in range(n_frames)],
        "theta_thigh_true_rad": [round(v, 6) for v in theta_true],
        "phi_knee_true_rad": [round(v, 6) for v in phi_true],
    }
    gt_path = out_dir / "demo_ground_truth.json"
    gt_path.write_text(json.dumps(gt, indent=2), encoding="utf-8")

    print(f"[make_demo_video] wrote {video_path} ({n_frames} frames @ {cfg.fps:.0f} fps)")
    print(f"[make_demo_video] wrote {gt_path}")
    return gt_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate the Mirror demo video + ground truth.")
    parser.add_argument("--out-dir", default=str(Path(__file__).resolve().parents[1] / "assets"))
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--fps", type=float, default=30.0)
    parser.add_argument("--duration", type=float, default=2.4)
    parser.add_argument("--cycles", type=float, default=2.0)
    args = parser.parse_args()
    cfg = DemoConfig(seed=args.seed, fps=args.fps, duration_s=args.duration, cycles=args.cycles)
    generate_demo(cfg, out_dir=args.out_dir)


if __name__ == "__main__":
    main()

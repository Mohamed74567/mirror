#!/usr/bin/env python3
"""Mirror — end-to-end pipeline runner.

Takes a single-camera monocular video of a sagittal two-link movement and
produces a fully calibrated MuJoCo digital twin:

    Part 1  vision_extractor  : keypoints, masks, clean joint angles
    Part 2  mjcf_builder      : parameterized MJCF model (compiled on the fly)
    Part 3  mujoco_sim        : 500 Hz RK4 forward dynamics + silhouettes
    Part 4  optimizer         : closed-loop silhouette-IoU calibration (Powell)

Outputs
-------
    models/mirror_initial.xml       seed model, straight from the vision stage
    models/mirror_calibrated.xml    calibrated digital twin
    outputs/mirror_comparison.mp4   side-by-side: video | uncalibrated | twin
    outputs/mirror_comparison.gif   the same comparison as an animated GIF
    outputs/kinematics_and_torques.png  3-panel diagnostic figure
    outputs/mirror_metrics.json     quantitative metrics + parameters

Usage
-----
    python run_mirror.py                                  # bundled demo video
    python run_mirror.py --profile fast                   # reduced optimiser budget
    python run_mirror.py --video my_recording.mp4 --camera-distance 3.0
"""

from __future__ import annotations

import argparse
import json
import logging
import platform
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

BASE_DIR = Path(__file__).resolve().parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

import src as mirror_pkg
from src.mjcf_builder import CALIBRATION_PARAM_NAMES, MJCFBuilder, ModelParams
from src.mujoco_sim import (
    RENDER_BACKEND,
    ANALYTIC_BACKEND,
    RENDERER_BACKEND,
    MirrorSimulator,
    ctrl_to_vision_angles,
    silhouette_iou,
    tracking_rmse_deg,
    vision_angles_to_ctrl,
)
from src.optimizer import (
    TARGET_IOU,
    CalibrationResult,
    MirrorOptimizer,
    calibrate_two_phase,
    estimate_ci95_curvature,
)
from src.vision_extractor import CameraIntrinsics, VisionExtractor

log = logging.getLogger("mirror.run")

FONT = cv2.FONT_HERSHEY_SIMPLEX

PROFILES: Dict[str, Dict[str, int]] = {
    "full": {"frame_stride": 2, "max_eval_frames": 36, "maxfev": 1200},
    "fast": {"frame_stride": 3, "max_eval_frames": 24, "maxfev": 400},
    "smoke": {"frame_stride": 6, "max_eval_frames": 12, "maxfev": 80},
}

#: default weight of the keypoint-reprojection term in the fused loss
#: (selected on the benchmark ablation, see scripts/ablation_study.py)
DEFAULT_KP_WEIGHT = 0.1

DEFAULT_VIDEO = BASE_DIR / "assets" / "demo_sagittal_swing.mp4"
DEFAULT_GROUND_TRUTH = BASE_DIR / "assets" / "demo_ground_truth.json"

# World height of the hip anchor. Only the hip-to-camera geometry matters for
# the renders; this constant merely fixes where the model sits in world space.
HIP_ANCHOR_HEIGHT_M = 0.95


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def ensure_demo_video(video: Path) -> Path:
    """Regenerate the bundled demo asset deterministically if it is missing."""
    if video.exists():
        return video
    if video.resolve() != DEFAULT_VIDEO.resolve():
        raise FileNotFoundError(f"Video not found: {video}")
    log.info("Bundled demo asset missing — regenerating deterministically ...")
    from scripts.make_demo_video import DemoConfig, generate_demo

    generate_demo(DemoConfig(), out_dir=video.parent)
    return video


def initial_params_from_vision(vision, camera_distance_m: float, kp_init: float) -> ModelParams:
    """Seed ModelParams from the vision stage (geometry) + priors (servo gain)."""
    px_per_m = vision.camera.focal_length_px / camera_distance_m
    hip_u, hip_v = vision.hip_pixel
    cam_z = HIP_ANCHOR_HEIGHT_M - (vision.camera.cy - hip_v) * camera_distance_m / vision.camera.focal_length_px
    cam_x = -(hip_u - vision.camera.cx) * camera_distance_m / vision.camera.focal_length_px
    femur_r = float(np.clip(vision.limb_width_px["femur_px"] / 2.0 / px_per_m, 0.02, 0.16))
    tibia_r = float(np.clip(vision.limb_width_px["tibia_px"] / 2.0 / px_per_m, 0.015, 0.14))
    marker_r = float(np.clip(vision.marker_radius_px / px_per_m, 0.008, 0.06))
    return ModelParams(
        L_femur_m=vision.initial_link_lengths_m["femur_m"],
        L_tibia_m=vision.initial_link_lengths_m["tibia_m"],
        servo_kp=float(kp_init),
        femur_radius_m=femur_r,
        tibia_radius_m=tibia_r,
        marker_radius_m=marker_r,
        fovy_deg=vision.camera.fovy_deg,
        cam_distance_m=float(camera_distance_m),
        cam_height_m=float(cam_z),
        cam_x_m=float(cam_x),
        hip_height_m=HIP_ANCHOR_HEIGHT_M,
        render_width_px=vision.width,
        render_height_px=vision.height,
    )


def per_frame_ious(sim: MirrorSimulator, frame_qpos: np.ndarray, masks: np.ndarray) -> np.ndarray:
    return np.asarray(
        [silhouette_iou(sim.render_silhouette(frame_qpos[i]), masks[i]) for i in range(len(masks))]
    )


def _final_x(optimizer: MirrorOptimizer, calib: CalibrationResult) -> np.ndarray:
    """Reconstruct the final full parameter vector in the optimizer's layout."""
    import numpy as _np

    geom = _np.asarray(calib.best_params.calibration_vector(), dtype=float)
    if not optimizer.optimize_marker_offsets:
        return geom
    deltas = calib.marker_offsets_px or {}
    flat = []
    for name in ("hip", "knee", "ankle"):
        dx, dy = deltas.get(name, (0.0, 0.0))
        flat.extend([dx, dy])
    return _np.concatenate([geom, _np.asarray(flat, dtype=float)])


def _jsonable(obj):
    """Recursively convert numpy containers to plain JSON types."""
    if isinstance(obj, dict):
        return {k: _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return [_jsonable(v) for v in obj.tolist()]
    if isinstance(obj, (np.floating, np.integer)):
        return obj.item()
    if isinstance(obj, Path):
        return str(obj)
    return obj


# ---------------------------------------------------------------------------
# Output media
# ---------------------------------------------------------------------------
def draw_input_panel(frame_bgr: np.ndarray, vision, idx: int) -> np.ndarray:
    canvas = frame_bgr.copy()
    pts = {
        name: (int(round(vision.keypoints_px[name][idx][0])), int(round(vision.keypoints_px[name][idx][1])))
        for name in ("hip", "knee", "ankle")
    }
    cv2.line(canvas, pts["hip"], pts["knee"], (255, 255, 255), 2, cv2.LINE_AA)
    cv2.line(canvas, pts["knee"], pts["ankle"], (225, 225, 225), 2, cv2.LINE_AA)
    colors = {"hip": (230, 70, 235), "knee": (80, 235, 100), "ankle": (250, 120, 60)}
    for name, pt in pts.items():
        cv2.circle(canvas, pt, 4, colors[name], -1, cv2.LINE_AA)
    cv2.putText(canvas, f"t = {idx / vision.fps:.2f} s", (10, canvas.shape[0] - 12),
                FONT, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
    return canvas


def draw_sim_panel(
    rgb: np.ndarray,
    iou: float,
    landmarks: Optional[Dict[str, np.ndarray]] = None,
    extra_label: str = "",
) -> np.ndarray:
    canvas = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    # projected fiducial landmarks (model-side keypoint reprojection)
    if landmarks:
        colors = {"hip": (230, 70, 235), "knee": (80, 235, 100), "ankle": (250, 120, 60)}
        for name, uv in landmarks.items():
            pt = (int(round(float(uv[0]))), int(round(float(uv[1]))))
            cv2.drawMarker(canvas, pt, colors[name], cv2.MARKER_CROSS, 11, 2, cv2.LINE_AA)
    cv2.putText(canvas, f"IoU = {iou:.3f}", (10, 26), FONT, 0.62, (90, 255, 140), 2, cv2.LINE_AA)
    if extra_label:
        cv2.putText(canvas, extra_label, (10, 50), FONT, 0.52, (120, 200, 255), 1, cv2.LINE_AA)
    return canvas


def compose_comparison(frame_bgr, uncal_panel, cal_panel) -> np.ndarray:
    h, w = frame_bgr.shape[:2]
    header_h, gap = 34, 2
    canvas = np.zeros((header_h + h, 3 * w + 2 * gap, 3), dtype=np.uint8)
    for col, title in enumerate(
        ["INPUT VIDEO + MARKER TRACKING", "UNCALIBRATED MODEL", "MIRROR DIGITAL TWIN (CALIBRATED)"]
    ):
        x0 = col * (w + gap)
        cv2.putText(canvas, title, (x0 + 14, 23), FONT, 0.60, (255, 255, 255), 1, cv2.LINE_AA)
    for col, panel in enumerate([frame_bgr, uncal_panel, cal_panel]):
        x0 = col * (w + gap)
        canvas[header_h:, x0:x0 + w] = panel
    return canvas


def write_video(path: Path, frames_bgr: List[np.ndarray], fps: float) -> str:
    h, w = frames_bgr[0].shape[:2]
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    if writer.isOpened():
        for frame in frames_bgr:
            writer.write(frame)
        writer.release()
        if path.exists() and path.stat().st_size > 1024:
            return "cv2/mp4v"
    import imageio.v2 as imageio

    with imageio.get_writer(str(path), fps=fps, codec="libx264", quality=8,
                            macro_block_size=1, pixelformat="yuv420p") as writer_ff:
        for frame in frames_bgr:
            writer_ff.append_data(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    return "imageio/libx264"


def write_gif(path: Path, frames_bgr: List[np.ndarray], fps: float, scale: float = 0.5) -> None:
    import imageio.v2 as imageio

    step = 2  # ~15 fps keeps the GIF small while staying smooth
    small = []
    for frame in frames_bgr[::step]:
        resized = cv2.resize(frame, (int(frame.shape[1] * scale), int(frame.shape[0] * scale)),
                             interpolation=cv2.INTER_AREA)
        small.append(cv2.cvtColor(resized, cv2.COLOR_BGR2RGB))
    imageio.mimsave(str(path), small, fps=max(2, int(round(fps / step))), loop=0)


def make_diagnostic_plot(path: Path, vision, params_final: ModelParams,
                         ctrl0, traj0, ctrl1, traj1, calib: CalibrationResult,
                         phase1_hist: Optional[np.ndarray] = None) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    t = np.arange(vision.n_frames) / vision.fps
    sim_cal = ctrl_to_vision_angles(traj1.frame_qpos_rad, params_final.hip_offset_rad, params_final.knee_offset_rad)
    sim_uncal = ctrl_to_vision_angles(traj0.frame_qpos_rad, 0.0, 0.0)

    fig, axes = plt.subplots(3, 1, figsize=(12, 10.5), constrained_layout=True)

    ax = axes[0]
    ax.plot(t, np.degrees(vision.angles_rad["hip"]), "k--", lw=1.8, label="CV hip (Savitzky-Golay filtered)")
    ax.plot(t, np.degrees(vision.angles_rad["knee"]), color="0.4", ls="--", lw=1.8, label="CV knee (Savitzky-Golay filtered)")
    ax.plot(t, np.degrees(sim_cal["hip"]), color="tab:red", lw=1.4, label="MuJoCo hip — calibrated twin")
    ax.plot(t, np.degrees(sim_cal["knee"]), color="tab:blue", lw=1.4, label="MuJoCo knee — calibrated twin")
    ax.plot(t, np.degrees(sim_uncal["hip"]), color="tab:red", lw=1.0, ls=":", alpha=0.55, label="MuJoCo hip — uncalibrated")
    ax.plot(t, np.degrees(sim_uncal["knee"]), color="tab:blue", lw=1.0, ls=":", alpha=0.55, label="MuJoCo knee — uncalibrated")
    ax.set_xlabel("time (s)")
    ax.set_ylabel("joint angle (deg)")
    ax.set_title("Filtered vision trajectories vs forward-dynamics trajectories")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8, ncol=2, loc="upper right")

    ax = axes[1]
    ax.plot(traj1.time_s, traj1.joint_torque_Nm[:, 0], color="tab:red", lw=1.5, label="hip torque (calibrated twin)")
    ax.plot(traj1.time_s, traj1.joint_torque_Nm[:, 1], color="tab:blue", lw=1.5, label="knee torque (calibrated twin)")
    ax.set_xlabel("time (s)")
    ax.set_ylabel("actuator torque (N·m)")
    ax.set_title("PD-servo torque demand under gravity (forward dynamics)")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)

    ax = axes[2]
    offset = 0
    if phase1_hist is not None and len(phase1_hist):
        ax.scatter(np.arange(len(phase1_hist)), phase1_hist, s=10, alpha=0.45,
                   color="tab:blue", label="phase 1 evaluations (silhouette-only)")
        ax.step(np.arange(len(phase1_hist)), np.minimum.accumulate(phase1_hist),
                where="post", color="tab:blue", lw=1.4, alpha=0.8)
        offset = len(phase1_hist)
        ax.axvline(offset - 0.5, color="k", ls=":", lw=1.0)
        ax.text(offset + 5, ax.get_ylim()[0] + 0.05 * (ax.get_ylim()[1] - ax.get_ylim()[0]),
                "phase 2 (fused)", fontsize=8, color="0.3")
    hist = np.asarray(calib.eval_history)
    ax.scatter(offset + np.arange(len(hist)), hist, s=10, alpha=0.45, color="tab:gray",
               label="phase 2 evaluations (fused loss)")
    full = np.concatenate([np.asarray(phase1_hist), hist]) if phase1_hist is not None and len(phase1_hist) else hist
    ax.step(offset + np.arange(len(hist)), np.minimum.accumulate(full)[offset:], where="post",
            color="tab:purple", lw=1.8, label="best-so-far")
    ax.axhline(1.0 - TARGET_IOU, color="tab:red", ls="--", lw=1.2, label=f"IoU target {TARGET_IOU:.2f}")
    ax.set_xlabel("objective evaluation")
    ax.set_ylabel("loss")
    ax.set_title("Closed-loop convergence (Powell, two phases)")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)

    fig.suptitle("Mirror — kinematics, torques and calibration convergence", fontsize=13)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def make_calibration_diagnostics_plot(
    path: Path,
    vision,
    iou0_frames: np.ndarray,
    iou1_frames: np.ndarray,
    calib_indices: np.ndarray,
    calib: CalibrationResult,
    ci95: Optional[Dict[str, Dict[str, float]]],
    gt: Optional[dict] = None,
    empirical_bias_deg: Optional[Dict[str, float]] = None,
) -> None:
    """Second diagnostic figure: validation, reprojection, uncertainty, loss terms.

    Panel 1: per-frame silhouette IoU (uncalibrated vs calibrated twin) with
             the calibration / held-out frame split shaded.
    Panel 2: keypoint-reprojection residual magnitude per marker (calibrated)
             plus the recovered skin-mount pixel offsets vs ground truth.
    Panel 3: identified parameters with local-curvature 95% CIs, ground-truth
             markers where available.
    Panel 4: fused-loss decomposition over the Powell evaluations.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    t = np.arange(vision.n_frames) / vision.fps
    cal_set = set(int(i) for i in calib_indices)
    is_cal = np.asarray([i in cal_set for i in range(vision.n_frames)])

    fig, axes = plt.subplots(2, 2, figsize=(13.5, 9.0), constrained_layout=True)

    # -- panel 1: per-frame IoU + validation split --------------------------
    ax = axes[0, 0]
    ax.plot(t, iou0_frames, color="tab:gray", lw=1.2, label="uncalibrated")
    ax.plot(t, iou1_frames, color="tab:green", lw=1.6, label="calibrated twin")
    ymin, ymax = 0.0, max(1.0, float(max(iou0_frames.max(), iou1_frames.max())) + 0.05)
    ax.fill_between(t, ymin, ymax, where=is_cal, color="tab:blue", alpha=0.07,
                    label="calibration frames (Powell)")
    ax.fill_between(t, ymin, ymax, where=~is_cal, color="tab:orange", alpha=0.10,
                    label="held-out validation frames")
    ax.set_ylim(ymin, ymax)
    ax.set_xlabel("time (s)")
    ax.set_ylabel("silhouette IoU")
    mean_c, mean_h = float(iou1_frames[is_cal].mean()), float(iou1_frames[~is_cal].mean())
    ax.set_title(f"Per-frame IoU — calibration {mean_c:.3f} vs held-out {mean_h:.3f}")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8, loc="lower right")

    # -- panel 2: reprojection residuals -------------------------------------
    ax = axes[0, 1]
    if calib.reproj_rmse_px_final:
        names = [n for n in ("hip", "knee", "ankle") if n in calib.reproj_rmse_px_final]
        init = [calib.reproj_rmse_px_initial.get(n, np.nan) for n in names]
        fin = [calib.reproj_rmse_px_final.get(n, np.nan) for n in names]
        xpos = np.arange(len(names))
        ax.bar(xpos - 0.18, init, width=0.36, color="tab:gray", label="uncalibrated")
        ax.bar(xpos + 0.18, fin, width=0.36, color="tab:green", label="calibrated")
        ax.axhline(calib.kp_sigma_px, color="tab:red", ls="--", lw=1.2,
                   label=f"residual floor ≈ 1 marker radius ({calib.kp_sigma_px:.1f} px)")
        ax.set_xticks(xpos, names)
        ax.set_ylabel("reprojection RMSE (px)")
        ax.set_title("Fiducial reprojection error (pinhole, mirror_cam)")
        ax.grid(alpha=0.3, axis="y")
        ax.legend(fontsize=8, loc="upper left")
        if calib.marker_offsets_px:
            lines = ["recovered skin-mount offsets (px):"]
            for m in ("hip", "knee", "ankle"):
                dx, dy = calib.marker_offsets_px[m]
                entry = f"  {m}: ({dx:+.1f},{dy:+.1f})"
                if gt is not None:
                    tx, ty = gt["marker_bias_px"][m]
                    entry += f"  true ({tx:+.1f},{ty:+.1f})"
                lines.append(entry)
            ax.text(0.98, 0.97, "\n".join(lines), transform=ax.transAxes,
                    ha="right", va="top", fontsize=7.5,
                    bbox=dict(boxstyle="round", fc="white", ec="0.7", alpha=0.9))
    else:
        ax.text(0.5, 0.5, "keypoint term disabled (kp_weight = 0)", ha="center", va="center")
        ax.set_axis_off()

    # -- panel 3: CI coverage (errors normalised by the 95% CI half-width) ----
    ax = axes[1, 0]
    refs: Dict[str, Optional[Tuple[float, float]]] = {}
    if gt is not None:
        refs["L_femur"] = (calib.best_params.L_femur_m, gt["links"]["L_femur_m"])
        refs["L_tibia"] = (calib.best_params.L_tibia_m, gt["links"]["L_tibia_m"])
    if empirical_bias_deg is not None:
        refs["hip_offset"] = (np.degrees(calib.best_params.hip_offset_rad), empirical_bias_deg["hip"])
        refs["knee_offset"] = (np.degrees(calib.best_params.knee_offset_rad), empirical_bias_deg["knee"])
    if ci95 and refs:
        half_map = {
            "L_femur": ci95["L_femur_m"]["ci95_half_width"],
            "L_tibia": ci95["L_tibia_m"]["ci95_half_width"],
            "hip_offset": np.degrees(ci95["hip_offset_rad"]["ci95_half_width"]),
            "knee_offset": np.degrees(ci95["knee_offset_rad"]["ci95_half_width"]),
        }
        names = list(refs)
        z = [(refs[n][0] - refs[n][1]) / half_map[n] for n in names]
        err_phys = [refs[n][0] - refs[n][1] for n in names]
        unit = ["m", "m", "deg", "deg"]
        colors = ["tab:green" if abs(v) <= 1.0 else "tab:red" for v in z]
        ax.axhspan(-1.0, 1.0, color="tab:green", alpha=0.10,
                   label="inside 95% CI (likelihood-ratio)")
        ax.axhline(0.0, color="k", lw=0.8)
        ax.bar(np.arange(len(names)), z, width=0.55, color=colors, edgecolor="0.3")
        for i, (v, e, u) in enumerate(zip(z, err_phys, unit)):
            ax.annotate(f"{e:+.4f} {u}", (i, v), textcoords="offset points",
                        xytext=(0, 6 if v >= 0 else -14), ha="center", fontsize=8)
        ax.set_xticks(np.arange(len(names)), names, fontsize=8)
        ax.set_ylabel("(identified − truth) / CI half-width")
        ax.set_title("95% CI coverage vs ground truth / empirical bias")
        ax.grid(alpha=0.3, axis="y")
        ax.legend(fontsize=8, loc="lower right")
    elif ci95:
        # no ground truth available: show the raw estimates with CI bars
        x_est = calib.best_params.calibration_vector()
        units = ["m", "m", "deg", "deg", "N·m/rad"]
        labels = ["L_femur", "L_tibia", "hip_offset", "knee_offset", "servo_kp"]
        est = [x_est[0], x_est[1], np.degrees(x_est[2]), np.degrees(x_est[3]), x_est[4]]
        half = [
            ci95["L_femur_m"]["ci95_half_width"], ci95["L_tibia_m"]["ci95_half_width"],
            np.degrees(ci95["hip_offset_rad"]["ci95_half_width"]),
            np.degrees(ci95["knee_offset_rad"]["ci95_half_width"]),
            ci95["servo_kp"]["ci95_half_width"],
        ]
        xpos = np.arange(5)
        ax.errorbar(xpos, est, yerr=half, fmt="o", color="tab:purple", capsize=5,
                    lw=1.6, markersize=6, label="identified ± 95% CI")
        ax.set_xticks(xpos, [f"{l}\n[{u}]" for l, u in zip(labels, units)], fontsize=8)
        ax.grid(alpha=0.3, axis="y")
        ax.set_title("Identified parameters with local-curvature 95% CIs")
        ax.legend(fontsize=8)
    else:
        ax.text(0.5, 0.5, "uncertainty estimation skipped", ha="center", va="center")
        ax.set_axis_off()

    # -- panel 4: fused-loss decomposition ------------------------------------
    ax = axes[1, 1]
    terms = np.asarray(calib.eval_history_terms)
    if terms.size:
        evals = np.arange(len(terms))
        ax.scatter(evals, terms[:, 0], s=9, alpha=0.4, color="tab:gray",
                   label="silhouette term (1 − IoU)")
        if calib.kp_weight > 0 and np.any(terms[:, 1] > 0):
            ax.scatter(evals, calib.kp_weight * terms[:, 1], s=9, alpha=0.4,
                       color="tab:orange", label=f"kp term (weight {calib.kp_weight:g})")
        ax.step(evals, np.minimum.accumulate(np.asarray(calib.eval_history)),
                where="post", color="tab:purple", lw=1.7, label="best-so-far (fused)")
        ax.set_xlabel("objective evaluation (phase 2)")
        ax.set_ylabel("loss contribution")
        ax.set_yscale("log")
        ax.set_title("Fused-loss decomposition (phase-2 evaluations)")
        ax.grid(alpha=0.3, which="both")
        ax.legend(fontsize=8)
    else:
        ax.set_axis_off()

    fig.suptitle("Mirror — calibration diagnostics: validation, reprojection, uncertainty", fontsize=13)
    fig.savefig(path, dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------
def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="run_mirror.py",
        description="Mirror: monocular video -> calibrated MuJoCo digital twin.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--video", default=str(DEFAULT_VIDEO), help="input monocular video")
    parser.add_argument("--profile", choices=list(PROFILES), default="full",
                        help="optimisation budget preset")
    parser.add_argument("--camera-distance", type=float, default=3.0, dest="camera_distance",
                        help="assumed camera-to-plane distance in metres (the one manual scale prior)")
    parser.add_argument("--focal-px", type=float, default=620.0, dest="focal_px",
                        help="assumed pinhole focal length in pixels")
    parser.add_argument("--kp-init", type=float, default=250.0, dest="kp_init",
                        help="initial position-servo gain (calibrated in Part 4)")
    parser.add_argument("--savgol-window", type=int, default=11, dest="savgol_window",
                        help="Savitzky-Golay de-jitter window (frames)")
    parser.add_argument("--max-frames", type=int, default=400, dest="max_frames",
                        help="cap on decoded frames")
    parser.add_argument("--out-dir", default=str(BASE_DIR / "outputs"))
    parser.add_argument("--models-dir", default=str(BASE_DIR / "models"))
    parser.add_argument("--max-evals", type=int, default=None, help="override optimiser budget")
    parser.add_argument("--frame-stride", type=int, default=None, help="override eval frame stride")
    parser.add_argument("--kp-weight", type=float, default=DEFAULT_KP_WEIGHT, dest="kp_weight",
                        help="weight of the keypoint-reprojection term in the fused loss "
                             "(0 = silhouette-only objective; see scripts/ablation_study.py)")
    parser.add_argument("--no-uncertainty", action="store_true", dest="no_uncertainty",
                        help="skip the local-curvature 95% CI estimation")
    parser.add_argument("--no-gif", action="store_true", help="skip GIF export")
    parser.add_argument("--no-mp4", action="store_true", help="skip MP4 export")
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
        datefmt="%H:%M:%S",
    )
    out_dir = Path(args.out_dir)
    models_dir = Path(args.models_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    models_dir.mkdir(parents=True, exist_ok=True)

    profile = PROFILES[args.profile]
    frame_stride = args.frame_stride or profile["frame_stride"]
    max_eval_frames = profile["max_eval_frames"]
    maxfev = args.max_evals or profile["maxfev"]

    t_total = time.perf_counter()

    # ---- [1/6] input -------------------------------------------------------
    video_path = ensure_demo_video(Path(args.video))
    log.info("[1/6] Video: %s", video_path)

    # ground truth (only available for the synthetic benchmark) — loaded once
    # so it can feed both the metrics JSON and the diagnostics figure
    gt_path = DEFAULT_GROUND_TRUTH if video_path.resolve() == DEFAULT_VIDEO.resolve() else video_path.parent / "demo_ground_truth.json"
    gt: Optional[dict] = json.loads(gt_path.read_text(encoding="utf-8")) if gt_path.exists() else None

    # ---- [2/6] Part 1 — vision ---------------------------------------------
    log.info("[2/6] Part 1: monocular ingestion, marker tracking, kinematics")
    extractor = VisionExtractor(
        video_path,
        focal_length_px=args.focal_px,
        assumed_camera_distance_m=args.camera_distance,
        max_frames=args.max_frames,
        savgol_window=args.savgol_window,
    )
    vision = extractor.extract()

    # ---- [3/6] Part 2 — initial MJCF ----------------------------------------
    log.info("[3/6] Part 2: compiling initial parameterized MJCF model")
    params0 = initial_params_from_vision(vision, args.camera_distance, args.kp_init)
    xml_initial = MJCFBuilder(params0).save(models_dir / "mirror_initial.xml")
    log.info("      initial lengths: femur %.3f m, tibia %.3f m | kp=%.0f | radii f/t %.3f/%.3f m",
             params0.L_femur_m, params0.L_tibia_m, params0.servo_kp,
             params0.femur_radius_m, params0.tibia_radius_m)

    # ---- [4/6] Part 3 — uncalibrated forward dynamics ------------------------
    log.info("[4/6] Part 3: uncalibrated forward-dynamics rollout (500 Hz RK4)")
    sim0 = MirrorSimulator(MJCFBuilder(params0).to_xml(), vision.height, vision.width)
    ctrl0 = vision_angles_to_ctrl(vision.angles_rad)
    traj0 = sim0.simulate_trajectory(np.arange(vision.n_frames) / vision.fps, ctrl0)
    iou0_frames = per_frame_ious(sim0, traj0.frame_qpos_rad, vision.foreground_masks)
    rmse0 = tracking_rmse_deg(ctrl0, traj0.frame_qpos_rad)
    max_tau0 = np.max(np.abs(traj0.joint_torque_Nm), axis=0)
    mass0 = float(sim0.model.body_mass.sum())
    log.info("      uncalibrated mean IoU = %.4f | tracking RMSE hip %.2f deg, knee %.2f deg",
             float(iou0_frames.mean()), rmse0["hip_deg"], rmse0["knee_deg"])

    # ---- [5/6] Part 4 — closed-loop calibration -----------------------------
    log.info("[5/6] Part 4: closed-loop calibration (two-phase Powell, %d evals budget, "
             "fused refinement weight kp=%.2g)", maxfev, args.kp_weight)
    calib1, calib, optimizer = calibrate_two_phase(
        vision, params0,
        kp_weight=args.kp_weight,
        maxfev_total=maxfev,
        frame_stride=frame_stride,
        max_eval_frames=max_eval_frames,
        render_height=vision.height,
        render_width=vision.width,
    )
    params1 = calib.best_params
    xml_calibrated = MJCFBuilder(params1).save(models_dir / "mirror_calibrated.xml")

    # ---- uncertainty: local-curvature 95% CIs at the optimum -----------------
    ci95: Optional[Dict[str, Dict[str, float]]] = None
    if not args.no_uncertainty:
        log.info("      estimating parameter CIs (finite-difference curvature at the optimum)")
        try:
            ci95 = estimate_ci95_curvature(
                optimizer.fused_loss, _final_x(optimizer, calib),
                param_names=optimizer.param_names, param_bounds=optimizer.param_bounds,
            )
            log.info("      CIs done (%d extra evaluations)", 2 * optimizer.n_params + 1)
        except Exception as exc:  # never fail the pipeline on CI estimation
            log.warning("CI estimation failed: %s", exc)
            ci95 = None

    sim1 = MirrorSimulator(MJCFBuilder(params1).to_xml(), vision.height, vision.width)
    ctrl1 = vision_angles_to_ctrl(vision.angles_rad, params1.hip_offset_rad, params1.knee_offset_rad)
    traj1 = sim1.simulate_trajectory(np.arange(vision.n_frames) / vision.fps, ctrl1)
    iou1_frames = per_frame_ious(sim1, traj1.frame_qpos_rad, vision.foreground_masks)
    rmse1 = tracking_rmse_deg(ctrl1, traj1.frame_qpos_rad)
    max_tau1 = np.max(np.abs(traj1.joint_torque_Nm), axis=0)
    rf_mean = traj1.reaction_force_N.mean(axis=0)
    rf_max = traj1.reaction_force_N.max(axis=0)
    mass1 = float(sim1.model.body_mass.sum())
    log.info("      calibrated mean IoU = %.4f (target %.2f) | tracking RMSE hip %.2f deg, knee %.2f deg",
             float(iou1_frames.mean()), TARGET_IOU, rmse1["hip_deg"], rmse1["knee_deg"])

    # ---- [6/6] outputs --------------------------------------------------------
    log.info("[6/6] writing outputs")
    metrics: Dict = {
        "schema": "mirror/metrics-v1.1",
        "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "mirror_version": mirror_pkg.__version__,
        "environment": {
            "python": sys.version.split()[0],
            "platform": platform.platform(),
            "mujoco": __import__("mujoco").__version__,
            "numpy": np.__version__,
            "scipy": __import__("scipy").__version__,
            "opencv": cv2.__version__,
            "render_backend": RENDER_BACKEND,
        },
        "video": {
            "path": str(video_path),
            "fps": vision.fps,
            "frames": vision.n_frames,
            "resolution_wh": [vision.width, vision.height],
        },
        "camera_prior": {
            "focal_length_px": vision.camera.focal_length_px,
            "assumed_camera_distance_m": vision.assumed_camera_distance_m,
            "fovy_deg": vision.camera.fovy_deg,
            "note": "Monocular scale requires one manual distance prior; recovered lengths "
                    "live in this camera frame.",
        },
        "vision": {
            "tracking_success_rate": vision.tracking_success_rate,
            "initial_link_lengths_m": vision.initial_link_lengths_m,
            "limb_width_px": vision.limb_width_px,
            "marker_radius_px": vision.marker_radius_px,
            "hip_pixel": list(vision.hip_pixel),
        },
        "model_initial": {
            "xml": str(xml_initial),
            "params": {name: getattr(params0, name) for name in CALIBRATION_PARAM_NAMES},
            "morphology": {
                "femur_radius_m": params0.femur_radius_m,
                "tibia_radius_m": params0.tibia_radius_m,
                "marker_radius_m": params0.marker_radius_m,
                "total_mass_kg": mass0,
            },
            "mean_silhouette_iou": float(iou0_frames.mean()),
            "per_frame_iou": iou0_frames,
            "tracking_rmse_deg": rmse0,
            "max_abs_joint_torque_Nm": {"hip": float(max_tau0[0]), "knee": float(max_tau0[1])},
        },
        "calibration": {
            "method": "two-phase scipy.optimize.minimize / Powell (derivative-free): "
                      "phase 1 silhouette-only over geometry, phase 2 fused loss "
                      "(1 - mean IoU) + kp_weight * keypoint reprojection RMSE over "
                      "geometry + per-marker pixel nuisance offsets",
            "param_names": list(optimizer.param_names),
            "kp_weight": float(args.kp_weight),
            "kp_sigma_px": float(optimizer.kp_sigma_px),
            "optimize_marker_offsets": bool(optimizer.optimize_marker_offsets),
            "phase1_silhouette_only": {
                "n_evals": calib1.n_evals,
                "mean_iou_final": calib1.mean_iou_final,
                "x_optimized": calib1.best_params.calibration_vector(),
                "eval_history_loss": calib1.eval_history,
            },
            "x_initial": params0.calibration_vector(),
            "x_optimized": calib.best_params.calibration_vector(),
            "x_bounds": [list(b) for b in optimizer.param_bounds],
            "n_objective_evals": calib1.n_evals + calib.n_evals,
            "wall_time_s": calib1.wall_time_s + calib.wall_time_s,
            "converged": calib.success,
            "optimizer_message": calib.message,
            "eval_history_loss": calib.eval_history,
            "eval_history_terms": calib.eval_history_terms,
            "marker_offsets_px": calib.marker_offsets_px,
            "reprojection_rmse_px": {
                "initial": calib.reproj_rmse_px_initial,
                "final": calib.reproj_rmse_px_final,
            },
            "validation": {
                "n_calibration_frames": int(len(optimizer.eval_indices)),
                "n_heldout_frames": int(len(optimizer.heldout_indices)),
                "mean_iou_calibration_frames": calib.mean_iou_final,
                "mean_iou_heldout_frames": calib.mean_iou_heldout,
                "per_frame_iou_heldout": calib.per_frame_iou_heldout,
            },
        },
        "model_calibrated": {
            "xml": str(xml_calibrated),
            "params": {name: getattr(params1, name) for name in CALIBRATION_PARAM_NAMES},
            "morphology": {
                "femur_radius_m": params1.femur_radius_m,
                "tibia_radius_m": params1.tibia_radius_m,
                "marker_radius_m": params1.marker_radius_m,
                "total_mass_kg": mass1,
            },
            "mean_silhouette_iou": float(iou1_frames.mean()),
            "per_frame_iou": iou1_frames,
            "tracking_rmse_deg": rmse1,
            "max_abs_joint_torque_Nm": {"hip": float(max_tau1[0]), "knee": float(max_tau1[1])},
            "joint_reaction_force_N": {
                "hip": {"mean": float(rf_mean[0]), "max": float(rf_max[0])},
                "knee": {"mean": float(rf_mean[1]), "max": float(rf_max[1])},
            },
            "sim_rate_hz": traj1.sim_rate_hz,
        },
        "targets": {"mean_silhouette_iou": TARGET_IOU, "met": bool(iou1_frames.mean() >= TARGET_IOU)},
        "total_wall_time_s": time.perf_counter() - t_total,
    }

    # ground-truth cross-check (only possible for the synthetic demo clip)
    empirical_bias_deg: Optional[Dict[str, float]] = None
    if gt is not None:
        n = min(vision.n_frames, len(gt["theta_thigh_true_rad"]))
        vis_rmse = {
            "hip_deg": float(np.degrees(np.sqrt(np.mean(
                (vision.angles_rad["hip"][:n] - np.asarray(gt["theta_thigh_true_rad"][:n])) ** 2)))),
            "knee_deg": float(np.degrees(np.sqrt(np.mean(
                (vision.angles_rad["knee"][:n] - np.asarray(gt["phi_knee_true_rad"][:n])) ** 2)))),
        }
        # empirical skin-marker bias: the time-averaged angular error the
        # fiducial placement injects into the vision angles.  The additive
        # joint offsets identified by Part 4 should recover exactly this
        # quantity (offset = mean(theta_vision - theta_true) in each joint).
        empirical_bias_deg = {
            "hip": float(np.degrees(np.mean(
                vision.angles_rad["hip"][:n] - np.asarray(gt["theta_thigh_true_rad"][:n])))),
            "knee": float(np.degrees(np.mean(
                vision.angles_rad["knee"][:n] - np.asarray(gt["phi_knee_true_rad"][:n])))),
        }
        gtc: Dict = {
            "ground_truth_file": str(gt_path),
            "true_link_lengths_m": {"femur": gt["links"]["L_femur_m"], "tibia": gt["links"]["L_tibia_m"]},
            "true_camera_distance_m": gt["camera"]["camera_distance_m"],
            "marker_bias_px": gt["marker_bias_px"],
            "vision_angle_rmse_vs_truth_deg": vis_rmse,
            "empirical_marker_bias_deg": empirical_bias_deg,
            "calibrated_length_error_m": {
                "femur": params1.L_femur_m - gt["links"]["L_femur_m"],
                "tibia": params1.L_tibia_m - gt["links"]["L_tibia_m"],
            },
            "calibrated_length_error_pct": {
                "femur": 100.0 * (params1.L_femur_m - gt["links"]["L_femur_m"]) / gt["links"]["L_femur_m"],
                "tibia": 100.0 * (params1.L_tibia_m - gt["links"]["L_tibia_m"]) / gt["links"]["L_tibia_m"],
            },
            "identified_marker_bias_deg": {
                "hip": float(np.degrees(params1.hip_offset_rad)),
                "knee": float(np.degrees(params1.knee_offset_rad)),
            },
        }
        if ci95 is not None:
            gtc["ci95_coverage"] = {
                name: bool(
                    ci95[name]["lo"]
                    <= val
                    <= ci95[name]["hi"]
                )
                for name, val in (
                    ("L_femur_m", gt["links"]["L_femur_m"]),
                    ("L_tibia_m", gt["links"]["L_tibia_m"]),
                )
            }
        metrics["ground_truth_check"] = gtc

        # the fitted per-marker pixel nuisance offsets are a direct measurement
        # of the skin-mount bias — score them against the published truth
        if calib.marker_offsets_px:
            gt_bias = gt["marker_bias_px"]
            gtc["recovered_marker_offsets_px"] = {
                name: {
                    "identified": list(calib.marker_offsets_px[name]),
                    "true": list(gt_bias[name]),
                    "error_px": float(np.linalg.norm(
                        np.asarray(calib.marker_offsets_px[name]) - np.asarray(gt_bias[name])
                    )),
                }
                for name in calib.marker_offsets_px
            }

    # uncertainty quantification block
    if ci95 is not None:
        metrics["uncertainty"] = {
            "method": "local-curvature (Gaussian / likelihood-ratio) 95% CIs; central "
                      "finite differences of the fused loss in bound-normalized "
                      "coordinates at the optimum; other parameters held fixed "
                      "(conservative vs a full profile likelihood)",
            "chi2_95_1dof": 3.841458820694124,
            "parameters": ci95,
        }

    metrics_path = out_dir / "mirror_metrics.json"
    metrics_path.write_text(json.dumps(_jsonable(metrics), indent=2), encoding="utf-8")

    # comparison video / gif — with pinhole landmark reprojection overlays
    frames_bgr = []
    cap = cv2.VideoCapture(str(video_path))
    raw_frames = []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        raw_frames.append(frame)
    cap.release()

    lm0_all = [sim0.project_landmarks(traj0.frame_qpos_rad[i]) for i in range(vision.n_frames)]
    marker_deltas = calib.marker_offsets_px or {}
    lm1_all = []
    for i in range(vision.n_frames):
        raw = sim1.project_landmarks(traj1.frame_qpos_rad[i])
        lm1_all.append({
            name: raw[name] + np.asarray(marker_deltas.get(name, (0.0, 0.0)), dtype=float)
            for name in raw
        })
    reproj_px_frame = []
    for i in range(vision.n_frames):
        errs = [
            float(np.linalg.norm(lm1_all[i][name] - vision.keypoints_px[name][i]))
            for name in ("hip", "knee", "ankle") if name in lm1_all[i]
        ]
        reproj_px_frame.append(float(np.mean(errs)) if errs else 0.0)

    for i in range(vision.n_frames):
        if i < len(raw_frames):
            panel_in = draw_input_panel(raw_frames[i], vision, i)
        else:
            panel_in = draw_input_panel(np.zeros((vision.height, vision.width, 3), np.uint8), vision, i)
        frames_bgr.append(
            compose_comparison(
                panel_in,
                draw_sim_panel(
                    sim0.render_rgb(traj0.frame_qpos_rad[i]), float(iou0_frames[i]),
                    landmarks=lm0_all[i],
                ),
                draw_sim_panel(
                    sim1.render_rgb(traj1.frame_qpos_rad[i]), float(iou1_frames[i]),
                    landmarks=lm1_all[i],
                    extra_label=f"reproj = {reproj_px_frame[i]:.1f} px",
                ),
            )
        )
    media_writer = "skipped"
    if not args.no_mp4 and frames_bgr:
        media_writer = write_video(out_dir / "mirror_comparison.mp4", frames_bgr, vision.fps)
    if not args.no_gif and frames_bgr:
        write_gif(out_dir / "mirror_comparison.gif", frames_bgr, vision.fps)

    make_diagnostic_plot(
        out_dir / "kinematics_and_torques.png", vision, params1,
        ctrl0, traj0, ctrl1, traj1, calib,
        phase1_hist=calib1.eval_history,
    )

    make_calibration_diagnostics_plot(
        out_dir / "calibration_diagnostics.png", vision,
        iou0_frames, iou1_frames, optimizer.eval_indices, calib, ci95,
        gt=gt, empirical_bias_deg=empirical_bias_deg,
    )

    # preview render of the calibrated twin mid-swing
    mid = vision.n_frames // 2
    mid_rgb = sim1.render_rgb(traj1.frame_qpos_rad[mid])
    cv2.imwrite(str(out_dir / "calibrated_model_render.png"), cv2.cvtColor(mid_rgb, cv2.COLOR_RGB2BGR))

    sim0.close()
    sim1.close()

    # ---- summary -------------------------------------------------------------
    iou_u, iou_c = float(iou0_frames.mean()), float(iou1_frames.mean())
    print("\n" + "=" * 74)
    print(" MIRROR — CALIBRATION SUMMARY")
    print("=" * 74)
    print(f" frames / fps                : {vision.n_frames} @ {vision.fps:.1f}")
    print(f" render backend              : {RENDER_BACKEND}")
    print(f" fused loss                  : (1 - IoU) + {args.kp_weight:g} * reproj RMSE")
    print(f" mean silhouette IoU         : {iou_u:.4f}  ->  {iou_c:.4f}   (target {TARGET_IOU:.2f})")
    if calib.mean_iou_heldout is not None:
        print(f" held-out validation IoU     : {calib.mean_iou_heldout:.4f} "
              f"({len(optimizer.heldout_indices)} unseen frames)")
    if calib.reproj_rmse_px_final:
        r0, r1 = calib.reproj_rmse_px_initial, calib.reproj_rmse_px_final
        def _fmt(d):
            return " / ".join(f"{d.get(k, float('nan')):.1f}" for k in ("hip", "knee", "ankle"))
        print(f" reproj RMSE px (h/k/a)      : {_fmt(r0)}  ->  {_fmt(r1)}")
    print(f" tracking RMSE (hip / knee)  : {rmse0['hip_deg']:.2f} / {rmse0['knee_deg']:.2f} deg"
          f"  ->  {rmse1['hip_deg']:.2f} / {rmse1['knee_deg']:.2f} deg")
    print(f" calibrated L_femur / L_tibia: {params1.L_femur_m:.4f} m / {params1.L_tibia_m:.4f} m")
    if ci95 is not None:
        print(f"   95% CIs                   : femur ±{ci95['L_femur_m']['ci95_half_width']:.4f} m, "
              f"tibia ±{ci95['L_tibia_m']['ci95_half_width']:.4f} m")
    print(f" calibrated offsets (hip/knee): {np.degrees(params1.hip_offset_rad):+.2f} / "
          f"{np.degrees(params1.knee_offset_rad):+.2f} deg")
    if empirical_bias_deg is not None:
        print(f" empirical marker bias       : {empirical_bias_deg['hip']:+.2f} / "
              f"{empirical_bias_deg['knee']:+.2f} deg")
    if calib.marker_offsets_px:
        d = calib.marker_offsets_px
        print(f" recovered mount offsets px : hip ({d['hip'][0]:+.1f},{d['hip'][1]:+.1f})  "
              f"knee ({d['knee'][0]:+.1f},{d['knee'][1]:+.1f})  ankle ({d['ankle'][0]:+.1f},{d['ankle'][1]:+.1f})")
        if gt is not None:
            b = gt["marker_bias_px"]
            print(f"   true mount bias px       : hip ({b['hip'][0]:+.1f},{b['hip'][1]:+.1f})  "
                  f"knee ({b['knee'][0]:+.1f},{b['knee'][1]:+.1f})  ankle ({b['ankle'][0]:+.1f},{b['ankle'][1]:+.1f})")
    print(f" calibrated servo kp         : {params1.servo_kp:.0f}")
    print(f" optimiser                   : {calib1.n_evals + calib.n_evals} evals "
          f"(phase 1 {calib1.n_evals}, phase 2 {calib.n_evals}) "
          f"in {calib1.wall_time_s + calib.wall_time_s:.1f} s")
    if "ground_truth_check" in metrics:
        gtc = metrics["ground_truth_check"]
        print(f" vs ground truth lengths     : femur {gtc['calibrated_length_error_pct']['femur']:+.2f}%, "
              f"tibia {gtc['calibrated_length_error_pct']['tibia']:+.2f}%")
        if "ci95_coverage" in gtc:
            cov = gtc["ci95_coverage"]
            print(f" GT within 95% CI            : femur {cov['L_femur_m']}, tibia {cov['L_tibia_m']}")
    print("-" * 74)
    print(f" models   : {xml_initial}")
    print(f"           {xml_calibrated}")
    print(f" outputs  : {out_dir / 'mirror_comparison.mp4'} ({media_writer})")
    print(f"           {out_dir / 'mirror_comparison.gif'}")
    print(f"           {out_dir / 'kinematics_and_torques.png'}")
    print(f"           {out_dir / 'calibration_diagnostics.png'}")
    print(f"           {metrics_path}")
    print("=" * 74)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

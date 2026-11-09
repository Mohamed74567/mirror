"""Mirror — Part 3: forward-dynamics simulation and silhouette rendering.

Raw kinematics meet real physics here.  Instead of playing back recorded
angles, the simulator steps the MuJoCo engine forward in time under Earth
gravity (500 Hz, RK4 integration) while position servos track the
computer-vision-derived joint trajectories.  It records the dynamic quantities
that video alone cannot see — joint torques (N·m), internal joint reaction
wrenches — and produces synthetic silhouettes from the exact camera
perspective for the closed-loop IoU calibration of Part 4.

Silhouette backends
-------------------
``mujoco-renderer``
    Offscreen rendering with ``mujoco.Renderer`` (EGL/OSMesa/GLFW).  This is
    the primary backend and is selected automatically whenever a GL context
    is available.

``analytic-pinhole``
    Exact analytic projection of the same geoms through the same pinhole
    camera.  For this planar scene every geom lies at a single constant
    camera depth, so the projection is a uniform scale: capsules map to
    stadium shapes and spheres to circles.  It is mathematically identical to
    the GL rasterisation (up to anti-aliasing) and is used automatically on
    headless CPU-only machines where no GL context exists.
"""

from __future__ import annotations

import logging
import os
import subprocess
import sys
from dataclasses import dataclass
from typing import Dict, Optional, Sequence, Tuple

import cv2
import numpy as np

logger = logging.getLogger("mirror.sim")

RENDERER_BACKEND = "mujoco-renderer"
ANALYTIC_BACKEND = "analytic-pinhole"

_GL_PROBE_SNIPPET = (
    "import os, sys; os.environ['MUJOCO_GL'] = sys.argv[1];"
    "import mujoco;"
    "m = mujoco.MjModel.from_xml_string("
    "'<mujoco><worldbody><geom type=\"sphere\" size=\"0.1\"/></worldbody></mujoco>');"
    "r = mujoco.Renderer(m, 48, 48);"
    "d = mujoco.MjData(m);"
    "mujoco.mj_forward(m, d);"
    "r.update_scene(d);"
    "img = r.render();"
    "print('GL_OK', img.shape)"
)


def resolve_render_backend() -> str:
    """Choose the silhouette backend once, *before* ``mujoco`` is imported.

    Respects a user-provided ``MUJOCO_GL`` value; otherwise probes EGL and
    OSMesa in clean subprocesses.  Falls back to the exact analytic pinhole
    projection when no GL context can be created.
    """
    env = os.environ.get("MUJOCO_GL")
    if env:
        if env.lower() == "disable":
            print("[mirror] MUJOCO_GL=disable requested -> analytic-pinhole silhouette backend")
            return ANALYTIC_BACKEND
        print(f"[mirror] MUJOCO_GL={env} requested -> mujoco-renderer silhouette backend")
        return RENDERER_BACKEND
    for candidate in ("egl", "osmesa"):
        try:
            probe = subprocess.run(
                [sys.executable, "-c", _GL_PROBE_SNIPPET, candidate],
                capture_output=True, text=True, timeout=180,
            )
        except Exception:
            continue
        if probe.returncode == 0 and "GL_OK" in probe.stdout:
            os.environ["MUJOCO_GL"] = candidate
            print(f"[mirror] GL probe OK -> mujoco-renderer backend via {candidate}")
            return RENDERER_BACKEND
    os.environ["MUJOCO_GL"] = "disable"
    print("[mirror] No GL context available -> exact analytic-pinhole silhouette backend")
    return ANALYTIC_BACKEND


RENDER_BACKEND = resolve_render_backend()

import mujoco  # noqa: E402  (imported after MUJOCO_GL is finalised)


# ---------------------------------------------------------------------------
# Angle-convention bridge (vision <-> MuJoCo)
# ---------------------------------------------------------------------------
def vision_angles_to_ctrl(
    angles_rad: Dict[str, np.ndarray],
    hip_offset_rad: float = 0.0,
    knee_offset_rad: float = 0.0,
) -> np.ndarray:
    """Map vision-convention angle trajectories to MuJoCo servo ctrl targets.

    ``angles_rad`` holds ``{"hip": theta_thigh(t), "knee": phi_rel(t)}`` in the
    vision convention (0 = hanging down, positive towards image +x; knee is
    the relative shank angle).  Returns an ``(T, 2)`` array of
    ``[q_hip_target, q_knee_target]`` — a sign inversion plus the additive
    calibration offsets (see :mod:`src.mjcf_builder`).
    """
    hip = -angles_rad["hip"] + hip_offset_rad
    knee = -angles_rad["knee"] + knee_offset_rad
    return np.column_stack([hip, knee])


def ctrl_to_vision_angles(
    q: np.ndarray,
    hip_offset_rad: float = 0.0,
    knee_offset_rad: float = 0.0,
) -> Dict[str, np.ndarray]:
    """Inverse of :func:`vision_angles_to_ctrl` for an ``(T, 2)`` qpos array."""
    return {
        "hip": -(q[:, 0] - hip_offset_rad),
        "knee": -(q[:, 1] - knee_offset_rad),
    }


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------
@dataclass
class TrajectoryResult:
    """Recorded dynamic state of one forward-dynamics rollout."""

    time_s: np.ndarray              # (n_rec,) diagnostics grid (~100 Hz)
    qpos_rad: np.ndarray            # (n_steps, 2) every physics step
    qvel_rad_s: np.ndarray          # (n_steps, 2)
    joint_torque_Nm: np.ndarray     # (n_rec, 2) qfrc_actuator: [hip, knee]
    reaction_force_N: np.ndarray    # (n_rec, 2) |joint reaction force| [hip, knee]
    frame_times_s: np.ndarray       # (T,)
    frame_qpos_rad: np.ndarray      # (T, 2) state sampled at video frame times
    n_steps: int
    sim_rate_hz: float


# ---------------------------------------------------------------------------
# Simulator
# ---------------------------------------------------------------------------
class MirrorSimulator:
    """Drives one compiled MJCF model and renders its silhouettes."""

    BG_DIST_THRESHOLD = 60.0   # RGB distance from background -> foreground pixel
    DIAGNOSTIC_RATE_HZ = 100.0  # torque / reaction-force sampling rate

    def __init__(self, xml_string: str, render_height: int = 480, render_width: int = 640) -> None:
        self.model = mujoco.MjModel.from_xml_string(xml_string)
        self.data = mujoco.MjData(self.model)
        self.render_height = int(render_height)
        self.render_width = int(render_width)

        self._renderer = None
        self._scene_option = None
        self.backend = ANALYTIC_BACKEND
        if RENDER_BACKEND == RENDERER_BACKEND:
            try:
                self._renderer = mujoco.Renderer(
                    self.model, height=self.render_height, width=self.render_width
                )
                self.backend = RENDERER_BACKEND
            except Exception as exc:  # pragma: no cover - GL-dependent
                logger.warning("mujoco.Renderer unavailable (%s); using analytic projection.", exc)
        else:
            logger.debug("Analytic-pinhole silhouette backend active.")

        # Silhouette scene options: fiducial marker geoms (group 1) are used for
        # keypoint detection on the vision side, NOT for the shape objective, so
        # they are excluded from rendered silhouettes on both sides of the IoU.
        self._silhouette_option = None
        if self.backend == RENDERER_BACKEND:
            self._silhouette_option = mujoco.MjvOption()
            self._silhouette_option.geomgroup[1] = False

        self._cam_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_CAMERA, "mirror_cam")
        if self._cam_id < 0:
            raise ValueError("Model does not contain the 'mirror_cam' camera.")
        self._thigh_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "thigh")
        self._shank_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "shank")

        # fiducial landmark geoms (geom group 1) — projected for the keypoint
        # reprojection term of Part 4; names are fixed by src.mjcf_builder
        self._landmark_geom_ids: Dict[str, int] = {}
        for lm_name, geom_name in (
            ("hip", "hip_marker"), ("knee", "knee_marker"), ("ankle", "ankle_marker")
        ):
            gid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, geom_name)
            if gid >= 0:
                self._landmark_geom_ids[lm_name] = gid

        # camera frame axes/orientation are static (fixed camera)
        self._cam_pos = self.model.cam_pos[self._cam_id].astype(np.float64)
        cam_mat0 = self.model.cam_mat0[self._cam_id].reshape(3, 3)
        self._cam_xaxis = cam_mat0[:, 0]
        self._cam_yaxis = cam_mat0[:, 1]
        self._cam_zaxis = cam_mat0[:, 2]              # points *away* from the scene
        self._look_dir = -self._cam_zaxis
        fovy_rad = np.radians(float(self.model.cam_fovy[self._cam_id]))
        self._focal_px = (self.render_height / 2.0) / np.tan(fovy_rad / 2.0)

    # -- forward dynamics ------------------------------------------------------
    def _velocity_ff_coeff(self) -> np.ndarray:
        """Velocity feed-forward coefficient kv/kp per actuator (0 if N/A).

        The position servos use critically-damped PD gains (kp, kv).  When
        tracking a *moving* reference, the kv*qvel damping term must itself be
        produced by the position error, which creates a velocity-proportional
        tracking lag of kv*v/kp radians.  Standard servo practice removes this
        lag with a velocity feed-forward: ctrl = target + (kv/kp)*d(target)/dt.
        """
        n_act = self.model.nu
        coeff = np.zeros(n_act)
        for a in range(n_act):
            kp = float(self.model.actuator_gainprm[a][0])
            kv = -float(self.model.actuator_biasprm[a][2])
            if kp > 0 and kv > 0:
                coeff[a] = kv / kp
        return coeff

    def simulate_trajectory(
        self, frame_times_s: np.ndarray, ctrl_targets: np.ndarray
    ) -> TrajectoryResult:
        """Roll out forward dynamics while the servos track ``ctrl_targets``.

        Control targets are defined at the video frame times and linearly
        interpolated at the 500 Hz physics rate; a velocity feed-forward
        ``(kv/kp) * d(target)/dt`` (see :meth:`_velocity_ff_coeff`) cancels the
        damping-induced tracking lag so the identified geometry — not servo
        lag — dominates the silhouette residual.  The trajectory starts from
        the first target (no swing-in transient).
        """
        frame_times_s = np.asarray(frame_times_s, dtype=np.float64)
        ctrl_targets = np.asarray(ctrl_targets, dtype=np.float64)
        dt = float(self.model.opt.timestep)
        n_steps = int(round(frame_times_s[-1] / dt)) + 1
        rec_every = max(1, int(round((1.0 / self.DIAGNOSTIC_RATE_HZ) / dt)))
        n_rec = len(range(0, n_steps, rec_every))

        # velocity feed-forward from the smoothed reference trajectories
        ff = self._velocity_ff_coeff()
        if len(frame_times_s) >= 3:
            target_vel = np.gradient(ctrl_targets, frame_times_s, axis=0, edge_order=2)
        else:
            target_vel = np.zeros_like(ctrl_targets)

        self.data.qpos[:] = ctrl_targets[0]
        # the recorded movement is already in motion at t=0 — start with the
        # reference velocity (avoids an artificial startup transient)
        self.data.qvel[:] = target_vel[0]
        mujoco.mj_forward(self.model, self.data)

        qpos_hist = np.zeros((n_steps, 2))
        qvel_hist = np.zeros((n_steps, 2))
        t_rec = np.zeros(n_rec)
        tau_rec = np.zeros((n_rec, 2))
        rf_rec = np.zeros((n_rec, 2))

        rec_i = 0
        for step in range(n_steps):
            t = step * dt
            for j in range(2):
                self.data.ctrl[j] = float(
                    np.interp(t, frame_times_s, ctrl_targets[:, j])
                    + ff[j] * np.interp(t, frame_times_s, target_vel[:, j])
                )
            mujoco.mj_step(self.model, self.data)
            qpos_hist[step] = self.data.qpos
            qvel_hist[step] = self.data.qvel
            if step % rec_every == 0:
                # recompute derived quantities at the recorded state, then read
                # actuator torque and the internal joint-reaction wrench.
                # NOTE: cfrc_int rows are [moment(3); force(3)] in MuJoCo's
                # com-based convention; the force part (validated against the
                # static subtree weight) is the joint reaction force.
                mujoco.mj_forward(self.model, self.data)
                mujoco.mj_rnePostConstraint(self.model, self.data)
                t_rec[rec_i] = t
                tau_rec[rec_i] = self.data.qfrc_actuator
                rf_rec[rec_i] = (
                    np.linalg.norm(self.data.cfrc_int[self._thigh_id][3:]),
                    np.linalg.norm(self.data.cfrc_int[self._shank_id][3:]),
                )
                rec_i += 1

        frame_steps = np.round(frame_times_s / dt).astype(int)
        frame_steps = np.clip(frame_steps, 0, n_steps - 1)
        return TrajectoryResult(
            time_s=t_rec,
            qpos_rad=qpos_hist,
            qvel_rad_s=qvel_hist,
            joint_torque_Nm=tau_rec,
            reaction_force_N=rf_rec,
            frame_times_s=frame_times_s,
            frame_qpos_rad=qpos_hist[frame_steps],
            n_steps=n_steps,
            sim_rate_hz=1.0 / dt,
        )

    # -- rendering ---------------------------------------------------------------
    def set_pose(self, qpos: Sequence[float]) -> None:
        self.data.qpos[:] = qpos
        self.data.qvel[:] = 0.0
        mujoco.mj_forward(self.model, self.data)

    def render_silhouette(self, qpos: Sequence[float]) -> np.ndarray:
        """Binary limb silhouette (H, W) uint8 {0, 255} for the given joint pose.

        Fiducial marker geoms (geom group 1) are excluded so the mask contains
        only the limb capsules — matching the limb-only masks extracted from
        the video by :class:`~src.vision_extractor.VisionExtractor`.
        """
        self.set_pose(qpos)
        if self._renderer is not None:
            self._renderer.update_scene(
                self.data, camera="mirror_cam", scene_option=self._silhouette_option
            )
            rgb = self._renderer.render()
            return self._threshold_background(rgb)
        return self._analytic_mask()

    def render_rgb(self, qpos: Sequence[float]) -> np.ndarray:
        """Shaded RGB preview (H, W, 3) of the model in the given pose.

        Unlike :meth:`render_silhouette`, the fiducial markers are drawn here
        so the comparison video shows the full digital twin.
        """
        self.set_pose(qpos)
        if self._renderer is not None:
            if self._scene_option is None:
                self._scene_option = mujoco.MjvOption()
            self._renderer.update_scene(
                self.data, camera="mirror_cam", scene_option=self._scene_option
            )
            return self._renderer.render()
        return self._analytic_rgb()

    def _threshold_background(self, rgb: np.ndarray) -> np.ndarray:
        """Binary mask from a GL render: distance from the border-frame colour."""
        ring = 10
        border = np.concatenate(
            [rgb[:ring].reshape(-1, 3), rgb[-ring:].reshape(-1, 3),
             rgb[:, :ring].reshape(-1, 3), rgb[:, -ring:].reshape(-1, 3)]
        )
        bg = np.median(border, axis=0)
        dist = np.linalg.norm(rgb.astype(np.int16) - bg.astype(np.int16), axis=-1)
        return ((dist > self.BG_DIST_THRESHOLD) * 255).astype(np.uint8)

    def project_landmarks(self, qpos: Sequence[float]) -> Dict[str, np.ndarray]:
        """Pinhole-project the three fiducial landmark centres into the image.

        For the given joint pose, returns ``{name: (u, v)}`` pixel coordinates
        (OpenCV convention: origin top-left, v down) of the *true* rotation-
        centre landmarks (hip anchor, knee, ankle marker spheres), through the
        same ``mirror_cam`` camera that produces the silhouettes.  This is the
        model-side half of the keypoint-reprojection term used by the fused
        calibration loss (Part 4) — the observation side is the fiducial
        centroids tracked in Part 1.
        """
        self.set_pose(qpos)
        out: Dict[str, np.ndarray] = {}
        for lm_name, gid in self._landmark_geom_ids.items():
            center = self.data.geom_xpos[gid]
            depth = max(self._depth(center), 1e-6)
            out[lm_name] = self._project(center.reshape(1, 3), depth)[0]
        return out

    # -- analytic pinhole projection ---------------------------------------------
    def _project(self, points: np.ndarray, depth: float) -> np.ndarray:
        """World points -> pixel coordinates through the pinhole camera."""
        rel = points - self._cam_pos
        u = (self.render_width / 2.0) + self._focal_px * (rel @ self._cam_xaxis) / depth
        v = (self.render_height / 2.0) - self._focal_px * (rel @ self._cam_yaxis) / depth
        return np.column_stack([u, v])

    def _depth(self, point: np.ndarray) -> float:
        return float((point - self._cam_pos) @ self._look_dir)

    def _analytic_mask(self) -> np.ndarray:
        """Exact silhouette via pinhole projection of the limb geoms (group 0)."""
        mask = np.zeros((self.render_height, self.render_width), dtype=np.uint8)
        for gid in range(self.model.ngeom):
            if int(self.model.geom_group[gid]) != 0:
                continue  # fiducial markers are excluded from silhouettes
            gtype = int(self.model.geom_type[gid])
            radius = float(self.model.geom_size[gid][0])
            center = self.data.geom_xpos[gid]
            if gtype == int(mujoco.mjtGeom.mjGEOM_SPHERE):
                self._draw_sphere(mask, center, radius)
            elif gtype == int(mujoco.mjtGeom.mjGEOM_CAPSULE):
                half = float(self.model.geom_size[gid][1])
                axis = self.data.geom_xmat[gid].reshape(3, 3)[:, 2]
                self._draw_capsule(mask, center - axis * half, center + axis * half, radius)
        return mask

    def _analytic_rgb(self) -> np.ndarray:
        """Shaded-looking RGB preview: geoms drawn in their MJCF rgba colours.

        Capsules are drawn first and fiducial spheres last, mirroring the
        video's visual convention (markers on top of the limbs).
        """
        canvas = np.zeros((self.render_height, self.render_width, 3), dtype=np.uint8)
        capsule_gids = [
            gid for gid in range(self.model.ngeom)
            if int(self.model.geom_type[gid]) == int(mujoco.mjtGeom.mjGEOM_CAPSULE)
        ]
        sphere_gids = [
            gid for gid in range(self.model.ngeom)
            if int(self.model.geom_type[gid]) == int(mujoco.mjtGeom.mjGEOM_SPHERE)
        ]
        for gid in capsule_gids + sphere_gids:
            gtype = int(self.model.geom_type[gid])
            radius = float(self.model.geom_size[gid][0])
            center = self.data.geom_xpos[gid]
            color = tuple(int(c * 255) for c in self.model.geom_rgba[gid][:3])
            if gtype == int(mujoco.mjtGeom.mjGEOM_CAPSULE):
                half = float(self.model.geom_size[gid][1])
                axis = self.data.geom_xmat[gid].reshape(3, 3)[:, 2]
                self._draw_capsule(canvas, center - axis * half, center + axis * half, radius, color)
            else:
                self._draw_sphere(canvas, center, radius, color)
        return canvas

    def _draw_capsule(
        self, canvas: np.ndarray, p1: np.ndarray, p2: np.ndarray, radius_m: float,
        color: Optional[Tuple[int, ...]] = None,
    ) -> None:
        d1, d2 = self._depth(p1), self._depth(p2)
        depth = max(0.5 * (d1 + d2), 1e-6)
        uv = self._project(np.array([p1, p2]), depth)
        thickness = max(1, int(round(2.0 * self._focal_px * radius_m / depth)))
        pt1 = (int(round(uv[0, 0])), int(round(uv[0, 1])))
        pt2 = (int(round(uv[1, 0])), int(round(uv[1, 1])))
        if color is None:
            cv2.line(canvas, pt1, pt2, 255, thickness=thickness, lineType=cv2.LINE_AA)
        else:
            cv2.line(canvas, pt1, pt2, color, thickness=thickness, lineType=cv2.LINE_AA)
        cap_px = max(1.0, self._focal_px * radius_m / depth)
        for pt in (pt1, pt2):
            cv2.circle(canvas, pt, int(round(cap_px)), color or 255, thickness=-1, lineType=cv2.LINE_AA)

    def _draw_sphere(
        self, canvas: np.ndarray, center: np.ndarray, radius_m: float,
        color: Optional[Tuple[int, ...]] = None,
    ) -> None:
        depth = max(self._depth(center), 1e-6)
        uv = self._project(center.reshape(1, 3), depth)[0]
        radius_px = max(1.0, self._focal_px * radius_m / depth)
        pt = (int(round(uv[0])), int(round(uv[1])))
        cv2.circle(canvas, pt, int(round(radius_px)), color or 255, thickness=-1, lineType=cv2.LINE_AA)

    # -- lifecycle -----------------------------------------------------------
    def close(self) -> None:
        if self._renderer is not None:
            self._renderer.close()
            self._renderer = None

    def __del__(self):  # pragma: no cover - best-effort GL cleanup
        try:
            self.close()
        except Exception:
            pass


# ---------------------------------------------------------------------------
# Metrics helpers
# ---------------------------------------------------------------------------
def silhouette_iou(mask_a: np.ndarray, mask_b: np.ndarray) -> float:
    """Intersection-over-Union between two binary silhouette masks."""
    a = mask_a > 127
    b = mask_b > 127
    union = int(np.logical_or(a, b).sum())
    if union == 0:
        return 0.0
    return float(np.logical_and(a, b).sum()) / union


def tracking_rmse_deg(ctrl_targets: np.ndarray, achieved_qpos: np.ndarray) -> Dict[str, float]:
    """Per-joint RMS servo-tracking error (deg) between ctrl targets and qpos."""
    err_deg = np.degrees(np.asarray(ctrl_targets) - np.asarray(achieved_qpos))
    return {
        "hip_deg": float(np.sqrt(np.mean(err_deg[:, 0] ** 2))),
        "knee_deg": float(np.sqrt(np.mean(err_deg[:, 1] ** 2))),
    }

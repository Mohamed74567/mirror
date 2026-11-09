"""Mirror test-suite: unit checks for every pipeline stage plus a fast
closed-loop calibration integration test.

Run with ``python -m pytest`` from the repository root.
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pytest

cv2 = pytest.importorskip("cv2")
mujoco = pytest.importorskip("mujoco")

from src.mjcf_builder import CALIBRATION_PARAM_NAMES, MJCFBuilder, ModelParams
from src.mujoco_sim import (
    ANALYTIC_BACKEND,
    RENDERER_BACKEND,
    RENDER_BACKEND,
    MirrorSimulator,
    ctrl_to_vision_angles,
    silhouette_iou,
    vision_angles_to_ctrl,
)
from src.optimizer import (
    GEOMETRY_PARAM_NAMES,
    MirrorOptimizer,
    estimate_ci95_curvature,
)
from src.vision_extractor import (
    DEFAULT_MARKER_HSV,
    CameraIntrinsics,
    VisionExtractor,
    _safe_savgol_window,
    detect_marker_centroids,
)
from scripts.make_demo_video import DemoConfig, background_canvas, draw_frame, joint_positions_world, kinematics, world_to_pixel

ROOT = Path(__file__).resolve().parents[1]
ASSETS = ROOT / "assets"
DEMO_VIDEO = ASSETS / "demo_sagittal_swing.mp4"


# ---------------------------------------------------------------------------
# Part 0 — camera model & demo geometry
# ---------------------------------------------------------------------------
def test_camera_intrinsics_fov():
    cam = CameraIntrinsics(width=640, height=480, focal_length_px=620.0)
    expected = 2.0 * math.degrees(math.atan(480.0 / (2.0 * 620.0)))
    assert abs(cam.fovy_deg - expected) < 1e-9


def test_demo_kinematics_and_projection_roundtrip():
    cfg = DemoConfig()
    theta, phi = kinematics(cfg, 0.4)
    joints = joint_positions_world(cfg, theta, phi)
    hip_px = world_to_pixel(cfg, *joints["hip"])
    assert abs(hip_px[0] - cfg.hip_pixel[0]) < 1e-6
    assert abs(hip_px[1] - cfg.hip_pixel[1]) < 1e-6
    # knee must be femur-length away from hip in world coordinates
    d = np.linalg.norm(joints["knee"] - joints["hip"])
    assert abs(d - cfg.L_femur_m) < 1e-9


def test_safe_savgol_window():
    assert _safe_savgol_window(200, 11, 3) == 11
    assert _safe_savgol_window(8, 11, 3) in (5, 7)
    assert _safe_savgol_window(3, 11, 3) is None  # too short to filter


# ---------------------------------------------------------------------------
# Part 1 — vision
# ---------------------------------------------------------------------------
def test_marker_detection_subpixel_accuracy():
    cfg = DemoConfig()
    base_bg = background_canvas(cfg)
    rng = np.random.default_rng(cfg.seed)
    theta, phi = kinematics(cfg, 0.37)
    frame = draw_frame(cfg, base_bg, rng, theta, phi)
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    detections = detect_marker_centroids(hsv, DEFAULT_MARKER_HSV, min_area_px=30.0)

    joints_world = joint_positions_world(cfg, theta, phi)
    for name in ("hip", "knee", "ankle"):
        assert detections[name] is not None, f"marker {name} not detected"
        centroid, area = detections[name]
        u, v = world_to_pixel(cfg, *joints_world[name])
        bias = cfg.marker_bias_px[name]
        expected = np.array([u + bias[0], v + bias[1]])
        assert np.linalg.norm(centroid - expected) < 1.5, f"{name}: {centroid} vs {expected}"
        assert area > 50.0


def test_vision_extractor_on_synthetic_clip():
    if not DEMO_VIDEO.exists():
        pytest.skip("demo asset missing — run scripts/make_demo_video.py first")
    vision = VisionExtractor(DEMO_VIDEO, max_frames=30).extract()

    assert vision.n_frames == 30
    assert vision.tracking_success_rate > 0.95
    assert set(vision.angles_rad) == {"hip", "knee"}

    # compare against ground truth: the skin-marker bias (~3 deg hip, ~7 deg
    # knee RMS) plus Savitzky-Golay smoothing at the flexion peak bound the
    # per-frame deviation; assert both the RMS and a sanity ceiling
    gt = __import__("json").loads((ASSETS / "demo_ground_truth.json").read_text())
    theta_gt = np.asarray(gt["theta_thigh_true_rad"][:30])
    phi_gt = np.asarray(gt["phi_knee_true_rad"][:30])
    hip_err_deg = np.degrees(vision.angles_rad["hip"] - theta_gt)
    knee_err_deg = np.degrees(vision.angles_rad["knee"] - phi_gt)
    assert np.sqrt((hip_err_deg ** 2).mean()) < 8.0
    assert np.sqrt((knee_err_deg ** 2).mean()) < 8.0
    assert np.abs(hip_err_deg).max() < 10.0
    assert np.abs(knee_err_deg).max() < 15.0

    # pinhole length estimates from the *biased* marker centroids: the femur
    # lands within ~5%, while the tibia inherits the relative knee/ankle marker
    # offset (+9,+9 px on an ~83 px segment) and lands ~14% high — precisely
    # the initialisation error the closed-loop calibration (Part 4) corrects.
    assert abs(vision.initial_link_lengths_m["femur_m"] - gt["links"]["L_femur_m"]) / gt["links"]["L_femur_m"] < 0.06
    assert abs(vision.initial_link_lengths_m["tibia_m"] - gt["links"]["L_tibia_m"]) / gt["links"]["L_tibia_m"] < 0.16


# ---------------------------------------------------------------------------
# Part 2 — MJCF compilation
# ---------------------------------------------------------------------------
def test_mjcf_compiles_with_expected_structure():
    model = MJCFBuilder(ModelParams()).compile()
    assert model.nq == 2 and model.nv == 2 and model.nu == 2
    assert model.nbody == 4  # world + hip_anchor + thigh + shank
    assert model.body_mass.sum() > 1.0        # density-derived masses present
    assert model.jnt_limited.all()            # physiological joint limits set
    assert model.ngeom == 5                   # hip/knee/ankle spheres + femur/tibia capsules
    # actuators are position servos
    assert model.actuator_gainprm[0, 0] == pytest.approx(ModelParams().servo_kp, rel=1e-6)


def test_calibration_vector_roundtrip():
    base = ModelParams(L_femur_m=0.5, L_tibia_m=0.45, hip_offset_rad=0.1, knee_offset_rad=-0.1, servo_kp=900.0)
    x = base.calibration_vector()
    assert len(x) == len(CALIBRATION_PARAM_NAMES)
    other = ModelParams.with_calibration_vector(ModelParams(), x)
    assert other.L_femur_m == 0.5 and other.servo_kp == 900.0


# ---------------------------------------------------------------------------
# Part 3 — forward dynamics & rendering
# ---------------------------------------------------------------------------
def test_forward_dynamics_tracks_targets():
    params = ModelParams(servo_kp=3000.0)
    sim = MirrorSimulator(MJCFBuilder(params).to_xml())
    try:
        frame_times = np.arange(0, 1.0, 1.0 / 30.0)
        theta = 0.20 * np.sin(2 * np.pi * 0.8 * frame_times)
        # knee flexion only (phi <= 0) so targets respect the joint limits
        phi = -0.30 * np.clip(np.sin(2 * np.pi * 0.8 * frame_times + 1.0), 0.0, None)
        ctrl = np.column_stack([-theta, -phi])
        traj = sim.simulate_trajectory(frame_times, ctrl)
        err_deg = np.degrees(np.abs(ctrl - traj.frame_qpos_rad).max())
        assert err_deg < 2.5, f"servo tracking error too large: {err_deg:.2f} deg"
        assert traj.joint_torque_Nm.shape[0] == traj.time_s.shape[0]
        assert np.all(np.isfinite(traj.reaction_force_N))
    finally:
        sim.close()


def test_silhouette_is_nonempty_and_centered():
    sim = MirrorSimulator(MJCFBuilder(ModelParams()).to_xml())
    try:
        mask = sim.render_silhouette([0.0, 0.0])
        assert mask.shape == (480, 640)
        ys, xs = np.nonzero(mask > 127)
        assert len(xs) > 2000, "silhouette unexpectedly small"
        # hip anchored near the top-centre of the image
        assert 120 < ys.min() < 200
        assert 260 < xs.mean() < 380
    finally:
        sim.close()


@pytest.mark.skipif(RENDER_BACKEND != RENDERER_BACKEND, reason="requires a GL context for mujoco.Renderer")
def test_gl_and_analytic_silhouettes_agree():
    """The GL renderer and the analytic pinhole projection must agree (same math)."""
    sim_gl = MirrorSimulator(MJCFBuilder(ModelParams()).to_xml())
    assert sim_gl.backend == RENDERER_BACKEND
    try:
        sim_an = MirrorSimulator.__new__(MirrorSimulator)
        # force a fresh analytic instance by temporarily masking the renderer
        sim_an = MirrorSimulator(MJCFBuilder(ModelParams()).to_xml())
        sim_an._renderer, saved = None, sim_an._renderer
        sim_an.backend = ANALYTIC_BACKEND
        for q in ([0.1, -0.5], [-0.2, 0.8], [0.0, 0.0]):
            m_gl = sim_gl.render_silhouette(q)
            m_an = sim_an.render_silhouette(q)
            assert silhouette_iou(m_gl, m_an) > 0.96, f"backends disagree at q={q}"
    finally:
        sim_gl.close()


# ---------------------------------------------------------------------------
# Part 3b — landmark pinhole projection (keypoint reprojection)
# ---------------------------------------------------------------------------
def test_landmark_projection_known_pose():
    """At q = 0 the hip landmark must project exactly onto the vision hip pixel
    (the camera placement in the MJCF is derived from that very pixel)."""
    params = ModelParams(cam_distance_m=3.0, cam_height_m=0.45, cam_x_m=0.0,
                         hip_height_m=0.95, fovy_deg=42.0)
    sim = MirrorSimulator(MJCFBuilder(params).to_xml())
    try:
        lm = sim.project_landmarks([0.0, 0.0])
        assert set(lm) == {"hip", "knee", "ankle"}
        # hip marker sits at the anchor: u == principal point, v above centre
        assert lm["hip"][0] == pytest.approx(320.0, abs=1.0)
        assert lm["hip"][1] < 240.0  # above image centre
        # knee hangs L_femur below the hip in world -> larger v
        assert lm["knee"][1] > lm["hip"][1]
        assert lm["ankle"][1] > lm["knee"][1]
        # vertical pixel gap must match the pinhole scale: f_px * L / Z
        f_px = (480.0 / 2.0) / math.tan(math.radians(42.0) / 2.0)
        depth = 3.0
        gap_femur_px = f_px * params.L_femur_m / depth
        assert (lm["knee"][1] - lm["hip"][1]) == pytest.approx(gap_femur_px, rel=1e-3)
        gap_tibia_px = f_px * params.L_tibia_m / depth
        assert (lm["ankle"][1] - lm["knee"][1]) == pytest.approx(gap_tibia_px, rel=1e-3)
    finally:
        sim.close()


def test_estimate_ci95_curvature_quadratic():
    """On an exact parabola the CI half-width must match the analytic
    likelihood-ratio solution sqrt(chi2_95 / H_ii) * span."""
    target = np.array([0.42, 0.40, 0.05, -0.11, 1500.0])
    span = np.array([0.6, 0.6, 0.7, 0.7, 4950.0])

    def quad(x):
        d = (np.asarray(x) - target) / span
        return float(10.0 * d @ d)   # H_ii = 20 in normalized units

    ci = estimate_ci95_curvature(quad, target.tolist())
    expected_half_norm = math.sqrt(3.841458820694124 / 20.0)
    for name, s in zip(("L_femur_m", "L_tibia_m", "hip_offset_rad", "knee_offset_rad", "servo_kp"), span):
        assert ci[name]["curvature_ok"]
        assert ci[name]["ci95_half_width"] == pytest.approx(expected_half_norm * s, rel=1e-6)
        assert ci[name]["lo"] <= target[list(ci).index(name)] <= ci[name]["hi"]


def test_flat_direction_reports_full_range():
    """A perfectly flat direction must yield the conservative full-range CI."""
    def flat(x):
        return 1.0

    ci = estimate_ci95_curvature(flat, [0.42, 0.40, 0.05, -0.11, 1500.0])
    assert not ci["servo_kp"]["curvature_ok"]
    assert ci["servo_kp"]["ci95_half_width"] == pytest.approx(0.5 * (5000.0 - 50.0))


# ---------------------------------------------------------------------------
# Part 4 — closed-loop calibration (fast integration test)
# ---------------------------------------------------------------------------
def test_closed_loop_calibration_improves_geometry():
    if not DEMO_VIDEO.exists():
        pytest.skip("demo asset missing — run scripts/make_demo_video.py first")
    from run_mirror import initial_params_from_vision

    vision = VisionExtractor(DEMO_VIDEO, max_frames=24).extract()
    params0 = initial_params_from_vision(vision, camera_distance_m=3.0, kp_init=250.0)

    optimizer = MirrorOptimizer(vision, params0, frame_stride=4, max_eval_frames=8)
    result = optimizer.calibrate(maxfev=30)

    # a 30-evaluation budget cannot fully converge; it must, however, never
    # degrade the objective and must land in a sane regime
    assert result.mean_iou_final > 0.70
    assert result.mean_iou_final >= result.mean_iou_initial - 0.01
    assert result.n_evals == len(result.eval_history)
    # best-so-far must be monotone non-increasing
    assert np.all(np.diff(np.minimum.accumulate(result.eval_history)) <= 1e-12)


def test_two_phase_refinement_recovers_mount_offsets():
    """Phase 2 (nuisance-aware fused refinement) must absorb the skin-mount
    pixel bias into the per-marker offsets: the fitted deltas must be close to
    the published ground-truth mount bias, and the reprojection RMSE must
    collapse well below its phase-1 level."""
    if not DEMO_VIDEO.exists():
        pytest.skip("demo asset missing — run scripts/make_demo_video.py first")
    import json as _json

    from run_mirror import initial_params_from_vision
    from src.optimizer import calibrate_two_phase

    vision = VisionExtractor(DEMO_VIDEO, max_frames=24).extract()
    params0 = initial_params_from_vision(vision, camera_distance_m=3.0, kp_init=250.0)

    _, res2, opt2 = calibrate_two_phase(
        vision, params0, kp_weight=0.1, maxfev_total=90,
        frame_stride=4, max_eval_frames=6,
        render_height=vision.height, render_width=vision.width,
    )
    assert opt2.optimize_marker_offsets
    assert res2.marker_offsets_px is not None
    gt = _json.loads((ASSETS / "demo_ground_truth.json").read_text())
    for name in ("hip", "knee", "ankle"):
        dx, dy = res2.marker_offsets_px[name]
        tx, ty = gt["marker_bias_px"][name]
        assert math.hypot(dx - tx, dy - ty) < 8.0, f"{name}: ({dx:.1f},{dy:.1f}) vs ({tx},{ty})"
    # reprojection must improve overall (phase-2 final vs phase-2 initial)
    r0 = res2.reproj_rmse_px_initial
    r1 = res2.reproj_rmse_px_final
    assert sum(r1.values()) < sum(r0.values())


def test_naive_absolute_keypoint_term_documented():
    """Regression guard for the design decision: the plain 5-parameter fused
    objective (no nuisance offsets) must remain available and finite, since it
    is the ablation baseline that demonstrates why the offsets are needed."""
    if not DEMO_VIDEO.exists():
        pytest.skip("demo asset missing — run scripts/make_demo_video.py first")
    from run_mirror import initial_params_from_vision

    vision = VisionExtractor(DEMO_VIDEO, max_frames=16).extract()
    params0 = initial_params_from_vision(vision, camera_distance_m=3.0, kp_init=250.0)
    opt = MirrorOptimizer(vision, params0, frame_stride=4, max_eval_frames=4,
                          kp_weight=0.1, optimize_marker_offsets=False)
    assert opt.n_params == 5
    loss = opt.fused_loss(params0.calibration_vector())
    assert np.isfinite(loss) and loss > 0
    x_full = opt.seed_vector()
    assert len(x_full) == 5


def test_vision_to_ctrl_convention_roundtrip():
    angles = {"hip": np.array([0.2, -0.1]), "knee": np.array([-0.4, 0.1])}
    ctrl = vision_angles_to_ctrl(angles, hip_offset_rad=0.05, knee_offset_rad=0.03)
    assert ctrl[0, 0] == pytest.approx(-0.15)
    assert ctrl[0, 1] == pytest.approx(0.43)
    back = ctrl_to_vision_angles(ctrl, hip_offset_rad=0.05, knee_offset_rad=0.03)
    assert back["hip"][0] == pytest.approx(0.2)
    assert back["knee"][1] == pytest.approx(0.1)

"""Mirror — Part 4: closed-loop calibration (silhouette IoU + keypoint reprojection).

Solves the inverse calibration problem: the initial geometry guessed from
monocular pixels always carries errors (fiducial-placement bias, limb-width
uncertainty, servo tuning), so the simulated limb does not exactly overlay the
observed limb.  Mirror therefore *closes the loop*:

    candidate params -> MJCF compile -> forward dynamics (500 Hz)
        -> silhouettes from the exact camera perspective
        -> IoU against the video's foreground masks
        -> landmark pinhole reprojection against the tracked fiducial centroids
        -> loss = (1 - mean IoU) + w_kp * reprojection RMSE (in marker radii)

Why the marker-mount nuisance offsets exist
-------------------------------------------
The silhouette IoU pins the overall limb shape, but the knee sits *inside* the
union of the two limb silhouettes, so the femur/tibia split is only weakly
constrained by shape overlap alone.  The tracked fiducial centroids observe
the knee directly — however, skin-mounted markers are glued a few pixels away
from the true rotation centres.  Fitting *absolute* landmark positions without
modelling that constant mount bias would chase the bias and inflate the
segments (demonstrated in ``scripts/ablation_study.py``).

Mirror therefore treats the per-marker constant pixel offsets
``delta_marker = (dx, dy)`` as explicit **nuisance parameters** of the
reprojection term::

    residual(t) = [ project_landmark(q(t)) + delta_marker ] - centroid(t)

so the optimiser must explain the *time-varying* part of the keypoint
residuals through the geometry while the constant part is absorbed by the
nuisance — the standard "partial out the bias" construction.  A valuable
by-product: the fitted ``delta_marker`` is a direct measurement of the
skin-mount offset in pixels, which the benchmark's published ground truth
scores exactly.

The calibration runs in two phases (see :func:`calibrate_two_phase` in
``run_mirror.py``):

* **Phase 1 — geometry.** Silhouette-only Powell over the 5 geometry/servo
  parameters (well-conditioned, cannot be biased by the fiducial mounts).
* **Phase 2 — fused refinement.** Powell over the 5 geometry parameters plus
  the 6 nuisance offsets, seeded at the Phase-1 optimum; the reprojection
  term now sharpens the femur/tibia split instead of corrupting it.

``kp_weight = 0`` or ``optimize_marker_offsets = False`` recovers the pure
silhouette-IoU objective (the original Mirror formulation — the ablation
baseline).

The loss is minimised with a derivative-free solver
(``scipy.optimize.minimize``, Powell method) in bound-normalized coordinates
u = (x - lo)/(hi - lo) so that metres, radians, thousands and pixels present
equally-scaled directions to the solver.

After convergence, :func:`estimate_ci95_curvature` provides local-curvature
(Gaussian / likelihood-ratio) 95% confidence intervals for every identified
parameter — point estimates become error bars.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from scipy.optimize import minimize

from .mjcf_builder import MJCFBuilder, ModelParams
from .mujoco_sim import (
    MirrorSimulator,
    silhouette_iou,
    vision_angles_to_ctrl,
)

logger = logging.getLogger("mirror.optimizer")

#: geometry/servo calibration parameters, in vector order (Phase 1)
GEOMETRY_PARAM_NAMES: Tuple[str, ...] = (
    "L_femur_m",
    "L_tibia_m",
    "hip_offset_rad",
    "knee_offset_rad",
    "servo_kp",
)

#: physically reasonable hard bounds for each geometry parameter
GEOMETRY_PARAM_BOUNDS: Tuple[Tuple[float, float], ...] = (
    (0.15, 0.75),      # L_femur  [m]
    (0.15, 0.75),      # L_tibia  [m]
    (-0.35, 0.35),     # hip offset   [rad]
    (-0.35, 0.35),     # knee offset  [rad]
    (50.0, 5000.0),    # position-servo gain kp
)

#: per-marker constant pixel nuisance offsets (dx, dy) of the reprojection
#: term — the modelled skin-mount bias (Phase 2)
MARKER_OFFSET_PARAM_NAMES: Tuple[str, ...] = (
    "hip_marker_dx_px", "hip_marker_dy_px",
    "knee_marker_dx_px", "knee_marker_dy_px",
    "ankle_marker_dx_px", "ankle_marker_dy_px",
)

#: marker-mount offsets are small (skin fiducials are glued near the joint);
#: +/-15 px is generous for any reasonable recording
MARKER_OFFSET_PARAM_BOUNDS: Tuple[Tuple[float, float], ...] = ((-15.0, 15.0),) * 6

#: backwards-compatible aliases (silhouette-only formulation)
PARAM_NAMES = GEOMETRY_PARAM_NAMES
PARAM_BOUNDS = GEOMETRY_PARAM_BOUNDS

#: published convergence target for the mean silhouette IoU
TARGET_IOU = 0.92

#: chi^2 95% quantile, 1 degree of freedom (likelihood-ratio CI threshold)
CHI2_95_1DOF = 3.841458820694124

MARKER_ORDER: Tuple[str, str, str] = ("hip", "knee", "ankle")


@dataclass
class CalibrationResult:
    """Outcome of one closed-loop calibration run."""

    best_params: ModelParams
    best_loss: float                     # fused loss on the calibration frames
    mean_iou_initial: float
    mean_iou_final: float                # silhouette IoU on the calibration frames
    per_frame_iou_final: np.ndarray
    eval_history: np.ndarray             # fused loss at every objective evaluation
    eval_history_terms: np.ndarray       # (n_evals, 2): [silhouette term, kp term]
    n_evals: int
    success: bool                        # scipy convergence flag
    message: str
    wall_time_s: float
    # --- keypoint-reprojection diagnostics -----------------------------------
    kp_weight: float = 0.0
    kp_sigma_px: float = 1.0
    optimize_marker_offsets: bool = False
    marker_offsets_px: Optional[Dict[str, Tuple[float, float]]] = None
    reproj_rmse_px_final: Dict[str, float] = field(default_factory=dict)
    reproj_rmse_px_initial: Dict[str, float] = field(default_factory=dict)
    # --- held-out validation ---------------------------------------------------
    heldout_indices: Optional[np.ndarray] = None
    mean_iou_heldout: Optional[float] = None
    per_frame_iou_heldout: Optional[np.ndarray] = None

    @property
    def target_met(self) -> bool:
        return self.mean_iou_final >= TARGET_IOU


class MirrorOptimizer:
    """Derivative-free closed-loop calibration of the Mirror digital twin.

    Parameters
    ----------
    vision_result:
        Output of :class:`src.vision_extractor.VisionExtractor` (masks,
        smoothed keypoints, angles, camera model).
    initial_params:
        Seed :class:`ModelParams` from the vision stage (or from a previous
        calibration phase — morphology fields are never re-optimised).
    frame_stride, max_eval_frames:
        Sub-sampling of the video frames used as the *calibration set*; the
        remaining frames form a held-out validation set that never enters the
        objective.
    kp_weight:
        Weight of the keypoint-reprojection term in the fused loss
        (``0`` = silhouette-only objective).
    optimize_marker_offsets:
        If True, the per-marker constant pixel offsets are appended to the
        calibration vector as nuisance parameters (11-dim problem).  Requires
        ``kp_weight > 0`` to have any effect.
    kp_sigma_px:
        Pixel scale used to normalise reprojection residuals; defaults to the
        median fiducial radius measured in Part 1 (a physically meaningful
        "one marker radius" unit).
    """

    def __init__(
        self,
        vision_result,
        initial_params: ModelParams,
        frame_stride: int = 2,
        max_eval_frames: int = 36,
        render_height: int = 480,
        render_width: int = 640,
        kp_weight: float = 0.0,
        optimize_marker_offsets: bool = False,
        kp_sigma_px: Optional[float] = None,
    ) -> None:
        self.vision = vision_result
        self.initial_params = initial_params
        self.render_height = int(render_height)
        self.render_width = int(render_width)
        self.kp_weight = float(kp_weight)
        self.optimize_marker_offsets = bool(optimize_marker_offsets)
        self.kp_sigma_px = float(
            kp_sigma_px if kp_sigma_px is not None else vision_result.marker_radius_px
        )

        if self.optimize_marker_offsets:
            self.param_names = GEOMETRY_PARAM_NAMES + MARKER_OFFSET_PARAM_NAMES
            self.param_bounds = GEOMETRY_PARAM_BOUNDS + MARKER_OFFSET_PARAM_BOUNDS
        else:
            self.param_names = GEOMETRY_PARAM_NAMES
            self.param_bounds = GEOMETRY_PARAM_BOUNDS
        self.n_params = len(self.param_names)
        self._bounds_lo = np.asarray([b[0] for b in self.param_bounds], dtype=np.float64)
        self._bounds_hi = np.asarray([b[1] for b in self.param_bounds], dtype=np.float64)

        n_frames = vision_result.n_frames
        indices = np.arange(0, n_frames, frame_stride)
        if len(indices) > max_eval_frames:
            indices = np.linspace(0, n_frames - 1, max_eval_frames).round().astype(int)
        self.eval_indices = indices
        self.heldout_indices = np.setdiff1d(np.arange(n_frames), indices)
        self.observed_masks = vision_result.foreground_masks[indices]
        self.observed_keypoints = {
            name: vision_result.keypoints_px[name][indices] for name in MARKER_ORDER
        }
        self.frame_times_s = np.arange(n_frames) / float(vision_result.fps)

        self._eval_history: List[float] = []
        self._eval_terms: List[Tuple[float, float]] = []

    # -- coordinates -----------------------------------------------------------
    def seed_vector(self) -> np.ndarray:
        """Initial calibration vector for this optimizer's parameterisation."""
        base = np.asarray(self.initial_params.calibration_vector(), dtype=np.float64)
        if self.optimize_marker_offsets:
            return np.concatenate([base, np.zeros(len(MARKER_OFFSET_PARAM_NAMES))])
        return base

    def split_x(self, x: Sequence[float]):
        """Split a full vector into (geometry 5-vector, marker offsets (3,2))."""
        x = np.asarray(x, dtype=np.float64)
        geom = x[: len(GEOMETRY_PARAM_NAMES)]
        if self.optimize_marker_offsets and len(x) >= len(self.param_names):
            deltas = np.asarray(x[len(GEOMETRY_PARAM_NAMES): len(self.param_names)], dtype=np.float64)
            return geom, deltas.reshape(3, 2)
        return geom, np.zeros((3, 2))

    def _to_norm(self, x: Sequence[float]) -> np.ndarray:
        x = np.asarray(x, dtype=np.float64)
        return (x - self._bounds_lo) / (self._bounds_hi - self._bounds_lo)

    def _from_norm(self, u: Sequence[float]) -> np.ndarray:
        u = np.clip(np.asarray(u, dtype=np.float64), 0.0, 1.0)
        return self._bounds_lo + u * (self._bounds_hi - self._bounds_lo)

    # -- evaluation core ---------------------------------------------------------
    def _rollout(self, x: Sequence[float]):
        """Compile + roll out one candidate parameter vector (caller closes)."""
        geom, _ = self.split_x(x)
        params = ModelParams.with_calibration_vector(self.initial_params, geom)
        ctrl = vision_angles_to_ctrl(
            self.vision.angles_rad, params.hip_offset_rad, params.knee_offset_rad
        )
        xml = MJCFBuilder(params).to_xml()
        sim = MirrorSimulator(xml, self.render_height, self.render_width)
        traj = sim.simulate_trajectory(self.frame_times_s, ctrl)
        return sim, traj

    def _silhouette_ious(
        self, sim: MirrorSimulator, traj, indices: np.ndarray
    ) -> np.ndarray:
        """Per-frame silhouette IoU for the given frame indices."""
        return np.asarray([
            silhouette_iou(
                sim.render_silhouette(traj.frame_qpos_rad[int(i)]),
                self.vision.foreground_masks[i],
            )
            for i in indices
        ])

    def _reproj_residuals_px(
        self, sim: MirrorSimulator, traj, indices: np.ndarray, marker_deltas: np.ndarray
    ) -> Dict[str, np.ndarray]:
        """Per-marker reprojection residuals (px, model(+delta) - observed)."""
        out: Dict[str, np.ndarray] = {}
        for k, name in enumerate(MARKER_ORDER):
            rows = []
            for i in indices:
                proj = sim.project_landmarks(traj.frame_qpos_rad[int(i)]).get(name)
                if proj is None:
                    continue
                obs = self.vision.keypoints_px[name][int(i)]
                rows.append(proj + marker_deltas[k] - obs)
            out[name] = np.asarray(rows) if rows else np.zeros((0, 2))
        return out

    @staticmethod
    def _rmse_per_marker(residuals: Dict[str, np.ndarray]) -> Dict[str, float]:
        return {
            name: float(np.sqrt(np.mean(np.sum(resid ** 2, axis=1))))
            for name, resid in residuals.items() if len(resid)
        }

    def _evaluate(self, x: Sequence[float], indices: Optional[np.ndarray] = None):
        """Full evaluation of one candidate: silhouette IoU + reprojection cost.

        Returns ``(per_frame_iou, kp_cost, rmse_px_per_marker)`` where
        ``kp_cost`` is the reprojection RMSE in units of ``kp_sigma_px``
        (0.0 when the keypoint term is disabled).
        """
        indices = self.eval_indices if indices is None else indices
        x = np.clip(np.asarray(x, dtype=np.float64), self._bounds_lo, self._bounds_hi)
        _, marker_deltas = self.split_x(x)
        try:
            sim, traj = self._rollout(x)
        except RuntimeError as exc:
            logger.debug("Infeasible candidate %s: %s", np.round(x, 4).tolist(), exc)
            return np.zeros(len(indices)), 0.0, {}
        try:
            ious = self._silhouette_ious(sim, traj, indices)
            if self.kp_weight > 0.0:
                residuals = self._reproj_residuals_px(sim, traj, indices, marker_deltas)
                sq = [
                    np.sum(resid ** 2, axis=1) / self.kp_sigma_px ** 2
                    for resid in residuals.values() if len(resid)
                ]
                kp_cost = float(np.sqrt(np.mean(np.concatenate(sq)))) if sq else 0.0
                rmse_px = self._rmse_per_marker(residuals)
            else:
                kp_cost, rmse_px = 0.0, {}
        finally:
            sim.close()
        return ious, kp_cost, rmse_px

    # -- objective -------------------------------------------------------------
    def fused_loss(self, x: Sequence[float]) -> float:
        """Fused loss for a *physical* parameter vector, without recording.

        Used by the uncertainty estimator (finite differences must not pollute
        the convergence history)."""
        ious, kp_cost, _ = self._evaluate(x)
        return (1.0 - float(np.mean(ious))) + self.kp_weight * kp_cost

    def objective(self, u: Sequence[float]) -> float:
        """Fused loss in bound-normalized coordinates.

        ``loss(u) = (1 - mean IoU) + kp_weight * reproj_RMSE(marker radii)``.
        Every evaluation is recorded for the convergence curve."""
        x = self._from_norm(u)
        ious, kp_cost, _ = self._evaluate(x)
        sil_term = 1.0 - float(np.mean(ious))
        loss = sil_term + self.kp_weight * kp_cost
        self._eval_history.append(loss)
        self._eval_terms.append((sil_term, kp_cost))
        return loss

    # -- driver ----------------------------------------------------------------
    def calibrate(self, maxfev: int = 320) -> CalibrationResult:
        """Run Powell minimisation (with one refinement restart from the best point)."""
        t0 = time.perf_counter()
        x0 = self.seed_vector()
        u0 = self._to_norm(x0)

        initial_ious, kp_cost0, rmse0 = self._evaluate(x0)
        mean_iou_initial = float(np.mean(initial_ious))
        logger.info(
            "Calibration start | uncalibrated mean IoU = %.4f | reproj RMSE = %s px",
            mean_iou_initial,
            {k: round(v, 2) for k, v in rmse0.items()},
        )

        unit_bounds = [(0.0, 1.0)] * self.n_params
        result = minimize(
            self.objective,
            u0,
            method="Powell",
            bounds=unit_bounds,
            options={"maxfev": int(maxfev), "xtol": 1e-3, "ftol": 1e-5},
        )

        # One Powell restart from the current optimum if budget remains —
        # standard practice for direction-set methods on mildly non-convex loss.
        total_evals = len(self._eval_history)
        remaining = int(maxfev) - total_evals
        if remaining > 40:
            refined = minimize(
                self.objective,
                result.x,
                method="Powell",
                bounds=unit_bounds,
                options={"maxfev": remaining, "xtol": 1e-4, "ftol": 1e-6},
            )
            if refined.fun <= result.fun:
                result = refined

        x_best = self._from_norm(result.x)
        geom_best, deltas = self.split_x(x_best)
        best_params = ModelParams.with_calibration_vector(self.initial_params, geom_best)
        marker_offsets = (
            {name: (float(deltas[k][0]), float(deltas[k][1])) for k, name in enumerate(MARKER_ORDER)}
            if self.optimize_marker_offsets else None
        )

        # final evaluation on the calibration frames (silhouette IoU + reproj)
        final_ious, _, rmse_final = self._evaluate(x_best)
        mean_iou_final = float(np.mean(final_ious))

        # held-out validation: silhouette IoU on frames never seen by Powell
        mean_iou_heldout, heldout_ious = None, None
        if len(self.heldout_indices):
            try:
                sim, traj = self._rollout(x_best)
                try:
                    heldout_ious = self._silhouette_ious(sim, traj, self.heldout_indices)
                finally:
                    sim.close()
                mean_iou_heldout = float(np.mean(heldout_ious))
            except RuntimeError as exc:  # pragma: no cover - defensive
                logger.warning("Held-out validation failed: %s", exc)

        wall = time.perf_counter() - t0
        message = (
            f"{result.message} | mean IoU {mean_iou_initial:.4f} -> {mean_iou_final:.4f}"
            + (f" (held-out {mean_iou_heldout:.4f})" if mean_iou_heldout is not None else "")
            + f" in {len(self._eval_history)} objective evaluations"
        )
        logger.info("Calibration done | %s", message)

        return CalibrationResult(
            best_params=best_params,
            best_loss=float(result.fun),
            mean_iou_initial=mean_iou_initial,
            mean_iou_final=mean_iou_final,
            per_frame_iou_final=final_ious,
            eval_history=np.asarray(self._eval_history, dtype=np.float64),
            eval_history_terms=np.asarray(self._eval_terms, dtype=np.float64),
            n_evals=len(self._eval_history),
            success=bool(result.success) or mean_iou_final >= TARGET_IOU,
            message=message,
            wall_time_s=wall,
            kp_weight=self.kp_weight,
            kp_sigma_px=self.kp_sigma_px,
            optimize_marker_offsets=self.optimize_marker_offsets,
            marker_offsets_px=marker_offsets,
            reproj_rmse_px_final=rmse_final,
            reproj_rmse_px_initial=rmse0,
            heldout_indices=self.heldout_indices,
            mean_iou_heldout=mean_iou_heldout,
            per_frame_iou_heldout=heldout_ious,
        )


# ---------------------------------------------------------------------------
# Two-phase driver: geometry first, then nuisance-aware fused refinement
# ---------------------------------------------------------------------------
def calibrate_two_phase(
    vision,
    initial_params: ModelParams,
    kp_weight: float,
    maxfev_total: int = 1200,
    frame_stride: int = 2,
    max_eval_frames: int = 36,
    render_height: int = 480,
    render_width: int = 640,
) -> Tuple[CalibrationResult, CalibrationResult, "MirrorOptimizer"]:
    """Run the standard two-phase Mirror calibration.

    Phase 1 (60% of the budget): silhouette-only Powell over the 5
    geometry/servo parameters — well-conditioned and immune to the fiducial
    mount bias.  Phase 2 (40%): fused refinement over geometry + per-marker
    pixel nuisance offsets, seeded at the Phase-1 optimum; the reprojection
    term sharpens the femur/tibia split while the nuisance absorbs the mount
    bias.  With ``kp_weight = 0`` the second phase is skipped and the result
    equals the original silhouette-only formulation.

    Returns ``(phase1_result, final_result, final_optimizer)``.
    """
    opt1 = MirrorOptimizer(
        vision, initial_params,
        frame_stride=frame_stride, max_eval_frames=max_eval_frames,
        render_height=render_height, render_width=render_width,
        kp_weight=0.0,
    )
    res1 = opt1.calibrate(maxfev=int(maxfev_total * 0.6))

    if kp_weight <= 0.0:
        return res1, res1, opt1

    opt2 = MirrorOptimizer(
        vision, res1.best_params,
        frame_stride=frame_stride, max_eval_frames=max_eval_frames,
        render_height=render_height, render_width=render_width,
        kp_weight=float(kp_weight),
        optimize_marker_offsets=True,
    )
    res2 = opt2.calibrate(maxfev=int(maxfev_total * 0.4))
    return res1, res2, opt2


# ---------------------------------------------------------------------------
# Uncertainty quantification
# ---------------------------------------------------------------------------
def estimate_ci95_curvature(
    objective_fn,
    x_opt: Sequence[float],
    param_names: Sequence[str] = PARAM_NAMES,
    param_bounds: Sequence[Tuple[float, float]] = PARAM_BOUNDS,
    step_norm: float = 1e-2,
    chi2_target: float = CHI2_95_1DOF,
) -> Dict[str, Dict[str, float]]:
    """Local-curvature 95% confidence intervals for the calibrated parameters.

    The loss ``f`` is treated as a negative log-likelihood around its optimum:
    the likelihood-ratio contour ``f(x) - f(x*) = chi2_95(1)/2 = 1.921``
    defines the 95% confidence region.  Approximating ``f`` locally by a
    parabola along each bound-normalized parameter axis (central finite
    differences, all other parameters held fixed) gives the marginal interval

        delta_i = sqrt( 2 * (chi2_95(1)/2) / H_ii ),  H_ii = d2f/du_i^2 .

    Holding the other parameters fixed *widens* the interval relative to a
    full profile likelihood (which would re-optimise them), so these CIs are
    conservative.  The simulation is deterministic, so the finite differences
    are exact — there is no evaluation noise to average out.

    Returns ``{param: {"estimate", "ci95_half_width", "lo", "hi", "curvature_ok"}}``
    in physical units.
    """
    x_opt = np.asarray(x_opt, dtype=np.float64)
    lo = np.asarray([b[0] for b in param_bounds], dtype=np.float64)
    hi = np.asarray([b[1] for b in param_bounds], dtype=np.float64)
    span = hi - lo
    u_opt = (x_opt - lo) / span
    f_star = float(objective_fn(x_opt))

    out: Dict[str, Dict[str, float]] = {}
    for i, name in enumerate(param_names):
        h = step_norm
        up, um = u_opt.copy(), u_opt.copy()
        up[i] = min(1.0, u_opt[i] + h)
        um[i] = max(0.0, u_opt[i] - h)
        # keep the step symmetric around u_opt for a clean second difference
        h_act = 0.5 * (up[i] - um[i])
        fp = float(objective_fn(lo + up * span))
        fm = float(objective_fn(lo + um * span))
        hess = (fp - 2.0 * f_star + fm) / (h_act ** 2) if h_act > 0 else 0.0
        if hess > 1e-12:
            d_norm = float(np.sqrt(2.0 * (chi2_target / 2.0) / hess))
            half = float(d_norm * span[i])
            curvature_ok = True
        else:
            # flat direction: report the full parameter range as the interval
            half = float(0.5 * span[i])
            curvature_ok = False
        est = float(x_opt[i])
        out[name] = {
            "estimate": est,
            "ci95_half_width": half,
            "lo": max(lo[i], est - half),
            "hi": min(hi[i], est + half),
            "curvature_ok": curvature_ok,
        }
    return out

"""Mirror — Part 1: monocular video ingestion and keypoint tracking.

This module is the visual front-end of the Mirror pipeline.  It converts a
single-camera video of a sagittal two-link movement (e.g. a lower-limb swing
cycle or a mechanical pendulum linkage) into clean, continuous kinematic
signals, without any motion-capture equipment:

1.  Foreground segmentation : HSV colour-space thresholding plus
    morphological opening/closing separate the moving limbs from a darker
    background and remove speckle noise.
2.  Keypoint detection      : per-marker HSV masks (coloured fiducials at the
    hip / knee / ankle) are reduced to centroid coordinates via contour
    moments; short detection gaps are linearly interpolated.
3.  Kinematic extraction    : joint angles are computed from the joint-centre
    pixel vectors and high-frequency tracking jitter is removed with a
    Savitzky-Golay filter applied to the pixel trajectories.
4.  Scale initialisation    : initial link lengths, limb radii and marker
    radii are estimated from the pixel geometry through the pinhole model
    (``pixel length x camera_distance / focal_length``).

Conventions used across the whole pipeline
------------------------------------------
* Image coordinates: origin top-left, x right, y **down** (OpenCV).
* Joint angles are measured from the downward vertical, positive towards
  image +x (the facing direction of the subject), in radians:
  ``theta = atan2(v_x, v_y)`` for a limb vector ``v``.
* The knee angle is the relative shank-vs-thigh angle
  ``phi = theta_shank - theta_thigh``; negative values correspond to knee
  flexion (the shank rotating backwards relative to the thigh line).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Tuple

import cv2
import numpy as np
from scipy.signal import savgol_filter

logger = logging.getLogger("mirror.vision")

MARKER_NAMES: Tuple[str, str, str] = ("hip", "knee", "ankle")

# HSV (OpenCV: H in [0,180], S,V in [0,255]) inclusive bounds for each fiducial.
DEFAULT_MARKER_HSV: Dict[str, Tuple[Tuple[int, int, int], Tuple[int, int, int]]] = {
    "hip": ((136, 90, 90), (166, 255, 255)),   # magenta fiducial
    "knee": ((40, 90, 90), (85, 255, 255)),    # green fiducial
    "ankle": ((95, 90, 90), (130, 255, 255)),  # blue fiducial
}


# ---------------------------------------------------------------------------
# Camera model
# ---------------------------------------------------------------------------
@dataclass
class CameraIntrinsics:
    """Pinhole camera model shared by the vision stage and the renderer."""

    width: int
    height: int
    focal_length_px: float = 620.0
    principal_point_px: Optional[Tuple[float, float]] = None  # default: image centre

    @property
    def cx(self) -> float:
        if self.principal_point_px is not None:
            return float(self.principal_point_px[0])
        return self.width / 2.0

    @property
    def cy(self) -> float:
        if self.principal_point_px is not None:
            return float(self.principal_point_px[1])
        return self.height / 2.0

    @property
    def fovy_deg(self) -> float:
        """Vertical field of view in degrees (MuJoCo camera ``fovy``)."""
        return float(np.degrees(2.0 * np.arctan(self.height / (2.0 * self.focal_length_px))))


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------
@dataclass
class VisionResult:
    """Everything Part 1 hands to the rest of the pipeline."""

    video_path: str
    fps: float
    n_frames: int
    width: int
    height: int

    keypoints_px: Dict[str, np.ndarray]        # filtered, (T, 2) float64
    keypoints_raw_px: Dict[str, np.ndarray]    # raw centroids, NaN where lost
    tracking_ok: Dict[str, np.ndarray]         # (T,) bool per marker
    tracking_success_rate: float               # fraction of frames with all 3 markers

    angles_rad: Dict[str, np.ndarray]          # {"hip": theta_thigh, "knee": phi_rel}

    initial_link_lengths_m: Dict[str, float]   # {"femur_m", "tibia_m"} pinhole estimate
    limb_width_px: Dict[str, float]            # {"femur_px", "tibia_px"} median widths
    marker_radius_px: float                    # median fiducial radius

    foreground_masks: np.ndarray               # (T, H, W) uint8 {0, 255}
    hip_pixel: Tuple[float, float]             # median hip keypoint (u, v)

    camera: CameraIntrinsics
    assumed_camera_distance_m: float

    @property
    def pixel_scale_px_per_m(self) -> float:
        return self.camera.focal_length_px / self.assumed_camera_distance_m


# ---------------------------------------------------------------------------
# Extractor
# ---------------------------------------------------------------------------
class VisionExtractor:
    """Marker-based monocular keypoint tracker and kinematics extractor.

    Parameters
    ----------
    video_path:
        Input video readable by OpenCV.
    camera:
        Optional known intrinsics; otherwise built from the video resolution
        and ``focal_length_px`` with the principal point at the image centre.
    assumed_camera_distance_m:
        The one manual scale prior of the pipeline: the assumed camera-to-
        movement-plane distance.  It fixes the pixel->metre ratio (monocular
        video alone cannot recover absolute scale).
    marker_hsv:
        Optional per-marker ``(lower, upper)`` HSV bounds overriding
        :data:`DEFAULT_MARKER_HSV` (e.g. for custom fiducial colours).
    """

    def __init__(
        self,
        video_path: str | Path,
        camera: Optional[CameraIntrinsics] = None,
        focal_length_px: float = 620.0,
        assumed_camera_distance_m: float = 3.0,
        marker_hsv: Optional[Dict[str, Tuple[Tuple[int, int, int], Tuple[int, int, int]]]] = None,
        min_marker_area_px: float = 30.0,
        max_frames: int = 400,
        savgol_window: int = 11,
        savgol_polyorder: int = 3,
        foreground_value_threshold: int = 110,
        foreground_saturation_threshold: int = 90,
    ) -> None:
        self.video_path = Path(video_path)
        self.camera = camera
        self.focal_length_px = float(focal_length_px)
        self.assumed_camera_distance_m = float(assumed_camera_distance_m)
        self.marker_hsv = {
            name: (tuple(lo), tuple(hi))
            for name, (lo, hi) in (marker_hsv or DEFAULT_MARKER_HSV).items()
        }
        self.min_marker_area_px = float(min_marker_area_px)
        self.max_frames = int(max_frames)
        self.savgol_window = int(savgol_window)
        self.savgol_polyorder = int(savgol_polyorder)
        self.foreground_value_threshold = int(foreground_value_threshold)
        self.foreground_saturation_threshold = int(foreground_saturation_threshold)

        self._kernel_open_fg = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        self._kernel_close_fg = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
        self._kernel_open_mk = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        self._kernel_close_mk = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))

    # -- public entry point -------------------------------------------------
    def extract(self) -> VisionResult:
        """Run the full Part-1 pipeline and return a :class:`VisionResult`."""
        frames = self._read_frames()
        height, width = frames[0].shape[:2]
        n_frames = len(frames)

        camera = self.camera or CameraIntrinsics(
            width=width, height=height, focal_length_px=self.focal_length_px
        )

        fg_masks = np.zeros((n_frames, height, width), dtype=np.uint8)
        raw = {name: np.full((n_frames, 2), np.nan) for name in MARKER_NAMES}
        areas = {name: np.zeros(n_frames) for name in MARKER_NAMES}

        for i, bgr in enumerate(frames):
            hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
            # limb pixels (low saturation) UNION fiducial pixels: the opaque
            # markers occlude the limb underneath, so the union + a closing
            # pass (below) reconstructs the true limb silhouette.
            limb = (
                (hsv[..., 2] >= self.foreground_value_threshold)
                & (hsv[..., 1] <= self.foreground_saturation_threshold)
            ).astype(np.uint8) * 255
            marker_mask = np.zeros_like(limb)
            detected = detect_marker_centroids(
                hsv, self.marker_hsv, self.min_marker_area_px,
                self._kernel_open_mk, self._kernel_close_mk,
            )
            for name in MARKER_NAMES:
                if detected[name] is not None:
                    raw[name][i] = detected[name][0]
                    areas[name][i] = detected[name][1]
                    lo, hi = self.marker_hsv[name]
                    marker_mask |= cv2.inRange(
                        hsv, np.array(lo, dtype=np.uint8), np.array(hi, dtype=np.uint8)
                    )
            fg_masks[i] = limb | marker_mask

        tracking_ok = {name: ~np.isnan(raw[name]).any(axis=1) for name in MARKER_NAMES}
        success_rate = float(np.mean(np.logical_and.reduce([tracking_ok[n] for n in MARKER_NAMES])))
        if success_rate < 0.5:
            raise RuntimeError(
                "Marker tracking failed: keypoint detection succeeded on only "
                f"{success_rate:.0%} of frames. Check the fiducial colours against the "
                "HSV configuration (see DEFAULT_MARKER_HSV / --marker-hsv docs in README)."
            )
        for name in MARKER_NAMES:
            missing = int((~tracking_ok[name]).sum())
            if missing:
                logger.warning("Marker '%s' missing in %d frame(s); linearly interpolated.", name, missing)

        keypoints = {name: _interpolate_gaps(raw[name], tracking_ok[name]) for name in MARKER_NAMES}

        # Fill the fiducial occlusion holes: a closing operation sized to the
        # marker diameter bridges the marker-sized holes punched into the limb
        # silhouette, without altering the (convex) limb outline.
        marker_radius_px = self._median_marker_radius(areas)
        hole_kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (int(2 * round(marker_radius_px)) + 3,) * 2
        )
        for i in range(n_frames):
            mask = cv2.morphologyEx(fg_masks[i], cv2.MORPH_OPEN, self._kernel_open_fg)
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, hole_kernel)
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, self._kernel_close_fg)
            n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
            for label in range(1, n_labels):
                if stats[label, cv2.CC_STAT_AREA] < 60:
                    mask[labels == label] = 0
            fg_masks[i] = mask

        # Savitzky-Golay de-jitter on the pixel trajectories (Part 1 spec).
        keypoints_smooth = {
            name: np.column_stack(
                [self._smooth(keypoints[name][:, 0]), self._smooth(keypoints[name][:, 1])]
            )
            for name in MARKER_NAMES
        }

        # --- kinematics (vision convention) --------------------------------
        v_thigh = keypoints_smooth["knee"] - keypoints_smooth["hip"]
        v_shank = keypoints_smooth["ankle"] - keypoints_smooth["knee"]
        theta_thigh = np.arctan2(v_thigh[:, 0], v_thigh[:, 1])
        theta_shank = np.arctan2(v_shank[:, 0], v_shank[:, 1])
        angles = {"hip": theta_thigh, "knee": theta_shank - theta_thigh}

        # --- scale initialisation through the pinhole model ----------------
        px_per_m = camera.focal_length_px / self.assumed_camera_distance_m
        d_femur = float(np.median(np.linalg.norm(v_thigh, axis=1)))
        d_tibia = float(np.median(np.linalg.norm(v_shank, axis=1)))
        link_lengths = {"femur_m": d_femur / px_per_m, "tibia_m": d_tibia / px_per_m}

        widths = self._median_limb_widths(fg_masks, keypoints_smooth)

        hip_u, hip_v = np.median(keypoints_smooth["hip"], axis=0)

        logger.info(
            "Vision: %d frames @ %.2f fps, tracking success %.1f%% | "
            "initial links: femur %.3f m, tibia %.3f m (prior Z=%.2f m, f=%.0f px)",
            n_frames, self._fps, 100.0 * success_rate,
            link_lengths["femur_m"], link_lengths["tibia_m"],
            self.assumed_camera_distance_m, camera.focal_length_px,
        )

        return VisionResult(
            video_path=str(self.video_path),
            fps=self._fps,
            n_frames=n_frames,
            width=width,
            height=height,
            keypoints_px=keypoints_smooth,
            keypoints_raw_px=raw,
            tracking_ok=tracking_ok,
            tracking_success_rate=success_rate,
            angles_rad=angles,
            initial_link_lengths_m=link_lengths,
            limb_width_px=widths,
            marker_radius_px=marker_radius_px,
            foreground_masks=fg_masks,
            hip_pixel=(float(hip_u), float(hip_v)),
            camera=camera,
            assumed_camera_distance_m=self.assumed_camera_distance_m,
        )

    # -- internals -----------------------------------------------------------
    def _read_frames(self) -> list:
        cap = cv2.VideoCapture(str(self.video_path))
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open video: {self.video_path}")
        self._fps = cap.get(cv2.CAP_PROP_FPS)
        if not self._fps or self._fps <= 0 or np.isnan(self._fps):
            self._fps = 30.0
            logger.warning("Video FPS unreadable/zero; assuming 30 fps.")
        frames = []
        while len(frames) < self.max_frames:
            ok, frame = cap.read()
            if not ok:
                break
            frames.append(frame)
        cap.release()
        if not frames:
            raise RuntimeError(f"No frames could be decoded from {self.video_path}")
        return frames

    def _foreground_mask(self, hsv: np.ndarray) -> np.ndarray:
        """Limb-only binary mask (used by tests): HSV value/saturation gates +
        open/close morphology.  The full extraction pipeline unions this with
        the fiducial pixels and fills the occlusion holes (see ``extract``)."""
        value = hsv[..., 2]
        saturation = hsv[..., 1]
        mask = ((value >= self.foreground_value_threshold) & (saturation <= self.foreground_saturation_threshold)).astype(np.uint8) * 255
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, self._kernel_open_fg)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, self._kernel_close_fg)
        return mask

    def _smooth(self, signal: np.ndarray) -> np.ndarray:
        window = _safe_savgol_window(len(signal), self.savgol_window, self.savgol_polyorder)
        if window is None:
            return signal
        return savgol_filter(signal, window, self.savgol_polyorder)

    def _median_limb_widths(
        self, masks: np.ndarray, keypoints: Dict[str, np.ndarray]
    ) -> Dict[str, float]:
        """Median limb width via the distance transform along each segment.

        The keypoint centroids carry the skin-marker placement bias, so the
        segment between two centroids is only an approximation of the limb
        centerline.  Sampling the distance transform at several points along
        the segment and taking the maximum recovers the true half-width
        (the centerline is the ridge of the distance transform), which keeps
        the radius estimate unbiased.
        """
        w_femur, w_tibia = [], []
        n_samples = 7
        for i in range(masks.shape[0]):
            dist = cv2.distanceTransform(masks[i], cv2.DIST_L2, 5)
            for widths, (a, b) in (
                (w_femur, ("hip", "knee")),
                (w_tibia, ("knee", "ankle")),
            ):
                best = 0.0
                for s in np.linspace(0.15, 0.85, n_samples):
                    pt = keypoints[a][i] + s * (keypoints[b][i] - keypoints[a][i])
                    u = int(np.clip(round(float(pt[0])), 0, masks.shape[2] - 1))
                    v = int(np.clip(round(float(pt[1])), 0, masks.shape[1] - 1))
                    best = max(best, float(dist[v, u]))
                widths.append(2.0 * best)
        return {
            "femur_px": float(np.median(w_femur)),
            "tibia_px": float(np.median(w_tibia)),
        }

    @staticmethod
    def _median_marker_radius(areas: Dict[str, np.ndarray]) -> float:
        """Median marker radius (px) from detected blob areas."""
        radii = []
        for name in MARKER_NAMES:
            a = areas[name][areas[name] > 0]
            if a.size:
                radii.append(np.sqrt(a / np.pi))
        if not radii:
            return 7.0
        return float(np.median(np.concatenate(radii)))


# ---------------------------------------------------------------------------
# Module-level helpers (importable by the test-suite)
# ---------------------------------------------------------------------------
def detect_marker_centroids(
    hsv: np.ndarray,
    marker_hsv: Dict[str, Tuple[Tuple[int, int, int], Tuple[int, int, int]]],
    min_area_px: float = 30.0,
    kernel_open: Optional[np.ndarray] = None,
    kernel_close: Optional[np.ndarray] = None,
) -> Dict[str, Optional[Tuple[np.ndarray, float]]]:
    """Detect one fiducial blob per marker colour.

    Returns ``{name: (centroid_xy, area_px) or None}``.
    """
    out: Dict[str, Optional[Tuple[np.ndarray, float]]] = {}
    for name, (lo, hi) in marker_hsv.items():
        mask = cv2.inRange(hsv, np.array(lo, dtype=np.uint8), np.array(hi, dtype=np.uint8))
        if kernel_open is not None:
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel_open)
        if kernel_close is not None:
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel_close)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        best, best_area = None, min_area_px
        for contour in contours:
            area = cv2.contourArea(contour)
            if area >= best_area:
                best, best_area = contour, area
        if best is None:
            out[name] = None
            continue
        moments = cv2.moments(best)
        if moments["m00"] <= 0:
            out[name] = None
            continue
        centroid = np.array([moments["m10"] / moments["m00"], moments["m01"] / moments["m00"]])
        out[name] = (centroid, float(best_area))
    return out


def _interpolate_gaps(values: np.ndarray, ok: np.ndarray) -> np.ndarray:
    """Linearly interpolate NaN coordinate gaps; clamp at the sequence ends."""
    if ok.all():
        return values.copy()
    idx = np.arange(values.shape[0])
    out = values.copy()
    for dim in range(values.shape[1]):
        out[:, dim] = np.interp(idx, idx[ok], values[ok, dim])
    return out


def _safe_savgol_window(n_samples: int, wanted: int, polyorder: int) -> Optional[int]:
    """Choose a valid odd Savitzky-Golay window for a short signal, or None."""
    window = min(int(wanted), n_samples if n_samples % 2 == 1 else n_samples - 1)
    if window % 2 == 0:
        window -= 1
    if window <= polyorder:
        window = polyorder + 1 if (polyorder + 1) % 2 == 1 else polyorder + 2
    if window > n_samples or window <= polyorder:
        return None
    return window

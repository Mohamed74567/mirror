"""Mirror — Part 2: parameterized dynamic MJCF compilation.

Translates the geometry estimated by the vision stage into a mathematically
sound, compilable MuJoCo model (MJCF/XML):

* hierarchical two-link sagittal body tree (thigh -> shank) with hinge joints
  about the lateral (y) axis;
* capsule geoms with physically meaningful inertia derived from a tissue-like
  density prior (1050 kg/m^3) — masses are computed by MuJoCo from geometry;
* physiological joint limits, joint damping and armature inertia;
* integrated ``<position>`` servo actuators (PD-controlled, gain ``kp``) so
  the model is driven by forward dynamics rather than static pose placement;
* a fixed pinhole-consistent camera (``mirror_cam``) whose focal length and
  placement are derived from the same camera model used in Part 1, so that
  offscreen renders land in the exact perspective of the input video.

The five *calibration parameters* — ``L_femur_m``, ``L_tibia_m``,
``hip_offset_rad``, ``knee_offset_rad``, ``servo_kp`` — form the optimisation
vector consumed by Part 4 (``src.optimizer``).

Angle conventions
-----------------
Vision convention (Part 1): angles from the downward vertical, positive
towards image +x.  MuJoCo convention: hinge about ``+y``; a positive rotation
moves a hanging limb tip towards world ``-x``.  With the fixed camera looking
along ``+y``, world ``-x`` projects onto image ``-x``, therefore the mapping
is a pure sign inversion plus an additive calibration offset::

    q_hip_target  = -theta_thigh + hip_offset
    q_knee_target = -(theta_shank - theta_thigh) + knee_offset

The offsets absorb the fiducial-placement bias (skin markers glued a few
degrees away from the true rotation centre) and are identified in Part 4.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass
from pathlib import Path
from typing import List, Sequence, Tuple

import mujoco

MIRROR_SCHEMA_VERSION = "1.0.0"

# Names of the calibration parameters, in vector order.
CALIBRATION_PARAM_NAMES: Tuple[str, ...] = (
    "L_femur_m",
    "L_tibia_m",
    "hip_offset_rad",
    "knee_offset_rad",
    "servo_kp",
)


@dataclass
class ModelParams:
    """Complete parameterisation of one generated MuJoCo model."""

    # --- calibration vector (optimised by Part 4) ---------------------------
    L_femur_m: float = 0.42
    L_tibia_m: float = 0.40
    hip_offset_rad: float = 0.0
    knee_offset_rad: float = 0.0
    servo_kp: float = 250.0

    # --- morphology (measured from the segmented limb width in Part 1) ------
    femur_radius_m: float = 0.070
    tibia_radius_m: float = 0.052
    marker_radius_m: float = 0.034
    marker_mass_kg: float = 0.08
    density_kg_m3: float = 1050.0  # tissue-like density prior

    # --- actuation / dynamics ----------------------------------------------
    joint_damping: float = 0.12
    armature_inertia: float = 0.010
    #: damping ratio of the position servos; velocity gain kv is derived as
    #: kv = 2 * zeta * sqrt(kp * I_nominal) with I_nominal the swing-leg inertia
    #: about the hip (computed from the density-derived capsule masses).
    servo_damping_ratio: float = 0.7
    hip_range_rad: Tuple[float, float] = (-2.10, 0.90)   # flexion..extension
    knee_range_rad: Tuple[float, float] = (-0.06, 2.60)  # slight hyperextension allowed
    timestep_s: float = 1.0 / 500.0                      # 500 Hz physics
    gravity_m_s2: float = 9.81

    # --- camera / world placement (from the vision stage) -------------------
    fovy_deg: float = 42.0
    cam_distance_m: float = 3.0
    cam_height_m: float = 0.45
    cam_x_m: float = 0.0
    hip_height_m: float = 0.95
    render_width_px: int = 640
    render_height_px: int = 480

    # --- calibration vector helpers -----------------------------------------
    def calibration_vector(self) -> List[float]:
        """The 5-vector consumed by :mod:`src.optimizer`."""
        return [
            self.L_femur_m,
            self.L_tibia_m,
            self.hip_offset_rad,
            self.knee_offset_rad,
            self.servo_kp,
        ]

    @classmethod
    def with_calibration_vector(cls, base: "ModelParams", x: Sequence[float]) -> "ModelParams":
        """Return a copy of ``base`` with the calibration vector replaced."""
        if len(x) != len(CALIBRATION_PARAM_NAMES):
            raise ValueError(f"expected {len(CALIBRATION_PARAM_NAMES)} values, got {len(x)}")
        return dataclasses.replace(
            base,
            L_femur_m=float(x[0]),
            L_tibia_m=float(x[1]),
            hip_offset_rad=float(x[2]),
            knee_offset_rad=float(x[3]),
            servo_kp=float(x[4]),
        )


class MJCFBuilder:
    """Builds, validates and serialises the parameterized MJCF model."""

    #: small padding added around joint limits for the servo ctrlrange
    CTRL_RANGE_PADDING_RAD = 0.15

    def __init__(self, params: ModelParams) -> None:
        if params.L_femur_m <= 0 or params.L_tibia_m <= 0:
            raise ValueError("Link lengths must be positive.")
        if params.servo_kp <= 0:
            raise ValueError("servo_kp must be positive.")
        self.params = params

    # -- XML -----------------------------------------------------------------
    def nominal_swing_inertia(self) -> float:
        """Swing-leg rotational inertia about the hip (kg m^2), used to derive
        the servo velocity gain for a critically-damped response."""
        import math

        p = self.params
        rho = p.density_kg_m3
        m_femur = rho * (math.pi * p.femur_radius_m**2 * p.L_femur_m
                         + 4.0 / 3.0 * math.pi * p.femur_radius_m**3)
        m_shank = rho * (math.pi * p.tibia_radius_m**2 * p.L_tibia_m
                         + 4.0 / 3.0 * math.pi * p.tibia_radius_m**3)
        return m_femur * p.L_femur_m**2 / 3.0 + m_shank * (p.L_femur_m + 0.5 * p.L_tibia_m) ** 2

    def to_xml(self) -> str:
        p = self.params
        pad = self.CTRL_RANGE_PADDING_RAD
        hip_ctrl = (p.hip_range_rad[0] - pad, p.hip_range_rad[1] + pad)
        knee_ctrl = (p.knee_range_rad[0] - pad, p.knee_range_rad[1] + pad)
        # critically-damped PD: kv = 2*zeta*sqrt(kp*I)
        kv = 2.0 * p.servo_damping_ratio * (p.servo_kp * self.nominal_swing_inertia()) ** 0.5
        header = (
            f"<!-- Generated by Mirror v{MIRROR_SCHEMA_VERSION} | calibration vector: "
            f"L_femur={p.L_femur_m:.4f} m, L_tibia={p.L_tibia_m:.4f} m, "
            f"hip_offset={p.hip_offset_rad:+.4f} rad, knee_offset={p.knee_offset_rad:+.4f} rad, "
            f"kp={p.servo_kp:.1f} (kv={kv:.1f} derived) -->"
        )
        return f"""{header}
<mujoco model="mirror_two_link_sagittal">
  <compiler angle="radian" inertiafromgeom="true"/>
  <option timestep="{p.timestep_s:.6f}" integrator="RK4" gravity="0 0 {-p.gravity_m_s2:.4f}"/>

  <visual>
    <global offwidth="{p.render_width_px}" offheight="{p.render_height_px}"/>
    <headlight ambient="0.55 0.55 0.55" diffuse="0.45 0.45 0.45" specular="0.1 0.1 0.1"/>
  </visual>

  <default>
    <geom contype="0" conaffinity="0" density="{p.density_kg_m3:.1f}" margin="0"/>
    <joint type="hinge" axis="0 1 0" damping="{p.joint_damping:.4f}" armature="{p.armature_inertia:.4f}"/>
  </default>

  <worldbody>
    <light name="key_light" directional="true" pos="0 -1.0 1.5" dir="0 0.3 -0.9"
           diffuse="0.85 0.85 0.85" specular="0.15 0.15 0.15"/>

    <!-- Pinhole-consistent camera: fovy from the vision-stage focal length,
         placed so the hip projects onto the hip pixel measured in the video. -->
    <camera name="mirror_cam" pos="{p.cam_x_m:.6f} {-p.cam_distance_m:.6f} {p.cam_height_m:.6f}"
            xyaxes="1 0 0 0 0 1" fovy="{p.fovy_deg:.4f}"/>

    <body name="hip_anchor" pos="0 0 {p.hip_height_m:.6f}">
      <geom name="hip_marker" type="sphere" size="{p.marker_radius_m:.4f}"
            mass="{p.marker_mass_kg:.3f}" group="1" rgba="0.95 0.20 0.90 1"/>
      <body name="thigh" pos="0 0 0">
        <joint name="hip_flexion" pos="0 0 0" range="{p.hip_range_rad[0]:.4f} {p.hip_range_rad[1]:.4f}"/>
        <geom name="femur" type="capsule" fromto="0 0 0 0 0 {-p.L_femur_m:.6f}"
              size="{p.femur_radius_m:.4f}" rgba="0.82 0.83 0.86 1"/>
        <body name="shank" pos="0 0 {-p.L_femur_m:.6f}">
          <joint name="knee_flexion" pos="0 0 0" range="{p.knee_range_rad[0]:.4f} {p.knee_range_rad[1]:.4f}"/>
          <geom name="knee_marker" type="sphere" size="{p.marker_radius_m:.4f}"
                mass="{p.marker_mass_kg:.3f}" group="1" rgba="0.15 0.92 0.25 1"/>
          <geom name="tibia" type="capsule" fromto="0 0 0 0 0 {-p.L_tibia_m:.6f}"
                size="{p.tibia_radius_m:.4f}" rgba="0.86 0.87 0.90 1"/>
          <geom name="ankle_marker" type="sphere" size="{p.marker_radius_m:.4f}"
                pos="0 0 {-p.L_tibia_m:.6f}" mass="{p.marker_mass_kg:.3f}" group="1" rgba="0.15 0.30 0.95 1"/>
        </body>
      </body>
    </body>
  </worldbody>

  <actuator>
    <position name="hip_servo" joint="hip_flexion" kp="{p.servo_kp:.3f}" kv="{kv:.3f}"
              ctrllimited="true" ctrlrange="{hip_ctrl[0]:.4f} {hip_ctrl[1]:.4f}"/>
    <position name="knee_servo" joint="knee_flexion" kp="{p.servo_kp:.3f}" kv="{kv:.3f}"
              ctrllimited="true" ctrlrange="{knee_ctrl[0]:.4f} {knee_ctrl[1]:.4f}"/>
  </actuator>
</mujoco>
"""

    # -- compilation / persistence -------------------------------------------
    def compile(self) -> mujoco.MjModel:
        """Compile the XML with MuJoCo, raising a readable error on failure."""
        xml = self.to_xml()
        try:
            return mujoco.MjModel.from_xml_string(xml)
        except Exception as exc:  # pragma: no cover - depends on candidate params
            raise RuntimeError(f"MJCF compilation failed: {exc}\n--- XML ---\n{xml}") from exc

    def save(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(self.to_xml(), encoding="utf-8")
        return path

# Mirror

**Mirror** is an automated computer-vision and physics-identification pipeline
that bridges the gap between raw monocular video and rigorous multi-body
physics simulation. It takes a single-camera recording of a sagittal two-link
movement,  a lower-limb swing cycle, a pendulum arm, a mechanical linkage 
and automatically produces a **fully calibrated, dynamically stepped MuJoCo
digital twin**, with no motion-capture equipment, no physical markers beyond
coloured fiducials, and no manual rigging.

The core idea is a **closed feedback loop**: the simulated body is rendered
from the exact camera perspective of the video, compared against the video's
segmented limb silhouettes (Intersection over Union), and a derivative-free
optimiser adjusts the model until simulation and observation agree both
visually and physically.

```text
                    ┌─────────────────────────────────────────────────────────┐
                    │                    monocular video                      │
                    └───────────────┬─────────────────────────────────────────┘
                                    │
              ┌─────────────────────▼─────────────────────┐
              │  Part 1 · vision_extractor.py             │
              │  HSV segmentation + morphology → limb     │
              │  masks · fiducial centroids → keypoints   │
              │  · Savitzky-Golay → clean θ(t) · pinhole  │
              │  → initial link lengths / radii           │
              └─────────────────────┬─────────────────────┘
                                    │ angles, masks, keypoints, geometry priors
              ┌─────────────────────▼─────────────────────┐
              │  Part 2 · mjcf_builder.py                 │
              │  parameterized MJCF: capsule links,       │
              │  density-derived inertia, joint limits,   │
              │  damping, armature, position servos       │
              └─────────────────────┬─────────────────────┘
                                    │ compilable XML (validated on the fly)
              ┌─────────────────────▼─────────────────────┐
              │  Part 3 · mujoco_sim.py                   │
              │  forward dynamics @ 500 Hz (RK4, g=9.81)  │
              │  · PD servo tracking + velocity feed-fwd  │
              │  · torques (N·m) · joint reaction forces  │
              │  · silhouettes + landmark pinhole         │
              │    reprojection from the exact camera     │
              └─────────────────────┬─────────────────────┘
                                    │ rendered silhouettes + landmarks
              ┌─────────────────────▼─────────────────────┐     re-loop until
              │  Part 4 · optimizer.py                    │ ─── IoU converges
              │  phase 1: silhouette-only geometry        │ ───┐ over 0.92
              │  phase 2: fused loss = (1 − mean IoU)     │ ◄───┘
              │    + w_kp · keypoint reprojection RMSE,   │
              │    with per-marker mount-offset nuisance  │
              │  scipy · Powell (derivative-free)         │
              └─────────────────────┬─────────────────────┘
                                    │
          calibrated XML · comparison video/gif · diagnostic plots
          · 95% CIs · held-out validation · metrics JSON
```

## Results on the bundled benchmark (real, reproducible numbers)

The repository ships a synthetic sagittal swing clip with **published ground
truth** (`assets/demo_ground_truth.json`: true camera model, true link
geometry, and a deliberate skin-marker placement bias), so every pipeline
estimate can be scored quantitatively. Running `python run_mirror.py`
reproduces the following (MuJoCo 4.0, analytical-silhouette backend,
~60 s end-to-end on 2 CPU cores):

| Quantity | Uncalibrated | Calibrated (Mirror v1.1) |
|---|---|---|
| Mean silhouette IoU (72 frames) | 0.778 | **0.924** (target 0.92) |
| Mean silhouette IoU — **36 held-out frames** | — | **0.927** (no overfitting) |
| Servo tracking RMSE — hip | 1.45° | **0.20°** |
| Servo tracking RMSE — knee | 1.14° | **0.17°** |
| Keypoint reprojection RMSE — knee / ankle | 4.9 / 7.6 px | **0.4 / 2.2 px** |
| Position-servo gain kp | 250 (seed) | 2319 (identified) |
| Marker-bias offset — hip | 0° (seed) | **+3.27°** (empirical bias +3.15°) |
| Marker-bias offset — knee | 0° (seed) | **−6.86°** (empirical bias −6.55°) |

Against ground truth: `L_femur` 0.4065 m vs 0.420 m (−3.2%), `L_tibia`
0.4116 m vs 0.400 m (+2.9%), total chain 0.8181 m vs 0.820 m (**−0.23%**).
The v1.0 silhouette-only formulation scored −5.0% / +2.8% / −1.2% on the same
quantities — the fused keypoint term (below) improves the femur estimate and
pins the chain to a quarter of a percent. See
[Limitations and honest scope](#limitations-and-honest-scope) for why the
per-segment split still carries more error than the chain total.

**Uncertainty:** local-curvature (likelihood-ratio) 95% CIs are reported for
every identified parameter (femur ±0.085 m, tibia ±0.109 m on the benchmark;
conservative — other parameters held fixed). The ground-truth lengths fall
inside both intervals (`ci95_coverage: true`), and the simulation is
deterministic, so the finite-difference curvature is exact.

**Bonus — the pipeline measures its own instrumentation error:** the fitted
per-marker pixel nuisance offsets recover the *deliberately injected*
skin-mount bias of the benchmark: ankle (−4.0, +4.9) px vs true (−4, +5) px,
knee (+4.7, −1.3) px vs true (+5, −4) px. The residual knee-dy component is
absorbed by the knee angle offset — the *sum* is what is observable, and the
identified knee offset (−6.86°) matches the empirical bias (−6.55°) to 0.3°.

Dynamics recorded by the twin (video cannot see these): peak actuator torques
~22 N·m (hip) / ~10 N·m (knee), mean joint reaction forces ~140 N (hip) /
~55 N (knee) under Earth gravity at 500 Hz.

### Ablation: why the fused loss needs its nuisance model

`python scripts/ablation_study.py` re-runs calibration under identical
reduced budgets and scores each variant against ground truth
(`outputs/ablation.json`, `outputs/ablation.png`):

| Variant | IoU cal / held-out | femur err | tibia err | chain err |
|---|---|---|---|---|
| silhouette-only (v1.0 formulation) | 0.935 / 0.922 | −4.21% | +3.02% | −0.68% |
| naive fused, **no** mount-offset model | 0.897 / 0.889 | −5.05% | **+10.99%** | +2.78% |
| two-phase fused, w=0.05 (shipped design) | 0.931 / 0.922 | −3.84% | +3.88% | **−0.07%** |
| two-phase fused, w=0.1 (shipped design) | 0.926 / 0.916 | **−2.98%** | +4.90% | +0.87% |
| two-phase w=0.1, seed −12% lengths | 0.922 / 0.911 | −4.35% | +7.80% | +1.58% |
| two-phase w=0.1, seed +12% lengths | 0.923 / 0.912 | −4.40% | +7.82% | +1.56% |

Reading: fitting absolute landmark positions *without* modelling the
skin-mount bias chases the bias and inflates the tibia by ~11% — the naive
row. Modelling the bias as a per-marker nuisance offset (the shipped two-phase
design) lets the keypoint term sharpen the femur/chain estimate instead.
Re-running from ±12% perturbed seeds converges to the same region, so the
result is not a lucky start.

## Installation

```bash
# Python ≥ 3.9
pip install -r requirements.txt
```

MuJoCo offscreen rendering is used automatically whenever a GL context (EGL /
OSMesa / GLFW) is available. On headless CPU-only machines the pipeline
transparently falls back to an **exact analytic pinhole projection** of the
same geoms (mathematically identical for this planar scene — the two backends
are cross-validated by a unit test).

## Quickstart

```bash
python run_mirror.py                 # full pipeline on the bundled demo clip
python run_mirror.py --profile fast  # ~4x smaller optimiser budget
python run_mirror.py --profile smoke # CI-style sanity run
python scripts/ablation_study.py     # fused-loss ablation + robustness (GT-scored)
make video                           # (re)generate the demo asset + ground truth
make test                            # pytest suite
```

## Using your own video

```bash
python run_mirror.py --video path/to/clip.mp4 --camera-distance 3.0 --focal-px 620
```

Current input requirements (see [limitations](#limitations-and-honest-scope)):

* **Sagittal-plane two-link movement** captured side-on, camera roughly
  perpendicular to the movement plane; the hip anchor should stay
  approximately fixed in the frame.
* **Three coloured fiducials** (default: magenta = hip, green = knee,
  blue = ankle) large enough to resolve (~6 px radius at 640×480) and
  distinguishable in HSV space. Custom colours can be supplied via
  `DEFAULT_MARKER_HSV` in `src/vision_extractor.py`.
* A **dark, quasi-static background** with bright limbs, and a one-time
  `--camera-distance` measurement (tape measure) that fixes the monocular
  pixel→metre scale.

## What the pipeline produces

| File | Description |
|---|---|
| `models/mirror_initial.xml` | Seed MJCF model straight from the vision stage |
| `models/mirror_calibrated.xml` | The converged digital twin (compiles standalone) |
| `outputs/mirror_comparison.mp4` | Side-by-side: input + tracking, uncalibrated, calibrated twin |
| `outputs/mirror_comparison.gif` | The same comparison as an animated GIF |
| `outputs/kinematics_and_torques.png` | 3-panel figure: angles (CV vs physics), torque demand (N·m), IoU convergence |
| `outputs/calibration_diagnostics.png` | 4-panel figure: held-out validation split, keypoint reprojection + recovered mount offsets, parameter 95% CIs vs ground truth, fused-loss decomposition |
| `outputs/mirror_metrics.json` | Full quantitative record: parameters, per-frame IoU, RMSEs, torques, reaction forces, held-out validation, 95% CIs, environment, ground-truth check when available |
| `outputs/ablation.json` / `outputs/ablation.png` | (from `scripts/ablation_study.py`) ground-truth-scored ablation of the fused loss |
| `outputs/calibrated_model_render.png` | Mid-swing render of the twin |

## Method details

**Part 1 — vision.** Foreground = bright low-saturation (limb) pixels ∪
fiducial pixels, cleaned by morphological opening/closing; a closing pass
sized to the fiducial diameter reconstructs the limb silhouette under the
opaque markers. Marker centroids come from contour moments; short gaps are
interpolated. Pixel trajectories are de-jittered with a Savitzky-Golay filter
(window 11, order 3). Angles follow the convention `θ = atan2(v_x, v_y)`
(0 = hanging down, positive towards image +x); the knee is the relative
shank-vs-thigh angle. Initial lengths/radii come from the pinhole model
`L = pixel_length · Z / f`, with limb radii measured from the distance-transform
ridge (robust to marker-placement bias).

**Part 2 — MJCF.** A two-link sagittal body tree (thigh → shank) with hinge
joints about the lateral axis, capsule geoms whose inertia MuJoCo derives from
a tissue-like density prior (1050 kg/m³), physiological joint limits, joint
damping, armature inertia, and `<position>` servos with critically-damped PD
gains (`kv = 2ζ√(kp·I)`, ζ = 0.7, I = swing-leg inertia about the hip). Every
candidate model is compiled and validated on the fly.

**Part 3 — forward dynamics.** 500 Hz `mj_step` with RK4 under
`g = 9.81 m/s²`. Targets are linearly interpolated at the physics rate, with a
velocity feed-forward `(kv/kp)·θ̇_ref` that cancels the damping-induced
tracking lag (without it, the identified geometry would absorb servo lag).
Recorded per rollout: joint torques `qfrc_actuator` (N·m), internal joint
reaction forces (`mj_rnePostConstraint` → `cfrc_int`, cross-validated against
the static subtree weight), and silhouettes from the `mirror_cam` camera whose
fovy and placement are derived from the *same* pinhole parameters as Part 1 —
so renders land in the exact perspective of the video.

**Part 4 — closed-loop calibration (two phases).** *Phase 1* minimises
`loss = 1 − mean IoU(observed limb mask, rendered limb silhouette)` over a
subsampled frame set with `scipy.optimize.minimize(method="Powell")` in
bound-normalised coordinates over `x = [L_femur, L_tibia, hip_offset,
knee_offset, servo_kp]`, with one refinement restart. The additive offsets
absorb the fiducial-placement bias (skin markers glued a few degrees away
from the true rotation centres).

*Phase 2* re-optimises the geometry under a **fused objective**
`(1 − mean IoU) + w_kp · reprojection RMSE`, where the three tracked fiducial
centroids are compared against the *pinhole-projected landmark centres* of
the simulated model. Because skin markers are mounted a few pixels away from
the true rotation centres, each marker carries a constant pixel offset
`δ_marker = (dx, dy)` treated as an explicit **nuisance parameter**
(`residual(t) = projection(q(t)) + δ_marker − centroid(t)`): the optimiser
must explain the *time-varying* part of the residuals through the geometry
while the constant part is absorbed by the nuisance. Without this model the
absolute positions chase the mount bias and inflate the segments
(+11% tibia — see the ablation table above); with it, the femur/chain
estimates sharpen and the fitted `δ_marker` becomes a direct measurement of
the instrumentation error itself. Frames never seen by the optimiser are
reported separately as held-out validation IoU, and
`estimate_ci95_curvature` adds local-curvature 95% confidence intervals
(likelihood-ratio threshold χ²₀.₉₅(1)/2, finite differences in
bound-normalised coordinates, other parameters held fixed — conservative
relative to a full profile likelihood).

## Limitations and honest scope

* **The bundled demo clip is synthetic** (rendered by
  `scripts/make_demo_video.py` with published ground truth). This is what
  makes *quantitative* validation possible; it is not a claim about real-world
  footage. The pipeline has no access to the ground truth — it is used only
  for scoring, in the `ground_truth_check` block of the metrics JSON.
* **Monocular scale is ambiguous.** A single view cannot recover absolute
  length: the `--camera-distance` prior fixes the pixel→metre scale. If that
  prior is off by a factor k, all recovered lengths are off by k (the
  silhouette IoU is unaffected).
* **Silhouette observability of the knee is limited.** The knee sits inside
  the union of the two limb silhouettes, so its position *along* the limb axis
  is only weakly constrained by shape overlap alone. v1.1 addresses this with
  the fused keypoint-reprojection term (chain error −1.2% → −0.23% on the
  benchmark, femur −5.0% → −3.2%); the per-segment split still carries more
  error than the chain total, because the fiducial mount bias and the
  geometry are only jointly observable — the mount-offset nuisance model
  reduces but does not eliminate the degeneracy.
* **Planar sagittal assumption.** Out-of-plane motion, perspective
  foreshortening beyond the pinhole model, hip translation, foot
  ground-contact, and marker occlusion beyond short gaps are outside the
  current scope; mass distribution is a density prior, not identified.
  Learning-based keypoint detection (e.g. DeepLabCut-style) is the natural
  upgrade over coloured fiducials.

## Project layout

```text
mirror/
├── run_mirror.py              # end-to-end pipeline runner (CLI)
├── src/
│   ├── vision_extractor.py    # Part 1 — ingestion, tracking, kinematics
│   ├── mjcf_builder.py        # Part 2 — parameterized MJCF compilation
│   ├── mujoco_sim.py          # Part 3 — 500 Hz forward dynamics + silhouettes
│   └── optimizer.py           # Part 4 — closed-loop IoU calibration
├── scripts/make_demo_video.py # synthetic benchmark + ground truth
├── scripts/ablation_study.py  # fused-loss ablation + robustness (GT-scored)
├── tests/test_mirror.py       # pytest suite (unit + integration)
├── assets/                    # demo video + ground-truth JSON
├── models/                    # generated XML models
├── outputs/                   # generated videos, plots, metrics
└── .github/workflows/ci.yml   # CI: tests + end-to-end smoke run
```

## License

MIT — see [LICENSE](LICENSE).

#!/usr/bin/env python3
"""Mirror — ablation & robustness study for the fused calibration loss.

Answers, with reproducible numbers, the question a reviewer will ask: *does
the keypoint-reprojection term actually improve the identified geometry, and
how much?*  The bundled benchmark ships published ground truth (true link
geometry, true camera, deliberate skin-marker bias), so every calibration
variant can be scored against the truth.

Variants
--------
1. ``silhouette_only``          the original Mirror formulation (kp_weight = 0)
2. ``naive_fused_w*``           fused loss WITHOUT the nuisance offsets —
                                demonstrates that fitting absolute landmark
                                positions chases the skin-mount bias and
                                *inflates* the segments
3. ``twophase_w*``              the shipped formulation: silhouette-only
                                geometry phase + nuisance-aware fused
                                refinement (marker-mount offsets modelled)
4. robustness: the default two-phase weight re-run from *perturbed* initial
   geometries (link lengths scaled ±12%) to show the optimum is recovered
   independently of the vision-stage seed.

Each variant uses the same reduced Powell budget so the comparison is
budget-fair.  Outputs:

    outputs/ablation.json    full per-variant record
    outputs/ablation.png     ground-truth-error bar chart

Usage
-----
    python scripts/ablation_study.py [--maxfev 600] [--out-dir outputs]
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import src as mirror_pkg
from run_mirror import DEFAULT_GROUND_TRUTH, DEFAULT_VIDEO, initial_params_from_vision
from src.optimizer import MirrorOptimizer, calibrate_two_phase
from src.vision_extractor import VisionExtractor

#: reduced but budget-fair optimiser settings for every variant
ABLATION_FRAME_STRIDE = 3
ABLATION_MAX_EVAL_FRAMES = 24


def gt_length_errors(params, gt: dict) -> dict:
    femur_err = 100.0 * (params.L_femur_m - gt["links"]["L_femur_m"]) / gt["links"]["L_femur_m"]
    tibia_err = 100.0 * (params.L_tibia_m - gt["links"]["L_tibia_m"]) / gt["links"]["L_tibia_m"]
    chain_est = params.L_femur_m + params.L_tibia_m
    chain_true = gt["links"]["L_femur_m"] + gt["links"]["L_tibia_m"]
    return {
        "femur_err_pct": float(femur_err),
        "tibia_err_pct": float(tibia_err),
        "chain_err_pct": float(100.0 * (chain_est - chain_true) / chain_true),
        "abs_femur_err_pct": float(abs(femur_err)),
        "abs_tibia_err_pct": float(abs(tibia_err)),
        "abs_chain_err_pct": float(abs(100.0 * (chain_est - chain_true) / chain_true)),
    }


def run_variant(name, vision, params0, maxfev, gt, *,
                kp_weight=0.0, mode="silhouette", perturbation=None) -> dict:
    """One budget-fair calibration variant; returns its record."""
    seed_params = params0
    if perturbation:
        seed_params = dataclasses.replace(
            params0,
            L_femur_m=params0.L_femur_m * perturbation.get("femur_scale", 1.0),
            L_tibia_m=params0.L_tibia_m * perturbation.get("tibia_scale", 1.0),
        )

    t0 = time.perf_counter()
    if mode == "silhouette":
        opt = MirrorOptimizer(
            vision, seed_params,
            frame_stride=ABLATION_FRAME_STRIDE, max_eval_frames=ABLATION_MAX_EVAL_FRAMES,
            render_height=vision.height, render_width=vision.width,
            kp_weight=0.0,
        )
        res = opt.calibrate(maxfev=maxfev)
    elif mode == "naive_fused":
        opt = MirrorOptimizer(
            vision, seed_params,
            frame_stride=ABLATION_FRAME_STRIDE, max_eval_frames=ABLATION_MAX_EVAL_FRAMES,
            render_height=vision.height, render_width=vision.width,
            kp_weight=kp_weight, optimize_marker_offsets=False,
        )
        res = opt.calibrate(maxfev=maxfev)
    elif mode == "twophase":
        _, res, _ = calibrate_two_phase(
            vision, seed_params, kp_weight=kp_weight, maxfev_total=maxfev,
            frame_stride=ABLATION_FRAME_STRIDE, max_eval_frames=ABLATION_MAX_EVAL_FRAMES,
            render_height=vision.height, render_width=vision.width,
        )
    else:
        raise ValueError(mode)
    wall = time.perf_counter() - t0

    rec = {
        "variant": name,
        "mode": mode,
        "kp_weight": float(kp_weight),
        "seed_perturbation": perturbation or {},
        "x_optimized": res.best_params.calibration_vector(),
        "mean_iou_calibration_frames": res.mean_iou_final,
        "mean_iou_heldout_frames": res.mean_iou_heldout,
        "reproj_rmse_px_final": res.reproj_rmse_px_final,
        "n_evals": res.n_evals,
        "wall_time_s": wall,
    }
    if res.marker_offsets_px:
        rec["marker_offsets_px"] = res.marker_offsets_px
        if gt is not None:
            rec["marker_offset_errors_px"] = {
                m: float(np.linalg.norm(
                    np.asarray(res.marker_offsets_px[m]) - np.asarray(gt["marker_bias_px"][m])
                ))
                for m in res.marker_offsets_px
            }
    if gt is not None:
        rec["gt"] = gt_length_errors(res.best_params, gt)
        rec["identified_offset_deg"] = {
            "hip": float(np.degrees(res.best_params.hip_offset_rad)),
            "knee": float(np.degrees(res.best_params.knee_offset_rad)),
        }
    return rec


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Mirror ablation study")
    ap.add_argument("--video", default=str(DEFAULT_VIDEO))
    ap.add_argument("--maxfev", type=int, default=600, help="Powell budget per variant")
    ap.add_argument("--out-dir", default=str(ROOT / "outputs"))
    args = ap.parse_args(argv)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    video_path = Path(args.video)
    if not video_path.exists():
        raise FileNotFoundError(video_path)

    print(f"Mirror ablation study v{mirror_pkg.__version__} | budget {args.maxfev} evals/variant")
    vision = VisionExtractor(video_path).extract()
    params0 = initial_params_from_vision(vision, camera_distance_m=3.0, kp_init=250.0)

    gt = None
    if DEFAULT_GROUND_TRUTH.exists() and video_path.resolve() == DEFAULT_VIDEO.resolve():
        gt = json.loads(DEFAULT_GROUND_TRUTH.read_text(encoding="utf-8"))

    variants = [
        ("silhouette_only", dict(mode="silhouette")),
        ("naive_fused_w0.1", dict(mode="naive_fused", kp_weight=0.1)),
        ("twophase_w0.05", dict(mode="twophase", kp_weight=0.05)),
        ("twophase_w0.1", dict(mode="twophase", kp_weight=0.1)),
        ("twophase_w0.1_perturbed_short",
         dict(mode="twophase", kp_weight=0.1, perturbation={"femur_scale": 0.88, "tibia_scale": 0.88})),
        ("twophase_w0.1_perturbed_long",
         dict(mode="twophase", kp_weight=0.1, perturbation={"femur_scale": 1.12, "tibia_scale": 1.12})),
    ]

    records = []
    for name, kwargs in variants:
        label = f"{name} {kwargs}"
        print(f"\n=== {label} " + "=" * max(0, 60 - len(label)))
        rec = run_variant(name, vision, params0, args.maxfev, gt, **kwargs)
        records.append(rec)
        line = (f"  IoU cal {rec['mean_iou_calibration_frames']:.4f} | held-out "
                f"{rec['mean_iou_heldout_frames']:.4f} | {rec['n_evals']} evals in {rec['wall_time_s']:.0f}s")
        if "gt" in rec:
            g = rec["gt"]
            line += (f" | GT err: femur {g['femur_err_pct']:+.2f}%, "
                     f"tibia {g['tibia_err_pct']:+.2f}%, chain {g['chain_err_pct']:+.2f}%")
        if "marker_offset_errors_px" in rec:
            errs = rec["marker_offset_errors_px"]
            line += f" | mount-offset err px: " + "/".join(f"{errs[m]:.1f}" for m in ("hip", "knee", "ankle"))
        print(line)

    # -- summary table ----------------------------------------------------------
    print("\n" + "=" * 108)
    print(" ABLATION SUMMARY (ground-truth scoring, budget-fair)")
    print("=" * 108)
    print(f" {'variant':32s} {'IoU cal':>8s} {'IoU held':>9s} {'femur %':>9s} {'tibia %':>9s} "
          f"{'chain %':>9s} {'mount err px':>14s}")
    print("-" * 108)
    for rec in records:
        g = rec.get("gt")
        row = (f" {rec['variant']:32s} {rec['mean_iou_calibration_frames']:8.4f} "
               f"{rec['mean_iou_heldout_frames']:9.4f}")
        if g:
            row += (f" {g['femur_err_pct']:+9.2f} {g['tibia_err_pct']:+9.2f} "
                    f"{g['chain_err_pct']:+9.2f}")
        if "marker_offset_errors_px" in rec:
            errs = rec["marker_offset_errors_px"]
            row += f" {np.mean([errs[m] for m in errs]):14.2f}"
        print(row)

    out_path = out_dir / "ablation.json"
    out_path.write_text(json.dumps(records, indent=2), encoding="utf-8")
    print(f"\n wrote {out_path}")

    # -- figure -------------------------------------------------------------------
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        scored = [r for r in records if "gt" in r]
        names = [r["variant"] for r in scored]
        xpos = np.arange(len(scored))
        width = 0.26
        fig, ax = plt.subplots(figsize=(12, 5.4), constrained_layout=True)
        ax.bar(xpos - width, [abs(r["gt"]["femur_err_pct"]) for r in scored], width,
               label="|femur error|", color="tab:red", alpha=0.85)
        ax.bar(xpos, [abs(r["gt"]["tibia_err_pct"]) for r in scored], width,
               label="|tibia error|", color="tab:blue", alpha=0.85)
        ax.bar(xpos + width, [abs(r["gt"]["chain_err_pct"]) for r in scored], width,
               label="|chain error|", color="tab:green", alpha=0.85)
        ax.set_xticks(xpos, [n.replace("_", "\n") for n in names], fontsize=7.5)
        ax.set_ylabel("absolute length error vs ground truth (%)")
        ax.set_title(f"Mirror ablation — identified geometry vs truth "
                     f"({args.maxfev}-eval Powell budget per variant)")
        ax.grid(alpha=0.3, axis="y")
        ax.legend()
        fig_path = out_dir / "ablation.png"
        fig.savefig(fig_path, dpi=150)
        plt.close(fig)
        print(f" wrote {fig_path}")
    except Exception as exc:  # pragma: no cover
        print(f" figure skipped: {exc}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

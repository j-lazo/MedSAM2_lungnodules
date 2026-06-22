#!/usr/bin/env python3
"""
Detector-guided MedSAM2 for LUNA-style nodule segmentation.

This script implements the first practical training stage for prompt-less / automatic
SAM2-based LUNA16 segmentation:

1) train-detector
   Train a 2D/2.5D nodule detector on top of frozen or partially trainable SAM2
   image encoder features. The detector predicts:
       - objectness heatmap
       - box distances [l, t, r, b] from each feature-grid location

2) infer-detector
   Run the trained detector and write candidate prompts to CSV.

3) infer-detect-sam2
   Run detector first, convert top candidates to point+box prompts, then run the
   frozen SAM2 image predictor slice-wise to produce a 3D mask volume.

Assumptions
-----------
- You run this from the SAM2 repository root, or have SAM2 importable.
- Your LUNA dataset structure matches your existing sam2_luna_inference.py:
      DATASET_DIR/CT_volumes/*.mhd
      DATASET_DIR/masks_nodules/nifti_data/*mask*contour*nodule*.nii.gz
      DATASET_DIR/annotations.csv
      DATASET_DIR/LUNA16_metadata_split_offical.csv
- If your metadata differs, pass --volumes-dir, --masks-dir, --annotations-csv,
  and --links-csv explicitly.

Recommended first run
---------------------
Train detector on a small subset first:

python train_sam2_luna_detector.py train-detector \
  --dataset-dir /path/to/LUNA16 \
  --model-cfg configs/sam2.1/sam2.1_hiera_l.yaml \
  --checkpoint checkpoints/sam2.1_hiera_large.pt \
  --output-dir results/sam2_detector_debug \
  --max-cases 20 --epochs 2 --batch-size 8 --num-workers 4 \
  --use-triplet-channels --amp-dtype bf16

Then run detector + SAM2 segmentation:

python train_sam2_luna_detector.py infer-detect-sam2 \
  --dataset-dir /path/to/LUNA16 \
  --model-cfg configs/sam2.1/sam2.1_hiera_l.yaml \
  --checkpoint checkpoints/sam2.1_hiera_large.pt \
  --detector-checkpoint results/sam2_detector_debug/best_detector.pt \
  --output-dir results/sam2_detect_sam2_debug \
  --max-cases 20 --use-triplet-channels --save-volumes
"""

from __future__ import annotations

import argparse
import contextlib
import csv
import gc
import hashlib
import json
import math
import os
import random
import re
import shutil
import socket
import tempfile
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import cv2
import numpy as np
import pandas as pd
import SimpleITK as sitk
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from scipy import ndimage
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor

# MedSAM2 uses the same build_sam2/SAM2ImagePredictor API in your inference script.


# =============================================================================
# General utilities
# =============================================================================


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def slugify(value: str, max_len: int = 160) -> str:
    value = str(value).replace("+", "plus")
    value = re.sub(r"[^A-Za-z0-9._-]+", "-", value)
    value = re.sub(r"-+", "-", value).strip("-_.")
    return value[:max_len]


def select_device(device_arg: str) -> torch.device:
    if device_arg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_arg)


def safe_autocast(device: torch.device, amp_dtype: str):
    if device.type != "cuda" or amp_dtype == "none":
        return contextlib.nullcontext()
    dtype = {"bf16": torch.bfloat16, "fp16": torch.float16}[amp_dtype]
    return torch.autocast("cuda", dtype=dtype)


def configure_torch(device: torch.device, allow_tf32: bool = True) -> None:
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = bool(allow_tf32)
        torch.backends.cudnn.allow_tf32 = bool(allow_tf32)
        if hasattr(torch, "set_float32_matmul_precision") and allow_tf32:
            torch.set_float32_matmul_precision("high")


def dice_score(mask1: np.ndarray, mask2: np.ndarray, smooth: float = 1e-6) -> float:
    mask1 = mask1.astype(bool)
    mask2 = mask2.astype(bool)
    inter = np.logical_and(mask1, mask2).sum(dtype=np.float64)
    total = mask1.sum(dtype=np.float64) + mask2.sum(dtype=np.float64)
    return float((2.0 * inter + smooth) / (total + smooth))


def write_pred_volume(pred: np.ndarray, reference_image: sitk.Image, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    img = sitk.GetImageFromArray(pred.astype(np.uint8))
    img.CopyInformation(reference_image)
    sitk.WriteImage(img, str(out_path))



def binary_confusion_counts(gt: np.ndarray, pred: np.ndarray) -> Tuple[int, int, int, int]:
    gt_b = gt.astype(bool)
    pred_b = pred.astype(bool)
    tp = int(np.logical_and(gt_b, pred_b).sum())
    fp = int(np.logical_and(~gt_b, pred_b).sum())
    fn = int(np.logical_and(gt_b, ~pred_b).sum())
    tn = int(np.logical_and(~gt_b, ~pred_b).sum())
    return tp, fp, fn, tn


def iou_score(gt: np.ndarray, pred: np.ndarray, smooth: float = 1e-6) -> float:
    gt_b = gt.astype(bool)
    pred_b = pred.astype(bool)
    inter = np.logical_and(gt_b, pred_b).sum(dtype=np.float64)
    union = np.logical_or(gt_b, pred_b).sum(dtype=np.float64)
    return float((inter + smooth) / (union + smooth))


def precision_recall_f1(gt: np.ndarray, pred: np.ndarray, smooth: float = 1e-6) -> Dict[str, float]:
    tp, fp, fn, tn = binary_confusion_counts(gt, pred)
    precision = float((tp + smooth) / (tp + fp + smooth))
    recall = float((tp + smooth) / (tp + fn + smooth))
    f1 = float((2.0 * precision * recall + smooth) / (precision + recall + smooth))
    return {"precision": precision, "recall": recall, "f1": f1, "tp": tp, "fp": fp, "fn": fn, "tn": tn}


def _surface(mask: np.ndarray) -> np.ndarray:
    mask = mask.astype(bool)
    if not mask.any():
        return mask
    structure = ndimage.generate_binary_structure(mask.ndim, 1)
    eroded = ndimage.binary_erosion(mask, structure=structure, border_value=0)
    return np.logical_and(mask, ~eroded)


def surface_distances_mm(mask_a: np.ndarray, mask_b: np.ndarray, spacing_zyx: Tuple[float, float, float]) -> np.ndarray:
    a = mask_a.astype(bool)
    b = mask_b.astype(bool)
    if not a.any() or not b.any():
        return np.array([], dtype=np.float32)
    surf_a = _surface(a)
    surf_b = _surface(b)
    dt_b = ndimage.distance_transform_edt(~surf_b, sampling=spacing_zyx)
    dt_a = ndimage.distance_transform_edt(~surf_a, sampling=spacing_zyx)
    d_ab = dt_b[surf_a]
    d_ba = dt_a[surf_b]
    return np.concatenate([d_ab, d_ba]).astype(np.float32)


def hd95_mm(gt: np.ndarray, pred: np.ndarray, spacing_zyx: Tuple[float, float, float]) -> float:
    if not gt.astype(bool).any() and not pred.astype(bool).any():
        return 0.0
    if not gt.astype(bool).any() or not pred.astype(bool).any():
        return float("nan")
    d = surface_distances_mm(gt, pred, spacing_zyx)
    return float(np.percentile(d, 95)) if d.size else float("nan")


def assd_mm(gt: np.ndarray, pred: np.ndarray, spacing_zyx: Tuple[float, float, float]) -> float:
    if not gt.astype(bool).any() and not pred.astype(bool).any():
        return 0.0
    if not gt.astype(bool).any() or not pred.astype(bool).any():
        return float("nan")
    d = surface_distances_mm(gt, pred, spacing_zyx)
    return float(np.mean(d)) if d.size else float("nan")


def volume_similarity(gt: np.ndarray, pred: np.ndarray, smooth: float = 1e-6) -> float:
    g = float(gt.astype(bool).sum())
    p = float(pred.astype(bool).sum())
    return float(1.0 - abs(p - g) / (p + g + smooth))


def get_spacing_zyx(image_itk: sitk.Image) -> Tuple[float, float, float]:
    sx, sy, sz = image_itk.GetSpacing()
    return float(sz), float(sy), float(sx)


def segmentation_metrics(gt: np.ndarray, pred: np.ndarray, spacing_zyx: Tuple[float, float, float]) -> Dict[str, float]:
    pr = precision_recall_f1(gt, pred)
    return {
        "DSC": dice_score(gt, pred),
        "IoU": iou_score(gt, pred),
        "seg_precision": pr["precision"],
        "seg_recall": pr["recall"],
        "seg_f1": pr["f1"],
        "HD95_mm": hd95_mm(gt, pred, spacing_zyx),
        "ASSD_mm": assd_mm(gt, pred, spacing_zyx),
        "volume_similarity": volume_similarity(gt, pred),
        "pred_voxels": int(pred.astype(bool).sum()),
        "gt_voxels": int(gt.astype(bool).sum()),
    }


def extract_gt_nodules_3d(gt_volume: np.ndarray, spacing_zyx: Tuple[float, float, float], min_area: int = 1) -> List[Dict]:
    labeled, num = ndimage.label(gt_volume.astype(bool), structure=ndimage.generate_binary_structure(3, 2))
    nodules: List[Dict] = []
    voxel_volume = float(spacing_zyx[0] * spacing_zyx[1] * spacing_zyx[2])
    for k in range(1, num + 1):
        comp = labeled == k
        vox = int(comp.sum())
        if vox < min_area:
            continue
        coords = np.argwhere(comp)
        zmin, ymin, xmin = coords.min(axis=0)
        zmax, ymax, xmax = coords.max(axis=0)
        center = coords.mean(axis=0)  # z,y,x in voxels
        extent_vox = np.array([zmax - zmin + 1, ymax - ymin + 1, xmax - xmin + 1], dtype=np.float32)
        extent_mm = extent_vox * np.array(spacing_zyx, dtype=np.float32)
        eq_diam_mm = float((6.0 * vox * voxel_volume / np.pi) ** (1.0 / 3.0))
        nodules.append({
            "nodule_id": int(k),
            "mask": comp.astype(np.uint8),
            "center_zyx": center.astype(np.float32),
            "bbox_zyx": [int(zmin), int(ymin), int(xmin), int(zmax), int(ymax), int(xmax)],
            "gt_volume_voxels": vox,
            "gt_volume_mm3": float(vox * voxel_volume),
            "gt_diameter_vox": float(max(extent_vox)),
            "gt_diameter_mm": float(max(extent_mm)),
            "gt_equivalent_diameter_mm": eq_diam_mm,
        })
    nodules.sort(key=lambda n: n["nodule_id"])
    return nodules


def candidate_fields(c) -> Tuple[float, float, float, float, Optional[np.ndarray]]:
    z = float(getattr(c, "z"))
    y = float(getattr(c, "y"))
    x = float(getattr(c, "x"))
    score = float(getattr(c, "score", np.nan))
    box = getattr(c, "box", None)
    if box is None:
        box = getattr(c, "box_xyxy", None)
    return z, y, x, score, None if box is None else np.asarray(box, dtype=np.float32)


def center_error(c, nodule: Dict, spacing_zyx: Tuple[float, float, float]) -> Tuple[float, float, int]:
    z, y, x, score, _ = candidate_fields(c)
    center = nodule["center_zyx"]
    dz, dy, dx = z - float(center[0]), y - float(center[1]), x - float(center[2])
    err_vox = float(np.sqrt(dz * dz + dy * dy + dx * dx))
    err_mm = float(np.sqrt((dz * spacing_zyx[0]) ** 2 + (dy * spacing_zyx[1]) ** 2 + (dx * spacing_zyx[2]) ** 2))
    return err_vox, err_mm, int(round(abs(dz)))


def match_candidates_to_gt(gt_volume: np.ndarray, candidates: List, spacing_zyx: Tuple[float, float, float], min_area: int = 1) -> Tuple[List[Dict], List[Dict], List[int]]:
    nodules = extract_gt_nodules_3d(gt_volume, spacing_zyx, min_area=min_area)
    matches: List[Dict] = []
    used = set()
    for ni, n in enumerate(nodules):
        best = None
        for ci, c in enumerate(candidates):
            if ci in used:
                continue
            ev, em, ez = center_error(c, n, spacing_zyx)
            radius_vox = max(2.0, 0.5 * float(n["gt_diameter_vox"]))
            radius_mm = max(2.0, 0.5 * float(n["gt_diameter_mm"]))
            if ev <= radius_vox or em <= radius_mm:
                z, y, x, score, box = candidate_fields(c)
                item = {"gt_index": ni, "cand_index": ci, "center_error_vox": ev, "center_error_mm": em, "slice_error": ez, "detector_confidence": score}
                if best is None or item["center_error_mm"] < best["center_error_mm"]:
                    best = item
        if best is not None:
            used.add(best["cand_index"])
            matches.append(best)
    unmatched_gt = [i for i in range(len(nodules)) if i not in {m["gt_index"] for m in matches}]
    unmatched_cand = [i for i in range(len(candidates)) if i not in used]
    return matches, nodules, unmatched_cand


def detection_localization_metrics(gt_volume: np.ndarray, candidates: List, spacing_zyx: Tuple[float, float, float], min_area: int = 1) -> Dict[str, float]:
    matches, nodules, unmatched_cand = match_candidates_to_gt(gt_volume, candidates, spacing_zyx, min_area=min_area)
    tp = len(matches)
    fn = max(0, len(nodules) - tp)
    fp = len(unmatched_cand)
    precision = float(tp / (tp + fp)) if (tp + fp) else 0.0
    recall = float(tp / (tp + fn)) if (tp + fn) else 0.0
    f1 = float(2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
    center_errs_mm = [m["center_error_mm"] for m in matches]
    center_errs_vox = [m["center_error_vox"] for m in matches]
    slice_errs = [m["slice_error"] for m in matches]
    return {
        "det_TP": tp,
        "det_FP": fp,
        "det_FN": fn,
        "det_precision": precision,
        "det_recall": recall,
        "det_f1": f1,
        "n_gt_nodules": len(nodules),
        "n_candidates": len(candidates),
        "mean_center_error_mm": float(np.mean(center_errs_mm)) if center_errs_mm else np.nan,
        "median_center_error_mm": float(np.median(center_errs_mm)) if center_errs_mm else np.nan,
        "mean_center_error_vox": float(np.mean(center_errs_vox)) if center_errs_vox else np.nan,
        "mean_slice_error": float(np.mean(slice_errs)) if slice_errs else np.nan,
        "best_det_score": float(max([candidate_fields(c)[3] for c in candidates], default=np.nan)),
    }


def per_nodule_metrics(case: CaseData, pred: Optional[np.ndarray], candidates: List, min_area: int = 1) -> List[Dict]:
    spacing_zyx = get_spacing_zyx(case.image_itk)
    matches, nodules, unmatched = match_candidates_to_gt(case.gt_volume, candidates, spacing_zyx, min_area=min_area)
    match_by_gt = {m["gt_index"]: m for m in matches}
    rows: List[Dict] = []
    for i, n in enumerate(nodules):
        gt_mask = n["mask"].astype(np.uint8)
        m = match_by_gt.get(i)
        if pred is None:
            dsc = np.nan
            hd = np.nan
        else:
            dsc = dice_score(gt_mask, pred)
            hd = hd95_mm(gt_mask, pred, spacing_zyx)
        row = {
            "SeriesUID": case.series_id,
            "nodule_id": n["nodule_id"],
            "GT_diameter_vox": n["gt_diameter_vox"],
            "GT_diameter_mm": n["gt_diameter_mm"],
            "GT_equivalent_diameter_mm": n["gt_equivalent_diameter_mm"],
            "GT_volume_voxels": n["gt_volume_voxels"],
            "GT_volume_mm3": n["gt_volume_mm3"],
            "Detector_confidence": m["detector_confidence"] if m else np.nan,
            "Center_error_vox": m["center_error_vox"] if m else np.nan,
            "Center_error_mm": m["center_error_mm"] if m else np.nan,
            "Slice_error": m["slice_error"] if m else np.nan,
            "Dice": dsc,
            "HD95_mm": hd,
            "detected": bool(m is not None),
        }
        rows.append(row)
    return rows


def summarize_detection_operating_points(volume_rows: List[Dict]) -> Dict[str, float]:
    if not volume_rows:
        return {}
    total_tp = int(np.nansum([r.get("det_TP", 0) for r in volume_rows]))
    total_fp = int(np.nansum([r.get("det_FP", 0) for r in volume_rows]))
    total_fn = int(np.nansum([r.get("det_FN", 0) for r in volume_rows]))
    n_scans = max(1, len([r for r in volume_rows if r.get("status") == "ok"]))
    precision = float(total_tp / (total_tp + total_fp)) if (total_tp + total_fp) else 0.0
    recall = float(total_tp / (total_tp + total_fn)) if (total_tp + total_fn) else 0.0
    f1 = float(2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
    return {"overall_det_TP": total_tp, "overall_det_FP": total_fp, "overall_det_FN": total_fn, "overall_det_precision": precision, "overall_det_recall": recall, "overall_det_f1": f1, "FP_per_scan": float(total_fp / n_scans)}


def save_detection_summary(rows: List[Dict], out_dir: Path) -> None:
    summary = summarize_detection_operating_points(rows)
    if summary:
        pd.DataFrame([summary]).to_csv(out_dir / "detection_summary.csv", index=False)
        with open(out_dir / "detection_summary.json", "w") as f:
            json.dump(summary, f, indent=2)


def plot_training_metrics(metrics_csv: Path, out_dir: Path) -> None:
    if not metrics_csv.exists():
        return
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:
        print(f"WARNING: could not import matplotlib for training plots: {exc}")
        return
    df = pd.read_csv(metrics_csv)
    if df.empty or "epoch" not in df.columns:
        return
    numeric_cols = [c for c in df.columns if c != "epoch" and pd.api.types.is_numeric_dtype(df[c])]
    if not numeric_cols:
        return
    loss_cols = [c for c in numeric_cols if "loss" in c.lower()]
    metric_cols = [c for c in numeric_cols if c not in loss_cols and c != "lr"]
    def _plot(cols: List[str], filename: str, title: str):
        if not cols:
            return
        plt.figure(figsize=(10, 6))
        for c in cols:
            plt.plot(df["epoch"], df[c], marker="o", linewidth=1.5, label=c)
        plt.xlabel("Epoch")
        plt.ylabel("Value")
        plt.title(title)
        plt.grid(True, alpha=0.3)
        plt.legend(loc="best", fontsize=8)
        plt.tight_layout()
        plt.savefig(out_dir / filename, dpi=200)
        plt.close()
    _plot(loss_cols, "training_validation_losses.png", "Training/validation losses")
    _plot(metric_cols, "training_validation_metrics.png", "Training/validation metrics")

def stable_uint32_seed(*items) -> int:
    payload = "::".join(str(x) for x in items).encode("utf-8")
    digest = hashlib.blake2b(payload, digest_size=8).digest()
    return int.from_bytes(digest, byteorder="little", signed=False) % (2**32)


# =============================================================================
# CT normalization and slice building
# =============================================================================


def normalize_ct_to_uint8(
    img_2d: np.ndarray,
    hu_min: float = -1000.0,
    hu_max: float = 400.0,
) -> np.ndarray:
    """Fixed CT windowing. Better for training than per-slice min-max."""
    img = img_2d.astype(np.float32)
    img = np.clip(img, hu_min, hu_max)
    img = (img - hu_min) / max(hu_max - hu_min, 1e-6)
    return np.round(img * 255.0).astype(np.uint8)


def build_sam2_input_slice(
    image_array: np.ndarray,
    z: int,
    use_triplet_channels: bool,
    hu_min: float,
    hu_max: float,
) -> np.ndarray:
    """Return H,W,3 uint8 image for SAM2."""
    if use_triplet_channels:
        z0 = max(z - 1, 0)
        z1 = z
        z2 = min(z + 1, image_array.shape[0] - 1)
        return np.stack(
            [
                normalize_ct_to_uint8(image_array[z0], hu_min, hu_max),
                normalize_ct_to_uint8(image_array[z1], hu_min, hu_max),
                normalize_ct_to_uint8(image_array[z2], hu_min, hu_max),
            ],
            axis=-1,
        )
    ch = normalize_ct_to_uint8(image_array[z], hu_min, hu_max)
    return np.stack([ch, ch, ch], axis=-1)


def numpy_rgb_to_tensor(rgb: np.ndarray) -> torch.Tensor:
    """H,W,3 uint8 -> 3,H,W float in [0,1]."""
    x = torch.from_numpy(rgb).permute(2, 0, 1).float() / 255.0
    return x


# =============================================================================
# Dataset indexing/loading, matching your current inference script layout
# =============================================================================


@dataclass(frozen=True)
class DatasetIndex:
    dataset_dir: Path
    volumes_dir: Path
    masks_dir: Path
    annotations_csv: Path
    links_csv: Path
    df_annotations: pd.DataFrame
    df_links: pd.DataFrame
    mask_id_to_file: Dict[int, str]
    volume_ids: List[str]


@dataclass(frozen=True)
class CaseData:
    series_id: str
    image_itk: sitk.Image
    image_array: np.ndarray
    gt_volume: np.ndarray
    mask_path: Path


def build_dataset_index(args: argparse.Namespace) -> DatasetIndex:
    dataset_dir = Path(args.dataset_dir).expanduser().resolve()
    volumes_dir = Path(args.volumes_dir).expanduser().resolve() if args.volumes_dir else dataset_dir / "CT_volumes"
    masks_dir = Path(args.masks_dir).expanduser().resolve() if args.masks_dir else dataset_dir / "masks_nodules" / "nifti_data"
    annotations_csv = Path(args.annotations_csv).expanduser().resolve() if args.annotations_csv else dataset_dir / "annotations.csv"
    links_csv = Path(args.links_csv).expanduser().resolve() if args.links_csv else dataset_dir / "LUNA16_metadata_split_offical.csv"

    for p, name in [
        (dataset_dir, "dataset_dir"),
        (volumes_dir, "volumes_dir"),
        (masks_dir, "masks_dir"),
        (annotations_csv, "annotations_csv"),
        (links_csv, "links_csv"),
    ]:
        if not p.exists():
            raise FileNotFoundError(f"{name} does not exist: {p}")

    df_annotations = pd.read_csv(annotations_csv)
    df_links = pd.read_csv(links_csv)
    if "seriesuid" not in df_annotations.columns:
        raise ValueError(f"{annotations_csv} must contain column 'seriesuid'")
    if not {"SeriesID", "CID"}.issubset(df_links.columns):
        raise ValueError(f"{links_csv} must contain columns 'SeriesID' and 'CID'")

    mask_files = [
        f.name for f in masks_dir.iterdir()
        if f.is_file()
        and "mask" in f.name
        and "contour" in f.name
        and "circle" not in f.name
        and "nodule" in f.name
    ]
    mask_id_to_file: Dict[int, str] = {}
    for fname in mask_files:
        try:
            mask_id_to_file[int(fname.split("_")[0])] = fname
        except ValueError:
            pass

    volume_ids = sorted(p.stem for p in volumes_dir.glob("*.mhd"))
    if args.only_annotated:
        annotated = set(df_annotations["seriesuid"].astype(str).tolist())
        volume_ids = [v for v in volume_ids if v in annotated]

    if args.case_list:
        wanted = [line.strip() for line in Path(args.case_list).read_text().splitlines() if line.strip()]
        wanted = [w[:-4] if w.endswith(".mhd") else w for w in wanted]
        wanted_set = set(wanted)
        volume_ids = [v for v in volume_ids if v in wanted_set]

    if args.shuffle:
        rng = np.random.default_rng(args.seed)
        volume_ids = list(rng.permutation(volume_ids))

    if args.dataset_fraction is not None:
        if not (0 < args.dataset_fraction <= 1):
            raise ValueError("--dataset-fraction must be in (0, 1]")
        n = max(1, int(math.ceil(len(volume_ids) * args.dataset_fraction)))
        volume_ids = volume_ids[:n]

    if args.max_cases is not None:
        volume_ids = volume_ids[: max(0, args.max_cases)]

    return DatasetIndex(
        dataset_dir=dataset_dir,
        volumes_dir=volumes_dir,
        masks_dir=masks_dir,
        annotations_csv=annotations_csv,
        links_csv=links_csv,
        df_annotations=df_annotations,
        df_links=df_links,
        mask_id_to_file=mask_id_to_file,
        volume_ids=volume_ids,
    )


def load_case(index: DatasetIndex, series_id: str) -> Optional[CaseData]:
    image_path = index.volumes_dir / f"{series_id}.mhd"
    if not image_path.is_file():
        return None

    links = index.df_links[index.df_links["SeriesID"].astype(str) == str(series_id)]
    if len(links) == 0:
        return None

    mask_id = int(links["CID"].iloc[0])
    mask_fname = index.mask_id_to_file.get(mask_id)
    if mask_fname is None:
        return None

    mask_path = index.masks_dir / mask_fname
    if not mask_path.is_file():
        return None

    image_itk = sitk.ReadImage(str(image_path))
    image_array = sitk.GetArrayFromImage(image_itk).astype(np.float32)
    mask_itk = sitk.ReadImage(str(mask_path))
    gt_volume = (sitk.GetArrayFromImage(mask_itk) >= 0.5).astype(np.uint8)
    if image_array.shape != gt_volume.shape:
        raise ValueError(f"Shape mismatch for {series_id}: image {image_array.shape}, mask {gt_volume.shape}")
    return CaseData(series_id, image_itk, image_array, gt_volume, mask_path)


# =============================================================================
# Train/val/test split utilities
# =============================================================================

def read_case_list_file(path: Optional[str]) -> Optional[List[str]]:
    """Read one SeriesUID per line. Empty lines and comments are ignored."""
    if path is None:
        return None
    ids: List[str] = []
    for line in Path(path).expanduser().read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if line.endswith(".mhd"):
            line = line[:-4]
        ids.append(line)
    return ids


def write_case_list_file(path: Path, ids: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(str(x) for x in ids) + ("\n" if ids else ""))


def split_volume_ids(
    volume_ids: Sequence[str],
    seed: int,
    val_ratio: float = 0.10,
    test_ratio: float = 0.0,
    train_case_list: Optional[str] = None,
    val_case_list: Optional[str] = None,
    test_case_list: Optional[str] = None,
    shuffle_splits: bool = True,
) -> Dict[str, List[str]]:
    """
    Build explicit train/val/test volume splits by SeriesUID.

    If any explicit split list is provided, those files define the split. Missing
    train list means "all remaining volumes not listed in val/test".
    Otherwise, ratios are applied after deterministic shuffling.
    """
    available = list(volume_ids)
    available_set = set(available)

    explicit_train = read_case_list_file(train_case_list)
    explicit_val = read_case_list_file(val_case_list)
    explicit_test = read_case_list_file(test_case_list)
    if explicit_train is not None or explicit_val is not None or explicit_test is not None:
        val_ids = [x for x in (explicit_val or []) if x in available_set]
        test_ids = [x for x in (explicit_test or []) if x in available_set]
        if explicit_train is None:
            used_nontrain = set(val_ids) | set(test_ids)
            train_ids = [x for x in available if x not in used_nontrain]
        else:
            train_ids = [x for x in explicit_train if x in available_set]

        overlap = (set(train_ids) & set(val_ids)) | (set(train_ids) & set(test_ids)) | (set(val_ids) & set(test_ids))
        if overlap:
            raise ValueError(f"Explicit split files overlap for {len(overlap)} SeriesUIDs, e.g. {sorted(overlap)[:5]}")
        used = set(train_ids) | set(val_ids) | set(test_ids)
        return {"train": train_ids, "val": val_ids, "test": test_ids, "all": [x for x in available if x in used] if used else available}

    if not (0.0 <= val_ratio < 1.0):
        raise ValueError("--val-ratio must be in [0, 1)")
    if not (0.0 <= test_ratio < 1.0):
        raise ValueError("--test-ratio must be in [0, 1)")
    if val_ratio + test_ratio >= 1.0:
        raise ValueError("--val-ratio + --test-ratio must be < 1")

    ids = list(available)
    if shuffle_splits:
        rng = np.random.default_rng(seed)
        ids = list(rng.permutation(ids))

    n_total = len(ids)
    n_test = int(round(n_total * test_ratio))
    n_val = int(round(n_total * val_ratio))
    if n_total > 1 and val_ratio > 0 and n_val == 0:
        n_val = 1
    if n_total > 2 and test_ratio > 0 and n_test == 0:
        n_test = 1
    if n_val + n_test >= n_total and n_total > 0:
        # Keep at least one train volume.
        excess = n_val + n_test - (n_total - 1)
        reduce_test = min(excess, n_test)
        n_test -= reduce_test
        excess -= reduce_test
        n_val = max(0, n_val - excess)

    test_ids = ids[:n_test]
    val_ids = ids[n_test:n_test + n_val]
    train_ids = ids[n_test + n_val:]
    return {"train": train_ids, "val": val_ids, "test": test_ids, "all": ids}


def write_split_files(splits: Dict[str, List[str]], out_dir: Path) -> None:
    split_dir = out_dir / "splits"
    for name in ["train", "val", "test", "all"]:
        write_case_list_file(split_dir / f"{name}.txt", splits.get(name, []))


def build_splits(index: DatasetIndex, args: argparse.Namespace) -> Dict[str, List[str]]:
    return split_volume_ids(
        index.volume_ids,
        seed=args.seed,
        val_ratio=getattr(args, "val_ratio", 0.10),
        test_ratio=getattr(args, "test_ratio", 0.0),
        train_case_list=getattr(args, "train_case_list", None),
        val_case_list=getattr(args, "val_case_list", None),
        test_case_list=getattr(args, "test_case_list", None),
        shuffle_splits=getattr(args, "shuffle_splits", True),
    )


def select_eval_volume_ids(index: DatasetIndex, args: argparse.Namespace) -> List[str]:
    splits = build_splits(index, args)
    split_name = getattr(args, "eval_split", "all")
    if split_name not in splits:
        raise ValueError(f"Unknown --eval-split {split_name!r}; expected one of {sorted(splits)}")
    return splits[split_name]


# =============================================================================
# Mask/object utilities
# =============================================================================


def get_bbox_from_2d_mask(mask_2d: np.ndarray) -> Optional[np.ndarray]:
    ys, xs = np.where(mask_2d > 0)
    if len(xs) == 0:
        return None
    return np.array([xs.min(), ys.min(), xs.max(), ys.max()], dtype=np.float32)


def get_center_from_2d_mask(mask_2d: np.ndarray) -> Optional[np.ndarray]:
    mask_bin = (mask_2d > 0).astype(np.uint8)
    if mask_bin.sum() == 0:
        return None
    dist = cv2.distanceTransform(mask_bin, cv2.DIST_L2, 5)
    y, x = np.unravel_index(np.argmax(dist), dist.shape)
    return np.array([float(x), float(y)], dtype=np.float32)


def extract_2d_objects(mask_2d: np.ndarray, min_area: int = 1) -> List[Dict]:
    mask_bin = (mask_2d > 0).astype(np.uint8)
    contours, _ = cv2.findContours(mask_bin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    objs: List[Dict] = []
    for contour in contours:
        if contour is None or len(contour) == 0:
            continue
        comp = np.zeros_like(mask_bin, dtype=np.uint8)
        cv2.drawContours(comp, [contour], -1, 1, thickness=-1)
        area = int(comp.sum())
        if area < min_area:
            continue
        bbox = get_bbox_from_2d_mask(comp)
        center = get_center_from_2d_mask(comp)
        if bbox is None or center is None:
            continue
        objs.append({"bbox": bbox, "center": center, "area": area, "mask": comp})
    objs.sort(key=lambda o: (o["bbox"][1], o["bbox"][0], o["bbox"][3], o["bbox"][2]))
    return objs


def clip_box_xyxy(box: np.ndarray, H: int, W: int) -> np.ndarray:
    box = np.asarray(box, dtype=np.float32).copy()
    box[0] = np.clip(box[0], 0, W - 1)
    box[2] = np.clip(box[2], 0, W - 1)
    box[1] = np.clip(box[1], 0, H - 1)
    box[3] = np.clip(box[3], 0, H - 1)
    if box[2] < box[0]:
        box[0], box[2] = box[2], box[0]
    if box[3] < box[1]:
        box[1], box[3] = box[3], box[1]
    return box


def expand_box_xyxy(box: np.ndarray, factor: float, H: int, W: int, min_size: float = 4.0) -> np.ndarray:
    box = np.asarray(box, dtype=np.float32)
    x1, y1, x2, y2 = box.tolist()
    cx = 0.5 * (x1 + x2)
    cy = 0.5 * (y1 + y2)
    bw = max(x2 - x1 + 1.0, min_size) * factor
    bh = max(y2 - y1 + 1.0, min_size) * factor
    out = np.array([cx - bw / 2.0, cy - bh / 2.0, cx + bw / 2.0, cy + bh / 2.0], dtype=np.float32)
    return clip_box_xyxy(out, H, W)


# =============================================================================
# Detector target building
# =============================================================================


def gaussian_radius_from_box(box: np.ndarray, min_radius: float = 1.0, max_radius: float = 4.0, scale: float = 0.25) -> float:
    x1, y1, x2, y2 = box.tolist()
    size = max(x2 - x1 + 1.0, y2 - y1 + 1.0)
    return float(np.clip(size * scale, min_radius, max_radius))


def draw_gaussian_heatmap(heatmap: np.ndarray, cx: float, cy: float, radius: float) -> None:
    H, W = heatmap.shape
    r = max(1, int(math.ceil(radius * 3)))
    x0 = max(0, int(round(cx)) - r)
    x1 = min(W - 1, int(round(cx)) + r)
    y0 = max(0, int(round(cy)) - r)
    y1 = min(H - 1, int(round(cy)) + r)
    if x1 < x0 or y1 < y0:
        return
    xs = np.arange(x0, x1 + 1, dtype=np.float32)
    ys = np.arange(y0, y1 + 1, dtype=np.float32)
    yy, xx = np.meshgrid(ys, xs, indexing="ij")
    g = np.exp(-((xx - cx) ** 2 + (yy - cy) ** 2) / (2 * radius**2))
    heatmap[y0 : y1 + 1, x0 : x1 + 1] = np.maximum(heatmap[y0 : y1 + 1, x0 : x1 + 1], g)


def build_detector_targets(
    boxes_xyxy: List[np.ndarray],
    image_hw: Tuple[int, int],
    out_hw: Tuple[int, int],
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Build heatmap and [l,t,r,b] box-distance targets on detector feature grid.

    Returns:
        heatmap: 1,oh,ow
        box_targets: 4,oh,ow, in image-pixel distances
        box_mask: 1,oh,ow, positive grid cells for box loss
    """
    H, W = image_hw
    oh, ow = out_hw
    sx = W / float(ow)
    sy = H / float(oh)

    heatmap = np.zeros((oh, ow), dtype=np.float32)
    box_t = np.zeros((4, oh, ow), dtype=np.float32)
    box_mask = np.zeros((oh, ow), dtype=np.float32)

    for box in boxes_xyxy:
        box = clip_box_xyxy(box, H, W)
        x1, y1, x2, y2 = box.tolist()
        cx = 0.5 * (x1 + x2)
        cy = 0.5 * (y1 + y2)
        gcx = cx / sx
        gcy = cy / sy
        radius = gaussian_radius_from_box(box) / max(sx, sy)
        draw_gaussian_heatmap(heatmap, gcx, gcy, radius=max(radius, 1.0))

        # Positive cells: grid cells whose image-space centers lie inside box.
        gx1 = max(0, int(math.floor(x1 / sx)))
        gx2 = min(ow - 1, int(math.ceil(x2 / sx)))
        gy1 = max(0, int(math.floor(y1 / sy)))
        gy2 = min(oh - 1, int(math.ceil(y2 / sy)))
        for gy in range(gy1, gy2 + 1):
            py = (gy + 0.5) * sy
            for gx in range(gx1, gx2 + 1):
                px = (gx + 0.5) * sx
                if x1 <= px <= x2 and y1 <= py <= y2:
                    box_mask[gy, gx] = 1.0
                    box_t[:, gy, gx] = np.array([px - x1, py - y1, x2 - px, y2 - py], dtype=np.float32)

        # Ensure at least center cell positive for very tiny boxes.
        cgx = int(np.clip(round(gcx), 0, ow - 1))
        cgy = int(np.clip(round(gcy), 0, oh - 1))
        px = (cgx + 0.5) * sx
        py = (cgy + 0.5) * sy
        box_mask[cgy, cgx] = 1.0
        box_t[:, cgy, cgx] = np.array(
            [max(px - x1, 0.0), max(py - y1, 0.0), max(x2 - px, 0.0), max(y2 - py, 0.0)],
            dtype=np.float32,
        )

    return (
        torch.from_numpy(heatmap[None]),
        torch.from_numpy(box_t),
        torch.from_numpy(box_mask[None]),
    )


# =============================================================================
# Dataset for detector training
# =============================================================================


@dataclass
class SliceSample:
    series_id: str
    z: int
    is_positive: bool


class LUNASliceDetectorDataset(Dataset):
    def __init__(
        self,
        index: DatasetIndex,
        split: str,
        val_ratio: float,
        seed: int,
        positive_fraction: float,
        neighbor_slices: int,
        max_negative_slices_per_case: int,
        use_triplet_channels: bool,
        hu_min: float,
        hu_max: float,
        box_expand: float,
        min_component_area: int,
        output_stride_hint: int = 16,
        cache_cases: bool = False,
        volume_ids_override: Optional[Sequence[str]] = None,
    ):
        self.index = index
        self.split = split
        self.use_triplet_channels = use_triplet_channels
        self.hu_min = hu_min
        self.hu_max = hu_max
        self.box_expand = box_expand
        self.min_component_area = min_component_area
        self.output_stride_hint = output_stride_hint
        self.cache_cases = cache_cases
        self._case_cache: Dict[str, CaseData] = {}

        if volume_ids_override is not None:
            self.volume_ids = list(volume_ids_override)
        else:
            # Backward-compatible fallback if dataset is constructed directly.
            splits = split_volume_ids(index.volume_ids, seed=seed, val_ratio=val_ratio, test_ratio=0.0)
            if split in splits:
                self.volume_ids = splits[split]
            else:
                raise ValueError(split)

        self.samples = self._build_samples(
            seed=seed,
            positive_fraction=positive_fraction,
            neighbor_slices=neighbor_slices,
            max_negative_slices_per_case=max_negative_slices_per_case,
        )
        if len(self.samples) == 0:
            raise RuntimeError(f"No samples built for split={split}")

    def _load_case(self, series_id: str) -> CaseData:
        if self.cache_cases and series_id in self._case_cache:
            return self._case_cache[series_id]
        case = load_case(self.index, series_id)
        if case is None:
            raise FileNotFoundError(f"Could not load case {series_id}")
        if self.cache_cases:
            self._case_cache[series_id] = case
        return case

    def _build_samples(
        self,
        seed: int,
        positive_fraction: float,
        neighbor_slices: int,
        max_negative_slices_per_case: int,
    ) -> List[SliceSample]:
        rng = np.random.default_rng(seed + (0 if self.split == "train" else 10000))
        pos: List[SliceSample] = []
        neg_pool: List[SliceSample] = []
        neigh: List[SliceSample] = []

        print(f"Building {self.split} slice index from {len(self.volume_ids)} volumes...")
        for series_id in tqdm(self.volume_ids, desc=f"index-{self.split}"):
            case = load_case(self.index, series_id)
            if case is None:
                continue
            gt = case.gt_volume
            positive_z = sorted(np.where(gt.reshape(gt.shape[0], -1).sum(axis=1) > 0)[0].tolist())
            positive_set = set(int(z) for z in positive_z)
            for z in positive_z:
                pos.append(SliceSample(series_id, int(z), True))
                for dz in range(-neighbor_slices, neighbor_slices + 1):
                    zz = int(z + dz)
                    if 0 <= zz < gt.shape[0] and zz not in positive_set:
                        neigh.append(SliceSample(series_id, zz, False))

            all_neg = [z for z in range(gt.shape[0]) if z not in positive_set]
            if len(all_neg) > 0:
                take = min(max_negative_slices_per_case, len(all_neg))
                selected = rng.choice(all_neg, size=take, replace=False)
                for z in selected:
                    neg_pool.append(SliceSample(series_id, int(z), False))

        if self.split == "train":
            desired_total = int(round(len(pos) / max(positive_fraction, 1e-6)))
            desired_neg = max(0, desired_total - len(pos))
            neg = neigh + neg_pool
            if len(neg) > desired_neg:
                idx = rng.choice(len(neg), size=desired_neg, replace=False)
                neg = [neg[i] for i in idx]
            samples = pos + neg
            rng.shuffle(samples)
        else:
            samples = pos + neigh + neg_pool

        print(f"{self.split}: {len(pos)} positive, {len(samples) - len(pos)} negative, total={len(samples)}")
        return samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict:
        s = self.samples[idx]
        case = self._load_case(s.series_id)
        image = case.image_array
        gt = case.gt_volume
        z = s.z
        rgb = build_sam2_input_slice(
            image,
            z,
            use_triplet_channels=self.use_triplet_channels,
            hu_min=self.hu_min,
            hu_max=self.hu_max,
        )
        x = numpy_rgb_to_tensor(rgb)
        H, W = gt.shape[1:]

        objs = extract_2d_objects(gt[z], min_area=self.min_component_area)
        boxes = [expand_box_xyxy(o["bbox"], self.box_expand, H, W) for o in objs]

        # We do not know the true SAM2 feature output size before model forward.
        # Build approximate targets for collation; train loop rebuilds exact targets
        # after seeing detector output size. Keep boxes as tensor here.
        boxes_arr = np.stack(boxes, axis=0).astype(np.float32) if boxes else np.zeros((0, 4), dtype=np.float32)
        return {
            "image": x,
            "boxes": torch.from_numpy(boxes_arr),
            "series_id": s.series_id,
            "z": int(z),
            "is_positive": bool(len(boxes) > 0),
            "image_hw": torch.tensor([H, W], dtype=torch.long),
        }


def collate_detector(batch: List[Dict]) -> Dict:
    return {
        "image": torch.stack([b["image"] for b in batch], dim=0),
        "boxes": [b["boxes"] for b in batch],
        "series_id": [b["series_id"] for b in batch],
        "z": torch.tensor([b["z"] for b in batch], dtype=torch.long),
        "is_positive": torch.tensor([b["is_positive"] for b in batch], dtype=torch.bool),
        "image_hw": torch.stack([b["image_hw"] for b in batch], dim=0),
    }


# =============================================================================
# SAM2 encoder detector
# =============================================================================


class ConvGNAct(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, groups: int = 8):
        super().__init__()
        g = min(groups, out_ch)
        while out_ch % g != 0 and g > 1:
            g -= 1
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
            nn.GroupNorm(g, out_ch),
            nn.SiLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class SAM2EncoderDetector(nn.Module):
    """
    Detector on top of SAM2 image encoder features.

    By default only the detector head is trainable. Use --unfreeze-encoder to fine-tune
    the SAM2 image encoder too, but that is not recommended for the first run.
    """

    def __init__(
        self,
        sam2_model: nn.Module,
        head_dim: int = 256,
        freeze_encoder: bool = True,
    ):
        super().__init__()
        self.sam2 = sam2_model
        self.freeze_encoder = freeze_encoder
        if freeze_encoder:
            for p in self.sam2.parameters():
                p.requires_grad = False

        # Lazy layers make this robust to different SAM2 feature dimensions.
        self.head = nn.Sequential(
            nn.LazyConv2d(head_dim, kernel_size=1),
            nn.GroupNorm(8, head_dim),
            nn.SiLU(inplace=True),
            ConvGNAct(head_dim, head_dim),
            ConvGNAct(head_dim, head_dim),
        )
        self.objectness = nn.Conv2d(head_dim, 1, kernel_size=1)
        self.box_dist = nn.Conv2d(head_dim, 4, kernel_size=1)

    def extract_sam2_feature(self, x: torch.Tensor) -> torch.Tensor:
        """Return last SAM2 image feature as B,C,Hf,Wf."""
        # SAM2 expects images normalized internally in forward_image for the standard predictor path.
        if self.freeze_encoder:
            with torch.no_grad():
                backbone_out = self.sam2.forward_image(x)
                _, vision_feats, _, feat_sizes = self.sam2._prepare_backbone_features(backbone_out)
        else:
            backbone_out = self.sam2.forward_image(x)
            _, vision_feats, _, feat_sizes = self.sam2._prepare_backbone_features(backbone_out)

        feat = vision_feats[-1]
        b = x.shape[0]
        c = feat.shape[-1]
        h, w = feat_sizes[-1]
        feat = feat.permute(1, 2, 0).reshape(b, c, h, w).contiguous()
        return feat

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        feat = self.extract_sam2_feature(x)
        y = self.head(feat)
        obj_logits = self.objectness(y)
        # Softplus keeps distances positive. Add epsilon to avoid zero-width boxes.
        box_dist = F.softplus(self.box_dist(y)) + 1e-3
        return {"objectness_logits": obj_logits, "box_dist": box_dist}


# =============================================================================
# Losses and decoding
# =============================================================================


def focal_heatmap_loss(logits: torch.Tensor, targets: torch.Tensor, alpha: float = 2.0, beta: float = 4.0) -> torch.Tensor:
    """CenterNet-style heatmap focal loss for Gaussian objectness maps."""
    pred = torch.sigmoid(logits).clamp(1e-4, 1.0 - 1e-4)
    pos_inds = targets.eq(1.0).float()
    neg_inds = targets.lt(1.0).float()
    neg_weights = torch.pow(1.0 - targets, beta)

    pos_loss = -torch.log(pred) * torch.pow(1.0 - pred, alpha) * pos_inds
    neg_loss = -torch.log(1.0 - pred) * torch.pow(pred, alpha) * neg_weights * neg_inds

    num_pos = pos_inds.sum().clamp(min=1.0)
    return (pos_loss.sum() + neg_loss.sum()) / num_pos


def build_batch_targets(
    boxes_list: List[torch.Tensor],
    image_hw_list: torch.Tensor,
    out_hw: Tuple[int, int],
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    heatmaps, box_targets, box_masks = [], [], []
    for boxes, hw in zip(boxes_list, image_hw_list):
        H, W = int(hw[0].item()), int(hw[1].item())
        boxes_np = [b.detach().cpu().numpy().astype(np.float32) for b in boxes]
        hm, bt, bm = build_detector_targets(boxes_np, (H, W), out_hw)
        heatmaps.append(hm)
        box_targets.append(bt)
        box_masks.append(bm)
    return (
        torch.stack(heatmaps).to(device),
        torch.stack(box_targets).to(device),
        torch.stack(box_masks).to(device),
    )


def detector_loss(
    outputs: Dict[str, torch.Tensor],
    heatmap_t: torch.Tensor,
    box_t: torch.Tensor,
    box_mask: torch.Tensor,
    box_loss_weight: float,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    obj_logits = outputs["objectness_logits"]
    box_pred = outputs["box_dist"]

    obj_loss = focal_heatmap_loss(obj_logits, heatmap_t)
    if box_mask.sum() > 0:
        bm = box_mask.expand_as(box_pred)
        reg_loss = F.smooth_l1_loss(box_pred * bm, box_t * bm, reduction="sum") / bm.sum().clamp(min=1.0)
    else:
        reg_loss = box_pred.sum() * 0.0
    total = obj_loss + box_loss_weight * reg_loss
    return total, {"loss": float(total.detach()), "obj_loss": float(obj_loss.detach()), "box_loss": float(reg_loss.detach())}


@dataclass
class Candidate:
    series_id: str
    z: int
    score: float
    x: float
    y: float
    box: np.ndarray


def nms_boxes(boxes: np.ndarray, scores: np.ndarray, iou_thresh: float) -> List[int]:
    if len(boxes) == 0:
        return []
    x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    areas = np.maximum(0, x2 - x1 + 1) * np.maximum(0, y2 - y1 + 1)
    order = scores.argsort()[::-1]
    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(int(i))
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        w = np.maximum(0, xx2 - xx1 + 1)
        h = np.maximum(0, yy2 - yy1 + 1)
        inter = w * h
        iou = inter / np.maximum(areas[i] + areas[order[1:]] - inter, 1e-6)
        inds = np.where(iou <= iou_thresh)[0]
        order = order[inds + 1]
    return keep


def decode_detector_outputs(
    outputs: Dict[str, torch.Tensor],
    series_ids: List[str],
    z_values: Sequence[int],
    image_hw: Tuple[int, int],
    score_thresh: float,
    topk_per_slice: int,
    nms_iou: float,
) -> List[Candidate]:
    obj = torch.sigmoid(outputs["objectness_logits"]).detach().float().cpu().numpy()
    dist = outputs["box_dist"].detach().float().cpu().numpy()
    B, _, oh, ow = obj.shape
    H, W = image_hw
    sx = W / float(ow)
    sy = H / float(oh)
    candidates: List[Candidate] = []

    for b in range(B):
        scores_map = obj[b, 0]
        flat = scores_map.reshape(-1)
        idxs = np.where(flat >= score_thresh)[0]
        if len(idxs) == 0:
            # Keep the best location as a fallback if threshold is too strict.
            idxs = np.array([int(flat.argmax())], dtype=np.int64)
        idxs = idxs[np.argsort(flat[idxs])[::-1]][: max(topk_per_slice * 4, topk_per_slice)]

        boxes = []
        scores = []
        centers = []
        for idx in idxs:
            gy = int(idx // ow)
            gx = int(idx % ow)
            score = float(scores_map[gy, gx])
            px = (gx + 0.5) * sx
            py = (gy + 0.5) * sy
            l, t, r, bb = dist[b, :, gy, gx].tolist()
            box = np.array([px - l, py - t, px + r, py + bb], dtype=np.float32)
            box = clip_box_xyxy(box, H, W)
            # Filter pathological boxes.
            if (box[2] - box[0] + 1) < 2 or (box[3] - box[1] + 1) < 2:
                continue
            boxes.append(box)
            scores.append(score)
            centers.append((px, py))

        if boxes:
            boxes_arr = np.stack(boxes)
            scores_arr = np.array(scores, dtype=np.float32)
            keep = nms_boxes(boxes_arr, scores_arr, nms_iou)[:topk_per_slice]
            for k in keep:
                cx, cy = centers[k]
                candidates.append(
                    Candidate(
                        series_id=series_ids[b],
                        z=int(z_values[b]),
                        score=float(scores_arr[k]),
                        x=float(cx),
                        y=float(cy),
                        box=boxes_arr[k].astype(np.float32),
                    )
                )
    return candidates


# =============================================================================
# Training and validation
# =============================================================================


def build_detector_model(args: argparse.Namespace, device: torch.device) -> SAM2EncoderDetector:
    sam2 = build_sam2(args.model_cfg, args.checkpoint, device=device)
    model = SAM2EncoderDetector(
        sam2_model=sam2,
        head_dim=args.detector_head_dim,
        freeze_encoder=not args.unfreeze_encoder,
    ).to(device)
    return model


def make_output_dir(args: argparse.Namespace, command: str) -> Path:
    out = Path(args.output_dir).expanduser().resolve()
    if args.create_experiment_dir:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        triplet = "triplet" if getattr(args, "use_triplet_channels", False) else "singlech"
        hdim = getattr(args, "detector_head_dim", "na")
        bs = getattr(args, "batch_size", getattr(args, "eval_batch_size", "na"))
        ep = getattr(args, "epochs", "na")
        lr = str(getattr(args, "lr", "na")).replace(".", "p")
        val = str(getattr(args, "val_ratio", "na")).replace(".", "p")
        test = str(getattr(args, "test_ratio", "na")).replace(".", "p")
        pos = str(getattr(args, "positive_fraction", "na")).replace(".", "p")
        max_cases = getattr(args, "max_cases", None)
        parts = [stamp, command, "medsam2det2d", triplet, f"hd{hdim}", f"ep{ep}", f"bs{bs}", f"lr{lr}", f"val{val}", f"test{test}", f"pos{pos}"]
        if max_cases is not None:
            parts.append(f"max{max_cases}")
        name = args.experiment_name or "_".join(map(str, parts))
        out = out / slugify(name)
    out.mkdir(parents=True, exist_ok=args.overwrite_experiment)
    return out


def save_config(args: argparse.Namespace, out_dir: Path, command: str, index: DatasetIndex) -> None:
    cfg = {
        "command": command,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "hostname": socket.gethostname(),
        "cwd": os.getcwd(),
        "argv": os.sys.argv,
        "args": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
        "dataset": {
            "dataset_dir": str(index.dataset_dir),
            "volumes_dir": str(index.volumes_dir),
            "masks_dir": str(index.masks_dir),
            "n_volumes": len(index.volume_ids),
        },
    }
    with open(out_dir / "config.json", "w") as f:
        json.dump(cfg, f, indent=2)


def run_epoch(
    model: SAM2EncoderDetector,
    loader: DataLoader,
    optimizer: Optional[torch.optim.Optimizer],
    scaler: Optional[torch.cuda.amp.GradScaler],
    device: torch.device,
    amp_dtype: str,
    box_loss_weight: float,
    grad_clip_norm: float,
    desc: str,
) -> Dict[str, float]:
    is_train = optimizer is not None
    model.train(is_train)
    totals = {"loss": 0.0, "obj_loss": 0.0, "box_loss": 0.0}
    n = 0

    pbar = tqdm(loader, desc=desc, leave=False)
    for batch in pbar:
        images = batch["image"].to(device, non_blocking=True)
        if is_train:
            optimizer.zero_grad(set_to_none=True)
        ctx = safe_autocast(device, amp_dtype)
        with torch.set_grad_enabled(is_train), ctx:
            outputs = model(images)
            _, _, oh, ow = outputs["objectness_logits"].shape
            heat_t, box_t, box_m = build_batch_targets(batch["boxes"], batch["image_hw"], (oh, ow), device)
            loss, logs = detector_loss(outputs, heat_t, box_t, box_m, box_loss_weight=box_loss_weight)

        if is_train:
            if scaler is not None and amp_dtype == "fp16":
                scaler.scale(loss).backward()
                if grad_clip_norm > 0:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                if grad_clip_norm > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
                optimizer.step()

        bs = images.shape[0]
        for k in totals:
            totals[k] += logs[k] * bs
        n += bs
        pbar.set_postfix({k: f"{totals[k]/max(n,1):.4f}" for k in totals})

    return {k: totals[k] / max(n, 1) for k in totals}


def save_checkpoint(
    path: Path,
    model: SAM2EncoderDetector,
    optimizer: Optional[torch.optim.Optimizer],
    epoch: int,
    best_val: float,
    args: argparse.Namespace,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "epoch": epoch,
        "best_val": best_val,
        "model_state": model.state_dict(),
        "args": vars(args),
    }
    if optimizer is not None:
        payload["optimizer_state"] = optimizer.state_dict()
    torch.save(payload, path)


def load_detector_checkpoint(model: SAM2EncoderDetector, ckpt_path: str, device: torch.device) -> Dict:
    ckpt = torch.load(ckpt_path, map_location=device)
    missing, unexpected = model.load_state_dict(ckpt["model_state"], strict=False)
    if missing:
        print(f"WARNING missing keys while loading detector: {missing[:10]}{'...' if len(missing)>10 else ''}")
    if unexpected:
        print(f"WARNING unexpected keys while loading detector: {unexpected[:10]}{'...' if len(unexpected)>10 else ''}")
    return ckpt


def run_test_prediction_after_training(args: argparse.Namespace, train_out_dir: Path) -> None:
    """Run detector-guided SAM2 prediction on the saved test split after training."""
    if not getattr(args, "run_test_after_training", True):
        print("Skipping automatic test prediction because --no-run-test-after-training was used.")
        return

    best_ckpt = train_out_dir / "best_detector.pt"
    if not best_ckpt.exists():
        print(f"WARNING: {best_ckpt} does not exist; skipping automatic test prediction.")
        return

    test_list = train_out_dir / "splits" / "test.txt"
    test_ids = read_case_list_file(str(test_list)) if test_list.exists() else []
    if not test_ids:
        print("WARNING: test split is empty; skipping automatic test prediction.")
        return

    test_args = argparse.Namespace(**vars(args))
    test_args.command = "infer-detect-sam2"
    test_args.detector_checkpoint = str(best_ckpt)
    test_args.output_dir = str(train_out_dir / "test_predictions")
    test_args.create_experiment_dir = False
    test_args.overwrite_experiment = True
    test_args.eval_split = "test"
    test_args.train_case_list = str(train_out_dir / "splits" / "train.txt")
    test_args.val_case_list = str(train_out_dir / "splits" / "val.txt")
    test_args.test_case_list = str(test_list)

    # Defaults for the inference-only arguments, unless the user provided training-time overrides.
    test_args.eval_batch_size = getattr(args, "test_eval_batch_size", 16)
    test_args.det_score_thresh = getattr(args, "test_det_score_thresh", 0.20)
    test_args.topk_per_slice = getattr(args, "test_topk_per_slice", 3)
    test_args.max_candidates_per_volume = getattr(args, "test_max_candidates_per_volume", 20)
    test_args.nms_iou = getattr(args, "test_nms_iou", 0.30)
    test_args.eval_positive_slices_only = getattr(args, "test_eval_positive_slices_only", False)
    test_args.sam2_prompt_mode = getattr(args, "test_sam2_prompt_mode", "point+box")
    test_args.sam2_multimask = getattr(args, "test_sam2_multimask", False)
    test_args.min_sam2_mask_area = getattr(args, "test_min_sam2_mask_area", 3)
    test_args.save_volumes = getattr(args, "test_save_volumes", True)
    test_args.save_gt_volume = getattr(args, "test_save_gt_volume", False)
    test_args.fail_fast = getattr(args, "test_fail_fast", False)

    print(f"Running automatic test prediction on {len(test_ids)} test volumes. Output: {test_args.output_dir}")
    infer_detect_sam2(test_args)



def _post_train_common_infer_args_2d(args: argparse.Namespace, out_dir: Path, test_file: Path, ckpt_path: Path, output_dir: Path, command: str) -> argparse.Namespace:
    """Build an inference Namespace for automatic post-training test evaluation."""
    pred_args = argparse.Namespace(**vars(args))
    pred_args.command = command
    pred_args.detector_checkpoint = str(ckpt_path)
    pred_args.output_dir = str(output_dir)
    pred_args.create_experiment_dir = False
    pred_args.overwrite_experiment = True
    pred_args.eval_split = "test"
    pred_args.train_case_list = str(out_dir / "splits" / "train.txt")
    pred_args.val_case_list = str(out_dir / "splits" / "val.txt")
    pred_args.test_case_list = str(test_file)

    pred_args.eval_batch_size = int(getattr(args, "post_train_eval_batch_size", 16))
    pred_args.det_score_thresh = float(getattr(args, "post_train_det_score_thresh", 0.20))
    pred_args.topk_per_slice = int(getattr(args, "post_train_topk_per_slice", 3))
    pred_args.max_candidates_per_volume = int(getattr(args, "post_train_max_candidates_per_volume", 20))
    pred_args.nms_iou = float(getattr(args, "post_train_nms_iou", 0.30))
    pred_args.eval_positive_slices_only = bool(getattr(args, "post_train_eval_positive_slices_only", False))
    pred_args.fail_fast = bool(getattr(args, "post_train_fail_fast", False))
    return pred_args


def _copy_if_exists(src: Path, dst: Path) -> None:
    if src.exists():
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)


def run_test_predictions_after_training(args: argparse.Namespace, out_dir: Path) -> None:
    """
    Run four automatic test evaluations after training:
      1) best weights -> detector-only metrics/candidates
      2) best weights -> detector+SAM2 segmentation metrics/volumes
      3) last weights -> detector-only metrics/candidates
      4) last weights -> detector+SAM2 segmentation metrics/volumes

    Output layout:
      test_predictions/{best_weights,last_weights}/detector/
          detector_metrics.csv, candidates.csv, detection_summary.*
      test_predictions/{best_weights,last_weights}/segmentation/
          segmentation_metrics.csv, nodule_metrics.csv, predicted_volumes/
    """
    if not getattr(args, "run_test_after_training", True):
        print("Skipping post-training test evaluations because --no-run-test-after-training was set.")
        return
    test_file = out_dir / "splits" / "test.txt"
    if not test_file.exists() or not test_file.read_text().strip():
        print("Skipping post-training test evaluations because the test split is empty.")
        return

    runs = [
        ("best_weights", out_dir / "best_detector.pt"),
        ("last_weights", out_dir / "last_detector.pt"),
    ]

    for tag, ckpt_path in runs:
        if not ckpt_path.exists():
            print(f"WARNING: skipping {tag}; checkpoint not found: {ckpt_path}")
            continue

        root = out_dir / "test_predictions" / tag
        detector_dir = root / "detector"
        segmentation_dir = root / "segmentation"

        det_args = _post_train_common_infer_args_2d(args, out_dir, test_file, ckpt_path, detector_dir, "infer-detector")
        print(f"Running post-training DETECTOR test evaluation with {tag}: {ckpt_path}")
        infer_detector(det_args)
        _copy_if_exists(detector_dir / "summary.csv", detector_dir / "detector_metrics.csv")

        seg_args = _post_train_common_infer_args_2d(args, out_dir, test_file, ckpt_path, segmentation_dir, "infer-detect-sam2")
        seg_args.sam2_prompt_mode = str(getattr(args, "post_train_sam2_prompt_mode", "point+box"))
        seg_args.sam2_multimask = bool(getattr(args, "post_train_sam2_multimask", False))
        seg_args.min_sam2_mask_area = int(getattr(args, "post_train_min_sam2_mask_area", 3))
        seg_args.save_volumes = bool(getattr(args, "post_train_save_volumes", True))
        seg_args.save_gt_volume = bool(getattr(args, "post_train_save_gt_volume", False))
        print(f"Running post-training SEGMENTATION test evaluation with {tag}: {ckpt_path}")
        infer_detect_sam2(seg_args)
        _copy_if_exists(segmentation_dir / "metrics.csv", segmentation_dir / "segmentation_metrics.csv")

def train_detector(args: argparse.Namespace) -> None:
    seed_everything(args.seed)
    device = select_device(args.device)
    configure_torch(device, args.allow_tf32)
    out_dir = make_output_dir(args, "train-detector")
    index = build_dataset_index(args)
    splits = build_splits(index, args)
    write_split_files(splits, out_dir)
    print(f"Split sizes: train={len(splits['train'])}, val={len(splits['val'])}, test={len(splits['test'])}, all={len(splits['all'])}")
    save_config(args, out_dir, "train-detector", index)

    train_ds = LUNASliceDetectorDataset(
        index=index,
        split="train",
        val_ratio=args.val_ratio,
        seed=args.seed,
        volume_ids_override=splits["train"],
        positive_fraction=args.positive_fraction,
        neighbor_slices=args.neighbor_slices,
        max_negative_slices_per_case=args.max_negative_slices_per_case,
        use_triplet_channels=args.use_triplet_channels,
        hu_min=args.hu_min,
        hu_max=args.hu_max,
        box_expand=args.box_expand,
        min_component_area=args.min_component_area,
        cache_cases=args.cache_cases,
    )
    val_ds = LUNASliceDetectorDataset(
        index=index,
        split="val",
        val_ratio=args.val_ratio,
        seed=args.seed,
        volume_ids_override=splits["val"] if splits["val"] else splits["train"],
        positive_fraction=args.positive_fraction,
        neighbor_slices=args.neighbor_slices,
        max_negative_slices_per_case=args.max_negative_slices_per_case,
        use_triplet_channels=args.use_triplet_channels,
        hu_min=args.hu_min,
        hu_max=args.hu_max,
        box_expand=args.box_expand,
        min_component_area=args.min_component_area,
        cache_cases=args.cache_cases,
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
        collate_fn=collate_detector,
        persistent_workers=args.num_workers > 0,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
        collate_fn=collate_detector,
        persistent_workers=args.num_workers > 0,
    )

    model = build_detector_model(args, device)
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(args.epochs, 1))
    scaler = torch.cuda.amp.GradScaler(enabled=(device.type == "cuda" and args.amp_dtype == "fp16"))

    best_val = float("inf")
    bad_epochs = 0
    metrics_rows = []
    print(f"Training detector. Output: {out_dir}")
    for epoch in range(1, args.epochs + 1):
        train_log = run_epoch(
            model,
            train_loader,
            optimizer,
            scaler,
            device,
            args.amp_dtype,
            args.box_loss_weight,
            args.grad_clip_norm,
            desc=f"train {epoch}/{args.epochs}",
        )
        with torch.no_grad():
            val_log = run_epoch(
                model,
                val_loader,
                None,
                None,
                device,
                args.amp_dtype,
                args.box_loss_weight,
                args.grad_clip_norm,
                desc=f"val {epoch}/{args.epochs}",
            )
        scheduler.step()
        row = {"epoch": epoch, **{f"train_{k}": v for k, v in train_log.items()}, **{f"val_{k}": v for k, v in val_log.items()}, "lr": optimizer.param_groups[0]["lr"]}
        metrics_rows.append(row)
        pd.DataFrame(metrics_rows).to_csv(out_dir / "metrics.csv", index=False)
        with open(out_dir / "metrics.json", "w") as f:
            json.dump(metrics_rows, f, indent=2)

        print(
            f"Epoch {epoch:03d}: train_loss={train_log['loss']:.5f}, "
            f"val_loss={val_log['loss']:.5f}, lr={optimizer.param_groups[0]['lr']:.3e}"
        )
        save_checkpoint(out_dir / "last_detector.pt", model, optimizer, epoch, best_val, args)
        if val_log["loss"] < best_val - args.min_delta:
            best_val = val_log["loss"]
            bad_epochs = 0
            save_checkpoint(out_dir / "best_detector.pt", model, optimizer, epoch, best_val, args)
            print(f"  saved best_detector.pt with val_loss={best_val:.5f}")
        else:
            bad_epochs += 1
            if args.patience > 0 and bad_epochs >= args.patience:
                print(f"Early stopping after {bad_epochs} bad epochs.")
                break

    plot_training_metrics(out_dir / "metrics.csv", out_dir)
    print(f"Done. Best detector: {out_dir / 'best_detector.pt'}")
    run_test_predictions_after_training(args, out_dir)


# =============================================================================
# Detector inference and SAM2 refinement
# =============================================================================


class CaseSliceDataset(Dataset):
    def __init__(
        self,
        case: CaseData,
        use_triplet_channels: bool,
        hu_min: float,
        hu_max: float,
        positive_slices_only: bool = False,
        min_component_area: int = 1,
    ):
        self.case = case
        self.use_triplet_channels = use_triplet_channels
        self.hu_min = hu_min
        self.hu_max = hu_max
        if positive_slices_only:
            zs = np.where(case.gt_volume.reshape(case.gt_volume.shape[0], -1).sum(axis=1) >= min_component_area)[0].tolist()
            self.zs = [int(z) for z in zs]
        else:
            self.zs = list(range(case.image_array.shape[0]))

    def __len__(self) -> int:
        return len(self.zs)

    def __getitem__(self, i: int) -> Dict:
        z = self.zs[i]
        rgb = build_sam2_input_slice(
            self.case.image_array,
            z,
            use_triplet_channels=self.use_triplet_channels,
            hu_min=self.hu_min,
            hu_max=self.hu_max,
        )
        return {
            "image": numpy_rgb_to_tensor(rgb),
            "z": int(z),
            "series_id": self.case.series_id,
        }


def collate_case_slices(batch: List[Dict]) -> Dict:
    return {
        "image": torch.stack([b["image"] for b in batch], dim=0),
        "z": torch.tensor([b["z"] for b in batch], dtype=torch.long),
        "series_id": [b["series_id"] for b in batch],
    }


def run_detector_on_case(
    model: SAM2EncoderDetector,
    case: CaseData,
    args: argparse.Namespace,
    device: torch.device,
) -> List[Candidate]:
    ds = CaseSliceDataset(
        case,
        use_triplet_channels=args.use_triplet_channels,
        hu_min=args.hu_min,
        hu_max=args.hu_max,
        positive_slices_only=args.eval_positive_slices_only,
        min_component_area=args.min_component_area,
    )
    loader = DataLoader(
        ds,
        batch_size=args.eval_batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
        collate_fn=collate_case_slices,
    )
    H, W = case.gt_volume.shape[1:]
    all_candidates: List[Candidate] = []
    model.eval()
    with torch.no_grad():
        for batch in loader:
            x = batch["image"].to(device, non_blocking=True)
            with safe_autocast(device, args.amp_dtype):
                outputs = model(x)
            cand = decode_detector_outputs(
                outputs,
                series_ids=batch["series_id"],
                z_values=batch["z"].tolist(),
                image_hw=(H, W),
                score_thresh=args.det_score_thresh,
                topk_per_slice=args.topk_per_slice,
                nms_iou=args.nms_iou,
            )
            all_candidates.extend(cand)
    all_candidates.sort(key=lambda c: c.score, reverse=True)
    if args.max_candidates_per_volume > 0:
        all_candidates = all_candidates[: args.max_candidates_per_volume]
    return all_candidates


def save_candidates_csv(candidates_by_case: Dict[str, List[Candidate]], out_csv: Path) -> None:
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    for series_id, cands in candidates_by_case.items():
        for rank, c in enumerate(cands, start=1):
            rows.append(
                {
                    "VolumeID": series_id,
                    "rank": rank,
                    "z": c.z,
                    "score": c.score,
                    "x": c.x,
                    "y": c.y,
                    "x1": float(c.box[0]),
                    "y1": float(c.box[1]),
                    "x2": float(c.box[2]),
                    "y2": float(c.box[3]),
                }
            )
    pd.DataFrame(rows).to_csv(out_csv, index=False)


def infer_detector(args: argparse.Namespace) -> None:
    seed_everything(args.seed)
    device = select_device(args.device)
    configure_torch(device, args.allow_tf32)
    out_dir = make_output_dir(args, "infer-detector")
    index = build_dataset_index(args)
    save_config(args, out_dir, "infer-detector", index)

    model = build_detector_model(args, device)
    load_detector_checkpoint(model, args.detector_checkpoint, device)
    model.eval()

    eval_volume_ids = select_eval_volume_ids(index, args)
    print(f"Evaluating split={args.eval_split} with {len(eval_volume_ids)} volumes")
    candidates_by_case: Dict[str, List[Candidate]] = {}
    rows = []
    for series_id in tqdm(eval_volume_ids, desc="cases"):
        case = load_case(index, series_id)
        if case is None:
            rows.append({"VolumeID": series_id, "status": "missing"})
            continue
        cands = run_detector_on_case(model, case, args, device)
        candidates_by_case[series_id] = cands
        spacing_zyx = get_spacing_zyx(case.image_itk)
        det_metrics = detection_localization_metrics(case.gt_volume, cands, spacing_zyx, min_area=args.min_component_area)
        rows.append({"VolumeID": series_id, "status": "ok", **det_metrics})
    save_candidates_csv(candidates_by_case, out_dir / "candidates.csv")
    pd.DataFrame(rows).to_csv(out_dir / "summary.csv", index=False)
    save_detection_summary(rows, out_dir)
    print(f"Saved: {out_dir / 'candidates.csv'}")


def merge_candidate_sam2_masks(
    case: CaseData,
    candidates: List[Candidate],
    predictor: SAM2ImagePredictor,
    args: argparse.Namespace,
    device: torch.device,
) -> Tuple[np.ndarray, List[Dict]]:
    Z, H, W = case.gt_volume.shape
    pred = np.zeros((Z, H, W), dtype=np.uint8)
    logs: List[Dict] = []
    by_z: Dict[int, List[Candidate]] = {}
    for c in candidates:
        by_z.setdefault(c.z, []).append(c)

    for z in sorted(by_z.keys()):
        rgb = build_sam2_input_slice(
            case.image_array,
            z,
            use_triplet_channels=args.use_triplet_channels,
            hu_min=args.hu_min,
            hu_max=args.hu_max,
        )
        with torch.inference_mode(), safe_autocast(device, args.amp_dtype):
            predictor.set_image(rgb)
            slice_mask = np.zeros((H, W), dtype=bool)
            for c in by_z[z]:
                point = np.array([[c.x, c.y]], dtype=np.float32)
                labels = np.array([1], dtype=np.int32)
                box = c.box.astype(np.float32)
                if args.sam2_prompt_mode == "point":
                    masks, scores, _ = predictor.predict(point_coords=point, point_labels=labels, multimask_output=args.sam2_multimask)
                elif args.sam2_prompt_mode == "box":
                    masks, scores, _ = predictor.predict(point_coords=None, point_labels=None, box=box, multimask_output=args.sam2_multimask)
                elif args.sam2_prompt_mode == "point+box":
                    masks, scores, _ = predictor.predict(point_coords=point, point_labels=labels, box=box, multimask_output=args.sam2_multimask)
                else:
                    raise ValueError(args.sam2_prompt_mode)
                best_i = int(np.argmax(scores)) if args.sam2_multimask else 0
                mask = masks[best_i].astype(bool)
                if args.min_sam2_mask_area > 0 and int(mask.sum()) < args.min_sam2_mask_area:
                    keep = False
                else:
                    keep = True
                if keep:
                    slice_mask |= mask
                logs.append(
                    {
                        "VolumeID": case.series_id,
                        "z": z,
                        "det_score": c.score,
                        "sam2_score": float(scores[best_i]),
                        "mask_area": int(mask.sum()),
                        "kept": keep,
                        "x": c.x,
                        "y": c.y,
                        "x1": float(c.box[0]),
                        "y1": float(c.box[1]),
                        "x2": float(c.box[2]),
                        "y2": float(c.box[3]),
                    }
                )
            pred[z] = slice_mask.astype(np.uint8)
    return pred, logs


def infer_detect_sam2(args: argparse.Namespace) -> None:
    seed_everything(args.seed)
    device = select_device(args.device)
    configure_torch(device, args.allow_tf32)
    out_dir = make_output_dir(args, "infer-detect-sam2")
    index = build_dataset_index(args)
    save_config(args, out_dir, "infer-detect-sam2", index)

    detector = build_detector_model(args, device)
    load_detector_checkpoint(detector, args.detector_checkpoint, device)
    detector.eval()

    sam2_model = build_sam2(args.model_cfg, args.checkpoint, device=device)
    predictor = SAM2ImagePredictor(sam2_model)

    rows = []
    prompt_rows = []
    nodule_rows = []
    pred_root = out_dir / "predicted_volumes"
    eval_volume_ids = select_eval_volume_ids(index, args)
    print(f"Evaluating split={args.eval_split} with {len(eval_volume_ids)} volumes")

    for series_id in tqdm(eval_volume_ids, desc="cases"):
        try:
            case = load_case(index, series_id)
            if case is None:
                rows.append({"VolumeID": series_id, "status": "missing"})
                continue
            cands = run_detector_on_case(detector, case, args, device)
            pred, logs = merge_candidate_sam2_masks(case, cands, predictor, args, device)
            spacing_zyx = get_spacing_zyx(case.image_itk)
            row = {
                "VolumeID": series_id,
                "status": "ok",
                **detection_localization_metrics(case.gt_volume, cands, spacing_zyx, min_area=args.min_component_area),
                **segmentation_metrics(case.gt_volume, pred, spacing_zyx),
            }
            rows.append(row)
            prompt_rows.extend(logs)
            nodule_rows.extend(per_nodule_metrics(case, pred, cands, min_area=args.min_component_area))
            if args.save_volumes:
                write_pred_volume(pred, case.image_itk, pred_root / "detect_medsam2" / f"{series_id}_detect_medsam2.nii.gz")
                if args.save_gt_volume:
                    write_pred_volume(case.gt_volume, case.image_itk, pred_root / "gt" / f"{series_id}_gt.nii.gz")
        except Exception as exc:
            if args.fail_fast:
                raise
            rows.append({"VolumeID": series_id, "status": "error", "error": repr(exc)})
            print(f"ERROR {series_id}: {repr(exc)}")
        pd.DataFrame(rows).to_csv(out_dir / "metrics.csv", index=False)
        if prompt_rows:
            pd.DataFrame(prompt_rows).to_csv(out_dir / "prompt_logs.csv", index=False)
        if nodule_rows:
            pd.DataFrame(nodule_rows).to_csv(out_dir / "nodule_metrics.csv", index=False)
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()

    df = pd.DataFrame(rows)
    df.to_csv(out_dir / "metrics.csv", index=False)
    if prompt_rows:
        pd.DataFrame(prompt_rows).to_csv(out_dir / "prompt_logs.csv", index=False)
    if nodule_rows:
        pd.DataFrame(nodule_rows).to_csv(out_dir / "nodule_metrics.csv", index=False)
    save_detection_summary(rows, out_dir)
    ok = df[df["status"] == "ok"]
    print(f"Saved metrics: {out_dir / 'metrics.csv'}")
    if len(ok) > 0:
        print(f"Mean DSC: {pd.to_numeric(ok['DSC'], errors='coerce').mean():.6f}")


# =============================================================================
# CLI
# =============================================================================


def add_common_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--dataset-dir", required=True)
    parser.add_argument("--volumes-dir", default=None)
    parser.add_argument("--masks-dir", default=None)
    parser.add_argument("--annotations-csv", default=None)
    parser.add_argument("--links-csv", default=None)
    parser.add_argument("--case-list", default=None, help="Optional global case filter applied before train/val/test splitting.")
    parser.add_argument("--train-case-list", default=None, help="Explicit train split SeriesUIDs, one per line.")
    parser.add_argument("--val-case-list", default=None, help="Explicit validation split SeriesUIDs, one per line.")
    parser.add_argument("--test-case-list", default=None, help="Explicit test split SeriesUIDs, one per line.")
    parser.add_argument("--val-ratio", type=float, default=0.10, help="Validation volume ratio when explicit split files are not used.")
    parser.add_argument("--test-ratio", type=float, default=0.10, help="Test volume ratio when explicit split files are not used.")
    parser.add_argument("--eval-split", choices=["all", "train", "val", "test"], default="all", help="Split used by inference/evaluation commands.")
    parser.add_argument("--shuffle-splits", action=argparse.BooleanOptionalAction, default=True, help="Shuffle before ratio-based train/val/test splitting.")
    parser.add_argument("--only-annotated", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--dataset-fraction", type=float, default=None)
    parser.add_argument("--max-cases", type=int, default=None)
    parser.add_argument("--shuffle", action="store_true", help="Shuffle global case order before optional dataset_fraction/max_cases filtering.")
    parser.add_argument("--seed", type=int, default=123)

    parser.add_argument("--model-cfg", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--amp-dtype", choices=["bf16", "fp16", "none"], default="bf16")
    parser.add_argument("--allow-tf32", action=argparse.BooleanOptionalAction, default=True)

    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--create-experiment-dir", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--experiment-name", default=None)
    parser.add_argument("--overwrite-experiment", action="store_true")

    parser.add_argument("--use-triplet-channels", action="store_true")
    parser.add_argument("--hu-min", type=float, default=-1000.0)
    parser.add_argument("--hu-max", type=float, default=400.0)
    parser.add_argument("--min-component-area", type=int, default=1)
    parser.add_argument("--box-expand", type=float, default=1.5)

    parser.add_argument("--detector-head-dim", type=int, default=256)
    parser.add_argument("--unfreeze-encoder", action="store_true")
    parser.add_argument("--num-workers", type=int, default=4)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Detector-guided MedSAM2 for LUNA nodule segmentation")
    sub = parser.add_subparsers(dest="command", required=True)

    p_train = sub.add_parser("train-detector", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    add_common_args(p_train)
    p_train.add_argument("--epochs", type=int, default=30)
    p_train.add_argument("--batch-size", type=int, default=8)
    p_train.add_argument("--lr", type=float, default=1e-4)
    p_train.add_argument("--weight-decay", type=float, default=1e-4)
    p_train.add_argument("--positive-fraction", type=float, default=0.50)
    p_train.add_argument("--neighbor-slices", type=int, default=2)
    p_train.add_argument("--max-negative-slices-per-case", type=int, default=20)
    p_train.add_argument("--box-loss-weight", type=float, default=0.25)
    p_train.add_argument("--grad-clip-norm", type=float, default=1.0)
    p_train.add_argument("--patience", type=int, default=10)
    p_train.add_argument("--min-delta", type=float, default=1e-5)
    p_train.add_argument("--cache-cases", action="store_true")
    p_train.add_argument("--run-test-after-training", action=argparse.BooleanOptionalAction, default=True,
                         help="After training, run infer-detect-sam2 on the test split with both best_detector.pt and last_detector.pt.")
    p_train.add_argument("--post-train-save-volumes", action=argparse.BooleanOptionalAction, default=True,
                         help="Save predicted NIfTI volumes during automatic post-training test prediction.")
    p_train.add_argument("--post-train-save-gt-volume", action=argparse.BooleanOptionalAction, default=False,
                         help="Also save GT masks during automatic post-training test prediction.")
    p_train.add_argument("--post-train-eval-batch-size", type=int, default=16)
    p_train.add_argument("--post-train-det-score-thresh", type=float, default=0.20)
    p_train.add_argument("--post-train-topk-per-slice", type=int, default=3)
    p_train.add_argument("--post-train-max-candidates-per-volume", type=int, default=20)
    p_train.add_argument("--post-train-nms-iou", type=float, default=0.30)
    p_train.add_argument("--post-train-eval-positive-slices-only", action="store_true")
    p_train.add_argument("--post-train-sam2-prompt-mode", choices=["point", "box", "point+box"], default="point+box")
    p_train.add_argument("--post-train-sam2-multimask", action="store_true")
    p_train.add_argument("--post-train-min-sam2-mask-area", type=int, default=3)
    p_train.add_argument("--post-train-fail-fast", action="store_true")

    p_det = sub.add_parser("infer-detector", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    add_common_args(p_det)
    p_det.add_argument("--detector-checkpoint", required=True)
    p_det.add_argument("--eval-batch-size", type=int, default=16)
    p_det.add_argument("--det-score-thresh", type=float, default=0.20)
    p_det.add_argument("--topk-per-slice", type=int, default=3)
    p_det.add_argument("--max-candidates-per-volume", type=int, default=20)
    p_det.add_argument("--nms-iou", type=float, default=0.30)
    p_det.add_argument("--eval-positive-slices-only", action="store_true")

    p_sam = sub.add_parser("infer-detect-sam2", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    add_common_args(p_sam)
    p_sam.add_argument("--detector-checkpoint", required=True)
    p_sam.add_argument("--eval-batch-size", type=int, default=16)
    p_sam.add_argument("--det-score-thresh", type=float, default=0.20)
    p_sam.add_argument("--topk-per-slice", type=int, default=3)
    p_sam.add_argument("--max-candidates-per-volume", type=int, default=20)
    p_sam.add_argument("--nms-iou", type=float, default=0.30)
    p_sam.add_argument("--eval-positive-slices-only", action="store_true")
    p_sam.add_argument("--sam2-prompt-mode", choices=["point", "box", "point+box"], default="point+box")
    p_sam.add_argument("--sam2-multimask", action="store_true")
    p_sam.add_argument("--min-sam2-mask-area", type=int, default=3)
    p_sam.add_argument("--save-volumes", action="store_true")
    p_sam.add_argument("--save-gt-volume", action="store_true")
    p_sam.add_argument("--fail-fast", action="store_true")

    return parser


def validate_args(args: argparse.Namespace) -> None:
    if not (0.0 <= args.val_ratio < 1.0):
        raise ValueError("--val-ratio must be in [0, 1)")
    if not (0.0 <= args.test_ratio < 1.0):
        raise ValueError("--test-ratio must be in [0, 1)")
    if args.val_ratio + args.test_ratio >= 1.0 and not (args.train_case_list or args.val_case_list or args.test_case_list):
        raise ValueError("Need --val-ratio + --test-ratio < 1 unless explicit split files are used")


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()
    validate_args(args)
    if args.command == "train-detector":
        train_detector(args)
    elif args.command == "infer-detector":
        infer_detector(args)
    elif args.command == "infer-detect-sam2":
        infer_detect_sam2(args)
    else:
        raise ValueError(args.command)


if __name__ == "__main__":
    main()
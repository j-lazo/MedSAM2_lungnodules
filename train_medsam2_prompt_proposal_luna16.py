#!/usr/bin/env python3
"""
Fully supervised prompt-proposal training for LUNA nodule segmentation with frozen MedSAM2/SAM2 guidance.

Compared with a promptless segmentation head, this model learns a small proposal module that outputs:
  1) proposed center point(s)          [B, N, 2] in xy pixel coordinates
  2) proposed bounding box(es)         [B, N, 4] in xyxy pixel coordinates
  3) coarse mask logits               [B, 1, H, W]
  4) frozen-MedSAM2 mask from point    [B, 1, H, W]
  5) frozen-MedSAM2 mask from box      [B, 1, H, W]

Training losses:
  - coarse mask Dice/BCE against GT mask
  - point SmoothL1 against GT interior center
  - box SmoothL1 + generalized IoU against GT mask-derived bbox
  - frozen MedSAM2 point-prompt Dice/BCE against GT mask
  - frozen MedSAM2 box-prompt Dice/BCE against GT mask

Two proposal-network options are provided:
  --proposal-backbone medsam2-fpn : frozen MedSAM2/SAM2 image encoder -> FPN decoder -> proposal heads
  --proposal-backbone unet        : trainable lightweight U-Net-like encoder/decoder -> proposal heads

Important implementation note:
  The SAM-guided loss uses frozen SAM2 modules but does NOT wrap the prompt encoder/mask decoder
  in torch.no_grad(), so gradients can flow from the SAM mask loss to the proposal heads. The SAM2
  weights remain frozen because requires_grad=False. Because SAM2/MedSAM2 forks differ internally,
  use --no-sam-guided-loss for debugging if your fork needs a small adapter change in the direct
  SAM forward path. Prompt coordinates are passed without any extra half-pixel offset.

Expected LUNA-style layout:
  DATASET_DIR/CT_volumes/*.mhd
  DATASET_DIR/masks_nodules/nifti_data/*mask*contour*nodule*.nii.gz
  DATASET_DIR/annotations.csv                         with column: seriesuid
  DATASET_DIR/LUNA16_metadata_split_offical.csv       with columns: SeriesID,CID
"""

from __future__ import annotations

import argparse
import contextlib
import gc
import importlib
import json
import math
import os
import random
import re
import socket
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np
import pandas as pd
import SimpleITK as sitk
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy import ndimage
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm


# =============================================================================
# General utilities
# =============================================================================


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def slugify(value: str, max_len: int = 180) -> str:
    value = str(value).replace("+", "plus")
    value = re.sub(r"[^A-Za-z0-9._-]+", "-", value)
    value = re.sub(r"-+", "-", value).strip("-_.")
    return value[:max_len]


def select_device(device_arg: str) -> torch.device:
    if device_arg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_arg)


def configure_torch(device: torch.device, allow_tf32: bool = True) -> None:
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = bool(allow_tf32)
        torch.backends.cudnn.allow_tf32 = bool(allow_tf32)
        if hasattr(torch, "set_float32_matmul_precision") and allow_tf32:
            torch.set_float32_matmul_precision("high")


def safe_autocast(device: torch.device, amp_dtype: str):
    if device.type != "cuda" or amp_dtype == "none":
        return contextlib.nullcontext()
    dtype = {"bf16": torch.bfloat16, "fp16": torch.float16}[amp_dtype]
    return torch.autocast("cuda", dtype=dtype)


def write_case_list_file(path: Path, ids: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(str(x) for x in ids) + ("\n" if ids else ""))


def read_case_list_file(path: Optional[str]) -> Optional[List[str]]:
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


def make_output_dir(args: argparse.Namespace) -> Path:
    out = Path(args.output_dir).expanduser().resolve()
    if args.create_experiment_dir:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        triplet = "2p5d" if args.use_triplet_channels else "2d_rgbcopy"
        size = f"sz{args.image_size}" if args.image_size else "native"
        aug = "aug" if getattr(args, "augment", False) else "noaug"
        name = args.experiment_name or "_".join(
            [
                stamp,
                "medsam2_prompt_proposal",
                args.proposal_backbone,
                args.decoder_type,
                getattr(args, "prompt_source", "heads"),
                "curr" if getattr(args, "curriculum_training", False) else "nocurr",
                aug,
                triplet,
                size,
                f"ep{args.epochs}",
                f"bs{args.batch_size}",
                f"lr{str(args.lr).replace('.', 'p')}",
            ]
        )
        out = out / slugify(name)
    out.mkdir(parents=True, exist_ok=args.overwrite_experiment)
    return out


# =============================================================================
# Metrics
# =============================================================================


def binary_confusion_counts(gt: np.ndarray, pred: np.ndarray) -> Tuple[int, int, int, int]:
    gt_b = gt.astype(bool)
    pred_b = pred.astype(bool)
    tp = int(np.logical_and(gt_b, pred_b).sum())
    fp = int(np.logical_and(~gt_b, pred_b).sum())
    fn = int(np.logical_and(gt_b, ~pred_b).sum())
    tn = int(np.logical_and(~gt_b, ~pred_b).sum())
    return tp, fp, fn, tn


def dice_score(gt: np.ndarray, pred: np.ndarray, smooth: float = 1e-6) -> float:
    gt_b = gt.astype(bool)
    pred_b = pred.astype(bool)
    inter = np.logical_and(gt_b, pred_b).sum(dtype=np.float64)
    total = gt_b.sum(dtype=np.float64) + pred_b.sum(dtype=np.float64)
    return float((2.0 * inter + smooth) / (total + smooth))


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


def surface_distances(mask_a: np.ndarray, mask_b: np.ndarray, spacing: Sequence[float]) -> np.ndarray:
    a = mask_a.astype(bool)
    b = mask_b.astype(bool)
    if not a.any() or not b.any():
        return np.array([], dtype=np.float32)
    surf_a = _surface(a)
    surf_b = _surface(b)
    dt_b = ndimage.distance_transform_edt(~surf_b, sampling=spacing)
    dt_a = ndimage.distance_transform_edt(~surf_a, sampling=spacing)
    return np.concatenate([dt_b[surf_a], dt_a[surf_b]]).astype(np.float32)


def hd95(gt: np.ndarray, pred: np.ndarray, spacing: Sequence[float]) -> float:
    if not gt.astype(bool).any() and not pred.astype(bool).any():
        return 0.0
    if not gt.astype(bool).any() or not pred.astype(bool).any():
        return float("nan")
    d = surface_distances(gt, pred, spacing)
    return float(np.percentile(d, 95)) if d.size else float("nan")


def assd(gt: np.ndarray, pred: np.ndarray, spacing: Sequence[float]) -> float:
    if not gt.astype(bool).any() and not pred.astype(bool).any():
        return 0.0
    if not gt.astype(bool).any() or not pred.astype(bool).any():
        return float("nan")
    d = surface_distances(gt, pred, spacing)
    return float(np.mean(d)) if d.size else float("nan")


def volume_similarity(gt: np.ndarray, pred: np.ndarray, smooth: float = 1e-6) -> float:
    g = float(gt.astype(bool).sum())
    p = float(pred.astype(bool).sum())
    return float(1.0 - abs(p - g) / (p + g + smooth))


def get_spacing_zyx(image_itk: sitk.Image) -> Tuple[float, float, float]:
    sx, sy, sz = image_itk.GetSpacing()
    return float(sz), float(sy), float(sx)


def get_spacing_yx(image_itk: sitk.Image) -> Tuple[float, float]:
    sx, sy, _ = image_itk.GetSpacing()
    return float(sy), float(sx)


def segmentation_metrics(gt: np.ndarray, pred: np.ndarray, spacing: Sequence[float]) -> Dict[str, float]:
    pr = precision_recall_f1(gt, pred)
    return {
        "DSC": dice_score(gt, pred),
        "IoU": iou_score(gt, pred),
        "precision": pr["precision"],
        "recall": pr["recall"],
        "F1": pr["f1"],
        "HD95_mm": hd95(gt, pred, spacing),
        "ASSD_mm": assd(gt, pred, spacing),
        "volume_similarity": volume_similarity(gt, pred),
        "pred_voxels": int(pred.astype(bool).sum()),
        "gt_voxels": int(gt.astype(bool).sum()),
        "TP": pr["tp"],
        "FP": pr["fp"],
        "FN": pr["fn"],
        "TN": pr["tn"],
    }


def aggregate_numeric(rows: List[Dict], group_key: str, metric_cols: List[str]) -> pd.DataFrame:
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    out_rows: List[Dict] = []
    for key, g in df.groupby(group_key):
        row = {group_key: key, "n_rows": int(len(g))}
        for c in metric_cols:
            vals = pd.to_numeric(g[c], errors="coerce") if c in g.columns else pd.Series(dtype=float)
            row[f"mean_{c}"] = float(vals.mean()) if vals.notna().any() else np.nan
            row[f"median_{c}"] = float(vals.median()) if vals.notna().any() else np.nan
            row[f"std_{c}"] = float(vals.std(ddof=0)) if vals.notna().any() else np.nan
        out_rows.append(row)
    return pd.DataFrame(out_rows)


# =============================================================================
# CT normalization and image construction
# =============================================================================


def normalize_ct_to_uint8(img_2d: np.ndarray, hu_min: float = -1000.0, hu_max: float = 400.0) -> np.ndarray:
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
    """Return H,W,3 uint8. For 2.5D, channels are z-1, z, z+1."""
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


def resize_rgb_and_mask(rgb: np.ndarray, mask: np.ndarray, image_size: Optional[int]) -> Tuple[np.ndarray, np.ndarray]:
    if image_size is None:
        return rgb, mask
    size = (int(image_size), int(image_size))
    rgb_r = cv2.resize(rgb, size, interpolation=cv2.INTER_LINEAR)
    mask_r = cv2.resize(mask.astype(np.uint8), size, interpolation=cv2.INTER_NEAREST)
    return rgb_r, mask_r


def build_augment_params(args: argparse.Namespace) -> Dict:
    return {
        "enabled": bool(args.augment),
        "hflip_p": float(args.aug_hflip_p),
        "vflip_p": float(args.aug_vflip_p),
        "rotation_deg": float(args.aug_rotation_deg),
        "shift_px": float(args.aug_shift_px),
        "scale_min": float(args.aug_scale_min),
        "scale_max": float(args.aug_scale_max),
        "intensity_p": float(args.aug_intensity_p),
        "brightness": float(args.aug_brightness),
        "contrast": float(args.aug_contrast),
        "noise_std": float(args.aug_noise_std),
        "blur_p": float(args.aug_blur_p),
    }


def apply_train_augmentations(rgb: np.ndarray, mask: np.ndarray, params: Dict) -> Tuple[np.ndarray, np.ndarray]:
    if not params or not params.get("enabled", False):
        return rgb, mask

    rgb = np.ascontiguousarray(rgb)
    mask = np.ascontiguousarray(mask.astype(np.uint8))

    if random.random() < params.get("hflip_p", 0.0):
        rgb = np.ascontiguousarray(rgb[:, ::-1])
        mask = np.ascontiguousarray(mask[:, ::-1])
    if random.random() < params.get("vflip_p", 0.0):
        rgb = np.ascontiguousarray(rgb[::-1, :])
        mask = np.ascontiguousarray(mask[::-1, :])

    H, W = mask.shape[:2]
    rot = float(params.get("rotation_deg", 0.0))
    shift = float(params.get("shift_px", 0.0))
    scale_min = float(params.get("scale_min", 1.0))
    scale_max = float(params.get("scale_max", 1.0))
    if rot > 0 or shift > 0 or abs(scale_min - 1.0) > 1e-6 or abs(scale_max - 1.0) > 1e-6:
        angle = random.uniform(-rot, rot) if rot > 0 else 0.0
        scale = random.uniform(scale_min, scale_max) if scale_max > 0 else 1.0
        tx = random.uniform(-shift, shift) if shift > 0 else 0.0
        ty = random.uniform(-shift, shift) if shift > 0 else 0.0
        M = cv2.getRotationMatrix2D((W / 2.0, H / 2.0), angle, scale)
        M[0, 2] += tx
        M[1, 2] += ty
        rgb = cv2.warpAffine(rgb, M, (W, H), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101)
        mask = cv2.warpAffine(mask, M, (W, H), flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT, borderValue=0)

    if random.random() < params.get("intensity_p", 0.0):
        x = rgb.astype(np.float32)
        contrast = float(params.get("contrast", 0.0))
        brightness = float(params.get("brightness", 0.0))
        if contrast > 0:
            x *= random.uniform(1.0 - contrast, 1.0 + contrast)
        if brightness > 0:
            x += random.uniform(-brightness, brightness) * 255.0
        noise_std = float(params.get("noise_std", 0.0))
        if noise_std > 0:
            x += np.random.normal(0.0, noise_std * 255.0, size=x.shape).astype(np.float32)
        rgb = np.clip(x, 0.0, 255.0).astype(np.uint8)

    if random.random() < params.get("blur_p", 0.0):
        rgb = cv2.GaussianBlur(rgb, ksize=(3, 3), sigmaX=0.0)

    return rgb.astype(np.uint8), (mask > 0).astype(np.uint8)


def numpy_rgb_to_tensor(rgb: np.ndarray) -> torch.Tensor:
    return torch.from_numpy(rgb).permute(2, 0, 1).float() / 255.0


def numpy_mask_to_tensor(mask: np.ndarray) -> torch.Tensor:
    return torch.from_numpy((mask > 0).astype(np.float32))[None]


def resize_pred_to_hw(mask: np.ndarray, hw: Tuple[int, int]) -> np.ndarray:
    H, W = hw
    if mask.shape == (H, W):
        return mask.astype(np.uint8)
    return cv2.resize(mask.astype(np.uint8), (W, H), interpolation=cv2.INTER_NEAREST).astype(np.uint8)


def resize_prob_to_hw(prob: np.ndarray, hw: Tuple[int, int]) -> np.ndarray:
    H, W = hw
    if prob.shape == (H, W):
        return prob.astype(np.float32)
    return cv2.resize(prob.astype(np.float32), (W, H), interpolation=cv2.INTER_LINEAR).astype(np.float32)


# =============================================================================
# Prompt target extraction from masks
# =============================================================================


def mask_to_box_and_point(mask_2d: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.float32]:
    """Return GT interior point xy, bbox xyxy, valid flag for a binary 2D mask."""
    mask = (mask_2d > 0).astype(np.uint8)
    H, W = mask.shape
    ys, xs = np.where(mask > 0)
    if len(xs) == 0:
        point = np.array([W / 2.0, H / 2.0], dtype=np.float32)
        box = np.array([0.0, 0.0, max(0.0, W - 1.0), max(0.0, H - 1.0)], dtype=np.float32)
        return point, box, np.float32(0.0)

    dist = cv2.distanceTransform(mask, cv2.DIST_L2, 5)
    py, px = np.unravel_index(int(np.argmax(dist)), dist.shape)
    point = np.array([float(px), float(py)], dtype=np.float32)
    box = np.array([float(xs.min()), float(ys.min()), float(xs.max()), float(ys.max())], dtype=np.float32)
    return point, box, np.float32(1.0)


# =============================================================================
# Dataset indexing/loading
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


@dataclass(frozen=True)
class SliceSample:
    series_id: str
    z: int


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
        f.name
        for f in masks_dir.iterdir()
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
            continue

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
    return CaseData(series_id=series_id, image_itk=image_itk, image_array=image_array, gt_volume=gt_volume, mask_path=mask_path)


def positive_slices_for_case(case: CaseData, min_slice_mask_pixels: int = 1) -> List[int]:
    counts = case.gt_volume.reshape(case.gt_volume.shape[0], -1).sum(axis=1)
    return [int(z) for z in np.where(counts >= min_slice_mask_pixels)[0].tolist()]


def filter_positive_volume_ids(index: DatasetIndex, args: argparse.Namespace) -> List[str]:
    ids: List[str] = []
    missing = 0
    no_positive = 0
    print(f"Filtering {len(index.volume_ids)} candidate volumes to volumes with positive nodule slices...")
    for sid in tqdm(index.volume_ids, desc="filter-volumes"):
        case = load_case(index, sid)
        if case is None:
            missing += 1
            continue
        zs = positive_slices_for_case(case, args.min_slice_mask_pixels)
        if zs:
            ids.append(sid)
        else:
            no_positive += 1
    print(f"Usable positive volumes: {len(ids)} | missing: {missing} | no positive slices: {no_positive}")
    return ids


def split_volume_ids(
    volume_ids: Sequence[str],
    seed: int,
    val_ratio: float,
    test_ratio: float,
    train_case_list: Optional[str] = None,
    val_case_list: Optional[str] = None,
    test_case_list: Optional[str] = None,
    shuffle_splits: bool = True,
) -> Dict[str, List[str]]:
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
        return {"train": train_ids, "val": val_ids, "test": test_ids, "all": available}

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
        excess = n_val + n_test - (n_total - 1)
        reduce_test = min(excess, n_test)
        n_test -= reduce_test
        excess -= reduce_test
        n_val = max(0, n_val - excess)

    test_ids = ids[:n_test]
    val_ids = ids[n_test : n_test + n_val]
    train_ids = ids[n_test + n_val :]
    return {"train": train_ids, "val": val_ids, "test": test_ids, "all": ids}


def write_split_files(splits: Dict[str, List[str]], out_dir: Path) -> None:
    split_dir = out_dir / "splits"
    for name in ["train", "val", "test", "all"]:
        write_case_list_file(split_dir / f"{name}.txt", splits.get(name, []))


class PositiveSlicePromptDataset(Dataset):
    def __init__(
        self,
        index: DatasetIndex,
        volume_ids: Sequence[str],
        use_triplet_channels: bool,
        hu_min: float,
        hu_max: float,
        min_slice_mask_pixels: int,
        image_size: Optional[int] = None,
        cache_cases: bool = False,
        augment_params: Optional[Dict] = None,
        desc: str = "dataset",
    ):
        self.index = index
        self.volume_ids = list(volume_ids)
        self.use_triplet_channels = use_triplet_channels
        self.hu_min = hu_min
        self.hu_max = hu_max
        self.min_slice_mask_pixels = min_slice_mask_pixels
        self.image_size = image_size
        self.cache_cases = cache_cases
        self.augment_params = augment_params or {"enabled": False}
        self._case_cache: Dict[str, CaseData] = {}
        self.samples = self._build_samples(desc=desc)
        if len(self.samples) == 0:
            raise RuntimeError(f"No positive nodule-slice samples built for {desc}")

    def _load_case(self, series_id: str) -> CaseData:
        if self.cache_cases and series_id in self._case_cache:
            return self._case_cache[series_id]
        case = load_case(self.index, series_id)
        if case is None:
            raise FileNotFoundError(f"Could not load case {series_id}")
        if self.cache_cases:
            self._case_cache[series_id] = case
        return case

    def _build_samples(self, desc: str) -> List[SliceSample]:
        samples: List[SliceSample] = []
        print(f"Building {desc} positive-slice index from {len(self.volume_ids)} volumes...")
        for sid in tqdm(self.volume_ids, desc=f"index-{desc}"):
            case = load_case(self.index, sid)
            if case is None:
                continue
            for z in positive_slices_for_case(case, self.min_slice_mask_pixels):
                samples.append(SliceSample(series_id=sid, z=z))
        print(f"{desc}: {len(samples)} positive slices from {len(set(s.series_id for s in samples))} patients/volumes")
        return samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict:
        s = self.samples[idx]
        case = self._load_case(s.series_id)
        rgb = build_sam2_input_slice(
            case.image_array,
            s.z,
            use_triplet_channels=self.use_triplet_channels,
            hu_min=self.hu_min,
            hu_max=self.hu_max,
        )
        mask = case.gt_volume[s.z].astype(np.uint8)
        orig_hw = mask.shape
        rgb, mask = resize_rgb_and_mask(rgb, mask, self.image_size)
        rgb, mask = apply_train_augmentations(rgb, mask, self.augment_params)
        point_xy, box_xyxy, valid = mask_to_box_and_point(mask)
        return {
            "image": numpy_rgb_to_tensor(rgb),
            "mask": numpy_mask_to_tensor(mask),
            "gt_point_xy": torch.from_numpy(point_xy)[None],  # [1, 2]
            "gt_box_xyxy": torch.from_numpy(box_xyxy)[None],  # [1, 4]
            "valid_prompt": torch.tensor([float(valid)], dtype=torch.float32),
            "series_id": s.series_id,
            "z": int(s.z),
            "orig_hw": torch.tensor(orig_hw, dtype=torch.long),
        }


def collate_prompt(batch: List[Dict]) -> Dict:
    return {
        "image": torch.stack([b["image"] for b in batch], dim=0),
        "mask": torch.stack([b["mask"] for b in batch], dim=0),
        "gt_point_xy": torch.stack([b["gt_point_xy"] for b in batch], dim=0),
        "gt_box_xyxy": torch.stack([b["gt_box_xyxy"] for b in batch], dim=0),
        "valid_prompt": torch.stack([b["valid_prompt"] for b in batch], dim=0),
        "series_id": [b["series_id"] for b in batch],
        "z": torch.tensor([b["z"] for b in batch], dtype=torch.long),
        "orig_hw": torch.stack([b["orig_hw"] for b in batch], dim=0),
    }


# =============================================================================
# Model blocks
# =============================================================================


class ConvGNAct(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, groups: int = 8, dropout: float = 0.0):
        super().__init__()
        g = min(groups, out_ch)
        while out_ch % g != 0 and g > 1:
            g -= 1
        layers: List[nn.Module] = [
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
            nn.GroupNorm(g, out_ch),
            nn.SiLU(inplace=True),
        ]
        if dropout > 0:
            layers.append(nn.Dropout2d(p=float(dropout)))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class LazyConvGNAct(nn.Module):
    def __init__(self, out_ch: int, groups: int = 8, dropout: float = 0.0):
        super().__init__()
        g = min(groups, out_ch)
        while out_ch % g != 0 and g > 1:
            g -= 1
        layers: List[nn.Module] = [
            nn.LazyConv2d(out_ch, kernel_size=1),
            nn.GroupNorm(g, out_ch),
            nn.SiLU(inplace=True),
        ]
        if dropout > 0:
            layers.append(nn.Dropout2d(p=float(dropout)))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class FPNFeatureDecoder(nn.Module):
    """Top-down FPN-like feature decoder over multiple MedSAM2/SAM2 backbone features."""

    def __init__(self, decoder_dim: int = 256, num_levels: int = 3, smooth_blocks: int = 1, dropout: float = 0.0):
        super().__init__()
        self.num_levels = max(1, int(num_levels))
        self.lateral_convs = nn.ModuleList([nn.LazyConv2d(decoder_dim, kernel_size=1) for _ in range(self.num_levels)])
        self.smooth_convs = nn.ModuleList()
        smooth_blocks = max(1, int(smooth_blocks))
        for _ in range(self.num_levels):
            blocks: List[nn.Module] = []
            for _ in range(smooth_blocks):
                blocks.append(ConvGNAct(decoder_dim, decoder_dim, dropout=dropout))
            self.smooth_convs.append(nn.Sequential(*blocks))

    def forward(self, features: Sequence[torch.Tensor]) -> torch.Tensor:
        if not features:
            raise ValueError("FPNFeatureDecoder received an empty feature list")
        feats = sorted(list(features), key=lambda t: int(t.shape[-2]) * int(t.shape[-1]))
        if len(feats) < self.num_levels:
            raise ValueError(f"FPN requested {self.num_levels} levels, but SAM2 returned only {len(feats)}")
        if len(feats) > self.num_levels:
            feats = feats[-self.num_levels:]
        y = self.lateral_convs[0](feats[0])
        y = self.smooth_convs[0](y)
        for i in range(1, len(feats)):
            lateral = self.lateral_convs[i](feats[i])
            y = F.interpolate(y, size=lateral.shape[-2:], mode="bilinear", align_corners=False)
            y = y + lateral
            y = self.smooth_convs[i](y)
        return y


class SimpleFeatureDecoder(nn.Module):
    def __init__(self, decoder_dim: int = 256, depth: int = 4, dropout: float = 0.0):
        super().__init__()
        depth = max(1, int(depth))
        layers: List[nn.Module] = [LazyConvGNAct(decoder_dim, dropout=dropout)]
        for _ in range(depth):
            layers.append(ConvGNAct(decoder_dim, decoder_dim, dropout=dropout))
        self.net = nn.Sequential(*layers)

    def forward(self, feat: torch.Tensor) -> torch.Tensor:
        return self.net(feat)


class UNetBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, dropout: float = 0.0):
        super().__init__()
        self.net = nn.Sequential(
            ConvGNAct(in_ch, out_ch, dropout=dropout),
            ConvGNAct(out_ch, out_ch, dropout=dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class SmallUNetFeatureExtractor(nn.Module):
    """Lightweight trainable U-Net-style feature extractor for proposal heads."""

    def __init__(self, in_ch: int = 3, base_ch: int = 32, out_ch: int = 128, dropout: float = 0.0):
        super().__init__()
        self.enc1 = UNetBlock(in_ch, base_ch, dropout=dropout)
        self.enc2 = UNetBlock(base_ch, base_ch * 2, dropout=dropout)
        self.enc3 = UNetBlock(base_ch * 2, base_ch * 4, dropout=dropout)
        self.bottleneck = UNetBlock(base_ch * 4, base_ch * 8, dropout=dropout)
        self.up3 = nn.ConvTranspose2d(base_ch * 8, base_ch * 4, kernel_size=2, stride=2)
        self.dec3 = UNetBlock(base_ch * 8, base_ch * 4, dropout=dropout)
        self.up2 = nn.ConvTranspose2d(base_ch * 4, base_ch * 2, kernel_size=2, stride=2)
        self.dec2 = UNetBlock(base_ch * 4, base_ch * 2, dropout=dropout)
        self.up1 = nn.ConvTranspose2d(base_ch * 2, base_ch, kernel_size=2, stride=2)
        self.dec1 = UNetBlock(base_ch * 2, base_ch, dropout=dropout)
        self.out = nn.Sequential(
            ConvGNAct(base_ch, out_ch, dropout=dropout),
            ConvGNAct(out_ch, out_ch, dropout=dropout),
        )

    @staticmethod
    def _crop_or_pad_to(x: torch.Tensor, target_hw: Tuple[int, int]) -> torch.Tensor:
        if x.shape[-2:] == target_hw:
            return x
        return F.interpolate(x, size=target_hw, mode="bilinear", align_corners=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        e1 = self.enc1(x)
        e2 = self.enc2(F.max_pool2d(e1, 2))
        e3 = self.enc3(F.max_pool2d(e2, 2))
        b = self.bottleneck(F.max_pool2d(e3, 2))
        d3 = self.up3(b)
        d3 = self._crop_or_pad_to(d3, e3.shape[-2:])
        d3 = self.dec3(torch.cat([d3, e3], dim=1))
        d2 = self.up2(d3)
        d2 = self._crop_or_pad_to(d2, e2.shape[-2:])
        d2 = self.dec2(torch.cat([d2, e2], dim=1))
        d1 = self.up1(d2)
        d1 = self._crop_or_pad_to(d1, e1.shape[-2:])
        d1 = self.dec1(torch.cat([d1, e1], dim=1))
        return self.out(d1)


class ProposalHeads(nn.Module):
    def __init__(self, in_ch: int, hidden_ch: int = 256, num_prompts: int = 1, dropout: float = 0.0):
        super().__init__()
        self.num_prompts = int(num_prompts)
        if self.num_prompts != 1:
            raise NotImplementedError(
                "This first implementation supports one proposal per positive slice. "
                "The tensors keep an [N] prompt dimension so multi-prompt matching can be added later."
            )
        self.refine = nn.Sequential(
            ConvGNAct(in_ch, hidden_ch, dropout=dropout),
            ConvGNAct(hidden_ch, hidden_ch, dropout=dropout),
        )
        self.coarse_head = nn.Conv2d(hidden_ch, 1, kernel_size=1)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.point_mlp = nn.Sequential(
            nn.Flatten(),
            nn.Linear(hidden_ch, hidden_ch),
            nn.SiLU(inplace=True),
            nn.Linear(hidden_ch, self.num_prompts * 2),
        )
        self.box_mlp = nn.Sequential(
            nn.Flatten(),
            nn.Linear(hidden_ch, hidden_ch),
            nn.SiLU(inplace=True),
            nn.Linear(hidden_ch, self.num_prompts * 4),
        )

    def forward(self, feat: torch.Tensor, input_hw: Tuple[int, int]) -> Dict[str, torch.Tensor]:
        H, W = int(input_hw[0]), int(input_hw[1])
        feat_full = F.interpolate(feat, size=input_hw, mode="bilinear", align_corners=False) if feat.shape[-2:] != input_hw else feat
        feat_full = self.refine(feat_full)
        coarse_logits = self.coarse_head(feat_full)

        pooled = self.pool(feat_full)
        point_raw = self.point_mlp(pooled).view(-1, self.num_prompts, 2)
        point01 = torch.sigmoid(point_raw)
        scale_xy = point01.new_tensor([max(W - 1, 1), max(H - 1, 1)]).view(1, 1, 2)
        point_xy = point01 * scale_xy

        # Predict center + width/height in normalized coordinates, then convert to xyxy pixels.
        box_raw = self.box_mlp(pooled).view(-1, self.num_prompts, 4)
        center01 = torch.sigmoid(box_raw[..., 0:2])
        wh01 = 0.02 + 0.98 * torch.sigmoid(box_raw[..., 2:4])
        center_xy = center01 * scale_xy
        wh_xy = wh01 * scale_xy
        x1y1 = center_xy - 0.5 * wh_xy
        x2y2 = center_xy + 0.5 * wh_xy
        box_xyxy = torch.cat([x1y1, x2y2], dim=-1)
        box_xyxy = clip_boxes_torch(box_xyxy, H=H, W=W)

        return {
            "proposal_features": feat_full,
            "coarse_logits": coarse_logits,
            "point_xy": point_xy,
            "box_xyxy": box_xyxy,
        }


# -----------------------------------------------------------------------------
# Prompt extraction from predicted coarse masks
# -----------------------------------------------------------------------------


def soft_prompts_from_coarse_logits(
    coarse_logits: torch.Tensor,
    box_std_scale: float = 2.0,
    min_box_size_px: float = 3.0,
    box_pad_px: float = 0.0,
    eps: float = 1e-6,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Differentiably derive one point and one xyxy box from coarse mask logits.

    The point is the probability-weighted center of mass. The box is a moment-based
    soft box around that center: center +/- box_std_scale * weighted std, with an
    optional pad. This keeps gradients from SAM losses connected to the coarse
    mask logits, unlike threshold + connected-components extraction.
    """
    if coarse_logits.ndim != 4 or coarse_logits.shape[1] != 1:
        raise ValueError(f"Expected coarse_logits [B,1,H,W], got {tuple(coarse_logits.shape)}")
    # Work in float32 even under fp16 autocast; coordinate variances can exceed fp16 range for 512x512 images.
    work_logits = coarse_logits.float()
    B, _, H, W = work_logits.shape
    dtype = work_logits.dtype
    device = work_logits.device
    probs = torch.sigmoid(work_logits).clamp_min(eps)

    ys = torch.arange(H, dtype=dtype, device=device).view(1, 1, H, 1)
    xs = torch.arange(W, dtype=dtype, device=device).view(1, 1, 1, W)
    mass = probs.sum(dim=(2, 3), keepdim=True).clamp_min(eps)

    cx = (probs * xs).sum(dim=(2, 3), keepdim=True) / mass
    cy = (probs * ys).sum(dim=(2, 3), keepdim=True) / mass

    var_x = (probs * (xs - cx).pow(2)).sum(dim=(2, 3), keepdim=True) / mass
    var_y = (probs * (ys - cy).pow(2)).sum(dim=(2, 3), keepdim=True) / mass
    std_x = torch.sqrt(var_x + eps)
    std_y = torch.sqrt(var_y + eps)

    min_half = 0.5 * float(max(min_box_size_px, 1.0))
    half_w = torch.clamp(float(box_std_scale) * std_x + float(box_pad_px), min=min_half)
    half_h = torch.clamp(float(box_std_scale) * std_y + float(box_pad_px), min=min_half)

    point_xy = torch.cat([cx, cy], dim=-1).view(B, 1, 2)
    x1 = (cx - half_w).view(B, 1)
    y1 = (cy - half_h).view(B, 1)
    x2 = (cx + half_w).view(B, 1)
    y2 = (cy + half_h).view(B, 1)
    box_xyxy = torch.stack([x1, y1, x2, y2], dim=-1)
    box_xyxy = clip_boxes_torch(box_xyxy, H=H, W=W)
    return point_xy, box_xyxy


def _largest_component_prompt_from_prob_np(
    prob_2d: np.ndarray,
    threshold: float,
    min_box_size_px: int,
    box_pad_px: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """Hard, non-differentiable prompt extraction from one probability map."""
    prob = np.asarray(prob_2d, dtype=np.float32)
    H, W = prob.shape
    mask = (prob >= float(threshold)).astype(np.uint8)
    num, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    if num > 1:
        # Ignore background row 0 and keep the largest predicted component.
        largest = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
        mask = (labels == largest).astype(np.uint8)
    if mask.sum() == 0:
        y, x = np.unravel_index(int(np.argmax(prob)), prob.shape)
        half = max(1, int(min_box_size_px)) / 2.0
        point = np.array([float(x), float(y)], dtype=np.float32)
        box = np.array([x - half, y - half, x + half, y + half], dtype=np.float32)
    else:
        point, box, _ = mask_to_box_and_point(mask)

    if box_pad_px != 0:
        box = box.astype(np.float32).copy()
        box[0] -= float(box_pad_px)
        box[1] -= float(box_pad_px)
        box[2] += float(box_pad_px)
        box[3] += float(box_pad_px)

    # Enforce a minimum size while preserving center when possible.
    cx = 0.5 * (float(box[0]) + float(box[2]))
    cy = 0.5 * (float(box[1]) + float(box[3]))
    min_size = max(1.0, float(min_box_size_px))
    if float(box[2] - box[0] + 1.0) < min_size:
        half = 0.5 * (min_size - 1.0)
        box[0], box[2] = cx - half, cx + half
    if float(box[3] - box[1] + 1.0) < min_size:
        half = 0.5 * (min_size - 1.0)
        box[1], box[3] = cy - half, cy + half

    box[0] = np.clip(box[0], 0, W - 1)
    box[2] = np.clip(box[2], 0, W - 1)
    box[1] = np.clip(box[1], 0, H - 1)
    box[3] = np.clip(box[3], 0, H - 1)
    x1, x2 = min(box[0], box[2]), max(box[0], box[2])
    y1, y2 = min(box[1], box[3]), max(box[1], box[3])
    return point.astype(np.float32), np.array([x1, y1, x2, y2], dtype=np.float32)


def hard_prompts_from_coarse_logits(
    coarse_logits: torch.Tensor,
    threshold: float = 0.5,
    min_box_size_px: int = 3,
    box_pad_px: int = 0,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Hard prompt extraction from predicted coarse masks.

    This exactly follows the usual threshold/component/bbox logic but is not
    differentiable. It is useful as an ablation and for inference, but SAM-guided
    losses will not backpropagate through this extraction step.
    """
    probs = torch.sigmoid(coarse_logits).detach().float().cpu().numpy()[:, 0]
    points: List[np.ndarray] = []
    boxes: List[np.ndarray] = []
    for prob in probs:
        p, b = _largest_component_prompt_from_prob_np(
            prob,
            threshold=threshold,
            min_box_size_px=min_box_size_px,
            box_pad_px=box_pad_px,
        )
        points.append(p)
        boxes.append(b)
    point_xy = torch.as_tensor(np.stack(points, axis=0), dtype=coarse_logits.dtype, device=coarse_logits.device)[:, None, :]
    box_xyxy = torch.as_tensor(np.stack(boxes, axis=0), dtype=coarse_logits.dtype, device=coarse_logits.device)[:, None, :]
    return point_xy, box_xyxy


# =============================================================================
# SAM2 direct frozen guidance
# =============================================================================


def load_builder(builder: str):
    if ":" not in builder:
        raise ValueError("--builder must have format module.submodule:function, e.g. sam2.build_sam:build_sam2")
    module_name, func_name = builder.split(":", 1)
    module = importlib.import_module(module_name)
    try:
        return getattr(module, func_name)
    except AttributeError as exc:
        raise AttributeError(f"Builder function {func_name!r} was not found in module {module_name!r}") from exc


def freeze_module(module: nn.Module) -> None:
    module.eval()
    for p in module.parameters():
        p.requires_grad = False


def reshape_sam2_feature(feat: torch.Tensor, feat_size: Sequence[int], batch_size: int) -> torch.Tensor:
    h, w = int(feat_size[0]), int(feat_size[1])
    if feat.ndim == 3 and feat.shape[1] == batch_size:
        c = int(feat.shape[-1])
        return feat.permute(1, 2, 0).reshape(batch_size, c, h, w).contiguous()
    if feat.ndim == 4 and feat.shape[0] == batch_size:
        return feat.contiguous()
    raise RuntimeError(f"Unexpected SAM2 feature shape {tuple(feat.shape)} for batch_size={batch_size}, feat_size={feat_size}")


def extract_sam2_features(sam2: nn.Module, x: torch.Tensor, no_grad: bool = True) -> List[torch.Tensor]:
    ctx = torch.no_grad() if no_grad else contextlib.nullcontext()
    with ctx:
        backbone_out = sam2.forward_image(x)
        _, vision_feats, _, feat_sizes = sam2._prepare_backbone_features(backbone_out)
        if getattr(sam2, "directly_add_no_mem_embed", False):
            vision_feats[-1] = vision_feats[-1] + sam2.no_mem_embed
        feats = [reshape_sam2_feature(f, s, x.shape[0]) for f, s in zip(vision_feats, feat_sizes)]
    return sorted(feats, key=lambda t: int(t.shape[-2]) * int(t.shape[-1]))


def build_sam2_image_feature_dict(sam2: nn.Module, x: torch.Tensor) -> Dict[str, List[torch.Tensor] | torch.Tensor]:
    """Return image_embed and high_res_feats for the SAM2 mask decoder.

    Important detail: ``_prepare_backbone_features`` returns several feature maps.
    The SAM2 mask decoder expects ``image_embeddings`` to have the same spatial
    size as the dense prompt embeddings from ``sam_prompt_encoder``. For the
    common 512x512 setup this is the *lowest-resolution* feature, e.g. 32x32,
    while the high-resolution skip features are typically 64x64 and 128x128.

    The previous version accidentally used the highest-resolution feature as
    ``image_embed``. That caused errors like:
        src 128x128 + dense_prompt_embeddings 32x32
    inside ``sam_mask_decoder.predict_masks``.
    """
    with torch.no_grad():
        backbone_out = sam2.forward_image(x)
        _, vision_feats, _, feat_sizes = sam2._prepare_backbone_features(backbone_out)
        if getattr(sam2, "directly_add_no_mem_embed", False):
            vision_feats[-1] = vision_feats[-1] + sam2.no_mem_embed
        feats = [reshape_sam2_feature(f, s, x.shape[0]) for f, s in zip(vision_feats, feat_sizes)]
        feats = sorted(feats, key=lambda t: int(t.shape[-2]) * int(t.shape[-1]))  # low -> high resolution

        # Mask decoder input: lowest-resolution image embedding, usually 32x32.
        image_embed = feats[0]

        # SAM2 high-res skips are usually expected as [highest_res, mid_res],
        # e.g. [128x128, 64x64], because the decoder adds them after each
        # upsampling stage from the 32x32 image embedding.
        high_res_feats = list(reversed(feats[1:]))

    return {"image_embed": image_embed, "high_res_feats": high_res_feats}


def _call_prompt_encoder(prompt_encoder, points, boxes, masks):
    """Call SAM2 prompt encoder with several fork-compatible signatures."""
    try:
        return prompt_encoder(points=points, boxes=boxes, masks=masks)
    except TypeError:
        # Some forks use positional arguments.
        return prompt_encoder(points, boxes, masks)


def _call_mask_decoder(mask_decoder, *, image_embeddings, image_pe, sparse_prompt_embeddings, dense_prompt_embeddings,
                       multimask_output, repeat_image, high_res_features):
    """Call SAM2 mask decoder with fallback signatures used by SAM/SAM2 forks."""
    kwargs = dict(
        image_embeddings=image_embeddings,
        image_pe=image_pe,
        sparse_prompt_embeddings=sparse_prompt_embeddings,
        dense_prompt_embeddings=dense_prompt_embeddings,
        multimask_output=multimask_output,
    )
    # SAM2 usually supports repeat_image and high_res_features.
    try:
        return mask_decoder(**kwargs, repeat_image=repeat_image, high_res_features=high_res_features)
    except TypeError:
        try:
            return mask_decoder(**kwargs, high_res_features=high_res_features)
        except TypeError:
            return mask_decoder(**kwargs)


class FrozenSAM2Guidance(nn.Module):
    """Frozen SAM2 forward path used only for SAM-guided proposal losses.

    Gradients are allowed through prompt coordinates into the proposal network. SAM2 weights remain frozen.

    Prompt handling intentionally mirrors the SAM2 image-predictor API:
      - point prompts are passed through ``points=(coords, labels)``
      - box prompts are passed through ``boxes=box_xyxy``
      - point+box prompts pass both arguments at the same time

    No half-pixel offset is added here. Coordinates are expected to be in the
    same pixel-coordinate convention as the predictor input prompts.
    """

    def __init__(self, sam2_model: nn.Module):
        super().__init__()
        self.sam2 = sam2_model
        freeze_module(self.sam2)

    def _make_points(self, point_xy: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # point_xy: [B, 1, 2] or [B, 2], xy pixel coordinates.
        if point_xy.ndim == 2:
            point_xy = point_xy[:, None, :]
        if point_xy.ndim != 3 or point_xy.shape[-1] != 2:
            raise ValueError(f"Expected point_xy with shape [B, N, 2] or [B, 2], got {tuple(point_xy.shape)}")
        coords = point_xy
        labels = torch.ones(coords.shape[:2], dtype=torch.int64, device=coords.device)
        return coords, labels

    def _make_boxes(self, box_xyxy: torch.Tensor) -> torch.Tensor:
        # box_xyxy: [B, 4] or [B, 1, 4], xyxy pixel coordinates.
        if box_xyxy.ndim == 3:
            if box_xyxy.shape[1] != 1:
                raise NotImplementedError("FrozenSAM2Guidance currently supports one box per image.")
            box_xyxy = box_xyxy[:, 0, :]
        if box_xyxy.ndim != 2 or box_xyxy.shape[-1] != 4:
            raise ValueError(f"Expected box_xyxy with shape [B, 4] or [B, 1, 4], got {tuple(box_xyxy.shape)}")
        return box_xyxy

    def decode(self, image: torch.Tensor, *, point_xy: Optional[torch.Tensor] = None, box_xyxy: Optional[torch.Tensor] = None) -> torch.Tensor:
        if point_xy is None and box_xyxy is None:
            raise ValueError("Need point_xy or box_xyxy")
        input_hw = image.shape[-2:]
        feats = build_sam2_image_feature_dict(self.sam2, image)

        points = self._make_points(point_xy) if point_xy is not None else None
        boxes = self._make_boxes(box_xyxy) if box_xyxy is not None else None

        sparse_embeddings, dense_embeddings = _call_prompt_encoder(
            self.sam2.sam_prompt_encoder,
            points=points,
            boxes=boxes,
            masks=None,
        )
        image_pe = self.sam2.sam_prompt_encoder.get_dense_pe()
        decoder_out = _call_mask_decoder(
            self.sam2.sam_mask_decoder,
            image_embeddings=feats["image_embed"],
            image_pe=image_pe,
            sparse_prompt_embeddings=sparse_embeddings,
            dense_prompt_embeddings=dense_embeddings,
            multimask_output=False,
            repeat_image=False,
            high_res_features=feats["high_res_feats"],
        )
        low_res_masks = decoder_out[0]
        if low_res_masks.ndim == 3:
            low_res_masks = low_res_masks[:, None]
        # Keep the first mask channel when a fork still returns multiple masks.
        logits = low_res_masks[:, :1]
        logits = F.interpolate(logits, size=input_hw, mode="bilinear", align_corners=False)
        return logits

    def forward(self, image: torch.Tensor, point_xy: torch.Tensor, box_xyxy: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.decode(image, point_xy=point_xy), self.decode(image, box_xyxy=box_xyxy)


class PromptProposalModel(nn.Module):
    def __init__(
        self,
        sam2_model: nn.Module,
        proposal_backbone: str = "medsam2-fpn",
        decoder_type: str = "fpn",
        decoder_dim: int = 256,
        decoder_depth: int = 2,
        fpn_levels: int = 3,
        decoder_dropout: float = 0.0,
        unet_base_ch: int = 32,
        unet_out_ch: int = 128,
        num_prompts: int = 1,
        prompt_source: str = "heads",
        coarse_prompt_threshold: float = 0.5,
        coarse_prompt_box_std_scale: float = 2.0,
        coarse_prompt_min_box_size_px: int = 3,
        coarse_prompt_box_pad_px: int = 0,
    ):
        super().__init__()
        self.sam2 = sam2_model
        self.proposal_backbone = str(proposal_backbone)
        self.decoder_type = str(decoder_type)
        self.prompt_source = str(prompt_source)
        self.coarse_prompt_threshold = float(coarse_prompt_threshold)
        self.coarse_prompt_box_std_scale = float(coarse_prompt_box_std_scale)
        self.coarse_prompt_min_box_size_px = int(coarse_prompt_min_box_size_px)
        self.coarse_prompt_box_pad_px = int(coarse_prompt_box_pad_px)
        freeze_module(self.sam2)
        self.sam_guidance = FrozenSAM2Guidance(self.sam2)

        if self.proposal_backbone == "medsam2-fpn":
            if self.decoder_type == "fpn":
                self.proposal_decoder = FPNFeatureDecoder(
                    decoder_dim=decoder_dim,
                    num_levels=fpn_levels,
                    smooth_blocks=decoder_depth,
                    dropout=decoder_dropout,
                )
                heads_in_ch = decoder_dim
            elif self.decoder_type in {"simple", "deep"}:
                self.proposal_decoder = SimpleFeatureDecoder(
                    decoder_dim=decoder_dim,
                    depth=decoder_depth,
                    dropout=decoder_dropout,
                )
                heads_in_ch = decoder_dim
            else:
                raise ValueError(f"Unknown decoder_type={decoder_type!r}")
        elif self.proposal_backbone == "unet":
            self.proposal_decoder = SmallUNetFeatureExtractor(
                in_ch=3,
                base_ch=unet_base_ch,
                out_ch=unet_out_ch,
                dropout=decoder_dropout,
            )
            heads_in_ch = unet_out_ch
        else:
            raise ValueError("--proposal-backbone must be medsam2-fpn or unet")

        self.heads = ProposalHeads(
            in_ch=heads_in_ch,
            hidden_ch=decoder_dim,
            num_prompts=num_prompts,
            dropout=decoder_dropout,
        )

    def extract_proposal_features(self, x: torch.Tensor) -> torch.Tensor:
        if self.proposal_backbone == "unet":
            return self.proposal_decoder(x)
        feats = extract_sam2_features(self.sam2, x, no_grad=True)
        if self.decoder_type == "fpn":
            return self.proposal_decoder(feats)
        return self.proposal_decoder(feats[-1])

    def forward(self, x: torch.Tensor, run_sam: bool = True) -> Dict[str, torch.Tensor]:
        feat = self.extract_proposal_features(x)
        out = self.heads(feat, input_hw=x.shape[-2:])

        # Keep raw head outputs for diagnostics even when the prompts actually sent
        # to MedSAM2 are derived from the predicted coarse mask.
        out["point_xy_head"] = out["point_xy"]
        out["box_xyxy_head"] = out["box_xyxy"]
        if self.prompt_source == "coarse-soft":
            point_xy, box_xyxy = soft_prompts_from_coarse_logits(
                out["coarse_logits"],
                box_std_scale=self.coarse_prompt_box_std_scale,
                min_box_size_px=self.coarse_prompt_min_box_size_px,
                box_pad_px=self.coarse_prompt_box_pad_px,
            )
            out["point_xy"] = point_xy
            out["box_xyxy"] = box_xyxy
        elif self.prompt_source == "coarse-hard":
            point_xy, box_xyxy = hard_prompts_from_coarse_logits(
                out["coarse_logits"],
                threshold=self.coarse_prompt_threshold,
                min_box_size_px=self.coarse_prompt_min_box_size_px,
                box_pad_px=self.coarse_prompt_box_pad_px,
            )
            out["point_xy"] = point_xy
            out["box_xyxy"] = box_xyxy
        elif self.prompt_source != "heads":
            raise ValueError(f"Unknown prompt_source={self.prompt_source!r}")

        if run_sam:
            sam_point_logits, sam_box_logits = self.sam_guidance(
                x,
                point_xy=out["point_xy"],
                box_xyxy=out["box_xyxy"],
            )
            out["sam_point_logits"] = sam_point_logits
            out["sam_box_logits"] = sam_box_logits
        return out


# =============================================================================
# Losses
# =============================================================================


def clip_boxes_torch(boxes: torch.Tensor, H: int, W: int) -> torch.Tensor:
    x1 = boxes[..., 0].clamp(0.0, max(float(W - 1), 0.0))
    y1 = boxes[..., 1].clamp(0.0, max(float(H - 1), 0.0))
    x2 = boxes[..., 2].clamp(0.0, max(float(W - 1), 0.0))
    y2 = boxes[..., 3].clamp(0.0, max(float(H - 1), 0.0))
    # Ensure non-degenerate ordering after clipping.
    xx1 = torch.minimum(x1, x2)
    yy1 = torch.minimum(y1, y2)
    xx2 = torch.maximum(x1, x2)
    yy2 = torch.maximum(y1, y2)
    return torch.stack([xx1, yy1, xx2, yy2], dim=-1)


def box_area(boxes: torch.Tensor) -> torch.Tensor:
    return (boxes[..., 2] - boxes[..., 0]).clamp(min=0) * (boxes[..., 3] - boxes[..., 1]).clamp(min=0)


def generalized_iou_loss(pred_boxes: torch.Tensor, target_boxes: torch.Tensor, valid: Optional[torch.Tensor] = None, eps: float = 1e-7) -> torch.Tensor:
    # pred/target: [B, N, 4]
    x1 = torch.maximum(pred_boxes[..., 0], target_boxes[..., 0])
    y1 = torch.maximum(pred_boxes[..., 1], target_boxes[..., 1])
    x2 = torch.minimum(pred_boxes[..., 2], target_boxes[..., 2])
    y2 = torch.minimum(pred_boxes[..., 3], target_boxes[..., 3])
    inter = (x2 - x1).clamp(min=0) * (y2 - y1).clamp(min=0)
    area_p = box_area(pred_boxes)
    area_t = box_area(target_boxes)
    union = area_p + area_t - inter + eps
    iou = inter / union

    cx1 = torch.minimum(pred_boxes[..., 0], target_boxes[..., 0])
    cy1 = torch.minimum(pred_boxes[..., 1], target_boxes[..., 1])
    cx2 = torch.maximum(pred_boxes[..., 2], target_boxes[..., 2])
    cy2 = torch.maximum(pred_boxes[..., 3], target_boxes[..., 3])
    c_area = (cx2 - cx1).clamp(min=0) * (cy2 - cy1).clamp(min=0) + eps
    giou = iou - (c_area - union) / c_area
    loss = 1.0 - giou
    if valid is not None:
        loss = loss * valid
        return loss.sum() / valid.sum().clamp(min=1.0)
    return loss.mean()


def soft_dice_loss_from_logits(logits: torch.Tensor, targets: torch.Tensor, smooth: float = 1.0) -> torch.Tensor:
    probs = torch.sigmoid(logits)
    dims = tuple(range(1, probs.ndim))
    inter = (probs * targets).sum(dim=dims)
    denom = probs.sum(dim=dims) + targets.sum(dim=dims)
    dice = (2.0 * inter + smooth) / (denom + smooth)
    return 1.0 - dice.mean()


def mask_loss(logits: torch.Tensor, targets: torch.Tensor, bce_weight: float, dice_weight: float) -> Tuple[torch.Tensor, Dict[str, float]]:
    bce = F.binary_cross_entropy_with_logits(logits, targets)
    dice = soft_dice_loss_from_logits(logits, targets)
    total = bce_weight * bce + dice_weight * dice
    return total, {"bce": float(bce.detach()), "dice": float(dice.detach()), "total": float(total.detach())}


def normalized_smooth_l1(pred: torch.Tensor, target: torch.Tensor, valid: torch.Tensor, input_hw: Tuple[int, int]) -> torch.Tensor:
    H, W = int(input_hw[0]), int(input_hw[1])
    if pred.shape[-1] == 2:
        scale = pred.new_tensor([max(W - 1, 1), max(H - 1, 1)]).view(1, 1, 2)
    elif pred.shape[-1] == 4:
        scale = pred.new_tensor([max(W - 1, 1), max(H - 1, 1), max(W - 1, 1), max(H - 1, 1)]).view(1, 1, 4)
    else:
        raise ValueError("Expected last dimension 2 or 4")
    loss = F.smooth_l1_loss(pred / scale, target / scale, reduction="none").mean(dim=-1)
    loss = loss * valid
    return loss.sum() / valid.sum().clamp(min=1.0)


def compute_losses(outputs: Dict[str, torch.Tensor], batch: Dict, args: argparse.Namespace, run_sam: bool) -> Tuple[torch.Tensor, Dict[str, float]]:
    masks = batch["mask"]
    gt_point = batch["gt_point_xy"]
    gt_box = batch["gt_box_xyxy"]
    valid = batch["valid_prompt"]
    input_hw = masks.shape[-2:]

    coarse, coarse_logs = mask_loss(outputs["coarse_logits"], masks, args.coarse_bce_weight, args.coarse_dice_weight)
    point = normalized_smooth_l1(outputs["point_xy"], gt_point, valid, input_hw)
    box_l1 = normalized_smooth_l1(outputs["box_xyxy"], gt_box, valid, input_hw)
    box_giou = generalized_iou_loss(outputs["box_xyxy"], gt_box, valid=valid)

    supervise_prompts = (
        args.prompt_source == "heads"
        or (args.prompt_source == "coarse-soft" and args.supervise_derived_prompts)
    )
    total = args.lambda_coarse * coarse
    if supervise_prompts:
        total = total + args.lambda_point * point + args.lambda_box_l1 * box_l1 + args.lambda_box_giou * box_giou
    logs: Dict[str, float] = {
        "loss_coarse": float(coarse.detach()),
        "coarse_bce": coarse_logs["bce"],
        "coarse_dice_loss": coarse_logs["dice"],
        "loss_point": float(point.detach()),
        "loss_box_l1": float(box_l1.detach()),
        "loss_box_giou": float(box_giou.detach()),
        "loss_sam_point": 0.0,
        "loss_sam_box": 0.0,
    }

    if run_sam:
        sam_point, _ = mask_loss(outputs["sam_point_logits"], masks, args.sam_bce_weight, args.sam_dice_weight)
        sam_box, _ = mask_loss(outputs["sam_box_logits"], masks, args.sam_bce_weight, args.sam_dice_weight)
        # Hard threshold/component prompt extraction is non-differentiable, so SAM losses
        # from coarse-hard prompts are logged but not added to the optimization objective.
        if args.prompt_source != "coarse-hard":
            total = total + args.lambda_sam_point * sam_point + args.lambda_sam_box * sam_box
        logs["loss_sam_point"] = float(sam_point.detach())
        logs["loss_sam_box"] = float(sam_box.detach())

    logs["loss"] = float(total.detach())
    return total, logs


# =============================================================================
# Training / validation
# =============================================================================


def batch_dice_from_logits(logits: torch.Tensor, masks: torch.Tensor, threshold: float) -> float:
    probs = torch.sigmoid(logits).detach().float().cpu().numpy()
    gt = masks.detach().float().cpu().numpy()
    pred = probs >= threshold
    vals = [dice_score(gt[i, 0] > 0.5, pred[i, 0]) for i in range(pred.shape[0])]
    return float(np.mean(vals)) if vals else 0.0


def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: Optional[torch.optim.Optimizer],
    scaler: Optional[torch.cuda.amp.GradScaler],
    device: torch.device,
    amp_dtype: str,
    threshold: float,
    desc: str,
    args: argparse.Namespace,
) -> Dict[str, float]:
    is_train = optimizer is not None
    model.train(is_train)
    # Keep frozen SAM2 in eval even when proposal heads are training.
    if hasattr(model, "sam2"):
        model.sam2.eval()
    totals: Dict[str, float] = {
        "loss": 0.0,
        "loss_coarse": 0.0,
        "loss_point": 0.0,
        "loss_box_l1": 0.0,
        "loss_box_giou": 0.0,
        "loss_sam_point": 0.0,
        "loss_sam_box": 0.0,
        "DSC_coarse": 0.0,
        "DSC_sam_point": 0.0,
        "DSC_sam_box": 0.0,
    }
    n = 0
    run_sam = bool(args.sam_guided_loss)

    pbar = tqdm(loader, desc=desc, leave=False)
    for batch in pbar:
        images = batch["image"].to(device, non_blocking=True)
        batch = {
            **batch,
            "mask": batch["mask"].to(device, non_blocking=True),
            "gt_point_xy": batch["gt_point_xy"].to(device, non_blocking=True),
            "gt_box_xyxy": batch["gt_box_xyxy"].to(device, non_blocking=True),
            "valid_prompt": batch["valid_prompt"].to(device, non_blocking=True),
        }

        if is_train:
            optimizer.zero_grad(set_to_none=True)

        with torch.set_grad_enabled(is_train), safe_autocast(device, amp_dtype):
            outputs = model(images, run_sam=run_sam)
            loss, logs = compute_losses(outputs, batch, args, run_sam=run_sam)

        if is_train:
            if scaler is not None and amp_dtype == "fp16":
                scaler.scale(loss).backward()
                if args.grad_clip_norm > 0:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip_norm)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                if args.grad_clip_norm > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip_norm)
                optimizer.step()

        with torch.no_grad():
            bs = images.shape[0]
            totals["loss"] += logs["loss"] * bs
            for k in ["loss_coarse", "loss_point", "loss_box_l1", "loss_box_giou", "loss_sam_point", "loss_sam_box"]:
                totals[k] += logs.get(k, 0.0) * bs
            totals["DSC_coarse"] += batch_dice_from_logits(outputs["coarse_logits"], batch["mask"], threshold) * bs
            if run_sam:
                totals["DSC_sam_point"] += batch_dice_from_logits(outputs["sam_point_logits"], batch["mask"], threshold) * bs
                totals["DSC_sam_box"] += batch_dice_from_logits(outputs["sam_box_logits"], batch["mask"], threshold) * bs
            n += bs
        pbar.set_postfix({k: f"{totals[k] / max(n, 1):.4f}" for k in ["loss", "DSC_coarse", "DSC_sam_box"]})

    return {k: totals[k] / max(n, 1) for k in totals}


def save_checkpoint(
    path: Path,
    model: nn.Module,
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


def load_checkpoint(model: nn.Module, ckpt_path: Path, device: torch.device) -> Dict:
    ckpt = torch.load(str(ckpt_path), map_location=device)
    missing, unexpected = model.load_state_dict(ckpt["model_state"], strict=False)
    if missing:
        print(f"WARNING missing keys while loading checkpoint: {missing[:10]}{'...' if len(missing) > 10 else ''}")
    if unexpected:
        print(f"WARNING unexpected keys while loading checkpoint: {unexpected[:10]}{'...' if len(unexpected) > 10 else ''}")
    return ckpt


def plot_training_metrics(metrics_csv: Path, out_dir: Path) -> None:
    if not metrics_csv.exists():
        return
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:
        print(f"WARNING: matplotlib not available for plots: {exc}")
        return
    df = pd.read_csv(metrics_csv)
    if df.empty or "epoch" not in df.columns:
        return
    groups = [
        ([c for c in df.columns if "loss" in c.lower()], "losses.png", "Training/validation losses"),
        ([c for c in df.columns if "DSC" in c], "dice_metrics.png", "Training/validation Dice metrics"),
    ]
    for cols, filename, title in groups:
        cols = [c for c in cols if c in df.columns]
        if not cols:
            continue
        plt.figure(figsize=(11, 6))
        for c in cols:
            plt.plot(df["epoch"], df[c], marker="o", linewidth=1.5, label=c)
        plt.xlabel("Epoch")
        plt.ylabel("Value")
        plt.title(title)
        plt.grid(True, alpha=0.3)
        plt.legend(fontsize=8)
        plt.tight_layout()
        plt.savefig(out_dir / filename, dpi=200)
        plt.close()


# =============================================================================
# Evaluation and prediction saving
# =============================================================================


def write_pred_volume(pred: np.ndarray, reference_image: sitk.Image, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    img = sitk.GetImageFromArray(pred.astype(np.uint8))
    img.CopyInformation(reference_image)
    sitk.WriteImage(img, str(out_path))


def write_prob_volume(prob: np.ndarray, reference_image: sitk.Image, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    img = sitk.GetImageFromArray(prob.astype(np.float32))
    img.CopyInformation(reference_image)
    sitk.WriteImage(img, str(out_path))


def predict_case_positive_slices(
    model: nn.Module,
    case: CaseData,
    args: argparse.Namespace,
    device: torch.device,
) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray], List[Dict]]:
    Z, H, W = case.gt_volume.shape
    mask_keys = ["coarse", "sam_point", "sam_box"]
    pred_volumes = {k: np.zeros((Z, H, W), dtype=np.uint8) for k in mask_keys}
    prob_volumes = {k: np.zeros((Z, H, W), dtype=np.float32) for k in mask_keys}
    slice_rows: List[Dict] = []
    positive_zs = positive_slices_for_case(case, args.min_slice_mask_pixels)
    spacing_yx = get_spacing_yx(case.image_itk)

    model.eval()
    with torch.no_grad():
        for z0 in range(0, len(positive_zs), args.eval_batch_size):
            batch_zs = positive_zs[z0 : z0 + args.eval_batch_size]
            imgs = []
            for z in batch_zs:
                rgb = build_sam2_input_slice(
                    case.image_array,
                    z,
                    use_triplet_channels=args.use_triplet_channels,
                    hu_min=args.hu_min,
                    hu_max=args.hu_max,
                )
                mask_dummy = case.gt_volume[z].astype(np.uint8)
                rgb, _ = resize_rgb_and_mask(rgb, mask_dummy, args.image_size)
                imgs.append(numpy_rgb_to_tensor(rgb))
            x = torch.stack(imgs, dim=0).to(device)
            with safe_autocast(device, args.amp_dtype):
                outputs = model(x, run_sam=args.eval_sam_outputs)

            logits_map = {"coarse": outputs["coarse_logits"]}
            if args.eval_sam_outputs:
                logits_map["sam_point"] = outputs["sam_point_logits"]
                logits_map["sam_box"] = outputs["sam_box_logits"]

            prompt_points = outputs["point_xy"].detach().float().cpu().numpy()[:, 0]
            prompt_boxes = outputs["box_xyxy"].detach().float().cpu().numpy()[:, 0]

            for key, logits in logits_map.items():
                probs = torch.sigmoid(logits).detach().float().cpu().numpy()[:, 0]
                preds = (probs >= args.threshold).astype(np.uint8)
                for i, z in enumerate(batch_zs):
                    p = resize_pred_to_hw(preds[i], (H, W))
                    prob = resize_prob_to_hw(probs[i], (H, W))
                    pred_volumes[key][z] = p
                    prob_volumes[key][z] = prob

            for i, z in enumerate(batch_zs):
                gt_slice = case.gt_volume[z].astype(np.uint8)
                row = {
                    "SeriesUID": case.series_id,
                    "z": int(z),
                    "status": "ok",
                    "pred_point_x": float(prompt_points[i, 0]),
                    "pred_point_y": float(prompt_points[i, 1]),
                    "pred_box_x1": float(prompt_boxes[i, 0]),
                    "pred_box_y1": float(prompt_boxes[i, 1]),
                    "pred_box_x2": float(prompt_boxes[i, 2]),
                    "pred_box_y2": float(prompt_boxes[i, 3]),
                }
                for key in logits_map.keys():
                    m = segmentation_metrics(gt_slice, pred_volumes[key][z], spacing_yx)
                    row.update({f"{key}_{mk}": mv for mk, mv in m.items()})
                slice_rows.append(row)
    return pred_volumes, prob_volumes, slice_rows


def evaluate_checkpoint(
    args: argparse.Namespace,
    out_dir: Path,
    ckpt_path: Path,
    tag: str,
    index: DatasetIndex,
    test_ids: Sequence[str],
    device: torch.device,
    sample_image: torch.Tensor,
) -> None:
    pred_dir = out_dir / "test_predictions" / tag
    pred_vol_dir = pred_dir / "predicted_volumes"
    prob_vol_dir = pred_dir / "probability_volumes"
    gt_vol_dir = pred_dir / "gt_volumes"
    pred_dir.mkdir(parents=True, exist_ok=True)

    model = build_model(args, device)
    materialize_lazy_modules(model, sample_image, device, args.amp_dtype, run_sam=False)
    load_checkpoint(model, ckpt_path, device)
    model.eval()

    slice_rows: List[Dict] = []
    volume_rows: List[Dict] = []

    print(f"Evaluating {tag} checkpoint on {len(test_ids)} positive test volumes: {ckpt_path}")
    for sid in tqdm(test_ids, desc=f"test-{tag}"):
        try:
            case = load_case(index, sid)
            if case is None:
                volume_rows.append({"SeriesUID": sid, "status": "missing"})
                continue
            pred_volumes, prob_volumes, rows = predict_case_positive_slices(model, case, args, device)
            slice_rows.extend(rows)
            spacing_zyx = get_spacing_zyx(case.image_itk)
            vol_row = {
                "SeriesUID": sid,
                "status": "ok",
                "n_positive_slices": len(positive_slices_for_case(case, args.min_slice_mask_pixels)),
            }
            for key, pred_vol in pred_volumes.items():
                # Skip empty sam volumes when eval_sam_outputs is false.
                if key in {"sam_point", "sam_box"} and not args.eval_sam_outputs:
                    continue
                m = segmentation_metrics(case.gt_volume, pred_vol, spacing_zyx)
                vol_row.update({f"{key}_{mk}": mv for mk, mv in m.items()})
            volume_rows.append(vol_row)

            if args.save_volumes:
                for key, pred_vol in pred_volumes.items():
                    if key in {"sam_point", "sam_box"} and not args.eval_sam_outputs:
                        continue
                    write_pred_volume(pred_vol, case.image_itk, pred_vol_dir / key / f"{sid}_{key}_{tag}.nii.gz")
                    if args.save_prob_volumes:
                        write_prob_volume(prob_volumes[key], case.image_itk, prob_vol_dir / key / f"{sid}_{key}_{tag}.nii.gz")
                if args.save_gt_volume:
                    write_pred_volume(case.gt_volume, case.image_itk, gt_vol_dir / f"{sid}_gt.nii.gz")
        except Exception as exc:
            if args.fail_fast:
                raise
            volume_rows.append({"SeriesUID": sid, "status": "error", "error": repr(exc)})
            print(f"ERROR {sid}: {repr(exc)}")
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()

    pd.DataFrame(slice_rows).to_csv(pred_dir / "slice_metrics_and_prompts.csv", index=False)
    pd.DataFrame(volume_rows).to_csv(pred_dir / "patient_volume_metrics.csv", index=False)

    # Summary over patient-volume metrics.
    dfv = pd.DataFrame(volume_rows)
    if not dfv.empty and "status" in dfv.columns:
        dfok = dfv[dfv["status"] == "ok"].copy()
    else:
        dfok = dfv
    summary = {}
    for c in dfok.columns:
        if c in {"SeriesUID", "status", "error"}:
            continue
        vals = pd.to_numeric(dfok[c], errors="coerce")
        if vals.notna().any():
            summary[f"mean_{c}"] = float(vals.mean())
            summary[f"median_{c}"] = float(vals.median())
    with open(pred_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    pd.DataFrame([summary]).to_csv(pred_dir / "summary.csv", index=False)
    print(f"Saved {tag} test outputs to: {pred_dir}")


# =============================================================================
# Build model / checkpoint materialization
# =============================================================================


def build_model(args: argparse.Namespace, device: torch.device) -> PromptProposalModel:
    builder_fn = load_builder(args.builder)
    sam2 = builder_fn(args.model_cfg, args.checkpoint, device=device)
    model = PromptProposalModel(
        sam2_model=sam2,
        proposal_backbone=args.proposal_backbone,
        decoder_type=args.decoder_type,
        decoder_dim=args.decoder_dim,
        decoder_depth=args.decoder_depth,
        fpn_levels=args.fpn_levels,
        decoder_dropout=args.decoder_dropout,
        unet_base_ch=args.unet_base_ch,
        unet_out_ch=args.unet_out_ch,
        num_prompts=args.num_prompts,
        prompt_source=args.prompt_source,
        coarse_prompt_threshold=args.coarse_prompt_threshold,
        coarse_prompt_box_std_scale=args.coarse_prompt_box_std_scale,
        coarse_prompt_min_box_size_px=args.coarse_prompt_min_box_size_px,
        coarse_prompt_box_pad_px=args.coarse_prompt_box_pad_px,
    ).to(device)
    return model


def materialize_lazy_modules(model: nn.Module, sample_image: torch.Tensor, device: torch.device, amp_dtype: str, run_sam: bool) -> None:
    was_training = model.training
    model.eval()
    # Do not use torch.no_grad() for the entire forward when materializing the SAM-guided path;
    # we want to catch prompt-gradient compatible direct-SAM errors early.
    with safe_autocast(device, amp_dtype):
        _ = model(sample_image[None].to(device), run_sam=run_sam)
    model.train(was_training)


# =============================================================================
# Main training routine
# =============================================================================


def copy_args_with_overrides(args: argparse.Namespace, **overrides) -> argparse.Namespace:
    d = vars(args).copy()
    d.update(overrides)
    return argparse.Namespace(**d)


def curriculum_state(args: argparse.Namespace, epoch: int) -> Dict[str, object]:
    """Return the effective stage and SAM-loss weights for one epoch.

    Stages when --curriculum-training is enabled:
      0) proposal_warmup: no SAM-guided losses
      1) sam_ramp: SAM-guided losses enabled with a linear lambda factor
      2) sam_full: full requested SAM-guided lambda values

    If --no-sam-guided-loss is set, SAM stays disabled in all stages.
    """
    if not getattr(args, "curriculum_training", False):
        run_sam = bool(args.sam_guided_loss)
        return {
            "stage_index": 0,
            "stage_name": "single",
            "sam_factor": 1.0 if run_sam else 0.0,
            "sam_guided_loss": run_sam,
            "lambda_sam_point": float(args.lambda_sam_point) if run_sam else 0.0,
            "lambda_sam_box": float(args.lambda_sam_box) if run_sam else 0.0,
        }

    warmup = int(args.curriculum_warmup_epochs)
    ramp = int(args.curriculum_ramp_epochs)
    base_run_sam = bool(args.sam_guided_loss)

    if epoch <= warmup:
        return {
            "stage_index": 0,
            "stage_name": "proposal_warmup",
            "sam_factor": 0.0,
            "sam_guided_loss": False,
            "lambda_sam_point": 0.0,
            "lambda_sam_box": 0.0,
        }

    if ramp > 0 and epoch <= warmup + ramp:
        if ramp == 1:
            progress = 1.0
        else:
            progress = float(epoch - warmup - 1) / float(max(ramp - 1, 1))
        start = float(args.curriculum_ramp_start_factor)
        end = float(args.curriculum_ramp_end_factor)
        factor = start + (end - start) * progress
        factor = float(np.clip(factor, 0.0, max(start, end, 1.0)))
        run_sam = base_run_sam and factor > 0
        return {
            "stage_index": 1,
            "stage_name": "sam_ramp",
            "sam_factor": factor if run_sam else 0.0,
            "sam_guided_loss": run_sam,
            "lambda_sam_point": float(args.lambda_sam_point) * factor if run_sam else 0.0,
            "lambda_sam_box": float(args.lambda_sam_box) * factor if run_sam else 0.0,
        }

    run_sam = base_run_sam
    return {
        "stage_index": 2 if warmup > 0 or ramp > 0 else 0,
        "stage_name": "sam_full" if run_sam else "proposal_only",
        "sam_factor": 1.0 if run_sam else 0.0,
        "sam_guided_loss": run_sam,
        "lambda_sam_point": float(args.lambda_sam_point) if run_sam else 0.0,
        "lambda_sam_box": float(args.lambda_sam_box) if run_sam else 0.0,
    }


def save_config(args: argparse.Namespace, out_dir: Path, index: DatasetIndex, positive_ids: Sequence[str], splits: Dict[str, List[str]]) -> None:
    cfg = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "hostname": socket.gethostname(),
        "cwd": os.getcwd(),
        "argv": os.sys.argv,
        "args": vars(args),
        "dataset": {
            "dataset_dir": str(index.dataset_dir),
            "volumes_dir": str(index.volumes_dir),
            "masks_dir": str(index.masks_dir),
            "annotations_csv": str(index.annotations_csv),
            "links_csv": str(index.links_csv),
            "n_initial_volumes": len(index.volume_ids),
            "n_positive_volumes": len(positive_ids),
            "split_sizes": {k: len(v) for k, v in splits.items()},
        },
    }
    with open(out_dir / "config.json", "w") as f:
        json.dump(cfg, f, indent=2)


def train(args: argparse.Namespace) -> None:
    seed_everything(args.seed)
    device = select_device(args.device)
    configure_torch(device, args.allow_tf32)
    out_dir = make_output_dir(args)

    index = build_dataset_index(args)
    positive_ids = filter_positive_volume_ids(index, args)
    splits = split_volume_ids(
        positive_ids,
        seed=args.seed,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        train_case_list=args.train_case_list,
        val_case_list=args.val_case_list,
        test_case_list=args.test_case_list,
        shuffle_splits=args.shuffle_splits,
    )
    write_split_files(splits, out_dir)
    save_config(args, out_dir, index, positive_ids, splits)
    print(f"Split sizes: train={len(splits['train'])}, val={len(splits['val'])}, test={len(splits['test'])}")

    train_ds = PositiveSlicePromptDataset(
        index=index,
        volume_ids=splits["train"],
        use_triplet_channels=args.use_triplet_channels,
        hu_min=args.hu_min,
        hu_max=args.hu_max,
        min_slice_mask_pixels=args.min_slice_mask_pixels,
        image_size=args.image_size,
        cache_cases=args.cache_cases,
        augment_params=build_augment_params(args),
        desc="train",
    )
    val_ids_for_loader = splits["val"] if splits["val"] else splits["train"]
    val_ds = PositiveSlicePromptDataset(
        index=index,
        volume_ids=val_ids_for_loader,
        use_triplet_channels=args.use_triplet_channels,
        hu_min=args.hu_min,
        hu_max=args.hu_max,
        min_slice_mask_pixels=args.min_slice_mask_pixels,
        image_size=args.image_size,
        cache_cases=args.cache_cases,
        augment_params={"enabled": False},
        desc="val",
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
        collate_fn=collate_prompt,
        persistent_workers=args.num_workers > 0,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
        collate_fn=collate_prompt,
        persistent_workers=args.num_workers > 0,
    )

    model = build_model(args, device)
    sample_image = train_ds[0]["image"]
    materialize_lazy_modules(model, sample_image, device, args.amp_dtype, run_sam=False)

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable_params, lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(args.epochs, 1))
    scaler = torch.cuda.amp.GradScaler(enabled=(device.type == "cuda" and args.amp_dtype == "fp16"))

    best_val = float("inf")
    bad_epochs = 0
    metrics_rows: List[Dict] = []

    print(f"Training prompt-proposal model. Output: {out_dir}")
    print(
        f"Proposal backbone: {args.proposal_backbone}; decoder={args.decoder_type}; "
        f"prompt_source={args.prompt_source}; sam_guided_loss={args.sam_guided_loss}; "
        f"curriculum={args.curriculum_training}"
    )
    print(f"Trainable parameters: {sum(p.numel() for p in trainable_params):,}")

    current_stage_index: Optional[int] = None
    stage_best_initialized = False
    current_stage_name = ""

    for epoch in range(1, args.epochs + 1):
        stage = curriculum_state(args, epoch)
        stage_index = int(stage["stage_index"])
        if current_stage_index != stage_index:
            current_stage_index = stage_index
            current_stage_name = str(stage["stage_name"])
            bad_epochs = 0
            best_val = float("inf")
            stage_best_initialized = False
            print(
                f"\n=== Starting curriculum stage {stage_index}: {current_stage_name} "
                f"at epoch {epoch} | sam_factor={float(stage['sam_factor']):.3f} | "
                f"run_sam={bool(stage['sam_guided_loss'])} ==="
            )

        epoch_args = copy_args_with_overrides(
            args,
            sam_guided_loss=bool(stage["sam_guided_loss"]),
            lambda_sam_point=float(stage["lambda_sam_point"]),
            lambda_sam_box=float(stage["lambda_sam_box"]),
        )

        train_log = run_epoch(
            model,
            train_loader,
            optimizer,
            scaler,
            device,
            args.amp_dtype,
            args.threshold,
            desc=f"train {epoch}/{args.epochs} [{current_stage_name}]",
            args=epoch_args,
        )
        val_log = run_epoch(
            model,
            val_loader,
            None,
            None,
            device,
            args.amp_dtype,
            args.threshold,
            desc=f"val {epoch}/{args.epochs} [{current_stage_name}]",
            args=epoch_args,
        )
        scheduler.step()

        row = {
            "epoch": epoch,
            "curriculum_stage_index": stage_index,
            "curriculum_stage": current_stage_name,
            "sam_factor": float(stage["sam_factor"]),
            "effective_lambda_sam_point": float(stage["lambda_sam_point"]),
            "effective_lambda_sam_box": float(stage["lambda_sam_box"]),
            **{f"train_{k}": v for k, v in train_log.items()},
            **{f"val_{k}": v for k, v in val_log.items()},
            "lr": optimizer.param_groups[0]["lr"],
        }
        metrics_rows.append(row)
        pd.DataFrame(metrics_rows).to_csv(out_dir / "metrics.csv", index=False)
        with open(out_dir / "metrics.json", "w") as f:
            json.dump(metrics_rows, f, indent=2)

        print(
            f"Epoch {epoch:03d}: "
            f"train_loss={train_log['loss']:.5f}, train_DSC_coarse={train_log['DSC_coarse']:.4f}, "
            f"train_DSC_sam_point={train_log['DSC_sam_point']:.4f}, train_DSC_sam_box={train_log['DSC_sam_box']:.4f}, "
            f"val_loss={val_log['loss']:.5f}, val_DSC_coarse={val_log['DSC_coarse']:.4f}, "
            f"val_DSC_sam_point={val_log['DSC_sam_point']:.4f}, val_DSC_sam_box={val_log['DSC_sam_box']:.4f}, "
            f"lr={optimizer.param_groups[0]['lr']:.3e}"
        )

        save_checkpoint(out_dir / "last_model.pt", model, optimizer, epoch, best_val, args)
        val_metric = float(val_log["loss"])
        if not stage_best_initialized:
            best_val = val_metric
            bad_epochs = 0
            stage_best_initialized = True
            save_checkpoint(out_dir / "best_model.pt", model, optimizer, epoch, best_val, args)
            save_checkpoint(out_dir / f"best_model_stage{stage_index}_{current_stage_name}.pt", model, optimizer, epoch, best_val, args)
            print(f"  reset stage best at val_loss={best_val:.5f} and saved best_model.pt")
        elif val_metric < best_val - args.min_delta:
            best_val = val_metric
            bad_epochs = 0
            save_checkpoint(out_dir / "best_model.pt", model, optimizer, epoch, best_val, args)
            save_checkpoint(out_dir / f"best_model_stage{stage_index}_{current_stage_name}.pt", model, optimizer, epoch, best_val, args)
            print(f"  saved best_model.pt with stage val_loss={best_val:.5f}")
        else:
            bad_epochs += 1
            if args.patience > 0 and bad_epochs >= args.patience:
                print(f"Early stopping in stage {stage_index} ({current_stage_name}) after {bad_epochs} bad epochs.")
                break

    plot_training_metrics(out_dir / "metrics.csv", out_dir)

    if args.run_test_after_training and splits["test"]:
        ckpts = []
        if "best" in args.eval_checkpoints:
            ckpts.append(("best_model", out_dir / "best_model.pt"))
        if "last" in args.eval_checkpoints:
            ckpts.append(("last_model", out_dir / "last_model.pt"))
        for tag, ckpt in ckpts:
            if ckpt.exists():
                evaluate_checkpoint(args, out_dir, ckpt, tag, index, splits["test"], device, sample_image)
            else:
                print(f"WARNING: checkpoint missing, skipping {tag}: {ckpt}")
    elif args.run_test_after_training:
        print("Test split is empty; skipping test prediction.")

    print(f"Done. Results: {out_dir}")


# =============================================================================
# CLI
# =============================================================================


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Frozen-MedSAM2-guided prompt proposal training for positive-slice 2.5D LUNA nodule segmentation",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Dataset and splitting
    p.add_argument("--dataset-dir", required=True)
    p.add_argument("--volumes-dir", default=None)
    p.add_argument("--masks-dir", default=None)
    p.add_argument("--annotations-csv", default=None)
    p.add_argument("--links-csv", default=None)
    p.add_argument("--case-list", default=None, help="Optional global case filter before positive-volume filtering and splitting.")
    p.add_argument("--train-case-list", default=None, help="Explicit train SeriesUIDs, one per line.")
    p.add_argument("--val-case-list", default=None, help="Explicit validation SeriesUIDs, one per line.")
    p.add_argument("--test-case-list", default=None, help="Explicit test SeriesUIDs, one per line.")
    p.add_argument("--val-ratio", type=float, default=0.10)
    p.add_argument("--test-ratio", type=float, default=0.10)
    p.add_argument("--shuffle-splits", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--only-annotated", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--dataset-fraction", type=float, default=None)
    p.add_argument("--max-cases", type=int, default=None)
    p.add_argument("--shuffle", action="store_true", help="Shuffle global case order before optional dataset_fraction/max_cases filtering.")
    p.add_argument("--seed", type=int, default=123)

    # MedSAM2/SAM2 model
    p.add_argument("--model-cfg", required=True)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--builder", default="sam2.build_sam:build_sam2", help="Model builder import path.")
    p.add_argument("--proposal-backbone", choices=["medsam2-fpn", "unet"], default="medsam2-fpn")
    p.add_argument("--decoder-type", choices=["simple", "deep", "fpn"], default="fpn", help="Used only with --proposal-backbone medsam2-fpn.")
    p.add_argument("--decoder-dim", type=int, default=256)
    p.add_argument("--decoder-depth", type=int, default=2, help="For deep/simple: number of ConvGNAct blocks. For fpn: smoothing blocks per FPN level.")
    p.add_argument("--fpn-levels", type=int, default=3)
    p.add_argument("--decoder-dropout", type=float, default=0.0)
    p.add_argument("--unet-base-ch", type=int, default=32)
    p.add_argument("--unet-out-ch", type=int, default=128)
    p.add_argument("--num-prompts", type=int, default=1, help="Currently only 1 is implemented; output tensors keep a prompt dimension for future extension.")
    p.add_argument(
        "--prompt-source",
        choices=["heads", "coarse-soft", "coarse-hard"],
        default="heads",
        help=(
            "Which prompts are sent to frozen MedSAM2. 'heads' uses the learned point/box heads. "
            "'coarse-soft' derives differentiable moment-based point/box prompts from the coarse mask. "
            "'coarse-hard' thresholds the coarse mask and extracts the largest component/bbox; this is not differentiable."
        ),
    )
    p.add_argument("--supervise-derived-prompts", action=argparse.BooleanOptionalAction, default=True,
                   help="When --prompt-source coarse-soft, also apply point/box losses to the derived prompts.")
    p.add_argument("--coarse-prompt-threshold", type=float, default=0.5, help="Threshold used by --prompt-source coarse-hard.")
    p.add_argument("--coarse-prompt-box-std-scale", type=float, default=2.0, help="Box half-size multiplier for --prompt-source coarse-soft.")
    p.add_argument("--coarse-prompt-min-box-size-px", type=int, default=3, help="Minimum derived prompt-box side length.")
    p.add_argument("--coarse-prompt-box-pad-px", type=int, default=0, help="Extra padding in pixels around coarse-derived prompt boxes.")

    # Image/slice setup
    p.add_argument("--use-triplet-channels", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--hu-min", type=float, default=-1000.0)
    p.add_argument("--hu-max", type=float, default=400.0)
    p.add_argument("--image-size", type=int, default=None, help="Optional square resize for training/inference. Strongly recommended to match the SAM2 config size, e.g. 512.")
    p.add_argument("--min-slice-mask-pixels", type=int, default=1)

    # Loss weights
    p.add_argument("--coarse-bce-weight", type=float, default=0.5)
    p.add_argument("--coarse-dice-weight", type=float, default=0.5)
    p.add_argument("--sam-bce-weight", type=float, default=0.5)
    p.add_argument("--sam-dice-weight", type=float, default=0.5)
    p.add_argument("--lambda-coarse", type=float, default=1.0)
    p.add_argument("--lambda-point", type=float, default=1.0)
    p.add_argument("--lambda-box-l1", type=float, default=2.0)
    p.add_argument("--lambda-box-giou", type=float, default=1.0)
    p.add_argument("--lambda-sam-point", type=float, default=1.0)
    p.add_argument("--lambda-sam-box", type=float, default=1.0)
    p.add_argument("--sam-guided-loss", action=argparse.BooleanOptionalAction, default=True, help="Use frozen MedSAM2 point/box output losses during training.")

    # Curriculum / staged training
    p.add_argument("--curriculum-training", action=argparse.BooleanOptionalAction, default=False,
                   help="Enable staged training: proposal-only warmup, SAM-loss ramp, then full SAM-guided training.")
    p.add_argument("--curriculum-warmup-epochs", type=int, default=30,
                   help="Proposal-only warmup epochs when --curriculum-training is enabled.")
    p.add_argument("--curriculum-ramp-epochs", type=int, default=30,
                   help="Epochs over which SAM loss lambdas are linearly ramped when --curriculum-training is enabled.")
    p.add_argument("--curriculum-ramp-start-factor", type=float, default=0.25,
                   help="Initial multiplier for lambda-sam-point/box in the SAM ramp stage.")
    p.add_argument("--curriculum-ramp-end-factor", type=float, default=1.0,
                   help="Final multiplier for lambda-sam-point/box in the SAM ramp stage.")

    # Training
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch-size", type=int, default=2)
    p.add_argument("--eval-batch-size", type=int, default=4)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--threshold", type=float, default=0.5)
    p.add_argument("--patience", type=int, default=10)
    p.add_argument("--min-delta", type=float, default=1e-5)
    p.add_argument("--grad-clip-norm", type=float, default=0.0)
    p.add_argument("--cache-cases", action="store_true")
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--device", default="auto")
    p.add_argument("--amp-dtype", choices=["bf16", "fp16", "none"], default="bf16")
    p.add_argument("--allow-tf32", action=argparse.BooleanOptionalAction, default=True)

    # Augmentation
    p.add_argument("--augment", action=argparse.BooleanOptionalAction, default=False)
    p.add_argument("--aug-hflip-p", type=float, default=0.5)
    p.add_argument("--aug-vflip-p", type=float, default=0.5)
    p.add_argument("--aug-rotation-deg", type=float, default=15.0)
    p.add_argument("--aug-shift-px", type=float, default=16.0)
    p.add_argument("--aug-scale-min", type=float, default=0.90)
    p.add_argument("--aug-scale-max", type=float, default=1.10)
    p.add_argument("--aug-intensity-p", type=float, default=0.8)
    p.add_argument("--aug-brightness", type=float, default=0.10)
    p.add_argument("--aug-contrast", type=float, default=0.10)
    p.add_argument("--aug-noise-std", type=float, default=0.02)
    p.add_argument("--aug-blur-p", type=float, default=0.10)

    # Outputs/evaluation
    p.add_argument("--output-dir", required=True)
    p.add_argument("--create-experiment-dir", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--experiment-name", default=None)
    p.add_argument("--overwrite-experiment", action="store_true")
    p.add_argument("--run-test-after-training", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--eval-checkpoints", nargs="+", choices=["best", "last"], default=["best", "last"])
    p.add_argument("--eval-sam-outputs", action=argparse.BooleanOptionalAction, default=True, help="Save/evaluate frozen SAM point/box masks at test time.")
    p.add_argument("--save-volumes", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--save-prob-volumes", action=argparse.BooleanOptionalAction, default=False)
    p.add_argument("--save-gt-volume", action=argparse.BooleanOptionalAction, default=False)
    p.add_argument("--fail-fast", action="store_true")

    return p


def validate_args(args: argparse.Namespace) -> None:
    if not (0.0 <= args.val_ratio < 1.0):
        raise ValueError("--val-ratio must be in [0, 1)")
    if not (0.0 <= args.test_ratio < 1.0):
        raise ValueError("--test-ratio must be in [0, 1)")
    if args.val_ratio + args.test_ratio >= 1.0 and not (args.train_case_list or args.val_case_list or args.test_case_list):
        raise ValueError("Need --val-ratio + --test-ratio < 1 unless explicit split files are used")
    if args.image_size is not None and args.image_size <= 0:
        raise ValueError("--image-size must be positive when provided")
    if args.decoder_depth < 1:
        raise ValueError("--decoder-depth must be >= 1")
    if args.fpn_levels < 1:
        raise ValueError("--fpn-levels must be >= 1")
    if not (0.0 <= args.decoder_dropout < 1.0):
        raise ValueError("--decoder-dropout must be in [0, 1)")
    if args.num_prompts != 1:
        raise ValueError("This first implementation supports --num-prompts 1 only")
    if args.prompt_source == "coarse-hard" and args.supervise_derived_prompts:
        print("NOTE: --supervise-derived-prompts has no effect with --prompt-source coarse-hard because hard extraction is non-differentiable.")
    if args.coarse_prompt_min_box_size_px < 1:
        raise ValueError("--coarse-prompt-min-box-size-px must be >= 1")
    if args.coarse_prompt_box_pad_px < 0:
        raise ValueError("--coarse-prompt-box-pad-px must be >= 0")
    if args.coarse_prompt_box_std_scale <= 0:
        raise ValueError("--coarse-prompt-box-std-scale must be > 0")
    if not (0.0 <= args.coarse_prompt_threshold <= 1.0):
        raise ValueError("--coarse-prompt-threshold must be in [0, 1]")
    if args.curriculum_warmup_epochs < 0 or args.curriculum_ramp_epochs < 0:
        raise ValueError("Curriculum epoch counts must be >= 0")
    if args.curriculum_ramp_start_factor < 0 or args.curriculum_ramp_end_factor < 0:
        raise ValueError("Curriculum SAM lambda factors must be >= 0")
    for name in [
        "coarse_bce_weight",
        "coarse_dice_weight",
        "sam_bce_weight",
        "sam_dice_weight",
        "lambda_coarse",
        "lambda_point",
        "lambda_box_l1",
        "lambda_box_giou",
        "lambda_sam_point",
        "lambda_sam_box",
    ]:
        if getattr(args, name) < 0:
            raise ValueError(f"--{name.replace('_', '-')} must be >= 0")
    if args.coarse_bce_weight + args.coarse_dice_weight <= 0:
        raise ValueError("At least one coarse mask loss weight must be positive")
    if args.sam_guided_loss and args.sam_bce_weight + args.sam_dice_weight <= 0:
        raise ValueError("At least one SAM mask loss weight must be positive when --sam-guided-loss is enabled")
    if args.proposal_backbone == "unet" and args.decoder_type != "fpn":
        print("NOTE: --decoder-type is ignored with --proposal-backbone unet")
    if args.aug_scale_min <= 0 or args.aug_scale_max <= 0 or args.aug_scale_min > args.aug_scale_max:
        raise ValueError("Need 0 < --aug-scale-min <= --aug-scale-max")
    for name in ["aug_hflip_p", "aug_vflip_p", "aug_intensity_p", "aug_blur_p"]:
        val = getattr(args, name)
        if not (0.0 <= val <= 1.0):
            raise ValueError(f"--{name.replace('_', '-')} must be in [0, 1]")
    for name in ["aug_rotation_deg", "aug_shift_px", "aug_brightness", "aug_contrast", "aug_noise_std"]:
        if getattr(args, name) < 0:
            raise ValueError(f"--{name.replace('_', '-')} must be >= 0")


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()
    validate_args(args)
    train(args)


if __name__ == "__main__":
    main()
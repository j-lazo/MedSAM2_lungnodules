#!/usr/bin/env python3
"""
Fully supervised 2.5D positive-slice LUNA nodule segmentation with a residual attention U-Net.

This script keeps the same positive-slice experiment protocol as the MedSAM/SAM script:
  - split by patient / SeriesUID
  - keep only volumes that have nodule mask voxels
  - train only on slices where the GT nodule mask is non-empty
  - predict only those selected positive slices at test time
  - write same-size predicted 3D volumes, with zeros on non-selected slices
  - save slice-wise, patient-wise, volume-wise, and optional nodule-wise metrics

Expected LUNA-style layout, matching the earlier MedSAM/SAM experiment script:
  DATASET_DIR/CT_volumes/*.mhd
  DATASET_DIR/masks_nodules/nifti_data/*mask*contour*nodule*.nii.gz
  DATASET_DIR/annotations.csv                         with column: seriesuid
  DATASET_DIR/LUNA16_metadata_split_offical.csv       with columns: SeriesID,CID

Run from anywhere with the listed Python dependencies available; no SAM/MedSAM checkpoint is required.
"""

from __future__ import annotations

import argparse
import contextlib
import gc
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
        arch = f"resattnunet-b{getattr(args, 'base_channels', 32)}-d{getattr(args, 'unet_depth', 4)}"
        aug = "aug" if getattr(args, "augment", False) else "noaug"
        name = args.experiment_name or "_".join(
            [
                stamp,
                "resattnunet_posslice_seg",
                arch,
                aug,
                triplet,
                size,
                f"ep{args.epochs}",
                f"bs{args.batch_size}",
                f"lr{str(args.lr).replace('.', 'p')}",
                f"val{str(args.val_ratio).replace('.', 'p')}",
                f"test{str(args.test_ratio).replace('.', 'p')}",
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
    """Apply lightweight CT-safe 2D augmentations to one resized slice/mask pair.

    Geometric transforms are shared by image and mask. Intensity transforms are
    applied only to the image. The image remains uint8 H,W,3 and mask remains
    uint8 H,W. This is intentionally conservative for small nodule masks.
    """
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
        rgb = cv2.warpAffine(
            rgb, M, (W, H), flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REFLECT_101,
        )
        mask = cv2.warpAffine(
            mask, M, (W, H), flags=cv2.INTER_NEAREST,
            borderMode=cv2.BORDER_CONSTANT, borderValue=0,
        )

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


class PositiveSliceSegDataset(Dataset):
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
        return {
            "image": numpy_rgb_to_tensor(rgb),
            "mask": numpy_mask_to_tensor(mask),
            "series_id": s.series_id,
            "z": int(s.z),
            "orig_hw": torch.tensor(orig_hw, dtype=torch.long),
        }


def collate_seg(batch: List[Dict]) -> Dict:
    return {
        "image": torch.stack([b["image"] for b in batch], dim=0),
        "mask": torch.stack([b["mask"] for b in batch], dim=0),
        "series_id": [b["series_id"] for b in batch],
        "z": torch.tensor([b["z"] for b in batch], dtype=torch.long),
        "orig_hw": torch.stack([b["orig_hw"] for b in batch], dim=0),
    }


# =============================================================================
# Model
# =============================================================================


def _valid_group_count(channels: int, requested_groups: int) -> int:
    g = min(int(requested_groups), int(channels))
    while g > 1 and channels % g != 0:
        g -= 1
    return max(1, g)


class ConvNormAct(nn.Module):
    """3x3 convolution + normalization + SiLU activation.

    GroupNorm is the default because it is stable for the small batch sizes that
    are common in CT nodule segmentation.
    """

    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        kernel_size: int = 3,
        stride: int = 1,
        groups: int = 8,
        dropout: float = 0.0,
        norm: str = "group",
    ):
        super().__init__()
        padding = kernel_size // 2
        layers: List[nn.Module] = [
            nn.Conv2d(in_ch, out_ch, kernel_size=kernel_size, stride=stride, padding=padding, bias=False),
        ]
        if norm == "group":
            layers.append(nn.GroupNorm(_valid_group_count(out_ch, groups), out_ch))
        elif norm == "batch":
            layers.append(nn.BatchNorm2d(out_ch))
        elif norm == "instance":
            layers.append(nn.InstanceNorm2d(out_ch, affine=True))
        elif norm == "none":
            pass
        else:
            raise ValueError(f"Unknown norm={norm!r}; use group, batch, instance, or none.")
        layers.append(nn.SiLU(inplace=True))
        if dropout > 0:
            layers.append(nn.Dropout2d(float(dropout)))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class SqueezeExcite2d(nn.Module):
    """Channel recalibration block used inside residual units."""

    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()
        hidden = max(4, channels // max(1, int(reduction)))
        self.net = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, hidden, kernel_size=1),
            nn.SiLU(inplace=True),
            nn.Conv2d(hidden, channels, kernel_size=1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.net(x)


class ResidualBlock(nn.Module):
    """Pre-activation style residual block with optional squeeze-excitation."""

    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        groups: int = 8,
        dropout: float = 0.0,
        norm: str = "group",
        use_se: bool = True,
        se_reduction: int = 16,
    ):
        super().__init__()
        self.conv1 = ConvNormAct(in_ch, out_ch, groups=groups, dropout=dropout, norm=norm)
        self.conv2 = ConvNormAct(out_ch, out_ch, groups=groups, dropout=dropout, norm=norm)
        self.se = SqueezeExcite2d(out_ch, reduction=se_reduction) if use_se else nn.Identity()
        self.proj = nn.Identity() if in_ch == out_ch else nn.Conv2d(in_ch, out_ch, kernel_size=1, bias=False)
        self.act = nn.SiLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.proj(x)
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.se(out)
        return self.act(out + residual)


class ResidualStage(nn.Module):
    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        num_blocks: int,
        groups: int,
        dropout: float,
        norm: str,
        use_se: bool,
        se_reduction: int,
    ):
        super().__init__()
        blocks: List[nn.Module] = []
        for i in range(max(1, int(num_blocks))):
            blocks.append(
                ResidualBlock(
                    in_ch if i == 0 else out_ch,
                    out_ch,
                    groups=groups,
                    dropout=dropout,
                    norm=norm,
                    use_se=use_se,
                    se_reduction=se_reduction,
                )
            )
        self.net = nn.Sequential(*blocks)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class ASPP(nn.Module):
    """Atrous Spatial Pyramid Pooling at the bottleneck."""

    def __init__(self, channels: int, out_ch: int, rates: Sequence[int], groups: int, dropout: float, norm: str):
        super().__init__()
        self.branches = nn.ModuleList()
        self.branches.append(ConvNormAct(channels, out_ch, kernel_size=1, groups=groups, dropout=dropout, norm=norm))
        for r in rates:
            self.branches.append(
                nn.Sequential(
                    nn.Conv2d(channels, out_ch, kernel_size=3, padding=int(r), dilation=int(r), bias=False),
                    nn.GroupNorm(_valid_group_count(out_ch, groups), out_ch) if norm == "group" else nn.BatchNorm2d(out_ch),
                    nn.SiLU(inplace=True),
                )
            )
        self.image_pool = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            ConvNormAct(channels, out_ch, kernel_size=1, groups=groups, dropout=0.0, norm=norm),
        )
        self.project = ConvNormAct(out_ch * (len(rates) + 2), out_ch, kernel_size=1, groups=groups, dropout=dropout, norm=norm)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        size = x.shape[-2:]
        outs = [branch(x) for branch in self.branches]
        pooled = self.image_pool(x)
        pooled = F.interpolate(pooled, size=size, mode="bilinear", align_corners=False)
        outs.append(pooled)
        return self.project(torch.cat(outs, dim=1))


class AttentionGate(nn.Module):
    """Additive attention gate for filtering encoder skip features."""

    def __init__(self, skip_ch: int, gate_ch: int, inter_ch: int):
        super().__init__()
        inter_ch = max(1, int(inter_ch))
        self.skip_proj = nn.Conv2d(skip_ch, inter_ch, kernel_size=1, bias=False)
        self.gate_proj = nn.Conv2d(gate_ch, inter_ch, kernel_size=1, bias=False)
        self.psi = nn.Sequential(
            nn.SiLU(inplace=True),
            nn.Conv2d(inter_ch, 1, kernel_size=1),
            nn.Sigmoid(),
        )

    def forward(self, skip: torch.Tensor, gate: torch.Tensor) -> torch.Tensor:
        if gate.shape[-2:] != skip.shape[-2:]:
            gate = F.interpolate(gate, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        alpha = self.psi(self.skip_proj(skip) + self.gate_proj(gate))
        return skip * alpha


class UpBlock(nn.Module):
    def __init__(
        self,
        in_ch: int,
        skip_ch: int,
        out_ch: int,
        num_blocks: int,
        groups: int,
        dropout: float,
        norm: str,
        use_se: bool,
        se_reduction: int,
        use_attention: bool,
    ):
        super().__init__()
        self.up_proj = nn.Conv2d(in_ch, out_ch, kernel_size=1, bias=False)
        self.attn = AttentionGate(skip_ch, out_ch, inter_ch=max(out_ch // 2, 1)) if use_attention else nn.Identity()
        self.fuse = ResidualStage(
            out_ch + skip_ch,
            out_ch,
            num_blocks=num_blocks,
            groups=groups,
            dropout=dropout,
            norm=norm,
            use_se=use_se,
            se_reduction=se_reduction,
        )

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        x = self.up_proj(x)
        if isinstance(self.attn, AttentionGate):
            skip = self.attn(skip, x)
        return self.fuse(torch.cat([x, skip], dim=1))


class ResidualAttentionUNet(nn.Module):
    """Strong 2D U-Net baseline for small medical-object segmentation.

    Features included by default:
      - residual encoder/decoder blocks
      - GroupNorm + SiLU for small-batch stability
      - squeeze-excitation in residual blocks
      - attention-gated skip connections
      - ASPP bottleneck for multi-scale context
      - bilinear upsampling, which is less checkerboard-prone than transposed convs
    """

    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 1,
        base_channels: int = 32,
        depth: int = 4,
        blocks_per_stage: int = 2,
        bottleneck_blocks: int = 2,
        channel_multiplier: int = 2,
        max_channels: int = 512,
        groups: int = 8,
        dropout: float = 0.0,
        norm: str = "group",
        use_se: bool = True,
        se_reduction: int = 16,
        use_attention: bool = True,
        use_aspp: bool = True,
        aspp_rates: Sequence[int] = (1, 2, 4, 8),
    ):
        super().__init__()
        depth = max(2, int(depth))
        channels = [min(int(base_channels) * (int(channel_multiplier) ** i), int(max_channels)) for i in range(depth + 1)]
        self.channels = channels
        self.depth = depth

        self.encoders = nn.ModuleList()
        in_ch = int(in_channels)
        for out_ch in channels[:-1]:
            self.encoders.append(
                ResidualStage(
                    in_ch,
                    out_ch,
                    num_blocks=blocks_per_stage,
                    groups=groups,
                    dropout=dropout,
                    norm=norm,
                    use_se=use_se,
                    se_reduction=se_reduction,
                )
            )
            in_ch = out_ch
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.bottleneck = ResidualStage(
            in_ch,
            channels[-1],
            num_blocks=bottleneck_blocks,
            groups=groups,
            dropout=dropout,
            norm=norm,
            use_se=use_se,
            se_reduction=se_reduction,
        )
        self.aspp = ASPP(channels[-1], channels[-1], rates=aspp_rates, groups=groups, dropout=dropout, norm=norm) if use_aspp else nn.Identity()

        self.decoders = nn.ModuleList()
        prev_ch = channels[-1]
        for skip_ch, out_ch in zip(reversed(channels[:-1]), reversed(channels[:-1])):
            self.decoders.append(
                UpBlock(
                    prev_ch,
                    skip_ch,
                    out_ch,
                    num_blocks=blocks_per_stage,
                    groups=groups,
                    dropout=dropout,
                    norm=norm,
                    use_se=use_se,
                    se_reduction=se_reduction,
                    use_attention=use_attention,
                )
            )
            prev_ch = out_ch
        self.head = nn.Conv2d(channels[0], out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        input_hw = x.shape[-2:]
        skips: List[torch.Tensor] = []
        y = x
        for enc in self.encoders:
            y = enc(y)
            skips.append(y)
            y = self.pool(y)
        y = self.bottleneck(y)
        y = self.aspp(y)
        for dec, skip in zip(self.decoders, reversed(skips)):
            y = dec(y, skip)
        logits = self.head(y)
        if logits.shape[-2:] != input_hw:
            logits = F.interpolate(logits, size=input_hw, mode="bilinear", align_corners=False)
        return logits


def parse_int_tuple(value: str) -> Tuple[int, ...]:
    if value is None or str(value).strip() == "":
        return tuple()
    return tuple(int(x.strip()) for x in str(value).split(",") if x.strip())


def build_model(args: argparse.Namespace, device: torch.device) -> ResidualAttentionUNet:
    model = ResidualAttentionUNet(
        in_channels=3,
        out_channels=1,
        base_channels=args.base_channels,
        depth=args.unet_depth,
        blocks_per_stage=args.blocks_per_stage,
        bottleneck_blocks=args.bottleneck_blocks,
        channel_multiplier=args.channel_multiplier,
        max_channels=args.max_channels,
        groups=args.norm_groups,
        dropout=args.unet_dropout,
        norm=args.norm_type,
        use_se=args.use_se,
        se_reduction=args.se_reduction,
        use_attention=args.use_attention_gates,
        use_aspp=args.use_aspp,
        aspp_rates=parse_int_tuple(args.aspp_rates),
    ).to(device)
    return model


def materialize_lazy_modules(model: nn.Module, sample_image: torch.Tensor, device: torch.device, amp_dtype: str) -> None:
    """Kept for API compatibility with the MedSAM script; this U-Net has no lazy modules."""
    return None


# =============================================================================
# Losses and training
# =============================================================================


def soft_dice_loss_from_logits(logits: torch.Tensor, targets: torch.Tensor, smooth: float = 1.0) -> torch.Tensor:
    probs = torch.sigmoid(logits)
    dims = tuple(range(1, probs.ndim))
    inter = (probs * targets).sum(dim=dims)
    denom = probs.sum(dim=dims) + targets.sum(dim=dims)
    dice = (2.0 * inter + smooth) / (denom + smooth)
    return 1.0 - dice.mean()


def segmentation_loss(logits: torch.Tensor, targets: torch.Tensor, bce_weight: float, dice_weight: float) -> Tuple[torch.Tensor, Dict[str, float]]:
    bce = F.binary_cross_entropy_with_logits(logits, targets)
    dice = soft_dice_loss_from_logits(logits, targets)
    total = bce_weight * bce + dice_weight * dice
    return total, {"loss": float(total.detach()), "bce_loss": float(bce.detach()), "dice_loss": float(dice.detach())}


def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: Optional[torch.optim.Optimizer],
    scaler: Optional[torch.cuda.amp.GradScaler],
    device: torch.device,
    amp_dtype: str,
    bce_weight: float,
    dice_weight: float,
    threshold: float,
    desc: str,
) -> Dict[str, float]:
    is_train = optimizer is not None
    model.train(is_train)
    totals = {"loss": 0.0, "bce_loss": 0.0, "dice_loss": 0.0, "DSC": 0.0, "IoU": 0.0}
    n = 0

    pbar = tqdm(loader, desc=desc, leave=False)
    for batch in pbar:
        images = batch["image"].to(device, non_blocking=True)
        masks = batch["mask"].to(device, non_blocking=True)

        if is_train:
            optimizer.zero_grad(set_to_none=True)

        with torch.set_grad_enabled(is_train), safe_autocast(device, amp_dtype):
            logits = model(images)
            loss, logs = segmentation_loss(logits, masks, bce_weight=bce_weight, dice_weight=dice_weight)

        if is_train:
            if scaler is not None and amp_dtype == "fp16":
                scaler.scale(loss).backward()
                if getattr(loader, "grad_clip_norm", 0) > 0:
                    scaler.unscale_(optimizer)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                optimizer.step()

        with torch.no_grad():
            probs = torch.sigmoid(logits).detach().float().cpu().numpy()
            gt = masks.detach().float().cpu().numpy()
            pred = probs >= threshold
            bs_dice = []
            bs_iou = []
            for i in range(pred.shape[0]):
                bs_dice.append(dice_score(gt[i, 0] > 0.5, pred[i, 0]))
                bs_iou.append(iou_score(gt[i, 0] > 0.5, pred[i, 0]))

        bs = images.shape[0]
        totals["loss"] += logs["loss"] * bs
        totals["bce_loss"] += logs["bce_loss"] * bs
        totals["dice_loss"] += logs["dice_loss"] * bs
        totals["DSC"] += float(np.mean(bs_dice)) * bs
        totals["IoU"] += float(np.mean(bs_iou)) * bs
        n += bs
        pbar.set_postfix({k: f"{totals[k] / max(n, 1):.4f}" for k in ["loss", "DSC", "IoU"]})

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

    for cols, filename, title in [
        ([c for c in df.columns if "loss" in c.lower()], "losses.png", "Training/validation losses"),
        ([c for c in df.columns if c.endswith("DSC") or c.endswith("IoU")], "metrics.png", "Training/validation segmentation metrics"),
    ]:
        cols = [c for c in cols if c in df.columns]
        if not cols:
            continue
        plt.figure(figsize=(10, 6))
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
) -> Tuple[np.ndarray, np.ndarray, List[Dict]]:
    """Predict only GT-positive slices and return same-size volume masks/probabilities."""
    Z, H, W = case.gt_volume.shape
    pred_volume = np.zeros((Z, H, W), dtype=np.uint8)
    prob_volume = np.zeros((Z, H, W), dtype=np.float32)
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
                logits = model(x)
            probs = torch.sigmoid(logits).detach().float().cpu().numpy()[:, 0]
            preds = (probs >= args.threshold).astype(np.uint8)

            for i, z in enumerate(batch_zs):
                p_small = preds[i]
                prob_small = probs[i]
                p = resize_pred_to_hw(p_small, (H, W))
                if prob_small.shape != (H, W):
                    prob = cv2.resize(prob_small.astype(np.float32), (W, H), interpolation=cv2.INTER_LINEAR)
                else:
                    prob = prob_small.astype(np.float32)
                pred_volume[z] = p
                prob_volume[z] = prob
                gt_slice = case.gt_volume[z].astype(np.uint8)
                m = segmentation_metrics(gt_slice, p, spacing_yx)
                slice_rows.append(
                    {
                        "SeriesUID": case.series_id,
                        "z": int(z),
                        "status": "ok",
                        **m,
                    }
                )
    return pred_volume, prob_volume, slice_rows


def extract_gt_nodules_3d(gt_volume: np.ndarray, spacing_zyx: Tuple[float, float, float], min_voxels: int = 1) -> List[Dict]:
    labeled, num = ndimage.label(gt_volume.astype(bool), structure=ndimage.generate_binary_structure(3, 2))
    nodules: List[Dict] = []
    voxel_volume = float(spacing_zyx[0] * spacing_zyx[1] * spacing_zyx[2])
    for k in range(1, num + 1):
        comp = labeled == k
        vox = int(comp.sum())
        if vox < min_voxels:
            continue
        coords = np.argwhere(comp)
        zmin, ymin, xmin = coords.min(axis=0)
        zmax, ymax, xmax = coords.max(axis=0)
        extent_vox = np.array([zmax - zmin + 1, ymax - ymin + 1, xmax - xmin + 1], dtype=np.float32)
        extent_mm = extent_vox * np.array(spacing_zyx, dtype=np.float32)
        nodules.append(
            {
                "nodule_id": int(k),
                "mask": comp.astype(np.uint8),
                "bbox_zyx": [int(zmin), int(ymin), int(xmin), int(zmax), int(ymax), int(xmax)],
                "gt_volume_voxels": vox,
                "gt_volume_mm3": float(vox * voxel_volume),
                "gt_diameter_vox": float(max(extent_vox)),
                "gt_diameter_mm": float(max(extent_mm)),
            }
        )
    return nodules


def pred_components(pred_volume: np.ndarray, min_voxels: int = 1) -> Tuple[np.ndarray, List[int]]:
    labeled, num = ndimage.label(pred_volume.astype(bool), structure=ndimage.generate_binary_structure(3, 2))
    labels: List[int] = []
    for k in range(1, num + 1):
        if int((labeled == k).sum()) >= min_voxels:
            labels.append(int(k))
    return labeled, labels


def per_nodule_metrics(case: CaseData, pred_volume: np.ndarray, args: argparse.Namespace) -> List[Dict]:
    spacing_zyx = get_spacing_zyx(case.image_itk)
    gt_nodules = extract_gt_nodules_3d(case.gt_volume, spacing_zyx, min_voxels=args.min_nodule_voxels)
    pred_labeled, pred_labels = pred_components(pred_volume, min_voxels=args.min_nodule_voxels)
    rows: List[Dict] = []

    for nod in gt_nodules:
        gt_mask = nod["mask"].astype(bool)
        best_label = 0
        best_iou = 0.0
        for lab in pred_labels:
            pm = pred_labeled == lab
            val = iou_score(gt_mask, pm)
            if val > best_iou:
                best_iou = val
                best_label = int(lab)
        matched_pred = (pred_labeled == best_label) if best_label > 0 else np.zeros_like(pred_volume, dtype=bool)
        m = segmentation_metrics(gt_mask, matched_pred, spacing_zyx)
        rows.append(
            {
                "SeriesUID": case.series_id,
                "nodule_id": nod["nodule_id"],
                "matched_pred_component_id": best_label if best_label > 0 else np.nan,
                "matched_component_IoU": best_iou if best_label > 0 else 0.0,
                "bbox_zyx": json.dumps(nod["bbox_zyx"]),
                "GT_volume_voxels": nod["gt_volume_voxels"],
                "GT_volume_mm3": nod["gt_volume_mm3"],
                "GT_diameter_vox": nod["gt_diameter_vox"],
                "GT_diameter_mm": nod["gt_diameter_mm"],
                **m,
            }
        )
    return rows


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
    materialize_lazy_modules(model, sample_image, device, args.amp_dtype)
    load_checkpoint(model, ckpt_path, device)
    model.eval()

    slice_rows: List[Dict] = []
    volume_rows: List[Dict] = []
    nodule_rows: List[Dict] = []

    print(f"Evaluating {tag} checkpoint on {len(test_ids)} positive test volumes: {ckpt_path}")
    for sid in tqdm(test_ids, desc=f"test-{tag}"):
        try:
            case = load_case(index, sid)
            if case is None:
                volume_rows.append({"SeriesUID": sid, "status": "missing"})
                continue
            pred_vol, prob_vol, rows = predict_case_positive_slices(model, case, args, device)
            slice_rows.extend(rows)
            spacing_zyx = get_spacing_zyx(case.image_itk)
            vol_metrics = segmentation_metrics(case.gt_volume, pred_vol, spacing_zyx)
            volume_rows.append(
                {
                    "SeriesUID": sid,
                    "status": "ok",
                    "n_positive_slices": len(positive_slices_for_case(case, args.min_slice_mask_pixels)),
                    **vol_metrics,
                }
            )
            nodule_rows.extend(per_nodule_metrics(case, pred_vol, args))

            if args.save_volumes:
                write_pred_volume(pred_vol, case.image_itk, pred_vol_dir / f"{sid}_pred_{tag}.nii.gz")
                if args.save_prob_volumes:
                    write_prob_volume(prob_vol, case.image_itk, prob_vol_dir / f"{sid}_prob_{tag}.nii.gz")
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

    metric_cols = ["DSC", "IoU", "precision", "recall", "F1", "HD95_mm", "ASSD_mm", "volume_similarity"]

    pd.DataFrame(slice_rows).to_csv(pred_dir / "slice_metrics.csv", index=False)
    pd.DataFrame(volume_rows).to_csv(pred_dir / "patient_volume_metrics.csv", index=False)

    patient_slice_summary = aggregate_numeric(slice_rows, "SeriesUID", metric_cols)
    patient_slice_summary.to_csv(pred_dir / "patient_slice_summary.csv", index=False)

    if nodule_rows:
        pd.DataFrame(nodule_rows).to_csv(pred_dir / "nodule_metrics.csv", index=False)
        patient_nodule_summary = aggregate_numeric(nodule_rows, "SeriesUID", metric_cols)
        patient_nodule_summary.to_csv(pred_dir / "patient_nodule_summary.csv", index=False)

    ok = pd.DataFrame(volume_rows)
    if not ok.empty and "status" in ok.columns:
        ok = ok[ok["status"] == "ok"]
    summary = {}
    for c in metric_cols:
        vals = pd.to_numeric(ok[c], errors="coerce") if c in ok.columns else pd.Series(dtype=float)
        summary[f"mean_patient_volume_{c}"] = float(vals.mean()) if vals.notna().any() else np.nan
        summary[f"median_patient_volume_{c}"] = float(vals.median()) if vals.notna().any() else np.nan
    with open(pred_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    pd.DataFrame([summary]).to_csv(pred_dir / "summary.csv", index=False)
    print(f"Saved {tag} test outputs to: {pred_dir}")


# =============================================================================
# Main training routine
# =============================================================================


def save_config(args: argparse.Namespace, out_dir: Path, index: DatasetIndex, positive_ids: Sequence[str], splits: Dict[str, List[str]]) -> None:
    cfg = {
        "model": "U-net",
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

    train_ds = PositiveSliceSegDataset(
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
    val_ds = PositiveSliceSegDataset(
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
        collate_fn=collate_seg,
        persistent_workers=args.num_workers > 0,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
        collate_fn=collate_seg,
        persistent_workers=args.num_workers > 0,
    )

    model = build_model(args, device)
    sample_image = train_ds[0]["image"]
    materialize_lazy_modules(model, sample_image, device, args.amp_dtype)

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable_params, lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(args.epochs, 1))
    scaler = torch.cuda.amp.GradScaler(enabled=(device.type == "cuda" and args.amp_dtype == "fp16"))

    best_val = float("inf")
    bad_epochs = 0
    metrics_rows: List[Dict] = []

    print(f"Training residual attention U-Net positive-slice segmentation. Output: {out_dir}")
    print(
        f"U-Net: base_channels={args.base_channels}; depth={args.unet_depth}; "
        f"blocks_per_stage={args.blocks_per_stage}; max_channels={args.max_channels}; "
        f"attention={args.use_attention_gates}; SE={args.use_se}; ASPP={args.use_aspp}; dropout={args.unet_dropout}"
    )
    print(f"Augmentation enabled: {args.augment}")
    print(f"Trainable parameters: {sum(p.numel() for p in trainable_params):,}")
    for epoch in range(1, args.epochs + 1):
        train_log = run_epoch(
            model,
            train_loader,
            optimizer,
            scaler,
            device,
            args.amp_dtype,
            args.bce_weight,
            args.dice_weight,
            args.threshold,
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
                args.bce_weight,
                args.dice_weight,
                args.threshold,
                desc=f"val {epoch}/{args.epochs}",
            )
        scheduler.step()

        row = {
            "epoch": epoch,
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
            f"train_loss={train_log['loss']:.5f}, train_DSC={train_log['DSC']:.4f}, "
            f"val_loss={val_log['loss']:.5f}, val_DSC={val_log['DSC']:.4f}, "
            f"lr={optimizer.param_groups[0]['lr']:.3e}"
        )

        save_checkpoint(out_dir / "last_model.pt", model, optimizer, epoch, best_val, args)
        if val_log["loss"] < best_val - args.min_delta:
            best_val = val_log["loss"]
            bad_epochs = 0
            save_checkpoint(out_dir / "best_model.pt", model, optimizer, epoch, best_val, args)
            print(f"  saved best_model.pt with val_loss={best_val:.5f}")
        else:
            bad_epochs += 1
            if args.patience > 0 and bad_epochs >= args.patience:
                print(f"Early stopping after {bad_epochs} bad epochs.")
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
        description="Fully supervised residual attention U-Net positive-slice 2.5D LUNA nodule segmentation",
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

    # Residual attention U-Net model
    p.add_argument("--base-channels", type=int, default=32, help="Width of the first U-Net stage. Use 48/64 for a larger model if memory allows.")
    p.add_argument("--unet-depth", type=int, default=4, help="Number of downsampling stages before the bottleneck.")
    p.add_argument("--blocks-per-stage", type=int, default=2, help="Residual blocks in each encoder/decoder stage.")
    p.add_argument("--bottleneck-blocks", type=int, default=2, help="Residual blocks at the bottleneck.")
    p.add_argument("--channel-multiplier", type=int, default=2)
    p.add_argument("--max-channels", type=int, default=512)
    p.add_argument("--norm-type", choices=["group", "batch", "instance", "none"], default="group")
    p.add_argument("--norm-groups", type=int, default=8)
    p.add_argument("--unet-dropout", type=float, default=0.05)
    p.add_argument("--use-se", action=argparse.BooleanOptionalAction, default=True, help="Use squeeze-excitation inside residual blocks.")
    p.add_argument("--se-reduction", type=int, default=16)
    p.add_argument("--use-attention-gates", action=argparse.BooleanOptionalAction, default=True, help="Use attention gates on skip connections.")
    p.add_argument("--use-aspp", action=argparse.BooleanOptionalAction, default=True, help="Use ASPP in the bottleneck.")
    p.add_argument("--aspp-rates", default="1,2,4,8", help="Comma-separated dilation rates for ASPP.")

    # Image/slice setup
    p.add_argument("--use-triplet-channels", action=argparse.BooleanOptionalAction, default=True, help="Use z-1,z,z+1 as 3 input channels. Disable for single-slice copied to RGB.")
    p.add_argument("--hu-min", type=float, default=-1000.0)
    p.add_argument("--hu-max", type=float, default=400.0)
    p.add_argument("--image-size", type=int, default=None, help="Optional square resize for training/inference. Predictions are resized back to original H,W before saving volumes.")
    p.add_argument("--min-slice-mask-pixels", type=int, default=1, help="A slice is selected if GT nodule mask pixels >= this value.")
    p.add_argument("--min-nodule-voxels", type=int, default=1, help="Minimum connected-component size for nodule-wise metrics.")

    # Training
    p.add_argument("--epochs", type=int, default=80)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--eval-batch-size", type=int, default=16)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--bce-weight", type=float, default=0.5)
    p.add_argument("--dice-weight", type=float, default=0.5)
    p.add_argument("--threshold", type=float, default=0.5)
    p.add_argument("--patience", type=int, default=20)
    p.add_argument("--min-delta", type=float, default=1e-5)
    p.add_argument("--cache-cases", action="store_true")
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--device", default="auto")
    p.add_argument("--amp-dtype", choices=["bf16", "fp16", "none"], default="bf16")
    p.add_argument("--allow-tf32", action=argparse.BooleanOptionalAction, default=True)

    # Augmentation; applied only to the training split, after optional resize.
    p.add_argument("--augment", action=argparse.BooleanOptionalAction, default=True, help="Enable conservative 2D augmentations for training slices only.")
    p.add_argument("--aug-hflip-p", type=float, default=0.5)
    p.add_argument("--aug-vflip-p", type=float, default=0.5)
    p.add_argument("--aug-rotation-deg", type=float, default=15.0)
    p.add_argument("--aug-shift-px", type=float, default=16.0)
    p.add_argument("--aug-scale-min", type=float, default=0.90)
    p.add_argument("--aug-scale-max", type=float, default=1.10)
    p.add_argument("--aug-intensity-p", type=float, default=0.8)
    p.add_argument("--aug-brightness", type=float, default=0.10, help="Brightness jitter as a fraction of 255.")
    p.add_argument("--aug-contrast", type=float, default=0.10, help="Contrast jitter fraction around 1.0.")
    p.add_argument("--aug-noise-std", type=float, default=0.02, help="Gaussian noise std as a fraction of 255.")
    p.add_argument("--aug-blur-p", type=float, default=0.10, help="Probability of mild 3x3 Gaussian blur.")

    # Outputs/evaluation
    p.add_argument("--output-dir", required=True)
    p.add_argument("--create-experiment-dir", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--experiment-name", default=None)
    p.add_argument("--overwrite-experiment", action="store_true")
    p.add_argument("--run-test-after-training", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--eval-checkpoints", nargs="+", choices=["best", "last"], default=["best", "last"])
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
    if args.bce_weight < 0 or args.dice_weight < 0 or (args.bce_weight + args.dice_weight) <= 0:
        raise ValueError("Need non-negative --bce-weight/--dice-weight and at least one positive loss weight")
    if args.base_channels < 4:
        raise ValueError("--base-channels must be >= 4")
    if args.unet_depth < 2:
        raise ValueError("--unet-depth must be >= 2")
    if args.blocks_per_stage < 1:
        raise ValueError("--blocks-per-stage must be >= 1")
    if args.bottleneck_blocks < 1:
        raise ValueError("--bottleneck-blocks must be >= 1")
    if args.channel_multiplier < 1:
        raise ValueError("--channel-multiplier must be >= 1")
    if args.max_channels < args.base_channels:
        raise ValueError("--max-channels must be >= --base-channels")
    if args.norm_groups < 1:
        raise ValueError("--norm-groups must be >= 1")
    if not (0.0 <= args.unet_dropout < 1.0):
        raise ValueError("--unet-dropout must be in [0, 1)")
    if args.se_reduction < 1:
        raise ValueError("--se-reduction must be >= 1")
    rates = parse_int_tuple(args.aspp_rates)
    if args.use_aspp and (not rates or any(r < 1 for r in rates)):
        raise ValueError("--aspp-rates must contain positive integers when --use-aspp is enabled")
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
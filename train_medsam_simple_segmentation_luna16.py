#!/usr/bin/env python3
"""
Fully supervised 2.5D positive-slice LUNA nodule segmentation with a MedSAM2/SAM2 encoder.

This script is intentionally simpler than detector-guided SAM2 pipelines:
  - split by patient / SeriesUID
  - keep only volumes that have nodule mask voxels
  - train only on slices where the GT nodule mask is non-empty
  - predict only those selected positive slices at test time
  - write same-size predicted 3D volumes, with zeros on non-selected slices
  - save slice-wise, patient-wise, volume-wise, and optional nodule-wise metrics

Expected LUNA-style layout, matching the earlier MedSAM2 detector script:
  DATASET_DIR/CT_volumes/*.mhd
  DATASET_DIR/masks_nodules/nifti_data/*mask*contour*nodule*.nii.gz
  DATASET_DIR/annotations.csv                         with column: seriesuid
  DATASET_DIR/LUNA16_metadata_split_offical.csv       with columns: SeriesID,CID

Run from the MedSAM2/SAM2 repository root, or anywhere where the requested builder module is importable.
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
import importlib
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
        decoder = getattr(args, "decoder_type", "simple")
        aug = "aug" if getattr(args, "augment", False) else "noaug"
        name = args.experiment_name or "_".join(
            [
                stamp,
                "medsam2_posslice_seg",
                decoder,
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
    """Lazy 1x1 projection followed by GroupNorm and activation."""

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


class SimpleDecoder(nn.Module):
    """Original lightweight decoder: project final SAM2 feature, refine, predict."""

    def __init__(self, decoder_dim: int = 256, dropout: float = 0.0):
        super().__init__()
        self.net = nn.Sequential(
            LazyConvGNAct(decoder_dim, dropout=dropout),
            ConvGNAct(decoder_dim, decoder_dim, dropout=dropout),
            ConvGNAct(decoder_dim, decoder_dim, dropout=dropout),
            nn.Conv2d(decoder_dim, 1, kernel_size=1),
        )

    def forward(self, feat: torch.Tensor) -> torch.Tensor:
        return self.net(feat)


class DeepDecoder(nn.Module):
    """Deeper low-resolution decoder using the final SAM2 image feature only."""

    def __init__(self, decoder_dim: int = 256, depth: int = 4, dropout: float = 0.0):
        super().__init__()
        depth = max(1, int(depth))
        layers: List[nn.Module] = [LazyConvGNAct(decoder_dim, dropout=dropout)]
        for _ in range(depth):
            layers.append(ConvGNAct(decoder_dim, decoder_dim, dropout=dropout))
        layers.append(nn.Conv2d(decoder_dim, 1, kernel_size=1))
        self.net = nn.Sequential(*layers)

    def forward(self, feat: torch.Tensor) -> torch.Tensor:
        return self.net(feat)


class FPNDecoder(nn.Module):
    """Top-down FPN-like decoder over multiple MedSAM2/SAM2 backbone features.

    Features are expected in low-to-high spatial resolution order. Each feature is
    projected to decoder_dim with a lazy 1x1 conv, then higher-resolution features
    are fused top-down with bilinear upsampling and smoothing convolutions.
    """

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
        self.out_conv = nn.Conv2d(decoder_dim, 1, kernel_size=1)

    def forward(self, features: Sequence[torch.Tensor]) -> torch.Tensor:
        if not features:
            raise ValueError("FPNDecoder received an empty feature list")
        # Sort low -> high resolution. If there are more levels than requested,
        # keep the highest-resolution range available while still preserving order.
        feats = sorted(list(features), key=lambda t: int(t.shape[-2]) * int(t.shape[-1]))
        if len(feats) < self.num_levels:
            raise ValueError(
                f"FPN requested {self.num_levels} feature levels, but SAM2 returned only {len(feats)}. "
                "Lower --fpn-levels or use --decoder-type deep/simple."
            )
        if len(feats) > self.num_levels:
            feats = feats[-self.num_levels:]
        n = len(feats)
        y = self.lateral_convs[0](feats[0])
        y = self.smooth_convs[0](y)
        for i in range(1, n):
            lateral = self.lateral_convs[i](feats[i])
            y = F.interpolate(y, size=lateral.shape[-2:], mode="bilinear", align_corners=False)
            y = y + lateral
            y = self.smooth_convs[i](y)
        return self.out_conv(y)



def build_2d_sincos_position_embedding(
    channels: int,
    height: int,
    width: int,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Return a [1, C, H, W] 2D sine-cosine positional encoding.

    The implementation is intentionally lightweight and computed on the fly so it
    can be used with arbitrary SAM2/MedSAM2 feature resolutions. If channels is
    not divisible by four, the embedding is padded or truncated to exactly C.
    """
    channels = int(channels)
    num_freqs = max(1, channels // 4)
    y = torch.linspace(0.0, 1.0, steps=int(height), device=device, dtype=dtype)
    x = torch.linspace(0.0, 1.0, steps=int(width), device=device, dtype=dtype)
    yy, xx = torch.meshgrid(y, x, indexing="ij")
    omega = torch.arange(num_freqs, device=device, dtype=dtype)
    omega = 1.0 / (10000.0 ** (omega / max(num_freqs, 1)))
    pos_x = xx[..., None] * omega
    pos_y = yy[..., None] * omega
    pe = torch.cat([pos_x.sin(), pos_x.cos(), pos_y.sin(), pos_y.cos()], dim=-1)
    if pe.shape[-1] < channels:
        pe = F.pad(pe, (0, channels - pe.shape[-1]))
    elif pe.shape[-1] > channels:
        pe = pe[..., :channels]
    return pe.permute(2, 0, 1).unsqueeze(0).contiguous()


class MultiLevelTransformerDecoder(nn.Module):
    """Transformer decoder that fuses multiple MedSAM2/SAM2 feature levels.

    This is the multi-level transformer counterpart of the FPN decoder. It uses
    the same backbone feature levels as FPN, projects each level to decoder_dim,
    adds 2D positional encodings plus a learned level embedding, concatenates all
    spatial locations as transformer tokens, and finally reshapes the transformed
    tokens back to feature maps for multi-scale fusion and mask prediction.

    To keep memory bounded, the projected feature maps can be downsampled before
    tokenization when the total number of tokens exceeds max_tokens. The final
    mask logits are still resized back to the input resolution by the outer model.
    """

    def __init__(
        self,
        decoder_dim: int = 256,
        num_levels: int = 3,
        depth: int = 2,
        num_heads: int = 8,
        mlp_ratio: float = 4.0,
        refine_blocks: int = 1,
        max_tokens: int = 1024,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.decoder_dim = int(decoder_dim)
        self.num_levels = max(1, int(num_levels))
        self.max_tokens = max(1, int(max_tokens))
        self.lateral_convs = nn.ModuleList([nn.LazyConv2d(self.decoder_dim, kernel_size=1) for _ in range(self.num_levels)])
        self.level_embed = nn.Parameter(torch.zeros(self.num_levels, 1, self.decoder_dim))

        depth = max(1, int(depth))
        ff_dim = int(round(float(mlp_ratio) * self.decoder_dim))
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.decoder_dim,
            nhead=int(num_heads),
            dim_feedforward=ff_dim,
            dropout=float(dropout),
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=depth)
        self.norm = nn.LayerNorm(self.decoder_dim)

        refine_blocks = max(0, int(refine_blocks))
        refine_layers: List[nn.Module] = []
        for _ in range(refine_blocks):
            refine_layers.append(ConvGNAct(self.decoder_dim, self.decoder_dim, dropout=dropout))
        self.refine = nn.Sequential(*refine_layers) if refine_layers else nn.Identity()
        self.out_conv = nn.Conv2d(self.decoder_dim, 1, kernel_size=1)

    def _select_features(self, features: Sequence[torch.Tensor]) -> List[torch.Tensor]:
        if not features:
            raise ValueError("MultiLevelTransformerDecoder received an empty feature list")
        feats = sorted(list(features), key=lambda t: int(t.shape[-2]) * int(t.shape[-1]))
        if len(feats) < self.num_levels:
            raise ValueError(
                f"Transformer decoder requested {self.num_levels} feature levels, but SAM2 returned only {len(feats)}. "
                "Lower --fpn-levels/--transformer-levels or use --decoder-type deep/simple."
            )
        if len(feats) > self.num_levels:
            feats = feats[-self.num_levels:]
        return feats

    def _maybe_downsample_projected_features(self, feats: List[torch.Tensor]) -> List[torch.Tensor]:
        total_tokens = sum(int(f.shape[-2]) * int(f.shape[-1]) for f in feats)
        if total_tokens <= self.max_tokens:
            return feats
        scale = math.sqrt(float(self.max_tokens) / float(max(total_tokens, 1)))
        out: List[torch.Tensor] = []
        for f in feats:
            h, w = int(f.shape[-2]), int(f.shape[-1])
            new_h = max(1, int(round(h * scale)))
            new_w = max(1, int(round(w * scale)))
            if (new_h, new_w) == (h, w):
                out.append(f)
            else:
                out.append(F.interpolate(f, size=(new_h, new_w), mode="bilinear", align_corners=False))
        return out

    def forward(self, features: Sequence[torch.Tensor]) -> torch.Tensor:
        feats = self._select_features(features)
        projected = [proj(feat) for proj, feat in zip(self.lateral_convs, feats)]
        projected = self._maybe_downsample_projected_features(projected)

        tokens: List[torch.Tensor] = []
        spatial_shapes: List[Tuple[int, int]] = []
        for level, feat in enumerate(projected):
            b, c, h, w = feat.shape
            spatial_shapes.append((int(h), int(w)))
            pos = build_2d_sincos_position_embedding(c, h, w, feat.device, feat.dtype)
            feat = feat + pos + self.level_embed[level].view(1, c, 1, 1).to(device=feat.device, dtype=feat.dtype)
            tokens.append(feat.flatten(2).transpose(1, 2))  # [B, H*W, C]

        x = torch.cat(tokens, dim=1)
        x = self.transformer(x)
        x = self.norm(x)

        # Split transformer tokens back into feature maps and fuse all levels at
        # the highest transformed spatial resolution.
        maps: List[torch.Tensor] = []
        start = 0
        for h, w in spatial_shapes:
            n = int(h) * int(w)
            part = x[:, start:start + n, :]
            start += n
            maps.append(part.transpose(1, 2).reshape(x.shape[0], self.decoder_dim, h, w).contiguous())

        target_hw = max((m.shape[-2:] for m in maps), key=lambda s: int(s[0]) * int(s[1]))
        fused = torch.zeros(
            (maps[0].shape[0], self.decoder_dim, int(target_hw[0]), int(target_hw[1])),
            device=maps[0].device,
            dtype=maps[0].dtype,
        )
        for m in maps:
            if m.shape[-2:] != target_hw:
                m = F.interpolate(m, size=target_hw, mode="bilinear", align_corners=False)
            fused = fused + m
        fused = fused / float(len(maps))
        fused = self.refine(fused)
        return self.out_conv(fused)


class MedSAM2EncoderSegmentationModel(nn.Module):
    """Promptless segmentation model: MedSAM2/SAM2 image encoder + selectable decoder/head."""

    def __init__(
        self,
        sam2_model: nn.Module,
        decoder_dim: int = 256,
        freeze_encoder: bool = True,
        decoder_type: str = "simple",
        decoder_depth: int = 4,
        fpn_levels: int = 3,
        decoder_dropout: float = 0.0,
        transformer_heads: int = 8,
        transformer_mlp_ratio: float = 4.0,
        transformer_refine_blocks: int = 1,
        transformer_max_tokens: int = 1024,
    ):
        super().__init__()
        self.sam2 = sam2_model
        self.freeze_encoder = freeze_encoder
        self.decoder_type = str(decoder_type)
        if freeze_encoder:
            for p in self.sam2.parameters():
                p.requires_grad = False

        if self.decoder_type == "simple":
            self.decoder = SimpleDecoder(decoder_dim=decoder_dim, dropout=decoder_dropout)
        elif self.decoder_type == "deep":
            self.decoder = DeepDecoder(decoder_dim=decoder_dim, depth=decoder_depth, dropout=decoder_dropout)
        elif self.decoder_type == "fpn":
            self.decoder = FPNDecoder(
                decoder_dim=decoder_dim,
                num_levels=fpn_levels,
                smooth_blocks=decoder_depth,
                dropout=decoder_dropout,
            )
        elif self.decoder_type == "transformer":
            self.decoder = MultiLevelTransformerDecoder(
                decoder_dim=decoder_dim,
                num_levels=fpn_levels,
                depth=decoder_depth,
                num_heads=transformer_heads,
                mlp_ratio=transformer_mlp_ratio,
                refine_blocks=transformer_refine_blocks,
                max_tokens=transformer_max_tokens,
                dropout=decoder_dropout,
            )
        else:
            raise ValueError(f"Unknown decoder_type={decoder_type!r}. Use simple, deep, fpn, or transformer.")

    def _forward_backbone(self, x: torch.Tensor):
        if self.freeze_encoder:
            with torch.no_grad():
                backbone_out = self.sam2.forward_image(x)
                _, vision_feats, _, feat_sizes = self.sam2._prepare_backbone_features(backbone_out)
        else:
            backbone_out = self.sam2.forward_image(x)
            _, vision_feats, _, feat_sizes = self.sam2._prepare_backbone_features(backbone_out)
        return vision_feats, feat_sizes

    @staticmethod
    def _reshape_sam2_feature(feat: torch.Tensor, feat_size: Sequence[int], batch_size: int) -> torch.Tensor:
        h, w = int(feat_size[0]), int(feat_size[1])
        # SAM2 commonly returns flattened features as [H*W, B, C].
        if feat.ndim == 3 and feat.shape[1] == batch_size:
            c = int(feat.shape[-1])
            return feat.permute(1, 2, 0).reshape(batch_size, c, h, w).contiguous()
        # Safety fallback for already batched features.
        if feat.ndim == 4 and feat.shape[0] == batch_size:
            return feat.contiguous()
        raise RuntimeError(f"Unexpected SAM2 feature shape {tuple(feat.shape)} for batch_size={batch_size}, feat_size={feat_size}")

    def extract_sam2_feature(self, x: torch.Tensor) -> torch.Tensor:
        vision_feats, feat_sizes = self._forward_backbone(x)
        feat = vision_feats[-1]
        return self._reshape_sam2_feature(feat, feat_sizes[-1], x.shape[0])

    def extract_sam2_features(self, x: torch.Tensor) -> List[torch.Tensor]:
        vision_feats, feat_sizes = self._forward_backbone(x)
        feats = [self._reshape_sam2_feature(f, s, x.shape[0]) for f, s in zip(vision_feats, feat_sizes)]
        return sorted(feats, key=lambda t: int(t.shape[-2]) * int(t.shape[-1]))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        input_hw = x.shape[-2:]
        if self.decoder_type in {"fpn", "transformer"}:
            feats = self.extract_sam2_features(x)
            logits_low = self.decoder(feats)
        else:
            feat = self.extract_sam2_feature(x)
            logits_low = self.decoder(feat)
        logits = F.interpolate(logits_low, size=input_hw, mode="bilinear", align_corners=False)
        return logits


def load_builder(builder: str):
    """Load model builder from a module path like 'sam2.build_sam:build_sam2'."""
    if ":" not in builder:
        raise ValueError("--builder must have format module.submodule:function, e.g. sam2.build_sam:build_sam2")
    module_name, func_name = builder.split(":", 1)
    module = importlib.import_module(module_name)
    try:
        return getattr(module, func_name)
    except AttributeError as exc:
        raise AttributeError(f"Builder function {func_name!r} was not found in module {module_name!r}") from exc


def build_model(args: argparse.Namespace, device: torch.device) -> MedSAM2EncoderSegmentationModel:
    builder_fn = load_builder(args.builder)
    sam2 = builder_fn(args.model_cfg, args.checkpoint, device=device)
    model = MedSAM2EncoderSegmentationModel(
        sam2_model=sam2,
        decoder_dim=args.decoder_dim,
        freeze_encoder=not args.unfreeze_encoder,
        decoder_type=args.decoder_type,
        decoder_depth=args.decoder_depth,
        fpn_levels=args.fpn_levels,
        decoder_dropout=args.decoder_dropout,
        transformer_heads=args.transformer_heads,
        transformer_mlp_ratio=args.transformer_mlp_ratio,
        transformer_refine_blocks=args.transformer_refine_blocks,
        transformer_max_tokens=args.transformer_max_tokens,
    ).to(device)
    return model


def materialize_lazy_modules(model: nn.Module, sample_image: torch.Tensor, device: torch.device, amp_dtype: str) -> None:
    was_training = model.training
    model.eval()
    with torch.no_grad(), safe_autocast(device, amp_dtype):
        _ = model(sample_image[None].to(device))
    model.train(was_training)


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

    print(f"Training positive-slice segmentation. Output: {out_dir}")
    print(f"Decoder: {args.decoder_type}; decoder_dim={args.decoder_dim}; decoder_depth={args.decoder_depth}; fpn_levels={args.fpn_levels}; dropout={args.decoder_dropout}")
    if args.decoder_type == "transformer":
        print(f"Transformer decoder: heads={args.transformer_heads}; mlp_ratio={args.transformer_mlp_ratio}; refine_blocks={args.transformer_refine_blocks}; max_tokens={args.transformer_max_tokens}")
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
        description="Fully supervised MedSAM2/MedSAM2/SAM2-encoder positive-slice 2.5D LUNA nodule segmentation",
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

    # MedSAM2/MedSAM2/SAM2 model
    p.add_argument("--model-cfg", required=True)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--builder", default="sam2.build_sam:build_sam2", help="Model builder import path. Default works for MedSAM2/SAM2 repos exposing build_sam2 via sam2.build_sam.")
    p.add_argument("--decoder-type", choices=["simple", "deep", "fpn", "transformer"], default="simple", help="Segmentation decoder/head. simple keeps the original head, deep uses more conv blocks on the final feature, fpn fuses multiple MedSAM2/SAM2 features top-down, and transformer uses multi-level feature tokens with self-attention.")
    p.add_argument("--decoder-dim", type=int, default=256)
    p.add_argument("--decoder-depth", type=int, default=4, help="For deep: number of ConvGNAct blocks. For fpn: number of smoothing ConvGNAct blocks per FPN level.")
    p.add_argument("--fpn-levels", type=int, default=3, help="Number of MedSAM2/SAM2 backbone feature levels to fuse when --decoder-type fpn.")
    p.add_argument("--decoder-dropout", type=float, default=0.0, help="Optional dropout probability inside decoder blocks/transformer layers.")
    p.add_argument("--transformer-heads", type=int, default=8, help="Number of self-attention heads for --decoder-type transformer.")
    p.add_argument("--transformer-mlp-ratio", type=float, default=4.0, help="Transformer feed-forward width as a multiple of --decoder-dim.")
    p.add_argument("--transformer-refine-blocks", type=int, default=1, help="Number of ConvGNAct refinement blocks after multi-level transformer fusion.")
    p.add_argument("--transformer-max-tokens", type=int, default=1024, help="Maximum total spatial tokens used by the multi-level transformer decoder; feature levels are downsampled if needed.")
    p.add_argument("--unfreeze-encoder", action="store_true", help="Fine-tune the SAM2 image encoder as well as the segmentation head.")

    # Image/slice setup
    p.add_argument("--use-triplet-channels", action=argparse.BooleanOptionalAction, default=True, help="Use z-1,z,z+1 as 3 input channels. Disable for single-slice copied to RGB.")
    p.add_argument("--hu-min", type=float, default=-1000.0)
    p.add_argument("--hu-max", type=float, default=400.0)
    p.add_argument("--image-size", type=int, default=None, help="Optional square resize for training/inference. Predictions are resized back to original H,W before saving volumes.")
    p.add_argument("--min-slice-mask-pixels", type=int, default=1, help="A slice is selected if GT nodule mask pixels >= this value.")
    p.add_argument("--min-nodule-voxels", type=int, default=1, help="Minimum connected-component size for nodule-wise metrics.")

    # Training
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--eval-batch-size", type=int, default=8)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--bce-weight", type=float, default=0.5)
    p.add_argument("--dice-weight", type=float, default=0.5)
    p.add_argument("--threshold", type=float, default=0.5)
    p.add_argument("--patience", type=int, default=10)
    p.add_argument("--min-delta", type=float, default=1e-5)
    p.add_argument("--cache-cases", action="store_true")
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--device", default="auto")
    p.add_argument("--amp-dtype", choices=["bf16", "fp16", "none"], default="bf16")
    p.add_argument("--allow-tf32", action=argparse.BooleanOptionalAction, default=True)

    # Augmentation; applied only to the training split, after optional resize.
    p.add_argument("--augment", action=argparse.BooleanOptionalAction, default=False, help="Enable conservative 2D augmentations for training slices only.")
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
    if args.decoder_depth < 1:
        raise ValueError("--decoder-depth must be >= 1")
    if args.fpn_levels < 1:
        raise ValueError("--fpn-levels must be >= 1")
    if not (0.0 <= args.decoder_dropout < 1.0):
        raise ValueError("--decoder-dropout must be in [0, 1)")
    if args.transformer_heads < 1:
        raise ValueError("--transformer-heads must be >= 1")
    if args.decoder_dim % args.transformer_heads != 0:
        raise ValueError("--decoder-dim must be divisible by --transformer-heads for --decoder-type transformer")
    if args.transformer_mlp_ratio <= 0:
        raise ValueError("--transformer-mlp-ratio must be > 0")
    if args.transformer_refine_blocks < 0:
        raise ValueError("--transformer-refine-blocks must be >= 0")
    if args.transformer_max_tokens < 1:
        raise ValueError("--transformer-max-tokens must be >= 1")
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
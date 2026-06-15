import os
from os.path import join
import math
import numpy as np
import pandas as pd
import SimpleITK as sitk
import torch
from PIL import Image
from scipy import ndimage

from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor
torch.set_float32_matmul_precision("high")
from glob import glob
from tqdm import tqdm
import re
import matplotlib.pyplot as plt
from collections import OrderedDict

from PIL import Image
import torch
import torch.multiprocessing as mp
import SimpleITK as sitk
from skimage import measure, morphology
import random
import cv2

import sys
print(sys.executable)


def normalize_to_uint8(img_2d):
    img_2d = img_2d.astype(np.float32)
    img_2d = img_2d - img_2d.min()
    max_val = img_2d.max()
    if max_val > 0:
        img_2d = img_2d / max_val
    return (img_2d * 255).astype(np.uint8)

def dice_score(mask1, mask2, smooth=1e-6):
    mask1 = mask1.astype(bool)
    mask2 = mask2.astype(bool)

    intersection = np.logical_and(mask1, mask2).sum()
    total = mask1.sum() + mask2.sum()

    return float((2 * intersection + smooth) / (total + smooth))


def preprocess_ct(image_data, window_level=-750, window_width=1500):
    lower_bound = window_level - window_width / 2
    upper_bound = window_level + window_width / 2
    image_data_pre = np.clip(image_data, lower_bound, upper_bound)

    denom = np.max(image_data_pre) - np.min(image_data_pre)
    if denom > 0:
        image_data_pre = (image_data_pre - np.min(image_data_pre)) / denom * 255.0
    else:
        image_data_pre = np.zeros_like(image_data_pre)

    return image_data_pre.astype(np.uint8)


def resize_grayscale_to_rgb_and_resize(array, image_size):
    """
    array: (D, H, W) uint8
    returns: (D, 3, image_size, image_size)
    """
    d, h, w = array.shape
    resized_array = np.zeros((d, 3, image_size, image_size), dtype=np.float32)

    for i in range(d):
        img_pil = Image.fromarray(array[i].astype(np.uint8))
        img_rgb = img_pil.convert("RGB")
        img_resized = img_rgb.resize((image_size, image_size))
        img_array = np.array(img_resized).transpose(2, 0, 1)  # (3, image_size, image_size)
        resized_array[i] = img_array

    return resized_array


def normalize_for_medsam2(img_resized):
    """
    img_resized: (D, 3, H, W), values in [0,255]
    returns torch tensor on CUDA
    """
    img_resized = img_resized / 255.0
    img_resized = torch.from_numpy(img_resized).float().cuda()

    img_mean = torch.tensor((0.485, 0.456, 0.406), dtype=torch.float32, device="cuda")[:, None, None]
    img_std = torch.tensor((0.229, 0.224, 0.225), dtype=torch.float32, device="cuda")[:, None, None]

    img_resized = (img_resized - img_mean) / img_std
    return img_resized


def get_3d_connected_components(mask_zyx, connectivity=2):
    """
    mask_zyx: (Z, Y, X) binary
    """
    structure = ndimage.generate_binary_structure(rank=3, connectivity=connectivity)
    labeled, num = ndimage.label(mask_zyx.astype(np.uint8), structure=structure)
    return labeled, num


def get_interior_center_3d(component_mask):
    """
    Returns (z, y, x) as the 3D distance-transform peak.
    Guaranteed to lie inside the component.
    """
    dist = ndimage.distance_transform_edt(component_mask)
    z, y, x = np.unravel_index(np.argmax(dist), dist.shape)
    return int(z), int(y), int(x)


def get_bbox_from_slice_mask(mask_2d):
    ys, xs = np.where(mask_2d > 0)
    if len(xs) == 0:
        return None
    x0, x1 = xs.min(), xs.max()
    y0, y1 = ys.min(), ys.max()
    return np.array([x0, y0, x1, y1], dtype=np.float32)


def pad_box_xyxy(box, H, W, pad=5):
    x0, y0, x1, y1 = box
    x0 = max(0, x0 - pad)
    y0 = max(0, y0 - pad)
    x1 = min(W - 1, x1 + pad)
    y1 = min(H - 1, y1 + pad)
    return np.array([x0, y0, x1, y1], dtype=np.float32)


def extract_nodule_prompts_from_mask_3d(gt_volume, pad_box=5):
    """
    Returns one prompt dict per 3D connected nodule.

    Each dict contains:
      - obj_id
      - center_zyx
      - point_xy
      - box_xyxy
      - component_mask
    """
    labeled, num = get_3d_connected_components(gt_volume)

    prompts = []
    obj_id = 1

    Z, H, W = gt_volume.shape

    for k in range(1, num + 1):
        comp = (labeled == k)
        if comp.sum() == 0:
            continue

        z, y, x = get_interior_center_3d(comp)

        center_slice_mask = comp[z].astype(np.uint8)
        box_xyxy = get_bbox_from_slice_mask(center_slice_mask)
        if box_xyxy is None:
            continue

        box_xyxy = pad_box_xyxy(box_xyxy, H=H, W=W, pad=pad_box)

        prompts.append({
            "obj_id": obj_id,
            "center_zyx": (z, y, x),
            "frame_idx": z,
            "point_xy": np.array([[float(x), float(y)]], dtype=np.float32),  # (1,2)
            "point_labels": np.array([1], dtype=np.int32),                    # (1,)
            "box_xyxy": box_xyxy,                                             # (4,)
            "component_mask": comp.astype(np.uint8),
        })
        obj_id += 1

    return prompts


@torch.inference_mode()
def run_medsam2_single_object(
    predictor,
    img_resized_torch,   # (D, 3, imsize, imsize)
    video_height,
    video_width,
    frame_idx,
    obj_id,
    prompt_mode="point",   # "point", "box", "point+box"
    point_xy=None,         # (1,2)
    point_labels=None,     # (1,)
    box_xyxy=None,         # (4,)
):
    """
    Returns a binary 3D mask for one object: (Z, H, W)
    """
    Z = img_resized_torch.shape[0]
    segs_3d = np.zeros((Z, video_height, video_width), dtype=np.uint8)

    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        inference_state = predictor.init_state(img_resized_torch, video_height, video_width)

        kwargs = dict(
            inference_state=inference_state,
            frame_idx=frame_idx,
            obj_id=obj_id,
        )

        if prompt_mode == "point":
            kwargs["points"] = point_xy
            kwargs["labels"] = point_labels

        elif prompt_mode == "box":
            kwargs["box"] = box_xyxy

        elif prompt_mode == "point+box":
            kwargs["points"] = point_xy
            kwargs["labels"] = point_labels
            kwargs["box"] = box_xyxy

        else:
            raise ValueError(f"Unknown prompt_mode: {prompt_mode}")

        _, out_obj_ids, out_mask_logits = predictor.add_new_points_or_box(**kwargs)

        mask_prompt = (out_mask_logits[0] > 0.0).squeeze(0).cpu().numpy().astype(np.uint8)

        # seed mask at center slice
        frame_idx_out, object_ids, masks = predictor.add_new_mask(
            inference_state, frame_idx=frame_idx, obj_id=obj_id, mask=mask_prompt
        )
        segs_3d[frame_idx, ((masks[0] > 0.0).cpu().numpy())[0]] = 1

        # forward
        for out_frame_idx, out_obj_ids, out_mask_logits in predictor.propagate_in_video(
            inference_state, start_frame_idx=frame_idx, reverse=False
        ):
            segs_3d[out_frame_idx, (out_mask_logits[0] > 0.0).cpu().numpy()[0]] = 1

        predictor.reset_state(inference_state)

        # backward
        inference_state = predictor.init_state(img_resized_torch, video_height, video_width)
        predictor.add_new_mask(inference_state, frame_idx=frame_idx, obj_id=obj_id, mask=mask_prompt)

        for out_frame_idx, out_obj_ids, out_mask_logits in predictor.propagate_in_video(
            inference_state, start_frame_idx=frame_idx, reverse=True
        ):
            segs_3d[out_frame_idx, (out_mask_logits[0] > 0.0).cpu().numpy()[0]] = 1

        predictor.reset_state(inference_state)

    return segs_3d


@torch.inference_mode()
def analyze_CT_volume_medsam2_from_masks(
    mhd_file,
    path_volumes,
    path_masks,
    df_ids_link,
    predictor,
    list_number_masks,
    list_maks_nodules,
    imsize=512,
    prompt_modes=("point", "box", "point+box"),
    pad_box=5,
    window_level=-750,
    window_width=1500,
):
    """
    Whole-volume MedSAM2 inference using mask-derived 3D centers.

    Returns
    -------
    dict or None
    """
    path_image = os.path.join(path_volumes, mhd_file + ".mhd")
    if not os.path.isfile(path_image):
        print("Missing image:", path_image)
        return None

    sub_df_links = df_ids_link[df_ids_link["SeriesID"] == mhd_file]
    if len(sub_df_links) == 0:
        print("No mask link found for:", mhd_file)
        return None

    mask_id_num = sub_df_links["CID"].tolist()[0]
    if mask_id_num not in list_number_masks:
        print("Mask id not found:", mask_id_num, mhd_file)
        return None

    idx = list_number_masks.index(mask_id_num)
    path_mask = os.path.join(path_masks, list_maks_nodules[idx])
    if not os.path.isfile(path_mask):
        print("Missing mask:", path_mask)
        return None

    sitk_img = sitk.ReadImage(path_image)
    img_3d = sitk.GetArrayFromImage(sitk_img)              # (Z, Y, X)
    img_3d = preprocess_ct(img_3d, window_level=window_level, window_width=window_width)

    mask_img = sitk.ReadImage(path_mask)
    gt_volume = sitk.GetArrayFromImage(mask_img)
    gt_volume = (gt_volume >= 0.5).astype(np.uint8)

    prompts = extract_nodule_prompts_from_mask_3d(gt_volume, pad_box=pad_box)
    if len(prompts) == 0:
        print("No nodules found in mask:", mhd_file)
        return None

    video_height, video_width = gt_volume.shape[1], gt_volume.shape[2]

    if video_height != imsize or video_width != imsize:
        img_resized = resize_grayscale_to_rgb_and_resize(img_3d, imsize)
    else:
        img_resized = img_3d[:, None].repeat(3, axis=1).astype(np.float32)

    img_resized_torch = normalize_for_medsam2(img_resized)

    # one full-volume prediction per prompt mode
    pred_volumes = {
        mode: np.zeros_like(gt_volume, dtype=np.uint8)
        for mode in prompt_modes
    }

    for p in prompts:
        frame_idx = p["frame_idx"]
        point_xy = p["point_xy"]
        point_labels = p["point_labels"]
        box_xyxy = p["box_xyxy"]

        for mode in prompt_modes:
            pred_obj = run_medsam2_single_object(
                predictor=predictor,
                img_resized_torch=img_resized_torch,
                video_height=video_height,
                video_width=video_width,
                frame_idx=frame_idx,
                obj_id=1,  # fresh state every time, so obj_id can stay 1
                prompt_mode=mode,
                point_xy=point_xy,
                point_labels=point_labels,
                box_xyxy=box_xyxy,
            )

            pred_volumes[mode] = np.logical_or(pred_volumes[mode], pred_obj).astype(np.uint8)

    dsc = {
        mode: dice_score(gt_volume, pred_volumes[mode])
        for mode in prompt_modes
    }

    return {
        "seriesuid": mhd_file,
        "gt_volume": gt_volume,
        "pred_volumes": pred_volumes,
        "dice": dsc,
        "prompts": prompts,
    }


def build_slice_prompt_dict(mask_3d):
    """
    Group prompts by z-slice.

    Parameters
    ----------
    mask_3d : np.ndarray
        Binary mask with shape [Z, Y, X]

    Returns
    -------
    slice_dicts : list of dict
        One dict per z-slice containing at least one blob.

        Each dict contains:
        - z
        - mask_slice
        - blobs
        - point_coords: np.ndarray of shape (N, 2)
        - point_labels: np.ndarray of shape (N,)
        - boxes: np.ndarray of shape (N, 4)
    """
    mask_3d = (mask_3d > 0).astype(np.uint8)
    slice_dicts = []

    for z in range(mask_3d.shape[0]):
        mask_slice = mask_3d[z]

        if not np.any(mask_slice > 0):
            continue

        blobs = extract_blobs_from_slice(mask_slice)

        if len(blobs) == 0:
            continue

        point_coords = np.array([blob["center"] for blob in blobs], dtype=np.float32)   # (N, 2)
        point_labels = np.ones(len(blobs), dtype=np.int32)                              # (N,)
        boxes = np.array([blob["bbox"] for blob in blobs], dtype=np.float32)            # (N, 4)

        slice_dicts.append({
            "z": z,
            "mask_slice": mask_slice,
            "blobs": blobs,
            "point_coords": point_coords,
            "point_labels": point_labels,
            "boxes": boxes,
        })

    return slice_dicts

    
def build_medsam2_input_slice(image_array, z, use_triplet_channels=False):
    """
    image_array: [Z, Y, X]
    returns [H, W, 3] uint8
    """
    if use_triplet_channels:
        z_prev = max(z - 1, 0)
        z_next = min(z + 1, image_array.shape[0] - 1)

        ch0 = normalize_to_uint8(image_array[z_prev])
        ch1 = normalize_to_uint8(image_array[z])
        ch2 = normalize_to_uint8(image_array[z_next])

        img_3ch = np.stack([ch0, ch1, ch2], axis=-1)
    else:
        img = normalize_to_uint8(image_array[z])
        img_3ch = np.stack([img, img, img], axis=-1)

    return img_3ch


def get_contours_and_centroids(mask_2d):
    """
    Extract every contour/blob from a binary 2D mask and compute its centroid.

    Parameters
    ----------
    mask_2d : np.ndarray
        Binary mask [H, W]

    Returns
    -------
    blobs : list of dict
        Each dict contains:
        - contour
        - centroid : [x, y]
        - bbox : [x_min, y_min, x_max, y_max]
        - component_mask : binary mask for this contour/blob
    """
    mask_bin = (mask_2d > 0).astype(np.uint8)

    contours, _ = cv2.findContours(mask_bin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    blobs = []

    for contour in contours:
        if contour.shape[0] == 0:
            continue

        M = cv2.moments(contour)

        # Handle degenerate contour
        if M["m00"] != 0:
            cx = M["m10"] / M["m00"]
            cy = M["m01"] / M["m00"]
        else:
            # fallback: mean of contour points
            pts = contour[:, 0, :]
            cx = np.mean(pts[:, 0])
            cy = np.mean(pts[:, 1])

        x, y, w, h = cv2.boundingRect(contour)

        component_mask = np.zeros_like(mask_bin, dtype=np.uint8)
        cv2.drawContours(component_mask, [contour], contourIdx=-1, color=1, thickness=-1)

        blobs.append({
            "contour": contour,
            "centroid": [float(cx), float(cy)],
            "bbox": [int(x), int(y), int(x + w - 1), int(y + h - 1)],
            "component_mask": component_mask
        })

    return blobs

def get_all_blobs_from_mask_3d(mask_3d):
    """
    Go through all slices in a 3D mask [Z, Y, X].
    For every non-empty slice, find all contours/blobs.

    Returns
    -------
    all_blobs : list of dict
        Each dict contains:
        - z
        - centroid
        - bbox
        - component_mask
        - contour
    """
    mask_3d = (mask_3d > 0).astype(np.uint8)
    all_blobs = []

    for z in range(mask_3d.shape[0]):
        mask_slice = mask_3d[z]

        if np.any(mask_slice > 0):
            blobs = get_contours_and_centroids(mask_slice)

            for blob in blobs:
                all_blobs.append({
                    "z": z,
                    "centroid": blob["centroid"],
                    "bbox": blob["bbox"],
                    "component_mask": blob["component_mask"],
                    "contour": blob["contour"]
                })

    return all_blobs


def get_interior_point(component_mask):
    """
    Return [x, y] point inside the component, good for SAM prompting.
    """
    component_mask = (component_mask > 0).astype(np.uint8)
    dist = cv2.distanceTransform(component_mask, cv2.DIST_L2, 5)
    y, x = np.unravel_index(np.argmax(dist), dist.shape)
    return [float(x), float(y)]


def extract_blobs_from_slice(mask_2d):
    """
    For one binary slice [H, W], extract all blobs/contours.

    Returns
    -------
    blobs : list of dict
        Each dict has:
        - center: [x, y]
        - bbox: [x_min, y_min, x_max, y_max]
        - component_mask: [H, W]
        - contour
    """
    mask_bin = (mask_2d > 0).astype(np.uint8)

    contours, _ = cv2.findContours(mask_bin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    blobs = []

    for contour in contours:
        if contour is None or len(contour) == 0:
            continue

        x, y, w, h = cv2.boundingRect(contour)

        component_mask = np.zeros_like(mask_bin, dtype=np.uint8)
        cv2.drawContours(component_mask, [contour], contourIdx=-1, color=1, thickness=-1)

        if component_mask.sum() == 0:
            continue

        center_xy = get_interior_point(component_mask)

        blobs.append({
            "center": center_xy,
            "bbox": [int(x), int(y), int(x + w - 1), int(y + h - 1)],
            "component_mask": component_mask,
            "contour": contour,
        })

    return blobs
    

def analyze_CT_volume_medsam2_image_single_output(
    mhd_file,
    path_volumes,
    path_masks,
    df_ids_link,
    predictor,   # image predictor, not video predictor
    list_number_masks,
    list_maks_nodules,
    use_triplet_channels=False,
):
    path_image = os.path.join(path_volumes, mhd_file + ".mhd")

    if not os.path.isfile(path_image):
        print("Missing image:", path_image)
        return None

    image = sitk.ReadImage(path_image)
    image_array = sitk.GetArrayFromImage(image)   # [Z, Y, X]

    sub_df_links = df_ids_link[df_ids_link["SeriesID"] == mhd_file]
    if len(sub_df_links) == 0:
        print("No mask link found for:", mhd_file)
        return None

    mask_id_num = sub_df_links["CID"].tolist()[0]
    if mask_id_num not in list_number_masks:
        print("Mask id not found:", mask_id_num, mhd_file)
        return None

    idx = list_number_masks.index(mask_id_num)
    path_mask = os.path.join(path_masks, list_maks_nodules[idx])

    if not os.path.isfile(path_mask):
        print("Missing mask:", path_mask)
        return None

    mask = sitk.ReadImage(path_mask)
    mask_array = sitk.GetArrayFromImage(mask)   # [Z, Y, X]
    gt_volume = (mask_array >= 0.5).astype(np.uint8)

    slice_prompt_dicts = build_slice_prompt_dict(gt_volume)
    if len(slice_prompt_dicts) == 0:
        print("No valid slices found for:", mhd_file)
        return None

    pred_point_vol = np.zeros_like(gt_volume, dtype=bool)
    pred_box_vol = np.zeros_like(gt_volume, dtype=bool)
    pred_combined_vol = np.zeros_like(gt_volume, dtype=bool)

    all_scores_points = []
    all_scores_boxes = []
    all_scores_combined = []

    for slice_info in slice_prompt_dicts:
        z = slice_info["z"]
        point_coords = slice_info["point_coords"]   # (N, 2)
        point_labels = slice_info["point_labels"]   # (N,)
        boxes = slice_info["boxes"]                 # (N, 4)
        blobs = slice_info["blobs"]

        img_3ch = build_medsam2_input_slice(
            image_array=image_array,
            z=z,
            use_triplet_channels=use_triplet_channels,
        )

        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            predictor.set_image(img_3ch)

            pred_point_slice = np.zeros_like(gt_volume[z], dtype=bool)
            pred_box_slice = np.zeros_like(gt_volume[z], dtype=bool)
            pred_combined_slice = np.zeros_like(gt_volume[z], dtype=bool)

            for i in range(len(blobs)):
                input_point = point_coords[i:i+1]   # (1, 2)
                input_label = point_labels[i:i+1]   # (1,)
                input_box = boxes[i:i+1]            # (1, 4)

                masks_points, scores_points, _ = predictor.predict(point_coords=input_point, point_labels=input_label, multimask_output=False,)
                masks_boxes, scores_boxes, _ = predictor.predict(point_coords=None, point_labels=None, box=input_box, multimask_output=False,)
                masks_combined, scores_combined, _ = predictor.predict(point_coords=input_point, point_labels=input_label, box=input_box, multimask_output=False,)

                pred_point_slice |= masks_points[0].astype(bool)
                pred_box_slice |= masks_boxes[0].astype(bool)
                pred_combined_slice |= masks_combined[0].astype(bool)

                all_scores_points.append(float(scores_points[0]))
                all_scores_boxes.append(float(scores_boxes[0]))
                all_scores_combined.append(float(scores_combined[0]))

            pred_point_vol[z] |= pred_point_slice
            pred_box_vol[z] |= pred_box_slice
            pred_combined_vol[z] |= pred_combined_slice

    pred_point_vol = pred_point_vol.astype(np.uint8)
    pred_box_vol = pred_box_vol.astype(np.uint8)
    pred_combined_vol = pred_combined_vol.astype(np.uint8)

    dsc_point_vol = dice_score(gt_volume, pred_point_vol)
    dsc_box_vol = dice_score(gt_volume, pred_box_vol)
    dsc_combined_vol = dice_score(gt_volume, pred_combined_vol)

    return (
        dsc_point_vol,
        dsc_box_vol,
        dsc_combined_vol,
        all_scores_points,
        all_scores_boxes,
        all_scores_combined,
        pred_point_vol,
        pred_box_vol,
        pred_combined_vol,
        gt_volume,
    )


def analyze_CT_volume_medsam2_image_multi_output(
    mhd_file,
    path_volumes,
    path_masks,
    df_ids_link,
    predictor,   # image predictor, not video predictor
    list_number_masks,
    list_maks_nodules,
    use_triplet_channels=False,
):
    path_image = os.path.join(path_volumes, mhd_file + ".mhd")

    if not os.path.isfile(path_image):
        print("Missing image:", path_image)
        return None

    image = sitk.ReadImage(path_image)
    image_array = sitk.GetArrayFromImage(image)   # [Z, Y, X]

    sub_df_links = df_ids_link[df_ids_link["SeriesID"] == mhd_file]
    if len(sub_df_links) == 0:
        print("No mask link found for:", mhd_file)
        return None

    mask_id_num = sub_df_links["CID"].tolist()[0]
    if mask_id_num not in list_number_masks:
        print("Mask id not found:", mask_id_num, mhd_file)
        return None

    idx = list_number_masks.index(mask_id_num)
    path_mask = os.path.join(path_masks, list_maks_nodules[idx])

    if not os.path.isfile(path_mask):
        print("Missing mask:", path_mask)
        return None

    mask = sitk.ReadImage(path_mask)
    mask_array = sitk.GetArrayFromImage(mask)   # [Z, Y, X]
    gt_volume = (mask_array >= 0.5).astype(np.uint8)

    slice_prompt_dicts = build_slice_prompt_dict(gt_volume)
    if len(slice_prompt_dicts) == 0:
        print("No valid slices found for:", mhd_file)
        return None

    pred_point_vol = [np.zeros_like(gt_volume, dtype=bool) for _ in range(3)]
    pred_box_vol = [np.zeros_like(gt_volume, dtype=bool) for _ in range(3)]
    pred_combined_vol = [np.zeros_like(gt_volume, dtype=bool) for _ in range(3)]

    scores_point_all = [[] for _ in range(3)]
    scores_box_all = [[] for _ in range(3)]
    scores_combined_all = [[] for _ in range(3)]

    for slice_info in slice_prompt_dicts:
        z = slice_info["z"]
        point_coords = slice_info["point_coords"]
        point_labels = slice_info["point_labels"]
        boxes = slice_info["boxes"]
        blobs = slice_info["blobs"]

        img_3ch = build_medsam2_input_slice(image_array=image_array,z=z,use_triplet_channels=use_triplet_channels,)

        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            predictor.set_image(img_3ch)

            pred_point_slice = [np.zeros_like(gt_volume[z], dtype=bool) for _ in range(3)]
            pred_box_slice = [np.zeros_like(gt_volume[z], dtype=bool) for _ in range(3)]
            pred_combined_slice = [np.zeros_like(gt_volume[z], dtype=bool) for _ in range(3)]

            for i in range(len(blobs)):
                input_point = point_coords[i:i+1]
                input_label = point_labels[i:i+1]
                input_box = boxes[i:i+1]

                masks_p, scores_p, _ = predictor.predict(point_coords=input_point, point_labels=input_label, multimask_output=True,)
                masks_b, scores_b, _ = predictor.predict(point_coords=None, point_labels=None, box=input_box, multimask_output=True,)
                masks_c, scores_c, _ = predictor.predict(point_coords=input_point, point_labels=input_label, box=input_box, multimask_output=True,)

                for ch in range(3):
                    pred_point_slice[ch] |= masks_p[ch].astype(bool)
                    pred_box_slice[ch] |= masks_b[ch].astype(bool)
                    pred_combined_slice[ch] |= masks_c[ch].astype(bool)

                    scores_point_all[ch].append(float(scores_p[ch]))
                    scores_box_all[ch].append(float(scores_b[ch]))
                    scores_combined_all[ch].append(float(scores_c[ch]))

            for ch in range(3):
                pred_point_vol[ch][z] |= pred_point_slice[ch]
                pred_box_vol[ch][z] |= pred_box_slice[ch]
                pred_combined_vol[ch][z] |= pred_combined_slice[ch]

    pred_point_vol = [v.astype(np.uint8) for v in pred_point_vol]
    pred_box_vol = [v.astype(np.uint8) for v in pred_box_vol]
    pred_combined_vol = [v.astype(np.uint8) for v in pred_combined_vol]

    dsc_point = [dice_score(gt_volume, pred_point_vol[ch]) for ch in range(3)]
    dsc_box = [dice_score(gt_volume, pred_box_vol[ch]) for ch in range(3)]
    dsc_combined = [dice_score(gt_volume, pred_combined_vol[ch]) for ch in range(3)]

    return (
        dsc_point,             # [ch0, ch1, ch2]
        dsc_box,               # [ch0, ch1, ch2]
        dsc_combined,          # [ch0, ch1, ch2]
        scores_point_all,
        scores_box_all,
        scores_combined_all,
        pred_point_vol,
        pred_box_vol,
        pred_combined_vol,
        gt_volume,
    )

def main():
    path_dataset = '/jorge/datasets/LUNA_dataset/'
    list_files_dataset = os.listdir(path_dataset)
    annoations_path = os.path.join(path_dataset, 'annotations.csv')
    df_annotations = pd.read_csv(annoations_path)
    
    unique_series_uids, repetitions_uids = np.unique(df_annotations['seriesuid'].tolist(), return_counts=True)
    print('Unique IDs with labels: ', len(unique_series_uids), 'Note: One ID could have more than one set of annotations (i.e. more than 1 nodule)')
    
    
    path_ct_volumes = os.path.join(path_dataset, 'CT_volumes')
    list_all_files = list()
    #sub_folders_ct_volues = os.listdir(path_ct_volumes)
    
    list_all_files = os.listdir(path_ct_volumes)
    
    annotations_name_list = df_annotations['seriesuid'].tolist()
    mhd_files = [f for f in list_all_files if f.endswith('.mhd')]
    only_name_files = [f.replace('.mhd', '') for f in mhd_files]
    print(len(only_name_files), 'CT Volumes in ', path_ct_volumes)
    
    # masks analysis 
    path_masks = '/datasets/LUNA_dataset/masks_nodules/nifti_data/'
    list_files_path = os.listdir(path_masks)
    list_maks_nodules = [f for f in list_files_path if 'mask' in f and 'contour' in f and 'circle' not in f and 'nodule' in f]
    print(len(list_maks_nodules), 'Masks in ', path_masks)
    list_number_masks = [int(s.split('_')[0]) for s in list_maks_nodules]
    
    path_annotaions_links = '/datasets/LUNA_dataset/LUNA16_metadata_split_offical.csv'
    df_ids_link = pd.read_csv(path_annotaions_links)
    
    # run this in case you want to get the cases in which there are more than 1 nodule per CT Volume 
    
    dup_rows = df_annotations[df_annotations["seriesuid"].duplicated(keep=False)]["seriesuid"].tolist()

    # select the device for computation
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"using device: {device}")
    
    if device.type == "cuda":
        # use bfloat16 for the entire notebook
        torch.autocast("cuda", dtype=torch.bfloat16).__enter__()
        # turn on tfloat32 for Ampere GPUs (https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices)
        if torch.cuda.get_device_properties(0).major >= 8:
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
    elif device.type == "mps":
        print(
            "\nSupport for MPS devices is preliminary. SAM 2 is trained with CUDA and might "
            "give numerically different outputs and sometimes degraded performance on MPS. "
            "See e.g. https://github.com/pytorch/pytorch/issues/84936 for a discussion."
        )
        
    
    # Set paths to the model checkpoint, image directory, and dataset information of the bounding boxes
    
    
    checkpoint = '/repos/MedSAM2/checkpoints/MedSAM2_latest.pt'
    print(os.path.isfile(checkpoint))
    model_cfg = "configs/sam2.1_hiera_t512.yaml"
    print(os.path.isfile(model_cfg))
    
    
    sam2_model = build_sam2(model_cfg, checkpoint, device="cuda")
    predictor = SAM2ImagePredictor(sam2_model)

    results = []
    path_volumes = path_ct_volumes
    print(len(only_name_files), 'unique file names')
    for mhd_file in tqdm(only_name_files):
        try:
            result = analyze_CT_volume_medsam2_image_single_output(
                                mhd_file=mhd_file,
                                path_volumes=path_ct_volumes,
                                path_masks=path_masks,
                                df_ids_link=df_ids_link,
                                predictor=predictor,
                                list_number_masks=list_number_masks,
                                list_maks_nodules=list_maks_nodules,
                                use_triplet_channels=False,)

    
            row = {
                    "VolumeID": mhd_file,
                    "DSC (video points)": result[0],
                    "DSC (video boxes)": result[1],
                    "DSC (video combines)":result[2],
                }
    
            results.append(row)

        except Exception as e:
            results.append({
                "VolumeID": mhd_file,
                "DSC (video points)": None,
                "DSC (video boxes)": None,
                "DSC (video combines)": None,
                "error": str(e),
            })
    
    df_results = pd.DataFrame(results)

    path_save_results = os.path.join(path_dataset, "predictions_MedSAM2_image_single_ch_output.csv")
    df_results.to_csv(path_save_results, index=False)
    print(f"results saved at: {path_save_results}")



if __name__ == "__main__":
    main()

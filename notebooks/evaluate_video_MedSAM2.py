import os
from os.path import join
import math
import numpy as np
import pandas as pd
import SimpleITK as sitk
import torch
from PIL import Image
from scipy import ndimage
from sam2.build_sam import build_sam2_video_predictor_npz
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

import sys
print(sys.executable)


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

def main():
    path_dataset = '/datasets/LUNA_dataset/'
    list_files_dataset = os.listdir(path_dataset)
    annoations_path = os.path.join(path_dataset, 'annotations.csv')
    df_annotations = pd.read_csv(annoations_path)
    
    unique_series_uids, repetitions_uids = np.unique(df_annotations['seriesuid'].tolist(), return_counts=True)
    print('Unique IDs with labels: ', len(unique_series_uids), 'Note: One ID could have more than one set of annotations (i.e. more than 1 nodule)')
    
    
    path_ct_volumes = os.path.join(path_dataset, 'CT_volumes')
    list_all_files = list()
    #sub_folders_ct_volues = os.listdir(path_ct_volumes)
    
    list_all_files += os.listdir(path_ct_volumes)
    
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
    
    path_annotaions_links = '/mimer/NOBACKUP/groups/naiss2025-6-383/jorge/datasets/LUNA_dataset/LUNA16_metadata_split_offical.csv'
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
    
    predictor = build_sam2_video_predictor_npz(model_cfg, checkpoint)

    results = []
    path_volumes = path_ct_volumes
    for mhd_file in tqdm(only_name_files):
        try:
            result = analyze_CT_volume_medsam2_from_masks(
                mhd_file=mhd_file,
                path_volumes=path_volumes,
                path_masks=path_masks,
                df_ids_link=df_ids_link,
                predictor=predictor,
                list_number_masks=list_number_masks,
                list_maks_nodules=list_maks_nodules,
                imsize=512,
                prompt_modes=("point", "box", "point+box"),
                pad_box=5,)
    
            row = {
                    "VolumeID": mhd_file,
                    "DSC (video points)": result["dice"]["point"],
                    "DSC (video boxes)": result["dice"]["box"],
                    "DSC (video combines)": result["dice"]["point+box"],
                }
    
            results.append(row)

        except Exception as e:
            print(e)

    df_results = pd.DataFrame(results)

    path_save_results = os.path.join(path_dataset, "predictions_MedSAM2_video.csv")
    df_results.to_csv(path_save_results, index=False)
    print(f"results saved at: {path_save_results}")

    #print("Mean SAM2 point-only Dice:", df_results["DSC (video points)"].mean())
    #print("Mean SAM2 box-only Dice:", df_results["DSC (video boxes)"].mean())
    #print("Mean SAM2 point+box Dice:", df_results["DSC (video combines)"].mean())

if __name__ == "__main__":
    main()

import json
import os
import random
from pathlib import Path
from typing import List, Optional, Tuple
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import cv2
import torchvision.transforms.functional as tvf
import math
from shapely.geometry import Polygon
import shapely

# ─────────────────────────────────────────────
#  Label helpers
# ─────────────────────────────────────────────
WIDTH = 2048
HEIGHT = 2048
IOU_THRESH = 0.7
TOLERANCE = 15
CONTAINMENT_THRESH = 0.6
DETECTOR_SIZE = 1024

def convert_to_coords(seg):
    return [np.array([seg[i], seg[i+1]]) for i in range(len(seg))[::2]]

def convert_poly_to_mask(segmentations: dict) -> np.typing.NDArray:
    '''converts segmentation polygon to a mask'''
    mask = np.zeros((HEIGHT, WIDTH), dtype=np.uint8)
    for category_id, segmentation in segmentations.items():
        # 1. Reshape [x1, y1, x2, y2...] into [[x1, y1], [x2, y2]...]
        for seg in segmentation:
            pts = convert_to_coords(seg)

            # 3. FIX: Coordinates MUST be np.int32 for OpenCV polygon drawing
            pts = np.array(pts, dtype=np.int32)

            # 4. FIX: Wrap 'pts' inside a list [pts]
            cv2.fillPoly(mask, [pts], color=1)
    return mask          # (3, H, W, D)

7
# ─────────────────────────────────────────────
#  Augmentations (minimal, applied to tensors)
# ─────────────────────────────────────────────

def random_flip(image: torch.Tensor, label: torch.Tensor):
    """Random axis flips — safe for 3D volumes."""
    for axis in [0, 1]:           # H, W, D axes (channel is 0)
        if random.random() > 0.5:
            image = torch.flip(image, [axis])
            label = torch.flip(label, [axis])
    return image, label


def random_intensity_shift(image: torch.Tensor, shift: float = 0.1, scale: float = 0.1):
    """Per-channel random brightness + contrast augmentation."""
    for c in range(image.shape[0]):
        image[c] = image[c] * (1 + scale * (2 * random.random() - 1)) \
                             + shift * (2 * random.random() - 1)
    return image

def search_segmentation_from_id(annotations,image_id:str):
    '''finds matching image in ground truth'''
    segmentations = {}
    for item in annotations:
        # print("item",item['image_id'])
        if item['image_id'] == image_id:
            cat_id = item['category_id']
            all_items = []
            for seg in item['segmentation']:
                all_items.extend(seg)

            if cat_id in segmentations:
                segmentations[cat_id].append(all_items)
            else:
                segmentations[cat_id] = []
                segmentations[cat_id].append(all_items)
    return segmentations

def polygon_to_bbox(coords: list) -> np.ndarray:
    """
    coords: flat list [x1, y1, x2, y2, ...]
    Returns (xmin, ymin, xmax, ymax) as float32.
    """
    xs = coords[0::2]
    ys = coords[1::2]
    return np.array([min(xs), min(ys), max(xs), max(ys)], dtype=np.float32)

def crop_division(image: np.typing.NDArray,
                   label: np.typing.NDArray,
                   overlap: float,
                   crop_size: Tuple[int, int] = (1024, 1024)):
    hd, wd = image.shape[0] // crop_size[0], image.shape[1] // crop_size[1]
    hr, wr = image.shape[0] % crop_size[0], image.shape[1] % crop_size[1]

    hp = 0 if hr == 0 else crop_size[0] * (hd + 1) - image.shape[0]
    wp = 0 if wr == 0 else crop_size[1] * (wd + 1) - image.shape[1]

    # guarantee at least one full crop_size tile even if image is smaller than crop_size
    hp = max(hp, crop_size[0] - image.shape[0])
    wp = max(wp, crop_size[1] - image.shape[1])

    pad_config = ((0, hp), (0, wp))
    padded_image = np.pad(image, pad_width=pad_config, mode='constant', constant_values=0)
    padded_label = np.pad(label, pad_width=pad_config, mode='constant', constant_values=0)

    H, W = padded_image.shape[0], padded_image.shape[1]
    hs = max(1, int(crop_size[0] * (1 - overlap)))
    ws = max(1, int(crop_size[1] * (1 - overlap)))

    start_h = list(range(0, H - crop_size[0] + 1, hs))
    start_w = list(range(0, W - crop_size[1] + 1, ws))
    if start_h[-1] + crop_size[0] < H:
        start_h.append(H - crop_size[0])
    if start_w[-1] + crop_size[1] < W:
        start_w.append(W - crop_size[1])

    img_list, label_list = [], []
    for sh in start_h:
        for sw in start_w:
            img_list.append(padded_image[sh:sh + crop_size[0], sw:sw + crop_size[1]])
            label_list.append(padded_label[sh:sh + crop_size[0], sw:sw + crop_size[1]])

    return img_list, label_list

def crop_division_bbox(image, labels, boxes, overlap, crop_size=(1024, 1024)):
    hd, wd = image.shape[0] // crop_size[0], image.shape[1] // crop_size[1]
    hr, wr = image.shape[0] % crop_size[0], image.shape[1] % crop_size[1]

    hp = 0 if hr == 0 else crop_size[0] * (hd + 1) - image.shape[0]
    wp = 0 if wr == 0 else crop_size[1] * (wd + 1) - image.shape[1]
    hp = max(hp, crop_size[0] - image.shape[0])
    wp = max(wp, crop_size[1] - image.shape[1])

    pad_config = ((0, hp), (0, wp))
    padded_image = np.pad(image, pad_width=pad_config, mode='constant', constant_values=0)
    padded_labels = np.stack([np.pad(label, pad_width=pad_config, mode='constant', constant_values=0) for label in labels])

    H, W = padded_image.shape[0], padded_image.shape[1]
    hs = max(1, int(crop_size[0] * (1 - overlap)))
    ws = max(1, int(crop_size[1] * (1 - overlap)))

    start_h = list(range(0, H - crop_size[0] + 1, hs))
    start_w = list(range(0, W - crop_size[1] + 1, ws))
    if start_h[-1] + crop_size[0] < H:
        start_h.append(H - crop_size[0])
    if start_w[-1] + crop_size[1] < W:
        start_w.append(W - crop_size[1])

    img_list, label_list, box_list = [], [], []
    for sh in start_h:
        for sw in start_w:
            crop_y1, crop_x1 = sh, sw
            crop_y2, crop_x2 = sh + crop_size[0], sw + crop_size[1]

            x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
            intersects = (x1 < crop_x2) & (x2 > crop_x1) & (y1 < crop_y2) & (y2 > crop_y1)

            cropped_boxes = boxes[intersects].copy()

            if len(cropped_boxes) > 0:
                cropped_boxes[:, [0, 2]] = np.clip(cropped_boxes[:, [0, 2]], crop_x1, crop_x2) - crop_x1
                cropped_boxes[:, [1, 3]] = np.clip(cropped_boxes[:, [1, 3]], crop_y1, crop_y2) - crop_y1

                img_list.append(padded_image[crop_y1:crop_y2, crop_x1:crop_x2])
                label_list.append(padded_labels[intersects,crop_y1:crop_y2, crop_x1:crop_x2])
                box_list.append(cropped_boxes)


    return img_list, label_list, box_list

def iou_xyxy(box1, box2) -> float:
    x1, y1, x2, y2 = box1
    X1, Y1, X2, Y2 = box2
    ix1, iy1 = max(x1, X1), max(y1, Y1)
    ix2, iy2 = min(x2, X2), min(y2, Y2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    area1 = (x2 - x1) * (y2 - y1)
    area2 = (X2 - X1) * (Y2 - Y1)
    union_area = area1 + area2 - inter
    return inter / union_area if union_area > 0 else 0.0


def merge_box_xyxy(box1, box2):
    x1, y1, x2, y2 = box1
    X1, Y1, X2, Y2 = box2
    return (min(x1, X1), min(y1, Y1), max(x2, X2), max(y2, Y2))

def box_contains(outer, inner, tolerance: float = 0.0) -> bool:
    """
    Returns True if `inner` box is fully contained within `outer` box.
    Boxes are (xmin, ymin, xmax, ymax).

    tolerance: small slack in pixels to allow near-containment
               (e.g. tolerance=2 allows inner to stick out by up to 2px).
    """
    ox1, oy1, ox2, oy2 = outer
    ix1, iy1, ix2, iy2 = inner

    return (
        ix1 >= ox1 - tolerance and
        iy1 >= oy1 - tolerance and
        ix2 <= ox2 + tolerance and
        iy2 <= oy2 + tolerance
    )


def either_contains(box1, box2, tolerance: float = 0.0):
    """
    Checks containment in both directions.
    Returns:
        "box1_contains_box2", "box2_contains_box1", or None
    """
    if box_contains(box1, box2, tolerance):
        return box1,box2,True
    if box_contains(box2, box1, tolerance):
        return box2,box1,True
    return None, None, False

def calculate_area(box):
    x1,y1,x2,y2 = box
    return abs((x2 -x1)*(y2 - y1))


def containment_score(box1,box2,tolerance:float = 0.0):
    outer, inner, is_contained = either_contains(box1,box2,tolerance)
    if not is_contained:
        return 0.0
    else:
        print(calculate_area(inner)/calculate_area(outer))
        return calculate_area(inner)/calculate_area(outer)


def merge_seg(seg1:np.typing.NDArray,seg2:np.typing.NDArray):
    new_seg = seg1 | seg2
    return new_seg

def flatten_dictionary(cat_dic:dict):
    ls = []
    for cat_id, value in cat_dic.items():
        ls.extend(value)
    return ls

def resize_everything(image, masks,boxes,resize = DETECTOR_SIZE):
    old_width, old_height = image.shape  # PIL: (W, H)
    new_height, new_width = resize, resize

    scale_x = new_width / old_width
    scale_y = new_height/ old_height

    resized_image = cv2.resize(image, (resize,resize), interpolation=cv2.INTER_NEAREST).astype(np.float32)
    resized_masks = [cv2.resize(mask, (resize,resize), interpolation=cv2.INTER_NEAREST).astype(bool) for mask in masks]

    resized_boxes = boxes
    resized_boxes[:, 0] *= scale_x # x1
    resized_boxes[:, 1] *= scale_y # y1
    resized_boxes[:, 2] *= scale_x  # x2
    resized_boxes[:, 3] *= scale_y
    return resized_image, resized_masks, resized_boxes


def check_intersection(seg1,seg2):
    poly1 = Polygon(convert_to_coords(seg1))
    poly2 = Polygon(convert_to_coords(seg2))

    return shapely.overlaps(poly1,poly2)

def crop_around_box(image,label,box):
    x1 = int(box[0])
    y1 = int(box[1])
    x2 = int(box[2])
    y2 = int(box[3])

    cropped_image = image[y1:y2,x1:x2]
    cropped_label = label[y1:y2,x1:x2]
    return cropped_image, cropped_label
def merge_overlapping_boxes(boxes,segments, iou_thresh: float = IOU_THRESH, contain_tolerance: float = CONTAINMENT_THRESH):
    """
    boxes: list of (xmin, ymin, xmax, ymax)
    Groups boxes that overlap (IoU > threshold) OR one contains the other,
    transitively, then merges each group into a single enclosing box.

    Returns: list of merged boxes.
    """
    n = len(boxes)
    if n == 0:
        return []

    parent = list(range(n))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    def union(i, j):
        ri, rj = find(i), find(j)
        if ri != rj:
            parent[ri] = rj

    # Pairwise compare every box, union if IoU passes OR either contains the other
    # iou_xyxy(boxes[i], boxes[j]) > iou_thresh or containment_score(boxes[i], boxes[j], tolerance = contain_tolerance) >= 0.6 or
    for i in range(n):
        for j in range(i + 1, n):
            if check_intersection(segments[i],segments[j]):
                union(i, j)

    # Group boxes by their root parent
    groups = {}
    seg_groups = {}
    for i in range(n):
        root = find(i)
        groups.setdefault(root, []).append(boxes[i])
        seg_groups.setdefault(root, []).append(convert_poly_to_mask({1:[segments[i]]}))

    # Merge each group into one enclosing box
    merged = []
    merged_seg = []
    for group in groups.values():
        cur = group[0]
        for b in group[1:]:
            cur = merge_box_xyxy(cur, b)
        merged.append(cur)

    for group in seg_groups.values():
        cur = group[0]
        for b in group[1:]:
            cur = merge_seg(cur, b)
        merged_seg.append(cur)

    return merged,merged_seg
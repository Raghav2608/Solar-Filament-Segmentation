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

# ─────────────────────────────────────────────
#  Dataset
# ─────────────────────────────────────────────

class SolarDataset(Dataset):
    """
    Args:
        data_root : path to the folder containing per-subject subdirectories
        subject_ids: list of subject folder names; if None, all subdirs are used
        roi_size   : (H, W) crop size for centre-crop; None = full volume
        augment    : apply random flips + intensity shift during training
    """


    def __init__(
        self,
        data_root: str,
        subject_ids:List[str],
        roi_size: Optional[Tuple[int, int]] = (2048,2048),
        augment: bool = False
    ):
        self.data_root  = Path(data_root)
        self.roi_size   = roi_size
        self.augment    = augment

        with open(str(self.data_root / "MAGFiLO_1.0_Annotations_kaggle2026_train.json"),"r") as f:
            self.gt = json.load(f)
        self.subjects = subject_ids

    def __len__(self):
        return len(self.subjects)

    # ── internal helpers ──────────────────────

    def _load_image(self, path: Path) -> np.ndarray:
        return cv2.imread(str(path),cv2.IMREAD_UNCHANGED)

    def _get_all_image_ids(self,image_path:str) -> List[str]:
        img_ids = []
        for an in self.gt["images"]:
            file_name = an["file_name"]
            if file_name == image_path:
                img_ids.append(an["id"])
        return img_ids

    def get_segmentations_by_index(self,image_id:str):
        '''gets segmentations masks by image id'''
        segmentation = search_segmentation_from_id(self.gt['annotations'],image_id)
        return convert_poly_to_mask(segmentation)

    def get_bbox_by_index(self, image_id: str):
        """
        Returns a dict {category_id: [bbox, bbox, ...]}, mirroring the
        structure of search_segmentation_from_id, so callers can align
        boxes with their originating segmentation/category.
        """
        segmentations = search_segmentation_from_id(self.gt['annotations'], image_id)
        bboxes = {}
        for cat_id, polygons in segmentations.items():
            bboxes[cat_id] = [polygon_to_bbox(poly) for poly in polygons]
        return bboxes

    def blender(self,image_path:str,technique:str = "union"):
        '''takes images from multiple annotators and blends them in a specific way to create a single gt
        '''
        all_image_ids = self._get_all_image_ids(image_path)
        # get all images and blend them
        cur_mask = np.zeros((HEIGHT, WIDTH), dtype=np.uint8)
        if technique == "union":
            for image_id in all_image_ids:
                cur_mask = cur_mask | self.get_segmentations_by_index(image_id)

        # add more blending technqiues
        return cur_mask


    def _normalise(self, img: np.ndarray) -> np.ndarray:
        """normalisation"""
        return (img) / (img.max() + 1e-8)

    def _centre_crop(
        self, image: np.ndarray, label: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Crop around the image centre to self.roi_size."""
        H, W = image.shape
        rH, rW = self.roi_size
        sH = max(0, (H - rH) // 2)
        sW = max(0, (W - rW) // 2)
        image = image[sH:sH+rH, sW:sW+rW]
        label = label[sH:sH+rH, sW:sW+rW]
        return image, label

    # ── __getitem__ ───────────────────────────

    def __getitem__(self, idx: int):
        subject = self.subjects[idx]
        image_path  = self.data_root / "train_images"/ subject

        image = self._load_image(image_path)
        label = self.blender(subject).astype(np.int32) # blend segmentations with multiple annotators lowkirkkenuinely

        # Convert to tensors
        image = torch.from_numpy(image)
        label = torch.from_numpy(label)
        # Crop
        if self.roi_size is not None:
            image, label = self._centre_crop(image, label)

        # Augmentations (training only)
        if self.augment:
            image, label = random_flip(image, label)
            image = random_intensity_shift(image)

        return {"image": image, "label": label, "subject": subject}

class SolarDatasetPatchBased(SolarDataset):
    def __getitem__(self, idx):
        subject = self.subjects[idx]
        image_path  = self.data_root / "train_images"/ subject

        image = self._load_image(image_path)
        label = self.blender(subject).astype(np.int32) #blend segmentations with multiple annotators lowkirkkenuinely
        img_list, label_list = crop_division(image,label,overlap=0.5)

        new_img_list = []
        new_label_list = []

        for img, lab in zip(img_list,label_list):
            if (lab > 0).any():
                new_img_list.append(img)
                new_label_list.append(lab)

        if random.random() < 0.8 and len(new_img_list) > 0:
            img_list, label_list = new_img_list, new_label_list

        random_index = random.randrange(len(img_list))
        image = img_list[random_index]
        label = label_list[random_index]

        # Convert to tensors
        image = torch.from_numpy(image).float()
        label = torch.from_numpy(label).float()
        # Crop
        if self.roi_size is not None:
            image, label = self._centre_crop(image, label)

        # Augmentations (training only)
        if self.augment:
            image, label = random_flip(image, label)
            image = random_intensity_shift(image)

        return {"image": image, "label": label, "subject": subject}

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

def crop_around_box(image,label,box,crop_size = (512,512)):
    x1 = int(box[0])
    y1 = int(box[1])
    x2 = int(box[2])
    y2 = int(box[3])

    padded_image = np.pad(image, ((crop_size[0] // 2, crop_size[0] // 2), (crop_size[1] // 2, crop_size[1] // 2)), mode='constant')
    padded_label = np.pad(label, ((crop_size[0] // 2, crop_size[0] // 2), (crop_size[1] // 2, crop_size[1] // 2)), mode='constant')
    # Crop the image and label

    x1 = x1 + crop_size[0] // 2
    y1 = y1 + crop_size[1] // 2
    x2 = x2 + crop_size[0] // 2
    y2 = y2 + crop_size[1] // 2

    mid_x = (x1 + x2) // 2
    mid_y = (y1 + y2) // 2

    x1 = mid_x - crop_size[0] // 2
    y1 = mid_y - crop_size[1] // 2
    x2 = mid_x + crop_size[0] // 2
    y2 = mid_y + crop_size[1] // 2


    cropped_image = padded_image[y1:y2,x1:x2]
    cropped_label = padded_label[y1:y2,x1:x2]
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

class SolarDatasetBBox(SolarDataset):
    def blender(self,image_path:str,technique:str = "bbox"):
        '''takes images from multiple annotators and blends them in a specific way to create a single gt
        '''
        all_image_ids = self._get_all_image_ids(image_path)
        cur_mask = super().blender(image_path,"union")
        if technique == "bbox":
            all_boxes = []
            all_segments = []
            for image_id in all_image_ids:
                segs = search_segmentation_from_id(self.gt['annotations'], image_id)
                for cat_id, polygons in segs.items():
                    all_segments.extend(polygons)
                    for poly in polygons:
                        all_boxes.append(polygon_to_bbox(poly))

            assert len(all_boxes) == len(all_segments)
            merged_boxes, merged_segmentations = merge_overlapping_boxes(all_boxes, all_segments,iou_thresh=0.7,contain_tolerance = 4)
            # add more blending technqiues
            return merged_boxes, merged_segmentations, cur_mask

    def __getitem__(self, idx):
        subject = self.subjects[idx]
        image_path  = self.data_root / "train_images"/ subject

        image = self._load_image(image_path)
        boxes, masks, _ = self.blender(subject) #blend segmentations with multiple annotators lowkirkkenuinely
        boxes = [np.array(box).astype(np.float16) for box in boxes]
        boxes = np.stack(boxes)

        img_list, mask_list, box_list = crop_division_bbox(image,masks,boxes,overlap=0.5)
        new_img_list = []
        new_mask_list = []
        new_box_list = []
        for img, labs,boxes in zip(img_list,mask_list,box_list):
            if any([(lab > 0).any() for lab in labs]):
                new_img_list.append(img)
                new_mask_list.append(labs)
                new_box_list.append(boxes)

        if random.random() < 0.8 and len(new_img_list) > 0:
            img_list, mask_list, box_list = new_img_list, new_mask_list, new_box_list

        retry = True
        while retry:
          random_index = random.randrange(len(img_list))
          image = img_list[random_index]
          masks = mask_list[random_index]
          boxes = box_list[random_index]
          if boxes.shape[0] > 0:
            retry = False

        image = self._normalise(image)


        image = torch.as_tensor(image,dtype=torch.float32)
        boxes = torch.as_tensor(boxes,dtype=torch.float32)
        masks = torch.as_tensor(masks,dtype=torch.uint8)

        return image,{
                "labels": torch.ones(len(masks), dtype=torch.int64),
                "masks": masks,
                "image_id": torch.tensor([idx]),
                "boxes":boxes,
                "area":torch.as_tensor(masks.sum((1, 2)), dtype=torch.float32),
                "iscrowd": torch.zeros(len(masks), dtype=torch.int64)
               }

class SolarDatasetYOLO(SolarDatasetBBox):
    def __getitem__(self, idx):
        subject = self.subjects[idx]
        image_path  = self.data_root / "train_images"/ subject

        image = self._load_image(image_path)
        boxes, masks, _ = self.blender(subject) #blend segmentations with multiple annotators lowkirkkenuinely
        boxes = [np.array(box).astype(np.float16) for box in boxes]
        boxes = np.stack(boxes)
        img_list, _, box_list = crop_division_bbox(image,masks,boxes,overlap=0.5,crop_size=(640,640))

        new_img_list, new_box_list = [], []
        for img, labs,boxes in zip(img_list,_,box_list):
            if any([(lab > 0).any() for lab in labs]):
                new_img_list.append(img)
                new_box_list.append(boxes)

        return {"image":new_img_list,
                "image_id": idx,
                "boxes":new_box_list,
               }

class SolarDatasetBBoxResize(SolarDatasetBBox):
    def __getitem__(self, idx):
        subject = self.subjects[idx]
        image_path  = self.data_root / "train_images"/ subject

        image = self._load_image(image_path)
        boxes, masks, _ = self.blender(subject) #blend segmentations with multiple annotators lowkirkkenuinely
        boxes = np.stack(boxes)
        image = self._normalise(image)
        # Convert to tensors
        image,masks,boxes = resize_everything(image,masks,boxes)
        masks = np.stack(masks)
        image = torch.as_tensor(image,dtype=torch.float32)
        boxes = torch.as_tensor(boxes,dtype=torch.float32)
        masks = torch.as_tensor(masks,dtype=torch.uint8)

        return image,{
                "labels": torch.ones(len(masks), dtype=torch.int64),
                "masks": masks,
                "image_id": torch.tensor([idx]),
                "boxes":boxes,
                "area":torch.as_tensor(masks.sum((1, 2)), dtype=torch.float32),
                "iscrowd": torch.zeros(len(masks), dtype=torch.int64)
               }
import random
import numpy as np
import torch
from scipy.ndimage import gaussian_filter, map_coordinates

import random

def jitter_bbox_asymmetric(box, img_w=None, img_h=None,
                            pos_jitter=0.08,
                            tight_prob=0.35, tight_range=(0.05, 0.25),
                            loose_prob=0.35, loose_range=(0.05, 0.3),
                            symmetric_jitter=0.05,
                            clip=True, seed=None):
    """
    Simulate realistic imperfect YOLO detections on a corner-format box,
    for training CropUNet to be robust to localization error.

    Each of the 4 edges independently gets one of:
      - "tight": edge moves inward (risking cutting off the filament)
      - "loose": edge moves outward (including extra background)
      - default: small symmetric jitter (normal detector noise)

    Args:
        box: (x1, y1, x2, y2), pixel or normalized coords — units are
             preserved in the output (no normalization assumed).
        img_w, img_h: needed only if clip=True, to bound the box to the image.
        pos_jitter: max fractional center drift, relative to box w/h.
        tight_prob: probability an individual edge shrinks inward.
        tight_range: fractional shrink amount (of that side's w or h).
        loose_prob: probability an individual edge expands outward.
        loose_range: fractional expand amount.
        symmetric_jitter: fallback small jitter when neither tight nor loose fires.
        clip: clamp final box to [0, img_w] / [0, img_h] if provided.
        seed: optional int for reproducibility.

    Returns:
        (x1, y1, x2, y2) jittered box, same units as input.
    """
    if seed is not None:
        random.seed(seed)

    x1, y1, x2, y2 = box
    w = x2 - x1
    h = y2 - y1

    def perturb_edge(extent):
        r = random.random()
        if r < tight_prob:
            return random.uniform(*tight_range) * extent      # shrinks inward
        elif r < tight_prob + loose_prob:
            return -random.uniform(*loose_range) * extent     # expands outward
        else:
            return random.uniform(-symmetric_jitter, symmetric_jitter) * extent

    x1_new = x1 + perturb_edge(w)
    x2_new = x2 - perturb_edge(w)
    y1_new = y1 + perturb_edge(h)
    y2_new = y2 - perturb_edge(h)

    # extra whole-box center drift, independent of edge jitter
    dx = random.uniform(-pos_jitter, pos_jitter) * w
    dy = random.uniform(-pos_jitter, pos_jitter) * h
    x1_new += dx
    x2_new += dx
    y1_new += dy
    y2_new += dy

    # guard against degenerate/inverted boxes
    if x2_new <= x1_new:
        cx = (x1 + x2) / 2
        x1_new, x2_new = cx - 1e-3, cx + 1e-3
    if y2_new <= y1_new:
        cy = (y1 + y2) / 2
        y1_new, y2_new = cy - 1e-3, cy + 1e-3

    if clip:
        max_x = img_w if img_w is not None else float("inf")
        max_y = img_h if img_h is not None else float("inf")
        x1_new = min(max(x1_new, 0.0), max_x)
        x2_new = min(max(x2_new, 0.0), max_x)
        y1_new = min(max(y1_new, 0.0), max_y)
        y2_new = min(max(y2_new, 0.0), max_y)

    return (x1_new, y1_new, x2_new, y2_new)


def jitter_bboxes_asymmetric(boxes, **kwargs):
    """Apply jitter_bbox_asymmetric to a list of (x1,y1,x2,y2) boxes."""
    return [jitter_bbox_asymmetric(b, **kwargs) for b in boxes]

class SolarDatasetBoxCrop(SolarDatasetBBox):
    def __init__(self, *args,
                 jitter_prob=0.7,
                 flip_prob=0.5,
                 rotate90_prob=0.5,
                 elastic_prob=0.3,
                 intensity_prob=0.5,
                 **kwargs):
        super().__init__(*args, **kwargs)
        self.jitter_prob = jitter_prob
        self.flip_prob = flip_prob
        self.rotate90_prob = rotate90_prob
        self.elastic_prob = elastic_prob
        self.intensity_prob = intensity_prob

    def __getitem__(self, idx):
        subject = self.subjects[idx]
        image_path = self.data_root / "train_images" / subject

        image = self._load_image(image_path)
        boxes, _, label = self.blender(subject)  # blend segmentations with multiple annotators

        img_list = []
        label_list = []
        for box in boxes:
            # jitter the box BEFORE cropping, only at train time, so CropUNet
            # sees the kind of imperfect localization YOLO will actually give it
            if self.augment and random.random() < self.jitter_prob:
                box = jitter_bbox_asymmetric(
                    box, img_w=image.shape[1], img_h=image.shape[0]
                )

            crop_image, crop_label = crop_around_box(image, label, box, crop_size=(512, 512))

            if self.augment:
                crop_image, crop_label = self._augment(crop_image, crop_label)

            img_list.append(crop_image)
            label_list.append(crop_label)

        img_list = [self._normalise(img.astype(np.float32)) for img in img_list]
        images = np.stack(img_list,axis=0)
        labels = np.stack(label_list,axis=0)

        # Convert to tensors
        images = torch.as_tensor(images, dtype=torch.float32)
        labels = torch.as_tensor(labels, dtype=torch.uint8)

        return {
                "image": images,
                "label": labels
                }

    def _augment(self, image, label):
        """Geometric + intensity augmentation for a single (image, label) crop pair."""

        # --- horizontal/vertical flips (safe for filament orientation, no bias) ---
        if random.random() < self.flip_prob:
            image = np.fliplr(image)
            label = np.fliplr(label)
        if random.random() < self.flip_prob:
            image = np.flipud(image)
            label = np.flipud(label)

        # --- 90-degree rotations (filaments have no canonical orientation) ---
        if random.random() < self.rotate90_prob:
            k = random.randint(1, 3)
            image = np.rot90(image, k)
            label = np.rot90(label, k)

        # --- elastic deformation, good for thin curvy structures like filaments ---
        if random.random() < self.elastic_prob:
            image, label = self._elastic_deform(image, label)

        # --- intensity jitter (brightness/contrast), image only ---
        if random.random() < self.intensity_prob:
            image = self._intensity_jitter(image)

        # np.rot90/flip return views with negative strides; torch dislikes that
        image = np.ascontiguousarray(image)
        label = np.ascontiguousarray(label)

        return image, label

    def _elastic_deform(self, image, label, alpha=30, sigma=5):
        """Elastic deformation applied identically to image and label."""
        shape = image.shape
        dx = gaussian_filter((np.random.rand(*shape) * 2 - 1), sigma) * alpha
        dy = gaussian_filter((np.random.rand(*shape) * 2 - 1), sigma) * alpha

        x, y = np.meshgrid(np.arange(shape[1]), np.arange(shape[0]))
        indices = (np.reshape(y + dy, (-1, 1)), np.reshape(x + dx, (-1, 1)))

        image_deformed = map_coordinates(image, indices, order=1, mode='reflect').reshape(shape)
        label_deformed = map_coordinates(label, indices, order=0, mode='reflect').reshape(shape)  # nearest-neighbor for masks

        return image_deformed, label_deformed

    def _intensity_jitter(self, image, brightness_range=(-0.1, 0.1), contrast_range=(0.9, 1.1)):
        """Random brightness/contrast shift. Assumes image is in a roughly [0,1] or raw grayscale range."""
        brightness = random.uniform(*brightness_range)
        contrast = random.uniform(*contrast_range)
        mean = image.mean()
        image = (image - mean) * contrast + mean + brightness * image.max()
        return image
# ─────────────────────────────────────────────
#  DataLoader factory
# ─────────────────────────────────────────────
import json

def build_coco_gt(dataset, categories):
    images = []
    annotations = []
    ann_id = 1

    for img_idx in range(len(dataset)):
      # print(dataset[img_idx])
      img, target = dataset[img_idx]
      image_id = int(target["image_id"])
      h, w = img.shape[-2], img.shape[-1]  # or however your dataset stores size

      images.append({"id": image_id, "file_name": f"{image_id}.png", "height": h, "width": w})

      boxes = target["boxes"]   # [N, 4], x1,y1,x2,y2
      labels = target["labels"]

      for box, label in zip(boxes, labels):
          x1, y1, x2, y2 = box.tolist()
          bw, bh = x2 - x1, y2 - y1
          annotations.append({
              "id": ann_id,
              "image_id": image_id,
              "category_id": int(label),
              "bbox": [x1, y1, bw, bh],
              "area": bw * bh,
              "iscrowd": 0
          })
          ann_id += 1

    return {"images": images, "annotations": annotations, "categories": categories}
def collate_fn(batch):
    return tuple(zip(*batch))

def get_dataloaders(
    data_root: str,
    val_fraction: float = 0.2,
    batch_size: int = 1,
    roi_size: Tuple[int, int] = (128, 128),
    num_workers: int = 1,
    seed: int = 42,
    train_dataset: str = "bbox",
    val_dataset: str = "standard",
) -> Tuple[DataLoader, DataLoader]:
    """
    Split subjects into train / val and return DataLoaders.
    """
    root = Path(data_root) / "train_images"
    all_subjects = os.listdir(str(root))

    random.seed(seed)
    random.shuffle(all_subjects)
    n_val = max(1, int(len(all_subjects) * val_fraction))
    val_ids   = all_subjects[:n_val]
    train_ids = all_subjects[n_val:]

    DATASET_CLASSES = {
        "standard": SolarDataset,
        "patch": SolarDatasetPatchBased,
        "bbox": SolarDatasetBBox,
        "bbox_resize": SolarDatasetBBoxResize,
        "boxcrop":SolarDatasetBoxCrop
    }

    NEEDS_COLLATE_FN = {"bbox", "bbox_resize"}

    train_ds = DATASET_CLASSES[train_dataset](
        data_root, train_ids, roi_size=roi_size, augment=True
    )
    val_ds = DATASET_CLASSES[val_dataset](
        data_root, val_ids, roi_size=roi_size, augment=False
    )


    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True, drop_last=True,
        collate_fn=collate_fn if train_dataset in NEEDS_COLLATE_FN else None,
    )

    val_loader = DataLoader(
        val_ds, batch_size=1, shuffle=False,
        num_workers=num_workers, pin_memory=True,
        collate_fn=collate_fn if val_dataset in NEEDS_COLLATE_FN else None,
    )

    return train_loader, val_loader
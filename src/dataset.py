import json
import os
import random
from pathlib import Path
from typing import List, Optional, Tuple

import nibabel as nib
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader 
import cv2


# ─────────────────────────────────────────────
#  Label helpers
# ─────────────────────────────────────────────
WIDTH = 2048
HEIGHT = 2048

def convert_poly_to_mask(segmentations: list) -> np.typing.NDArray:
    '''converts segmentation polygon to a mask'''
    mask = np.zeros((HEIGHT, WIDTH), dtype=np.uint8) 
    for category_id, segmentation in segmentations.items():
        # 1. Reshape [x1, y1, x2, y2...] into [[x1, y1], [x2, y2]...]
        for seg in segmentation:
            pts = [np.array([seg[i], seg[i+1]]) for i in range(len(seg))[::2]]
            
            # 3. FIX: Coordinates MUST be np.int32 for OpenCV polygon drawing
            pts = np.array(pts, dtype=np.int32)
            
            # 4. FIX: Wrap 'pts' inside a list [pts]
            cv2.fillPoly(mask, [pts], color=category_id)
    return mask          # (3, H, W, D)


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


# ─────────────────────────────────────────────
#  Dataset
# ─────────────────────────────────────────────

class SolarDataset(Dataset):
    """
    Args:
        data_root : path to the folder containing per-subject subdirectories
        subject_ids: list of subject folder names; if None, all subdirs are used
        roi_size   : (H, W, D) crop size for centre-crop; None = full volume
        augment    : apply random flips + intensity shift during training
    """


    def __init__(
        self,
        data_root: str,
        roi_size: Optional[Tuple[int, int]] = (2048,2048),
        augment: bool = False
    ):
        self.data_root  = Path(data_root)
        self.roi_size   = roi_size
        self.augment    = augment

        with open(str(self.data_root / "MAGFiLO_1.0_Annotations_kaggle2026_train.json"),"r") as f:
            self.gt = json.load(f)
        self.subjects = os.listdir(str(self.data_root / "train_images"))

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
        
     
    def blender(self,image_path:str,technique:str = "union"):
        '''takes images from multiple annotators and blends them in a specific way to create a single gt
        '''
        all_image_ids = self._get_all_image_ids(image_path)
        # get all images and blend them
        cur_mask = np.zeros((HEIGHT, WIDTH), dtype=np.uint8) 
        if technique == "union":
            for image_id in all_image_ids:
                cur_mask = cur_mask | self.get_segmentations_by_index(image_id)
        return cur_mask
    
    def _normalise(self, vol: np.ndarray) -> np.ndarray:
        """Z-score normalisation over non-zero voxels (brain mask)."""
        mask = vol > 0
        if mask.sum() == 0:
            return vol
        mean = vol[mask].mean()
        std  = vol[mask].std() + 1e-8
        out  = np.zeros_like(vol)
        out[mask] = (vol[mask] - mean) / std
        return out

    def _centre_crop(
        self, image: np.ndarray, label: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Crop around the image centre to self.roi_size."""
        H, W, D = image.shape[1:]
        rH, rW, rD = self.roi_size
        sH = max(0, (H - rH) // 2)
        sW = max(0, (W - rW) // 2)
        sD = max(0, (D - rD) // 2)
        image = image[:, sH:sH+rH, sW:sW+rW, sD:sD+rD]
        label = label[:, sH:sH+rH, sW:sW+rW, sD:sD+rD]
        return image, label

    # ── __getitem__ ───────────────────────────

    def __getitem__(self, idx: int):
        subject = self.subjects[idx]
        image_path  = self.data_root / "train_images"/ subject

        image = self._load_image(image_path) 
        
        # Load segmentation & convert to multi-label
        
        label = self.blender(subject).astype(np.int32)
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


# ─────────────────────────────────────────────
#  DataLoader factory
# ─────────────────────────────────────────────

def get_dataloaders(
    data_root: str,
    val_fraction: float = 0.2,
    batch_size: int = 1,
    roi_size: Tuple[int, int, int] = (128, 128, 128),
    num_workers: int = 4,
    seed: int = 42,
) -> Tuple[DataLoader, DataLoader]:
    """
    Split subjects into train / val and return DataLoaders.
    """
    root = Path(data_root)
    all_subjects = sorted(d.name for d in root.iterdir() if d.is_dir())

    random.seed(seed)
    random.shuffle(all_subjects)
    n_val = max(1, int(len(all_subjects) * val_fraction))
    val_ids   = all_subjects[:n_val]
    train_ids = all_subjects[n_val:]

    train_ds = SolarDataset(data_root, train_ids, roi_size=roi_size, augment=True)
    val_ds   = SolarDataset(data_root, val_ids,   roi_size=roi_size, augment=False)

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True, drop_last=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=1, shuffle=False,
        num_workers=num_workers, pin_memory=True,
    )
    return train_loader, val_loader
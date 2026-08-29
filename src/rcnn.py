from pycocotools.coco import COCO
from pycocotools import mask as mask_utils
from scipy.optimize import linear_sum_assignment
import torchvision
from torchvision.models.detection import maskrcnn_resnet50_fpn_v2, MaskRCNN_ResNet50_FPN_V2_Weights
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
from torchvision.models.detection.rpn import AnchorGenerator
from torchvision.ops import MultiScaleRoIAlign
import torch
import numpy as np 
import torch.nn.functional as F

DETECTOR_SIZE = 1024
IOU_THRESH = 0.7
CONTAINMENT_THRESH = 0.6
def build_detector(pretrained=True):
    weights = MaskRCNN_ResNet50_FPN_V2_Weights.DEFAULT if pretrained else None
    model = maskrcnn_resnet50_fpn_v2(weights=weights, weights_backbone=None)
    box_features = model.roi_heads.box_predictor.cls_score.in_features
    mask_features = model.roi_heads.mask_predictor.conv5_mask.in_channels
    hidden = model.roi_heads.mask_predictor.conv5_mask.out_channels
    model.roi_heads.box_predictor = FastRCNNPredictor(box_features, 2)
    model.roi_heads.mask_predictor = MaskRCNNPredictor(mask_features, hidden, 2)
    model.rpn.anchor_generator = AnchorGenerator(
        sizes=((16,), (32,), (64,), (128,), (256,)),
        aspect_ratios=((0.5, 1.0, 2.0),) * 5,
    )
    model.roi_heads.mask_roi_pool = MultiScaleRoIAlign(
        featmap_names=["0", "1", "2", "3"], output_size=28, sampling_ratio=2
    )
    model.transform.min_size = (DETECTOR_SIZE,)
    model.transform.max_size = DETECTOR_SIZE
    return model

def encode_mask(mask):
    return mask_utils.encode(np.asfortranarray(mask.astype(np.uint8)))

def make_masks_disjoint(masks, min_area=1):
    '''Assign every predicted pixel to at most one instance.

    Small/thin filaments receive contested pixels first so a large coarse mask
    cannot swallow them. The function preserves the original instance order.
    '''
    masks = [np.asarray(mask, dtype=bool) for mask in masks if np.any(mask)]
    if len(masks) < 2:
        return [mask for mask in masks if int(mask.sum()) >= min_area]
    shape = masks[0].shape
    if any(mask.shape != shape for mask in masks):
        raise ValueError("All masks from one image must have the same shape")
    coverage = np.zeros(shape, dtype=np.uint16)
    for mask in masks:
        coverage += mask.astype(np.uint16)
    if int(coverage.max()) <= 1:
        return [mask for mask in masks if int(mask.sum()) >= min_area]
    priority = sorted(range(len(masks)), key=lambda index: int(masks[index].sum()))
    occupied = np.zeros(shape, dtype=bool)
    result = [np.zeros(shape, dtype=bool) for _ in masks]
    for index in priority:
        result[index] = masks[index] & ~occupied
        occupied |= result[index]
    return [mask for mask in result if int(mask.sum()) >= min_area]

def merge_duplicate_masks(masks):
    masks = [np.asarray(mask, bool) for mask in masks if np.any(mask)]
    if len(masks) < 2:
        return make_masks_disjoint(masks)
    rles = [encode_mask(mask) for mask in masks]
    iou = mask_utils.iou(rles, rles, [0] * len(rles))
    areas = np.asarray([float(mask_utils.area(rle)) for rle in rles])
    intersection = iou * (areas[:, None] + areas[None, :]) / (1 + iou + 1e-7)
    containment = intersection / np.minimum(areas[:, None], areas[None, :]).clip(min=1)
    adjacency = (iou >= IOU_THRESH) | (containment >= CONTAINMENT_THRESH)
    groups, seen = [], set()
    for start in range(len(masks)):
        if start in seen:
            continue
        stack, group = [start], []
        while stack:
            index = stack.pop()
            if index in seen:
                continue
            seen.add(index); group.append(index)
            stack.extend(np.flatnonzero(adjacency[index]).tolist())
        groups.append(group)
    merged = [np.logical_or.reduce([masks[index] for index in group]) for group in groups]
    return make_masks_disjoint(merged)

def instance_metrics(pred_masks, true_masks):
    p, t = len(pred_masks), len(true_masks)
    if p and t:
        pred_rles, true_rles = [encode_mask(x) for x in pred_masks], [encode_mask(x) for x in true_masks]
        iou = mask_utils.iou(pred_rles, true_rles, [0] * t)
        dice = 2 * iou / (1 + iou + 1e-7)
        rows, cols = linear_sum_assignment(-dice)
        mean_dice = float(dice[rows, cols].sum() / max(p, t))
        tp = int((iou[rows, cols] >= 0.5).sum())
    else:
        mean_dice, tp = float(p == 0 and t == 0), 0
    precision, recall = (tp / p if p else 0), (tp / t if t else 0)
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0
    return {"mean_dice": mean_dice, "f1_50": f1, "count_error": abs(p - t), "predictions": p}


def sliding_window_inference(
    model: torch.nn.Module,
    image: torch.Tensor,
    roi_size: tuple = (1024, 1024),
    sw_batch_size: int = 1,
    overlap: float = 0.5,
    device: torch.device = torch.device("cpu"),
    score_threshold: float = 0.6,
    mask_threshold: float = 0.5

) -> torch.Tensor:
    """
    Tile the volume with overlapping windows, run the model on each tile,
    and Gaussian-weight-blend the outputs.

    Args:
        image : (1, C, H, W, D) tensor on CPU
        Returns: (1, 1, H, W, D) probability tensor (after sigmoid)
    """
    model.eval()
    _, C, H, W = image.shape
    rH, rW = roi_size

    # Stride
    sH = max(1, int(rH * (1 - overlap)))
    sW = max(1, int(rW * (1 - overlap)))

    # Pad so the volume is divisible by roi
    pH = max(0, rH - H)
    pW = max(0, rW - W)
    image_padded = F.pad(image, [0, pW, 0, pH])
    _, _, H2, W2 = image_padded.shape

    # Gaussian importance map
    def gauss_kernel_1d(size):
        x = np.arange(size) - size // 2
        g = np.exp(-0.5 * (x / (size / 6)) ** 2)
        return g / g.max()

    gH = gauss_kernel_1d(rH)
    gW = gauss_kernel_1d(rW)

    gauss3d = torch.from_numpy(
        gH[:, None] * gW[None, :]
    ).float()  # (rH, rW)

    output = torch.zeros(1, 1, H2, W2)
    count  = torch.zeros(1, 1, H2, W2)

    # Collect all window start coords
    starts_H = list(range(0, H2 - rH + 1, sH))
    starts_W = list(range(0, W2 - rW + 1, sW))
    # Ensure last window reaches the end
    if starts_H[-1] + rH < H2: starts_H.append(H2 - rH)
    if starts_W[-1] + rW < W2: starts_W.append(W2 - rW)

    tiles, coords = [], []
    for sh in starts_H:
        for sw in starts_W:
            tile = image_padded[:, :, sh:sh+rH, sw:sw+rW]
            tiles.append(tile)
            coords.append((sh, sw))

    # Run in micro-batches
    with torch.no_grad():
        for i in range(0, len(tiles), sw_batch_size):
            batch  = torch.cat(tiles[i:i+sw_batch_size], dim=0).to(device)
            out = model(batch)[0]
            keep = out["scores"] >= score_threshold
            probs = (out["masks"][keep, 0] >= mask_threshold).cpu().numpy()
            probs = merge_duplicate_masks(probs)
            if len(probs) > 0:
              probs = np.stack(merge_duplicate_masks(probs))
              probs = torch.as_tensor(probs.sum(axis=0))
            else:
              probs = torch.zeros((roi_size[0],roi_size[1]))
            probs = probs.unsqueeze(0)

            for j, (sh, sw) in enumerate(coords[i:i+sw_batch_size]):
                w = gauss3d.unsqueeze(0).unsqueeze(0) # (1,1,rH,rW)
                output[:, :, sh:sh+rH, sw:sw+rW] += probs[j:j+1] * w
                count[:,  :, sh:sh+rH, sw:sw+rW] += w

    output = output / (count + 1e-8)
    # Crop back to original size
    output = output[:, :, :H, :W]
    return output  # (1, 1, H, W) — probabilities

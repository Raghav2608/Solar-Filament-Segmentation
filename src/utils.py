import cv2
import numpy as np
from typing import List
import json
gt_path = "/kaggle/input/competitions/filament-segmentation-2026/MAGFiLO_1.0_Kaggle_2026/train/MAGFiLO_1.0_Annotations_kaggle2026_train.json"

with open(gt_path,"r") as f:
    annotations = json.loads(f.read())

def convert_seg_to_mask(segmentations: list) -> np.typing.NDArray:
    '''converts segmentation polygon to a mask'''
    mask = np.zeros((2048, 2048), dtype=np.uint8) 
    for category_id, segmentation in segmentations.items():
        # 1. Reshape [x1, y1, x2, y2...] into [[x1, y1], [x2, y2]...]
        for seg in segmentation:
            pts = [np.array([seg[i], seg[i+1]]) for i in range(len(seg))[::2]]
            
            # 3. FIX: Coordinates MUST be np.int32 for OpenCV polygon drawing
            pts = np.array(pts, dtype=np.int32)
            
            # 4. FIX: Wrap 'pts' inside a list [pts]
            cv2.fillPoly(mask, [pts], color=1)
    return mask 

def blender(image_path:str,technique:str = "union"):
    '''takes images from multiple annotators and blends them in a specific way to create a single gt
    '''
    all_image_ids = get_all_image_ids(image_path)
    # get all images and blend them
    cur_mask = np.zeros((2048, 2048), dtype=np.uint8) 
    if technique == "union":
        for image_id in all_image_ids:
            cur_mask = cur_mask | get_segmentations_by_index(image_id)
    return cur_mask

def get_all_image_ids(image_path:str) -> List[str]:
    img_ids = []
    for an in annotations["images"]:
        file_name = an["file_name"]
        if file_name == image_path:
            img_ids.append(an["id"])
    return img_ids
    
def get_segmentations_by_index(image_id:str):
    '''gets segmentations masks by image id'''
    segmentation = search_segmentation_from_id(annotations['annotations'],image_id)
    return convert_seg_to_mask(segmentation)

def search_segmentation_from_id(annotations:list,image_id:str):
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
        else:
            pass
            # print("done")
            # print(image_id)
    return segmentations

import matplotlib.pyplot as plt
import os

def visualise_segmentation(index:int,root_dir:str):
    image_ids = [image_id['id'] for image_id in annotations['images']]
    img_id = image_ids[index]
    mask = get_segmentations_by_index(img_id)
    image_path = os.path.join(root_dir,img_id.split('-')[-1] + '.jpeg')
    
    # # 1. Load the original image (BGR)
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    # 2. Create a dummy binary mask (True where the object is)
    color_mask = np.zeros_like(image)
    
    color_mask[mask == 1] = [0, 255, 0]
    color_mask[mask == 2] = [255,0 , 0]
    color_mask[mask == 3] = [0,0, 255]
    # 3. Blend the original image and the color mask
    alpha = 0.4  # Transparency of the mask
    beta = 1 # Transparency of the original image
    overlay = cv2.addWeighted(image, beta, color_mask, alpha, 0)
    
    # Save or display the resul
    plt.figure(figsize=(10, 10))
    plt.imshow(overlay)
    plt.title(img_id)
    plt.axis('off')  # Optional: Hides the graph tick marks and pixel axes
    plt.show()


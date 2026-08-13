import matplotlib.pyplot as plt
import cv2
import numpy as np
import os
from .dataset import SolarDataset

def visualise_segmentation(image,label):
    image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    # 2. Create a dummy binary mask (True where the object is)
    mask = label
    color_mask = np.zeros_like(image)
    
    color_mask[mask == 1] = [0, 255, 0]
    color_mask[mask == 2] = [255,0 , 0]
    color_mask[mask == 3] = [0,0, 255]
    # 3. Blend the original image and the color mask
    alpha = 0.4  # Transparency of the mask
    beta = 1 # Transparency of the original image
    overlay = cv2.addWeighted(image, beta, color_mask, alpha, 0)
    
    # Save or display the result
    plt.figure(figsize=(10, 10))
    plt.imshow(overlay)
    plt.axis('off')  # Optional: Hides the graph tick marks and pixel axes
    plt.show()

if __name__ == "__main__":
    dataset = SolarDataset(data_root="/kaggle/input/competitions/filament-segmentation-2026/MAGFiLO_1.0_Kaggle_2026/train",
                      roi_size=None)

    for i in dataset:
        image = i["image"]
        label = i["label"]
        visualise_segmentation(image.numpy(),label.numpy())

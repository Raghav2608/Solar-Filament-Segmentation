

import cv2
import numpy as np
import matplotlib.pyplot as plt
from .dataset import SolarDataset

def detect_roi(gray):
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # 3. Apply Otsu's thresholding to separate foreground from background
    # This automatically calculates the optimal threshold value
    _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    # 4. Find external boundaries (contours) of the isolated shapes
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # 5. Extract the target Region of Interest (assuming the largest shape)
    if contours:
        # Identify the contour with the largest area
        largest_contour = max(contours, key=cv2.contourArea)
        
        # Get standard bounding box coordinates: x, y, width, height
        x, y, w, h = cv2.boundingRect(largest_contour)
        
        # Crop the exact region out of the original grayscale image
        roi = gray[y:y+h, x:x+w]
        # Optional: Draw a visual anchor box on the original image
        # visual_output = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
        # cv2.rectangle(visual_output, (x, y), (x + w, y + h), (0, 255, 0), 2)
        # plt.imshow(visual_output)
        # plt.axis("off")
        # plt.show()
        return roi
        
        print(f"ROI coordinates found: X={x}, Y={y}, Width={w}, Height={h}")
    else:
        print("No regions of interest detected.")


if __name__ == "__main__":
    dataset = SolarDataset(data_root="/kaggle/input/competitions/filament-segmentation-2026/MAGFiLO_1.0_Kaggle_2026/train",
                        roi_size=None)
    all_roi = {
        0:[],
        1:[]
    }
    for i in dataset:
        image = i["image"]
        roi = detect_roi(image.numpy())
        for i,dim in enumerate(roi.shape):
            all_roi[i].append(dim)
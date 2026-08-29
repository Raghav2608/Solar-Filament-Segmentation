from pathlib import Path
import os
from ..dataset import SolarDataset, SolarDatasetYOLO
import cv2 

root_path = Path("/root/.cache/kagglehub/competitions/filament-segmentation-2026/MAGFiLO_1.0_Kaggle_2026/train")
subject_ids = os.listdir(str(root_path / "train_images"))

n_val = max(1, int(len(subject_ids) * 0.1))
val_ids   = subject_ids[:n_val]
train_ids = subject_ids[n_val:]

id_map = {
    "train":train_ids,
    "val":val_ids
}


def convert_to_yolo(box, img_w, img_h, mode="xyxy"):
    """
    Converts absolute pixel coordinates into normalized YOLO format.

    Parameters:
    - box (tuple/list): The bounding box values (e.g., [100, 150, 300, 350]).
    - img_w (int): The total width of the image in pixels.
    - img_h (int): The total height of the image in pixels.
    - mode (str): The format of your input box. Options:
                  'xyxy' -> [xmin, ymin, xmax, ymax] (Pascal VOC / OpenCV)
                  'xywh' -> [xmin, ymin, width, height] (COCO / PIL)

    Returns:
    - tuple: (x_center, y_center, width, height) normalized between 0.0 and 1.0.
    """
    if mode == "xyxy":
        xmin, ymin, xmax, ymax = box
        box_w = xmax - xmin
        box_h = ymax - ymin
        x_center = xmin + (box_w / 2.0)
        y_center = ymin + (box_h / 2.0)

    elif mode == "xywh":
        xmin, ymin, box_w, box_h = box
        x_center = xmin + (box_w / 2.0)
        y_center = ymin + (box_h / 2.0)

    else:
        raise ValueError("Mode must be either 'xyxy' or 'xywh'")

    # Normalize by dividing by total image dimensions
    nx = x_center / img_w
    ny = y_center / img_h
    nw = box_w / img_w
    nh = box_h / img_h

    # Safety clip to force coordinates strictly between 0.0 and 1.0
    nx = min(max(nx, 0.0), 1.0)
    ny = min(max(ny, 0.0), 1.0)
    nw = min(max(nw, 0.0), 1.0)
    nh = min(max(nh, 0.0), 1.0)

    # Round to 6 decimal places for YOLO standard precision
    return round(nx, 6), round(ny, 6), round(nw, 6), round(nh, 6)

for purpose, ids in id_map.items():
    dataset = SolarDatasetYOLO(root_path,
                      ids,
                      None)
    for i in dataset:
        img_list = i["image"]
        box_list =  i["boxes"]

        for index, img in enumerate(img_list):
            boxes = box_list[index]

            img_path = Path(f"dataset/{purpose}/images/{i["image_id"]}_{index}.jpeg")
            img_path.parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(f"dataset/{purpose}/images/{i["image_id"]}_{index}.jpeg",img)

            lines = []
            for box in boxes:
                if len(box) > 0:
                    x,y,w,h = convert_to_yolo(box.tolist(),640,640,mode="xyxy")
                    box_str = f"0 {x} {y} {w} {h}"
                    lines.append(box_str)

            boxes_string = "\n".join(lines)

            file_path = Path(f"dataset/{purpose}/labels/{i['image_id']}_{index}.txt")
            file_path.parent.mkdir(parents=True, exist_ok=True)
            file_path.write_text(boxes_string)


import json

def build_bbox_results(image_id, boxes_xyxy, scores, labels):
    """
    boxes_xyxy: [N,4] tensor, [x1,y1,x2,y2]
    scores: [N] tensor
    labels: [N] tensor, category_ids matching your categories list
    """
    boxes_xyxy = boxes_xyxy.cpu().numpy()
    scores = scores.cpu().numpy()
    labels = labels.cpu().numpy()

    results = []
    for i in range(len(scores)):
        x1, y1, x2, y2 = boxes_xyxy[i]
        results.append({
            "image_id": int(image_id),
            "category_id": int(labels[i]),
            "bbox": [float(x1), float(y1), float(x2 - x1), float(y2 - y1)],
            "score": float(scores[i]),
        })
    return results


# all_results = []
# model.eval()
# with torch.no_grad():
#     for images, targets in val_loader:
#         outputs = model(images[0].unsqueeze(0).unsqueeze(0).to("cuda"))  # list of dicts, eval mode
#         for target, output in zip(targets, outputs):
#             image_id = int(target["image_id"].item())
#             all_results.extend(build_bbox_results(
#                 image_id, output["boxes"], output["scores"], output["labels"]
#             ))

# with open("results.json", "w") as f:
#     json.dump(all_results, f)
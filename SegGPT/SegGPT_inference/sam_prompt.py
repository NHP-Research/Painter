import numpy as np
import torch
import matplotlib.pyplot as plt
import cv2
import sys
import os
import json
sys.path.append("..")
from segment_anything import sam_model_registry, SamPredictor

dataset_path = "/content/drive/MyDrive/Research/RatSoles"

def show_mask(mask, ax, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([30/255, 144/255, 255/255, 0.6])
        # color = np.array([0, 0, 0, 1])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)

def show_points(coords, labels, ax, marker_size=375):
    pos_points = coords[labels==1]
    neg_points = coords[labels==0]
    ax.scatter(pos_points[:, 0], pos_points[:, 1], color='green', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)
    ax.scatter(neg_points[:, 0], neg_points[:, 1], color='red', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)

def show_box(box, ax):
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='green', facecolor=(0,0,0,0), lw=2))

def generate_target_image(image_path, input_point, input_label, predictor, target_path):
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    predictor.set_image(image)
    masks, scores, logits = predictor.predict(
        point_coords=input_point,
        point_labels=input_label,
        multimask_output=True,
    )
    mask_expanded = np.expand_dims(masks[0], axis=-1)  # Change shape from (w, h) to (w, h, 1)
    mask_expanded = np.repeat(mask_expanded, image.shape[-1], axis=-1)  # Change shape from (w, h, 1) to (w, h, c)
    bw_image = np.full(image.shape, 255)
    masked_image = mask_expanded * bw_image
    cv2.imwrite(target_path, masked_image)

if __name__ == '__main__':
    sam_checkpoint = "sam_vit_h_4b8939.pth"
    model_type = "vit_h"
    device = "cuda"
    sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
    sam.to(device=device)
    predictor = SamPredictor(sam)

    with open("annotation.json", "r") as f:
        annotation = json.load(f)
    
    for anno in annotation:
        print(f"Processing {anno['path']}")
        generate_target_image(os.path.join(dataset_path, anno["path"]), np.array(anno["points"]), np.array(anno["labels"]), predictor, os.path.join(dataset_path, "Targets", anno["path"]))
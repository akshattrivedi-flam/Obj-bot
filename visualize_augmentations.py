import os
import sys
import cv2
import torch
import numpy as np
import json
from PIL import Image
import albumentations as A

# Add project root to sys.path
sys.path.append(os.getcwd())

from train_overfit import BottleDataset, get_transforms
from objectron.dataset import graphics

def visualize_augments(data_dir='bottle_data', output_dir='output/augmentations'):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Use augmenting transforms WITHOUT normalization/ToTensor for visualization
    def get_viz_transforms():
        return A.ReplayCompose([
            A.Resize(224, 224),
            A.HorizontalFlip(p=0.5),
            A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.1, rotate_limit=15, p=0.5),
            A.RandomBrightnessContrast(p=0.5),
            A.HueSaturationValue(p=0.5),
            A.GaussNoise(p=0.3),
        ], keypoint_params=A.KeypointParams(format='xy', remove_invisible=False))

    # We need to monkey-patch or use a modified version of the dataset to return numpy images
    # Alternatively, just use the logic directly here.
    
    anno_file = os.path.join(data_dir, 'annotations.json')
    img_dir = os.path.join(data_dir, 'images')
    
    with open(anno_file, 'r') as f:
        annotations = json.load(f)
    
    transform = get_viz_transforms()
    
    # Pick one frame
    idx = 10
    anno = annotations[idx]
    img_path = os.path.join(img_dir, anno['image'])
    raw_image = np.array(Image.open(img_path).convert('RGB'))
    
    kps_2d = np.array(anno['keypoints_2d'], dtype=np.float32)
    kps_xy = kps_2d[:, :2]
    depths = kps_2d[:, 2]
    
    for i in range(10):
        transformed = transform(image=raw_image, keypoints=kps_xy)
        image = transformed['image']
        curr_kps_xy = np.array(transformed['keypoints'], dtype=np.float32)
        curr_depths = depths.copy()
        
        # Apply remapping if flipped
        is_flipped = False
        for t in transformed['replay']['transforms']:
            if t['__class_fullname__'] == 'HorizontalFlip' and t['applied']:
                is_flipped = not is_flipped
        
        if is_flipped:
            remap = [0, 5, 6, 7, 8, 1, 2, 3, 4]
            curr_kps_xy = curr_kps_xy[remap]
            curr_depths = curr_depths[remap]
            print(f"Sample {i}: Flipped!")
        else:
            print(f"Sample {i}: Not flipped.")
            
        # Draw keypoints
        # graphics.draw_annotation_on_image expects (9, 3) keypoints
        viz_kps = np.column_stack((curr_kps_xy, curr_depths))
        
        # graphics.draw_annotation_on_image uses pixel coordinates if we are not careful
        # Wait, the graphics script expects normalized UV coordinates if it does the multiplication internaly
        # but it uses image.shape to denormalize.
        # Let's check objectron/dataset/graphics.py again.
        # Line 32: h, w, _ = image.shape
        # Line 38: np.multiply(keypoint, np.asarray([w, h, 1.], np.float32)).astype(int)
        # So it expects normalized coords.
        
        # Albumentations with Resize(224, 224) and normalized keypoints:
        # If we pass normalized coords [0, 1] to A.Resize, they remain normalized.
        
        # Convert RGB to BGR for OpenCV
        frame_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        annotated_frame = graphics.draw_annotation_on_image(frame_bgr, viz_kps, [9])
        
        out_path = os.path.join(output_dir, f'aug_{i}.jpg')
        cv2.imwrite(out_path, annotated_frame)
        print(f"Saved {out_path}")

if __name__ == "__main__":
    visualize_augments()

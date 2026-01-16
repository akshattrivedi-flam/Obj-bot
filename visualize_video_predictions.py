import os
import sys
import cv2
import torch
import numpy as np
import json
from PIL import Image
from torchvision import transforms
from tqdm import tqdm

# Add project root to sys.path
sys.path.append(os.getcwd())

from train_overfit import BottleModel, get_transforms
from objectron.dataset import graphics

def generate_video(model_path, data_dir='bottle_data', output_path='output/overfit_results.mp4'):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load model
    model = BottleModel().to(device)
    state_dict = torch.load(model_path, map_location=device)
    
    # Handle DataParallel prefix if present (though new model doesn't use it)
    from collections import OrderedDict
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        if k.startswith('module.'):
            name = k[7:] # remove `module.`
        else:
            name = k
        new_state_dict[name] = v
    
    model.load_state_dict(new_state_dict)
    model.eval()

    # Load annotations to get image list
    anno_file = os.path.join(data_dir, 'annotations.json')
    with open(anno_file, 'r') as f:
        annotations = json.load(f)

    # Transforms (using the same baseline transforms as training)
    transform = get_transforms(augment=False)

    # Video properties
    first_img_path = os.path.join(data_dir, 'images', annotations[0]['image'])
    first_frame = cv2.imread(first_img_path)
    height, width, _ = first_frame.shape
    fps = 30
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    print(f"Generating video from {len(annotations)} frames...")
    
    for anno in tqdm(annotations):
        img_filename = anno['image']
        img_path = os.path.join(data_dir, 'images', img_filename)
        
        # Load and transform image for model
        image = np.array(Image.open(img_path).convert('RGB'))
        # Albumentations expects (H, W, C)
        transformed = transform(image=image, keypoints=[[0.0, 0.0]]) # Dummy keypoints
        input_tensor = transformed['image'].unsqueeze(0).to(device)
        
        # Inference
        with torch.no_grad():
            output = model(input_tensor)
        
        # Reshape output to (9, 3)
        predictions = output.view(9, 3).cpu().numpy()
        
        # Load original image for visualization
        frame = cv2.imread(img_path)
        
        # Draw predictions
        num_keypoints_per_obj = [9]
        annotated_frame = graphics.draw_annotation_on_image(frame, predictions, num_keypoints_per_obj)
        
        # Write to video
        out.write(annotated_frame)

    out.release()
    print(f"Video saved to {output_path}")

if __name__ == "__main__":
    checkpoint = 'bottle_overfit_model.pth'
    generate_video(checkpoint)

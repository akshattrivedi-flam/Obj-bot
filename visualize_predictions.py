import os
import sys
import cv2
import torch
import numpy as np
import json
from PIL import Image
from torchvision import transforms

# Add project root to sys.path
sys.path.append(os.getcwd())

from train_overfit import BottleModel
from objectron.dataset import graphics

def visualize_predictions(model_path, data_dir='bottle_data', output_dir='output'):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load model
    model = BottleModel().to(device)
    state_dict = torch.load(model_path, map_location=device)
    
    # Handle DataParallel prefix if present
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

    # Transforms (same as training)
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Select frames (first, middle, last)
    indices = [0, len(annotations)//2, len(annotations)-1]
    
    for idx in indices:
        anno = annotations[idx]
        img_filename = anno['image']
        img_path = os.path.join(data_dir, 'images', img_filename)
        
        # Load and transform image for model
        image_pil = Image.open(img_path).convert('RGB')
        input_tensor = transform(image_pil).unsqueeze(0).to(device)
        
        # Inference
        with torch.no_grad():
            output = model(input_tensor)
        
        # Reshape output to (1, 9, 3) -> (9, 3)
        predictions = output.view(9, 3).cpu().numpy()
        
        # Load original image for visualization
        frame = cv2.imread(img_path)
        
        # predicted keypoints are 9 x 3 (x, y, depth)
        # the graphics.draw_annotation_on_image expects a flat list of (x, y, depth)
        # but wait, the graphics function split it using num_keypoints
        num_keypoints_per_obj = [9]
        
        # Draw predictions
        annotated_frame = graphics.draw_annotation_on_image(frame, predictions, num_keypoints_per_obj)
        
        # Save results
        out_path = os.path.join(output_dir, f'pred_{img_filename}')
        cv2.imwrite(out_path, annotated_frame)
        print(f"Saved prediction visualization to {out_path}")

if __name__ == "__main__":
    visualize_predictions('bottle_overfit_model.pth')

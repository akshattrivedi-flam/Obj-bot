import cv2
import numpy as np
import tensorflow as tf
import os
from tqdm import tqdm
from PIL import Image
from objectron.dataset import graphics

def get_tflite_predictions(interpreter, image):
    # Prepare input
    # image is (224, 224, 3) from PIL, already normalized
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    
    # Add batch dimension and convert to float32
    input_data = np.expand_dims(image.astype(np.float32), axis=0)
    # TFLite expects (Batch, Channel, Height, Width) based on our PyTorch export
    # PyTorch: (1, 3, 224, 224). Our converter preserved this.
    
    interpreter.set_tensor(input_details[0]['index'], input_data)
    interpreter.invoke()
    
    output_data = interpreter.get_tensor(output_details[0]['index'])
    return output_data.reshape(9, 3)

def generate_video_tflite(data_dir='bottle_data', output_path='output/overfit_results_tflite.mp4'):
    if not os.path.exists('output'):
        os.makedirs('output')

    model_path = 'bottle_model.tflite'
    print(f"Loading TFLite model: {model_path}")
    interpreter = tf.lite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()

    # Load annotations to get image list and order
    anno_file = os.path.join(data_dir, 'annotations.json')
    import json
    with open(anno_file, 'r') as f:
        annotations = json.load(f)
    
    # Get first image to set up video writer
    first_img_path = os.path.join(data_dir, 'images', annotations[0]['image'])
    first_img = cv2.imread(first_img_path)
    h, w, _ = first_img.shape
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, 30.0, (w, h))

    # Normalization parameters (must match training)
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])

    print(f"Generating video from {len(annotations)} frames...")
    for anno in tqdm(annotations):
        img_filename = anno['image']
        img_path = os.path.join(data_dir, 'images', img_filename)
        frame = cv2.imread(img_path)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Preprocess frame exactly as in training
        img_resized = cv2.resize(frame_rgb, (224, 224))
        img_norm = (img_resized / 255.0 - mean) / std
        img_final = img_norm.transpose(2, 0, 1)
        
        # Inference
        predictions = get_tflite_predictions(interpreter, img_final)
        
        # Draw predictions
        annotated_frame = graphics.draw_annotation_on_image(frame.copy(), predictions, [9])
        
        out.write(annotated_frame)

    out.release()
    print(f"Video saved to {output_path}")

if __name__ == "__main__":
    # Ensure CUDA is disabled for TFLite stability in this env
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    generate_video_tflite()

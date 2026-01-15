import os
import sys
import cv2
import numpy as np

# Add the project root to sys.path
sys.path.append(os.getcwd())

from objectron.schema import annotation_data_pb2 as annotation_protocol
from objectron.dataset import graphics

def visualize_bottle(video_file, pbdata_file, output_dir='output'):
    if not os.path.exists(video_file) or not os.path.exists(pbdata_file):
        print("Error: Video or annotation file missing.")
        return

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Load annotations
    sequence = annotation_protocol.Sequence()
    with open(pbdata_file, 'rb') as f:
        sequence.ParseFromString(f.read())

    # Open video
    cap = cv2.VideoCapture(video_file)
    if not cap.isOpened():
        print("Error: Could not open video.")
        return

    # Select a few frames to visualize (e.g., 0, 50, 100, 150, 200)
    total_frames = len(sequence.frame_annotations)
    frame_indices = [0, total_frames // 4, total_frames // 2, (3 * total_frames) // 4, total_frames - 1]

    for idx in frame_indices:
        frame_anno = sequence.frame_annotations[idx]
        
        # Set video to correct frame
        # Note: frame_id in annotation might not strictly match CV2 frame index if there are drops,
        # but for these samples they usually align or we can use the timestamp.
        # We'll use the frame_id from annotation as a proxy for index here.
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if not ret:
            print(f"Warning: Could not read frame {idx}")
            continue

        # Prepare keypoints for drawing
        # graphics.draw_annotation_on_image expects a flat list of (x, y, depth)
        all_keypoints = []
        num_keypoints_per_obj = []
        
        for anno in frame_anno.annotations:
            num_kp = len(anno.keypoints)
            num_keypoints_per_obj.append(num_kp)
            for kp in anno.keypoints:
                all_keypoints.append([kp.point_2d.x, kp.point_2d.y, kp.point_2d.depth])
        
        if all_keypoints:
            # Convert to numpy array
            all_keypoints = np.array(all_keypoints, dtype=np.float32)
            
            # Draw on frame
            # The draw_annotation_on_image function modifies the image in-place
            annotated_frame = graphics.draw_annotation_on_image(frame, all_keypoints, num_keypoints_per_obj)
            
            # Save frame
            out_path = os.path.join(output_dir, f'bottle_frame_{idx:03d}.png')
            cv2.imwrite(out_path, annotated_frame)
            print(f"Saved annotated frame to {out_path}")

    cap.release()
    print("Visualization complete. Check the 'output' directory.")

if __name__ == "__main__":
    visualize_bottle('bottle_sample.MOV', 'bottle_sample.pbdata')

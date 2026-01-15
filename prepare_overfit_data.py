import os
import cv2
import sys
import numpy as np
import json

# Add project root to sys.path
sys.path.append(os.getcwd())

from objectron.schema import annotation_data_pb2 as annotation_protocol

def prepare_data(video_file, pbdata_file, data_dir='bottle_data'):
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
    
    img_dir = os.path.join(data_dir, 'images')
    if not os.path.exists(img_dir):
        os.makedirs(img_dir)

    # Load annotations
    sequence = annotation_protocol.Sequence()
    with open(pbdata_file, 'rb') as f:
        sequence.ParseFromString(f.read())

    # Open video
    cap = cv2.VideoCapture(video_file)
    if not cap.isOpened():
        print("Error: Could not open video.")
        return

    annotations_list = []
    
    print(f"Extracting {len(sequence.frame_annotations)} frames...")
    for idx, frame_anno in enumerate(sequence.frame_annotations):
        # Set video to correct frame
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if not ret:
            print(f"Warning: Could not read frame {idx}")
            continue

        # Save frame
        img_filename = f'frame_{idx:03d}.jpg'
        img_path = os.path.join(img_dir, img_filename)
        cv2.imwrite(img_path, frame)

        # Extract keypoints (assuming 1 object per frame for this sample)
        if len(frame_anno.annotations) > 0:
            anno = frame_anno.annotations[0]
            kp_2d = [[kp.point_2d.x, kp.point_2d.y, kp.point_2d.depth] for kp in anno.keypoints]
            kp_3d = [[kp.point_3d.x, kp.point_3d.y, kp.point_3d.z] for kp in anno.keypoints]
            visibility = anno.visibility
        else:
            kp_2d = []
            kp_3d = []
            visibility = 0.0

        annotations_list.append({
            'frame_id': idx,
            'image': img_filename,
            'keypoints_2d': kp_2d,
            'keypoints_3d': kp_3d,
            'visibility': visibility
        })

    cap.release()

    # Save annotations to JSON
    with open(os.path.join(data_dir, 'annotations.json'), 'w') as f:
        json.dump(annotations_list, f)

    print(f"Data preparation complete. Saved to {data_dir}")

if __name__ == "__main__":
    prepare_data('bottle_sample.MOV', 'bottle_sample.pbdata')

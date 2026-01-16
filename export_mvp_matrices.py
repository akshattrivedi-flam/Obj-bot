import os
import sys
import numpy as np
import json
from tqdm import tqdm

# Add the project root to sys.path
sys.path.append(os.getcwd())

from objectron.schema import annotation_data_pb2 as annotation_protocol

def export_mvp_data(pbdata_file, output_json='output/bottle_mvp_data.json'):
    if not os.path.exists(pbdata_file):
        print(f"Error: {pbdata_file} not found.")
        return

    if not os.path.exists('output'):
        os.makedirs('output')

    print(f"Loading annotations from {pbdata_file}...")
    sequence = annotation_protocol.Sequence()
    with open(pbdata_file, 'rb') as f:
        sequence.ParseFromString(f.read())

    # Static Object Data (Model Matrix components)
    if len(sequence.objects) == 0:
        print("Error: No objects found in sequence.")
        return
        
    obj = sequence.objects[0]
    R = np.array(obj.rotation).reshape(3, 3)
    T = np.array(obj.translation)
    S = np.array(obj.scale)
    
    # Construct Model Matrix (M) - Standard Column-Major internal for math
    # M = T * R * S
    M = np.eye(4)
    # R @ diag(S) handles the scaling in the rotation basis
    M[:3, :3] = R @ np.diag(S)
    M[:3, 3] = T
    
    all_mvp_data = []

    print(f"Exporting MVP matrices for {len(sequence.frame_annotations)} frames...")
    for frame_anno in tqdm(sequence.frame_annotations):
        camera = frame_anno.camera
        
        # In Objectron, matrices in protobuf are row-major
        P = np.array(camera.projection_matrix).reshape(4, 4)
        V = np.array(camera.view_matrix).reshape(4, 4)
        
        # MVP = P @ V @ M
        MVP = P @ V @ M
        
        all_mvp_data.append({
            'frame_id': frame_anno.frame_id,
            'timestamp': frame_anno.timestamp,
            'mvp': MVP.tolist(),
            'view': V.tolist(),
            'projection': P.tolist(),
            'model': M.tolist()
        })

    with open(output_json, 'w') as f:
        json.dump(all_mvp_data, f)
    
    print(f"MVP data exported to {output_json}")

if __name__ == "__main__":
    export_mvp_data('bottle_sample.pbdata')

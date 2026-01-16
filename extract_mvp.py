import os
import sys
import numpy as np
import json

# Add the project root to sys.path
sys.path.append(os.getcwd())

from objectron.schema import annotation_data_pb2 as annotation_protocol

def get_mvp_matrix(pbdata_file, frame_idx=0):
    sequence = annotation_protocol.Sequence()
    with open(pbdata_file, 'rb') as f:
        sequence.ParseFromString(f.read())

    frame = sequence.frame_annotations[frame_idx]
    camera = frame.camera
    obj = sequence.objects[0]
    
    # ARCore/Objectron matrices are usually column-major
    P = np.array(camera.projection_matrix).reshape(4, 4, order='F')
    V = np.array(camera.view_matrix).reshape(4, 4, order='F')
    
    # Model Matrix
    R = np.array(obj.rotation).reshape(3, 3)
    T = np.array(obj.translation)
    S = np.array(obj.scale)
    
    M = np.eye(4)
    M[:3, :3] = R @ np.diag(S) # Scale before rotation or during?
    # In Objectron, rotation is applied to scaled axis-aligned vertices.
    # So M = [ R | T ] @ [ S | 0 ] ? No, usually M = [ R*S | T ]
    M[:3, 3] = T
    
    # MVP = P @ V @ M
    MVP = P @ V @ M
    
    return MVP, P, V, M

def verify(mvp, target_2d):
    # Local center is [0, 0, 0, 1]
    p_local = np.array([0, 0, 0, 1])
    p_clip = mvp @ p_local
    p_ndc = p_clip[:2] / p_clip[3]
    
    # NDC to Screen [0, 1]
    # ARCore/Objectron screen space: x in [0, 1], y in [0, 1]
    # Note: clip space to NDC is (x/w, y/w)
    # Then NDC to [0, 1] is usually (p_ndc + 1) / 2
    # But wait, ARCore projection might already be set up for a specific convention.
    
    # Let's see if p_clip matches directly
    print(f"Projected p_clip: {p_clip}")
    print(f"Target 2D (from JSON): {target_2d}")
    
    # In Objectron, the 'd' in [x, y, d] is often the depth in camera space.
    # Let's check if p_clip[3] matches target_2d[2]
    print(f"p_clip[3]: {p_clip[3]}, target depth: {target_2d[2]}")

if __name__ == "__main__":
    mvp, p, v, m = get_mvp_matrix('bottle_sample.pbdata', 0)
    
    with open('bottle_data/annotations.json', 'r') as f:
        annotations = json.load(f)
    target_2d = annotations[0]['keypoints_2d'][0]
    
    verify(mvp, target_2d)

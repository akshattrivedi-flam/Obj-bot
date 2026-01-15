import os
import sys
import numpy as np

# Add the project root to sys.path to import objectron
sys.path.append(os.getcwd())

from objectron.schema import annotation_data_pb2 as annotation_protocol
from objectron.schema import object_pb2 as object_protocol

def inspect_sample(pbdata_file):
    if not os.path.exists(pbdata_file):
        print(f"Error: {pbdata_file} not found.")
        return

    # Read the sequence from the pbdata file
    sequence = annotation_protocol.Sequence()
    with open(pbdata_file, 'rb') as f:
        sequence.ParseFromString(f.read())

    print(f"--- Inspection of {pbdata_file} ---")
    print(f"Number of objects in sequence: {len(sequence.objects)}")
    for i, obj in enumerate(sequence.objects):
        print(f"  Object {i}:")
        print(f"    ID: {obj.id}")
        print(f"    Category: {obj.category}")
        print(f"    Type: {object_protocol.Object.Type.Name(obj.type)}")
        print(f"    Scale: {obj.scale}")

    print(f"Number of annotated frames: {len(sequence.frame_annotations)}")
    
    if len(sequence.frame_annotations) > 0:
        first_frame = sequence.frame_annotations[0]
        print(f"First frame ID: {first_frame.frame_id}")
        print(f"Timestamp: {first_frame.timestamp}")
        print(f"Number of annotations in first frame: {len(first_frame.annotations)}")
        
        if len(first_frame.annotations) > 0:
            first_annotation = first_frame.annotations[0]
            print(f"  First annotation object ID: {first_annotation.object_id}")
            print(f"  Number of keypoints: {len(first_annotation.keypoints)}")
            print(f"  Visibility: {first_annotation.visibility}")

if __name__ == "__main__":
    inspect_sample('bottle_sample.pbdata')

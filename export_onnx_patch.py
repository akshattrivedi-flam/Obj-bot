import sys
import ml_dtypes
import numpy as np

# Monkey-patch ml_dtypes to satisfy new ONNX
if not hasattr(ml_dtypes, 'float4_e2m1fn'):
    # Define it as a dummy float8 type if float4 is missing
    ml_dtypes.float4_e2m1fn = getattr(ml_dtypes, 'float8_e4m3fn', np.float32)
if not hasattr(ml_dtypes, 'float4_e2m1fn_uz'):
    ml_dtypes.float4_e2m1fn_uz = getattr(ml_dtypes, 'float8_e4m3fn', np.float32)
if not hasattr(ml_dtypes, 'float8_e4m3fnuz'):
    ml_dtypes.float8_e4m3fnuz = getattr(ml_dtypes, 'float8_e4m3fn', np.float32)
if not hasattr(ml_dtypes, 'float8_e5m2fnuz'):
    ml_dtypes.float8_e5m2fnuz = getattr(ml_dtypes, 'float8_e5m2', np.float32)

import torch
import torchvision.models as models
import torch.nn as nn
import os

class BottleModel(nn.Module):
    def __init__(self, num_keypoints=9):
        super(BottleModel, self).__init__()
        mv3 = models.mobilenet_v3_small(weights=None)
        self.features = mv3.features
        self.avgpool = mv3.avgpool
        last_channel = 576
        self.classifier = nn.Sequential(
            nn.Linear(last_channel, 1024),
            nn.Hardswish(inplace=True),
            nn.Dropout(p=0.2, inplace=True),
            nn.Linear(1024, num_keypoints * 3)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

def export_onnx_patched(model_path='bottle_overfit_model.pth', output_path='bottle_model_final.onnx'):
    print(f"Loading PyTorch model from {model_path}...")
    model = BottleModel()
    if not os.path.exists(model_path):
        print(f"ERROR: {model_path} not found.")
        return

    checkpoint = torch.load(model_path, map_location='cpu')
    from collections import OrderedDict
    new_state_dict = OrderedDict()
    for k, v in checkpoint.items():
        name = k[7:] if k.startswith('module.') else k
        new_state_dict[name] = v
        
    model.load_state_dict(new_state_dict)
    model.eval()

    dummy_input = torch.randn(1, 3, 224, 224)
    
    print(f"Exporting to ONNX: {output_path}...")
    torch.onnx.export(model, 
                      dummy_input, 
                      output_path, 
                      export_params=True,
                      opset_version=12,
                      do_constant_folding=True,
                      input_names = ['input'],
                      output_names = ['output'],
                      dynamic_axes={'input' : {0 : 'batch_size'},
                                    'output' : {0 : 'batch_size'}})
    
    print("Successfully exported to ONNX.")
    size_mb = os.path.getsize(output_path) / (1024 * 1024)
    print(f"ONNX model size: {size_mb:.2f} MB")

if __name__ == "__main__":
    export_onnx_patched()

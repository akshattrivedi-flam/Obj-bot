import torch
import torchvision.models as models
import torch.nn as nn
import os

class BottleModel(nn.Module):
    def __init__(self, num_keypoints=9):
        super(BottleModel, self).__init__()
        # MobileNetV3 Small backbone
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

def export_onnx(model_path='bottle_overfit_model.pth', output_path='bottle_model_stable.onnx'):
    print(f"Loading PyTorch model from {model_path}...")
    model = BottleModel()
    # Check if file exists
    if not os.path.exists(model_path):
        print(f"ERROR: {model_path} not found.")
        return

    state_dict = torch.load(model_path, map_location='cpu')
    
    # Handle DataParallel prefix
    from collections import OrderedDict
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = k[7:] if k.startswith('module.') else k
        new_state_dict[name] = v
        
    model.load_state_dict(new_state_dict)
    model.eval()

    sample_input = torch.randn(1, 3, 224, 224)
    print(f"Exporting to ONNX: {output_path}...")
    
    # Using opset 11 or 12 for better TFLite compatibility
    torch.onnx.export(model, sample_input, output_path, 
                      input_names=['input'], output_names=['output'],
                      opset_version=12,
                      do_constant_folding=True)
    
    print("Successfully exported to ONNX.")
    size_mb = os.path.getsize(output_path) / (1024 * 1024)
    print(f"ONNX model size: {size_mb:.2f} MB")

if __name__ == "__main__":
    export_onnx()

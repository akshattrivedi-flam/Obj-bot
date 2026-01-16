import torch
import torchvision.models as models
import torch.nn as nn
import ai_edge_torch
import os
from collections import OrderedDict

# Define the model architecture exactly as used for training
class BottleModel(nn.Module):
    def __init__(self, num_keypoints=9):
        super(BottleModel, self).__init__()
        # Use MobileNetV3 Small backbone
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

def export_tflite(model_path='bottle_overfit_model.pth', output_path='bottle_model.tflite'):
    print(f"Loading PyTorch model from {model_path}...")
    model = BottleModel()
    if not os.path.exists(model_path):
        print(f"ERROR: {model_path} not found.")
        return

    # Load state dict
    state_dict = torch.load(model_path, map_location='cpu')
    
    # Clean state dict (remove module prefix if present)
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = k[7:] if k.startswith('module.') else k
        new_state_dict[name] = v
    
    model.load_state_dict(new_state_dict)
    model.eval()

    # Define dummy input (Batch, Channel, Height, Width)
    sample_input = torch.randn(1, 3, 224, 224)

    print("Converting to TFLite using ai_edge_torch...")
    try:
        # Convert and serialize
        edge_model = ai_edge_torch.convert(model, (sample_input,))
        edge_model.export(output_path)
        
        print(f"Successfully exported TFLite model to {output_path}")
        size_mb = os.path.getsize(output_path) / (1024 * 1024)
        print(f"Final TFLite model size: {size_mb:.2f} MB")
    except Exception as e:
        print(f"Conversion failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    export_tflite()

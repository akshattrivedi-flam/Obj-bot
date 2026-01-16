import torch
import torchvision.models as models
import torch.nn as nn
import os

class BottleModel(nn.Module):
    def __init__(self, num_keypoints=9):
        super(BottleModel, self).__init__()
        mv3 = models.mobilenet_v3_small(pretrained=False)
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

def export_onnx(model_path='bottle_overfit_model_epoch_200.pth', output_path='bottle_model.onnx'):
    print(f"Loading PyTorch model from {model_path}...")
    model = BottleModel()
    state_dict = torch.load(model_path, map_location='cpu')
    model.load_state_dict(state_dict)
    model.eval()

    sample_input = torch.randn(1, 3, 224, 224)
    print(f"Exporting to ONNX: {output_path}...")
    torch.onnx.export(model, sample_input, output_path, 
                      input_names=['input'], output_names=['output'],
                      opset_version=12)
    print("Successfully exported to ONNX.")
    
    size_mb = os.path.getsize(output_path) / (1024 * 1024)
    print(f"ONNX model size: {size_mb:.2f} MB")

if __name__ == "__main__":
    export_onnx()

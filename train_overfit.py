import os
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
from PIL import Image
import numpy as np

import albumentations as A
from albumentations.pytorch import ToTensorV2

# Configuration
DATA_DIR = 'bottle_data'
IMG_DIR = os.path.join(DATA_DIR, 'images')
ANNO_FILE = os.path.join(DATA_DIR, 'annotations.json')
EPOCHS = 1000
BATCH_SIZE = 64 # Reduced for stability
LEARNING_RATE = 1e-4
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
USE_AUGMENTATION = False

class BottleDataset(Dataset):
    def __init__(self, anno_file, img_dir, transform=None):
        with open(anno_file, 'r') as f:
            self.annotations = json.load(f)
        self.img_dir = img_dir
        self.transform = transform
        self.images_cache = {}
        print("Caching raw images in memory...")
        for anno in self.annotations:
            img_path = os.path.join(self.img_dir, anno['image'])
            image = np.array(Image.open(img_path).convert('RGB'))
            self.images_cache[anno['image']] = image
        print(f"Cached {len(self.images_cache)} images.")

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):
        anno = self.annotations[idx]
        image = self.images_cache[anno['image']]
        h, w, _ = image.shape
        
        # Keypoints are stored as normalized [0, 1]
        kps_2d = np.array(anno['keypoints_2d'], dtype=np.float32)
        kps_xy = kps_2d[:, :2]
        depths = kps_2d[:, 2]
        
        # CRITICAL FIX: Scale to pixel coordinates before Albumentations
        kps_xy_pixels = kps_xy * np.array([w, h], dtype=np.float32)
        
        if self.transform:
            transformed = self.transform(image=image, keypoints=kps_xy_pixels)
            image = transformed['image']
            kps_xy_pixels = np.array(transformed['keypoints'], dtype=np.float32)
            
            # Re-normalize to [0, 1] based on the NEW image size (after transforms/Resize)
            # Albumentations transforms include Resize(224, 224)
            new_h, new_w = image.shape[1], image.shape[2] # ToTensorV2 makes it (C, H, W)
            kps_xy = kps_xy_pixels / np.array([new_w, new_h], dtype=np.float32)
        
        target = np.column_stack((kps_xy, depths)).flatten()
        return image, torch.tensor(target, dtype=torch.float32)

class BottleModel(nn.Module):
    def __init__(self, num_keypoints=9):
        super(BottleModel, self).__init__()
        # Use MobileNetV3 Small as backbone, keeping it separate for stable DataParallel
        mv3 = models.mobilenet_v3_small(pretrained=True)
        # Extract features (everything except the classifier)
        self.features = mv3.features
        self.avgpool = mv3.avgpool
        # New classifier head
        last_channel = 576 # Feature output channel for mobilenet_v3_small
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

def get_model_size(model):
    param_size = 0
    for param in model.parameters():
        param_size += param.nelement() * param.element_size()
    buffer_size = 0
    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()
    size_all_mb = (param_size + buffer_size) / 1024**2
    return size_all_mb

def get_transforms(augment=False):
    # Plain baseline transforms
    return A.Compose([
        A.Resize(224, 224),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2()
    ], keypoint_params=A.KeypointParams(format='xy', remove_invisible=False))

def train():
    # Transforms
    transform = get_transforms(augment=USE_AUGMENTATION)

    dataset = BottleDataset(ANNO_FILE, IMG_DIR, transform=transform)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    model = BottleModel().to(DEVICE)
    print(f"Estimated model size: {get_model_size(model):.2f} MB")
    
    # Using single GPU for stability as DataParallel was causing SegFaults
    # With 252 images and MobileNetV3, single GPU is more than enough.
    if torch.cuda.device_count() > 1:
        print(f"Detected {torch.cuda.device_count()} GPUs. Using primary device {DEVICE} for stability.")
        
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    print("Starting fresh baseline training with MobileNetV3 Small...")
    print(f"Total epochs: {EPOCHS}, Device: {DEVICE}")
    
    for epoch in range(EPOCHS):
        model.train()
        running_loss = 0.0
        for images, targets in dataloader:
            images, targets = images.to(DEVICE), targets.to(DEVICE)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        
        avg_loss = running_loss / len(dataloader)
        print(f"Epoch {epoch+1}/{EPOCHS}, Loss: {avg_loss:6f}")
        
        if (epoch + 1) % 20 == 0:
            checkpoint_path = f'bottle_overfit_model_epoch_{epoch+1}.pth'
            torch.save(model.state_dict(), checkpoint_path)
            print(f"Saved checkpoint to {checkpoint_path}")

    # Save the overfitted model
    torch.save(model.state_dict(), 'bottle_overfit_model.pth')
    print("Training complete. Model saved as bottle_overfit_model.pth")

if __name__ == "__main__":
    train()

import os
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
from PIL import Image
import numpy as np

# Configuration
DATA_DIR = 'bottle_data'
IMG_DIR = os.path.join(DATA_DIR, 'images')
ANNO_FILE = os.path.join(DATA_DIR, 'annotations.json')
EPOCHS = 1000
BATCH_SIZE = 16
LEARNING_RATE = 1e-4
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class BottleDataset(Dataset):
    def __init__(self, anno_file, img_dir, transform=None):
        with open(anno_file, 'r') as f:
            self.annotations = json.load(f)
        self.img_dir = img_dir
        self.transform = transform
        self.cache = {}
        print("Caching images in memory...")
        for anno in self.annotations:
            img_path = os.path.join(self.img_dir, anno['image'])
            image = Image.open(img_path).convert('RGB')
            if self.transform:
                image = self.transform(image)
            self.cache[anno['image']] = (image, torch.tensor(np.array(anno['keypoints_2d'], dtype=np.float32).flatten(), dtype=torch.float32))
        print(f"Cached {len(self.cache)} images.")

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):
        anno = self.annotations[idx]
        return self.cache[anno['image']]

class BottleModel(nn.Module):
    def __init__(self, num_keypoints=9):
        super(BottleModel, self).__init__()
        # Using a pretrained ResNet18
        self.backbone = models.resnet18(pretrained=True)
        # Replace the last fully connected layer
        # 9 keypoints * 3 (x, y, depth) = 27 outputs
        num_ftrs = self.backbone.fc.in_features
        self.backbone.fc = nn.Linear(num_ftrs, num_keypoints * 3)

    def forward(self, x):
        return self.backbone(x)

def train():
    # Transforms: standard ImageNet normalization
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    dataset = BottleDataset(ANNO_FILE, IMG_DIR, transform=transform)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    model = BottleModel().to(DEVICE)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    print(f"Starting training for {EPOCHS} epochs on {DEVICE}...")
    
    for epoch in range(EPOCHS):
        model.train()
        running_loss = 0.0
        for images, keypoints in dataloader:
            images = images.to(DEVICE)
            keypoints = keypoints.to(DEVICE)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, keypoints)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * images.size(0)

        epoch_loss = running_loss / len(dataset)
        print(f"Epoch {epoch+1}/{EPOCHS}, Loss: {epoch_loss:.6f}")

    # Save the overfitted model
    torch.save(model.state_dict(), 'bottle_overfit_model.pth')
    print("Training complete. Model saved as bottle_overfit_model.pth")

if __name__ == "__main__":
    train()

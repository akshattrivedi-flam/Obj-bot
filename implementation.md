# Implementation Plan: Overfitting Objectron on Dual RTX 4090

This document outlines the steps to migrate and run the overfitting training for the bottle dataset on your high-performance GPU system.

## 1. Environment Setup
Ensure your new system has the necessary drivers and libraries:
- **NVIDIA Drivers**: Latest compatible drivers for RTX 4090.
- **CUDA Toolkit**: 11.8 or 12.x recommended.
- **Python Libraries**:
  ```bash
  pip install torch torchvision numpy opencv-python Pillow
  ```

## 2. Data Preparation
Regenerate the training data on the local machine:
1. Ensure `bottle_sample.MOV` and `bottle_sample.pbdata` are in the root directory.
2. Run the extraction script:
   ```bash
   python3 prepare_overfit_data.py
   ```
   This will create the `bottle_data/` folder with 252 frames and `annotations.json`.

## 3. Training Optimization for Dual 4090s
The current `train_overfit.py` is configured for CPU. Update it for your dual GPUs:

### Update Device Selection
In `train_overfit.py`, change:
```python
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
```

### Enable Multi-GPU Support
In the `train()` function of `train_overfit.py`, modify the model initialization:
```python
model = BottleModel().to(DEVICE)
if torch.cuda.device_count() > 1:
    print(f"Using {torch.cuda.device_count()} GPUs!")
    model = nn.DataParallel(model)
```

### Scale Batch Size
Since you have 48GB of VRAM (24GB x 2), you can significantly increase the batch size to speed up training:
```python
BATCH_SIZE = 128 # Or higher
```

## 4. Execution
Run the training:
```bash
python3 train_overfit.py
```

## 5. Verification
After training completes (1000 epochs):
1. Check for the generated `bottle_overfit_model.pth`.
2. Monitor the loss; it should drop close to zero (e.g., < 0.0001).
3. (Optional) Create a validation script to project the predicted 3D boxes back onto the video frames to visually confirm the overfit.

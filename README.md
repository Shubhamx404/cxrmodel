# Triage Challenge - Inference Solution

## 📋 Project Overview

This is an automated Chest X-Ray (CXR) triage classification system designed to predict severity levels for chest radiographs. The solution uses a ResNet-18 deep learning model trained on CXR images to classify each image into one of **4 triage categories** (classes 0-3).


## 🎯 Task Description

### Objective
Classify chest X-ray images into 4 severity triage categories (p0, p1, p2, p3) representing different urgency levels for patient assessment.

### Input
- **Images**: PNG/JPG chest X-ray images (224×224 resolution after preprocessing)
- **Metadata**: CSV containing image identifiers and clinical metadata (age, gender, view position, etc.)
- **Checkpoint**: Pre-trained ResNet-18 model weights (PyTorch .pt format)

### Output
- **Predictions CSV**: Contains image_id and probability predictions for each triage class
  - Columns: `image_id`, `p0`, `p1`, `p2`, `p3`
  - Each row has probabilities that sum to 1.0

## 📁 Directory Structure

```
yolov11/
│
├── predict.py                 # ✅ Main inference script
├── requirements.txt           # Python dependencies
├── README.md                  # This file
│
├── models/
│   └── best.pt                # Pre-trained ResNet-18 checkpoint
│
├── test_public/
│   └── *.png                  # Test images (CXR radiographs)
│
├── metadata.csv               # Image metadata (image_id, age, gender, etc.)
│
├── outputs/
│   └── test_public_predictions.csv  # Model predictions
│
└── .venv/                     # Python virtual environment
```

## 🔧 Installation & Setup

### Prerequisites
- Python 3.8+
- PyTorch 2.0+ (CPU/GPU compatible)
- pip package manager

### 1. Create Virtual Environment
```bash
# Navigate to project directory
cd yolov11

# Create virtual environment
python -m venv .venv

# Activate virtual environment
# On Windows:
.venv\Scripts\activate
# On macOS/Linux:
source .venv/bin/activate
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

**Required packages:**
- `torch` - Deep learning framework
- `torchvision` - Computer vision utilities
- `numpy` - Numerical computing
- `pandas` - Data manipulation
- `pillow` - Image processing
- `tqdm` - Progress bars

## 🚀 Usage

### Command Line Interface (CLI)

Run inference on test images:

```bash
python predict.py \
  --images_dir test_public \
  --metadata_csv metadata.csv \
  --checkpoint models/best.pt \
  --out_csv outputs/test_public_predictions.csv
```

### Arguments

| Argument | Type | Required | Description |
|----------|------|----------|-------------|
| `--images_dir` | str | ✅ Yes | Path to directory containing CXR images |
| `--metadata_csv` | str | ✅ Yes | Path to metadata CSV file with `image_id` column |
| `--checkpoint` | str | ✅ Yes | Path to pre-trained model checkpoint (.pt file) |
| `--out_csv` | str | ✅ Yes | Output path for predictions CSV |

### Example Output

```
Running inference: 100%|██████████| 8000/8000 [12:34<00:00, 10.63it/s]
✅ Predictions written to: outputs/test_public_predictions.csv
Total predictions: 8000
```

**Predictions CSV format:**
```csv
image_id,p0,p1,p2,p3
00000177_000.png,0.92,0.05,0.02,0.01
00000177_001.png,0.15,0.72,0.10,0.03
00000974_000.png,0.03,0.08,0.85,0.04
```

Where:
- `p0` - Probability of class 0 (Normal/Low urgency)
- `p1` - Probability of class 1 (Mild/Moderate urgency)
- `p2` - Probability of class 2 (Urgent)
- `p3` - Probability of class 3 (Critical/High urgency)

## 📊 Model Architecture

### ResNet-18 with Custom Classification Head

```python
# Base: Pre-trained ResNet-18
model = models.resnet18(weights=None)

# Custom FC layer for 4-class classification
model.fc = nn.Linear(512, 4)  # 512 input features → 4 output classes
```

### Key Features
- **Architecture**: ResNet-18 (18 layers)
- **Input Size**: 224×224 RGB images
- **Output**: 4-class logits (softmax → probabilities)
- **Device**: CPU (optimized for inference)
- **Seed**: 42 (reproducible results)

## 🖼️ Image Processing Pipeline

### Preprocessing Steps
1. **Load**: Read image as RGB (PIL)
2. **Resize**: Scale to 224×224 pixels
3. **Normalize**: Standard ImageNet normalization
   - Mean: [0.485, 0.456, 0.406]
   - Std: [0.229, 0.224, 0.225]
4. **Convert**: Tensor format for PyTorch
5. **Batch**: Add batch dimension for inference

### Code
```python
VAL_TRANSFORM = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    ),
])
```

## 📝 Metadata Format

The `metadata.csv` must contain at minimum:
- `image_id` - Image filename (required)
- `followup` - Follow-up study index (0, 1, 2, ...)
- `age` - Patient age in years
- `gender` - 'M' or 'F'
- `view_position` - X-ray view (AP, PA, etc.)
- `width` - Image width in pixels
- `height` - Image height in pixels
- `pixel_spacing_x` - Horizontal pixel spacing (mm)
- `pixel_spacing_y` - Vertical pixel spacing (mm)

**Example:**
```csv
image_id,followup,age,gender,view_position,width,height,pixel_spacing_x,pixel_spacing_y
00000177_000.png,0,55,F,AP,2500,2048,0.168,0.168
00000177_001.png,1,55,F,AP,2500,2048,0.168,0.168
```

## 🔍 Implementation Details

### Inference Process

1. **Load Metadata**: Read CSV and extract image IDs
2. **Initialize Model**: Load ResNet-18 and restore checkpoint weights
3. **Process Images**: For each image in metadata:
   - Load from disk
   - Apply preprocessing transforms
   - Forward pass through model
   - Apply softmax → normalize probabilities
4. **Export Results**: Save predictions to CSV

### Error Handling
- ✅ Missing image files are skipped gracefully
- ✅ Corrupted images are caught and logged
- ✅ Invalid probability values are clipped to [0, 1]
- ✅ Probabilities are normalized to sum = 1.0

### Probability Normalization

```python
def normalize_probs(probs: np.ndarray) -> np.ndarray:
    """Safe normalization ensuring probabilities sum to 1.0"""
    probs = np.clip(probs, 0.0, 1.0)  # Clip to valid range
    s = probs.sum()
    if s > 0:
        return probs / s
    # Fallback for zero sum (extremely rare)
    return np.array([0.25, 0.25, 0.25, 0.25], dtype=np.float32)
```

## ⚙️ Configuration

### Performance Settings
- **Device**: CPU (set in code, change to `cuda` if GPU available)
- **Reproducibility**: Fixed seed = 42
- **Num Threads**: 1 (deterministic single-threaded execution)
- **Batch Size**: 1 (process images individually)
- **Model Loading**: Maps to CPU (device='cpu')

### Modification for GPU

To use GPU acceleration, modify line 24 in `predict.py`:

```python
# Change from:
DEVICE = torch.device("cpu")

# To:
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
```

## 📈 Expected Performance

- **Inference Speed**: ~10-15 images/second (CPU)
- **GPU Speed**: ~100-200 images/second (NVIDIA GPU)
- **Memory**: ~500MB (model + batch processing)
- **Output Size**: ~1KB per 100 predictions

## 🐛 Troubleshooting

### Common Issues

**Issue**: `FileNotFoundError: Models not found`
```bash
# Solution: Ensure models/best.pt exists
ls -la models/best.pt
```

**Issue**: `AssertionError: metadata.csv must contain image_id column`
```bash
# Solution: Verify CSV headers
head -1 metadata.csv
```

**Issue**: `CUDA out of memory`
```bash
# Solution: Use CPU or reduce batch size
DEVICE = torch.device("cpu")
```

**Issue**: `PIL.UnidentifiedImageError: cannot identify image file`
```bash
# Solution: Check image format (must be PNG/JPG)
file test_public/*.png
```

## 📚 Dataset Statistics

Based on test_public/ directory:
- **Total Images**: ~8000 CXR radiographs
- **Image Format**: PNG (grayscale CXR images)
- **Image Size**: 2500×2048 pixels (original)
- **Processing**: Resized to 224×224 for model
- **Patient Studies**: ~100 unique patients with multiple follow-ups
- **Follow-up Range**: 0-62 images per patient (longitudinal study)

## 🔐 Reproducibility

All inference is reproducible via:
- Fixed random seed (42)
- Single-threaded execution
- CPU device (deterministic)
- No dropout/augmentation during inference

**Expected Output**: Same probabilities across multiple runs

## 📄 Output CSV Schema

```csv
image_id,p0,p1,p2,p3
00000177_000.png,0.92,0.05,0.02,0.01
00000177_001.png,0.15,0.72,0.10,0.03
...
```

- **image_id** (str): Filename of the CXR image
- **p0** (float): Probability of class 0 [0.0-1.0]
- **p1** (float): Probability of class 1 [0.0-1.0]
- **p2** (float): Probability of class 2 [0.0-1.0]
- **p3** (float): Probability of class 3 [0.0-1.0]

**Constraint**: p0 + p1 + p2 + p3 = 1.0 for each row

## 🎓 References

- **PyTorch**: https://pytorch.org/
- **ResNet**: https://arxiv.org/abs/1512.03385
- **ImageNet Normalization**: Standard mean/std for pre-trained models
- **Medical Imaging**: DICOM→PNG conversion standard

## 📝 License & Attribution

**Challenge**: ImagingIQ IITBHU CXR Triage Challenge v1.4  
**Date**: January 26, 2025  
**Model**: ResNet-18 (PyTorch torchvision)  
**Framework**: PyTorch

## 🤝 Support

For issues or questions:
1. Check [Troubleshooting](#-troubleshooting) section
2. Verify all input files exist and are correctly formatted
3. Ensure all dependencies are installed: `pip list`
4. Check Python version: `python --version`

---

**Last Updated**: January 26, 2025  
**Submission Status**: ✅ Ready for inference

# CLI Reference Guide - ImagingIQ CXR Triage Inference

## Quick Start

```bash
# Activate virtual environment (Windows)
.venv\Scripts\activate

# Run inference with default paths
# Single LINE 

python predict.py --images_dir test_public --metadata_csv metadata.csv --checkpoint models/best.pt --out_csv outputs/test_public_predictions.csv

python predict.py \
  --images_dir test_public \
  --metadata_csv metadata.csv \
  --checkpoint models/best.pt \
  --out_csv outputs/test_public_predictions.csv
```

---

## Complete CLI Documentation

### Command Syntax
```bash
python predict.py --images_dir PATH --metadata_csv PATH --checkpoint PATH --out_csv PATH
```

### Required Arguments

#### `--images_dir` (Directory Path)
- **Type**: String
- **Required**: Yes
- **Description**: Path to directory containing CXR PNG/JPG images
- **Default**: None
- **Example**: `test_public`

```bash
python predict.py --images_dir /path/to/images ...
```

#### `--metadata_csv` (File Path)
- **Type**: String (CSV file path)
- **Required**: Yes
- **Description**: Path to CSV with `image_id` column listing all images to process
- **Default**: None
- **Example**: `metadata.csv`

```bash
python predict.py --metadata_csv data/metadata.csv ...
```

#### `--checkpoint` (File Path)
- **Type**: String (PyTorch .pt file)
- **Required**: Yes
- **Description**: Path to pre-trained ResNet-18 model weights
- **Default**: None
- **Example**: `models/best.pt`

```bash
python predict.py --checkpoint models/best.pt ...
```

#### `--out_csv` (File Path)
- **Type**: String (output CSV file path)
- **Required**: Yes
- **Description**: Output path for predictions CSV
- **Default**: None
- **Example**: `outputs/predictions.csv`

```bash
python predict.py --out_csv results/predictions.csv ...
```

---

## Usage Examples

### Example 1: Standard Inference (Default Paths)
```bash
python predict.py \
  --images_dir test_public \
  --metadata_csv metadata.csv \
  --checkpoint models/best.pt \
  --out_csv outputs/test_public_predictions.csv
```

**Output:**
```
Running inference: 100%|██████████| 8000/8000 [12:34<00:00, 10.63it/s]
Predictions written to: outputs/test_public_predictions.csv
Total predictions: 8000
```

### Example 2: Custom Paths
```bash
python predict.py \
  --images_dir /data/medical_images \
  --metadata_csv /data/cxr_metadata.csv \
  --checkpoint /models/resnet18_checkpoint.pt \
  --out_csv /results/inference_output.csv
```

### Example 3: Relative Paths
```bash
python predict.py \
  --images_dir ./test_data/images \
  --metadata_csv ./test_data/meta.csv \
  --checkpoint ./checkpoints/model_v2.pt \
  --out_csv ./output/predictions_v2.csv
```

### Example 4: Absolute Paths (Windows)
```cmd
python predict.py ^
  --images_dir C:\data\test_images ^
  --metadata_csv C:\data\metadata.csv ^
  --checkpoint C:\models\best.pt ^
  --out_csv C:\results\predictions.csv
```

### Example 5: Absolute Paths (Linux/Mac)
```bash
python predict.py \
  --images_dir /home/user/data/images \
  --metadata_csv /home/user/data/meta.csv \
  --checkpoint /home/user/models/best.pt \
  --out_csv /home/user/results/predictions.csv
```

---

## Input Format Requirements

### Image Directory Structure
```
test_public/
├── 00000177_000.png
├── 00000177_001.png
├── 00000974_000.png
├── 00000974_001.png
└── ... (more PNG/JPG files)
```

### Metadata CSV Requirements

**Minimum Columns:**
```csv
image_id
```

**Recommended Columns:**
```csv
image_id,followup,age,gender,view_position,width,height,pixel_spacing_x,pixel_spacing_y
```

**Example Content:**
```csv
image_id,followup,age,gender,view_position,width,height,pixel_spacing_x,pixel_spacing_y
00000177_000.png,0,55,F,AP,2500,2048,0.168,0.168
00000177_001.png,1,55,F,AP,2500,2048,0.168,0.168
00000974_000.png,0,62,M,PA,2500,2048,0.168,0.168
```

---

## Output Format

### Predictions CSV Structure
```csv
image_id,p0,p1,p2,p3
00000177_000.png,0.92,0.05,0.02,0.01
00000177_001.png,0.15,0.72,0.10,0.03
00000974_000.png,0.03,0.08,0.85,0.04
```

### Column Definitions
- **image_id**: Filename of processed image
- **p0**: Probability of class 0 (Normal/Low urgency)
- **p1**: Probability of class 1 (Mild/Moderate urgency)
- **p2**: Probability of class 2 (Urgent)
- **p3**: Probability of class 3 (Critical/High urgency)

**Constraint**: p0 + p1 + p2 + p3 = 1.0 (probabilities sum to 1)

---

## Error Handling & Validation

### Input Validation
The script performs these checks:
```python
assert images_dir.exists(), f"Images dir not found: {images_dir}"
assert metadata_csv.exists(), f"Metadata CSV not found: {metadata_csv}"
assert checkpoint_path.exists(), f"Checkpoint not found: {checkpoint_path}"
assert "image_id" in meta.columns, "metadata.csv must contain image_id column"
```

### Common Errors & Solutions

**Error 1: Images directory not found**
```
AssertionError: Images dir not found: test_public
```
**Solution:**
```bash
# Check if directory exists
ls test_public/

# Verify path is correct
python predict.py --images_dir ./test_public ...
```

**Error 2: Missing image_id column**
```
AssertionError: metadata.csv must contain image_id column
```
**Solution:**
```bash
# Check CSV headers
head -1 metadata.csv

# Ensure first column is image_id
# Example: image_id,followup,age,gender,...
```

**Error 3: Checkpoint file not found**
```
AssertionError: Checkpoint not found: models/best.pt
```
**Solution:**
```bash
# Verify checkpoint exists
ls -la models/best.pt

# Check file size (should be ~40-50 MB for ResNet-18)
```

**Error 4: Image loading failures**
```
PIL.UnidentifiedImageError: cannot identify image file
```
**Solution:**
```bash
# Check image format (must be PNG or JPG)
file test_public/*.png

# Verify image integrity
identify test_public/00000177_000.png
```

---

## Performance Monitoring


```

- **Total Images**: 8000
- **Processing Time**: ~12 minutes (CPU)
- **Speed**: ~10.63 images/second

### Speed Estimates

**CPU (Intel i7/Ryzen 7):**
- Speed: 8-15 images/second
- For 8000 images: ~10-15 minutes
- Memory: 400-600 MB

**GPU (NVIDIA RTX 3080):**
- Speed: 100-200 images/second
- For 8000 images: ~40-80 seconds
- Memory: 2-4 GB

---

## Advanced Usage

### Batch Processing Multiple Datasets

**Script: `batch_inference.sh`**
```bash
#!/bin/bash

# Process multiple test sets
for dataset in test_public test_private test_external; do
    echo "Processing: $dataset"
    python predict.py \
        --images_dir $dataset \
        --metadata_csv ${dataset}_metadata.csv \
        --checkpoint models/best.pt \
        --out_csv outputs/${dataset}_predictions.csv
done
```

**Run it:**
```bash
chmod +x batch_inference.sh
./batch_inference.sh
```

### Logging Output to File

```bash
python predict.py \
  --images_dir test_public \
  --metadata_csv metadata.csv \
  --checkpoint models/best.pt \
  --out_csv outputs/test_public_predictions.csv \
  2>&1 | tee inference.log
```

### Timing the Inference

```bash
time python predict.py \
  --images_dir test_public \
  --metadata_csv metadata.csv \
  --checkpoint models/best.pt \
  --out_csv outputs/test_public_predictions.csv
```

**Output:**
```
real    12m34.567s
user    12m45.123s
sys     0m5.234s
```

---

## GPU Acceleration Setup

### Enable GPU (Modify predict.py)

**Line 24:**
```python
# Before (CPU only):
DEVICE = torch.device("cpu")

# After (GPU if available):
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
```

### Verify GPU is being used

```python
# Add to predict.py after line 24:
print(f"Using device: {DEVICE}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
```

---

## Environment Variables (Optional)

### PyTorch Configuration
```bash
# Limit CPU threads
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1

# Enable GPU memory growth
export CUDA_LAUNCH_BLOCKING=1

# Suppress warnings
export PYTHONWARNINGS=ignore

# Run inference
python predict.py --images_dir test_public ...
```

---

## Reproducibility

### Ensure Consistent Results

The script already implements:
- ✅ Fixed random seed (42)
- ✅ Single-threaded execution
- ✅ Deterministic operations
- CPU device (default)

**Expected behavior**: Running the same command multiple times produces identical probabilities.

---

## Output Verification

### Check Predictions File

```bash
# View first few rows
head -5 outputs/test_public_predictions.csv

# Count total predictions
wc -l outputs/test_public_predictions.csv

# Validate probability sums (should all equal ~1.0)
python -c "
import pandas as pd
df = pd.read_csv('outputs/test_public_predictions.csv')
df['sum'] = df['p0'] + df['p1'] + df['p2'] + df['p3']
print(f'Min sum: {df[\"sum\"].min():.6f}')
print(f'Max sum: {df[\"sum\"].max():.6f}')
print(f'Mean sum: {df[\"sum\"].mean():.6f}')
"
```

---

## Support & Documentation

- **Prediction Script**: [predict.py](predict.py)
- **Full README**: [README.md](README.md)
- **Dependencies**: [requirements.txt](requirements.txt)
- **Model Checkpoint**: [models/best.pt](models/best.pt)

---

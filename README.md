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

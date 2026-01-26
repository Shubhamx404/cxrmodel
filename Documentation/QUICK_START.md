# 🚀 QUICK START CARD - CXR Triage Inference

## ⚡ 60-Second Setup

```bash
# 1. Activate environment
.venv\Scripts\activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run inference
python predict.py \
  --images_dir test_public \
  --metadata_csv metadata.csv \
  --checkpoint models/best.pt \
  --out_csv outputs/test_public_predictions.csv

# 4. Check results
head outputs/test_public_predictions.csv
```

---

## 📋 What This Does

| Step | Input | Process | Output |
|------|-------|---------|--------|
| Load | metadata.csv | Parse image list | 8000 image IDs |
| Model | models/best.pt | ResNet-18 checkpoint | 512 features → 4 classes |
| Process | test_public/*.png | Resize 224×224, normalize | 4 probability scores |
| Output | p0,p1,p2,p3 | Save to CSV | predictions.csv |

---

## 📊 Output Format

```
image_id          | p0    | p1    | p2    | p3
===============================================
00000177_000.png  | 0.92  | 0.05  | 0.02  | 0.01
00000177_001.png  | 0.15  | 0.72  | 0.10  | 0.03
00000974_000.png  | 0.03  | 0.08  | 0.85  | 0.04

✓ Sum = 1.0 for each row
```

---

## 🔧 CLI Arguments (Required)

```bash
python predict.py \
  --images_dir    PATH        # Directory with PNG/JPG images
  --metadata_csv  PATH        # CSV with image_id column
  --checkpoint    PATH        # PyTorch model weights (.pt)
  --out_csv       PATH        # Output predictions CSV
```

---

## ⚠️ Common Issues & Fixes

### Issue 1: "Directory not found"
```bash
# Check if directory exists
ls test_public/
# Fix: Use correct path
python predict.py --images_dir ./test_public ...
```

### Issue 2: "Missing image_id column"
```bash
# Check CSV headers
head -1 metadata.csv
# Should show: image_id,followup,age,gender,...
```

### Issue 3: "Model checkpoint not found"
```bash
# Verify file exists
ls -la models/best.pt
# File should be ~40-50 MB
```

---

## 📈 Performance

| Hardware | Speed | Time for 8000 |
|----------|-------|---------------|
| CPU (i7/Ryzen 7) | 10 img/sec | ~13 min |
| GPU (RTX 3080) | 150 img/sec | ~1 min |

---

## 🎯 What's Happening

```
Input Images (8000)
        ↓
   [Resize to 224×224]
        ↓
   [ImageNet Normalize]
        ↓
   [ResNet-18 Forward]
        ↓
   [Softmax Activation]
        ↓
Output: 4 Probabilities per image
        ↓
Save to CSV (outputs/predictions.csv)
```

---

## 📚 Documentation Files

1. **README.md** - Full documentation (341 lines)
2. **CLI_REFERENCE.md** - CLI commands & examples (450+ lines)
3. **DOCUMENTATION_SUMMARY.txt** - Overview of all docs
4. **QUICK_START.md** - This file

---

## ✅ Verify Installation

```bash
# Check Python
python --version

# Check PyTorch
python -c "import torch; print(torch.__version__)"

# Check files exist
ls models/best.pt metadata.csv test_public/

# Run help
python predict.py --help
```

---

## 🎓 Model Details

- **Name**: ResNet-18
- **Input**: 224×224 RGB images
- **Output**: 4 class probabilities
- **Classes**: p0, p1, p2, p3 (triage levels)
- **Framework**: PyTorch

---

## 🔐 Reproducibility

- ✅ Same results every run
- ✅ Fixed seed (42)
- ✅ CPU deterministic
- ✅ No randomization during inference

---

## 📞 Help

- Stuck? Check **README.md**
- Need CLI examples? Check **CLI_REFERENCE.md**
- Troubleshooting? See README.md → Troubleshooting section

---

**Quick Tip**: Run `python predict.py --help` for built-in documentation

**Status**: ✅ Ready to Use  
**Test Images**: 8000+  
**Expected Time**: 10-15 min (CPU)

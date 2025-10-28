# Medical-Image-Analyzer

Research-style **histopathology analyzer** for gigapixel WSIs (e.g., SVS), featuring:

- **Patch tiling** with OpenSlide
- **MONAI** training pipeline (patch-level classification)
- **WSI inference** with heatmap overlay
- **Explainability** via Grad-CAM on patches
- **Streamlit** UI for interactive exploration

> Works with public datasets (e.g., CAMELYON16, TCGA) and private data. Ships with a small demo mode.

---

## Quickstart

```bash
# 1) Create environment (Python 3.10+ recommended)
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate

# 2) Install requirements
pip install -r requirements.txt

# 3) (Optional) Install Torch explicitly for your platform
#    e.g., CPU-only:
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu

# 4) Launch the Streamlit app
streamlit run app/streamlit_app.py
```

The app supports uploading SVS/TIFF/PNG, tiling patches, running a demo classifier, visualizing patch scores, and exporting heatmaps.

---

## Repository Layout

```
Medical-Image-Analyzer/
├── app/
│   └── streamlit_app.py           # UI for tiling + inference + heatmaps
├── configs/
│   └── default.yaml               # data/model/inference configs
├── data/
│   └── patches/                   # patch dataset (train/val) if you build one
├── models/
│   └── (best.pth)                 # trained MONAI model checkpoint
├── notebooks/
│   └── EDA.ipynb
├── scripts/
│   ├── tile_wsi.py                # SVS->patches + csv labels (if available)
│   ├── train_monai.py             # patch-level classifier training
│   ├── infer_wsi.py               # slide-level inference + heatmap
│   └── gradcam_patch.py           # Grad-CAM on a single patch
├── utils/
│   ├── tiler.py                   # OpenSlide tiler utilities
│   ├── dataset.py                 # MONAI Dataset wrappers
│   ├── model.py                   # Backbone & head
│   ├── transforms.py              # MONAI transforms
│   └── viz.py                     # Heatmap & overlay
├── tests/
│   └── test_tiler.py              # sanity tests for tiler
├── requirements.txt
├── .gitignore
└── README.md
```

---

## Training (patch-level)

```bash
python scripts/train_monai.py   --data_dir data/patches   --epochs 10   --batch 32   --imgsz 224   --num_classes 2
```

Produces `models/best.pth`. Use it in the app and `infer_wsi.py`.

---

## Roadmap

- [ ] Multi-class support with per-class heatmaps
- [ ] Slide-level MIL (multiple instance learning)
- [ ] Mixed precision + distributed training
- [ ] On-the-fly stain normalization (Macenko/Reinhard)
- [ ] DICOM/WSI metadata viewer and QC suite

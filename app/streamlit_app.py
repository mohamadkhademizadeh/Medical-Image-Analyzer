import streamlit as st
import numpy as np
import yaml, io
import cv2
from PIL import Image
import torch
import torchvision.transforms as T

from utils.tiler import read_region_thumb, iterate_patches, is_tissue
from utils.model import build_model
from utils.viz import make_heatmap, overlay

st.set_page_config(page_title="Medical Image Analyzer", layout="wide")
st.title("🩺 Medical Image Analyzer — WSI Patching & Heatmaps")

with open('configs/default.yaml','r') as f:
    CFG = yaml.safe_load(f)

uploaded = st.file_uploader("Upload a slide (SVS/TIFF/PNG)", type=["svs","tif","tiff","png","jpg","jpeg"])
col1, col2 = st.columns(2)

ps = st.sidebar.number_input("Patch size", 64, 1024, CFG['data']['patch_size'], step=32)
stp = st.sidebar.number_input("Stride", 32, 1024, CFG['data']['patch_stride'], step=32)
tissue_thr = st.sidebar.slider("Tissue keep threshold", 0.0, 1.0, float(CFG['data']['tissue_threshold']), 0.05)
heat_ds = st.sidebar.number_input("Heatmap downsample", 2, 32, CFG['inference']['heatmap_downsample'], step=2)

model_name = st.sidebar.selectbox("Backbone", ["resnet18","resnet50"], index=0)
num_classes = st.sidebar.number_input("Num classes", 2, 10, CFG['model']['num_classes'])
ckpt_path = st.sidebar.text_input("Checkpoint path", "models/best.pth")

@st.cache_resource
def load_model(name, nc, ckpt):
    m = build_model(name=name, num_classes=nc, pretrained=True)
    if ckpt and ckpt.strip() and os.path.exists(ckpt):
        m.load_state_dict(torch.load(ckpt, map_location='cpu'))
    m.eval()
    return m

if uploaded is not None:
    if uploaded.name.lower().endswith(('.png','.jpg','.jpeg')):
        img = Image.open(uploaded).convert('RGB')
        rgb = np.array(img)
        slide = None
        W, H = rgb.shape[1], rgb.shape[0]
    else:
        rgb, slide = read_region_thumb(uploaded.name)

    with col1:
        st.subheader("Thumbnail")
        st.image(rgb, use_column_width=True)

    # Simple patch scoring using torchvision resize + softmax over class 1
    model = build_model(model_name, num_classes)
    model.eval()
    preprocess = T.Compose([
        T.ToTensor(),
        T.Resize((ps, ps)),
        T.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225]),
    ])

    points = []
    values = []

    if slide is None:
        # whole image patching
        H, W = rgb.shape[:2]
        for y in range(0, H-ps+1, stp):
            for x in range(0, W-ps+1, stp):
                patch = rgb[y:y+ps, x:x+ps]
                if not is_tissue(patch, tissue_thr):
                    continue
                tens = preprocess(Image.fromarray(patch)).unsqueeze(0)
                with torch.no_grad():
                    logits = model(tens)
                    prob = torch.softmax(logits, dim=1)[0, min(1, num_classes-1)].item()
                points.append((y//heat_ds, x//heat_ds))
                values.append(prob)
    else:
        # slide patching (thumbnail coordinates mapped approximately)
        for (x,y), patch in iterate_patches(slide, patch_size=ps, stride=stp):
            if not is_tissue(patch, tissue_thr):
                continue
            tens = preprocess(Image.fromarray(patch)).unsqueeze(0)
            with torch.no_grad():
                logits = model(tens)
                prob = torch.softmax(logits, dim=1)[0, min(1, num_classes-1)].item()
            points.append((y//heat_ds, x//heat_ds))
            values.append(prob)

    Hh, Wh = rgb.shape[0]//heat_ds, rgb.shape[1]//heat_ds
    heat = make_heatmap((Hh, Wh), points, values, sigma=8)
    overlay_img = overlay(rgb, heat, alpha=0.5)

    with col2:
        st.subheader("Heatmap Overlay")
        st.image(overlay_img, use_column_width=True)

    st.caption("Tip: train your model with scripts/train_monai.py and place the checkpoint at models/best.pth to use it here.")
else:
    st.info("Upload a slide or image to begin.")

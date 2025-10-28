import argparse, os
import numpy as np
from PIL import Image
import torch, torchvision.transforms as T
from utils.tiler import read_region_thumb, iterate_patches, is_tissue
from utils.model import build_model
from utils.viz import make_heatmap, overlay

def main(args):
    rgb, slide = read_region_thumb(args.slide)
    ps, stp, thr = args.size, args.stride, args.tissue_thr

    model = build_model(args.backbone, num_classes=args.num_classes)
    if os.path.exists(args.weights):
        model.load_state_dict(torch.load(args.weights, map_location='cpu'))
    model.eval()
    preprocess = T.Compose([T.ToTensor(), T.Resize((ps, ps)),
                            T.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])])

    points, values = [], []
    for (x,y), patch in iterate_patches(slide, patch_size=ps, stride=stp):
        if not is_tissue(patch, thr):
            continue
        tens = preprocess(Image.fromarray(patch)).unsqueeze(0)
        with torch.no_grad():
            logits = model(tens)
            prob = torch.softmax(logits, dim=1)[0, min(1, args.num_classes-1)].item()
        points.append((y//args.heat_ds, x//args.heat_ds))
        values.append(prob)

    Hh, Wh = rgb.shape[0]//args.heat_ds, rgb.shape[1]//args.heat_ds
    heat = make_heatmap((Hh, Wh), points, values, sigma=8)
    out = overlay(rgb, heat, alpha=0.5)
    os.makedirs(args.out_dir, exist_ok=True)
    Image.fromarray(out).save(os.path.join(args.out_dir, "heatmap_overlay.png"))
    print("Saved", os.path.join(args.out_dir, "heatmap_overlay.png"))

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--slide", required=True)
    ap.add_argument("--weights", default="models/best.pth")
    ap.add_argument("--out_dir", default="out")
    ap.add_argument("--size", type=int, default=224)
    ap.add_argument("--stride", type=int, default=224)
    ap.add_argument("--tissue_thr", type=float, default=0.8)
    ap.add_argument("--heat_ds", type=int, default=8)
    ap.add_argument("--num_classes", type=int, default=2)
    ap.add_argument("--backbone", type=str, default="resnet18")
    args = ap.parse_args()
    main(args)

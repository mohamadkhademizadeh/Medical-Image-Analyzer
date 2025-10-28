import argparse, os, csv
from utils.tiler import read_region_thumb, iterate_patches, is_tissue
from PIL import Image

def main(args):
    os.makedirs(args.out_dir, exist_ok=True)
    rgb, slide = read_region_thumb(args.slide)
    W,H = rgb.shape[1], rgb.shape[0]
    count = 0
    for (x,y), patch in iterate_patches(slide, patch_size=args.size, stride=args.stride):
        if not is_tissue(patch, args.tissue_thr):
            continue
        out_path = os.path.join(args.out_dir, f"patch_{y}_{x}.png")
        Image.fromarray(patch).save(out_path)
        count += 1
    print(f"Saved {count} patches to {args.out_dir}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--slide", required=True)
    ap.add_argument("--out_dir", default="data/patches")
    ap.add_argument("--size", type=int, default=224)
    ap.add_argument("--stride", type=int, default=224)
    ap.add_argument("--tissue_thr", type=float, default=0.8)
    args = ap.parse_args()
    main(args)

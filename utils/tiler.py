from typing import Tuple, List
import numpy as np
import openslide
import cv2

def read_region_thumb(slide_path: str, level: int = 0, max_dim: int = 2048):
    slide = openslide.OpenSlide(slide_path)
    w, h = slide.dimensions
    scale = max(w, h) / max_dim if max(w, h) > max_dim else 1.0
    thumb = slide.get_thumbnail((int(w/scale), int(h/scale)))
    return np.array(thumb)[:, :, :3], slide

def iterate_patches(slide, patch_size: int, stride: int):
    W, H = slide.dimensions
    for y in range(0, H - patch_size + 1, stride):
        for x in range(0, W - patch_size + 1, stride):
            patch = slide.read_region((x, y), 0, (patch_size, patch_size))
            patch = np.array(patch)[:, :, :3]
            yield (x, y), patch

def is_tissue(patch, threshold=0.8):
    gray = cv2.cvtColor(patch, cv2.COLOR_RGB2GRAY)
    white = (gray > 220).mean()
    return (1.0 - white) >= threshold

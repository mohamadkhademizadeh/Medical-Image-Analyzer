import numpy as np
import cv2
import matplotlib.pyplot as plt

def make_heatmap(canvas_hw, points, values, sigma=8):
    H,W = canvas_hw
    heat = np.zeros((H,W), dtype=np.float32)
    for (y,x), v in zip(points, values):
        y = int(max(0, min(H-1, y)))
        x = int(max(0, min(W-1, x)))
        heat[y, x] += float(v)
    heat = cv2.GaussianBlur(heat, (0,0), sigmaX=sigma, sigmaY=sigma)
    heat = heat - heat.min()
    if heat.max() > 1e-8:
        heat = heat / heat.max()
    return heat

def overlay(image_rgb, heatmap, alpha=0.5):
    hm = (plt.cm.jet(heatmap)[:,:,:3]*255).astype('uint8')
    hm = cv2.resize(hm, (image_rgb.shape[1], image_rgb.shape[0]))
    out = cv2.addWeighted(image_rgb, 1-alpha, hm, alpha, 0)
    return out

from monai.transforms import (
    Compose, LoadImage, EnsureChannelFirst, ScaleIntensity, RandFlip, RandRotate,
    RandZoom, ToTensor, Resized, Lambda
)
import numpy as np
from PIL import Image

def train_transforms(size=224):
    return Compose([
        Lambda(func=lambda d: {"img": d["img"], "label": d["label"]}),
        EnsureChannelFirst(keys=["img"]),
        Resized(keys=["img"], spatial_size=(size, size)),
        ScaleIntensity(keys=["img"]),
        RandFlip(keys=["img"], prob=0.5, spatial_axis=1),
        RandRotate(keys=["img"], prob=0.2, range_x=0.2),
        RandZoom(keys=["img"], prob=0.2, min_zoom=0.9, max_zoom=1.1),
        ToTensor(keys=["img"]),
    ])

def val_transforms(size=224):
    return Compose([
        Lambda(func=lambda d: {"img": d["img"], "label": d["label"]}),
        EnsureChannelFirst(keys=["img"]),
        Resized(keys=["img"], spatial_size=(size, size)),
        ScaleIntensity(keys=["img"]),
        ToTensor(keys=["img"]),
    ])

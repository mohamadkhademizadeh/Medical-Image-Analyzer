import os, glob
import numpy as np
from monai.data import Dataset
from monai.transforms import Compose
from PIL import Image

class PatchFolder(Dataset):
    def __init__(self, root, transform=None):
        self.root = root
        self.transform = transform
        self.samples = []
        for cls in sorted(os.listdir(root)):
            cdir = os.path.join(root, cls)
            if not os.path.isdir(cdir):
                continue
            for p in glob.glob(os.path.join(cdir, '*.png')) + glob.glob(os.path.join(cdir, '*.jpg')):
                self.samples.append((p, cls))
        self.classes = sorted({c for _, c in self.samples})
        self.class_to_idx = {c:i for i,c in enumerate(self.classes)}

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, cls = self.samples[idx]
        img = Image.open(path).convert('RGB')
        item = {"img": img, "label": self.class_to_idx[cls]}
        if self.transform:
            item = self.transform(item)
        return item

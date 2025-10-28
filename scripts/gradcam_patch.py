import argparse, os
import torch
from PIL import Image
import torchvision.transforms as T
from torchcam.methods import SmoothGradCAMpp
from utils.model import build_model
import numpy as np

def main(args):
    model = build_model(args.backbone, num_classes=args.num_classes)
    if os.path.exists(args.weights):
        model.load_state_dict(torch.load(args.weights, map_location='cpu'))
    model.eval()

    preprocess = T.Compose([
        T.ToTensor(), T.Resize((args.size, args.size)),
        T.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
    ])
    img = Image.open(args.image).convert('RGB')
    x = preprocess(img).unsqueeze(0)
    target_class = args.target_class

    cam_extractor = SmoothGradCAMpp(model)
    out = model(x)
    cams = cam_extractor(target_class, out)
    cam = cams[0].squeeze().detach().cpu().numpy()
    cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
    cam = (cam * 255).astype('uint8')
    Image.fromarray(cam).save(args.out)

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", required=True)
    ap.add_argument("--weights", default="models/best.pth")
    ap.add_argument("--out", default="cam.png")
    ap.add_argument("--size", type=int, default=224)
    ap.add_argument("--num_classes", type=int, default=2)
    ap.add_argument("--backbone", type=str, default="resnet18")
    ap.add_argument("--target_class", type=int, default=1)
    args = ap.parse_args()
    main(args)

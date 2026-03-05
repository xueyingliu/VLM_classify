#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.models import resnet50
from PIL import Image


IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}


def collect_images(p: str):
    p = Path(p)
    if p.is_file():
        return [str(p)]
    files = [x for x in p.iterdir() if x.is_file() and x.suffix.lower() in IMG_EXTS]
    return [str(x) for x in sorted(files)]


class ImgDS(Dataset):
    def __init__(self, paths, tfm):
        self.paths = paths
        self.tfm = tfm

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        path = self.paths[i]
        img = Image.open(path).convert("RGB")
        return self.tfm(img), path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, type=str)
    ap.add_argument("--ckpt", required=True, type=str)
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--num_workers", type=int, default=4)
    ap.add_argument("--image_size", type=int, default=224)
    args = ap.parse_args()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    ckpt = torch.load(args.ckpt, map_location="cpu")
    state = ckpt["model_state_dict"]
    class_names = ckpt["class_names"]
    num_classes = len(class_names)

    print(f"[INFO] device   = {device}")
    print(f"[INFO] ckpt     = {args.ckpt}")
    print(f"[INFO] backbone = resnet50")
    print(f"[INFO] num_classes = {num_classes}")

    tfm = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(args.image_size),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    paths = collect_images(args.input)
    if not paths:
        raise RuntimeError(f"No images found: {args.input}")
    print(f"[INFO] num_images = {len(paths)}")

    loader = DataLoader(
        ImgDS(paths, tfm),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
    )

    model = resnet50(weights=None)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    model.load_state_dict(state, strict=False)
    model.to(device).eval()

    with torch.inference_mode():
        for imgs, pths in loader:
            imgs = imgs.to(device, non_blocking=True)
            prob = torch.softmax(model(imgs), dim=1).cpu()
            v, i = prob.max(dim=1)
            for p, vv, ii in zip(pths, v.tolist(), i.tolist()):
                print(f"{p}\t->\t{ii:02d}:{class_names[ii]} prob={vv:.6f}")


if __name__ == "__main__":
    main()
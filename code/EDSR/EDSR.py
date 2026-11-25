import numpy as np

import torch
import torch.nn as nn
import torchvision.transforms.functional as TF

from PIL import Image

import argparse
import math
import glob
import os

# Model adapted from the Enhanced Deep Super-Resolution Network (Lim et al. https://arxiv.org/abs/1707.02921)

parser = argparse.ArgumentParser(
    description="Train an Enhanced Deep Residual Network on DIV2k",
    formatter_class=argparse.ArgumentDefaultsHelpFormatter,
)
parser.add_argument("--img-id", default="0801")
parser.add_argument("--scale", default=8, type=int, choices=[8, 16])
parser.add_argument("--seed", default=0, type=int)
parser.add_argument("--noise-sigma", default=0.0, type=float)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dtype = torch.cuda.FloatTensor

HR_DIR = '../../data/DIV2K_valid_HR'
LR_DIR = '../../data/DIV2K_valid_LR_x8'

# TODO: take hyperparameters from paper

class MeanShift(nn.Module):
    def __init__(self, rgbMean, sign):
        super().__init__()
        self.register_buffer("mean", torch.tensor(rgbMean).view(1, 3, 1, 1) * sign)
    
    def forward(self, x):
        return x + x.mean   

# TODO: note the lack of batch norm
# Residual block as defined in paper
class ResBlock(nn.Module):
    def __init__(self, nFeatures=64):
        super.__init__()
        self.cn1 = nn.Conv2d(nFeatures, nFeatures, 3, padding=1)
        self.cn2 = nn.Conv2d(nFeatures, nFeatures, 3, padding=1)
        
    def forward(self, x):
        res = self.cn2(nn.ReLU(self.cn1(x)))
        return x + res

# Upsample block as defined in paper and repository
class UpBlock(nn.Module):
    def __init__(self, nFeatures=64, scale=8):
        super().__init__()
        self.layers = nn.ModuleList()
        for i in range(int(math.log2(scale))):
            self.layers.append(nn.Sequential(
                nn.Conv2d(nFeatures, nFeatures * 4, 3, padding=1),
                nn.PixelShuffle(2),
            ))
        
    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

class EDSR(nn.Module):
    def __init__(self, rgbMean, nFeatures=64, nBlocks=16, scale=8):
        super().__init__()
        self.nFeatures = nFeatures
        self.nBlocks = nBlocks
        self.scale = scale
        self.subMean = MeanShift(rgbMean, -1)
        self.head = nn.Conv2d(3, nFeatures, 3, padding=1)
        self.body = nn.ModuleList()
        for _ in range(nBlocks):
            self.body.append(ResBlock(nFeatures))
        self.body.append(nn.Conv2d(nFeatures, nFeatures, 3, padding=1))
        self.upsample = UpBlock(nFeatures, scale)
        self.tail = nn.Conv2d(nFeatures, 3, 3, padding=1)
        self.addMean = MeanShift(rgbMean, 1)
    
    def forward(self, x):
        x = self.subMean(x)
        x = self.head(x)
        x = self.body(x) + x
        x = self.tail(self.upsample(x))
        return self.addMean(x)

# TODO comment out
def calculateMeanRGB(hrDir):
    total, count = torch.zeros(3), 0
    for file in sorted(glob.glob(os.path.join(hrDir, "*.png"))):
        img = TF.to_tensor(Image.open(file).convert("RGB"))
        total += img.sum(dim=(1, 2))
        count += img.shape[1] * img.shape[2]
    return (total / count).tolist()

def trainEDSR(args):
    return

def main(args):
    torch.manual_seed(args.seed)
    mean = calculateMeanRGB(HR_DIR)

if __name__ == "__main__":
    main(parser.parse_args())
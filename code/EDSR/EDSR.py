import numpy as np

import torch
import torch.nn as nn

import argparse

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

class ResBlock(nn.Module):
    def __init__(self):
        self.cn1 = nn.Conv2d()

class EDSR(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

def main(args):
    torch.manual_seed(args.seed)

if __name__ == "__main__":
    main(parser.parse_args())
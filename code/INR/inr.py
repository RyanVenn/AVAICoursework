import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.utils import save_image
import torchvision.transforms.functional as TF

from skimage.metrics import peak_signal_noise_ratio as psnr_fn, structural_similarity as ssim_fn
from utils.denoising_utils import *

from models.siren import Siren

import lpips

import argparse
import os
import time

parser = argparse.ArgumentParser(
    description="Train a SIREN (INR) on DIV2k",
    formatter_class=argparse.ArgumentDefaultsHelpFormatter,
)
parser.add_argument("--img-id", default="0801")
parser.add_argument("--scale", default=8, type=int, choices=[8, 16])
parser.add_argument("--crop", default=512, type=int)
parser.add_argument("--iters", default=2000, type=int)
parser.add_argument("--learning-rate", default=1e-4, type=float)
parser.add_argument("--seed", default=0, type=int)
parser.add_argument("--noise-sigma", default=0.0, type=float)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dtype = torch.cuda.FloatTensor

HR_DIR = '../../data/DIV2K_valid_HR'
LR_DIR = '../../data/DIV2K_valid_LR_x8'

# Define the degredation operator
def downsampler(x, factor):
    return F.interpolate(x, scale_factor=1/factor, mode='bicubic',
                         antialias=True, align_corners=False)
    
def get_mgrid(h, w):
    ys = (torch.arange(h) + 0.5) / h * 2 - 1
    xs = (torch.arange(w) + 0.5) / w * 2 - 1
    return torch.stack(torch.meshgrid(ys, xs, indexing="ij"), dim=-1).reshape(-1, 2)
    
# Crop images so that the HR and LR pairs align with their scaling
def crop_image_pair(hr_img, lr_img, scale, crop):
    # d=4 ensires hr is a multiple of 32 for skip (4*8)
    lr_crop = crop_image(lr_img, d=4)
    x0 = (lr_img.size[0] - lr_crop.size[0]) // 2
    y0 = (lr_img.size[1] - lr_crop.size[1]) // 2
    w, h = lr_crop.size
    
    if crop:
        # 4 * 4 ensures it remains a multiple of 4
        cw = min(w, crop // scale) // 4 * 4
        ch = min(h, crop // scale) // 4 * 4
        x0 += (w - cw) // 2
        y0 += (h - ch) // 2
        w, h = cw, ch
    
    # Crop the images
    lr_crop = lr_img.crop((x0, y0, x0 + w, y0 + h))
    hr_crop = hr_img.crop((x0*scale, y0*scale, (x0+w)*scale, (y0+h)*scale))
    return hr_crop, lr_crop

def evaluate(sr, hr, lpips_fn):
    sr, hr = sr.clamp(0, 1), hr.clamp(0, 1)
    s = sr[0].permute(1, 2, 0).cpu().numpy()
    h = hr[0].permute(1, 2, 0).cpu().numpy()
    with torch.no_grad():
        lp = lpips_fn(sr * 2 - 1, hr * 2 - 1).item()
    return psnr_fn(h, s, data_range=1), ssim_fn(h, s, channel_axis=2, data_range=1), lp

@torch.no_grad()
def render(siren, coords, H, W):
    out = torch.cat([siren(c)[0] for c in coords.split(65536)])
    return ((out + 1) / 2).reshape(H, W, 3).permute(2, 0, 1)[None]

def main(args):
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    # Open high-res and low-res images
    hr_pil = get_image(f'{HR_DIR}/{args.img_id}.png')[0]
    lr_pil = get_image(f'{LR_DIR}/{args.img_id}x8.png')[0]
    hr_pil, lr_pil = crop_image_pair(hr_pil, lr_pil, 8, args.crop)
    
    # Convert to tensor
    hr_img = TF.to_tensor(hr_pil)[None].to(DEVICE)
    lr_img = TF.to_tensor(lr_pil)[None].to(DEVICE)
    
    # task 2.2: add Gaussian noise to the LR image
    if args.noise_sigma > 0:
        _, lr_img = get_noisy_image(lr_img[0].cpu().numpy(), args.noise_sigma)
        lr_img = np_to_torch(lr_img).type(dtype)
        
    # task 2.3: downscale x8 image to x16
    if args.scale == 16:
        lr_img = downsampler(lr_img, 2).clamp(0, 1)

    # Initialize a SIREN model
    siren = Siren(in_features=2, out_features=3, hidden_features=256,
                 hidden_layers=3, outermost_linear=True).to(DEVICE)
    H, W = hr_img.shape[-2:]
    h, w = lr_img.shape[-2:]
    hr_coords = get_mgrid(H, W).to(DEVICE)
    lr_coords = get_mgrid(h, w).to(DEVICE)
    
    # Initialize optimizer and evaluation metrics
    optim = torch.optim.Adam(lr=args.learning_rate, params=siren.parameters())
    lpips_fn = lpips.LPIPS(net="alex").to(DEVICE)
    bicubic = F.interpolate(lr_img, size=(H, W), mode="bicubic", align_corners=False).clamp(0, 1)
    mse = torch.nn.MSELoss().type(dtype)
    
    os.makedirs("results", exist_ok=True)
    start = time.time()
    
    lr_pixels = (lr_img * 2 - 1)[0].permute(1, 2, 0).reshape(-1, 3)
    
    for i in range(1, args.iters + 1):
        out, _ = siren(lr_coords)
        loss = mse(out, lr_pixels)
        optim.zero_grad()
        loss.backward()
        optim.step()
        
        if i % 100 == 0 or i == args.iters:
            sr = render(siren, hr_coords, H, W)
            p = psnr_fn(hr_img[0].cpu().numpy(), sr.clamp(0, 1)[0].cpu().numpy(), data_range=1)
            print ('Iteration: ', i, ' Loss: ', loss.item(), ' PSNR_gt: ', p)   

    elapsed = time.time() - start
    
    for name, img in [("bicubic", bicubic), ("siren", sr)]:
        p, s, l = evaluate(img, hr_img, lpips_fn)
        print (f"{name} PSNR {p:.4f} SSIM {s:.4f} LPIPS {l:.4f}")
    print(f"time elapsed: {elapsed:.2f}s")     
    
    save_image(torch.cat([bicubic, sr.clamp(0, 1), hr_img]), f"results/inr_{args.img_id}.png", nrow=3)
        
if __name__ == "__main__":
    main(parser.parse_args())
    
import numpy as np
from models import *

import torch
import torch.nn.functional as F

from skimage.metrics import peak_signal_noise_ratio as psnr_fn, structural_similarity as ssim_fn
from utils.denoising_utils import *
import matplotlib.pyplot as plt 

import lpips
import argparse
import os
import time
import csv
import itertools
import copy

parser = argparse.ArgumentParser(
    description="Train a Deep Image Prior model on DIV2k",
    formatter_class=argparse.ArgumentDefaultsHelpFormatter,
)
parser.add_argument("--img-id", default="0801")
parser.add_argument("--scale", default=8, type=int, choices=[8, 16])
parser.add_argument("--crop", default=512, type=int)
parser.add_argument("--iters", default=2000, type=int)
parser.add_argument("--learning-rate", default=0.003, type=float)
parser.add_argument("--reg-noise-std", default=1./30, type=float)
parser.add_argument("--seed", default=0, type=int)
parser.add_argument("--noise-sigma", default=0.0, type=float)
parser.add_argument("--tune-hyperparams", action="store_true")
parser.add_argument("--tune-images", nargs="+", default=["0801", "0806", "0840"])

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dtype = torch.cuda.FloatTensor

HR_DIR = '../../data/DIV2K_valid_HR'
LR_DIR = '../../data/DIV2K_valid_LR_x8'

# Crop images so that the HR and LR pairs align with their scaling
def cropImagePair(hr_img, lr_img, scale, crop):
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

# Define the degredation operator
def downsampler(x, factor):
    return F.interpolate(x, scale_factor=1/factor, mode='bicubic',
                         antialias=True, align_corners=False)

# Plots the LR, HR, and reconstructed image at iteration i
def plotIteration(i, lr, hr, recon):
    recon_display = np.clip(torch_to_np(recon), 0, 1)
    recon_display = np.transpose(recon_display, (1, 2, 0))
    
    hr_display = np.transpose(hr, (1, 2, 0))
    
    lr_display = F.interpolate(
        lr,
        size=(hr.shape[1], hr.shape[2]),
        mode='bicubic',
        align_corners=False,
    )
    lr_display = np.transpose(torch_to_np(lr_display), (1, 2, 0))
    lr_display = np.clip(lr_display, 0, 1)
    
    fig, ax = plt.subplots(1, 3, figsize=(15, 5))
    
    ax[0].imshow(hr_display)
    ax[0].set_title("High Resolution (Ground Truth)")
    ax[0].axis('off')
    
    ax[1].imshow(lr_display)
    ax[1].set_title("Low Resolution (Upscaled)")
    ax[1].axis('off')
    
    ax[2].imshow(recon_display)
    ax[2].set_title("Reconstruction")
    ax[2].axis('off')
    
    fig.suptitle(f"Iteration {i}")
    
    plt.tight_layout()
    
    plt.savefig(
        f"results/iteration_{i}.png"
    )
    plt.close(fig)
    
def runDIP(args, lpips_fn, cps=(), log=True):
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    # Open high-res and low-res images
    hr_pil = get_image(f'{HR_DIR}/{args.img_id}.png')[0]
    lr_pil = get_image(f'{LR_DIR}/{args.img_id}x8.png')[0]
    hr_pil, lr_pil = cropImagePair(hr_pil, lr_pil, 8, args.crop)
    
    img_np, lr_np = pil_to_np(hr_pil), pil_to_np(lr_pil)
    lr_torch = np_to_torch(lr_np).type(dtype) 
    img_torch = np_to_torch(img_np).type(dtype)
    
    # task 2.2: add Gaussian noise to the LR image
    if args.noise_sigma > 0:
        _, lr_np = get_noisy_image(lr_np, args.noise_sigma)
        lr_torch = np_to_torch(lr_np).type(dtype)
        
    # task 2.3: downscale x8 image to x16
    if args.scale == 16:
        lr_torch = downsampler(lr_torch, 2).clamp(0, 1)
        lr_np = torch_to_np(lr_torch)
    
    # Network structure
    net = get_net(32, 'skip', 'reflection',
                    skip_n33d=128, skip_n33u=128, skip_n11=4,
                    num_scales=5, upsample_mode='bilinear').type(dtype)
    
    # Get input for network
    net_input = get_noise(32, 'noise', (img_np.shape[1], img_np.shape[2])).type(dtype).detach()
    net_input_saved, noise = net_input.detach().clone(), net_input.detach().clone()
    
    # Compute number of parameters
    s  = sum([np.prod(list(p.size())) for p in net.parameters()]);
    print ('Number of params: %d' % s)
    
    # Loss
    mse = torch.nn.MSELoss().type(dtype)
    
    i = 0
    os.makedirs("results", exist_ok=True)
    
    def evaluate(evalImg):
            evalImg = evalImg.clamp(0, 1)
            a = torch_to_np(evalImg)
            with torch.no_grad():
                # * 2 - 1 takes image from [0, 1] to [-1, 1]
                l = lpips_fn(evalImg * 2 - 1, img_torch * 2 - 1).item()
            return psnr_fn(img_np, a, data_range=1), ssim_fn(img_np, a, channel_axis=0, data_range=1), l
    
    results = []
    
    def closure():
        nonlocal i, net_input
        
        if i in cps:
            with torch.no_grad():
                results.append((i,) + evaluate(net(net_input_saved)))
        
        if args.reg_noise_std > 0:
            net_input = net_input_saved + (noise.normal_() * args.reg_noise_std)
            
        out = net(net_input)
        
        loss = mse(downsampler(out, args.scale), lr_torch)
        loss.backward()
        
        if i % 10 == 0 and log:
            p = psnr_fn(img_np, np.clip(torch_to_np(out), 0, 1), data_range=1)
            print('Iteration: ', i, ' Loss: ', loss.item(), ' PSNR_gt: ', p)
        
        if i % 100 == 0 and log:
            plotIteration(i, lr_torch, img_np, out)
            
        i += 1
        return loss
    
    start = time.time()
    params = get_params('net', net, net_input)
    optimize('adam', params, closure, args.learning_rate, args.iters)
    
    if log:
        print(f"time elapsed: {time.time() - start:.2f}s")  

    with torch.no_grad():
        recon = net(net_input_saved)
    bicubic = F.interpolate(lr_torch, size=img_torch.shape[-2:], mode='bicubic', align_corners=False)

    if log:
        for name, img in [('bicubic', bicubic), ('DIP', recon)]:
            p, s, l = evaluate(img)
            print(f"{name} PSNR {p:.4f} SSIM {s:.4f} LPIPS {l:.4f}")

    return results

# Performs a search over selected hyperparameters
def searchParams(args, lpips_fn):
    cps = [500, 1000, 1500, 2000]
    os.makedirs("results", exist_ok=True)
    results = {}
    
    with open("results/dipTuning.csv", "a", newline="") as f:
        w = csv.writer(f)
        for lr, reg in itertools.product([0.003, 0.01, 0.03], [0.0, 1./30, 0.1]):
            for img_id in args.tune_images:
                a = copy.copy(args)
                a.img_id, a.learning_rate, a.reg_noise_std = img_id, lr, reg
                a.iters = max(cps) + 1
                start = time.time()
                res = runDIP(a, lpips_fn, cps, log=False)
                for iteration, p, s, l in res:
                    results.setdefault((lr, reg, iteration), []).append((p, s, l))
                    w.writerow([img_id, lr, reg, iteration, f"PSNR {p:.4f} SSIM {s:.4f} LPIPS {l:.4f}"])
                f.flush()
                torch.cuda.empty_cache()
                print(f"{img_id} lr: {lr} reg: {reg} PSNR: {p:.4f} SSIM: {s:.4f} LPIPS: {l:.4f} "
                      f"time elapsed: {time.time() - start:.2f}s")
        
        sortedRes = sorted(((np.mean([r[0] for r in v]), np.mean([r[1] for r in v]),
                            np.mean([r[2] for r in v]), k) for k, v in results.items()), reverse=True)
        for p, s, l, (lr, reg, iteration) in sortedRes[:10]:
            print(f"lr: {lr} reg: {reg} iter: {iteration} PSNR: {p:.4f} SSIM: {s:.4f} LPIPS: {l:.4f}")

def main(args):
    lpips_fn = lpips.LPIPS(net="alex").to(DEVICE)
    if args.tune_hyperparams:
        searchParams(args, lpips_fn)
    else:
        runDIP(args, lpips_fn)
    
if __name__ == "__main__":
    main(parser.parse_args())
    
    # TODO: seed noise for comparison
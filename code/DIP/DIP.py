import numpy as np
from models import *

import torch
import torch.nn.functional as F

from skimage.metrics import peak_signal_noise_ratio as psnr_fn
from utils.denoising_utils import *
import matplotlib.pyplot as plt 

from PIL import Image

import argparse
import os

parser = argparse.ArgumentParser(
    description="Train a Deep Image Prior model on DIV2k",
    formatter_class=argparse.ArgumentDefaultsHelpFormatter,
)
parser.add_argument("--img-id", default="0801")
parser.add_argument("--scale", default=8, type=int)
parser.add_argument("--crop", default=512, type=int)
parser.add_argument("--iters", type=int, default=2000)
parser.add_argument("--learning-rate", default=0.01, type=float)
parser.add_argument("--reg-noise-std", default=1./30, type=float)

DEVICE = torch.device('cuda')
torch.backends.cudnn.benchmark = True
dtype = torch.cuda.FloatTensor

HR_DIR = '../../data/DIV2K_valid_HR'
LR_DIR = '../../data/DIV2K_valid_LR_x8'

# Crop images so that the HR and LR pairs align with their scaling
# TODO: comment out
def crop_image_pair(hr_img, lr_img, scale, crop):
    w = min(lr_img.size[0], hr_img.size[0] // scale)
    h = min(lr_img.size[1], hr_img.size[1] // scale)
    
    if crop:
        w, h = min(w, crop // scale), min(h, crop // scale)
    
    w, h = w // 4 * 4, h // 4 * 4
    
    return (hr_img.crop((0, 0, w * scale, h * scale)),
            lr_img.crop((0, 0, w, h)))    

# Define the degredation operator
def downsampler(x, factor):
    return F.interpolate(x, scale_factor=1/factor, mode='bicubic',
                         antialias=True, align_corners=False)

# Plots the LR, HR, and reconstructed image at iteration i
def plot_iteration(i, lr, hr, recon):
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

def main(args):
    # TODO: convert rgb?
    hr_pil = Image.open(f'{HR_DIR}/{args.img_id}.png').convert('RGB')
    lr_pil = Image.open(f'{LR_DIR}/{args.img_id}x8.png').convert('RGB')
    
    # TODO: double check downscaler
    
    hr_pil, lr_pil = crop_image_pair(hr_pil, lr_pil, args.scale, args.crop)
    
    img_np, lr_np = pil_to_np(hr_pil), pil_to_np(lr_pil)
    img_torch = np_to_torch(img_np).type(dtype)
    lr_torch = np_to_torch(lr_np).type(dtype) 
    print('HR', img_np.shape, 'LR', lr_np.shape)
    
    net = get_net(32, 'skip', 'reflection',
                  skip_n33d=128, skip_n33u=128, skip_n11=4,
                  num_scales=5, upsample_mode='bilinear').type(dtype)
    
    net_input = get_noise(32, 'noise', (img_np.shape[1], img_np.shape[2])).type(dtype).detach()
    net_input_saved, noise = net_input.detach().clone(), net_input.detach().clone()
    
    # Compute number of parameters
    s  = sum([np.prod(list(p.size())) for p in net.parameters()]);
    print ('Number of params: %d' % s)
    
    mse = torch.nn.MSELoss().type(dtype)
    
    i = 0
    
    os.makedirs("results", exist_ok=True)
    
    def closure():
        nonlocal i, net_input
        if args.reg_noise_std > 0:
            net_input = net_input_saved + (noise.normal_() * args.reg_noise_std)
            
        out = net(net_input)
        
        loss = mse(downsampler(out, args.scale), lr_torch)
        loss.backward()
        
        if i % 10 == 0:
            # TODO: ?
            p = psnr_fn(img_np, np.clip(torch_to_np(out), 0, 1), data_range=1)
            print ('Iteration: ', i, ' Loss: ', loss.item(), ' PSRN_gt: ', p)
        
        if i % 100 == 0:
            # plot_image_grid([
            #     img_np,
            #     np.clip(torch_to_np(out), 0, 1)
            # ], nrow=3)
            plot_iteration(i, lr_torch, img_np, out)
            
        i += 1
        return loss
    
    params = get_params('net', net, net_input)
    optimize('adam', params, closure, args.learning_rate, args.iters)

if __name__ == "__main__":
    main(parser.parse_args())

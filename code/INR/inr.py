import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
from torchvision.utils import save_image
import torchvision.transforms.functional as TF

from skimage.metrics import peak_signal_noise_ratio as psnr_fn, structural_similarity as ssim_fn

from PIL import Image

import lpips

import argparse
import os
import time

parser = argparse.ArgumentParser(
    description="Train a SIREN (INR) on DIV2k",
    formatter_class=argparse.ArgumentDefaultsHelpFormatter,
)
parser.add_argument("--img-id", default="0801")
parser.add_argument("--scale", default=8, type=int)
parser.add_argument("--crop", default=512, type=int)
parser.add_argument("--iters", default=2000, type=int)
parser.add_argument("--learning-rate", default=1e-4, type=float)
parser.add_argument("--seed", default=0, type=int)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

HR_DIR = '../../data/DIV2K_valid_HR'
LR_DIR = '../../data/DIV2K_valid_LR_x8'

class SineLayer(nn.Module):

    """ Linear layer followed by the sine activation

    If `is_first == True`, then it represents the first layer of the network.
    In this case, omega_0 is a frequency factor, which simply multiplies the activations before the nonlinearity.
    Note that it influences the initialization scheme.

    If `is_first == False`, then the weights will be divided by omega_0 so as to keep the magnitude of activations constant,
    but boost gradients to the weight matrix.
    """

    def __init__(self, in_features, out_features, bias=True, is_first=False, omega_0=30):
        super().__init__()
        self.omega_0 = omega_0
        self.is_first = is_first

        self.in_features = in_features
        # Initialize a linear layer with specified input and output features
        # 'bias' indicates whether to include a bias term
        self.linear = nn.Linear(in_features, out_features, bias=bias)
        self.init_weights()

    # initialize weights uniformly
    def init_weights(self):
        # diasble gradient calculation in initialization
        with torch.no_grad():
            if self.is_first:
                self.linear.weight.uniform_(-1 / self.in_features,
                                             1 / self.in_features)
            else:
                self.linear.weight.uniform_(-np.sqrt(6 / self.in_features) / self.omega_0,
                                             np.sqrt(6 / self.in_features) / self.omega_0)

    def forward(self, input):
        # 1. pass input through linear layer (self.linear layer performs the linear transformation on the input)
        x = self.linear(input)

        # 2. scale the output of the linear transformation by the frequency factor
        x = x * self.omega_0

        # 3. apply sine activation
        x = torch.sin(x)

        return x
    
class Siren(nn.Module):
    """ SIREN architecture """

    def __init__(self, in_features, out_features, hidden_features=256, hidden_layers=3, outermost_linear=False,
                 first_omega_0=30, hidden_omega_0=30.):
        super().__init__()

        self.net = []
        # add the first layer
        self.net.append(SineLayer(in_features, hidden_features,
                                  is_first=True, omega_0=first_omega_0))

        # append hidden layers
        for i in range(hidden_layers):
            self.net.append(SineLayer(hidden_features, hidden_features,
                                      is_first=False, omega_0=hidden_omega_0))


        if outermost_linear:
            # add a final Linear layer
            final_linear = nn.Linear(hidden_features, out_features)

            with torch.no_grad(): # weights intialization
                final_linear.weight.uniform_(-np.sqrt(6 / hidden_features) / hidden_omega_0,
                                              np.sqrt(6 / hidden_features) / hidden_omega_0)

            self.net.append(final_linear)
        else:
            # otherwise, add a SineLayer
            self.net.append(SineLayer(hidden_features, out_features,
                                      is_first=False, omega_0=hidden_omega_0))

        self.net = nn.Sequential(*self.net) # sequential wrapper of SineLayer and Linear

    def forward(self, coords):
        # coords represents the 2D pixel coordinates
        coords = coords.clone().detach().requires_grad_(True) # allows to take derivative w.r.t. input
        output = self.net(coords)
        return output, coords

# Define the degredation operator
def downsampler(x, factor):
    return F.interpolate(x, scale_factor=1/factor, mode='bicubic',
                         antialias=True, align_corners=False)
    
def mgrid(h, w):
    ys = (torch.arange(h) + 0.5) / h * 2 - 1
    xs = (torch.arange(w) + 0.5) / w * 2 - 1
    return torch.stack(torch.meshgrid(ys, xs, indexing="ij"), dim=-1).reshape(-1, 2)
    
def crop_image(img, d=32):
    '''Make dimensions divisible by `d`'''

    new_size = (img.size[0] - img.size[0] % d, 
                img.size[1] - img.size[1] % d)

    bbox = [
            int((img.size[0] - new_size[0])/2), 
            int((img.size[1] - new_size[1])/2),
            int((img.size[0] + new_size[0])/2),
            int((img.size[1] + new_size[1])/2),
    ]

    img_cropped = img.crop(bbox)
    return img_cropped

def crop_image_pair(hr_img, lr_img, scale, crop):
    lr_crop = crop_image(lr_img, d=4)
    x0 = (lr_img.size[0] - lr_crop.size[0]) // 2
    y0 = (lr_img.size[1] - lr_crop.size[1]) // 2
    w, h = lr_crop.size
    
    if crop:
        cw = min(w, crop // scale) // 4 * 4
        ch = min(h, crop // scale) // 4 * 4
        x0 += (w - cw) // 2
        y0 += (h - ch) // 2
        w, h = cw, ch
        
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
    hr_pil = Image.open(f'{HR_DIR}/{args.img_id}.png').convert('RGB')
    lr_pil = Image.open(f'{LR_DIR}/{args.img_id}x8.png').convert('RGB')
    hr_pil, lr_pil = crop_image_pair(hr_pil, lr_pil, args.scale, args.crop)
    hr_img = TF.to_tensor(hr_pil)[None].to(DEVICE)
    lr_img = TF.to_tensor(lr_pil)[None].to(DEVICE)
    H, W = hr_img.shape[-2:]
    
    siren = Siren(in_features=2, out_features=3, hidden_features=256,
                 hidden_layers=3, outermost_linear=True).to(DEVICE)
    hr_coords = mgrid(H, W).to(DEVICE)
    optim = torch.optim.Adam(lr=args.learning_rate, params=siren.parameters())
    lpips_fn = lpips.LPIPS(net="alex").to(DEVICE)
    bicubic = F.interpolate(lr_img, size=(H, W), mode="bicubic", align_corners=False).clamp(0, 1)
    
    os.makedirs("results", exist_ok=True)
    start = time.time()
    for i in range(1, args.iters + 1):
        out, _ = siren(hr_coords)
        pred = ((out + 1) / 2).reshape(H, W, 3).permute(2, 0, 1)[None]
        loss = F.mse_loss(downsampler(pred, args.scale), lr_img)
        optim.zero_grad()
        loss.backward()
        optim.step()
        
        if i % 10 == 0 or i == args.iters:
            sr = render(siren, hr_coords, H, W)
            p = psnr_fn(hr_img[0].cpu().numpy(), sr.clamp(0, 1)[0].cpu().numpy(), data_range=1)
            print ('Iteration: ', i, ' Loss: ', loss.item(), ' PSRN_gt: ', p)        

    elapsed = time.time() - start
    
    for name, img in [("bicubic", bicubic), ("siren", sr)]:
        p, s, l = evaluate(img, hr_img, lpips_fn)
        print (f"{name} PSNR {p:.4f} SSIM {s:.4f} LPIPS {l:.4f}")
    print(f"time elapsed: {elapsed:.2f}s")     
    
    save_image(torch.cat([bicubic, sr.clamp(0, 1), hr_img]), f"results/inr_{args.img_id}.png", nrow=3)
        
if __name__ == "__main__":
    main(parser.parse_args())
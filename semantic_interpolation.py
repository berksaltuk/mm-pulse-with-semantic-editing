from manipulator import linear_interpolate
from tqdm import tqdm
import torch
import pickle
import torchvision
from PIL import Image
import numpy as np


def semantic_interpolation(latent_codes, boundary, out_path):

    toPIL = torchvision.transforms.ToPILImage()
    # Select a sample ID to perform interpolation
    i = 0
    for l in latent_codes:

        for latent in l:

            interpolations = linear_interpolate(
                latent, boundary, -3.0, 3.0, 10)

            G = None
            with open('stylegan2-celebahq-256x256.pkl', 'rb') as f:
                G = pickle.load(f)['G_ema'].cuda()
            c = None
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
            interpolation_id = 0
            gen_im = (G.synthesis(
                torch.from_numpy(interpolations).to(device), noise_mode='const', force_fp32=True) + 1) / 2
            image_name = i
            toPIL(gen_im[0].cpu().detach().clamp(0, 1)).save(
                out_path / f"edit/{image_name}.png")
            i += 1

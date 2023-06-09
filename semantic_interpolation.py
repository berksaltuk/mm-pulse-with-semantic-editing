from manipulator import linear_interpolate
from tqdm import tqdm
import torch
import pickle
import torchvision
from PIL import Image
import numpy as np
from pathlib import Path

toPIL = torchvision.transforms.ToPILImage()
# Select a sample ID to perform interpolation

out_path = Path("edit")
out_path.mkdir(parents=True, exist_ok=True)

G = None
with open('stylegan2-celebahq-256x256.pkl', 'rb') as f:
    G = pickle.load(f)['G_ema'].cuda()
device = 'cuda' if torch.cuda.is_available() else 'cpu'
latent_codes = np.load('/content/pulse/latent_vectors/w.npy')
latent_codes = latent_codes.reshape((100, 14, 512))
latent_codes = latent_codes[0]
boundary = np.load('/content/pulse/our_boundaries/boundary.npy')
j = 0
for latent in latent_codes.reshape(1, 14, 512):
    interpolations = [linear_interpolate(latent.reshape((1, 14, 512)), boun.reshape(
        1, 512), -3, 3, 10) for boun in boundary.reshape((14, 512))]

    i = 0
    print(len(interpolations))
    for pol in interpolations:
        gen_im = (G.synthesis(
            torch.from_numpy(pol.reshape((10, 14, 512))).to(device), noise_mode='const', force_fp32=True) + 1) / 2
        toPIL(gen_im[0].cpu().detach().clamp(0, 1)).save(
            out_path / f"{j}_{i}.png")
        i += 1
    j += 1

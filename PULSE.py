from stylegan import G_synthesis, G_mapping
from dataclasses import dataclass
from SphericalOptimizer import SphericalOptimizer
from pathlib import Path
import numpy as np
import time
import torch
from loss import LossBuilder
from functools import partial
from drive import open_url
import dnnlib
import torch_utils
import pickle


class PULSE(torch.nn.Module):
    def __init__(self, cache_dir, verbose=True):
        super(PULSE, self).__init__()

        self.G = None
        with open('stylegan2-ffhq-1024x1024.pkl', 'rb') as f:
            self.G = pickle.load(f)['G_ema'].cuda()  # torch.nn.Module

        self.verbose = verbose

        cache_dir = Path(cache_dir)
        cache_dir.mkdir(parents=True, exist_ok=True)

        self.lrelu = torch.nn.LeakyReLU(negative_slope=0.2)

        if self.verbose:
            print("\tLoading Mapping Network")
        self.mapping = self.G.mapping(z, c,
                                      truncation_psi=0.5, truncation_cutoff=8).cuda()

    def forward(self, ref_im,
                seed,
                loss_str,
                eps,
                noise_type,
                num_trainable_noise_layers,
                tile_latent,
                bad_noise_layers,
                opt_name,
                learning_rate,
                steps,
                lr_schedule,
                save_intermediate,
                **kwargs):

        if seed:
            torch.manual_seed(seed)
            torch.cuda.manual_seed(seed)
            torch.backends.cudnn.deterministic = True

        batch_size = ref_im.shape[0]

        # Generate latent tensor
        if (tile_latent):
            latent = torch.randn(
                (batch_size, 1, 512), dtype=torch.float, requires_grad=True, device='cuda')
        else:
            latent = torch.randn(
                (batch_size, 18, 512), dtype=torch.float, requires_grad=True, device='cuda')

        # Generate list of noise tensors
        noise = []  # stores all of the noise tensors
        noise_vars = []  # stores the noise tensors that we want to optimize on

        for i in range(18):
            # dimension of the ith noise tensor
            res = (batch_size, 1, 2**(i//2+2), 2**(i//2+2))

            if (noise_type == 'zero' or i in [int(layer) for layer in bad_noise_layers.split('.')]):
                new_noise = torch.zeros(res, dtype=torch.float, device='cuda')
                new_noise.requires_grad = False
            elif (noise_type == 'fixed'):
                new_noise = torch.randn(res, dtype=torch.float, device='cuda')
                new_noise.requires_grad = False
            elif (noise_type == 'trainable'):
                new_noise = torch.randn(res, dtype=torch.float, device='cuda')
                if (i < num_trainable_noise_layers):
                    new_noise.requires_grad = True
                    noise_vars.append(new_noise)
                else:
                    new_noise.requires_grad = False
            else:
                raise Exception("unknown noise type")

            noise.append(new_noise)

        var_list = [latent]+noise_vars

        opt_dict = {
            'sgd': torch.optim.SGD,
            'adam': torch.optim.Adam,
            'sgdm': partial(torch.optim.SGD, momentum=0.9),
            'adamax': torch.optim.Adamax
        }
        opt_func = opt_dict[opt_name]
        opt = SphericalOptimizer(opt_func, var_list, lr=learning_rate)

        schedule_dict = {
            'fixed': lambda x: 1,
            'linear1cycle': lambda x: (9*(1-np.abs(x/steps-1/2)*2)+1)/10,
            'linear1cycledrop': lambda x: (9*(1-np.abs(x/(0.9*steps)-1/2)*2)+1)/10 if x < 0.9*steps else 1/10 + (x-0.9*steps)/(0.1*steps)*(1/1000-1/10),
        }
        schedule_func = schedule_dict[lr_schedule]
        scheduler = torch.optim.lr_scheduler.LambdaLR(opt.opt, schedule_func)

        loss_builder = LossBuilder(ref_im, loss_str, eps).cuda()

        min_loss = np.inf
        min_l2 = np.inf
        best_summary = ""
        start_t = time.time()
        gen_im = None

        if self.verbose:
            print("Optimizing")

        mapping_mean = torch.mean(self.mapping)
        mapping_std = torch.std(self.mapping)
        print("Mean of G.mapping:",
              mapping_mean.item())
        print("Std deviation of G.mapping:",
              mapping_std.item())
        for j in range(steps):
            opt.opt.zero_grad()

            # Duplicate latent in case tile_latent = True
            if (tile_latent):
                latent_in = latent.expand(-1, 18, -1)
            else:
                latent_in = latent

            # Apply learned linear mapping to match latent distribution to that of the mapping network
            latent_in = self.lrelu(
                latent_in*mapping_std + mapping_mean)

            # Normalize image to [0,1] instead of [-1,1]
            gen_im = (self.G.synthesis(
                latent_in, noise_mode='random', force_fp32=True)+1)/2

            # Calculate Losses
            loss, loss_dict = loss_builder(latent_in, gen_im)
            loss_dict['TOTAL'] = loss

            # Save best summary for log
            if (loss < min_loss):
                min_loss = loss
                best_summary = f'BEST ({j+1}) | '+' | '.join(
                    [f'{x}: {y:.4f}' for x, y in loss_dict.items()])
                best_im = gen_im.clone()

            loss_l2 = loss_dict['L2']

            if (loss_l2 < min_l2):
                min_l2 = loss_l2

            # Save intermediate HR and LR images
            if (save_intermediate):
                yield (best_im.cpu().detach().clamp(0, 1), loss_builder.D(best_im).cpu().detach().clamp(0, 1))

            loss.backward()
            opt.step()
            scheduler.step()

        total_t = time.time()-start_t
        current_info = f' | time: {total_t:.1f} | it/s: {(j+1)/total_t:.2f} | batchsize: {batch_size}'
        if self.verbose:
            print(best_summary+current_info)

        yield (gen_im.clone().cpu().detach().clamp(0, 1), loss_builder.D(best_im).cpu().detach().clamp(0, 1))

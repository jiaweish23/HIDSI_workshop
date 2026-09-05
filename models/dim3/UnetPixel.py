"""Standard multi-scale 3D UNet for PIXEL-space diffusion at 32^3.

Identical to models.dim3.Modules.Unet (reuses its ResnetBlock / attention /
time-embedding / Down-Upsample blocks and the exact same forward) EXCEPT the
down/up-sampling schedule: the shared `Unet` was hand-tuned for the 8^3 VAE
latent (it downsamples only ONCE), which is wrong for a 32^3 voxel field.  Here
every level except the bottleneck down/upsamples, giving 32->16->8->4 and full
self-attention only at the 4^3 bottleneck (64 tokens, cheap).  Nothing in the
shared Modules.py is modified, so all latent models are unaffected.
"""
import torch
import torch.nn as nn
from functools import partial

from models.dim3.Modules import (
    default, ResnetBlock, Residual, PreNorm, LinearAttention, Attention,
    Downsample, Upsample, SinusoidalPositionEmbeddings,
)


class UnetPixel(nn.Module):
    def __init__(self, dim, init_dim=None, out_dim=None, dim_mults=(1, 2, 4, 8),
                 channels=1, resnet_block_groups=8):
        super().__init__()
        self.channels = channels
        init_dim = default(init_dim, dim)
        self.init_conv = nn.Conv3d(channels, init_dim, 3, padding=1)

        dims = [init_dim, *map(lambda m: dim * m, dim_mults)]
        in_out = list(zip(dims[:-1], dims[1:]))
        block_klass = partial(ResnetBlock, kernel_size=3, groups=resnet_block_groups)

        time_dim = dim
        self.time_mlp = nn.Sequential(
            SinusoidalPositionEmbeddings(dim),
            nn.Linear(dim, time_dim), nn.GELU(), nn.Linear(time_dim, time_dim),
        )

        self.downs = nn.ModuleList([])
        self.ups = nn.ModuleList([])
        num_resolutions = len(in_out)

        for ind, (dim_in, dim_out) in enumerate(in_out):
            is_last = ind >= (num_resolutions - 1)
            self.downs.append(nn.ModuleList([
                block_klass(dim_in, dim_in, time_emb_dim=time_dim),
                block_klass(dim_in, dim_in, time_emb_dim=time_dim),
                Residual(PreNorm(dim_in, LinearAttention(dim_in))),
                Downsample(dim_in, dim_out) if not is_last
                else nn.Conv3d(dim_in, dim_out, 3, padding=1),
            ]))

        mid_dim = dims[-1]
        self.mid_block1 = block_klass(mid_dim, mid_dim, time_emb_dim=time_dim)
        self.mid_attn = Residual(PreNorm(mid_dim, Attention(mid_dim)))
        self.mid_block2 = block_klass(mid_dim, mid_dim, time_emb_dim=time_dim)

        for ind, (dim_in, dim_out) in enumerate(reversed(in_out)):
            is_last = ind >= (num_resolutions - 1)
            self.ups.append(nn.ModuleList([
                block_klass(dim_out + dim_in, dim_out, time_emb_dim=time_dim),
                block_klass(dim_out + dim_in, dim_out, time_emb_dim=time_dim),
                Residual(PreNorm(dim_out, LinearAttention(dim_out))),
                Upsample(dim_out, dim_in) if not is_last
                else nn.Conv3d(dim_out, dim_in, 3, padding=1),
            ]))

        self.out_dim = default(out_dim, channels)
        self.final_res_block = block_klass(dim * 2, dim, time_emb_dim=time_dim)
        self.final_conv = nn.Conv3d(dim, self.out_dim, 1)

    def forward(self, x, time, x_self_cond=None):
        x = self.init_conv(x)
        r = x.clone()
        t = self.time_mlp(time)
        h = []
        for block1, block2, attn, downsample in self.downs:
            x = block1(x, t); h.append(x)
            x = block2(x, t); x = attn(x); h.append(x)
            x = downsample(x)
        x = self.mid_block1(x, t); x = self.mid_attn(x); x = self.mid_block2(x, t)
        for block1, block2, attn, upsample in self.ups:
            x = torch.cat((x, h.pop()), dim=1); x = block1(x, t)
            x = torch.cat((x, h.pop()), dim=1); x = block2(x, t); x = attn(x)
            x = upsample(x)
        x = torch.cat((x, r), dim=1)
        x = self.final_res_block(x, t)
        return self.final_conv(x)

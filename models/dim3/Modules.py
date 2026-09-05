import torch
from torch import nn, einsum
import torch.nn.functional as F
import math
from einops import rearrange, reduce
# einops tutorial: https://github.com/arogozhnikov/einops/blob/master/docs/1-einops-basics.ipynb
from functools import partial
from inspect import isfunction
from einops.layers.torch import Rearrange

from collections import deque


def exists(x):
    return x is not None


def default(val, d):
    if exists(val):
        return val
    return d() if isfunction(d) else d


class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, *args, **kwargs):
        return self.fn(x, *args, **kwargs) + x


# Define an upsample operation
def Upsample(dim, dim_out=None):
    return nn.Sequential(
        nn.Upsample(scale_factor=2, mode="nearest"),
        nn.Conv3d(dim, default(dim_out, dim), 3, padding=1),
    )


# Define a downsample operation
def Downsample(dim, dim_out=None):
    # No More Strided Convolutions or Pooling
    return nn.Sequential(
        Rearrange("b c (h p1) (w p2) (d p3)-> b (c p1 p2 p3) h w d", p1=2, p2=2, p3=2),
        nn.Conv3d(dim * 8, default(dim_out, dim), 1),
    )


class SinusoidalPositionEmbeddings(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, time):
        device = time.device
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings


class WeightStandardizedConv3d(nn.Conv3d):
    """
    https://arxiv.org/abs/1903.10520
    weight standardization purportedly works synergistically with group normalization
    """

    def forward(self, x):
        eps = 1e-5 if x.dtype == torch.float32 else 1e-3

        weight = self.weight
        mean = reduce(weight, "o ... -> o 1 1 1 1", "mean")
        var = reduce(weight, "o ... -> o 1 1 1 1", partial(torch.var, unbiased=False))
        normalized_weight = (weight - mean) * (var + eps).rsqrt()

        return F.conv3d(
            x,
            normalized_weight,
            self.bias,
            self.stride,
            self.padding,
            self.dilation,
            self.groups,
        )


class Block(nn.Module):
    def __init__(self, dim, dim_out, kernel_size=3, groups=8):
        super().__init__()
        self.proj = WeightStandardizedConv3d(dim, dim_out, kernel_size, padding=kernel_size // 2)
        self.norm = nn.GroupNorm(groups, dim_out)
        self.act = nn.SiLU()

    def forward(self, x, scale_shift=None):
        x = self.proj(x)
        x = self.norm(x)

        if scale_shift:
            scale, shift = scale_shift
            x = x * (scale + 1) + shift

        x = self.act(x)
        return x


class ResnetBlock(nn.Module):
    """https://arxiv.org/abs/1512.03385"""

    def __init__(self, dim, dim_out, *, kernel_size=3, time_emb_dim=None, groups=8):
        super().__init__()
        self.mlp = (
            nn.Sequential(nn.SiLU(), nn.Linear(time_emb_dim, dim_out * 2))
            if time_emb_dim
            else None
        )

        self.block1 = Block(dim, dim_out, kernel_size=kernel_size, groups=groups)
        self.block2 = Block(dim_out, dim_out, kernel_size=kernel_size, groups=groups)
        self.res_conv = nn.Conv3d(dim, dim_out, 1) if dim != dim_out else nn.Identity()

    def forward(self, x, time_emb=None):
        scale_shift = None
        if self.mlp is not None and time_emb is not None:
            time_emb = self.mlp(time_emb)
            time_emb = rearrange(time_emb, "b c -> b c 1 1 1")
            scale_shift = time_emb.chunk(2, dim=1)

        h = self.block1(x, scale_shift=scale_shift)
        h = self.block2(h)
        return h + self.res_conv(x)


class Attention(nn.Module):
    def __init__(self, dim, heads=4, dim_head=32):
        super().__init__()
        self.scale = dim_head**-0.5
        self.heads = heads
        hidden_dim = dim_head * heads
        self.to_qkv = nn.Conv3d(dim, hidden_dim * 3, 1, bias=False)
        self.to_out = nn.Conv3d(hidden_dim, dim, 1)

    def forward(self, x):
        b, c, h, w, d = x.shape
        qkv = self.to_qkv(x).chunk(3, dim=1)
        q, k, v = map(
            lambda t: rearrange(t, "b (h c) x y z -> b h c (x y z)", h=self.heads), qkv
        )
        q = q * self.scale

        sim = einsum("b h d i, b h d j -> b h i j", q, k)
        sim = sim - sim.amax(dim=-1, keepdim=True).detach()
        attn = sim.softmax(dim=-1)

        out = einsum("b h i j, b h d j -> b h i d", attn, v)
        out = rearrange(out, "b h (x y z) d -> b (h d) x y z", x=h, y=w, z=d)
        return self.to_out(out)


class LinearAttention(nn.Module):
    def __init__(self, dim, heads=4, dim_head=32):
        super().__init__()
        self.scale = dim_head**-0.5
        self.heads = heads
        hidden_dim = dim_head * heads
        self.to_qkv = nn.Conv3d(dim, hidden_dim * 3, 1, bias=False)

        self.to_out = nn.Sequential(nn.Conv3d(hidden_dim, dim, 1),
                                    nn.GroupNorm(1, dim))

    def forward(self, x):
        b, c, h, w, d = x.shape
        qkv = self.to_qkv(x).chunk(3, dim=1)
        q, k, v = map(
            lambda t: rearrange(t, "b (h c) x y z -> b h c (x y z)", h=self.heads), qkv
        )

        q = q.softmax(dim=-2)
        k = k.softmax(dim=-1)

        q = q * self.scale
        context = torch.einsum("b h d n, b h e n -> b h d e", k, v)

        out = torch.einsum("b h d e, b h d n -> b h e n", context, q)
        out = rearrange(out, "b h c (x y z) -> b (h c) x y z", h=self.heads, x=h, y=w, z=d)
        return self.to_out(out)


class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.fn = fn
        self.norm = nn.GroupNorm(1, dim)

    def forward(self, x):
        x = self.norm(x)
        return self.fn(x)


class Unet(nn.Module):
    def __init__(
            self,
            dim,
            init_dim=None,
            out_dim=None,
            dim_mults=(1, 2, 4, 8),
            channels=3,
            self_condition=False,
            resnet_block_groups=4,
    ):
        super().__init__()

        # determine dimensions
        self.channels = channels
        self.self_condition = self_condition
        input_channels = channels * (2 if self_condition else 1)

        init_dim = default(init_dim, dim)
        self.init_conv = nn.Conv3d(input_channels, init_dim, 3, padding=1)

        dims = [init_dim, *map(lambda m: dim * m, dim_mults)]
        in_out = list(zip(dims[:-1], dims[1:]))

        block_klass = partial(ResnetBlock, kernel_size=3, groups=resnet_block_groups)

        # time embeddings
        time_dim = dim

        self.time_mlp = nn.Sequential(
            SinusoidalPositionEmbeddings(dim),
            nn.Linear(dim, time_dim),
            nn.GELU(),
            nn.Linear(time_dim, time_dim),
        )

        # layers
        self.downs = nn.ModuleList([])
        self.ups = nn.ModuleList([])
        num_resolutions = len(in_out)

        # downsampling stage
        for ind, (dim_in, dim_out) in enumerate(in_out):
            is_last = ind >= (num_resolutions - 1)
            downsample = ind > 1.5  and ind < (num_resolutions - 1)

            self.downs.append(
                nn.ModuleList(
                    [
                        block_klass(dim_in, dim_in, time_emb_dim=time_dim),
                        block_klass(dim_in, dim_in, time_emb_dim=time_dim),
                        Residual(PreNorm(dim_in, LinearAttention(dim_in))),
                        Downsample(dim_in, dim_out)
                        if downsample
                        else nn.Conv3d(dim_in, dim_out, 3, padding=1),
                    ]
                )
            )

        # middle part
        mid_dim = dims[-1]
        self.mid_block1 = block_klass(mid_dim, mid_dim, time_emb_dim=time_dim)
        self.mid_attn = Residual(PreNorm(mid_dim, Attention(mid_dim)))
        self.mid_block2 = block_klass(mid_dim, mid_dim, time_emb_dim=time_dim)

        # upsampling stage
        for ind, (dim_in, dim_out) in enumerate(reversed(in_out)):
            is_last = ind >= (len(in_out) - 1)
            upsample = ind < (num_resolutions - 3.5)

            self.ups.append(
                nn.ModuleList(
                    [
                        block_klass(dim_out + dim_in, dim_out, time_emb_dim=time_dim),
                        block_klass(dim_out + dim_in, dim_out, time_emb_dim=time_dim),
                        Residual(PreNorm(dim_out, LinearAttention(dim_out))),
                        Upsample(dim_out, dim_in)
                        if upsample
                        else nn.Conv3d(dim_out, dim_in, 3, padding=1),
                    ]
                )
            )

        self.out_dim = default(out_dim, channels)

        self.final_res_block = block_klass(dim * 2, dim, time_emb_dim=time_dim)
        self.final_conv = nn.Conv3d(dim, self.out_dim, 1)


    def forward(self, x, time, x_self_cond=None):
        if self.self_condition:
            x_self_cond = default(x_self_cond, lambda: torch.zeros_like(x))
            x = torch.cat((x_self_cond, x), dim=1)

        x = self.init_conv(x)
        r = x.clone()

        t = self.time_mlp(time)

        h = []

        for block1, block2, attn, downsample in self.downs:
            x = block1(x, t)
            h.append(x)

            x = block2(x, t)
            x = attn(x)
            h.append(x)

            x = downsample(x)

        x = self.mid_block1(x, t)
        x = self.mid_attn(x)
        x = self.mid_block2(x, t)

        for block1, block2, attn, upsample in self.ups:
            x = torch.cat((x, h.pop()), dim=1)
            x = block1(x, t)

            x = torch.cat((x, h.pop()), dim=1)
            x = block2(x, t)
            x = attn(x)

            x = upsample(x)

        x = torch.cat((x, r), dim=1)

        x = self.final_res_block(x, t)
        return self.final_conv(x)


class Unet_SR(nn.Module):
    """3D denoising UNet with channel-concat conditioning (3D analogue of the 2D
    Unet_SR).  A 1-channel `context` volume (same spatial shape as the latent) is
    concatenated to the input before init_conv; when context is None a zero volume
    is concatenated so the unconditional path (CFG dropout) shares the same weights.
    Architecture mirrors the unconditional 3D `Unet` above exactly, except init_conv
    takes one extra input channel for the context.
    """

    def __init__(
            self,
            dim,
            init_dim=None,
            out_dim=None,
            dim_mults=(1, 2, 4, 8),
            channels=1,
            self_condition=False,
            resnet_block_groups=4,
    ):
        super().__init__()

        self.channels = channels
        self.self_condition = self_condition
        input_channels = channels * (2 if self_condition else 1)

        init_dim = default(init_dim, dim)
        # +1 input channel for the concatenated context volume
        self.init_conv = nn.Conv3d(input_channels + 1, init_dim, 3, padding=1)

        dims = [init_dim, *map(lambda m: dim * m, dim_mults)]
        in_out = list(zip(dims[:-1], dims[1:]))

        block_klass = partial(ResnetBlock, kernel_size=3, groups=resnet_block_groups)

        time_dim = dim
        self.time_mlp = nn.Sequential(
            SinusoidalPositionEmbeddings(dim),
            nn.Linear(dim, time_dim),
            nn.GELU(),
            nn.Linear(time_dim, time_dim),
        )

        self.downs = nn.ModuleList([])
        self.ups = nn.ModuleList([])
        num_resolutions = len(in_out)

        for ind, (dim_in, dim_out) in enumerate(in_out):
            is_last = ind >= (num_resolutions - 1)
            downsample = ind > 1.5 and ind < (num_resolutions - 1)
            self.downs.append(
                nn.ModuleList(
                    [
                        block_klass(dim_in, dim_in, time_emb_dim=time_dim),
                        block_klass(dim_in, dim_in, time_emb_dim=time_dim),
                        Residual(PreNorm(dim_in, LinearAttention(dim_in))),
                        Downsample(dim_in, dim_out)
                        if downsample
                        else nn.Conv3d(dim_in, dim_out, 3, padding=1),
                    ]
                )
            )

        mid_dim = dims[-1]
        self.mid_block1 = block_klass(mid_dim, mid_dim, time_emb_dim=time_dim)
        self.mid_attn = Residual(PreNorm(mid_dim, Attention(mid_dim)))
        self.mid_block2 = block_klass(mid_dim, mid_dim, time_emb_dim=time_dim)

        for ind, (dim_in, dim_out) in enumerate(reversed(in_out)):
            is_last = ind >= (len(in_out) - 1)
            upsample = ind < (num_resolutions - 3.5)
            self.ups.append(
                nn.ModuleList(
                    [
                        block_klass(dim_out + dim_in, dim_out, time_emb_dim=time_dim),
                        block_klass(dim_out + dim_in, dim_out, time_emb_dim=time_dim),
                        Residual(PreNorm(dim_out, LinearAttention(dim_out))),
                        Upsample(dim_out, dim_in)
                        if upsample
                        else nn.Conv3d(dim_out, dim_in, 3, padding=1),
                    ]
                )
            )

        self.out_dim = default(out_dim, channels)
        self.final_res_block = block_klass(dim * 2, dim, time_emb_dim=time_dim)
        self.final_conv = nn.Conv3d(dim, self.out_dim, 1)

    def forward(self, x, time, context=None, x_self_cond=None):
        if self.self_condition:
            x_self_cond = default(x_self_cond, lambda: torch.zeros_like(x))
            x = torch.cat((x_self_cond, x), dim=1)

        if context is not None:
            x = torch.cat((x, context), dim=1)
        else:
            x = torch.cat((x, torch.zeros_like(x[:, :1])), dim=1)

        x = self.init_conv(x)
        r = x.clone()

        t = self.time_mlp(time)

        h = []
        for block1, block2, attn, downsample in self.downs:
            x = block1(x, t)
            h.append(x)
            x = block2(x, t)
            x = attn(x)
            h.append(x)
            x = downsample(x)

        x = self.mid_block1(x, t)
        x = self.mid_attn(x)
        x = self.mid_block2(x, t)

        for block1, block2, attn, upsample in self.ups:
            x = torch.cat((x, h.pop()), dim=1)
            x = block1(x, t)
            x = torch.cat((x, h.pop()), dim=1)
            x = block2(x, t)
            x = attn(x)
            x = upsample(x)

        x = torch.cat((x, r), dim=1)
        x = self.final_res_block(x, t)
        return self.final_conv(x)


class UinUnet(nn.Module):
    def __init__(
            self,
            dim,
            init_dim=None,
            out_dim=None,
            dim_mults=(1, 2, 4),
            channels=3,
            self_condition=False,
            resnet_block_groups=4,
    ):
        super().__init__()

        # determine dimensions
        self.channels = channels
        self.self_condition = self_condition
        input_channels = channels * (2 if self_condition else 1)

        init_dim = default(init_dim, dim)
        self.init_conv = nn.Conv3d(input_channels, init_dim, 3, padding=1)

        dims = [init_dim, *map(lambda m: dim * m, dim_mults)]
        in_out = list(zip(dims[:-1], dims[1:]))

        block_klass = partial(ResnetBlock, kernel_size=3, groups=resnet_block_groups)
        block_klass_Uin = partial(ResnetBlock, kernel_size=1, groups=resnet_block_groups)

        # time embeddings
        time_dim = dim

        self.time_mlp = nn.Sequential(
            SinusoidalPositionEmbeddings(dim),
            nn.Linear(dim, time_dim),
            nn.GELU(),
            nn.Linear(time_dim, time_dim),
        )

        # layers
        self.downs = nn.ModuleList([])
        self.ups = nn.ModuleList([])

        self.downs_Uin = nn.ModuleList([])
        self.ups_Uin = nn.ModuleList([])

        num_resolutions = len(in_out)

        # downsampling stage
        for ind, (dim_in, dim_out) in enumerate(in_out):
            is_last = ind >= (num_resolutions - 1)
            #downsample = ind > 1.5 and ind < (num_resolutions - 1)


            if ind < 1.5:
                self.downs_Uin.append(
                    nn.ModuleList(
                        [
                        block_klass_Uin(dim_in, dim_in, time_emb_dim=time_dim),
                        block_klass_Uin(dim_in, dim_in, time_emb_dim=time_dim),
                        Residual(PreNorm(dim_in, LinearAttention(dim_in))),
                        nn.Conv3d(dim_in, dim_out, 1),
                        ]
                    )
                )
                mid_dim_Uin = dim_out

                self.downs.append(
                    nn.ModuleList(
                        [
                            block_klass(dim_in, dim_in, time_emb_dim=time_dim),
                            block_klass(dim_in, dim_in, time_emb_dim=time_dim),
                            Residual(PreNorm(dim_in, LinearAttention(dim_in))),
                            nn.Conv3d(dim_in, dim_out, 3, padding=1),
                        ]
                    )
                )

            else:
                self.downs.append(
                    nn.ModuleList(
                        [
                            block_klass(dim_in, dim_in, time_emb_dim=time_dim),
                            block_klass(dim_in, dim_in, time_emb_dim=time_dim),
                            Residual(PreNorm(dim_in, LinearAttention(dim_in))),
                            Downsample(dim_in, dim_out)
                            if not is_last
                            else nn.Conv3d(dim_in, dim_out, 3, padding=1),
                        ]
                    )
                )

        # middle part
        mid_dim = dims[-1]
        self.mid_block1 = block_klass(mid_dim, mid_dim, time_emb_dim=time_dim)
        self.mid_attn = Residual(PreNorm(mid_dim, Attention(mid_dim)))
        self.mid_block2 = block_klass(mid_dim, mid_dim, time_emb_dim=time_dim)

        self.mid_block1_Uin = block_klass_Uin(mid_dim_Uin, mid_dim_Uin, time_emb_dim=time_dim)
        self.mid_attn_Uin = Residual(PreNorm(mid_dim_Uin, Attention(mid_dim_Uin)))
        self.mid_block2_Uin = block_klass_Uin(mid_dim_Uin, mid_dim_Uin, time_emb_dim=time_dim)

        self.num_resolutions = num_resolutions

        # upsampling stage
        for ind, (dim_in, dim_out) in enumerate(reversed(in_out)):
            is_last = ind == (len(in_out) - 1)
            #upsample = ind <= (num_resolutions - 2)
            upsample = ind < (num_resolutions - 3.5)

            if ind > (num_resolutions-2.5):
                self.ups_Uin.append(
                    nn.ModuleList(
                        [
                            block_klass_Uin(dim_out + dim_in, dim_out, time_emb_dim=time_dim),
                            block_klass_Uin(dim_out + dim_in, dim_out, time_emb_dim=time_dim),
                            Residual(PreNorm(dim_out, LinearAttention(dim_out))),
                            nn.Conv3d(dim_out, dim_in, 1),
                        ]
                    )
                )

                self.ups.append(
                    nn.ModuleList(
                        [
                            block_klass(dim_out*2 + dim_in, dim_out, time_emb_dim=time_dim),
                            block_klass(dim_out + dim_in*2, dim_out, time_emb_dim=time_dim),
                            Residual(PreNorm(dim_out, LinearAttention(dim_out))),
                            nn.Conv3d(dim_out, dim_in, 3, padding=1),
                        ]
                    )
                )
            else:
                self.ups.append(
                    nn.ModuleList(
                        [
                            block_klass(dim_out + dim_in, dim_out, time_emb_dim=time_dim),
                            block_klass(dim_out + dim_in, dim_out, time_emb_dim=time_dim),
                            Residual(PreNorm(dim_out, LinearAttention(dim_out))),
                            Upsample(dim_out, dim_in)
                            if upsample
                            else nn.Conv3d(dim_out, dim_in, 3, padding=1),
                        ]
                    )
                )


        self.out_dim = default(out_dim, channels)

        self.final_res_block = block_klass(dim * 2, dim, time_emb_dim=time_dim)
        self.final_conv = nn.Conv3d(dim, self.out_dim, 1)

    def forward(self, x, time, x_self_cond=None):
        if self.self_condition:
            x_self_cond = default(x_self_cond, lambda: torch.zeros_like(x))
            x = torch.cat((x_self_cond, x), dim=1)

        x = self.init_conv(x)
        r = x.clone()

        t = self.time_mlp(time)

        h = []
        h_Uin = []
        h_Uin2 = deque()

        for block1, block2, attn, downsample in self.downs_Uin:
            x = block1(x, t)
            h_Uin.append(x)

            x = block2(x, t)
            x = attn(x)
            h_Uin.append(x)

            x = downsample(x)

        x = self.mid_block1_Uin(x, t)
        x = self.mid_attn_Uin(x)
        x = self.mid_block2_Uin(x, t)

        for block1, block2, attn, upsample in self.ups_Uin:
            x = torch.cat((x, h_Uin.pop()), dim=1)
            x = block1(x, t)
            h_Uin2.append(x)

            x = torch.cat((x, h_Uin.pop()), dim=1)
            x = block2(x, t)
            x = attn(x)
            x = upsample(x)
            h_Uin2.append(x)

        #x_Uin = x.clone()
        x = r
        for block1, block2, attn, downsample in self.downs:
            x = block1(x, t)
            h.append(x)

            x = block2(x, t)
            x = attn(x)
            h.append(x)

            x = downsample(x)

        x = self.mid_block1(x, t)
        x = self.mid_attn(x)
        x = self.mid_block2(x, t)

        for ind, (block1, block2, attn, upsample) in enumerate(self.ups):
            if ind > (self.num_resolutions - 2.5):
                x = torch.cat((x, h.pop(), h_Uin2.popleft()), dim=1)
                x = block1(x, t)

                x = torch.cat((x, h.pop(), h_Uin2.popleft()), dim=1)
                x = block2(x, t)
                x = attn(x)

                x = upsample(x)

            else:
                x = torch.cat((x, h.pop()), dim=1)
                x = block1(x, t)

                x = torch.cat((x, h.pop()), dim=1)
                x = block2(x, t)
                x = attn(x)

                x = upsample(x)

        x = torch.cat((x, r), dim=1)

        x = self.final_res_block(x, t)
        return self.final_conv(x)


class SE(nn.Module):
    def __init__(self, channel, reduction=4):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, inputs):
        return inputs * self.fc(inputs)


class ResBlockSEDrop(nn.Module):
    """
    fixed the conv0 not used error in ResBlockSE
    """

    def __init__(self, input_dim, output_dim, dropout):
        super().__init__()
        self.non_linearity = nn.ReLU(inplace=True)
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.conv1 = nn.Linear(input_dim, output_dim)
        self.conv2 = nn.Linear(output_dim, output_dim)
        in_ch = self.output_dim
        self.SE = SE(in_ch)
        self.dropout = nn.Dropout(dropout)
        self.dropout_ratio = dropout

        dim = input_dim
        time_dim = input_dim

        self.time_mlp = nn.Sequential(
            SinusoidalPositionEmbeddings(dim),
            nn.Linear(dim, time_dim),
            nn.GELU(),
            nn.Linear(time_dim, time_dim),
        )

    def forward(self, x, t):
        t = self.time_mlp(t)
        output = x + t
        output = self.conv1(output)
        output = self.non_linearity(output)
        output = self.dropout(output)
        output = self.conv2(output)
        output = self.non_linearity(output)
        output = self.SE(output)
        shortcut = x
        return shortcut + output


class Block1d(nn.Module):
    def __init__(self, dim, dim_out=None, groups=8):
        super().__init__()
        dim_out = default(dim_out, dim)
        self.proj = nn.Linear(dim, dim_out)
        self.norm = nn.GroupNorm(groups, dim_out)
        self.act = nn.SiLU()

    def forward(self, x, scale_shift=None):
        x = self.proj(x)
        x = self.norm(x)

        if scale_shift:
            scale, shift = scale_shift
            x = x * (scale + 1) + shift

        x = self.act(x)
        return x


class ResnetBlock1d(nn.Module):
    """https://arxiv.org/abs/1512.03385"""

    def __init__(self, dim, *, dim_out=None, time_emb_dim=None, groups=8):
        super().__init__()
        dim_out = default(dim_out, dim)
        self.mlp = (
            nn.Sequential(nn.SiLU(), nn.Linear(time_emb_dim, dim_out * 2))
            if time_emb_dim
            else None
        )

        self.block1 = Block1d(dim, dim_out, groups=groups)
        self.block2 = Block1d(dim_out, dim_out, groups=groups)
        self.linear = nn.Linear(dim, dim_out) if dim != dim_out else nn.Identity()


    def forward(self, x, time_emb=None):
        scale_shift = None
        if self.mlp is not None and time_emb is not None:
            time_emb = self.mlp(time_emb)
            scale_shift = time_emb.chunk(2, dim=1)

        h = self.block1(x, scale_shift=scale_shift)
        h = self.block2(h)
        return h + self.linear(x)


class MLP(nn.Module):
    def __init__(self, input_size, dim=512, n_blocks=4):
        super(MLP, self).__init__()

        time_dim = dim

        self.fc_in = nn.Linear(input_size, dim)

        self.time_mlp = nn.Sequential(
            SinusoidalPositionEmbeddings(dim),
            nn.Linear(dim, time_dim),
            nn.GELU(),
            nn.Linear(time_dim, time_dim),
        )

        block_klass = partial(ResnetBlock1d, groups=4)

        self.mlp = nn.ModuleList([])

        for i in range(n_blocks):

            self.mlp.append(
                nn.ModuleList(
                    [
                        block_klass(dim, time_emb_dim=time_dim),
                        block_klass(dim, time_emb_dim=time_dim)])

                        #nn.Sequential(nn.Linear(dim, dim), nn.BatchNorm1d(dim), nn.ReLU())

            )

        self.fc_out = nn.Linear(dim, input_size)
        self.SE = SE(input_size)


    def forward(self, x, time):

        output = self.fc_in(x)

        t = self.time_mlp(time)

        for block1, block2 in self.mlp:
            output = block1(output, t)
            output = block2(output, t)
            #x = fc(x)

        output = self.fc_out(output)

        output = self.SE(output)
        shortcut = x
        return shortcut + output


class Unet1D(nn.Module):
    def __init__(self, init_dim=100, dim=128, dim_mults=(1, 2, 4), n_blocks=4):
        super(Unet1D, self).__init__()

        init_dim = default(init_dim, dim)

        dims = [init_dim, *map(lambda m: dim * m, dim_mults)]
        in_out = list(zip(dims[:-1], dims[1:]))

        block_klass = partial(ResnetBlock1d, groups=4)

        # time embeddings
        time_dim = dim

        self.time_mlp = nn.Sequential(
            SinusoidalPositionEmbeddings(dim),
            nn.Linear(dim, time_dim),
            nn.GELU(),
            nn.Linear(time_dim, time_dim),
        )

        # layers
        self.downs = nn.ModuleList([])
        self.ups = nn.ModuleList([])
        num_resolutions = len(in_out)

        # downsampling stage
        for ind, (dim_in, dim_out) in enumerate(in_out):

            self.downs.append(
                nn.ModuleList(
                    [
                        block_klass(dim_in, dim_out=dim_in, time_emb_dim=time_dim),
                        block_klass(dim_in, dim_out=dim_in, time_emb_dim=time_dim),
                        nn.Linear(dim_in, dim_out),

                    ]
                )
            )

        # middle part
        mid_dim = dims[-1]
        self.mid_block1 = block_klass(mid_dim, dim_out=mid_dim, time_emb_dim=time_dim)

        self.mid_block2 = block_klass(mid_dim, dim_out=mid_dim, time_emb_dim=time_dim)

        # upsampling stage
        for ind, (dim_in, dim_out) in enumerate(reversed(in_out)):


            self.ups.append(
                nn.ModuleList(
                    [
                        block_klass(dim_out + dim_in, dim_out=dim_out, time_emb_dim=time_dim),
                        block_klass(dim_out + dim_in, dim_out=dim_in, time_emb_dim=time_dim),
                    ]
                )
            )


        self.final_res_block = block_klass(init_dim * 2, dim_out=init_dim, time_emb_dim=time_dim)
        #self.fc_out = nn.Linear(dim, init_dim)


    def forward(self, x, time):

        r = x.clone()

        t = self.time_mlp(time)

        h = []

        for block1, block2, linear in self.downs:
            x = block1(x, t)
            h.append(x)

            x = block2(x, t)
            h.append(x)

            x = linear(x)


        x = self.mid_block1(x, t)

        x = self.mid_block2(x, t)
        #print("mid block2 shape", x.shape)

        for block1, block2 in self.ups:
            x = torch.cat((x, h.pop()), dim=1)
            x = block1(x, t)

            x = torch.cat((x, h.pop()), dim=1)
            x = block2(x, t)


        x = torch.cat((x, r), dim=1)

        x = self.final_res_block(x, t)
        return x




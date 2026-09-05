import torch
from torch import nn
from abc import abstractmethod
from typing import List, Callable, Union, Any, TypeVar, Tuple
# from torch import tensor as Tensor
import torch.nn.functional as F
from .Modules import *

Tensor = TypeVar('torch.tensor')


class BaseVAE(nn.Module):

    def __init__(self) -> None:
        super(BaseVAE, self).__init__()

    def encode(self, input: Tensor) -> List[Tensor]:
        raise NotImplementedError

    def decode(self, input: Tensor) -> Any:
        raise NotImplementedError

    def sample(self, batch_size: int, **kwargs) -> Tensor:
        raise NotImplementedError

    def generate(self, x: Tensor, **kwargs) -> Tensor:
        raise NotImplementedError

    @abstractmethod
    def forward(self, *inputs: Tensor) -> Tensor:
        pass

    @abstractmethod
    def loss_function(self, *inputs: Any, **kwargs) -> Tensor:
        pass


class VanillaVAE(BaseVAE):

    def __init__(self,
                 in_channels: int,
                 latent_dim: int,
                 dim,
                 dim_mults,
                 **kwargs) -> None:
        super(VanillaVAE, self).__init__()

        self.latent_dim = latent_dim
        self.dim = dim
        self.in_channels = in_channels

        Downsample_type = kwargs.get("Downsample_type", "Stride Convolution")

        modules = []

        hidden_dims = [*map(lambda m: dim * m, dim_mults)]

        # Build Encoder
        for i, h_dim in enumerate(hidden_dims):
            if Downsample_type == "Stride Convolution":
                modules.append(
                    nn.Sequential(
                        nn.Conv3d(in_channels, out_channels=h_dim,
                                  kernel_size=4, stride=2, padding=1),
                        nn.BatchNorm3d(h_dim),
                        nn.LeakyReLU(0.2, inplace=True),
                        # Downsample(h_dim, h_dim)
                        # nn.Dropout(0.2)
                    )
                )
                '''
                modules.append(
                    nn.Sequential(
                        nn.Conv2d(in_channels, out_channels=h_dim,
                                  kernel_size= 3, stride= 2, padding= 1),
                        nn.BatchNorm2d(h_dim),
                        nn.LeakyReLU(0.2, inplace=True),
                        #nn.Dropout(0.2)
                    )
                )
                '''
            elif Downsample_type == "Maxpool":
                # print("Maxpool for downsampling")
                modules.append(
                    nn.Sequential(
                        nn.Conv3d(in_channels, out_channels=h_dim,
                                  kernel_size=3, stride=1, padding=1),
                        nn.BatchNorm3d(h_dim),
                        nn.LeakyReLU(),
                        nn.MaxPool3d(kernel_size=2, stride=2)
                    )
                )

            else:
                # print(Downsample_type + "for downsampling")
                modules.append(
                    nn.Sequential(
                        nn.Conv3d(in_channels, out_channels=h_dim,
                                  kernel_size=3, stride=1, padding=1),
                        nn.BatchNorm3d(h_dim),
                        nn.LeakyReLU(),
                        Downsample(h_dim, h_dim)
                    )
                )

            in_channels = h_dim

        self.encoder = nn.Sequential(*modules)

        self.fc_mu = nn.Linear(hidden_dims[-1] * 64, latent_dim)
        self.fc_var = nn.Linear(hidden_dims[-1] * 64, latent_dim)

        # self.conv_out = nn.Conv2d(hidden_dims[-1], 32, 3, padding=1)

        # the above is encoder, the following is decoder

        # self.conv_in = nn.Conv2d(16, hidden_dims[-1], 3, padding=1)

        # Build Decoder
        modules = []

        self.decoder_input = nn.Linear(latent_dim, hidden_dims[-1] * 64)

        hidden_dims.reverse()
        hidden_dims.append(dim)

        for i in range(len(hidden_dims) - 1):
            if Downsample_type == "Stride Convolution":
                print("ConvTranspose3d for upsampling")

                modules.append(
                    nn.Sequential(
                        # nn.Upsample(scale_factor=2, mode="nearest"),
                        nn.ConvTranspose3d(hidden_dims[i],
                                           hidden_dims[i + 1],
                                           kernel_size=4,
                                           stride=2,
                                           padding=1),
                        nn.BatchNorm3d(hidden_dims[i + 1]),
                        nn.LeakyReLU(0.2, inplace=True),
                        # nn.Dropout(0.2)
                    )

                )

                '''
                modules.append(
                    nn.Sequential(
                        nn.ConvTranspose2d(hidden_dims[i],
                                           hidden_dims[i + 1],
                                           kernel_size=3,
                                           stride = 2,
                                           padding=1,
                                           output_padding=1),
                        nn.BatchNorm2d(hidden_dims[i + 1]),
                        nn.LeakyReLU(0.2, inplace=True),
                        #nn.Dropout(0.2)
                    )

                )
                '''
            else:
                print("nn.Upsampling")
                modules.append(
                    nn.Sequential(
                        nn.Upsample(scale_factor=2, mode="nearest"),
                        nn.Conv3d(hidden_dims[i],
                                  hidden_dims[i + 1],
                                  kernel_size=3,
                                  stride=1,
                                  padding=1),
                        nn.BatchNorm3d(hidden_dims[i + 1]),
                        nn.LeakyReLU())
                )

        self.decoder = nn.Sequential(*modules)

        self.fc_out = nn.Sequential(
            nn.Linear(dim * 32 * 32, 3072),
            nn.Sigmoid()
        )

    def encode(self, input: Tensor) -> List[Tensor]:
        """
        Encodes the input by passing through the encoder network
        and returns the latent codes.
        :param input: (Tensor) Input tensor to encoder [N x C x H x W]
        :return: (Tensor) List of latent codes
        """
        result = self.encoder(input)
        result = torch.flatten(result, start_dim=1)

        # Split the result into mu and var components
        # of the latent Gaussian distribution

        mu = self.fc_mu(result)
        log_var = self.fc_var(result)

        # mu, logvar = self.conv_out(result).chunk(2, dim=1)
        # self.h = mu.size(-2)
        # self.w = mu.size(-1)
        # mu = rearrange(mu, 'b c h w -> b (c h w)')
        # log_var = rearrange(logvar, 'b c h w -> b (c h w)')

        return [mu, log_var]

    def decode(self, z: Tensor) -> Tensor:
        """
        Maps the given latent codes
        onto the image space.
        :param z: (Tensor) [B x D]
        :return: (Tensor) [B x C x H x W]
        """
        # result = self.conv_in(z)

        result = self.decoder_input(z)
        result = result.view(z.size(0), -1, 4, 4, 4)
        result = self.decoder(result)

        result = rearrange(result, "b c h w d -> (b d) (c h w)")
        result = self.fc_out(result)
        result = rearrange(result, "(b d) (c h w) -> b c h w d", d=32, h=32, w=32)
        return result

    def reparameterize(self, mu: Tensor, logvar: Tensor) -> Tensor:
        """
        Reparameterization trick to sample from N(mu, var) from
        N(0,1).
        :param mu: (Tensor) Mean of the latent Gaussian [B x D]
        :param logvar: (Tensor) Standard deviation of the latent Gaussian [B x D]
        :return: (Tensor) [B x D]
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return eps * std + mu

    def forward(self, input: Tensor, **kwargs) -> List[Tensor]:
        mu, log_var = self.encode(input)
        z = self.reparameterize(mu, log_var)
        # z = rearrange(z, 'b (c h w) -> b c h w', c=16, h=self.h)
        return [self.decode(z), input, mu, log_var]

    def loss_function(self,
                      *args,
                      **kwargs) -> dict:
        """
        Computes the VAE loss function.
        KL(N(\mu, \sigma), N(0, 1)) = \log \frac{1}{\sigma} + \frac{\sigma^2 + \mu^2}{2} - \frac{1}{2}
        :param args:
        :param kwargs:
        :return:
        """
        recons = args[0]
        input = args[1]
        mu = args[2]
        log_var = args[3]

        kld_weight = kwargs['kld_weight']  # Account for the minibatch samples from the dataset
        recons_loss = F.mse_loss(recons, input)

        kld_loss = torch.mean(-0.5 * torch.sum(1 + log_var - mu ** 2 - log_var.exp(), dim=1), dim=0)

        loss = recons_loss + kld_weight * kld_loss
        return {'loss': loss, 'Reconstruction_Loss': recons_loss.detach(), 'KLD': -kld_loss.detach()}

    def sample(self,
               num_samples: int, **kwargs) -> Tensor:
        """
        Samples from the latent space and return the corresponding
        image space map.
        :param num_samples: (Int) Number of samples
        :param current_device: (Int) Device to run the model
        :return: (Tensor)
        """
        device = "cuda" if torch.cuda.is_available() else "cpu"
        z = torch.randn(num_samples,
                        self.latent_dim, device=device)

        samples = self.decode(z)
        return samples

    def generate(self, x: Tensor, **kwargs) -> Tensor:
        """
        Given an input image x, returns the reconstructed image
        :param x: (Tensor) [B x C x H x W]
        :return: (Tensor) [B x C x H x W]
        """

        return self.forward(x)[0]


class UNetAE(BaseVAE):
    def __init__(self,
                 dim,
                 latent_dim,
                 out_channels,
                 dim_mults,
                 flatten=False,
                 output_activation=None,
                 in_channels=3,
                 resnet_block_groups=4,
                 **kwargs) -> None:
        super(UNetAE, self).__init__()

        self.flatten = flatten
        self.init_conv = nn.Conv3d(in_channels, dim, kernel_size=1, padding=0)

        dims = [dim, *map(lambda m: dim * m, dim_mults)]
        in_out = list(zip(dims[:-1], dims[1:]))

        block_klass = partial(ResnetBlock, groups=resnet_block_groups)

        # layers
        self.downs = nn.ModuleList([])
        self.ups = nn.ModuleList([])
        num_resolutions = len(in_out)

        # downsampling stage
        for ind, (dim_in, dim_out) in enumerate(in_out):
            is_last = ind >= (num_resolutions - 1)

            self.downs.append(
                nn.ModuleList(
                    [
                        block_klass(dim_in, dim_in, time_emb_dim=None),
                        block_klass(dim_in, dim_in, time_emb_dim=None),
                        # torch.nn.Identity() if ind < (len(in_out) - 2.5)
                        # else Residual(PreNorm(dim_in, LinearAttention(dim_in))),
                        Downsample(dim_in, dim_out)
                        #if not is_last
                        #else nn.Conv2d(dim_in, dim_out, 3, padding=1),
                    ]
                )
            )

        # middle part
        mid_dim = dims[-1]
        self.mid_dim = mid_dim
        self.mid_block1 = block_klass(mid_dim, mid_dim, time_emb_dim=None)
        # self.mid_attn = Residual(PreNorm(mid_dim, Attention(mid_dim)))
        self.mid_block2 = block_klass(mid_dim, mid_dim, time_emb_dim=None)

        if flatten:
            self.linear_out = nn.Linear(mid_dim * 128, latent_dim)
            self.linear_in = nn.Linear(latent_dim, mid_dim * 128)
        else:
            self.attn_out = Residual(PreNorm(mid_dim, Attention(mid_dim)))
            self.conv_out = nn.Conv3d(mid_dim, out_channels, 3, padding=1)
            self.conv_in = nn.Conv3d(out_channels, mid_dim, 3, padding=1)
            self.attn_in = Residual(PreNorm(mid_dim, Attention(mid_dim)))

        #self.mid_block3 = block_klass(mid_dim, mid_dim, time_emb_dim=None)
        # self.mid_attn2 = Residual(PreNorm(mid_dim, Attention(mid_dim)))
        #self.mid_block4 = block_klass(mid_dim, mid_dim, time_emb_dim=None)

        # upsampling stage
        for ind, (dim_in, dim_out) in enumerate(reversed(in_out)):
            is_last = ind == (len(in_out) - 1)

            self.ups.append(
                nn.ModuleList(
                    [
                        block_klass(dim_out, dim_out, time_emb_dim=None),
                        block_klass(dim_out, dim_out, time_emb_dim=None),
                        # torch.nn.Identity() if ind > 1.5
                        # else Residual(PreNorm(dim_out, LinearAttention(dim_out))),
                        Upsample(dim_out, dim_in)
                        #if not is_last
                        #else nn.Conv3d(dim_out, dim_in, 3, padding=1),
                    ]
                )
            )

        self.out_dim = in_channels

        self.final_res_block = block_klass(dim, dim, time_emb_dim=None)

        if output_activation == "Sigmoid":
            print("Sigmoid")
            self.final_conv = nn.Sequential(
                nn.Conv3d(dim, self.out_dim, 1),
                nn.Sigmoid()
            )
        elif output_activation == "Tanh":
            self.final_conv = nn.Sequential(
                nn.Conv3d(dim, self.out_dim, 1),
                nn.Tanh()
            )
        else:
            self.final_conv = nn.Conv3d(dim, self.out_dim, 1)

    def encode(self, x):

        h = self.init_conv(x)
        t = None
        for block1, block2, downsample in self.downs:
            h = block1(h, t)
            h = block2(h, t)
            h = downsample(h)

        #h = self.mid_block1(h, t)
        # h = self.mid_attn(h)
        #h = self.mid_block2(h, t)

        if self.flatten:
            h = h.view(h.size(0), -1)
            h = self.linear_out(h)
        else:
            h= self.attn_out(h)
            h = self.conv_out(h)

        return h

    def decode(self, input):
        ###add noise
        # device = "cuda" if torch.cuda.is_available() else "cpu"
        # z = torch.randn_like(input, device=device)
        # input = input + z
        #h = input

        if self.flatten:
            h = self.linear_in(input)
            h = h.view(h.size(0), self.mid_dim, 4, 4, 8)
        else:
            h = self.conv_in(input)
            h = self.attn_in(h)

        t = None

        #h = self.mid_block1(h, t)
        # h = self.mid_attn(h)
        #h = self.mid_block2(h, t)

        for block1, block2, upsample in self.ups:
            h = block1(h, t)
            h = block2(h, t)

            h = upsample(h)

        h = self.final_res_block(h, t)
        return self.final_conv(h)

    def forward(self, x):
        z = self.encode(x)
        # print(z.size())
        return [self.decode(z), x]

    def loss_function(self, *args: Any, **kwargs) -> Tensor:
        recons = args[0]
        input = args[1]
        recons_loss = F.mse_loss(recons, input)
        # recons_loss = F.l1_loss(recons, input)
        return {'loss': recons_loss}

    def generate(self, x: Tensor, **kwargs) -> Tensor:
        return self.forward(x)[0]



class UNetVAE(BaseVAE):
    def __init__(self,
                 dim,
                 latent_dim,
                 out_channels,
                 dim_mults,
                 flatten=False,
                 output_activation=None,
                 in_channels=3,
                 resnet_block_groups=4,
                 **kwargs) -> None:
        super(UNetVAE, self).__init__()

        self.latent_dim = latent_dim
        self.out_channels = out_channels
        self.flatten = flatten
        self.init_conv = nn.Conv3d(in_channels, dim, kernel_size=1, padding=0)

        dims = [dim, *map(lambda m: dim * m, dim_mults)]
        in_out = list(zip(dims[:-1], dims[1:]))

        block_klass = partial(ResnetBlock, groups=resnet_block_groups)

        # layers
        self.downs = nn.ModuleList([])
        self.ups = nn.ModuleList([])
        num_resolutions = len(in_out)

        # downsampling stage
        for ind, (dim_in, dim_out) in enumerate(in_out):
            is_last = ind >= (num_resolutions - 1)

            self.downs.append(
                nn.ModuleList(
                    [
                        block_klass(dim_in, dim_in, time_emb_dim=None),
                        block_klass(dim_in, dim_in, time_emb_dim=None),
                        # torch.nn.Identity() if ind < (len(in_out) - 2.5)
                        # else Residual(PreNorm(dim_in, LinearAttention(dim_in))),
                        Downsample(dim_in, dim_out)
                        # if not is_last
                        # else nn.Conv2d(dim_in, dim_out, 3, padding=1),
                    ]
                )
            )

        # middle part
        mid_dim = dims[-1]
        self.mid_dim = mid_dim
        self.mid_block1 = block_klass(mid_dim, mid_dim, time_emb_dim=None)
        # self.mid_attn = Residual(PreNorm(mid_dim, Attention(mid_dim)))
        self.mid_block2 = block_klass(mid_dim, mid_dim, time_emb_dim=None)

        if flatten:
            self.linear_out = nn.Linear(mid_dim * 64, latent_dim * 2)
            self.linear_in = nn.Linear(latent_dim, mid_dim * 64)
        else:
            #self.attn_out = Residual(PreNorm(mid_dim, Attention(mid_dim)))
            self.conv_out = nn.Conv3d(mid_dim, out_channels * 2, 3, padding=1)
            self.conv_in = nn.Conv3d(out_channels, mid_dim, 3, padding=1)
            #self.attn_in = Residual(PreNorm(mid_dim, Attention(mid_dim)))

        self.mid_block3 = block_klass(mid_dim, mid_dim, time_emb_dim=None)
        # self.mid_attn2 = Residual(PreNorm(mid_dim, Attention(mid_dim)))
        self.mid_block4 = block_klass(mid_dim, mid_dim, time_emb_dim=None)

        # upsampling stage
        for ind, (dim_in, dim_out) in enumerate(reversed(in_out)):
            is_last = ind == (len(in_out) - 1)

            self.ups.append(
                nn.ModuleList(
                    [
                        block_klass(dim_out, dim_out, time_emb_dim=None),
                        block_klass(dim_out, dim_out, time_emb_dim=None),
                        # torch.nn.Identity() if ind > 1.5
                        # else Residual(PreNorm(dim_out, LinearAttention(dim_out))),
                        Upsample(dim_out, dim_in)
                        # if not is_last
                        # else nn.Conv3d(dim_out, dim_in, 3, padding=1),
                    ]
                )
            )

        self.out_dim = in_channels

        self.final_res_block = block_klass(dim, dim, time_emb_dim=None)

        if output_activation == "Sigmoid":
            print("Sigmoid")
            self.final_conv = nn.Sequential(
                nn.Conv3d(dim, self.out_dim, 1),
                nn.Sigmoid()
            )
        elif output_activation == "Tanh":
            self.final_conv = nn.Sequential(
                nn.Conv3d(dim, self.out_dim, 1),
                nn.Tanh()
            )
        else:
            self.final_conv = nn.Conv3d(dim, self.out_dim, 1)

    def encode(self, x):
        h = self.init_conv(x)
        t = None
        for block1, block2, downsample in self.downs:
            h = block1(h, t)
            h = block2(h, t)
            #h = attn(h)
            h = downsample(h)

        h = self.mid_block1(h, t)
        #h = self.mid_attn(h)
        h = self.mid_block2(h, t)
        if self.flatten:
            h = h.view(h.size(0), -1)
            mu, logvar = self.linear_out(h).chunk(2, dim=1)
        else:
            mu, logvar = self.conv_out(h).chunk(2, dim=1)
            self.h = mu.size(-3)
            self.w = mu.size(-2)
            self.d = mu.size(-1)
            mu = rearrange(mu, 'b c h w d -> b (c h w d)')
            logvar = rearrange(logvar, 'b c h w d -> b (c h w d)')
            

        return [mu, logvar]

    def reparameterize(self, mu: Tensor, logvar: Tensor) -> Tensor:
        """
        Reparameterization trick to sample from N(mu, var) from
        N(0,1).
        :param mu: (Tensor) Mean of the latent Gaussian [B x D]
        :param logvar: (Tensor) Standard deviation of the latent Gaussian [B x D]
        :return: (Tensor) [B x D]
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std, requires_grad=False)
        z = eps * std + mu
        if self.flatten:
            return z
        else:
            z = rearrange(z, 'b (c h w d) -> b c h w d', h=self.h, w=self.w, d=self.d)
            return z


    def decode(self, z):
        if self.flatten:
            h = self.linear_in(z)
            h = h.view(h.size(0), self.mid_dim, 4, 4, 4)
        else:
            h = self.conv_in(z)
            
        t=None

        h = self.mid_block3(h, t)
        #h = self.mid_attn(h)
        h = self.mid_block4(h, t)

        for block1, block2, upsample in self.ups:

            h = block1(h, t)
            h = block2(h, t)
            #h = attn(h)
            h = upsample(h)

        h = self.final_res_block(h, t)
        return self.final_conv(h)

    def forward(self, x):
        mu, log_var = self.encode(x)
        z = self.reparameterize(mu, log_var)

        return [self.decode(z), x, mu, log_var]

    def loss_function(self, *args: Any, **kwargs) -> Tensor:
        recons = args[0]
        input = args[1]
        mu = args[2]
        log_var = args[3]

        kld_weight = kwargs['kld_weight']  # Account for the minibatch samples from the dataset
        recon_loss_type = kwargs.get("loss_type", "l2")
        if recon_loss_type == "l2":
            recons_loss = F.mse_loss(recons, input)
        elif recon_loss_type == "BCEloss":
            recons_loss = F.binary_cross_entropy(recons, input)
        elif recon_loss_type in ("dice", "tversky"):
            # overlap losses for sparse binary fracture networks (immune to imbalance)
            B = recons.size(0)
            p = recons.reshape(B, -1)
            g = input.reshape(B, -1)
            eps = 1.0
            if recon_loss_type == "dice":
                num = 2.0 * (p * g).sum(1) + eps
                den = p.sum(1) + g.sum(1) + eps
                recons_loss = (1.0 - num / den).mean()
            else:
                a = kwargs.get("tversky_alpha", 0.3)
                b = kwargs.get("tversky_beta", 0.7)
                tp = (p * g).sum(1)
                fp = (p * (1 - g)).sum(1)
                fn = ((1 - p) * g).sum(1)
                recons_loss = (1.0 - (tp + eps) / (tp + a * fp + b * fn + eps)).mean()
        else:
            raise ValueError(f"Unknown loss_type: {recon_loss_type}")

        kld_loss = torch.mean(-0.5 * torch.sum(1 + log_var - mu ** 2 - log_var.exp(), dim=1), dim=0)

        loss = recons_loss + kld_weight * kld_loss
        return {'loss': loss, 'Reconstruction_Loss': recons_loss.detach(), 'KLD': -kld_loss.detach()}


    def generate(self, x: Tensor, **kwargs) -> Tensor:
        return self.forward(x)[0]


    def sample(self, num_samples:int, **kwargs) -> Tensor:
        """
        Samples from the latent space and return the corresponding
        image space map.
        :param num_samples: (Int) Number of samples
        :param current_device: (Int) Device to run the model
        :return: (Tensor)
        """
        device = next(self.parameters()).device
        if self.flatten:
            z = torch.randn(num_samples, self.latent_dim, device=device)
        else:
            z = torch.randn(num_samples, self.out_channels, self.h, self.w, self.d, device=device)

        samples = self.decode(z)
        return samples



class Discriminator(nn.Module):
    def __init__(self, in_channels: int,
                 dim=16,
                 dim_mults=[1, 2, 4],
                 **kwargs):
        super(Discriminator, self).__init__()

        Downsample_type = kwargs.get("Downsample_type", "Stride Convolution")

        modules = []

        hidden_dims = [*map(lambda m: dim * m, dim_mults)]

        for h_dim in hidden_dims:
            if Downsample_type == "Stride Convolution":
                #print("Stride Convolution for downsampling")
                modules.append(
                    nn.Sequential(
                        nn.Conv3d(in_channels, out_channels=h_dim,
                                  kernel_size= 3, stride= 2, padding= 1),
                        nn.LeakyReLU(0.2, inplace=True)
                    )
                )
            elif Downsample_type == "Maxpool":
                #print("Maxpool for downsampling")
                modules.append(
                    nn.Sequential(
                        nn.Conv3d(in_channels, out_channels=h_dim,
                                  kernel_size=3, stride=1, padding=1),
                        nn.LeakyReLU(0.2, inplace=True),
                        nn.MaxPool3d(kernel_size=2, stride=2)
                    )
                )

            else:
                #print(Downsample_type + "for downsampling")
                modules.append(
                    nn.Sequential(
                        nn.Conv3d(in_channels, out_channels=h_dim,
                                  kernel_size=3, stride=1, padding=1),
                        nn.LeakyReLU(0.2, inplace=True),
                        Downsample(h_dim, h_dim)
                    )
                )

            in_channels = h_dim

        self.conv = nn.Sequential(*modules)

        self.fc1 = nn.Sequential(
            nn.Linear(hidden_dims[-1] * 64, 512),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.fc2 = nn.Sequential(
            nn.Linear(512, 128),
            nn.LeakyReLU(0.2, inplace=True)
        )

        self.out = nn.Linear(128, 1)

    def forward(self, input):

        out = self.conv(input)
        out = torch.flatten(out, start_dim=1)
        out = self.out(self.fc2(self.fc1(out)))
        return out





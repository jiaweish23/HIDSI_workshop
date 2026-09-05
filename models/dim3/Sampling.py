import torch
from torch import nn, einsum
import torch.nn.functional as F
from torchvision.transforms import Compose, ToPILImage
from tqdm.auto import tqdm

#forward diffusion

timesteps = 1000

device = "cuda" if torch.cuda.is_available() else "cpu"

# define beta schedule
def linear_beta_schedule(timesteps):
    beta_start = 0.0001
    beta_end = 0.02
    return torch.linspace(beta_start, beta_end, timesteps)


betas = linear_beta_schedule(timesteps=timesteps)

# define alphas
alphas = 1. - betas
alphas_cumprod = torch.cumprod(alphas, axis=0)
alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.0)
sqrt_recip_alphas = torch.sqrt(1.0 / alphas)

# calculations for diffusion q(x_t | x_{t-1}) and others
sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod)
sqrt_one_minus_alphas_cumprod = torch.sqrt(1. - alphas_cumprod)

# calculations for posterior q(x_{t-1} | x_t, x_0)
posterior_variance = betas * (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)


def extract(a, t, x_shape):
    batch_size = t.shape[0]
    out = a.gather(-1, t.cpu())
    return out.reshape(batch_size, *((1,) * (len(x_shape) - 1))).to(t.device)


def q_sample(x_start, t, noise=None):
    if noise is None:
        noise = torch.randn_like(x_start, device=device)

    sqrt_alphas_cumprod_t = extract(sqrt_alphas_cumprod, t, x_start.shape)
    sqrt_one_minus_alphas_cumprod_t = extract(
        sqrt_one_minus_alphas_cumprod, t, x_start.shape
    )

    return sqrt_alphas_cumprod_t * x_start + sqrt_one_minus_alphas_cumprod_t * noise


def get_noisy_image(x_start, t):
  # add noise
  x_noisy = q_sample(x_start, t=t)

  # turn back into PIL image
  transform = Compose([
      ToPILImage(),
  ])

  noisy_image = transform(x_noisy.squeeze())

  return noisy_image


# DDPM Sampling

@torch.no_grad()
def p_sample(model, x, t, t_index):
    betas_t = extract(betas, t, x.shape)
    sqrt_one_minus_alphas_cumprod_t = extract(
        sqrt_one_minus_alphas_cumprod, t, x.shape
    )
    sqrt_recip_alphas_t = extract(sqrt_recip_alphas, t, x.shape)

    # Equation 11 in the paper
    # Use our model (noise predictor) to predict the mean
    model_mean = sqrt_recip_alphas_t * (
            x - betas_t * model(x, t) / sqrt_one_minus_alphas_cumprod_t
    )

    if t_index == 0:
        return model_mean
    else:
        posterior_variance_t = extract(posterior_variance, t, x.shape)
        noise = torch.randn_like(x)
        # Algorithm 2 line 4:
        return model_mean + torch.sqrt(posterior_variance_t) * noise

    # Algorithm 2 (including returning all images)


@torch.no_grad()
def p_sample_loop(model, shape):
    device = next(model.parameters()).device

    b = shape[0]
    # start from pure noise (for each example in the batch)
    img = torch.randn(shape, device=device)
    imgs = []

    for i in tqdm(reversed(range(0, timesteps)), desc='sampling loop time step', total=timesteps):
        img = p_sample(model, img, torch.full((b,), i, device=device, dtype=torch.long), i)
        imgs.append(img.cpu().numpy())
    return imgs


@torch.no_grad()
def sample(model, shape):
    return p_sample_loop(model, shape)


#DDIM Sampling
@torch.no_grad()
def p_sample_ddim(model, x, t, t_index, sigma_t=0., Guided=False,
                  c=None, unconditional_guidance_scale=1., unconditional_conditioning=None):
    x_shape = x.size()
    alphas_cumprod_prev_t = extract(alphas_cumprod_prev, t, x_shape)
    alphas_cumprod_t = extract(alphas_cumprod, t, x_shape)
    e_t = model(x, t)
    if Guided and c is not None:
        if unconditional_conditioning is None or unconditional_guidance_scale == 1.:
            e_t = model(x, t, c)
        else:
            x_in = torch.cat([x] * 2)
            t_in = torch.cat([t] * 2)
            c_in = torch.cat([unconditional_conditioning, c])
            e_t_uncond, e_t = model(x_in, t_in, c_in).chunk(2)
            e_t = e_t_uncond + unconditional_guidance_scale * (e_t - e_t_uncond)
    # current prediction for x_0
    pred_x0 = (x - torch.sqrt(1 - alphas_cumprod_t) * e_t) / alphas_cumprod_t.sqrt()
    model_mean = alphas_cumprod_prev_t.sqrt() * pred_x0 + torch.sqrt(1 - alphas_cumprod_prev_t - sigma_t ** 2) * e_t

    if t_index == 0:
        return model_mean
    else:
        noise = torch.randn_like(x)
        return model_mean + sigma_t * noise


@torch.no_grad()
def p_sample_ddim_loop(model, shape, img=None):
    device = next(model.parameters()).device

    b = shape[0]
    # start from pure noise (for each example in the batch)
    if img is None:
        img = torch.randn(shape, device=device)
    imgs = []

    for i in tqdm(reversed(range(0, timesteps)), desc='sampling loop time step', total=timesteps):
        img = p_sample_ddim(model, img, torch.full((b,), i, device=device, dtype=torch.long), i)
        imgs.append(img.cpu().numpy())
    return imgs


@torch.no_grad()
def sample_ddim(model, image_size, batch_size=16, channels=1):
    return p_sample_ddim_loop(model, shape=(batch_size, channels, *image_size))





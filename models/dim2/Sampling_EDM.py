"""
EDM (Karras et al. 2022, "Elucidating the Design Space of Diffusion Models")
formulation for latent diffusion — Route B (dimension-agnostic; used by 2D and 3D).

Self-contained and physically separate from the DDPM/DDIM path in Sampling.py.
Wraps the existing `Unet` (2D or 3D) with EDM preconditioning; the raw
Unet is never modified. Everything (training loss, Heun sampler) hangs off the
denoiser interface D_theta(x; sigma) so that hard-data conditioning can later be
attached cleanly on x0_hat = decode(D_theta(...)).
"""

import copy

import torch
import torch.nn as nn


# --------------------------------------------------------------------------- #
#  Preconditioning:  D_theta(x; sigma) = c_skip*x + c_out * F_theta(c_in*x, c_noise)
# --------------------------------------------------------------------------- #
class EDMPrecond(nn.Module):
    """EDM preconditioning wrapper around a raw noise/denoiser network.

    The wrapped `model` is the existing 3D Unet; it receives a continuous
    `c_noise` as its `time` argument (the sinusoidal embedding accepts floats).
    `D_theta` returns the denoised estimate x0_hat directly (an x0-predictor).
    """

    def __init__(self, model, sigma_data=1.0):
        super().__init__()
        self.model = model
        self.sigma_data = float(sigma_data)

    def _broadcast(self, sigma, x):
        # sigma: (B,) or scalar  ->  (B, 1, 1, ...) matching x.ndim
        if not torch.is_tensor(sigma):
            sigma = torch.as_tensor(sigma, device=x.device, dtype=x.dtype)
        sigma = sigma.to(x.device, x.dtype).reshape(-1)
        if sigma.numel() == 1:
            sigma = sigma.expand(x.shape[0])
        return sigma.view(-1, *([1] * (x.ndim - 1)))

    def forward(self, x, sigma):
        sd = self.sigma_data
        sigma = self._broadcast(sigma, x)

        c_skip = sd ** 2 / (sigma ** 2 + sd ** 2)
        c_out = sigma * sd / (sigma ** 2 + sd ** 2).sqrt()
        c_in = 1.0 / (sigma ** 2 + sd ** 2).sqrt()
        c_noise = (sigma.reshape(-1).log() / 4.0)         # (B,) for time-embed

        F = self.model(c_in * x, c_noise)
        return c_skip * x + c_out * F


# --------------------------------------------------------------------------- #
#  Training loss:  log-normal sigma sampling + EDM weighting
# --------------------------------------------------------------------------- #
def edm_loss(precond, x, P_mean=-1.2, P_std=1.2):
    """EDM denoising-score-matching loss on clean latents `x` (B, C, ...)."""
    rnd = torch.randn(x.shape[0], device=x.device)
    sigma = (rnd * P_std + P_mean).exp().view(-1, *([1] * (x.ndim - 1)))

    sd = precond.sigma_data
    weight = (sigma ** 2 + sd ** 2) / (sigma * sd) ** 2

    n = torch.randn_like(x) * sigma
    D = precond(x + n, sigma)
    return (weight * (D - x) ** 2).mean()


# --------------------------------------------------------------------------- #
#  Sampling:  deterministic Heun 2nd-order ODE solver (optional stochastic churn)
# --------------------------------------------------------------------------- #
@torch.no_grad()
def edm_sample(precond, shape, device, num_steps=32,
               sigma_min=0.002, sigma_max=80.0, rho=7.0,
               S_churn=0.0, S_min=0.0, S_max=float("inf"), S_noise=1.0):
    """Generate clean latents via the EDM Heun sampler. Returns x0 (B, C, ...)."""
    # time-step discretization (sigma schedule)
    step = torch.arange(num_steps, device=device, dtype=torch.float64)
    t = (sigma_max ** (1 / rho)
         + step / (num_steps - 1) * (sigma_min ** (1 / rho) - sigma_max ** (1 / rho))) ** rho
    t = torch.cat([t, torch.zeros_like(t[:1])])           # t_N = 0

    x = torch.randn(shape, device=device, dtype=torch.float64) * t[0]

    for i in range(num_steps):
        x_cur = x
        t_cur, t_next = t[i], t[i + 1]

        # optional stochastic churn (S_churn=0 -> deterministic ODE)
        gamma = (min(S_churn / num_steps, 2 ** 0.5 - 1)
                 if S_min <= t_cur <= S_max else 0.0)
        t_hat = t_cur * (1 + gamma)
        if gamma > 0:
            x_cur = x_cur + (t_hat ** 2 - t_cur ** 2).sqrt() * S_noise * torch.randn_like(x_cur)

        # Euler step
        d_cur = (x_cur - precond(x_cur.float(), t_hat.float()).double()) / t_hat
        x = x_cur + (t_next - t_hat) * d_cur

        # Heun 2nd-order correction
        if t_next > 0:
            d_prime = (x - precond(x.float(), t_next.float()).double()) / t_next
            x = x_cur + (t_next - t_hat) * 0.5 * (d_cur + d_prime)

    return x.float()


# --------------------------------------------------------------------------- #
#  EMA of the raw model weights (decay ~0.9999) — what we sample from.
# --------------------------------------------------------------------------- #
class EMA:
    def __init__(self, model, decay=0.9999):
        self.decay = decay
        self.shadow = {k: v.detach().clone() for k, v in model.state_dict().items()}

    @torch.no_grad()
    def update(self, model):
        for k, v in model.state_dict().items():
            if v.dtype.is_floating_point:
                self.shadow[k].mul_(self.decay).add_(v.detach(), alpha=1 - self.decay)
            else:
                self.shadow[k].copy_(v)

    def copy_to(self, model):
        model.load_state_dict(self.shadow, strict=True)

    def state_dict(self):
        return self.shadow

    def load_state_dict(self, state):
        """Counterpart of state_dict(), for resuming training from a checkpoint."""
        self.shadow = {k: v.detach().clone().to(self.shadow[k].device)
                       for k, v in state.items()}


def clone_model(model):
    """Detached deep copy for an EMA-eval network (no grad)."""
    m = copy.deepcopy(model)
    for p in m.parameters():
        p.requires_grad = False
    return m


# --------------------------------------------------------------------------- #
#  Hard-data conditioning (Route B): CondEncoder context concatenated at the
#  UNet input.  Same imposition mechanism as the DDPM hard-data design, but on
#  EDM — the denoiser D_theta IS x0_hat, so the fine-resolution consistency loss
#  needs no alpha_t reconstruction.  DPS guidance can be layered on at sampling.
# --------------------------------------------------------------------------- #
class EDMPrecondCond(EDMPrecond):
    """EDM preconditioning whose wrapped model takes (x, c_noise, context)."""

    def forward(self, x, sigma, context=None):
        sd = self.sigma_data
        sigma = self._broadcast(sigma, x)
        c_skip = sd ** 2 / (sigma ** 2 + sd ** 2)
        c_out = sigma * sd / (sigma ** 2 + sd ** 2).sqrt()
        c_in = 1.0 / (sigma ** 2 + sd ** 2).sqrt()
        c_noise = (sigma.reshape(-1).log() / 4.0)
        F = self.model(c_in * x, c_noise, context)
        return c_skip * x + c_out * F


def edm_loss_cond(precond, x, context, P_mean=-1.2, P_std=1.2):
    """Conditional EDM loss. Returns (loss, x0_hat) so the caller can reuse the
    same x0_hat = D_theta for the fine-resolution consistency term."""
    rnd = torch.randn(x.shape[0], device=x.device)
    sigma = (rnd * P_std + P_mean).exp().view(-1, *([1] * (x.ndim - 1)))
    sd = precond.sigma_data
    weight = (sigma ** 2 + sd ** 2) / (sigma * sd) ** 2
    n = torch.randn_like(x) * sigma
    D = precond(x + n, sigma, context)
    loss = (weight * (D - x) ** 2).mean()
    return loss, D


@torch.no_grad()
def edm_sample_cond(precond, context, shape, device, num_steps=32,
                    sigma_min=0.002, sigma_max=80.0, rho=7.0):
    """Deterministic conditional Heun sampler (no DPS). For training monitors."""
    step = torch.arange(num_steps, device=device, dtype=torch.float64)
    t = (sigma_max ** (1 / rho) + step / (num_steps - 1) *
         (sigma_min ** (1 / rho) - sigma_max ** (1 / rho))) ** rho
    t = torch.cat([t, torch.zeros_like(t[:1])])
    x = torch.randn(shape, device=device, dtype=torch.float64) * t[0]
    for i in range(num_steps):
        t_cur, t_next = t[i], t[i + 1]
        d_cur = (x - precond(x.float(), t_cur.float(), context).double()) / t_cur
        x_next = x + (t_next - t_cur) * d_cur
        if t_next > 0:
            d_prime = (x_next - precond(x_next.float(), t_next.float(), context).double()) / t_next
            x_next = x + (t_next - t_cur) * 0.5 * (d_cur + d_prime)
        x = x_next
    return x.float()


def edm_sample_cfg_dps(precond, context, shape, device,
                       vae=None, vae_decode_fn=None, value_map=None, mask=None,
                       latent_mean=0.0, w_cfg=1.0, w_dps=0.0,
                       num_steps=32, sigma_min=0.002, sigma_max=80.0, rho=7.0):
    """Conditional Heun sampler with Classifier-Free Guidance (CFG) and
    Diffusion Posterior Sampling (DPS) measurement guidance.

    Unified guidance-strength names (use these everywhere; do NOT reintroduce
    bare `w`, `zeta`, `guidance_scale`, or `dps_weight`):
      w_cfg : CFG guidance scale  -> amplifies the LEARNED condition (CondEncoder).
              D = D_uncond + w_cfg * (D_cond - D_uncond).  w_cfg=1 -> plain conditional.
              (legacy DDPM name: guidance_scale)
      w_dps : DPS measurement step -> enforces the ACTUAL well observations.
              each step adds  -w_dps * grad/||resid||  of ||(decode(D)-y)*mask||^2
              through the frozen VAE decoder, full strength (NOT step-length scaled).
              (legacy DDPM name: dps_weight)

    Args:
        vae_decode_fn : callable z -> image (B,1,H,W) in data space (no latent_mean added).
        value_map     : (B,1,H,W) observed fracture values at wells, 0 elsewhere.
        mask          : (B,1,H,W) 1 at wells. w_dps>0 requires value_map/mask/vae_decode_fn.
        latent_mean   : scalar/tensor added before decoding (latent standardisation).
    """
    use_dps = w_dps > 0 and mask is not None and bool(mask.any())

    def cfg_denoise(x32, sigma32):
        """CFG denoiser; x32, sigma32 float32. Returns D (float32)."""
        if w_cfg == 1.0 or context is None:
            return precond(x32, sigma32, context)
        D_c = precond(x32, sigma32, context)
        D_u = precond(x32, sigma32, None)
        return D_u + w_cfg * (D_c - D_u)

    step = torch.arange(num_steps, device=device, dtype=torch.float64)
    t = (sigma_max ** (1 / rho) + step / (num_steps - 1) *
         (sigma_min ** (1 / rho) - sigma_max ** (1 / rho))) ** rho
    t = torch.cat([t, torch.zeros_like(t[:1])])
    x = torch.randn(shape, device=device, dtype=torch.float64) * t[0]

    for i in range(num_steps):
        t_cur, t_next = t[i], t[i + 1]
        sig_cur = t_cur.float()

        # ---- denoise at current step (+ optional DPS gradient) ----
        if use_dps:
            x_req = x.float().detach().requires_grad_(True)
            with torch.enable_grad():
                D_cur = cfg_denoise(x_req, sig_cur)
                img = vae_decode_fn(D_cur + latent_mean)
                resid = (img - value_map) * mask
                meas = (resid ** 2).sum()                       # un-normalised SSE
                grad = torch.autograd.grad(meas, x_req)[0]
            # DPS-paper normalisation: step = w_dps * grad / ||resid|| (full strength,
            # NOT scaled by step length — matches the proven DDPM recipe that hit ~100%).
            rnorm = resid.detach().norm().clamp_min(1e-8)
            dps_step = (w_dps * grad / rnorm).double()
            D_cur = D_cur.detach().double()
        else:
            with torch.no_grad():
                D_cur = cfg_denoise(x.float(), sig_cur).double()
            dps_step = 0.0

        d_cur = (x - D_cur) / t_cur
        x_next = x + (t_next - t_cur) * d_cur

        # Heun 2nd-order correction (no DPS grad needed here)
        if t_next > 0:
            with torch.no_grad():
                D_nxt = cfg_denoise(x_next.float(), t_next.float()).double()
            d_prime = (x_next - D_nxt) / t_next
            x_next = x + (t_next - t_cur) * 0.5 * (d_cur + d_prime)

        # DPS measurement correction at full strength every step
        if use_dps:
            x_next = x_next - dps_step

        x = x_next

    return x.float()

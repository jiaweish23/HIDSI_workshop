"""
EDM (Karras et al. 2022, "Elucidating the Design Space of Diffusion Models")
formulation for the 3D latent diffusion — Route B.

Self-contained and physically separate from the DDPM/DDIM path in Sampling.py.
Wraps the existing `models.dim3.Modules.Unet` with EDM preconditioning; the raw
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
#  Conditional EDM (3D hard-data): context-aware preconditioning, loss, sampler
#  Mirrors models/dim2/Sampling_EDM.py.  The EDM math is dimension-agnostic
#  (EDMPrecond._broadcast already handles any x.ndim); only the wrapped model and
#  the VAE decode are 3D.  Unified guidance names: w_cfg (CFG), w_dps (DPS).
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
                       num_steps=32, sigma_min=0.002, sigma_max=80.0, rho=7.0,
                       dps_balanced=False):
    """Conditional Heun sampler with CFG (w_cfg) + DPS (w_dps).  3D analogue of the
    2D edm_sample_cfg_dps.  DPS update is full strength: x <- x - w_dps*grad/||resid||
    (NOT scaled by step length).  vae_decode_fn: z -> volume (B,1,D,H,W).

    dps_balanced: weight the fracture and matrix observations EQUALLY, matching the
    training objective (class_balanced_cons averages the two classes separately).  The
    default plain sum weights each voxel alike, so the majority matrix observations take
    ~2/3 of the gradient and the rare fracture ones only ~1/3 -- the opposite of how the
    model was trained.  The balanced measure is rescaled by the observation count so it
    keeps the same magnitude as the sum (and reduces to it when the classes are equinumerous),
    leaving the useful w_dps range unchanged."""
    use_dps = w_dps > 0 and mask is not None and bool(mask.any())

    def cfg_denoise(x32, sigma32):
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

        if use_dps:
            x_req = x.float().detach().requires_grad_(True)
            with torch.enable_grad():
                D_cur = cfg_denoise(x_req, sig_cur)
                img = vae_decode_fn(D_cur + latent_mean)
                resid = (img - value_map) * mask
                if dps_balanced:
                    wb = mask.bool()
                    fm = wb & (value_map > 0.5)
                    mm = wb & (value_map <= 0.5)
                    terms = [(resid[c] ** 2).mean() for c in (fm, mm) if c.any()]
                    meas = wb.sum() * sum(terms) / len(terms)
                else:
                    meas = (resid ** 2).sum()
                grad = torch.autograd.grad(meas, x_req)[0]
            rnorm = resid.detach().norm().clamp_min(1e-8)
            dps_step = (w_dps * grad / rnorm).double()
            D_cur = D_cur.detach().double()
        else:
            with torch.no_grad():
                D_cur = cfg_denoise(x.float(), sig_cur).double()
            dps_step = 0.0

        d_cur = (x - D_cur) / t_cur
        x_next = x + (t_next - t_cur) * d_cur

        if t_next > 0:
            with torch.no_grad():
                D_nxt = cfg_denoise(x_next.float(), t_next.float()).double()
            d_prime = (x_next - D_nxt) / t_next
            x_next = x + (t_next - t_cur) * 0.5 * (d_cur + d_prime)

        if use_dps:
            x_next = x_next - dps_step

        x = x_next

    return x.float()

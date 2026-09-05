"""
TUDF hard-data extensions to the 3D EDM sampler (Route C, continuous field).

Physically separate from Sampling_EDM.py: the EDM math, preconditioning, EMA and
the plain conditional Heun sampler are *reused* unchanged from Sampling_EDM; only
the hard-data measurement changes.

The binary-occupancy path used a symmetric class-balanced BCE / squared residual
`(x0_hat - y)`.  On a TUDF field that is wrong (it flattens the 0..0.5 halo and
kills the gradient information; see compare_binarize).  Here the measurement is
the *asymmetric* one:
    frac observation (y=1):   (x0_hat - 1)^2                  # equality: pull to the ridge
    matrix observation (y=0): relu(x0_hat - (0.5 - m))^2      # inequality: only cap a
                                                              # spurious surface, halo free
class-balanced (fracture and matrix voxels averaged separately).  The SAME function
is used for the training consistency loss and for the sampling-time DPS gradient,
so training and inference enforce identical observation likelihoods.
"""

import torch

from .Sampling_EDM import (  # reuse everything dimension-agnostic / unchanged
    EDMPrecond, EDMPrecondCond, edm_loss, edm_loss_cond, edm_sample,
    edm_sample_cond, EMA, clone_model,
)

__all__ = [
    "EDMPrecond", "EDMPrecondCond", "edm_loss", "edm_loss_cond", "edm_sample",
    "edm_sample_cond", "EMA", "clone_model",
    "asym_terms", "asym_cons", "edm_sample_cfg_dps_asym",
    "plane_terms", "plane_cons", "edm_sample_cfg_dps_plane",
]


# --------------------------------------------------------------------------- #
#  Asymmetric TUDF measurement  (shared by training consistency and DPS)
# --------------------------------------------------------------------------- #
def asym_terms(pred, gt, margin=0.0):
    """Class-balanced asymmetric measurement on flat well-voxel tensors.

    pred : predicted TUDF values x0_hat at the well voxels (1-D)
    gt   : binary observation y at the same voxels (1-D, 1=fracture / 0=matrix)
    margin (m): push matrix voxels just below 0.5 - m for a decisive binarization
                (default 0 -> constraint boundary coincides with the 0.5 threshold).
    Returns a scalar; fracture and matrix terms are averaged separately then the
    present terms are averaged, so rare fracture hits are not drowned by matrix.
    """
    fm = gt > 0.5
    mm = ~fm
    terms = []
    if fm.any():
        terms.append(((pred[fm] - 1.0) ** 2).mean())          # equality pull -> 1
    if mm.any():
        viol = torch.relu(pred[mm] - (0.5 - margin))          # one-sided: only T>0.5-m
        terms.append((viol ** 2).mean())
    if not terms:
        return pred.sum() * 0.0
    return sum(terms) / len(terms)


def asym_cons(x0_img, value_map, mask, margin=0.0):
    """Training consistency loss: decode(D_theta) TUDF `x0_img` (B,1,D,H,W) vs the
    binary well labels `value_map` at `mask` voxels, via the asymmetric measurement."""
    wb = mask.bool()
    if not wb.any():
        return x0_img.sum() * 0.0
    return asym_terms(x0_img[wb], value_map[wb], margin)


# --------------------------------------------------------------------------- #
#  Plane-aware TUDF measurement  (replaces asym_*; see models/dim3/well_planes.py)
# --------------------------------------------------------------------------- #
def plane_terms(pred, target, tmask, fmask, mmask, thr=0.5):
    """Plane-aware well measurement.  `pred` is the decoded TUDF x0_hat (B,1,D,H,W).

    The old `asym_terms` pulls every lit well voxel to T=1.  Measured on the real data,
    those voxels' true T is median 0.763 and a well run is >=2 stacked voxels 86% of the
    time, so that demand implies a plane containing a vertical segment -- infeasible.  The
    ground truth scores 0.0793 on it while an impossible field scores 0.

    Here a log reports a sub-voxel pierce point p and a dip n, from which the TUDF over the
    3x3x3 patch around p is *analytic*:  T = clip(1 - |n.(v-p)|/tau, 0, 1).  The 27 values
    recover n to 0.00 degrees, so orientation is imposed implicitly -- there is no separate
    orientation term.  All three terms are ONE-SIDED and the ground truth scores exactly 0
    on each (measured: 0.000000 / 0.000000 / 0.000000).

        tmask : T >= analytic target   (the plane passes here, tilted like this)
        fmask : T >= thr               (log says fracture, but no pierce/dip available)
        mmask : T <= thr               (log says matrix)

    Terms are averaged separately then averaged, so the rare fracture voxels are not
    drowned by the matrix ones.
    """
    tb, fb, mb = tmask.bool(), fmask.bool(), mmask.bool()
    terms = []
    if tb.any():
        terms.append((torch.relu(target[tb] - pred[tb]) ** 2).mean())
    if fb.any():
        terms.append((torch.relu(thr - pred[fb]) ** 2).mean())
    if mb.any():
        terms.append((torch.relu(pred[mb] - thr) ** 2).mean())
    if not terms:
        return pred.sum() * 0.0
    return sum(terms) / len(terms)


def plane_cons(x0_img, target, tmask, fmask, mmask, thr=0.5):
    """Training consistency loss -- same measurement the sampler's DPS uses."""
    return plane_terms(x0_img, target, tmask, fmask, mmask, thr)


# --------------------------------------------------------------------------- #
#  Conditional Heun sampler with CFG (w_cfg) + asymmetric DPS (w_dps)
# --------------------------------------------------------------------------- #
def _edm_sample_cfg_dps(precond, context, shape, device, meas_fn=None,
                        vae_decode_fn=None, latent_mean=0.0, w_cfg=1.0, w_dps=0.0,
                        num_steps=32, sigma_min=0.002, sigma_max=80.0, rho=7.0):
    """3D conditional Heun sampler: CFG + full-strength DPS.

    Structurally identical to Sampling_EDM.edm_sample_cfg_dps (CFG combination,
    DPS step x <- x - w_dps*grad/||.||, decode through the frozen VAE); the observation
    likelihood is pluggable via `meas_fn(decoded_tudf) -> scalar`, so training and
    inference always enforce the *same* measurement.
    vae_decode_fn: z -> continuous TUDF volume (B,1,D,H,W) in [0,1].
    """
    use_dps = w_dps > 0 and meas_fn is not None

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
                meas = meas_fn(img)
                grad = torch.autograd.grad(meas, x_req)[0]
            rnorm = meas.detach().sqrt().clamp_min(1e-8)       # DPS 1/||.|| normalization
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


def edm_sample_cfg_dps_asym(precond, context, shape, device,
                            vae_decode_fn=None, value_map=None, mask=None,
                            margin=0.0, **kw):
    """Legacy binary-label measurement (asym_terms).  Kept for the existing eval scripts."""
    meas = None
    if mask is not None and bool(mask.any()):
        wb = mask.bool()
        meas = lambda img: asym_terms(img[wb], value_map[wb], margin)     # noqa: E731
    return _edm_sample_cfg_dps(precond, context, shape, device, meas_fn=meas,
                               vae_decode_fn=vae_decode_fn, **kw)


def edm_sample_cfg_dps_plane(precond, context, shape, device,
                             vae_decode_fn=None, target=None, tmask=None,
                             fmask=None, mmask=None, thr=0.5, **kw):
    """Plane-aware measurement (plane_terms) -- the same one the training loss uses."""
    meas = None
    if tmask is not None and bool(tmask.any()):
        meas = lambda img: plane_terms(img, target, tmask, fmask, mmask, thr)  # noqa: E731
    return _edm_sample_cfg_dps(precond, context, shape, device, meas_fn=meas,
                               vae_decode_fn=vae_decode_fn, **kw)

"""Plane-aware well observations for TUDF hard data.

A borehole image log reports, for every fracture it intersects, a **sub-voxel pierce
depth** and a **dip**.  From those two numbers the TUDF is *analytic* over the whole
neighbourhood of the pierce:

    T_target(v) = clip(1 - |n . (v - p)| / tau, 0, 1)

so the observation constrains a 3x3x3 patch (27 voxels) rather than a single voxel, and
it constrains the plane's **orientation** implicitly — the 27 values recover n to 0.00
degrees.  This replaces the old per-voxel `T=1` label, which is geometrically infeasible
(a well run is >=2 stacked voxels 86% of the time; forcing them all onto one plane demands
a vertical plane, while the real |n_z| ~ 0.56).

Because the true field is a **min** over all fractures, a single plane's analytic value is
a *lower* bound on the truth (validated: only 1.7% of voxels fall below it, from the
finite-polygon edge).  The loss must therefore be one-sided — see `plane_cons` in
`Sampling_EDM_tudf.py`.

Depth axis = spatial dim 0, matching `build_well_inputs_tudf`.
"""
import numpy as np
import torch

PATCH_HALF = 1                    # 3x3x3; 5x5x5 measured worse (err 0.056 vs 0.021)


class PlaneBank:
    """Holds the precomputed per-sample plane parameters (from precompute_planes.py)."""

    def __init__(self, npz_path, device):
        z = np.load(npz_path)
        self.n = torch.from_numpy(z["normals"]).to(device)      # (N,P,3)
        self.c = torch.from_numpy(z["cents"]).to(device)        # (N,P,3)
        self.e1 = torch.from_numpy(z["e1"]).to(device)
        self.e2 = torch.from_numpy(z["e2"]).to(device)
        self.ring = torch.from_numpy(z["ring"]).to(device)      # (N,P,V,2)
        self.nplane = torch.from_numpy(z["nplane"].astype("int64")).to(device)  # (N,)
        self.P = self.n.shape[1]
        self.device = device

    def batch(self, idx):
        return (self.n[idx], self.c[idx], self.e1[idx], self.e2[idx],
                self.ring[idx], self.nplane[idx])


def _inside(a, b, ring):
    """Ray-crossing point-in-polygon.  a,b: (...,);  ring: (...,V,2) padded by repeating
    the last real vertex, which makes the padded edges degenerate and the closing edge
    correct, so no vertex count is needed."""
    x1 = ring[..., 0]                                   # (...,V)
    y1 = ring[..., 1]
    x2 = torch.roll(x1, -1, dims=-1)
    y2 = torch.roll(y1, -1, dims=-1)
    a = a.unsqueeze(-1)
    b = b.unsqueeze(-1)
    straddle = (y1 > b) != (y2 > b)
    xint = x1 + (b - y1) * (x2 - x1) / (y2 - y1 + 1e-12)
    cross = straddle & (a < xint)
    return cross.sum(-1) % 2 == 1                       # (...,)


def pierce_table(planes, R=32, n_eps=1e-3):
    """Intersect every vertical well column with every plane.

    Returns z (B,P,R,R) pierce depth and hit (B,P,R,R) bool — True where the column
    actually pierces the finite polygon inside the volume."""
    n, c, e1, e2, ring, nplane = planes
    B, P = n.shape[:2]
    dev = n.device
    hi, wi = torch.meshgrid(torch.arange(R, device=dev, dtype=n.dtype),
                            torch.arange(R, device=dev, dtype=n.dtype), indexing="ij")
    hi = hi.reshape(1, 1, R, R)
    wi = wi.reshape(1, 1, R, R)

    nc = (n * c).sum(-1)[..., None, None]                              # (B,P,1,1)
    n0 = n[..., 0][..., None, None]
    n1 = n[..., 1][..., None, None]
    n2 = n[..., 2][..., None, None]
    z = (nc - n1 * hi - n2 * wi) / torch.where(n0.abs() < n_eps,
                                               torch.full_like(n0, n_eps), n0)

    ok = (n0.abs() >= n_eps) & (z >= 0) & (z <= R - 1)
    ok = ok & (torch.arange(P, device=dev).view(1, P, 1, 1) < nplane.view(B, 1, 1, 1))

    p = torch.stack([z, hi.expand_as(z), wi.expand_as(z)], -1)         # (B,P,R,R,3)
    rel = p - c[:, :, None, None, :]
    a = (rel * e1[:, :, None, None, :]).sum(-1)
    b = (rel * e2[:, :, None, None, :]).sum(-1)
    inside = _inside(a, b, ring[:, :, None, None, :, :].expand(B, P, R, R, ring.shape[2], 2))
    return z, ok & inside


def build_well_inputs_plane(x, planes, n_wells, tau, R=32, beta=1.0, eps=0.5,
                            half=PATCH_HALF, thr=0.5, generator=None,
                            return_meta=False, spread=False, keep=None):
    """x: (B,1,R,R,R) TUDF volume -> target, tmask, fmask, mmask  (each (B,1,R,R,R))

    Three constraints, each of which the ground truth satisfies exactly:

      target/tmask : the analytic TUDF on the 3x3x3 patch around every pierce (element-wise
                     max where patches overlap, matching the true min-distance field).
                     Loss: one-sided  relu(target - That)^2
      fmask        : well voxels the log calls FRACTURE (binary obs = 1) that no patch
                     covers -- a fracture grazing the borehole without piercing it, or one
                     nearly parallel to it whose pierce falls outside the volume.  We know
                     a fracture is there but have no pierce point/dip for it.
                     Loss: weak  relu(thr - That)^2
      mmask        : well voxels the log calls MATRIX (binary obs = 0).
                     Loss: relu(That - thr)^2

    NOTE fmask/mmask are keyed off the *binary log reading* 1[T>thr], NOT off "is a patch
    here" -- getting that wrong makes the truth violate the matrix term 11% of the time.

    return_meta: additionally return (cols, wmask, pierces) where cols (B,n_wells) are the
    flattened (h,w) footprints, wmask (B,1,R,R,R) marks the logged columns, and pierces is a
    per-sample list of (p (M,3), n (M,3)) -- the observations themselves.  Evaluation needs
    these to check whether a *generated* fracture actually passes through p with dip n; the
    training path never asks for them.
    """
    n, c, e1, e2, ring, nplane = planes
    B, P = n.shape[:2]
    dev = n.device
    z, hit = pierce_table(planes, R=R)                                 # (B,P,R,R)

    npierce = hit.sum(1).reshape(B, R * R).float()                     # pierces per column
    if spread:
        # farthest-point sampling among pierce-carrying columns: maximally spread wells
        # that each still intersect >=1 plane (start from the pierce-richest column).
        k = min(n_wells, R * R)
        cols_list = []
        for bi in range(B):
            cand = torch.nonzero(npierce[bi] > 0).squeeze(1)           # flat col indices
            if cand.numel() <= k:
                sel = cand
            else:
                hw = torch.stack([cand // R, cand % R], 1).float()     # (M,2) footprints
                start = int(npierce[bi][cand].argmax())
                chosen = [start]
                while len(chosen) < k:
                    ref_hw = hw[chosen]                                # (c,2)
                    d = torch.cdist(hw, ref_hw).min(1).values          # (M,)
                    d[chosen] = -1.0
                    chosen.append(int(d.argmax()))
                sel = cand[torch.tensor(chosen, device=dev)]
            cols_list.append(sel)
        cols = torch.stack(cols_list)
    else:
        w = (npierce + eps) ** beta
        cols = torch.multinomial(w, min(n_wells, R * R), replacement=False, generator=generator)

    if keep is not None and keep < cols.shape[1]:
        # STRICT SUBSET: keep only `keep` of the already-chosen columns (farthest-point among
        # them, per batch), i.e. "remove n_wells-keep wells" -- the kept pierces are a subset.
        kept = []
        for bi in range(B):
            cc = cols[bi]                                             # (n_wells,) chosen cols
            hw = torch.stack([cc // R, cc % R], 1).float()
            chosen = [0]                                              # start from first chosen
            while len(chosen) < keep:
                d = torch.cdist(hw, hw[chosen]).min(1).values
                d[chosen] = -1.0
                chosen.append(int(d.argmax()))
            kept.append(cc[torch.tensor(chosen, device=dev)])
        cols = torch.stack(kept)

    off = torch.stack(torch.meshgrid(*[torch.arange(-half, half + 1, device=dev)] * 3,
                                     indexing="ij"), -1).reshape(-1, 3).to(n.dtype)  # (27,3)

    target = torch.zeros(B, R * R * R, device=dev, dtype=n.dtype)
    tmask = torch.zeros_like(target)
    wmask = torch.zeros_like(target)
    pierces = []

    for bi in range(B):
        sel = cols[bi]
        hh, ww = sel // R, sel % R
        # whole logged column
        zz = torch.arange(R, device=dev)
        widx = (zz.view(1, R) * R * R + hh.view(-1, 1) * R + ww.view(-1, 1)).reshape(-1)
        wmask[bi, widx] = 1.0

        h_bp = hit[bi][:, hh, ww]                                      # (P, n_wells)
        z_bp = z[bi][:, hh, ww]
        pk, pw = torch.nonzero(h_bp, as_tuple=True)                    # plane idx, well idx
        if len(pk) == 0:
            pierces.append((n.new_zeros(0, 3), n.new_zeros(0, 3)))
            continue
        p = torch.stack([z_bp[pk, pw], hh[pw].to(n.dtype), ww[pw].to(n.dtype)], -1)  # (M,3)
        nn = n[bi, pk]                                                 # (M,3)
        pierces.append((p.detach().clone(), nn.detach().clone()))

        v = torch.round(p)[:, None, :] + off[None]                     # (M,27,3) voxel centres
        sd = ((v - p[:, None, :]) * nn[:, None, :]).sum(-1)             # signed dist to the plane
        t = (1.0 - sd.abs() / tau).clamp(0.0, 1.0)

        # the analytic value assumes an INFINITE plane.  Beyond the polygon's rim the true
        # distance is to the rim (larger), so the target would overshoot -- drop those.
        cc, ee1, ee2 = c[bi, pk], e1[bi, pk], e2[bi, pk]                # (M,3)
        foot = v - sd[..., None] * nn[:, None, :]                       # (M,27,3)
        rel_f = foot - cc[:, None, :]
        fa = (rel_f * ee1[:, None, :]).sum(-1)
        fb = (rel_f * ee2[:, None, :]).sum(-1)
        rg = ring[bi, pk][:, None].expand(-1, off.shape[0], -1, -1)     # (M,27,V,2)
        on_face = _inside(fa, fb, rg)

        vi = v.long()
        good = ((vi >= 0) & (vi <= R - 1)).all(-1) & on_face
        vi, t = vi[good], t[good]
        flat = vi[:, 0] * R * R + vi[:, 1] * R + vi[:, 2]
        target[bi].scatter_reduce_(0, flat, t, reduce="amax", include_self=True)
        tmask[bi, flat] = 1.0

    sh = (B, 1, R, R, R)
    target = target.reshape(sh)
    tmask = tmask.reshape(sh)
    wmask = wmask.reshape(sh)

    # the log's binary reading along the column -> the two remaining constraints
    lit = (x > thr).to(x.dtype)
    fmask = wmask * lit * (1.0 - tmask)          # 'fracture here' but no pierce/dip for it
    mmask = wmask * (1.0 - lit)                  # 'no fracture here'
    if return_meta:
        return target, tmask, fmask, mmask, cols, wmask, pierces
    return target, tmask, fmask, mmask

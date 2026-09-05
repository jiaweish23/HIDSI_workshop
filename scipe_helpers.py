"""
scipe_helpers.py  --  self-contained helpers for the SCIPE 3D LDM workshop.

These functions are copied (verbatim in behaviour) from the research repo
`fracture_ldm_reproduce` so the workshop folder runs standalone -- it imports only
from the local `models/` package, never from the research repo.

Featured model: the 3D BINARY fracture LDM with HARD-DATA (well) conditioning.
Pipeline:  VAE encodes a 32^3 binary volume -> 8^3 latent;  an EDM diffusion model
denoises in that latent space;  a CondEncoder turns sparse well observations into a
context volume that steers generation (classifier-free guidance), and DPS adds a
gradient pull so the decoded volume matches the observed well voxels.
"""
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
import yaml

from models.dim3.VAE_Modules import UNetAE, UNetVAE
from models.dim3.Modules import Unet_SR
from models.dim3.CondEncoder import CondEncoder3D
from models.dim3.Sampling_EDM import EDMPrecondCond, edm_sample_cfg_dps, clone_model


# --------------------------------------------------------------------------- #
#  config / model loading
# --------------------------------------------------------------------------- #
def load_yaml(path):
    with open(path) as f:
        return yaml.safe_load(f)


def load_vae(vae_config_path, vae_checkpoint_path, device):
    """Load the frozen 3D VAE (UNetVAE) used by the LDM."""
    cfg = load_yaml(vae_config_path)
    vae = UNetVAE(
        latent_dim=cfg["latent_dim"], out_channels=cfg["latent_channels"],
        dim=cfg["dim"], in_channels=cfg["channels"], dim_mults=cfg["dim_mults"],
        Downsample_type=cfg["Downsample_type"], flatten=cfg["flatten"],
        output_activation=cfg["output_activation"],
    )
    vae.load_state_dict(torch.load(vae_checkpoint_path, map_location=device))
    for p in vae.parameters():
        p.requires_grad = False
    vae.to(device).eval()
    return vae, cfg


def load_ldm_and_encoder(ldm_ckpt, enc_ckpt, device, dim=128, dim_mults=(1, 2, 4, 8),
                         channels=1, cond_hidden_dim=32, image_size=32):
    """Load the EDM denoising UNet (Unet_SR) and the well CondEncoder."""
    ldm = Unet_SR(dim=dim, channels=channels, dim_mults=tuple(dim_mults))
    ldm.load_state_dict(torch.load(ldm_ckpt, map_location=device))
    ldm.to(device).eval()
    enc = CondEncoder3D(hidden_dim=cond_hidden_dim, image_size=image_size)
    enc.load_state_dict(torch.load(enc_ckpt, map_location=device))
    enc.to(device).eval()
    return ldm, enc


def encode(vae, vae_cfg, batch):
    if vae_cfg["model_name"] == "UnetAE":
        return vae.encode(batch)
    return vae.reparameterize(*vae.encode(batch))


def vae_decode(vae, vae_cfg, z):
    if vae_cfg.get("flatten", False):
        z = z.reshape(-1, vae_cfg["latent_dim"])
    return vae.decode(z)


# --------------------------------------------------------------------------- #
#  hard-data (well) conditioning
# --------------------------------------------------------------------------- #
def build_well_inputs_3d(x, n_wells, device, beta=1.0, eps=0.5, thr=0.5, well_margin=0):
    """Sample `n_wells` vertical boreholes through a binary volume x (B,1,D,H,W).
    Depth axis = dim 2; each well is a full depth column at some (h,w) footprint.
    Placement bias P(column) ~ (frac_count + eps)**beta (beta=0 uniform; ~1 favours
    fracture-rich columns).  Returns (value_map, mask), both (B,1,D,H,W): value_map
    carries the observed 0/1 values along well columns, mask marks the observed voxels.
    """
    B, _, D, H, W = x.shape
    value_map = torch.zeros_like(x)
    mask = torch.zeros_like(x)
    frac = (x[:, 0] > thr).float()
    col_count = frac.sum(dim=1)
    valid = None
    if well_margin > 0:
        m = well_margin
        valid = torch.zeros(H, W, device=device)
        valid[m:H - m, m:W - m] = 1.0
        valid = valid.reshape(-1)
    for b in range(B):
        w = (col_count[b].reshape(-1) + eps) ** beta
        if valid is not None:
            w = w * valid
        w = w / w.sum()
        n_avail = int((w > 0).sum().item())
        idx = torch.multinomial(w, min(n_wells, n_avail), replacement=False)
        hi = (idx // W).long(); wi = (idx % W).long()
        value_map[b, 0, :, hi, wi] = x[b, 0, :, hi, wi]
        mask[b, 0, :, hi, wi] = 1.0
    return value_map, mask


def verify_hard_data(generated, value_map, mask, threshold=0.5):
    """Per-voxel satisfaction at well voxels, split fracture vs matrix.
    Returns (mf, mb, nf, nb): matched-fracture, matched-matrix, n-fracture, n-matrix."""
    gen_b = (generated > threshold).float()
    obs_b = (value_map > threshold).float()
    wl = mask.bool()
    match = (gen_b[wl] == obs_b[wl])
    is_frac = obs_b[wl].bool()
    nf = int(is_frac.sum().item()); nb = int((~is_frac).sum().item())
    mf = int(match[is_frac].sum().item()) if nf > 0 else 0
    mb = int(match[~is_frac].sum().item()) if nb > 0 else 0
    return mf, mb, nf, nb


# --------------------------------------------------------------------------- #
#  conditioned generation (CFG + DPS)
# --------------------------------------------------------------------------- #
def generate_conditioned(ldm, enc, vae, vae_cfg, value_map, mask, n_samples, device,
                         latent_mean, sigma_data, latent_shape,
                         w_cfg=3.0, w_dps=4.0, num_steps=32, dps_balanced=False):
    """CFG + DPS conditional Heun sampling. Returns (n,1,D,H,W) volumes in [0,1]."""
    precond = EDMPrecondCond(clone_model(ldm).to(device), sigma_data=sigma_data).to(device)
    with torch.no_grad():
        ctx = enc(value_map, mask).expand(n_samples, -1, -1, -1, -1)
    vmr = value_map.expand(n_samples, -1, -1, -1, -1)
    mkr = mask.expand(n_samples, -1, -1, -1, -1)
    z = edm_sample_cfg_dps(
        precond, ctx, (n_samples, *latent_shape), device,
        vae=vae, vae_decode_fn=lambda zz: vae_decode(vae, vae_cfg, zz),
        value_map=vmr, mask=mkr, latent_mean=latent_mean,
        w_cfg=w_cfg, w_dps=w_dps, num_steps=num_steps, dps_balanced=dps_balanced,
    )
    with torch.no_grad():
        return vae_decode(vae, vae_cfg, z + latent_mean)


def generate_unconditional(ldm, vae, vae_cfg, n_samples, device,
                           latent_mean, sigma_data, latent_shape, num_steps=32):
    """Unconditional EDM sampling: no wells, context=None (w_cfg=1, w_dps=0).
    Same denoiser as the conditioned sampler, just with the well-context and DPS
    pull switched off -- the pure prior the model learned. Returns (n,1,D,H,W) in [0,1]."""
    precond = EDMPrecondCond(clone_model(ldm).to(device), sigma_data=sigma_data).to(device)
    z = edm_sample_cfg_dps(
        precond, None, (n_samples, *latent_shape), device,
        vae=vae, vae_decode_fn=lambda zz: vae_decode(vae, vae_cfg, zz),
        latent_mean=latent_mean, w_cfg=1.0, w_dps=0.0, num_steps=num_steps,
    )
    with torch.no_grad():
        return vae_decode(vae, vae_cfg, z + latent_mean)


# --------------------------------------------------------------------------- #
#  visualization
# --------------------------------------------------------------------------- #
def render_iso(ax, vol, color="indianred", level=0.5, alpha=0.6):
    """Marching-cubes isosurface of a binary/probability volume (D,H,W) onto a 3D ax."""
    from skimage import measure
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection
    vol = np.asarray(vol); n = vol.shape[0]
    try:
        v, f, _, _ = measure.marching_cubes(vol, level=level)
        mesh = Poly3DCollection(v[f], alpha=alpha)
        mesh.set_facecolor(color); mesh.set_edgecolor("none")
        ax.add_collection3d(mesh)
    except (ValueError, RuntimeError):
        pass
    ax.set_xlim(0, n); ax.set_ylim(0, n); ax.set_zlim(0, n)
    ax.set_box_aspect((1, 1, 1)); ax.view_init(20, 35)
    ax.set_xticks([]); ax.set_yticks([]); ax.set_zticks([])


# --------------------------------------------------------------------------- #
#  DFN plane reconstruction  (paper representation: fitted fracture polygons)
#  A raw isosurface is the weakest view; the paper shows each realization as the
#  set of fracture PLANES fitted to the binary field.  Normals come from local
#  PCA (a binary field has a degenerate gradient), then the same RANSAC / merge /
#  convex-hull polygon fit the research repo uses.  numpy + scipy only.
# --------------------------------------------------------------------------- #
C_HIT, C_MISS, C_OBS, C_WELL = "#1a9850", "#d73027", "#2c3e50", "#0570b0"


def surface_frac(occ):
    """Fraction of fracture voxels touching an empty 6-neighbour: thin sheet ~0.9,
    solid blob low.  A cell that hits wells by over-painting shows up as low surf."""
    o = np.asarray(occ).astype(bool)
    n = np.zeros(o.shape, dtype=np.int8)
    for ax in range(o.ndim):
        for sh in (1, -1):
            n += np.roll(o, sh, axis=ax)
    tot = o.sum()
    return float(1.0 - (o & (n == 6)).sum() / tot) if tot else float("nan")


def pca_normals(pts, radius=2.0, min_nb=6):
    """Local-PCA normal per point + planarity (1 - l0/l_mean); ~1 on a clean sheet,
    ~0 inside an isotropic blob.  Replaces grad(T) for a binary field."""
    from scipy.spatial import cKDTree
    tree = cKDTree(pts)
    nn = np.zeros_like(pts)
    planar = np.zeros(len(pts))
    for i, nb in enumerate(tree.query_ball_point(pts, radius)):
        if len(nb) < min_nb:
            nn[i] = (1.0, 0.0, 0.0)
            continue
        q = pts[nb] - pts[nb].mean(0)
        w, v = np.linalg.eigh(q.T @ q / len(nb))
        nn[i] = v[:, 0]                       # smallest eigenvalue -> sheet normal
        planar[i] = 1.0 - w[0] / (w.mean() + 1e-9)
    nn /= np.linalg.norm(nn, axis=1, keepdims=True) + 1e-9
    return nn, planar


def _fit(ip):
    from scipy.spatial import ConvexHull
    c0 = ip.mean(0); _, _, vt = np.linalg.svd(ip - c0, full_matrices=False)
    e1, e2 = vt[0], vt[1]
    proj = np.stack([(ip - c0) @ e1, (ip - c0) @ e2], 1)
    ring = proj[ConvexHull(proj).vertices]
    return c0 + ring[:, :1] * e1 + ring[:, 1:] * e2


def reconstruct_binary(vol, level=0.5, radius=2.0, angle=0.90, dist=2.2,
                       min_inliers=150, merge_ang=0.96, merge_off=3.5):
    """Fit DFN fracture polygons to a binary field: PCA normals -> RANSAC sheets ->
    merge coplanar -> convex-hull polygon.  Returns (list_of_polys, mean_planarity)."""
    pts = np.argwhere(vol > level).astype(float)
    if len(pts) < min_inliers:
        return [], float("nan")
    nn, planar = pca_normals(pts, radius)
    mean_planarity = float(planar.mean())
    clusters, rng = [], np.random.RandomState(0)
    while len(pts) >= min_inliers and len(clusters) < 40:
        best, bn = None, None
        for _ in range(300):
            i = rng.randint(len(pts)); nh = nn[i]; ch = nh @ pts[i]
            inl = (np.abs(pts @ nh - ch) < dist) & (np.abs(nn @ nh) > angle)
            if best is None or inl.sum() > best.sum():
                best, bn = inl, nh
        if best.sum() < min_inliers:
            break
        clusters.append([pts[best], bn, bn @ pts[best].mean(0)])
        pts, nn = pts[~best], nn[~best]
    merged = []
    for ip, n, off in clusters:
        hit = next((m for m in merged if abs(n @ m[1]) > merge_ang
                    and abs(off - np.sign(n @ m[1]) * m[2]) < merge_off), None)
        if hit:
            hit[0] = np.vstack([hit[0], ip])
        else:
            merged.append([ip, n, off])
    out = []
    for ip, n, off in merged:
        try:
            out.append(_fit(ip))
        except Exception:
            pass
    return out, mean_planarity


# Volumes are (D,H,W) with boreholes along axis 0.  ax.voxels maps axis 0 -> x,
# which draws wells horizontal; plotting the transpose (H,W,D) sends the borehole
# axis to z so wells stand upright and z reads as depth.
def _to_plot(pts_dhw):
    """(d,h,w) array coords -> (x,y,z) plot coords, matching vol.transpose(1,2,0)."""
    return pts_dhw[:, [1, 2, 0]]


def _frame(ax, title, D=32):
    ax.set_title(title, fontsize=9)
    ax.set_xticks([]); ax.set_yticks([]); ax.set_zticks([])
    ax.set_xlim(0, 32); ax.set_ylim(0, 32); ax.set_zlim(D, 0)   # depth downward
    ax.set_box_aspect((1, 1, 1)); ax.view_init(elev=18, azim=-60); ax.grid(False)


def _draw_wells(ax, wells, obs_dhw=None, obs_c=None, D=32):
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection  # noqa: F401 (ensures 3d proj)
    for (h, w) in wells:                       # vertical borehole at (x=h, y=w)
        ax.plot([h + 0.5, h + 0.5], [w + 0.5, w + 0.5], [0, D],
                color=C_WELL, lw=2.0, alpha=1.0, zorder=20)
    if obs_dhw is not None and len(obs_dhw):
        q = _to_plot(np.asarray(obs_dhw, float)) + 0.5
        ax.scatter(q[:, 0], q[:, 1], q[:, 2], c=obs_c, s=40, depthshade=False,
                   edgecolors="k", linewidths=0.4, zorder=30)


def render_planes(ax, polys, title, color, wells=(), obs=None, obs_c=None, alpha=0.55):
    """The clean paper view: fitted DFN fracture polygons."""
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection
    _frame(ax, title)
    if polys:
        ax.add_collection3d(Poly3DCollection([_to_plot(p) for p in polys], alpha=alpha,
                                             facecolor=color, edgecolor="k",
                                             linewidths=0.3))
    _draw_wells(ax, wells, obs, obs_c)


def render_voxels(ax, vol, title, color, wells=(), obs=None, obs_c=None, level=0.5,
                  alpha=0.85):
    """The raw binary occupancy the planes were extracted FROM."""
    _frame(ax, title)
    ax.voxels(np.asarray(vol).transpose(1, 2, 0) > level, facecolors=color,
              shade=True, alpha=alpha)
    _draw_wells(ax, wells, obs, obs_c)


def wells_and_obs(value_map, mask, thr=0.5):
    """Extract borehole footprints and fracture-observation voxels from the
    (B,1,D,H,W) value_map/mask that build_well_inputs_3d returns (batch 0).
    Returns wells [(h,w),...] and obs_f (N,3) fracture-intersection voxels (d,h,w)."""
    mkn = mask[0, 0].detach().cpu().numpy() > thr
    vmn = value_map[0, 0].detach().cpu().numpy() > thr
    wells = [tuple(int(c) for c in x) for x in np.argwhere(mkn.any(axis=0))]
    obs_f = np.argwhere(mkn & vmn)             # (d,h,w) fracture observations
    return wells, obs_f


# --------------------------------------------------------------------------- #
#  statistical realism  (paper Figure-3 style: three PoreSpy metrics)
#  porosity box + two-point correlation + chord-length CDF, real vs generated.
# --------------------------------------------------------------------------- #
_STAT_REAL, _STAT_GEN = "#333333", "#2c7fb8"        # grey real, blue generated


def _calc_porosity(d):
    import porespy as ps
    return np.array([ps.metrics.porosity(d[i]) for i in range(len(d))])


def _calc_two_point(d):
    import porespy as ps
    probs = []
    for i in range(len(d)):
        r = ps.metrics.two_point_correlation(d[i])
        probs.append(np.array(r.probability))
    x = np.array(ps.metrics.two_point_correlation(d[0]).distance)
    L = min(len(p) for p in probs)
    return x[:L], np.vstack([p[:L] for p in probs])


def _calc_chord_cdf(d, axis=0):
    import porespy as ps
    cdfs, Lref = [], None
    for i in range(len(d)):
        im = ps.filters.apply_chords(1 - d[i], axis=axis)
        r = ps.metrics.chord_length_distribution(im=im, bins=100, log=True)
        cdfs.append(np.array(r.cdf)); Lref = np.array(r.LogL)
    L = min(len(c) for c in cdfs)
    return Lref[:L], np.vstack([c[:L] for c in cdfs])


def _plot_ci(ax, x, y, color, label, marks):
    m = y.mean(0); ci = 1.96 * y.std(0)
    ax.plot(x, m, color=color, linestyle="--", label=label)
    for idx in marks:
        if idx < len(m):
            ax.errorbar(x[idx], m[idx], yerr=ci[idx], fmt="none", color=color,
                        capsize=3, linewidth=1)


def plot_stats_figure(real, gen):
    """Paper Figure-3 style statistical realism, real vs generated (3D volumes,
    boolean or 0/1).  1 row x 3 cols: fracture volume-fraction box, two-point
    correlation, chord-length CDF.  Requires porespy.  Returns the Figure."""
    real = (np.asarray(real) > 0.5).astype(np.float32)
    gen = (np.asarray(gen) > 0.5).astype(np.float32)
    por_r, por_g = _calc_porosity(real), _calc_porosity(gen)
    tx, tp_r = _calc_two_point(real); _, tp_g = _calc_two_point(gen)
    cx, cc_r = _calc_chord_cdf(real, 0); _, cc_g = _calc_chord_cdf(gen, 0)

    fig, ax = plt.subplots(1, 3, figsize=(13.5, 4.2))
    bp = ax[0].boxplot([por_r, por_g], labels=["Real (held-out)", "Generated"],
                       patch_artist=True, widths=0.5, showfliers=False)
    for patch, c in zip(bp["boxes"], [_STAT_REAL, _STAT_GEN]):
        patch.set_facecolor(c); patch.set_alpha(0.5)
    ax[0].set_ylabel("volume fraction"); ax[0].set_title("Fracture volume fraction")

    marks = list(range(3, len(tx), max(1, len(tx) // 6)))
    _plot_ci(ax[1], tx, tp_r, _STAT_REAL, "Real (held-out)", marks)
    _plot_ci(ax[1], tx, tp_g, _STAT_GEN, "Generated", marks)
    ax[1].set_xlabel("distance (voxels)"); ax[1].set_ylabel("probability")
    ax[1].set_title("Two-point correlation"); ax[1].legend(fontsize=9)

    ax[2].plot(cx, cc_r.mean(0), color=_STAT_REAL, linestyle="--", label="Real (held-out)")
    ax[2].plot(cx, cc_g.mean(0), color=_STAT_GEN, linestyle="--", label="Generated")
    ax[2].set_xlabel("chord length (log)"); ax[2].set_ylabel("CDF")
    ax[2].set_title("Chord length distribution"); ax[2].legend(fontsize=9)
    fig.tight_layout()
    return fig


def plot_realizations_triptych(refv, gens, wells, obs_f, min_inliers=150,
                               alpha_vox=0.25, alpha_pln=0.5, figscale=3.1):
    """Paper Figure-7 style: reference + N realizations, each shown three ways stacked
    in a column -- raw voxels, fitted DFN planes, and one borehole cross-section per
    well with hit/miss markers.  Returns the matplotlib Figure."""
    from matplotlib.colors import ListedColormap
    from matplotlib.lines import Line2D
    gens = [np.asarray(g) for g in gens]
    refv = np.asarray(refv)
    n = len(gens); ncol = 1 + n
    nrow = 2 + len(wells)
    cmap = ListedColormap(["#ffffff", "#2c7fb8"])
    ref_polys, _ = reconstruct_binary(refv, min_inliers=min_inliers)
    gen_polys = [reconstruct_binary(g, min_inliers=min_inliers)[0] for g in gens]

    fig = plt.figure(figsize=(figscale * ncol, figscale * 1.1 * nrow))
    panels = [(refv, ref_polys, None)] + [(gens[j], gen_polys[j], j) for j in range(n)]
    for k, (vol, polys, j) in enumerate(panels):
        is_ref = j is None
        hit = None if is_ref else vol[obs_f[:, 0], obs_f[:, 1], obs_f[:, 2]] > 0.5
        title0 = "reference (truth)" if is_ref else f"realization {j + 1}"
        wsat = "" if is_ref else f"   well {100 * hit.mean():.0f}%"
        face = "#9a9a9a" if is_ref else "#2c7fb8"
        oc = [C_OBS] * len(obs_f) if is_ref else np.where(hit, C_HIT, C_MISS)

        ax = fig.add_subplot(nrow, ncol, 0 * ncol + k + 1, projection="3d")
        render_voxels(ax, vol, f"{title0}{wsat}\nfracvol {100 * (vol > 0.5).mean():.0f}%"
                      f"  surf {surface_frac(vol > 0.5):.2f}", face,
                      wells, obs_f, oc, alpha=alpha_vox)
        ax = fig.add_subplot(nrow, ncol, 1 * ncol + k + 1, projection="3d")
        render_planes(ax, polys, f"{len(polys)} planes", face, wells, obs_f, oc, alpha_pln)

        for r, (h, w) in enumerate(wells):
            ax = fig.add_subplot(nrow, ncol, (2 + r) * ncol + k + 1)
            ax.imshow(vol[:, h, :] > 0.5, cmap=cmap, vmin=0, vmax=1,
                      interpolation="nearest", origin="upper")
            ax.axvline(w, color=C_WELL, lw=2.0, alpha=0.9)
            on = obs_f[obs_f[:, 1] == h]
            c = [C_OBS] * len(on) if is_ref else np.where(
                vol[on[:, 0], on[:, 1], on[:, 2]] > 0.5, C_HIT, C_MISS)
            ax.scatter([w] * len(on), on[:, 0], c=c, s=30, zorder=5,
                       edgecolors="k", linewidths=0.5)
            ax.set_title(f"slice fracvol {100 * (vol[:, h, :] > 0.5).mean():.0f}%",
                         fontsize=8)
            ax.set_xticks([]); ax.set_yticks([])

    rlabels = ["voxel", "fitted planes"] + \
              [f"slice: well {r + 1} (h={h})" for r, (h, w) in enumerate(wells)]
    for r, lab in enumerate(rlabels):
        ax = fig.axes[r]
        if r < 2:
            ax.text2D(-0.14, 0.5, lab, transform=ax.transAxes, fontsize=11, rotation=90,
                      va="center", ha="center", fontweight="bold")
        else:
            ax.set_ylabel(lab, fontsize=11, fontweight="bold")
    fig.legend(handles=[Line2D([], [], color=C_WELL, lw=2.4, label="borehole"),
                        Line2D([], [], marker="o", ls="", color=C_OBS,
                               label="fracture obs (reference)"),
                        Line2D([], [], marker="o", ls="", color=C_HIT, label="honoured"),
                        Line2D([], [], marker="o", ls="", color=C_MISS, label="missed")],
               loc="lower center", ncol=4, fontsize=10, frameon=False)
    fig.tight_layout(rect=[0.02, 0.04, 1, 0.98])
    return fig

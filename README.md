# SCIPE Workshop — 3D Latent Diffusion for Fracture Networks (on Koa)

A hands-on, **self-contained** workshop notebook that runs a pretrained **3D
Latent Diffusion Model (LDM)** to generate discrete fracture networks (DFNs) on a
$32^3$ voxel grid, and conditions the generation on sparse **well observations**
(hard data). Based on the 3D binary model from the "voxel" fracture-generation
paper.

Everything the notebook needs is in this folder — it does **not** import from any
research repository.

## Folder layout
```
SCIPE_workshop/
├── workshop_3d_ldm.ipynb     # the workshop (9 sections; run top-to-bottom)
├── scipe_helpers.py          # self-contained load / condition / generate / eval helpers
├── models/                   # copied model code (VAE, diffusion UNet, CondEncoder, EDM sampler)
├── checkpoints/
│   ├── UnetVAE.pt            # frozen VAE            (63 MB)
│   ├── VAE_config.yml
│   ├── LDM_ema.pt            # diffusion denoiser, EMA weights (1.5 GB — NOT in git; download, see below)
│   ├── CondEncoder_ema.pt    # wells -> context     (0.4 MB)
│   └── latent_stats.yml      # latent normalization for the sampler
├── data/binary3d_subset.npy  # 600 held-out binary volumes (32^3, uint8, ~20 MB)
├── environment.yml           # conda env "scipe-ldm"
├── download_checkpoint.sh    # fetches LDM_ema.pt from the GitHub Release
├── koa_train.sbatch          # SLURM job for full training (SHOW-ONLY, section 9)
└── README.md
```

## Setup on Koa
```bash
git clone https://github.com/YOUR_GITHUB_USER/SCIPE_workshop.git
cd SCIPE_workshop

# 1. get the large diffusion checkpoint (1.5 GB) — it is NOT in the repo,
#    it lives as a GitHub Release asset (see "Notes on the large checkpoint")
./download_checkpoint.sh                  # -> checkpoints/LDM_ema.pt

# 2. build the environment
module load lang/Anaconda3
conda env create -f environment.yml      # one-time; creates env "scipe-ldm"
conda activate scipe-ldm
```
> `download_checkpoint.sh` uses the `gh` CLI if available, otherwise `wget`.
> Edit the `REPO`/`TAG` lines at the top of the script to match where you
> uploaded the release asset (or pass them as env vars).
The notebook is **inference only** and runs in seconds per step on a GPU — fine on
an interactive GPU session. It also runs on CPU (slower). Do **not** run heavy
training on the login node; that goes through `sbatch` (see section 9).

## Run the notebook
```bash
conda activate scipe-ldm
jupyter lab workshop_3d_ldm.ipynb        # or: jupyter notebook
```
Run the cells top to bottom. Sections are marked:
- 🟢 **RUN** — execute during the workshop
- 📄 **SHOW-ONLY** — read, don't run (full training; takes hours)

| # | section | | # | section |
|---|---|---|---|---|
| 1 | Load dataset 🟢 | | 6 | Load checkpoints 🟢 |
| 2 | Visualize geometries 🟢 | | 7 | Generate (well-conditioned) 🟢 |
| 3 | Dataset / DataLoader 🟢 | | 8 | Evaluate 🟢 |
| 4 | Define models 🟢 | | 9 | Full training on Koa 📄 |
| 5 | Training code 📄 | | | |

## The model (provenance)
- **Representation:** binary voxel occupancy, $32^3$; a voxel is 1 where a fracture
  passes through it (obtained by thresholding a distance field at 0.5).
- **VAE:** `UNetVAE`, compresses $32^3 \to 8^3$ latent (16.4 M params).
- **Diffusion:** EDM-formulation denoiser `Unet_SR` in the frozen latent (386 M params).
- **Conditioning:** `CondEncoder3D` turns 1–2 vertical-borehole observations into a
  context volume; generation uses classifier-free guidance (`w_cfg`) + DPS
  measurement guidance (`w_dps`). Recommended operating point: `w_cfg=3, w_dps=4,
  32 steps`. The shipped model was trained on **1–2 wells**, so condition on 1–2
  wells (more is out-of-distribution).
- **Data shipped:** 600 **held-out** volumes (never seen in training), so section 8's
  comparison measures generalization.

## Notes on the large checkpoint
`checkpoints/LDM_ema.pt` is 1.5 GB (a 386 M-parameter model in fp32), which
exceeds GitHub's 100 MB per-file limit, so it is **not** committed to the repo
(it is listed in `.gitignore`). Instead it is distributed as an asset on a
GitHub **Release** (assets may be up to 2 GB), and `download_checkpoint.sh`
pulls it into `checkpoints/` after cloning. Its MD5 is
`51dad29c3d697584b0b68dede5f1ecd3` (the download script verifies this).

**To publish the checkpoint (one-time, repo owner):**
```bash
# after creating the GitHub repo and pushing the code:
gh release create checkpoints checkpoints/LDM_ema.pt \
    --title "Pretrained checkpoints" \
    --notes "LDM_ema.pt (1.5 GB) — fetched by download_checkpoint.sh"
```
The smaller checkpoints (VAE 63 MB, CondEncoder 0.4 MB) are under the 100 MB
limit and stay in the repo, so only `LDM_ema.pt` needs downloading.

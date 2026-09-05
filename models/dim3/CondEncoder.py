import torch
import torch.nn as nn


class CondEncoder3D(nn.Module):
    """Encode sparse vertical-well observations into a (B, 1, 8, 8, 8) context volume.

    3D analogue of the 2D CondEncoder.  Input: two (B, 1, D, H, W) volumes at the
    original image resolution (32^3)
      - value_map: observed fracture values at well voxels, 0 elsewhere
      - mask:      1.0 at well voxels (a full vertical borehole column), 0 elsewhere
    Output: (B, 1, 8, 8, 8) context ready for Unet_SR (3D) channel-concat conditioning.

    Two stride-2 conv blocks shrink 32 -> 16 -> 8, matching the 1x8x8x8 VAE latent.
    """

    def __init__(self, hidden_dim: int = 32, image_size: int = 32, in_ch: int = 2):
        super().__init__()
        d = hidden_dim
        self.in_ch = in_ch
        self.net = nn.Sequential(
            # 32 -> 16
            nn.Conv3d(in_ch, d, kernel_size=3, stride=2, padding=1),
            nn.GroupNorm(8, d),
            nn.SiLU(),
            # 16 -> 8
            nn.Conv3d(d, d * 2, kernel_size=3, stride=2, padding=1),
            nn.GroupNorm(8, d * 2),
            nn.SiLU(),
            # refine at 8^3
            nn.Conv3d(d * 2, d, kernel_size=3, stride=1, padding=1),
            nn.GroupNorm(8, d),
            nn.SiLU(),
            # project to 1 channel
            nn.Conv3d(d, 1, kernel_size=1),
        )

    def forward(self, *maps: torch.Tensor) -> torch.Tensor:
        """
        Args:
            *maps: `in_ch` volumes of shape (B, 1, D, H, W), concatenated on the channel dim.
              in_ch=2 (legacy binary wells): value_map, mask
              in_ch=4 (plane-aware wells):   target, tmask, fmask, mmask  -- see
                                             models/dim3/well_planes.py
        Returns:
            (B, 1, D/4, H/4, W/4) context volume (8^3 for 32^3 input)
        """
        assert len(maps) == self.in_ch, f"CondEncoder3D(in_ch={self.in_ch}) got {len(maps)} maps"
        return self.net(torch.cat(maps, dim=1))

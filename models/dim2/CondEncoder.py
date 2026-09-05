import torch
import torch.nn as nn


class CondEncoder(nn.Module):
    """Encode sparse well observations into a (B, 1, 16, 16) context map.

    Input: two (B, 1, H, W) maps at the original image resolution
      - value_map: observed fracture values at well locations, 0 elsewhere
      - mask:      1.0 at well locations, 0 elsewhere
    Output: (B, 1, 16, 16) context ready for Unet_SR conditioning.

    Three stride-2 conv blocks shrink 128 → 64 → 32 → 16.
    """

    def __init__(self, hidden_dim: int = 32, image_size: int = 128):
        super().__init__()
        d = hidden_dim
        self.net = nn.Sequential(
            # 128 → 64
            nn.Conv2d(2, d, kernel_size=3, stride=2, padding=1),
            nn.GroupNorm(8, d),
            nn.SiLU(),
            # 64 → 32
            nn.Conv2d(d, d * 2, kernel_size=3, stride=2, padding=1),
            nn.GroupNorm(8, d * 2),
            nn.SiLU(),
            # 32 → 16
            nn.Conv2d(d * 2, d, kernel_size=3, stride=2, padding=1),
            nn.GroupNorm(8, d),
            nn.SiLU(),
            # project to 1 channel
            nn.Conv2d(d, 1, kernel_size=1),
        )

    def forward(self, value_map: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """
        Args:
            value_map: (B, 1, H, W) – well values at observed locations, 0 elsewhere
            mask:      (B, 1, H, W) – 1.0 at well locations, 0 elsewhere
        Returns:
            (B, 1, H/8, W/8) context map
        """
        x = torch.cat([value_map, mask], dim=1)  # (B, 2, H, W)
        return self.net(x)

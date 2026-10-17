"""``DenseBlock``: Dense block: L layers with all-pairs skip connections."""

from __future__ import annotations

import torch
import torch.nn as nn

from hale_core.nn.vision.cnn.densenet._helpers import _DenseLayer


class DenseBlock(nn.Module):
    """Dense block: L layers with all-pairs skip connections.

    Each layer's input is the concatenation of ALL previous layer outputs
    plus the block's initial input.  This means layer ℓ has
    in_ch + ℓ * growth_rate input channels.

    Args:
        num_layers:          L — number of dense layers.
        num_input_features:  Channels at block entry.
        growth_rate:         k — channels added per layer.
    """

    def __init__(self, num_layers: int, num_input_features: int, growth_rate: int) -> None:
        super().__init__()
        self.layers = nn.ModuleList()
        for i in range(num_layers):
            self.layers.append(
                _DenseLayer(
                    num_input_features=num_input_features + i * growth_rate,
                    growth_rate=growth_rate,
                )
            )
        # Channels after this block
        self.out_channels = num_input_features + num_layers * growth_rate

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of :class:`DenseBlock`.

        Args:
            x: input (``torch.Tensor``).

        Returns:
            ``torch.Tensor``.
        """
        # Accumulate all feature maps in a list and concat for each new layer
        features = [x]
        for layer in self.layers:
            new_feat = layer(torch.cat(features, dim=1))
            features.append(new_feat)
        return torch.cat(features, dim=1)  # [B, num_input + L*k, H, W]

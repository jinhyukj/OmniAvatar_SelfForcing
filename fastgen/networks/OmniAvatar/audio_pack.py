# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
AudioPack module for OmniAvatar audio conditioning.

Adapted from OmniAvatar/OmniAvatar/models/audio_pack.py.
Projects Wav2Vec2 features (10752-dim) into a compact representation (32-dim)
via temporal patching and linear projection.
"""

import torch
from typing import Tuple, Union
from einops import rearrange
from torch import nn


def make_triple(value: Union[int, Tuple[int, int, int]]) -> Tuple[int, int, int]:
    value = (value,) * 3 if isinstance(value, int) else value
    assert len(value) == 3
    return value


class AudioPack(nn.Module):
    """Temporal patchification + linear projection for audio features.

    Takes Wav2Vec2 all-layer features [B, C_audio, T_video, 1, 1] and produces
    compact audio tokens [B, T_groups, 1, 1, dim] where T_groups = T_video / patch_t.

    Args:
        in_channels: Input feature dimension (10752 for Wav2Vec2 all-layer concat).
        patch_size: Temporal patching size, typically (4, 1, 1).
        dim: Output dimension (32 = audio_hidden_size).
        layernorm: Whether to apply LayerNorm after projection.
    """

    def __init__(
        self,
        in_channels: int,
        patch_size: Union[int, Tuple[int, int, int]],
        dim: int,
        layernorm: bool = False,
    ):
        super().__init__()
        t, h, w = make_triple(patch_size)
        self.patch_size = t, h, w
        self.proj = nn.Linear(in_channels * t * h * w, dim)
        if layernorm:
            self.norm_out = nn.LayerNorm(dim)
        else:
            self.norm_out = None

    def forward(self, vid: torch.Tensor) -> torch.Tensor:
        """
        Args:
            vid: [B, C_audio, T_video+padding, 1, 1]

        Returns:
            [B, T_groups, 1, 1, dim]
        """
        t, h, w = self.patch_size
        vid = rearrange(vid, "b c (T t) (H h) (W w) -> b T H W (t h w c)", t=t, h=h, w=w)
        vid = self.proj(vid)
        if self.norm_out is not None:
            vid = self.norm_out(vid)
        return vid

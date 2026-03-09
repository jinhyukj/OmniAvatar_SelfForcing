# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
OmniAvatar Self-Forcing model.

Minimal override of SelfForcingModel to handle OmniAvatar's audio + V2V conditioning.
The only change is in _prepare_training_data(), which constructs the condition dict
with audio, reference frames, masked video, and spatial mask.
"""

from __future__ import annotations

from typing import Any, Dict, TYPE_CHECKING

import torch

from fastgen.methods.distribution_matching.self_forcing import SelfForcingModel

if TYPE_CHECKING:
    from fastgen.configs.methods.config_omniavatar_sf import ModelConfig


class OmniAvatarSelfForcingModel(SelfForcingModel):
    """Self-Forcing model with OmniAvatar-specific data preparation.

    Inherits the full Self-Forcing / DMD2 training pipeline:
    - rollout_with_gradient() for causal student generation
    - VSD loss (fake_score_x0 - teacher_x0)
    - DSM loss for fake_score training
    - Alternating student / fake_score updates

    The only override is _prepare_training_data() which packages
    OmniAvatar-specific fields (audio, ref, mask, masked_video)
    into the condition dict that all three networks (student, teacher,
    fake_score) receive.
    """

    def __init__(self, config: ModelConfig):
        super().__init__(config)

    def _prepare_training_data(self, data: Dict[str, Any]) -> tuple[torch.Tensor, Any, Any]:
        """Prepare OmniAvatar training data.

        Constructs condition dicts that carry all information needed by
        I2V teacher/fake_score and V2V student. Each network's forward()
        extracts what it needs from the dict.

        Args:
            data: Raw data dict from OmniAvatarDataLoader containing:
                - real: [B, 16, T, H, W] target latents
                - condition: [B, L, D] text embeddings
                - neg_condition: [B, L, D] negative text embeddings
                - audio_emb: [B, T_video, 10752] Wav2Vec2 features
                - ref_latent: [B, 16, 1, H, W] reference frame latent
                - masked_video: [B, 16, T, H, W] mouth-masked video latents
                - mask: [B, H, W] or [H, W] spatial mask (1=keep, 0=generate)

        Returns:
            (real_data, condition, neg_condition) where condition dicts
            contain all fields needed by all three networks.
        """
        real_data = data["real"]

        condition = {
            "text_embeds": data["condition"],
            "audio_emb": data["audio_emb"],
            "ref_latent": data["ref_latent"],
            "masked_video": data["masked_video"],
            "mask": data["mask"],
        }

        # Negative condition: no text, no audio (for CFG at teacher)
        neg_condition = {
            "text_embeds": data["neg_condition"],
            "audio_emb": torch.zeros_like(data["audio_emb"]),
            "ref_latent": data["ref_latent"],
            "masked_video": data["masked_video"],
            "mask": data["mask"],
        }

        return real_data, condition, neg_condition

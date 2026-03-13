# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
OmniAvatar Causal KD (Knowledge Distillation) model.

Thin wrapper around CausalKDModel that reformats the data dict to build
the OmniAvatar condition dict (audio, ref, mask, masked_video) before
calling the causal KD training logic with inhomogeneous timesteps.

Same pattern as OmniAvatarSelfForcingModel — the only override is
data preparation, all KD training logic is inherited from CausalKDModel.
"""

from __future__ import annotations

from typing import Any, Dict, TYPE_CHECKING

from fastgen.methods.knowledge_distillation.KD import CausalKDModel

if TYPE_CHECKING:
    from fastgen.configs.config import BaseModelConfig as ModelConfig


class OmniAvatarKDModel(CausalKDModel):
    """Causal KD with OmniAvatar-specific condition dict construction.

    The OmniAvatarDataLoader returns condition as a raw text embedding tensor,
    plus separate fields for audio_emb, ref_latent, masked_video, mask.
    CausalOmniAvatarWan.forward() expects a single condition dict with all of these.
    This override builds that dict before calling CausalKDModel's training logic,
    which uses inhomogeneous timesteps (different t per chunk) matching the
    causal student's inference behavior.
    """

    def __init__(self, config: ModelConfig):
        super().__init__(config)

    def single_train_step(
        self, data: Dict[str, Any], iteration: int
    ) -> tuple[dict, dict]:
        # Rebuild condition as the dict that CausalOmniAvatarWan.forward() expects
        data["condition"] = {
            "text_embeds": data["condition"],
            "audio_emb": data["audio_emb"],
            "ref_latent": data["ref_latent"],
            "masked_video": data["masked_video"],
            "mask": data["mask"],
        }
        return super().single_train_step(data, iteration)

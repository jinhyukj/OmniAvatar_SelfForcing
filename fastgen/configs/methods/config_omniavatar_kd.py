# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Method config for OmniAvatar KD (Knowledge Distillation) pretraining.

Uses standard non-causal KDModel with OmniAvatar condition dict wrapper.
Extends the base config_kd with OmniAvatar-specific settings.
"""

from omegaconf import DictConfig

from fastgen.utils import LazyCall as L
import fastgen.configs.methods.config_kd as config_kd_default
from fastgen.methods.omniavatar_kd import OmniAvatarKDModel


def create_config():
    config = config_kd_default.create_config()

    # Use OmniAvatar KD model (condition dict wrapper)
    config.model_class = L(OmniAvatarKDModel)(config=None)

    # 4-step KD aligned with ODE pair timesteps
    config.model.student_sample_steps = 4
    config.model.sample_t_cfg.t_list = [0.999, 0.937, 0.833, 0.624, 0.0]

    config.model.precision = "bfloat16"
    config.model.enable_preprocessors = False

    return config

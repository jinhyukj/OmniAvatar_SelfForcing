# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Experiment config: OmniAvatar Causal KD pretraining.

Student: V2V 1.3B CausalOmniAvatarWan (causal, chunk-by-chunk with KV cache)
Loss: MSE on ODE pairs with inhomogeneous timesteps (different t per chunk)

After KD training, load the checkpoint into the causal V2V student
for Self-Forcing (config_sf.py) via pretrained_student_net_path / KD_CKPT_PATH.
"""

import os

from fastgen.utils import LazyCall as L
import fastgen.configs.methods.config_omniavatar_kd as config_omniavatar_kd
from fastgen.datasets.omniavatar_dataloader import OmniAvatarDataLoader
from fastgen.networks.OmniAvatar import CausalOmniAvatarWan

# Default paths — override via environment variables or CLI
CKPT_ROOT_DIR = os.getenv("CKPT_ROOT_DIR", "pretrained_models")
OMNIAVATAR_ROOT = os.getenv("OMNIAVATAR_ROOT", os.path.join(CKPT_ROOT_DIR, "OmniAvatar_LoRA"))

# --------------------------------------------------------------------------- #
# Network config: V2V 1.3B causal student (same arch as SF student)
# --------------------------------------------------------------------------- #
CausalOmniAvatar_V2V_1_3B_Config = L(CausalOmniAvatarWan)(
    in_dim=49,  # V2V: noise(16) + ref(16) + mask(1) + masked_vid(16)
    dim=1536,
    num_heads=12,
    ffn_dim=8960,
    num_layers=30,
    chunk_size=3,
    total_num_frames=21,
    use_audio=True,
    audio_hidden_size=32,
    has_image_input=False,
    base_model_paths=f"{CKPT_ROOT_DIR}/Wan2.1-T2V-1.3B/diffusion_pytorch_model.safetensors",
    omniavatar_ckpt_path=f"{OMNIAVATAR_ROOT}/OmniAvatar-1.3B",
    merge_lora=True,
    net_pred_type="flow",
    schedule_type="rf",
)


# --------------------------------------------------------------------------- #
# Full experiment config
# --------------------------------------------------------------------------- #
def create_config():
    config = config_omniavatar_kd.create_config()

    # Student: V2V 1.3B causal
    config.model.net = CausalOmniAvatar_V2V_1_3B_Config
    config.model.net.total_num_frames = 21  # = input_shape[1]

    # Input shape: [C=16, T_lat=21, H_lat=64, W_lat=64] for 81-frame 512x512
    config.model.input_shape = [16, 21, 64, 64]

    # Timestep sampling for causal model
    config.model.sample_t_cfg.time_dist_type = "shifted"
    config.model.sample_t_cfg.min_t = 0.001
    config.model.sample_t_cfg.max_t = 0.999

    # Optimizer
    config.model.net_optimizer.lr = 1e-5

    # Weights loaded by network __init__, not by KD build_model
    config.model.pretrained_model_path = ""
    config.model.load_student_weights = False

    # Dataloader
    config.dataloader_train = L(OmniAvatarDataLoader)(
        datatags=["PLACEHOLDER"],  # Override via CLI
        latentsync_mask_path="PLACEHOLDER",  # Override via CLI
        num_frames=81,
        mask_all_frames=True,
        batch_size=1,
        num_workers=2,
    )

    # Training
    config.trainer.max_iter = 10000
    config.trainer.logging_iter = 100
    config.trainer.save_ckpt_iter = 1000
    config.trainer.ddp = True

    # Logging
    config.log_config.group = "omniavatar_kd_causal"
    config.log_config.name = "omniavatar_kd_causal_v2v_1.3B"

    return config

# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Experiment config: OmniAvatar Self-Forcing distillation.

Teacher: I2V 14B OmniAvatar (bidirectional, frozen)
Student: V2V 1.3B OmniAvatar (causal, trainable)
Fake_score: I2V 1.3B OmniAvatar (bidirectional, trainable on DSM loss)
Loss: VSD only (no GAN discriminator)
"""

import os

from fastgen.utils import LazyCall as L
import fastgen.configs.methods.config_omniavatar_sf as config_omniavatar_sf
from fastgen.datasets.omniavatar_dataloader import OmniAvatarDataLoader
from fastgen.networks.OmniAvatar import OmniAvatarWan, CausalOmniAvatarWan

# Default paths — override via environment variables or CLI
CKPT_ROOT_DIR = os.getenv("CKPT_ROOT_DIR", "pretrained_models")
OMNIAVATAR_ROOT = os.getenv("OMNIAVATAR_ROOT", os.path.join(CKPT_ROOT_DIR, "OmniAvatar_LoRA"))

# --------------------------------------------------------------------------- #
# Network configs
# --------------------------------------------------------------------------- #
# Teacher: I2V 14B OmniAvatar (bidirectional, frozen)
OmniAvatar_I2V_14B_Config = L(OmniAvatarWan)(
    in_dim=33,  # noise(16) + ref(16) + mask(1)
    dim=5120,
    num_heads=40,
    ffn_dim=13824,
    num_layers=40,
    mode="i2v",
    use_audio=True,
    audio_hidden_size=32,
    has_image_input=False,
    base_model_paths=",".join(
        [
            f"{CKPT_ROOT_DIR}/Wan2.1-T2V-14B/diffusion_pytorch_model-0000{i}-of-00006.safetensors"
            for i in range(1, 7)
        ]
    ),
    omniavatar_ckpt_path=f"{OMNIAVATAR_ROOT}/OmniAvatar-14B",
    merge_lora=True,
    net_pred_type="flow",
    schedule_type="rf",
)

# Fake_score: I2V 1.3B OmniAvatar (bidirectional, trainable on DSM)
OmniAvatar_I2V_1_3B_Config = L(OmniAvatarWan)(
    in_dim=33,
    dim=1536,
    num_heads=12,
    ffn_dim=8960,
    num_layers=30,
    mode="i2v",
    use_audio=True,
    audio_hidden_size=32,
    has_image_input=False,
    base_model_paths=f"{CKPT_ROOT_DIR}/Wan2.1-T2V-1.3B/diffusion_pytorch_model.safetensors",
    omniavatar_ckpt_path=f"{OMNIAVATAR_ROOT}/OmniAvatar-1.3B",
    merge_lora=True,
    net_pred_type="flow",
    schedule_type="rf",
)

# Student: V2V 1.3B causal OmniAvatar (causal, trainable)
CausalOmniAvatar_V2V_1_3B_Config = L(CausalOmniAvatarWan)(
    in_dim=49,  # noise(16) + ref(16) + mask(1) + masked_vid(16)
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
    omniavatar_ckpt_path=f"{OMNIAVATAR_ROOT}/OmniAvatar-1.3B",  # I2V ckpt, expanded to V2V
    merge_lora=True,
    net_pred_type="flow",
    schedule_type="rf",
)


# --------------------------------------------------------------------------- #
# Full experiment config
# --------------------------------------------------------------------------- #
def create_config():
    config = config_omniavatar_sf.create_config()

    # Model architecture
    config.model.net = CausalOmniAvatar_V2V_1_3B_Config
    config.model.teacher = OmniAvatar_I2V_14B_Config
    config.model.fake_score_net = OmniAvatar_I2V_1_3B_Config

    # No discriminator (pure VSD)
    config.model.gan_loss_weight_gen = 0

    # Load KD-pretrained student weights if available (from Stage 2 KD pretraining)
    # Set KD_CKPT_PATH env var or override via CLI: model.pretrained_student_net_path=...
    kd_ckpt = os.getenv("KD_CKPT_PATH", "")
    config.model.pretrained_student_net_path = kd_ckpt
    config.model.load_student_weights = bool(kd_ckpt)

    # Base model weights loaded by network __init__
    config.model.pretrained_model_path = ""

    # Precision
    config.model.precision = "bfloat16"

    # Input shape: [C=16, T_lat=21, H_lat=64, W_lat=64] for 81-frame 512x512
    config.model.input_shape = [16, 21, 64, 64]

    # Self-Forcing settings
    config.model.student_sample_steps = 4
    config.model.fake_score_pred_type = "x0"
    config.model.guidance_scale = 4.0

    # Timestep sampling
    config.model.sample_t_cfg.time_dist_type = "shifted"
    config.model.sample_t_cfg.min_t = 0.001
    config.model.sample_t_cfg.max_t = 0.999
    config.model.sample_t_cfg.t_list = [0.999, 0.937, 0.833, 0.624, 0.0]

    # Optimizer
    config.model.net_optimizer.lr = 5e-5
    config.model.fake_score_optimizer.lr = 5e-5

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
    config.trainer.max_iter = 5000
    config.trainer.logging_iter = 100
    config.trainer.save_ckpt_iter = 500
    config.trainer.ddp = True

    # Logging
    config.log_config.group = "omniavatar_sf"
    config.log_config.name = "omniavatar_sf_v2v_1.3B"

    return config

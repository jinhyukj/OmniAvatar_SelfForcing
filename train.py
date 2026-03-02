# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
The training entry script for the FastGen project. Works for both DDP and FSDP training.
"""

import argparse
import warnings

from fastgen.configs.config import BaseConfig
from fastgen.utils import instantiate
from fastgen.trainer import Trainer
import fastgen.utils.logging_utils as logger
from fastgen.utils.distributed import synchronize, clean_up
from fastgen.utils.scripts import parse_args, setup

warnings.filterwarnings(
    "ignore", "Grad strides do not match bucket view strides"
)  # False warning printed by PyTorch 2.6.


def main(config: BaseConfig):
    # initiate the model
    """
      From BaseModelConfig:
        - net = CausalVACE_Wan_1_3B_Config (LazyCall for the student network)
        - teacher = VACE_Wan_1_3B_Config (LazyCall for the teacher network)
        - precision = "bfloat16"
        - precision_fsdp = "float32"
        - input_shape = [16, 21, 60, 104]
        - guidance_scale = 4.0
        - student_sample_steps = 4
        - net_optimizer, net_scheduler (lr=5e-6)

        Added by DMD2 ModelConfig:
        - discriminator = Discriminator_Wan_1_3B_Config
        - fake_score_optimizer, fake_score_scheduler (lr=5e-6)
        - discriminator_optimizer, discriminator_scheduler (lr=5e-6)
        - gan_loss_weight_gen = 0.003
        - fake_score_pred_type = "x0"
        - student_update_freq = 5

        Added by SelfForcing ModelConfig:
        - enable_gradient_in_rollout = True
        - start_gradient_frame = 0
        - same_step_across_blocks = True
        - last_step_only = False
        - context_noise = 0.0
    """
    config.model_class.config = config.model        ##  {"_target_": SelfForcingModel, "config": <the ModelConfig object>}

    """
    1. Sees it's a Mapping with "_target_" in it (line 72)
    2. Recursively instantiates all values — config (the ModelConfig) is not a LazyCall itself, so it's returned as-is
    3. Pops _target_ → gets SelfForcingModel class
    4. Calls SelfForcingModel(config=<ModelConfig>) (line 94)
    """
    model = instantiate(config.model_class)
    config.model_class.config = None
    synchronize()

    # initiate the trainer
    logger.info("Initializing trainer...")
    fastgen_trainer = Trainer(config)
    logger.success("Trainer initialized successfully")
    synchronize()

    # Start training
    fastgen_trainer.run(model)
    synchronize()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Training")
    args = parse_args(parser)
    config = setup(args)

    main(config)

    clean_up()
    logger.info("Training finished.")

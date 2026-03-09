# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Non-causal OmniAvatar network wrapper for FastGen.

Used for teacher (I2V 14B) and fake_score (I2V 1.3B) — bidirectional models
that score perturbed data in a single forward pass.
"""

from __future__ import annotations

import os
from typing import Any, Dict, List, Optional, Set, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.init as init

from fastgen.networks.network import FastGenNetwork
from .wan_model import WanModel

import fastgen.utils.logging_utils as logger


def _smart_load_state_dict(
    model: nn.Module,
    state_dict: Dict[str, torch.Tensor],
    strict: bool = False,
) -> None:
    """Load state dict with shape-mismatch handling (partial copy for patch_embedding etc.)."""
    model_sd = model.state_dict()
    filtered_sd = {}
    for key, value in state_dict.items():
        if key in model_sd:
            if model_sd[key].shape == value.shape:
                filtered_sd[key] = value
            else:
                # Shape mismatch — copy what fits
                target = model_sd[key].clone()
                src_shape = value.shape
                tgt_shape = target.shape
                slices = tuple(slice(0, min(s, t)) for s, t in zip(src_shape, tgt_shape))
                target[slices] = value[slices]
                filtered_sd[key] = target
                logger.info(f"Partial load for {key}: {src_shape} -> {tgt_shape}")
        else:
            if strict:
                raise KeyError(f"Unexpected key in state_dict: {key}")
    info = model.load_state_dict(filtered_sd, strict=False)
    if info.missing_keys:
        logger.debug(f"Missing keys (expected for new/audio params): {info.missing_keys[:10]}...")


def _load_safetensors(paths: Union[str, List[str]]) -> Dict[str, torch.Tensor]:
    """Load state dict from one or more safetensor files."""
    from safetensors.torch import load_file

    if isinstance(paths, str):
        paths = [paths]
    state_dict = {}
    for p in paths:
        state_dict.update(load_file(p, device="cpu"))
    return state_dict


def _merge_lora_into_base(
    model: nn.Module,
    lora_sd: Dict[str, torch.Tensor],
    lora_scale: float = 1.0,
) -> None:
    """Merge LoRA weights (lora_A, lora_B) into base linear layers.

    Expects keys like: blocks.0.self_attn.q.lora_A.default.weight
    Merges into: blocks.0.self_attn.q.weight via W += lora_B @ lora_A * scale
    """
    model_sd = dict(model.named_parameters())
    merged_count = 0

    # Collect all lora_B keys
    lora_b_keys = [k for k in lora_sd if "lora_B" in k]
    for lora_b_key in lora_b_keys:
        lora_a_key = lora_b_key.replace("lora_B", "lora_A")
        if lora_a_key not in lora_sd:
            continue

        # Derive the base weight key
        # e.g., "blocks.0.self_attn.q.lora_B.default.weight" -> "blocks.0.self_attn.q.weight"
        base_key = lora_b_key.split(".lora_B")[0] + ".weight"
        if base_key not in model_sd:
            logger.warning(f"Cannot find base param {base_key} for LoRA merge, skipping")
            continue

        lora_a = lora_sd[lora_a_key].to(model_sd[base_key].dtype).to(model_sd[base_key].device)
        lora_b = lora_sd[lora_b_key].to(model_sd[base_key].dtype).to(model_sd[base_key].device)
        model_sd[base_key].data += (lora_b @ lora_a) * lora_scale
        merged_count += 1

    logger.info(f"Merged {merged_count} LoRA weight pairs into base weights")


class OmniAvatarWan(FastGenNetwork):
    """Non-causal OmniAvatar Wan network for FastGen (teacher / fake_score).

    Wraps OmniAvatar's custom WanModel and provides the FastGenNetwork interface.
    Each forward call constructs the y conditioning tensor based on mode (I2V or V2V)
    from the condition dict.

    Args:
        in_dim: Total patch_embedding input channels (I2V=33, V2V=49).
        dim: Hidden dimension (5120 for 14B, 1536 for 1.3B).
        num_heads: Number of attention heads.
        ffn_dim: FFN intermediate dim.
        num_layers: Number of DiT blocks.
        mode: "i2v" or "v2v" — determines y tensor construction.
        use_audio: Whether to use audio conditioning.
        audio_hidden_size: AudioPack output dim (32).
        has_image_input: Whether CrossAttention has CLIP image branch.
        base_model_paths: Path(s) to base Wan 2.1 safetensors.
        omniavatar_ckpt_path: Path to OmniAvatar checkpoint (.pt with LoRA + audio weights).
        merge_lora: Whether to merge LoRA weights into base (for full fine-tuning).
        load_pretrained: Whether to load pretrained weights during init.
        out_dim: Output channels (16).
        text_dim: Text embedding dim (4096).
        freq_dim: Sinusoidal embedding dim (256).
        eps: LayerNorm epsilon.
        patch_size: 3D patch size.
        net_pred_type: Prediction type ("flow").
        schedule_type: Noise schedule type ("rf").
    """

    def __init__(
        self,
        in_dim: int = 33,
        dim: int = 5120,
        num_heads: int = 40,
        ffn_dim: int = 13824,
        num_layers: int = 40,
        mode: str = "i2v",
        use_audio: bool = True,
        audio_hidden_size: int = 32,
        has_image_input: bool = False,
        base_model_paths: Union[str, List[str]] = "",
        omniavatar_ckpt_path: str = "",
        merge_lora: bool = True,
        load_pretrained: bool = True,
        out_dim: int = 16,
        text_dim: int = 4096,
        freq_dim: int = 256,
        eps: float = 1e-6,
        patch_size: Tuple[int, int, int] = (1, 2, 2),
        net_pred_type: str = "flow",
        schedule_type: str = "rf",
        **kwargs,
    ):
        super().__init__(net_pred_type=net_pred_type, schedule_type=schedule_type)
        self.mode = mode
        self.in_dim = in_dim

        # Build the underlying WanModel
        self.model = WanModel(
            dim=dim,
            in_dim=in_dim,
            ffn_dim=ffn_dim,
            out_dim=out_dim,
            text_dim=text_dim,
            freq_dim=freq_dim,
            eps=eps,
            patch_size=patch_size,
            num_heads=num_heads,
            num_layers=num_layers,
            has_image_input=has_image_input,
            use_audio=use_audio,
            audio_hidden_size=audio_hidden_size,
        )

        # Xavier-init all parameters (OmniAvatar convention)
        self._xavier_init()

        # Load pretrained weights
        if load_pretrained and not self._is_in_meta_context():
            self._load_weights(base_model_paths, omniavatar_ckpt_path, merge_lora)

    def _xavier_init(self) -> None:
        """Xavier uniform init for all weight matrices, zeros for biases (OmniAvatar convention)."""

        def _init(m: nn.Module) -> None:
            if isinstance(m, nn.Linear):
                init.xavier_uniform_(m.weight, gain=0.05)
                if m.bias is not None:
                    init.zeros_(m.bias)
            elif isinstance(m, nn.Conv3d):
                init.xavier_uniform_(m.weight, gain=0.05)
                if m.bias is not None:
                    init.zeros_(m.bias)

        self.model.apply(_init)

    def _load_weights(
        self,
        base_model_paths: Union[str, List[str]],
        omniavatar_ckpt_path: str,
        merge_lora: bool,
    ) -> None:
        """Load base Wan 2.1 weights + OmniAvatar checkpoint."""
        # Step 1: Load base Wan 2.1 weights (civitai format — keys match directly)
        if base_model_paths:
            if isinstance(base_model_paths, str):
                paths = [p.strip() for p in base_model_paths.split(",") if p.strip()]
            else:
                paths = base_model_paths
            if paths and paths[0]:
                logger.info(f"Loading base Wan 2.1 weights from {len(paths)} file(s)")
                base_sd = _load_safetensors(paths)
                _smart_load_state_dict(self.model, base_sd)
                del base_sd
                logger.success("Base weights loaded")

        # Step 2: Load OmniAvatar checkpoint (LoRA + audio + patch_embedding)
        if omniavatar_ckpt_path and os.path.exists(omniavatar_ckpt_path):
            logger.info(f"Loading OmniAvatar checkpoint from {omniavatar_ckpt_path}")
            ckpt_sd = torch.load(omniavatar_ckpt_path, map_location="cpu", weights_only=True)

            # Map LoRA key format: lora_A.weight -> lora_A.default.weight
            mapped_sd = {}
            for k, v in ckpt_sd.items():
                new_key = k.replace("lora_A.weight", "lora_A.default.weight").replace(
                    "lora_B.weight", "lora_B.default.weight"
                )
                mapped_sd[new_key] = v

            if merge_lora:
                # Merge LoRA into base weights
                _merge_lora_into_base(self.model, mapped_sd)

                # Load non-LoRA weights (audio_proj, audio_cond_projs, patch_embedding)
                non_lora_sd = {k: v for k, v in mapped_sd.items() if "lora_" not in k}
                _smart_load_state_dict(self.model, non_lora_sd)
            else:
                # Load everything as-is (model must have LoRA modules)
                _smart_load_state_dict(self.model, mapped_sd)

            del ckpt_sd, mapped_sd
            logger.success("OmniAvatar checkpoint loaded")

    def _build_y(
        self,
        condition: Dict[str, Any],
        num_frames: int,
    ) -> torch.Tensor:
        """Build the y conditioning tensor based on mode.

        Args:
            condition: Dict with ref_latent, mask, masked_video, etc.
            num_frames: Number of latent temporal frames.

        Returns:
            y: [B, y_channels, T, H, W]
        """
        ref_latent = condition["ref_latent"]  # [B, 16, 1, H, W]
        mask = condition["mask"]  # [H, W]
        B = ref_latent.shape[0]
        H, W = ref_latent.shape[-2], ref_latent.shape[-1]

        # Reference frame repeated across time
        ref_repeated = ref_latent.repeat(1, 1, num_frames, 1, 1)  # [B, 16, T, H, W]

        # Mask channel: 0 for frame 0 (keep), inverted mask for frames 1+
        mask_ch = torch.zeros(B, 1, num_frames, H, W, device=ref_latent.device, dtype=ref_latent.dtype)
        inverted_mask = 1.0 - mask.to(ref_latent.device, ref_latent.dtype)  # 1=generate, 0=keep
        mask_ch[:, :, 1:] = inverted_mask.unsqueeze(0).unsqueeze(0).unsqueeze(0)

        if self.mode == "i2v":
            # I2V: y = [ref(16) + mask(1)] = 17 channels
            y = torch.cat([ref_repeated, mask_ch], dim=1)
        elif self.mode == "v2v":
            # V2V: y = [ref(16) + mask(1) + masked_video(16)] = 33 channels
            masked_video = condition["masked_video"]  # [B, 16, T, H, W]
            y = torch.cat([ref_repeated, mask_ch, masked_video], dim=1)
        else:
            raise ValueError(f"Unknown mode: {self.mode}")

        return y

    def forward(
        self,
        x_t: torch.Tensor,
        t: torch.Tensor,
        condition: Any = None,
        r: Optional[torch.Tensor] = None,
        return_features_early: bool = False,
        feature_indices: Optional[Set[int]] = None,
        return_logvar: bool = False,
        fwd_pred_type: Optional[str] = None,
        **fwd_kwargs,
    ) -> Union[torch.Tensor, List[torch.Tensor], Tuple[torch.Tensor, torch.Tensor]]:
        """Forward pass through OmniAvatar WanModel.

        Args:
            x_t: Noisy latents [B, 16, T, H, W].
            t: Timesteps [B].
            condition: Dict with keys: text_embeds, audio_emb, ref_latent, mask,
                       and optionally masked_video (for V2V mode).
            fwd_pred_type: Override prediction type (e.g., "x0").
        """
        # Unpack condition
        text_embeds = condition["text_embeds"]  # [B, L, 4096]
        audio_emb = condition.get("audio_emb")  # [B, T_video, 10752] or None

        # Build y tensor
        num_frames = x_t.shape[2]
        y = self._build_y(condition, num_frames)

        # Forward through WanModel
        raw_output = self.model(
            x=x_t,
            timestep=t,
            context=text_embeds,
            y=y,
            audio_emb=audio_emb,
        )

        # Convert prediction type (model outputs flow, caller may want x0)
        target_type = fwd_pred_type or self.net_pred_type
        if target_type != "flow":
            raw_output = self.noise_scheduler.convert_model_output(
                x_t, raw_output, t, src_pred_type="flow", target_pred_type=target_type
            )

        return raw_output

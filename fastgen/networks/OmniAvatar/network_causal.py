# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Causal OmniAvatar network wrapper for FastGen.

Used for the student (V2V 1.3B causal) — generates video chunk-by-chunk
with KV cache during Self-Forcing rollout.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Set, Tuple, Union

import torch
import torch.nn as nn

from fastgen.networks.network import CausalFastGenNetwork
from .wan_model import WanModel, build_rope_freqs
from .network import OmniAvatarWan, _smart_load_state_dict, _load_safetensors, _merge_lora_into_base

import fastgen.utils.logging_utils as logger


class CausalOmniAvatarWan(CausalFastGenNetwork):
    """Causal OmniAvatar Wan network for FastGen (student).

    Extends CausalFastGenNetwork with OmniAvatar's WanModel and KV cache.
    Generates video in chunks with autoregressive KV caching.

    Args:
        in_dim: Total patch_embedding input channels (V2V=49).
        dim: Hidden dimension (1536 for 1.3B).
        num_heads: Number of attention heads (12 for 1.3B).
        ffn_dim: FFN intermediate dim (8960 for 1.3B).
        num_layers: Number of DiT blocks (30 for 1.3B).
        chunk_size: Latent frames per autoregressive chunk.
        total_num_frames: Total latent frames in full video.
        use_audio: Whether to use audio conditioning.
        audio_hidden_size: AudioPack output dim.
        has_image_input: Whether CrossAttention has CLIP image branch.
        base_model_paths: Path(s) to base Wan 2.1 safetensors.
        omniavatar_ckpt_path: Path to OmniAvatar checkpoint.
        merge_lora: Whether to merge LoRA into base weights.
        load_pretrained: Whether to load pretrained weights.
        out_dim: Output channels.
        text_dim: Text embedding dim.
        freq_dim: Sinusoidal embedding dim.
        eps: LayerNorm epsilon.
        patch_size: 3D patch size.
        net_pred_type: Prediction type.
        schedule_type: Noise schedule type.
    """

    def __init__(
        self,
        in_dim: int = 49,
        dim: int = 1536,
        num_heads: int = 12,
        ffn_dim: int = 8960,
        num_layers: int = 30,
        chunk_size: int = 3,
        total_num_frames: int = 21,
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
        super().__init__(
            net_pred_type=net_pred_type,
            schedule_type=schedule_type,
            chunk_size=chunk_size,
            total_num_frames=total_num_frames,
        )
        self.mode = "v2v"
        self.in_dim = in_dim
        self._dim = dim
        self._num_layers = num_layers
        self._num_heads = num_heads

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

        # Xavier-init
        OmniAvatarWan._xavier_init(self)

        # KV caches (lazily initialized on first forward)
        self._self_kv_caches: List[Dict] = []
        self._cross_kv_caches: List[Dict] = []
        self._caches_initialized = False

        # Pre-computed audio (cached per-sequence to avoid recomputation)
        self._cached_audio: Optional[torch.Tensor] = None

        # Load pretrained weights
        if load_pretrained and not self._is_in_meta_context():
            self._load_weights(base_model_paths, omniavatar_ckpt_path, merge_lora)

    def _load_weights(
        self,
        base_model_paths: Union[str, List[str]],
        omniavatar_ckpt_path: str,
        merge_lora: bool,
    ) -> None:
        """Load base Wan 2.1 weights + OmniAvatar checkpoint.

        Reuses OmniAvatarWan's loading logic.
        """
        # Use the same loading logic as the non-causal wrapper
        OmniAvatarWan._load_weights(self, base_model_paths, omniavatar_ckpt_path, merge_lora)

    def _init_caches(self, batch_size: int, device: torch.device, dtype: torch.dtype) -> None:
        """Initialize per-block KV caches."""
        # Compute max total tokens: total_num_frames * (H_lat/2) * (W_lat/2)
        # We don't know H, W at init time, so we allocate lazily on first forward
        pass  # Handled in _ensure_caches

    def _ensure_caches(
        self,
        batch_size: int,
        total_tokens: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> None:
        """Ensure KV caches are allocated and properly sized."""
        if self._caches_initialized and len(self._self_kv_caches) == self._num_layers:
            return

        dim = self._dim
        self._self_kv_caches = []
        self._cross_kv_caches = []

        for _ in range(self._num_layers):
            self._self_kv_caches.append(
                {
                    "k": torch.zeros(batch_size, total_tokens, dim, device=device, dtype=dtype),
                    "v": torch.zeros(batch_size, total_tokens, dim, device=device, dtype=dtype),
                    "len": 0,
                }
            )
            self._cross_kv_caches.append(
                {
                    "k": None,
                    "v": None,
                    "is_init": False,
                }
            )
        self._caches_initialized = True

    def clear_caches(self) -> None:
        """Clear all KV caches. Called before and after Self-Forcing rollout."""
        for cache in self._self_kv_caches:
            cache["len"] = 0
            if cache["k"] is not None:
                cache["k"].zero_()
                cache["v"].zero_()
        for cache in self._cross_kv_caches:
            cache["is_init"] = False
            cache["k"] = None
            cache["v"] = None
        self._cached_audio = None
        self._caches_initialized = False

    def _build_v2v_y(
        self,
        condition: Dict[str, Any],
        cur_start_frame: int,
        chunk_frames: int,
    ) -> torch.Tensor:
        """Build V2V y tensor for the current chunk.

        Args:
            condition: Full condition dict.
            cur_start_frame: Start frame index in latent space.
            chunk_frames: Number of latent frames in this chunk.

        Returns:
            y: [B, 33, chunk_frames, H, W]
        """
        ref_latent = condition["ref_latent"]  # [B, 16, 1, H, W]
        mask = condition["mask"]  # [H, W]
        masked_video = condition["masked_video"]  # [B, 16, T_total, H, W]

        B = ref_latent.shape[0]
        H, W = ref_latent.shape[-2], ref_latent.shape[-1]

        # Reference repeated for chunk
        ref_repeated = ref_latent.repeat(1, 1, chunk_frames, 1, 1)

        # Mask channel for chunk
        mask_ch = torch.zeros(B, 1, chunk_frames, H, W, device=ref_latent.device, dtype=ref_latent.dtype)
        inverted_mask = 1.0 - mask.to(ref_latent.device, ref_latent.dtype)
        # Frame 0 of the entire video gets mask=0 (keep all); for chunk processing,
        # only the global first frame (cur_start_frame==0, frame index 0) should be 0
        if cur_start_frame == 0:
            # First chunk: frame 0 is the reference frame
            mask_ch[:, :, 1:] = inverted_mask.unsqueeze(0).unsqueeze(0).unsqueeze(0)
        else:
            # Later chunks: all frames get the mask
            mask_ch[:, :, :] = inverted_mask.unsqueeze(0).unsqueeze(0).unsqueeze(0)

        # Slice masked video for this chunk
        masked_video_chunk = masked_video[:, :, cur_start_frame : cur_start_frame + chunk_frames]

        return torch.cat([ref_repeated, mask_ch, masked_video_chunk], dim=1)

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
        # Causal-specific kwargs (passed by SelfForcingModel.rollout_with_gradient)
        cache_tag: str = "pos",
        cur_start_frame: int = 0,
        store_kv: bool = False,
        is_ar: bool = False,
        **fwd_kwargs,
    ) -> Union[torch.Tensor, List[torch.Tensor], Tuple[torch.Tensor, torch.Tensor]]:
        """Causal forward pass with KV cache.

        Args:
            x_t: Noisy latents for current chunk [B, 16, chunk_frames, H, W].
            t: Timesteps [B].
            condition: Dict with text_embeds, audio_emb, ref_latent, mask, masked_video.
            cache_tag: KV cache tag (unused, kept for interface compatibility).
            cur_start_frame: Start frame index in latent space for this chunk.
            store_kv: Whether to write K/V into cache after this forward pass.
            is_ar: Whether in autoregressive mode.
            fwd_pred_type: Override prediction type.
        """
        B = x_t.shape[0]
        chunk_frames = x_t.shape[2]
        H_lat, W_lat = x_t.shape[-2], x_t.shape[-1]

        # Ensure caches are allocated
        # Total tokens per frame after patchify: (H_lat / patch_h) * (W_lat / patch_w)
        h = H_lat // self.model.patch_size[1]
        w = W_lat // self.model.patch_size[2]
        total_tokens = self.total_num_frames * h * w
        self._ensure_caches(B, total_tokens, x_t.device, x_t.dtype)

        # Unpack condition
        text_embeds = condition["text_embeds"]  # [B, L, 4096]
        audio_emb = condition.get("audio_emb")  # [B, T_video, 10752] or None

        # Build V2V y tensor for this chunk
        y_chunk = self._build_v2v_y(condition, cur_start_frame, chunk_frames)

        # Compute RoPE frequencies for this chunk's temporal positions
        freqs = build_rope_freqs(
            self.model.freqs, chunk_frames, h, w, x_t.device, f_offset=cur_start_frame
        )

        # Pre-compute audio for the full sequence (cached)
        audio_processed = None
        if audio_emb is not None and self.model.use_audio:
            if self._cached_audio is None:
                self._cached_audio = self.model._prepare_audio(audio_emb, B)
            audio_processed = self._cached_audio

        # Forward through WanModel causal path
        raw_output = self.model.forward_causal(
            x=x_t,
            timestep=t,
            context=text_embeds,
            y=y_chunk,
            freqs=freqs,
            self_kv_caches=self._self_kv_caches,
            cross_kv_caches=self._cross_kv_caches,
            store_kv=store_kv,
            audio_processed=audio_processed,
            audio_chunk_start=cur_start_frame,
        )

        # Convert prediction type
        target_type = fwd_pred_type or self.net_pred_type
        if target_type != "flow":
            raw_output = self.noise_scheduler.convert_model_output(
                x_t, raw_output, t, src_pred_type="flow", target_pred_type=target_type
            )

        return raw_output

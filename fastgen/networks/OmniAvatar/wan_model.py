# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
OmniAvatar WanModel adapted for FastGen.

Adapted from OmniAvatar/OmniAvatar/models/wan_video_dit.py.
Changes from original:
  - Removed global `args` singleton dependency; all config is via constructor params.
  - Removed sequence parallel (sp_size) code.
  - Removed TeaCache code.
  - Added KV cache support to SelfAttention and CrossAttention for causal generation.
  - Added `forward_causal()` method for chunk-by-chunk generation with KV cache.
"""

import math
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

from .audio_pack import AudioPack

# --------------------------------------------------------------------------- #
# Attention backends
# --------------------------------------------------------------------------- #
try:
    import flash_attn_interface

    FLASH_ATTN_3_AVAILABLE = True
except ModuleNotFoundError:
    FLASH_ATTN_3_AVAILABLE = False

try:
    import flash_attn

    FLASH_ATTN_2_AVAILABLE = True
except ModuleNotFoundError:
    FLASH_ATTN_2_AVAILABLE = False


def flash_attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, num_heads: int) -> torch.Tensor:
    """Multi-head attention with flash_attn_3 > flash_attn_2 > SDPA fallback."""
    if FLASH_ATTN_3_AVAILABLE:
        q = rearrange(q, "b s (n d) -> b s n d", n=num_heads)
        k = rearrange(k, "b s (n d) -> b s n d", n=num_heads)
        v = rearrange(v, "b s (n d) -> b s n d", n=num_heads)
        x = flash_attn_interface.flash_attn_func(q, k, v)
        return rearrange(x, "b s n d -> b s (n d)")
    elif FLASH_ATTN_2_AVAILABLE:
        q = rearrange(q, "b s (n d) -> b s n d", n=num_heads)
        k = rearrange(k, "b s (n d) -> b s n d", n=num_heads)
        v = rearrange(v, "b s (n d) -> b s n d", n=num_heads)
        x = flash_attn.flash_attn_func(q, k, v)
        return rearrange(x, "b s n d -> b s (n d)")
    else:
        q = rearrange(q, "b s (n d) -> b n s d", n=num_heads)
        k = rearrange(k, "b s (n d) -> b n s d", n=num_heads)
        v = rearrange(v, "b s (n d) -> b n s d", n=num_heads)
        x = F.scaled_dot_product_attention(q, k, v)
        return rearrange(x, "b n s d -> b s (n d)")


# --------------------------------------------------------------------------- #
# Utilities
# --------------------------------------------------------------------------- #
def modulate(x: torch.Tensor, shift: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    return x * (1 + scale) + shift


def sinusoidal_embedding_1d(dim: int, position: torch.Tensor) -> torch.Tensor:
    sinusoid = torch.outer(
        position.to(torch.float64),
        torch.pow(10000, -torch.arange(dim // 2, dtype=torch.float64, device=position.device).div(dim // 2)),
    )
    x = torch.cat([torch.cos(sinusoid), torch.sin(sinusoid)], dim=1)
    return x.to(position.dtype)


def precompute_freqs_cis(dim: int, end: int = 1024, theta: float = 10000.0) -> torch.Tensor:
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].double() / dim))
    freqs = torch.outer(torch.arange(end, device=freqs.device), freqs)
    return torch.polar(torch.ones_like(freqs), freqs)  # complex64


def precompute_freqs_cis_3d(
    dim: int, end: int = 1024, theta: float = 10000.0
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    f_freqs = precompute_freqs_cis(dim - 2 * (dim // 3), end, theta)
    h_freqs = precompute_freqs_cis(dim // 3, end, theta)
    w_freqs = precompute_freqs_cis(dim // 3, end, theta)
    return f_freqs, h_freqs, w_freqs


def rope_apply(x: torch.Tensor, freqs: torch.Tensor, num_heads: int) -> torch.Tensor:
    x = rearrange(x, "b s (n d) -> b s n d", n=num_heads)
    x_out = torch.view_as_complex(x.to(torch.float64).reshape(x.shape[0], x.shape[1], x.shape[2], -1, 2))
    x_out = torch.view_as_real(x_out * freqs).flatten(2)
    return x_out.to(x.dtype)


def build_rope_freqs(
    freqs_3d: Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    f: int,
    h: int,
    w: int,
    device: torch.device,
    f_offset: int = 0,
) -> torch.Tensor:
    """Build 3D RoPE frequencies for a given grid, with optional temporal offset."""
    freqs = torch.cat(
        [
            freqs_3d[0][f_offset : f_offset + f].view(f, 1, 1, -1).expand(f, h, w, -1),
            freqs_3d[1][:h].view(1, h, 1, -1).expand(f, h, w, -1),
            freqs_3d[2][:w].view(1, 1, w, -1).expand(f, h, w, -1),
        ],
        dim=-1,
    ).reshape(f * h * w, 1, -1)
    return freqs.to(device)


# --------------------------------------------------------------------------- #
# Norms
# --------------------------------------------------------------------------- #
class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dtype = x.dtype
        normed = x.float() * torch.rsqrt(x.float().pow(2).mean(dim=-1, keepdim=True) + self.eps)
        return normed.to(dtype) * self.weight


# --------------------------------------------------------------------------- #
# Attention modules (with KV cache support)
# --------------------------------------------------------------------------- #
class SelfAttention(nn.Module):
    def __init__(self, dim: int, num_heads: int, eps: float = 1e-6):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.q = nn.Linear(dim, dim)
        self.k = nn.Linear(dim, dim)
        self.v = nn.Linear(dim, dim)
        self.o = nn.Linear(dim, dim)
        self.norm_q = RMSNorm(dim, eps=eps)
        self.norm_k = RMSNorm(dim, eps=eps)

    def forward(
        self,
        x: torch.Tensor,
        freqs: torch.Tensor,
        kv_cache: Optional[Dict] = None,
        store_kv: bool = False,
    ) -> torch.Tensor:
        """
        Args:
            x: [B, N, D]
            freqs: [N, 1, head_dim_complex] RoPE frequencies for current positions
            kv_cache: Optional dict with keys "k", "v", "len" for KV caching
            store_kv: If True and kv_cache is not None, write K/V into cache
        """
        q = self.norm_q(self.q(x))
        k = self.norm_k(self.k(x))
        v = self.v(x)

        # Apply RoPE to current Q and K
        q = rope_apply(q, freqs, self.num_heads)
        k = rope_apply(k, freqs, self.num_heads)

        # Prepend cached K/V if available
        if kv_cache is not None and kv_cache["len"] > 0:
            cached_len = kv_cache["len"]
            k = torch.cat([kv_cache["k"][:, :cached_len], k], dim=1)
            v = torch.cat([kv_cache["v"][:, :cached_len], v], dim=1)

        out = flash_attention(q, k, v, self.num_heads)

        # Store K/V into cache
        if store_kv and kv_cache is not None:
            new_len = k.shape[1]
            kv_cache["k"][:, :new_len] = k
            kv_cache["v"][:, :new_len] = v
            kv_cache["len"] = new_len

        return self.o(out)


class CrossAttention(nn.Module):
    def __init__(self, dim: int, num_heads: int, eps: float = 1e-6, has_image_input: bool = False):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.q = nn.Linear(dim, dim)
        self.k = nn.Linear(dim, dim)
        self.v = nn.Linear(dim, dim)
        self.o = nn.Linear(dim, dim)
        self.norm_q = RMSNorm(dim, eps=eps)
        self.norm_k = RMSNorm(dim, eps=eps)
        self.has_image_input = has_image_input
        if has_image_input:
            self.k_img = nn.Linear(dim, dim)
            self.v_img = nn.Linear(dim, dim)
            self.norm_k_img = RMSNorm(dim, eps=eps)

    def forward(
        self,
        x: torch.Tensor,
        context: torch.Tensor,
        cross_cache: Optional[Dict] = None,
        store_kv: bool = False,
    ) -> torch.Tensor:
        """
        Args:
            x: [B, N, D] query tokens
            context: [B, L, D] text context (optionally with CLIP image prefix)
            cross_cache: Optional dict with "k", "v", "is_init" for caching text K/V
            store_kv: If True and cross_cache not None, write K/V into cache
        """
        if self.has_image_input:
            img = context[:, :257]
            ctx = context[:, 257:]
        else:
            ctx = context

        q = self.norm_q(self.q(x))

        # Use cached K/V if available
        if cross_cache is not None and cross_cache["is_init"]:
            k = cross_cache["k"]
            v = cross_cache["v"]
        else:
            k = self.norm_k(self.k(ctx))
            v = self.v(ctx)
            if store_kv and cross_cache is not None:
                cross_cache["k"] = k
                cross_cache["v"] = v
                cross_cache["is_init"] = True

        out = flash_attention(q, k, v, self.num_heads)

        if self.has_image_input:
            k_img = self.norm_k_img(self.k_img(img))
            v_img = self.v_img(img)
            out = out + flash_attention(q, k_img, v_img, self.num_heads)

        return self.o(out)


# --------------------------------------------------------------------------- #
# DiT Block
# --------------------------------------------------------------------------- #
class DiTBlock(nn.Module):
    def __init__(self, has_image_input: bool, dim: int, num_heads: int, ffn_dim: int, eps: float = 1e-6):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.ffn_dim = ffn_dim
        self.self_attn = SelfAttention(dim, num_heads, eps)
        self.cross_attn = CrossAttention(dim, num_heads, eps, has_image_input=has_image_input)
        self.norm1 = nn.LayerNorm(dim, eps=eps, elementwise_affine=False)
        self.norm2 = nn.LayerNorm(dim, eps=eps, elementwise_affine=False)
        self.norm3 = nn.LayerNorm(dim, eps=eps)
        self.ffn = nn.Sequential(nn.Linear(dim, ffn_dim), nn.GELU(approximate="tanh"), nn.Linear(ffn_dim, dim))
        self.modulation = nn.Parameter(torch.randn(1, 6, dim) / dim**0.5)

    def forward(
        self,
        x: torch.Tensor,
        context: torch.Tensor,
        t_mod: torch.Tensor,
        freqs: torch.Tensor,
        self_kv_cache: Optional[Dict] = None,
        cross_kv_cache: Optional[Dict] = None,
        store_kv: bool = False,
    ) -> torch.Tensor:
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (
            self.modulation.to(dtype=t_mod.dtype, device=t_mod.device) + t_mod
        ).chunk(6, dim=1)

        x = x + gate_msa * self.self_attn(
            modulate(self.norm1(x), shift_msa, scale_msa),
            freqs,
            kv_cache=self_kv_cache,
            store_kv=store_kv,
        )
        x = x + self.cross_attn(
            self.norm3(x),
            context,
            cross_cache=cross_kv_cache,
            store_kv=store_kv,
        )
        x = x + gate_mlp * self.ffn(modulate(self.norm2(x), shift_mlp, scale_mlp))
        return x


# --------------------------------------------------------------------------- #
# Head
# --------------------------------------------------------------------------- #
class Head(nn.Module):
    def __init__(self, dim: int, out_dim: int, patch_size: Tuple[int, int, int], eps: float):
        super().__init__()
        self.dim = dim
        self.patch_size = patch_size
        self.norm = nn.LayerNorm(dim, eps=eps, elementwise_affine=False)
        self.head = nn.Linear(dim, out_dim * math.prod(patch_size))
        self.modulation = nn.Parameter(torch.randn(1, 2, dim) / dim**0.5)

    def forward(self, x: torch.Tensor, t_mod: torch.Tensor) -> torch.Tensor:
        shift, scale = (self.modulation.to(dtype=t_mod.dtype, device=t_mod.device) + t_mod).chunk(2, dim=1)
        return self.head(self.norm(x) * (1 + scale) + shift)


# --------------------------------------------------------------------------- #
# WanModel
# --------------------------------------------------------------------------- #
class WanModel(nn.Module):
    """OmniAvatar Wan DiT model adapted for FastGen.

    This is OmniAvatar's custom WanModel (NOT diffusers' WanTransformer3DModel).
    It supports audio conditioning via AudioPack + per-layer audio_cond_projs.

    Args:
        dim: Hidden dimension (5120 for 14B, 1536 for 1.3B).
        in_dim: Total input channels to patch_embedding (x + y channels).
            I2V: 33 (noise=16 + ref=16 + mask=1). V2V: 49 (+ masked_vid=16).
        ffn_dim: FFN intermediate dimension.
        out_dim: Output latent channels (16).
        text_dim: Text embedding dimension (4096 for T5).
        freq_dim: Sinusoidal timestep embedding dimension (256).
        eps: LayerNorm epsilon.
        patch_size: 3D patch size for Conv3d, typically (1, 2, 2).
        num_heads: Number of attention heads.
        num_layers: Number of DiT blocks (40 for 14B, 30 for 1.3B).
        has_image_input: Whether CrossAttention has CLIP image branch.
        use_audio: Whether to use audio conditioning.
        audio_hidden_size: AudioPack output dimension (32).
    """

    def __init__(
        self,
        dim: int,
        in_dim: int,
        ffn_dim: int,
        out_dim: int = 16,
        text_dim: int = 4096,
        freq_dim: int = 256,
        eps: float = 1e-6,
        patch_size: Tuple[int, int, int] = (1, 2, 2),
        num_heads: int = 40,
        num_layers: int = 40,
        has_image_input: bool = False,
        use_audio: bool = True,
        audio_hidden_size: int = 32,
    ):
        super().__init__()
        self.dim = dim
        self.in_dim = in_dim
        self.freq_dim = freq_dim
        self.has_image_input = has_image_input
        self.patch_size = patch_size
        self.num_layers = num_layers
        self.use_audio = use_audio

        self.patch_embedding = nn.Conv3d(in_dim, dim, kernel_size=patch_size, stride=patch_size)
        self.text_embedding = nn.Sequential(nn.Linear(text_dim, dim), nn.GELU(approximate="tanh"), nn.Linear(dim, dim))
        self.time_embedding = nn.Sequential(nn.Linear(freq_dim, dim), nn.SiLU(), nn.Linear(dim, dim))
        self.time_projection = nn.Sequential(nn.SiLU(), nn.Linear(dim, dim * 6))

        self.blocks = nn.ModuleList([DiTBlock(has_image_input, dim, num_heads, ffn_dim, eps) for _ in range(num_layers)])
        self.head = Head(dim, out_dim, patch_size, eps)

        head_dim = dim // num_heads
        self.freqs = precompute_freqs_cis_3d(head_dim)

        if has_image_input:
            self.img_emb = nn.Sequential(
                nn.LayerNorm(1280), nn.Linear(1280, 1280), nn.GELU(), nn.Linear(1280, dim), nn.LayerNorm(dim)
            )

        # Audio conditioning
        if use_audio:
            audio_input_dim = 10752
            self.audio_proj = AudioPack(audio_input_dim, [4, 1, 1], audio_hidden_size, layernorm=True)
            self.audio_cond_projs = nn.ModuleList(
                [nn.Linear(audio_hidden_size, dim) for _ in range(num_layers // 2 - 1)]
            )

    def patchify(self, x: torch.Tensor) -> Tuple[torch.Tensor, Tuple[int, int, int]]:
        grid_size = x.shape[2:]
        x = rearrange(x, "b c f h w -> b (f h w) c").contiguous()
        return x, grid_size

    def unpatchify(self, x: torch.Tensor, grid_size: Tuple[int, int, int]) -> torch.Tensor:
        return rearrange(
            x,
            "b (f h w) (x y z c) -> b c (f x) (h y) (w z)",
            f=grid_size[0],
            h=grid_size[1],
            w=grid_size[2],
            x=self.patch_size[0],
            y=self.patch_size[1],
            z=self.patch_size[2],
        )

    def _prepare_audio(self, audio_emb: torch.Tensor, batch_size: int) -> torch.Tensor:
        """Pre-process audio embeddings through AudioPack and all per-layer projections.

        Args:
            audio_emb: [B, T_video, 10752]
            batch_size: B

        Returns:
            [B, num_audio_layers, T_groups, 1, 1, dim]
        """
        # [B, T_video, 10752] -> [B, 10752, T_video, 1, 1]
        audio = audio_emb.permute(0, 2, 1)[:, :, :, None, None]
        # Prepend 3 duplicate frames for alignment: [B, 10752, T_video+3, 1, 1]
        audio = torch.cat([audio[:, :, :1].repeat(1, 1, 3, 1, 1), audio], dim=2)
        # AudioPack: [B, T_groups, 1, 1, audio_hidden_size]
        audio = self.audio_proj(audio)
        # Apply all per-layer projections and stack: [num_layers, B, T_groups, 1, 1, dim]
        audio = torch.cat([proj(audio) for proj in self.audio_cond_projs], dim=0)
        # Reshape to [B, num_audio_layers, T_groups, 1, 1, dim]
        audio = audio.reshape(batch_size, len(self.audio_cond_projs), -1, *audio.shape[2:])
        return audio

    def forward(
        self,
        x: torch.Tensor,
        timestep: torch.Tensor,
        context: torch.Tensor,
        y: torch.Tensor,
        audio_emb: Optional[torch.Tensor] = None,
        clip_feature: Optional[torch.Tensor] = None,
        use_gradient_checkpointing: bool = False,
    ) -> torch.Tensor:
        """Standard (non-causal) forward pass.

        Args:
            x: Noisy latents [B, 16, T_lat, H_lat, W_lat].
            timestep: [B] scalar timesteps.
            context: Text embeddings [B, L, text_dim].
            y: Conditioning tensor (I2V: [B, 17, T, H, W], V2V: [B, 33, T, H, W]).
            audio_emb: Optional Wav2Vec2 features [B, T_video, 10752].
            clip_feature: Optional CLIP image features [B, 257, 1280].
            use_gradient_checkpointing: Enable gradient checkpointing for memory savings.

        Returns:
            Predicted output [B, 16, T_lat, H_lat, W_lat] (flow prediction).
        """
        B = x.shape[0]
        lat_h, lat_w = x.shape[-2], x.shape[-1]

        # Time embedding (sinusoidal_embedding_1d computes in float64 for precision;
        # cast to model dtype before passing through time_embedding Linear layers)
        t_emb = sinusoidal_embedding_1d(self.freq_dim, timestep).to(dtype=x.dtype)
        t = self.time_embedding(t_emb)
        t_mod = self.time_projection(t).unflatten(1, (6, self.dim))

        # Text embedding
        context = self.text_embedding(context)

        # CLIP image embedding (if applicable)
        if clip_feature is not None and self.has_image_input:
            context = torch.cat([self.img_emb(clip_feature), context], dim=1)

        # Audio processing
        audio_processed = None
        if audio_emb is not None and self.use_audio:
            audio_processed = self._prepare_audio(audio_emb, B)

        # Patchify: concat x and y, then apply patch embedding
        x = torch.cat([x, y], dim=1)
        x = self.patch_embedding(x)
        x, (f, h, w) = self.patchify(x)

        # RoPE frequencies
        freqs = build_rope_freqs(self.freqs, f, h, w, x.device)

        # Transformer blocks
        def create_custom_forward(module):
            def custom_forward(*inputs):
                return module(*inputs)

            return custom_forward

        for layer_i, block in enumerate(self.blocks):
            # Audio conditioning injection (blocks 2 through num_layers//2)
            if audio_processed is not None and 1 < layer_i <= self.num_layers // 2:
                au_idx = layer_i - 2
                audio_tokens = audio_processed[:, au_idx]  # [B, T_groups, 1, 1, dim]
                audio_tokens = audio_tokens.repeat(1, 1, lat_h // 2, lat_w // 2, 1)
                audio_tokens = self.patchify(audio_tokens.permute(0, 4, 1, 2, 3))[0]
                x = x + audio_tokens

            if self.training and use_gradient_checkpointing:
                x = torch.utils.checkpoint.checkpoint(create_custom_forward(block), x, context, t_mod, freqs, use_reentrant=False)
            else:
                x = block(x, context, t_mod, freqs)

        # Head + unpatchify
        x = self.head(x, t)
        x = self.unpatchify(x, (f, h, w))
        return x

    def forward_causal(
        self,
        x: torch.Tensor,
        timestep: torch.Tensor,
        context: torch.Tensor,
        y: torch.Tensor,
        freqs: torch.Tensor,
        self_kv_caches: List[Dict],
        cross_kv_caches: List[Dict],
        store_kv: bool = False,
        audio_emb: Optional[torch.Tensor] = None,
        audio_processed: Optional[torch.Tensor] = None,
        audio_chunk_start: int = 0,
        clip_feature: Optional[torch.Tensor] = None,
        use_gradient_checkpointing: bool = False,
    ) -> torch.Tensor:
        """Causal forward pass with KV cache for chunk-by-chunk generation.

        Args:
            x: Noisy latents for current chunk [B, 16, chunk_frames, H_lat, W_lat].
            timestep: [B] scalar timesteps.
            context: Text embeddings [B, L, text_dim].
            y: Conditioning for current chunk [B, y_ch, chunk_frames, H_lat, W_lat].
            freqs: Pre-computed RoPE frequencies for this chunk's positions [N_chunk, 1, D].
            self_kv_caches: Per-block self-attention KV caches.
            cross_kv_caches: Per-block cross-attention KV caches.
            store_kv: Whether to store K/V into caches.
            audio_emb: Full audio features [B, T_video, 10752] (used if audio_processed is None).
            audio_processed: Pre-processed audio [B, num_audio_layers, T_groups, 1, 1, dim].
            audio_chunk_start: Temporal start index for audio slicing.
            clip_feature: Optional CLIP features.
            use_gradient_checkpointing: Enable gradient checkpointing.

        Returns:
            Predicted output for this chunk [B, 16, chunk_frames, H_lat, W_lat].
        """
        B = x.shape[0]
        lat_h, lat_w = x.shape[-2], x.shape[-1]
        chunk_frames = x.shape[2]

        # Time embedding (cast sinusoidal float64 to model dtype)
        t_emb = sinusoidal_embedding_1d(self.freq_dim, timestep).to(dtype=x.dtype)
        t = self.time_embedding(t_emb)
        t_mod = self.time_projection(t).unflatten(1, (6, self.dim))

        # Text embedding
        context = self.text_embedding(context)

        # CLIP image embedding
        if clip_feature is not None and self.has_image_input:
            context = torch.cat([self.img_emb(clip_feature), context], dim=1)

        # Pre-process audio if not already done
        if audio_processed is None and audio_emb is not None and self.use_audio:
            audio_processed = self._prepare_audio(audio_emb, B)

        # Patchify
        x = torch.cat([x, y], dim=1)
        x = self.patch_embedding(x)
        x, (f, h, w) = self.patchify(x)

        # Transformer blocks with KV cache
        for layer_i, block in enumerate(self.blocks):
            # Audio conditioning injection
            if audio_processed is not None and 1 < layer_i <= self.num_layers // 2:
                au_idx = layer_i - 2
                # Slice audio to current chunk's temporal range
                audio_tokens = audio_processed[:, au_idx, audio_chunk_start : audio_chunk_start + chunk_frames]
                audio_tokens = audio_tokens.repeat(1, 1, lat_h // 2, lat_w // 2, 1)
                audio_tokens = self.patchify(audio_tokens.permute(0, 4, 1, 2, 3))[0]
                x = x + audio_tokens

            if self.training and use_gradient_checkpointing:
                # NOTE: gradient checkpointing with KV cache requires care;
                # we only use checkpointing when NOT storing KV
                if not store_kv:
                    x = torch.utils.checkpoint.checkpoint(
                        block,
                        x,
                        context,
                        t_mod,
                        freqs,
                        self_kv_caches[layer_i] if self_kv_caches else None,
                        cross_kv_caches[layer_i] if cross_kv_caches else None,
                        store_kv,
                        use_reentrant=False,
                    )
                else:
                    x = block(
                        x,
                        context,
                        t_mod,
                        freqs,
                        self_kv_cache=self_kv_caches[layer_i] if self_kv_caches else None,
                        cross_kv_cache=cross_kv_caches[layer_i] if cross_kv_caches else None,
                        store_kv=store_kv,
                    )
            else:
                x = block(
                    x,
                    context,
                    t_mod,
                    freqs,
                    self_kv_cache=self_kv_caches[layer_i] if self_kv_caches else None,
                    cross_kv_cache=cross_kv_caches[layer_i] if cross_kv_caches else None,
                    store_kv=store_kv,
                )

        # Head + unpatchify
        x = self.head(x, t)
        x = self.unpatchify(x, (f, h, w))
        return x

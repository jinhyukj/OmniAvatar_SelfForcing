# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Dataset and DataLoader for OmniAvatar precomputed training data.

Reads from OmniAvatar's per-directory format where each sample directory contains
precomputed .pt files for VAE latents, audio embeddings, and text embeddings.

Expected directory structure:
    /path/to/sample_dir/
        vae_latents.pt          -> {"input_latents": [16,T,H,W], "masked_latents": [16,T,H,W]}
        audio_emb_omniavatar.pt -> {"audio_emb": [T_video, 10752]}
        text_emb.pt             -> T5 text embeddings
"""

from __future__ import annotations

import os
from typing import Dict, List, Optional

import torch
from torch.utils.data import Dataset, DataLoader, DistributedSampler

import fastgen.utils.logging_utils as logger


class OmniAvatarDataset(Dataset):
    """PyTorch Dataset for OmniAvatar precomputed training data.

    Args:
        data_list_path: Path to text file with one video directory per line.
        latentsync_mask_path: Path to the LatentSync spatial mask image (PNG).
        num_frames: Number of video frames (81). Used for audio slicing.
        mask_all_frames: If True, mask frame 0 too (no clean reference frame in y).
        neg_text_emb_path: Path to precomputed negative text embedding.
            If None, uses zeros.
    """

    def __init__(
        self,
        data_list_path: str,
        latentsync_mask_path: str,
        num_frames: int = 81,
        mask_all_frames: bool = True,
        neg_text_emb_path: Optional[str] = None,
        latent_size: tuple = (64, 64),
    ):
        super().__init__()
        self.num_frames = num_frames
        self.mask_all_frames = mask_all_frames

        # Read video directory paths
        with open(data_list_path, "r") as f:
            self.video_dirs = [line.strip() for line in f if line.strip()]
        logger.info(f"OmniAvatarDataset: {len(self.video_dirs)} samples from {data_list_path}")

        # Load spatial mask (LatentSync format: 255=keep, 0=mask)
        self.latent_mask = self._load_latentsync_mask(latentsync_mask_path, latent_size=latent_size)

        # Load negative text embedding
        if neg_text_emb_path and os.path.exists(neg_text_emb_path):
            self.neg_text_emb = torch.load(neg_text_emb_path, map_location="cpu", weights_only=True)
        else:
            self.neg_text_emb = None

    @staticmethod
    def _load_latentsync_mask(mask_path: str, latent_size: tuple = (64, 64)) -> torch.Tensor:
        """Load LatentSync mask and resize to latent resolution.

        The mask image may not match the training latent resolution
        (e.g., 256x256 mask for 512x512 video with 64x64 latents).
        We resize directly to the target latent_size.

        Args:
            mask_path: Path to mask PNG (grayscale, 255=keep, 0=mask).
            latent_size: Target (H_lat, W_lat) to resize to.

        Returns:
            mask: [H_lat, W_lat] float tensor, 1=keep upper face, 0=generate mouth.
        """
        from PIL import Image
        import torchvision.transforms.functional as TF

        img = Image.open(mask_path).convert("L")
        mask = TF.to_tensor(img).squeeze(0)  # [H, W] in [0, 1]

        # Resize to target latent resolution
        mask = torch.nn.functional.interpolate(
            mask.unsqueeze(0).unsqueeze(0), size=latent_size, mode="nearest"
        ).squeeze(0).squeeze(0)

        # Binarize: > 0.5 = keep (1), <= 0.5 = generate (0)
        mask = (mask > 0.5).float()
        return mask

    def __len__(self) -> int:
        return len(self.video_dirs)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        video_dir = self.video_dirs[idx]

        # Load precomputed VAE latents
        vae_data = torch.load(
            os.path.join(video_dir, "vae_latents.pt"), map_location="cpu", weights_only=True
        )
        input_latents = vae_data["input_latents"]  # [16, T_lat, H_lat, W_lat]
        masked_latents = vae_data["masked_latents"]  # [16, T_lat, H_lat, W_lat]

        # Load precomputed audio embeddings
        audio_data = torch.load(
            os.path.join(video_dir, "audio_emb_omniavatar.pt"), map_location="cpu", weights_only=True
        )
        audio_emb = audio_data["audio_emb"]  # [T_video, 10752]
        # Slice to num_frames
        audio_emb = audio_emb[: self.num_frames]

        # Load precomputed text embedding
        text_emb_path = os.path.join(video_dir, "text_emb.pt")
        if os.path.exists(text_emb_path):
            text_emb = torch.load(text_emb_path, map_location="cpu", weights_only=True)
            # Handle various formats
            if isinstance(text_emb, dict):
                text_emb = text_emb.get("text_emb", text_emb.get("context", next(iter(text_emb.values()))))
            if text_emb.dim() == 3:
                text_emb = text_emb.squeeze(0)  # [L, D] or keep [1, L, D]
        else:
            # Fallback: zero embedding
            text_emb = torch.zeros(512, 4096)

        # Ensure text_emb is 2D [L, D]
        if text_emb.dim() == 3:
            text_emb = text_emb.squeeze(0)

        # Extract reference latent (first frame)
        ref_latent = input_latents[:, :1]  # [16, 1, H_lat, W_lat]

        # Negative text embedding
        if self.neg_text_emb is not None:
            neg_text_emb = self.neg_text_emb
            if isinstance(neg_text_emb, dict):
                neg_text_emb = next(iter(neg_text_emb.values()))
            if neg_text_emb.dim() == 3:
                neg_text_emb = neg_text_emb.squeeze(0)
        else:
            neg_text_emb = torch.zeros_like(text_emb)

        sample = {
            "real": input_latents,         # [16, T_lat, H_lat, W_lat]
            "condition": text_emb,         # [L, D]
            "neg_condition": neg_text_emb, # [L, D]
            "audio_emb": audio_emb,        # [T_video, 10752]
            "ref_latent": ref_latent,      # [16, 1, H_lat, W_lat]
            "masked_video": masked_latents,# [16, T_lat, H_lat, W_lat]
            "mask": self.latent_mask,      # [H_lat, W_lat]
        }

        # Load ODE path if available (for KD pretraining)
        path_file = os.path.join(video_dir, "path.pth")
        if os.path.exists(path_file):
            sample["path"] = torch.load(path_file, map_location="cpu", weights_only=True)

        return sample


class OmniAvatarDataLoader:
    """DataLoader wrapper for OmniAvatar data.

    Returns an iterable DataLoader when instantiated. Compatible with FastGen's
    `instantiate()` pattern.

    Args:
        datatags: List containing the data_list_path as first element.
            Following FastGen convention where datatags holds dataset path(s).
        latentsync_mask_path: Path to LatentSync mask.
        num_frames: Number of video frames.
        mask_all_frames: Whether to mask all frames.
        batch_size: Batch size per GPU.
        num_workers: DataLoader workers.
        neg_text_emb_path: Path to negative text embedding.
    """

    def __init__(
        self,
        datatags: List[str],
        latentsync_mask_path: str,
        num_frames: int = 81,
        mask_all_frames: bool = True,
        batch_size: int = 1,
        num_workers: int = 2,
        neg_text_emb_path: Optional[str] = None,
        **kwargs,
    ):
        data_list_path = datatags[0]

        dataset = OmniAvatarDataset(
            data_list_path=data_list_path,
            latentsync_mask_path=latentsync_mask_path,
            num_frames=num_frames,
            mask_all_frames=mask_all_frames,
            neg_text_emb_path=neg_text_emb_path,
        )

        # Use DistributedSampler if in distributed training
        sampler = None
        shuffle = True
        if torch.distributed.is_initialized():
            sampler = DistributedSampler(dataset, shuffle=True)
            shuffle = False

        self._dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            sampler=sampler,
            num_workers=num_workers,
            pin_memory=True,
            drop_last=True,
            persistent_workers=num_workers > 0,
        )

    def __iter__(self):
        return iter(self._dataloader)

    def __len__(self):
        return len(self._dataloader)

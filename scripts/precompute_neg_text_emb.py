# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Precompute negative prompt text embedding for OmniAvatar ODE pair generation.

Encodes OmniAvatar's standard negative prompt using the T5 text encoder and saves
the embedding as a .pt file. This only needs to be run once.

Usage:
    python scripts/precompute_neg_text_emb.py \
        --text_encoder_path pretrained_models/Wan2.1-T2V-1.3B/models_t5_umt5-xxl-enc-bf16.pth \
        --output_path neg_text_emb.pt
"""

from __future__ import annotations

import argparse

import torch

NEGATIVE_PROMPT = (
    "Vivid color tones, background/camera moving quickly, screen switching, "
    "subtitles and special effects, mutation, overexposed, static, blurred details, "
    "subtitles, style, work, painting, image, still, overall grayish, worst quality, "
    "low quality, JPEG compression residue, ugly, incomplete, extra fingers, "
    "poorly drawn hands, poorly drawn face, deformed, disfigured, malformed limbs, "
    "fingers merging, motionless image, chaotic background, three legs, crowded "
    "background with many people, walking backward"
)


def main():
    parser = argparse.ArgumentParser(description="Precompute negative prompt text embedding")
    parser.add_argument("--text_encoder_path", type=str, required=True,
                        help="Path to T5 text encoder weights (.pth)")
    parser.add_argument("--output_path", type=str, default="neg_text_emb.pt",
                        help="Output path for the embedding (default: neg_text_emb.pt)")
    parser.add_argument("--prompt", type=str, default=NEGATIVE_PROMPT,
                        help="Negative prompt text (default: OmniAvatar standard)")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load T5 text encoder (same as OmniAvatar's training pipeline)
    from wan.modules.t5 import T5EncoderModel

    print(f"Loading text encoder from {args.text_encoder_path}...")
    text_encoder = T5EncoderModel(
        text_len=512,
        dtype=torch.bfloat16,
        device=device,
        checkpoint_path=args.text_encoder_path,
    )
    text_encoder.eval()

    print(f"Encoding negative prompt: {args.prompt[:80]}...")
    with torch.no_grad():
        neg_emb = text_encoder([args.prompt], device=device)

    # neg_emb: [1, 512, 4096] -> squeeze batch dim -> [512, 4096]
    neg_emb = neg_emb.squeeze(0).cpu().to(torch.bfloat16)

    torch.save({"text_emb": neg_emb}, args.output_path)
    print(f"Saved negative text embedding to {args.output_path}: {neg_emb.shape}")


if __name__ == "__main__":
    main()

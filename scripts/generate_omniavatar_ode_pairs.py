# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Generate ODE pair data for OmniAvatar Causal KD training (Stage 1).

Uses OmniAvatar's EXACT I2V inference pipeline (ModelManager + peft LoRA +
FlowMatchScheduler). The only addition is trajectory capture at target timesteps.

Output structure (saved into each sample directory):
    path.pth    — ODE trajectory intermediates [4, 16, 21, 64, 64] bf16
    latent.pth  — Clean teacher output x_0 [16, 21, 64, 64] bf16

Examples:

    # Single GPU, 14B teacher (default)
    python scripts/generate_omniavatar_ode_pairs.py \
        --data-list /path/to/video_square_path.txt

    # Multi-GPU with torchrun
    torchrun --nproc_per_node=4 --standalone \
        scripts/generate_omniavatar_ode_pairs.py \
        --data-list /path/to/video_square_path.txt

    # 1.3B teacher, subset for smoke testing
    python scripts/generate_omniavatar_ode_pairs.py \
        --data-list /path/to/video_square_path.txt \
        --teacher-size 1.3B --num-samples 5
"""

from __future__ import annotations

import argparse
import os
import sys
import time

import torch
import torch.nn as nn

# --- OmniAvatar setup (must be done before importing OmniAvatar modules) ---
OMNIAVATAR_PATH = os.getenv("OMNIAVATAR_PATH", "/home/work/.local/OmniAvatar")
sys.path.insert(0, OMNIAVATAR_PATH)

import OmniAvatar.utils.args_config as _args_cfg  # noqa: E402

_args_cfg.args = argparse.Namespace(
    use_audio=True,
    i2v=True,
    random_prefix_frames=True,
    train_architecture="lora",
    debug=False,
    local_rank=0,
    rank=0,
    sp_size=1,
)

# --- Constants ---
T_LIST = [0.999, 0.937, 0.833, 0.624, 0.0]
PATH_TIMESTEPS = [t for t in T_LIST if t > 0]
NUM_FRAMES = 81
NUM_STEPS = 50
GUIDANCE_SCALE = 4.5
NEGATIVE_PROMPT = (
    "Vivid color tones, background/camera moving quickly, screen switching, "
    "subtitles and special effects, mutation, overexposed, static, blurred details, "
    "subtitles, style, work, painting, image, still, overall grayish, worst quality, "
    "low quality, JPEG compression residue, ugly, incomplete, extra fingers, "
    "poorly drawn hands, poorly drawn face, deformed, disfigured, malformed limbs, "
    "fingers merging, motionless image, chaotic background, three legs, crowded "
    "background with many people, walking backward"
)

CKPT_ROOT = os.getenv("CKPT_ROOT_DIR", os.path.join(OMNIAVATAR_PATH, "pretrained_models"))

CONFIGS = {
    "14B": {
        "dit_paths": [
            os.path.join(CKPT_ROOT, f"Wan2.1-T2V-14B/diffusion_pytorch_model-0000{i}-of-00006.safetensors")
            for i in range(1, 7)
        ],
        "text_encoder_path": os.path.join(CKPT_ROOT, "Wan2.1-T2V-14B/models_t5_umt5-xxl-enc-bf16.pth"),
        "vae_path": os.path.join(CKPT_ROOT, "Wan2.1-T2V-14B/Wan2.1_VAE.pth"),
        "omniavatar_ckpt": os.path.join(CKPT_ROOT, "OmniAvatar-14B/pytorch_model.pt"),
        "lora_rank": 128,
        "lora_alpha": 64.0,
    },
    "1.3B": {
        "dit_paths": [os.path.join(CKPT_ROOT, "Wan2.1-T2V-1.3B/diffusion_pytorch_model.safetensors")],
        "text_encoder_path": os.path.join(CKPT_ROOT, "Wan2.1-T2V-1.3B/models_t5_umt5-xxl-enc-bf16.pth"),
        "vae_path": os.path.join(CKPT_ROOT, "Wan2.1-T2V-1.3B/Wan2.1_VAE.pth"),
        "omniavatar_ckpt": os.path.join(CKPT_ROOT, "OmniAvatar-1.3B/pytorch_model.pt"),
        "lora_rank": 128,
        "lora_alpha": 64.0,
    },
}


def load_pipeline(model_size: str, device: torch.device):
    """Load OmniAvatar pipeline using the EXACT same method as inference.py."""
    from OmniAvatar.models.model_manager import ModelManager
    from OmniAvatar.wan_video import WanVideoPipeline
    from peft import LoraConfig, inject_adapter_in_model

    cfg = CONFIGS[model_size]
    dtype = torch.bfloat16

    # Step 1: Load base models via ModelManager (same as inference.py:126-135)
    print("  Loading base models via ModelManager...")
    model_manager = ModelManager(device="cpu", infer=True)
    model_manager.load_models(
        [cfg["dit_paths"], cfg["text_encoder_path"], cfg["vae_path"]],
        torch_dtype=dtype,
        device="cpu",
    )

    # Step 2: Create pipeline (same as inference.py:137-141, use_usp=False for single GPU)
    print("  Creating WanVideoPipeline...")
    pipe = WanVideoPipeline.from_model_manager(
        model_manager, torch_dtype=dtype, device=str(device), use_usp=False, infer=True,
    )

    # Step 3: Inject LoRA (same as inference.py:142-151)
    print(f"  Injecting LoRA (rank={cfg['lora_rank']}, alpha={cfg['lora_alpha']})...")
    lora_config = LoraConfig(
        r=cfg["lora_rank"],
        lora_alpha=cfg["lora_alpha"],
        init_lora_weights=True,
        target_modules=["q", "k", "v", "o", "ffn.0", "ffn.2"],
    )
    pipe.dit = inject_adapter_in_model(lora_config, pipe.dit)

    # Load LoRA + audio + patch_embedding weights from OmniAvatar checkpoint
    ckpt_sd = torch.load(cfg["omniavatar_ckpt"], map_location="cpu", weights_only=True)

    # Handle patch_embedding size mismatch: base model has in_dim=16,
    # OmniAvatar expands to in_dim=33 (I2V: 16 noisy + 16 ref + 1 mask).
    if "patch_embedding.weight" in ckpt_sd:
        ckpt_pe_w = ckpt_sd.pop("patch_embedding.weight")
        ckpt_pe_b = ckpt_sd.pop("patch_embedding.bias", None)
        new_pe = nn.Conv3d(
            ckpt_pe_w.shape[1], ckpt_pe_w.shape[0],
            kernel_size=(1, 2, 2), stride=(1, 2, 2),
        )
        new_pe.weight = nn.Parameter(ckpt_pe_w)
        if ckpt_pe_b is not None:
            new_pe.bias = nn.Parameter(ckpt_pe_b)
        pipe.dit.patch_embedding = new_pe.to(device=pipe.device, dtype=dtype)
        print(f"  Replaced patch_embedding: in_dim=16 -> {ckpt_pe_w.shape[1]}")

    missing, unexpected = pipe.dit.load_state_dict(ckpt_sd, strict=False)
    all_keys = [name for name, _ in pipe.dit.named_parameters()]
    print(f"  Loaded {len(all_keys) - len(missing)} params from checkpoint. "
          f"{len(unexpected)} unexpected.")

    # Move all models to device
    pipe.dit = pipe.dit.to(device=device, dtype=dtype)
    pipe.text_encoder = pipe.text_encoder.to(device=device, dtype=dtype)
    pipe.vae = pipe.vae.to(device=device)

    # Freeze and eval (same as inference.py:155-156)
    pipe.requires_grad_(False)
    pipe.eval()

    return pipe


@torch.no_grad()
def generate_with_trajectory(
    pipe,
    img_lat,
    prompt,
    audio_emb_tensor,
    target_timesteps,
    negative_prompt=NEGATIVE_PROMPT,
    cfg_scale=GUIDANCE_SCALE,
    num_inference_steps=NUM_STEPS,
):
    """Run ODE solve identical to pipe.log_video() but capture trajectory."""
    device = pipe.device
    dtype = pipe.torch_dtype

    # Scheduler setup (same as log_video)
    pipe.scheduler.set_timesteps(num_inference_steps, denoising_strength=1.0, shift=5.0)

    # Latent init
    T = img_lat.shape[2]
    H, W = img_lat.shape[3], img_lat.shape[4]
    latents = torch.randn_like(img_lat)

    # Encode prompts (same as log_video)
    prompt_emb_posi = pipe.encode_prompt(prompt, positive=True)
    prompt_emb_nega = pipe.encode_prompt(negative_prompt, positive=False)

    # Build image_emb / y (same as inference.py:272-275)
    ref_frame = img_lat[:, :, :1]
    image_cat = ref_frame.repeat(1, 1, T, 1, 1)
    msk = torch.zeros(1, 1, T, H, W, device=device, dtype=dtype)
    msk[:, :, 1:] = 1
    image_emb = {"y": torch.cat([image_cat, msk], dim=1)}

    # Audio emb
    audio_emb = {"audio_emb": audio_emb_tensor} if audio_emb_tensor is not None else {}

    # Extra input
    extra_input = pipe.prepare_extra_input(latents)

    # Trajectory capture setup
    target_set = sorted(target_timesteps, reverse=True)
    captured = {}

    # Denoise (same as log_video:254-277, with trajectory capture)
    fixed_frame = 1
    for progress_id, timestep in enumerate(pipe.scheduler.timesteps):
        # Inject reference frame (same as log_video:256-257)
        if fixed_frame > 0:
            latents[:, :, :fixed_frame] = img_lat[:, :, :fixed_frame]

        # Trajectory capture: record state at target sigmas
        sigma = pipe.scheduler.sigmas[progress_id].item()
        for target_t in target_set:
            if target_t not in captured and sigma <= target_t:
                captured[target_t] = latents.clone()

        timestep = timestep.unsqueeze(0).to(dtype=dtype, device=device)

        # Positive prediction (same as log_video:261)
        noise_pred_posi = pipe.dit(
            latents, timestep=timestep, **prompt_emb_posi, **image_emb, **audio_emb, **extra_input,
            tea_cache=None,
        )

        # Negative prediction + CFG (same as log_video:262-268)
        audio_emb_uc = {k: torch.zeros_like(v) for k, v in audio_emb.items()}
        noise_pred_nega = pipe.dit(
            latents, timestep=timestep, **prompt_emb_nega, **image_emb, **audio_emb_uc, **extra_input,
            tea_cache=None,
        )
        noise_pred = noise_pred_nega + cfg_scale * (noise_pred_posi - noise_pred_nega)

        # Scheduler step (same as log_video:277)
        latents = pipe.scheduler.step(noise_pred, pipe.scheduler.timesteps[progress_id], latents)

    # Final reference frame injection (same as log_video:279-280)
    if fixed_frame > 0:
        latents[:, :, :fixed_frame] = img_lat[:, :, :fixed_frame]

    # Capture any remaining targets
    for target_t in target_set:
        if target_t not in captured:
            print(f"    WARNING: t={target_t} not captured, using final latents")
            captured[target_t] = latents.clone()

    x_0 = latents
    path = torch.stack([captured[t].squeeze(0) for t in target_set], dim=0)
    return x_0, path


@torch.no_grad()
def visualize_ode_pairs(pipe, path_save, x_0_save, gt_latents, viz_dir, sample_name, device):
    """Decode latents with VAE and save a visualization grid."""
    import torchvision
    import torchvision.transforms.functional as TF
    from collections import OrderedDict

    os.makedirs(viz_dir, exist_ok=True)

    def decode_latent(lat):
        frames = pipe.decode_video(lat.unsqueeze(0).to(device=device, dtype=torch.bfloat16))
        frames = (frames.permute(0, 2, 1, 3, 4).float() + 1) / 2
        return frames.squeeze(0).clamp(0, 1).cpu()

    frames_dict = OrderedDict()
    frames_dict["GT (original)"] = decode_latent(gt_latents)
    for i, t_val in enumerate(PATH_TIMESTEPS):
        frames_dict[f"t={t_val}"] = decode_latent(path_save[i])
    frames_dict["x_0 (clean)"] = decode_latent(x_0_save)

    T = frames_dict["GT (original)"].shape[0]
    frame_indices = [0, T // 4, T // 2, 3 * T // 4, T - 1]

    grid_images = []
    for frames in frames_dict.values():
        for fi in frame_indices:
            grid_images.append(frames[min(fi, frames.shape[0] - 1)])

    grid = torchvision.utils.make_grid(grid_images, nrow=len(frame_indices), padding=2, normalize=False)
    grid_pil = TF.to_pil_image(grid.cpu())
    grid_path = os.path.join(viz_dir, f"{sample_name}_ode_grid.png")
    grid_pil.save(grid_path)
    print(f"    Saved grid: {grid_path}")


def main():
    parser = argparse.ArgumentParser(description="Generate ODE pairs for OmniAvatar Causal KD")
    parser.add_argument("--data-list", type=str, required=True,
                        help="Text file with one video directory per line")
    parser.add_argument("--teacher-size", type=str, default="14B", choices=["14B", "1.3B"],
                        help="Teacher model size (default: 14B)")
    parser.add_argument("--num-steps", type=int, default=NUM_STEPS,
                        help="Number of Euler solver steps (default: 50)")
    parser.add_argument("--guidance-scale", type=float, default=GUIDANCE_SCALE,
                        help="CFG scale (default: 4.5)")
    parser.add_argument("--num-samples", type=int, default=0,
                        help="Max samples to process, 0=all (default: 0)")
    parser.add_argument("--no-skip", action="store_true",
                        help="Regenerate even if path.pth already exists")
    parser.add_argument("--visualize", action="store_true",
                        help="Decode and save visualization grids after generating each sample")
    parser.add_argument("--viz-dir", type=str, default=None,
                        help="Directory to save visualization grids (default: <data-list-dir>/viz)")

    args = parser.parse_args()

    # Distributed setup
    rank = int(os.environ.get("RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    if world_size > 1:
        torch.distributed.init_process_group(backend="nccl")
    device = torch.device(f"cuda:{rank}")
    torch.cuda.set_device(device)

    viz_dir = args.viz_dir
    if args.visualize and viz_dir is None:
        viz_dir = os.path.join(os.path.dirname(args.data_list), "viz")

    print("=" * 60)
    print("ODE Pair Generation for OmniAvatar Causal KD (Native Pipeline)")
    print(f"Teacher: I2V {args.teacher_size} | Steps: {args.num_steps} | CFG: {args.guidance_scale}")
    print(f"Target timesteps: {PATH_TIMESTEPS}")
    if args.visualize:
        print(f"Visualization: {viz_dir}")
    print(f"Rank: {rank}/{world_size}")
    print("=" * 60)

    """
    all_dirs = 
    [
        "/home/work/stableavatar_data/v2v_training_data_filtered_10/RD_Radio10_000_shot_001_001",
        "/home/work/stableavatar_data/v2v_training_data_filtered_10/RD_Radio10_000_shot_001_008",
        "/home/work/stableavatar_data/v2v_training_data_filtered_10/RD_Radio10_000_shot_001_003",
        # ... 7 more
    ]
    """
    # Load data list
    with open(args.data_list, "r") as f:
        all_dirs = [line.strip() for line in f if line.strip()]

    all_dirs = [d for d in all_dirs if os.path.isdir(d)]
    print(f"[Rank {rank}] Total available samples: {len(all_dirs)}")

    if args.num_samples > 0:
        all_dirs = all_dirs[:args.num_samples]
        print(f"[Rank {rank}] Using subset of {args.num_samples} samples")

    """
        With single GPU (rank=0, world_size=1), it's the same — all 10 entries.

        With multi-GPU, e.g. torchrun --nproc_per_node=4 (world_size=4):

        # Rank 0: indices 0, 4, 8
        [
            ".../RD_Radio10_000_shot_001_001",
            ".../RD_Radio10_000_shot_001_006",
            ".../RD_Radio10_000_shot_001_005",
        ]

        # Rank 1: indices 1, 5, 9
        [
            ".../RD_Radio10_000_shot_001_008",
            ".../RD_Radio10_000_shot_001_007",
            ".../RD_Radio10_000_shot_001_010",
        ]

        # Rank 2: indices 2, 6
        [
            ".../RD_Radio10_000_shot_001_003",
            ".../RD_Radio11_000_shot_001_000",
        ]

        # Rank 3: indices 3, 7
        [
            ".../RD_Radio10_000_shot_001_002",
            ".../RD_Radio10_000_shot_001_013",
        ]

        It's round-robin distribution — each rank takes every world_size-th sample.
    """
    # Distribute across ranks
    all_dirs = [all_dirs[i] for i in range(rank, len(all_dirs), world_size)]
    print(f"[Rank {rank}] Assigned {len(all_dirs)} samples")

    # Skip existing
    if not args.no_skip:
        pending = [d for d in all_dirs if not os.path.exists(os.path.join(d, "path.pth"))]
        skipped = len(all_dirs) - len(pending)
        if skipped > 0:
            print(f"[Rank {rank}] Skipping {skipped} samples with existing path.pth")
        all_dirs = pending

    if len(all_dirs) == 0:
        print(f"[Rank {rank}] No samples to process. Done!")
        return

    print(f"[Rank {rank}] Samples to process: {len(all_dirs)}")

    # Load pipeline (native OmniAvatar)
    print(f"\n[Rank {rank}] Loading {args.teacher_size} I2V pipeline (native OmniAvatar)...")
    t0 = time.time()
    pipe = load_pipeline(args.teacher_size, device)
    print(f"[Rank {rank}] Pipeline loaded in {time.time() - t0:.1f}s, "
          f"VRAM: {torch.cuda.memory_allocated(device) / (1024**3):.1f} GB")

    # Process samples
    total = len(all_dirs)
    success_count = 0
    error_count = 0

    for idx, video_dir in enumerate(all_dirs):
        try:
            t_start = time.time()

            # Load precomputed data
            vae_data = torch.load(
                os.path.join(video_dir, "vae_latents.pt"),
                map_location="cpu", weights_only=True,
            )
            input_latents = vae_data["input_latents"]  # [16, T_lat, H_lat, W_lat]

            audio_data = torch.load(
                os.path.join(video_dir, "audio_emb_omniavatar.pt"),
                map_location="cpu", weights_only=True,
            )
            audio_emb = audio_data["audio_emb"][:NUM_FRAMES]  # [T_video, 10752]
            audio_emb_tensor = audio_emb.unsqueeze(0).to(device, dtype=torch.bfloat16)

            # Read prompt text (same as inference.py — encode on the fly)
            prompt_path = os.path.join(video_dir, "prompt.txt")
            with open(prompt_path, "r") as f:
                prompt = f.read().strip()

            # Build img_lat: first frame = reference, rest = zeros (same as inference.py:306)
            T_lat = input_latents.shape[1]
            img_lat = input_latents[:, :1].unsqueeze(0).to(device, dtype=torch.bfloat16)  # [1,16,1,H,W]
            img_lat = torch.cat([
                img_lat,
                torch.zeros_like(img_lat[:, :, :1].repeat(1, 1, T_lat - 1, 1, 1)),
            ], dim=2)  # [1, 16, T_lat, H, W]

            # Run ODE solve with trajectory capture
            x_0, path = generate_with_trajectory(
                pipe=pipe,
                img_lat=img_lat,
                prompt=prompt,
                audio_emb_tensor=audio_emb_tensor,
                target_timesteps=PATH_TIMESTEPS,
                cfg_scale=args.guidance_scale,
                num_inference_steps=args.num_steps,
            )

            # Save
            x_0_save = x_0.squeeze(0).cpu().to(torch.bfloat16)
            path_save = path.cpu().to(torch.bfloat16)
            torch.save(path_save, os.path.join(video_dir, "path.pth"))
            torch.save(x_0_save, os.path.join(video_dir, "latent.pth"))

            # Visualize
            if args.visualize:
                visualize_ode_pairs(
                    pipe, path_save, x_0_save, input_latents,
                    viz_dir, os.path.basename(video_dir), device,
                )

            elapsed = time.time() - t_start
            success_count += 1

            if (idx + 1) % 5 == 0 or (idx + 1) == total:
                print(f"[Rank {rank}][{idx+1}/{total}] {os.path.basename(video_dir)} "
                      f"path={path_save.shape} {elapsed:.1f}s "
                      f"({success_count} ok, {error_count} err)")

        except Exception as e:
            error_count += 1
            print(f"[Rank {rank}][{idx+1}/{total}] ERROR {os.path.basename(video_dir)}: {e}")
            import traceback
            traceback.print_exc()

    print(f"\n[Rank {rank}] Done! {success_count} ok, {error_count} err out of {total}")

    if world_size > 1:
        torch.distributed.destroy_process_group()


if __name__ == "__main__":
    main()

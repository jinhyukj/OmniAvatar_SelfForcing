"""
Generate ODE pairs for OmniAvatar Causal KD pretraining on Modal.

Uses OmniAvatar's EXACT I2V inference pipeline (ModelManager + peft LoRA +
FlowMatchScheduler). The only addition is trajectory capture at target timesteps.

Output per sample directory:
    path.pth    — ODE trajectory intermediates [4, 16, 21, 64, 64] bf16
    latent.pth  — Clean teacher output x_0 [16, 21, 64, 64] bf16

Usage:
    # Smoke test with 5 samples using 14B teacher (default)
    modal run generate_ode_pairs_modal.py --subset 5

    # 1.3B teacher
    modal run generate_ode_pairs_modal.py --teacher-size 1.3B --subset 5

    # Full dataset
    modal run generate_ode_pairs_modal.py

    # Parallelise across multiple runs
    modal run generate_ode_pairs_modal.py --start-idx 0 --end-idx 10000
    modal run generate_ode_pairs_modal.py --start-idx 10000 --end-idx 20000

    # Background (survives terminal close)
    modal run --detach generate_ode_pairs_modal.py --subset 50
"""

import modal

app = modal.App("fastgen-ode-pairs")

data_vol = modal.Volume.from_name("fastgen-data", create_if_missing=True)
output_vol = modal.Volume.from_name("fastgen-output", create_if_missing=True)

image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("git", "ffmpeg")
    .pip_install(
        "torch==2.10.0",
        "torchvision",
        "transformers==4.49.0",
        "accelerate",
        "omegaconf",
        "loguru",
        "attrs",
        "einops",
        "numpy<2.0.0",
        "psutil",
        "pandas",
        "safetensors",
        "Pillow",
        "peft==0.15.1",
        "ftfy",
        "scipy==1.14.0",
    )
    # Add OmniAvatar code (for native pipeline)
    .add_local_dir("/home/work/.local/OmniAvatar", remote_path="/root/OmniAvatar", copy=True, ignore=[
        ".git", "__pycache__", "*.pyc", "pretrained_models", "outputs",
    ])
    .add_local_dir(".", remote_path="/root/FastGen", copy=True, ignore=[
        ".git", "__pycache__", "*.pyc", "third_party",
        "FASTGEN_OUTPUT", "runs", "tmp", ".claude",
    ])
    .run_commands("cd /root/FastGen && pip install -e .")
    .env({
        "FASTGEN_OUTPUT_ROOT": "/tmp/fastgen_output",
        "CKPT_ROOT_DIR": "/mnt/data/models",
    })
)


# --- Constants ---
T_LIST = [0.999, 0.937, 0.833, 0.624, 0.0]
PATH_TIMESTEPS = [t for t in T_LIST if t > 0]
GUIDANCE_SCALE = 4.5
NUM_STEPS = 50
NUM_FRAMES = 81
NEGATIVE_PROMPT = (
    "Vivid color tones, background/camera moving quickly, screen switching, "
    "subtitles and special effects, mutation, overexposed, static, blurred details, "
    "subtitles, style, work, painting, image, still, overall grayish, worst quality, "
    "low quality, JPEG compression residue, ugly, incomplete, extra fingers, "
    "poorly drawn hands, poorly drawn face, deformed, disfigured, malformed limbs, "
    "fingers merging, motionless image, chaotic background, three legs, crowded "
    "background with many people, walking backward"
)

# Data paths (on Modal volume)
DATA_LIST_PATH = "/mnt/data/v2v_training_data/video_square_path.txt"

# Model configs (paths on Modal volume)
CONFIGS = {
    "14B": {
        "dit_paths": [
            f"/mnt/data/models/Wan2.1-T2V-14B/diffusion_pytorch_model-0000{i}-of-00006.safetensors"
            for i in range(1, 7)
        ],
        "text_encoder_path": "/mnt/data/models/Wan2.1-T2V-14B/models_t5_umt5-xxl-enc-bf16.pth",
        "vae_path": "/mnt/data/models/Wan2.1-T2V-14B/Wan2.1_VAE.pth",
        "omniavatar_ckpt": "/mnt/data/models/OmniAvatar-14B/pytorch_model.pt",
        "lora_rank": 128,
        "lora_alpha": 64.0,
    },
    "1.3B": {
        "dit_paths": ["/mnt/data/models/Wan2.1-T2V-1.3B/diffusion_pytorch_model.safetensors"],
        "text_encoder_path": "/mnt/data/models/Wan2.1-T2V-1.3B/models_t5_umt5-xxl-enc-bf16.pth",
        "vae_path": "/mnt/data/models/Wan2.1-T2V-1.3B/Wan2.1_VAE.pth",
        "omniavatar_ckpt": "/mnt/data/models/OmniAvatar-1.3B/pytorch_model.pt",
        "lora_rank": 128,
        "lora_alpha": 64.0,
    },
}


def _setup_omniavatar():
    """Set up OmniAvatar imports and global args (must be called before any OmniAvatar import)."""
    import argparse
    import sys

    sys.path.insert(0, "/root/OmniAvatar")

    import OmniAvatar.utils.args_config as _args_cfg

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


def _load_pipeline(model_size, device):
    """Load OmniAvatar pipeline using the EXACT same method as inference.py."""
    import torch
    import torch.nn as nn
    from OmniAvatar.models.model_manager import ModelManager
    from OmniAvatar.wan_video import WanVideoPipeline
    from peft import LoraConfig, inject_adapter_in_model

    cfg = CONFIGS[model_size]
    dtype = torch.bfloat16

    print("  Loading base models via ModelManager...")
    model_manager = ModelManager(device="cpu", infer=True)
    model_manager.load_models(
        [cfg["dit_paths"], cfg["text_encoder_path"], cfg["vae_path"]],
        torch_dtype=dtype,
        device="cpu",
    )

    print("  Creating WanVideoPipeline...")
    pipe = WanVideoPipeline.from_model_manager(
        model_manager, torch_dtype=dtype, device=str(device), use_usp=False, infer=True,
    )

    print(f"  Injecting LoRA (rank={cfg['lora_rank']}, alpha={cfg['lora_alpha']})...")
    lora_config = LoraConfig(
        r=cfg["lora_rank"],
        lora_alpha=cfg["lora_alpha"],
        init_lora_weights=True,
        target_modules=["q", "k", "v", "o", "ffn.0", "ffn.2"],
    )
    pipe.dit = inject_adapter_in_model(lora_config, pipe.dit)

    ckpt_sd = torch.load(cfg["omniavatar_ckpt"], map_location="cpu", weights_only=True)

    # Handle patch_embedding size mismatch
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

    pipe.dit = pipe.dit.to(device=device, dtype=dtype)
    pipe.text_encoder = pipe.text_encoder.to(device=device, dtype=dtype)
    pipe.vae = pipe.vae.to(device=device)

    pipe.requires_grad_(False)
    pipe.eval()

    return pipe


def _generate_with_trajectory(
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
    import torch

    device = pipe.device
    dtype = pipe.torch_dtype

    pipe.scheduler.set_timesteps(num_inference_steps, denoising_strength=1.0, shift=5.0)

    T = img_lat.shape[2]
    H, W = img_lat.shape[3], img_lat.shape[4]
    latents = torch.randn_like(img_lat)

    prompt_emb_posi = pipe.encode_prompt(prompt, positive=True)
    prompt_emb_nega = pipe.encode_prompt(negative_prompt, positive=False)

    ref_frame = img_lat[:, :, :1]
    image_cat = ref_frame.repeat(1, 1, T, 1, 1)
    msk = torch.zeros(1, 1, T, H, W, device=device, dtype=dtype)
    msk[:, :, 1:] = 1
    image_emb = {"y": torch.cat([image_cat, msk], dim=1)}

    audio_emb = {"audio_emb": audio_emb_tensor} if audio_emb_tensor is not None else {}
    extra_input = pipe.prepare_extra_input(latents)

    target_set = sorted(target_timesteps, reverse=True)
    captured = {}

    fixed_frame = 1
    for progress_id, timestep in enumerate(pipe.scheduler.timesteps):
        if fixed_frame > 0:
            latents[:, :, :fixed_frame] = img_lat[:, :, :fixed_frame]

        sigma = pipe.scheduler.sigmas[progress_id].item()
        for target_t in target_set:
            if target_t not in captured and sigma <= target_t:
                captured[target_t] = latents.clone()

        timestep = timestep.unsqueeze(0).to(dtype=dtype, device=device)

        noise_pred_posi = pipe.dit(
            latents, timestep=timestep, **prompt_emb_posi, **image_emb, **audio_emb, **extra_input,
            tea_cache=None,
        )

        audio_emb_uc = {k: torch.zeros_like(v) for k, v in audio_emb.items()}
        noise_pred_nega = pipe.dit(
            latents, timestep=timestep, **prompt_emb_nega, **image_emb, **audio_emb_uc, **extra_input,
            tea_cache=None,
        )
        noise_pred = noise_pred_nega + cfg_scale * (noise_pred_posi - noise_pred_nega)

        latents = pipe.scheduler.step(noise_pred, pipe.scheduler.timesteps[progress_id], latents)

    if fixed_frame > 0:
        latents[:, :, :fixed_frame] = img_lat[:, :, :fixed_frame]

    for target_t in target_set:
        if target_t not in captured:
            print(f"    WARNING: t={target_t} not captured, using final latents")
            captured[target_t] = latents.clone()

    x_0 = latents
    path = torch.stack([captured[t].squeeze(0) for t in target_set], dim=0)
    return x_0, path


def _generate_ode_pairs(
    teacher_size: str = "14B",
    subset: int = 0,
    start_idx: int = 0,
    end_idx: int = -1,
    skip_existing: bool = True,
    num_steps: int = NUM_STEPS,
    guidance_scale: float = GUIDANCE_SCALE,
):
    """Generate ODE pairs using OmniAvatar's native pipeline."""
    import os
    import time
    import torch

    # Setup OmniAvatar imports
    _setup_omniavatar()

    print("=" * 60)
    print("ODE Pair Generation for OmniAvatar Causal KD (Native Pipeline)")
    print(f"Teacher: I2V {teacher_size} | Steps: {num_steps} | CFG: {guidance_scale}")
    print(f"Target timesteps: {PATH_TIMESTEPS}")
    print("=" * 60)

    # --- Load data list ---
    with open(DATA_LIST_PATH, "r") as f:
        raw_dirs = [line.strip() for line in f if line.strip()]

    all_dirs = []
    for d in raw_dirs:
        sample_name = os.path.basename(d)
        modal_path = f"/mnt/data/v2v_training_data/{sample_name}"
        if os.path.isdir(modal_path):
            all_dirs.append(modal_path)
    print(f"Total samples in data list: {len(raw_dirs)}, available on volume: {len(all_dirs)}")

    if subset > 0:
        all_dirs = all_dirs[:subset]
        print(f"Using subset of {subset} samples")
    elif end_idx > 0:
        all_dirs = all_dirs[start_idx:end_idx]
        print(f"Using range [{start_idx}, {end_idx}): {len(all_dirs)} samples")
    elif start_idx > 0:
        all_dirs = all_dirs[start_idx:]
        print(f"Using range [{start_idx}, end): {len(all_dirs)} samples")

    if skip_existing:
        pending_dirs = [d for d in all_dirs if not os.path.exists(os.path.join(d, "path.pth"))]
        skipped = len(all_dirs) - len(pending_dirs)
        if skipped > 0:
            print(f"Skipping {skipped} samples with existing path.pth")
        all_dirs = pending_dirs

    if len(all_dirs) == 0:
        print("No samples to process. Done!")
        return

    print(f"Samples to process: {len(all_dirs)}")

    # --- Load pipeline (native OmniAvatar) ---
    device = torch.device("cuda")
    print(f"\nLoading {teacher_size} I2V pipeline (native OmniAvatar)...")
    t0 = time.time()
    pipe = _load_pipeline(teacher_size, device)
    load_time = time.time() - t0
    vram_gb = torch.cuda.memory_allocated() / (1024**3)
    print(f"Pipeline loaded in {load_time:.1f}s, VRAM: {vram_gb:.1f} GB")

    # --- Process samples ---
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
            audio_emb = audio_data["audio_emb"][:NUM_FRAMES]
            audio_emb_tensor = audio_emb.unsqueeze(0).to(device, dtype=torch.bfloat16)

            # Read prompt text
            prompt_path = os.path.join(video_dir, "prompt.txt")
            with open(prompt_path, "r") as f:
                prompt = f.read().strip()

            # Build img_lat: first frame = reference, rest = zeros (same as inference.py:306)
            T_lat = input_latents.shape[1]
            img_lat = input_latents[:, :1].unsqueeze(0).to(device, dtype=torch.bfloat16)
            img_lat = torch.cat([
                img_lat,
                torch.zeros_like(img_lat[:, :, :1].repeat(1, 1, T_lat - 1, 1, 1)),
            ], dim=2)

            # Run ODE solve with trajectory capture
            x_0, path = _generate_with_trajectory(
                pipe=pipe,
                img_lat=img_lat,
                prompt=prompt,
                audio_emb_tensor=audio_emb_tensor,
                target_timesteps=PATH_TIMESTEPS,
                cfg_scale=guidance_scale,
                num_inference_steps=num_steps,
            )

            # Save tensors
            x_0_save = x_0.squeeze(0).cpu().to(torch.bfloat16)
            path_save = path.cpu().to(torch.bfloat16)
            torch.save(path_save, os.path.join(video_dir, "path.pth"))
            torch.save(x_0_save, os.path.join(video_dir, "latent.pth"))

            elapsed = time.time() - t_start
            success_count += 1

            if (idx + 1) % 5 == 0 or (idx + 1) == total:
                print(f"[{idx+1}/{total}] {video_dir.split('/')[-1]} "
                      f"path={path_save.shape} latent={x_0_save.shape} "
                      f"{elapsed:.1f}s ({success_count} ok, {error_count} err)")

        except Exception as e:
            error_count += 1
            print(f"[{idx+1}/{total}] ERROR {video_dir.split('/')[-1]}: {e}")
            import traceback
            traceback.print_exc()

    print(f"\nDone! {success_count} succeeded, {error_count} failed out of {total}")


@app.function(
    image=image,
    gpu="A100-80GB",
    volumes={"/mnt/data": data_vol, "/mnt/output": output_vol},
    timeout=24 * 3600,
)
def generate_ode_pairs_14b(
    subset: int = 0,
    start_idx: int = 0,
    end_idx: int = -1,
    skip_existing: bool = True,
    num_steps: int = NUM_STEPS,
    guidance_scale: float = GUIDANCE_SCALE,
):
    _generate_ode_pairs(
        teacher_size="14B",
        subset=subset, start_idx=start_idx, end_idx=end_idx,
        skip_existing=skip_existing,
        num_steps=num_steps, guidance_scale=guidance_scale,
    )


@app.function(
    image=image,
    gpu="L4",
    volumes={"/mnt/data": data_vol, "/mnt/output": output_vol},
    timeout=24 * 3600,
)
def generate_ode_pairs_1_3b(
    subset: int = 0,
    start_idx: int = 0,
    end_idx: int = -1,
    skip_existing: bool = True,
    num_steps: int = NUM_STEPS,
    guidance_scale: float = GUIDANCE_SCALE,
):
    _generate_ode_pairs(
        teacher_size="1.3B",
        subset=subset, start_idx=start_idx, end_idx=end_idx,
        skip_existing=skip_existing,
        num_steps=num_steps, guidance_scale=guidance_scale,
    )


@app.local_entrypoint()
def main(
    teacher_size: str = "14B",
    subset: int = 0,
    start_idx: int = 0,
    end_idx: int = -1,
    no_skip: bool = False,
    num_steps: int = NUM_STEPS,
    guidance_scale: float = GUIDANCE_SCALE,
):
    if teacher_size == "14B":
        generate_ode_pairs_14b.remote(
            subset=subset, start_idx=start_idx, end_idx=end_idx,
            skip_existing=not no_skip,
            num_steps=num_steps, guidance_scale=guidance_scale,
        )
    else:
        generate_ode_pairs_1_3b.remote(
            subset=subset, start_idx=start_idx, end_idx=end_idx,
            skip_existing=not no_skip,
            num_steps=num_steps, guidance_scale=guidance_scale,
        )

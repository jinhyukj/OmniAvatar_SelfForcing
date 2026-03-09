"""
Generate ODE pairs for OmniAvatar KD pretraining using the 14B I2V teacher on Modal.

For each training sample, runs a 4-step deterministic ODE denoising with CFG using the
14B I2V OmniAvatar teacher, and saves the intermediate noisy states as path.pth.

The path tensor has shape [4, 16, 21, 64, 64] (bf16) and contains the noisy latent
states at timesteps [0.999, 0.937, 0.833, 0.624].

Usage:
    # Smoke test with 50 samples
    modal run generate_ode_pairs_modal.py --subset 50

    # Full dataset
    modal run generate_ode_pairs_modal.py

    # Parallelise across multiple runs
    modal run generate_ode_pairs_modal.py --start-idx 0 --end-idx 10000
    modal run generate_ode_pairs_modal.py --start-idx 10000 --end-idx 20000

    # Background (survives terminal close)
    modal run --detach generate_ode_pairs_modal.py --subset 50

Setup (one-time):
    pip install modal
    python3 -m modal setup
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
        "diffusers==0.35.1",
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
    )
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
GUIDANCE_SCALE = 4.0
NUM_FRAMES = 81  # video frames -> 21 latent frames
LATENT_SHAPE = (16, 21, 64, 64)  # C, T_lat, H_lat, W_lat

# Checkpoint paths (relative to /mnt/data/models)
BASE_14B_PATHS = [
    f"/mnt/data/models/Wan2.1-T2V-14B/diffusion_pytorch_model-0000{i}-of-00006.safetensors"
    for i in range(1, 7)
]
OMNIAVATAR_14B_PATH = "/mnt/data/models/OmniAvatar-14B/pytorch_model.pt"

# Data paths
DATA_LIST_PATH = "/mnt/data/v2v_training_data/video_square_path.txt"
LATENTSYNC_MASK_PATH = "/mnt/data/models/latentsync_mask.png"


def _generate_ode_pairs(
    subset: int = 0,
    start_idx: int = 0,
    end_idx: int = -1,
    skip_existing: bool = True,
):
    """Generate ODE pairs for OmniAvatar KD pretraining."""
    import os
    import time
    import torch

    os.chdir("/root/FastGen")

    from fastgen.networks.OmniAvatar.network import OmniAvatarWan

    print("=" * 60)
    print("ODE Pair Generation for OmniAvatar KD")
    print("=" * 60)

    # --- Load data list ---
    # The path list may contain local paths (/home/work/stableavatar_data/v2v_training_data/...)
    # Remap to Modal volume paths (/mnt/data/v2v_training_data/...)
    with open(DATA_LIST_PATH, "r") as f:
        raw_dirs = [line.strip() for line in f if line.strip()]

    all_dirs = []
    for d in raw_dirs:
        # Extract the sample directory name and remap to Modal path
        sample_name = os.path.basename(d)
        modal_path = f"/mnt/data/v2v_training_data/{sample_name}"
        if os.path.isdir(modal_path):
            all_dirs.append(modal_path)
    print(f"Total samples in data list: {len(raw_dirs)}, available on volume: {len(all_dirs)}")

    # Apply subset / range
    if subset > 0:
        all_dirs = all_dirs[:subset]
        print(f"Using subset of {subset} samples")
    elif end_idx > 0:
        all_dirs = all_dirs[start_idx:end_idx]
        print(f"Using range [{start_idx}, {end_idx}): {len(all_dirs)} samples")
    elif start_idx > 0:
        all_dirs = all_dirs[start_idx:]
        print(f"Using range [{start_idx}, end): {len(all_dirs)} samples")

    # Filter out samples that already have path.pth
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

    # --- Load teacher model ---
    device = torch.device("cuda")
    print("\nLoading 14B I2V teacher...")
    t0 = time.time()

    teacher = OmniAvatarWan(
        in_dim=33,  # I2V: noise(16) + ref(16) + mask(1)
        dim=5120,
        num_heads=40,
        ffn_dim=13824,
        num_layers=40,
        mode="i2v",
        use_audio=True,
        audio_hidden_size=32,
        has_image_input=False,
        base_model_paths=",".join(BASE_14B_PATHS),
        omniavatar_ckpt_path=OMNIAVATAR_14B_PATH,
        merge_lora=True,
        load_pretrained=True,
        net_pred_type="flow",
        schedule_type="rf",
    )
    teacher = teacher.to(device=device, dtype=torch.bfloat16)
    teacher.eval()
    for p in teacher.parameters():
        p.requires_grad_(False)

    load_time = time.time() - t0
    vram_gb = torch.cuda.memory_allocated() / (1024**3)
    print(f"Teacher loaded in {load_time:.1f}s, VRAM: {vram_gb:.1f} GB")

    # --- Load LatentSync mask ---
    # Mask is 256x256 but latents are 64x64 (512x512 video / 8).
    # Resize mask directly to latent resolution.
    from PIL import Image
    import torchvision.transforms.functional as TF

    mask_img = Image.open(LATENTSYNC_MASK_PATH).convert("L")
    mask = TF.to_tensor(mask_img).squeeze(0)  # [H, W] in [0, 1]
    mask = torch.nn.functional.interpolate(
        mask.unsqueeze(0).unsqueeze(0),
        size=(LATENT_SHAPE[2], LATENT_SHAPE[3]),  # (64, 64)
        mode="nearest",
    ).squeeze(0).squeeze(0)
    mask = (mask > 0.5).float().to(device)

    # --- Build t_list tensor ---
    noise_scheduler = teacher.noise_scheduler
    t_list = torch.tensor(T_LIST, device=device, dtype=noise_scheduler.t_precision)

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
            audio_emb = audio_data["audio_emb"][:NUM_FRAMES]  # [T_video, 10752]

            text_emb_path = os.path.join(video_dir, "text_emb.pt")
            if os.path.exists(text_emb_path):
                text_emb = torch.load(text_emb_path, map_location="cpu", weights_only=True)
                if isinstance(text_emb, dict):
                    text_emb = text_emb.get("text_emb", text_emb.get("context", next(iter(text_emb.values()))))
                if text_emb.dim() == 3:
                    text_emb = text_emb.squeeze(0)
            else:
                text_emb = torch.zeros(512, 4096)

            if text_emb.dim() == 3:
                text_emb = text_emb.squeeze(0)

            # Extract reference latent (first frame)
            ref_latent = input_latents[:, :1]  # [16, 1, H_lat, W_lat]

            # Move to device and add batch dim
            ref_latent = ref_latent.unsqueeze(0).to(device, dtype=torch.bfloat16)
            text_emb = text_emb.unsqueeze(0).to(device, dtype=torch.bfloat16)
            audio_emb = audio_emb.unsqueeze(0).to(device, dtype=torch.bfloat16)

            # Build positive condition (I2V mode: text + audio + ref + mask)
            pos_cond = {
                "text_embeds": text_emb,
                "audio_emb": audio_emb,
                "ref_latent": ref_latent,
                "mask": mask,
            }

            # Build negative condition (no text, no audio)
            neg_cond = {
                "text_embeds": torch.zeros_like(text_emb),
                "audio_emb": torch.zeros_like(audio_emb),
                "ref_latent": ref_latent,
                "mask": mask,
            }

            # --- Run 4-step ODE with CFG ---
            with torch.no_grad():
                # Start from scaled noise
                noise = torch.randn(1, *LATENT_SHAPE, device=device, dtype=torch.bfloat16)
                x = noise_scheduler.latents(noise=noise, t_init=t_list[0])
                x = x.to(torch.bfloat16)

                path_states = []
                for t_cur, t_next in zip(t_list[:-1], t_list[1:]):
                    # Noise scheduler uses float64 for precision, but model expects bf16
                    t_batch = t_cur.unsqueeze(0).to(torch.bfloat16)

                    # CFG: conditional + unconditional predictions
                    x0_cond = teacher(x, t_batch, condition=pos_cond, fwd_pred_type="x0")
                    x0_uncond = teacher(x, t_batch, condition=neg_cond, fwd_pred_type="x0")
                    x0_pred = x0_uncond + GUIDANCE_SCALE * (x0_cond - x0_uncond)

                    # Save current noisy state
                    path_states.append(x.squeeze(0).cpu())

                    # ODE step to next timestep (use float64 t for scheduler precision)
                    if t_next > 0:
                        eps = noise_scheduler.x0_to_eps(xt=x, x0=x0_pred, t=t_cur.unsqueeze(0))
                        x = noise_scheduler.forward_process(x0_pred, eps, t_next.unsqueeze(0))
                        x = x.to(torch.bfloat16)  # scheduler may upcast to float64

            # Stack and save: [4, 16, 21, 64, 64] in bf16
            path_tensor = torch.stack(path_states)  # [4, 16, 21, 64, 64]
            save_path = os.path.join(video_dir, "path.pth")
            torch.save(path_tensor.to(torch.bfloat16), save_path)

            elapsed = time.time() - t_start
            success_count += 1

            if (idx + 1) % 10 == 0 or (idx + 1) == total:
                print(f"[{idx+1}/{total}] {video_dir.split('/')[-1]} "
                      f"path={path_tensor.shape} {elapsed:.1f}s "
                      f"({success_count} ok, {error_count} err)")

        except Exception as e:
            error_count += 1
            print(f"[{idx+1}/{total}] ERROR {video_dir.split('/')[-1]}: {e}")

    print(f"\nDone! {success_count} succeeded, {error_count} failed out of {total}")


@app.function(
    image=image,
    gpu="A100-80GB",
    volumes={"/mnt/data": data_vol, "/mnt/output": output_vol},
    timeout=24 * 3600,
)
def generate_ode_pairs(
    subset: int = 0,
    start_idx: int = 0,
    end_idx: int = -1,
    skip_existing: bool = True,
):
    _generate_ode_pairs(
        subset=subset,
        start_idx=start_idx,
        end_idx=end_idx,
        skip_existing=skip_existing,
    )


@app.local_entrypoint()
def main(
    subset: int = 0,
    start_idx: int = 0,
    end_idx: int = -1,
    no_skip: bool = False,
):
    generate_ode_pairs.remote(
        subset=subset,
        start_idx=start_idx,
        end_idx=end_idx,
        skip_existing=not no_skip,
    )

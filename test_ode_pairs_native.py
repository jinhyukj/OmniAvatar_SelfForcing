"""Test ODE pair generation using the EXACT OmniAvatar I2V inference pipeline.

Uses OmniAvatar's native model loading (ModelManager + peft LoRA), FlowMatchScheduler,
text encoder, and denoising loop. The only addition is trajectory capture.

Usage:
    # Generate ODE pairs + visualize
    python test_ode_pairs_native.py --device cuda:1

    # Visualize only (reuse saved pairs)
    python test_ode_pairs_native.py --device cuda:1 --viz-only
"""

import argparse
import os
import sys
import time

import torch
import torchvision.transforms.functional as TF
from collections import OrderedDict
from tqdm import tqdm

# Add OmniAvatar to path
sys.path.insert(0, "/home/work/.local/OmniAvatar")

# OmniAvatar's WanModel.__init__ reads global `args` from args_config.
# Set it up before any OmniAvatar imports.
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

# --- Constants ---
CKPT_ROOT = "/home/work/.local/OmniAvatar/pretrained_models"
DATA_LIST = "/home/work/stableavatar_data/v2v_training_data/video_square_path.txt"
OUTPUT_DIR = "/home/work/.local/FastGen/tmp/ode_pairs_test_native"

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

# Model configs
CONFIGS = {
    "14B": {
        "dit_paths": [
            f"{CKPT_ROOT}/Wan2.1-T2V-14B/diffusion_pytorch_model-0000{i}-of-00006.safetensors"
            for i in range(1, 7)
        ],
        "text_encoder_path": f"{CKPT_ROOT}/Wan2.1-T2V-14B/models_t5_umt5-xxl-enc-bf16.pth",
        "vae_path": f"{CKPT_ROOT}/Wan2.1-T2V-14B/Wan2.1_VAE.pth",
        "omniavatar_ckpt": f"{CKPT_ROOT}/OmniAvatar-14B/pytorch_model.pt",
        "lora_rank": 128,
        "lora_alpha": 64.0,
    },
    "1.3B": {
        "dit_paths": [f"{CKPT_ROOT}/Wan2.1-T2V-1.3B/diffusion_pytorch_model.safetensors"],
        "text_encoder_path": f"{CKPT_ROOT}/Wan2.1-T2V-1.3B/models_t5_umt5-xxl-enc-bf16.pth",
        "vae_path": f"{CKPT_ROOT}/Wan2.1-T2V-1.3B/Wan2.1_VAE.pth",
        "omniavatar_ckpt": f"{CKPT_ROOT}/OmniAvatar-1.3B/pytorch_model.pt",
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
        init_lora_weights=True,  # "kaiming" maps to True in peft
        target_modules=["q", "k", "v", "o", "ffn.0", "ffn.2"],
    )
    pipe.dit = inject_adapter_in_model(lora_config, pipe.dit)

    # Load LoRA + audio + patch_embedding weights from OmniAvatar checkpoint
    ckpt_sd = torch.load(cfg["omniavatar_ckpt"], map_location="cpu", weights_only=True)

    # Handle patch_embedding size mismatch: base model has in_dim=16,
    # OmniAvatar expands to in_dim=33 (I2V: 16 noisy + 16 ref + 1 mask).
    # Replace the model's patch_embedding with the checkpoint's version.
    if "patch_embedding.weight" in ckpt_sd:
        import torch.nn as nn
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

    # Move all models to device (we have enough VRAM on H200)
    pipe.dit = pipe.dit.to(device=device, dtype=dtype)
    pipe.text_encoder = pipe.text_encoder.to(device=device, dtype=dtype)
    pipe.vae = pipe.vae.to(device=device)

    # Step 4: Freeze and eval (same as inference.py:155-156)
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
    """Run ODE solve identical to pipe.log_video() but capture trajectory.

    This is a copy of WanVideoPipeline.log_video() with trajectory capture added.
    """
    device = pipe.device
    dtype = pipe.torch_dtype

    # --- Scheduler setup (same as log_video:229-230) ---
    pipe.scheduler.set_timesteps(num_inference_steps, denoising_strength=1.0, shift=5.0)

    # --- Latent init (same as log_video:232-233) ---
    # img_lat is [B, 16, T_lat, H, W] where T_lat frames are already expanded
    # (first frame = reference, rest = zeros). log_video replaces all with noise.
    T = img_lat.shape[2]  # latent temporal frames
    latents = torch.randn_like(img_lat)

    # --- Encode prompts (same as log_video:236-239) ---
    # Models already on device — no need for load_models_to_device
    prompt_emb_posi = pipe.encode_prompt(prompt, positive=True)
    prompt_emb_nega = pipe.encode_prompt(negative_prompt, positive=False)

    # --- Build image_emb / y (same as inference.py:272-275) ---
    # Use the single reference frame (first temporal frame), repeated T times
    ref_frame = img_lat[:, :, :1]  # [B, 16, 1, H, W]
    image_cat = ref_frame.repeat(1, 1, T, 1, 1)  # [B, 16, T, H, W]
    msk = torch.zeros(1, 1, T, img_lat.shape[3], img_lat.shape[4], device=device, dtype=dtype)
    msk[:, :, 1:] = 1
    image_emb = {"y": torch.cat([image_cat, msk], dim=1)}  # [B, 17, T, H, W]

    # --- Audio emb ---
    audio_emb = {"audio_emb": audio_emb_tensor} if audio_emb_tensor is not None else {}

    # --- Extra input ---
    extra_input = pipe.prepare_extra_input(latents)

    # --- Trajectory capture setup ---
    target_set = sorted(target_timesteps, reverse=True)
    captured = {}

    # --- Denoise (same as log_video:254-277, with trajectory capture) ---
    fixed_frame = 1  # I2V: keep first latent frame fixed
    # Models already on device
    for progress_id, timestep in enumerate(tqdm(pipe.scheduler.timesteps, desc="ODE solve")):
        # Inject reference frame (same as log_video:256-257)
        if fixed_frame > 0:
            latents[:, :, :fixed_frame] = img_lat[:, :, :fixed_frame]

        # --- Trajectory capture: record state at target sigmas ---
        sigma = pipe.scheduler.sigmas[progress_id].item()
        for target_t in target_set:
            if target_t not in captured and sigma <= target_t:
                captured[target_t] = latents.clone()
                print(f"    Captured t={target_t} at step {progress_id} (sigma={sigma:.6f})")

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


def decode_and_visualize(pipe, path_save, x_0_save, gt_latents, output_dir, sample_name, device):
    """Decode latents with VAE and save visualization grid."""
    import torchvision

    # Models already on device

    def decode_latent(lat):
        """Decode [C, T, H, W] -> [T, 3, H, W] in [0, 1]."""
        frames = pipe.decode_video(lat.unsqueeze(0).to(device=device, dtype=torch.bfloat16))
        frames = (frames.permute(0, 2, 1, 3, 4).float() + 1) / 2  # [B, T, C, H, W]
        return frames.squeeze(0).clamp(0, 1).cpu()  # [T, C, H, W]

    frames_dict = OrderedDict()
    for i, t_val in enumerate(PATH_TIMESTEPS):
        print(f"  Decoding path[{i}] (t={t_val})...")
        frames_dict[f"t={t_val}"] = decode_latent(path_save[i])

    print("  Decoding x_0 (clean)...")
    frames_dict["x_0 (clean)"] = decode_latent(x_0_save)

    print("  Decoding GT latent...")
    frames_dict["GT (original)"] = decode_latent(gt_latents)

    # Build grid
    labels = list(frames_dict.keys())
    all_frames = list(frames_dict.values())
    T = all_frames[0].shape[0]
    frame_indices = [0, T // 4, T // 2, 3 * T // 4, T - 1]

    grid_images = []
    for frames in all_frames:
        for fi in frame_indices:
            fi = min(fi, frames.shape[0] - 1)
            grid_images.append(frames[fi])

    ncol = len(frame_indices)
    grid = torchvision.utils.make_grid(grid_images, nrow=ncol, padding=2, normalize=False)
    grid_pil = TF.to_pil_image(grid.cpu())

    grid_path = os.path.join(output_dir, f"{sample_name}_ode_grid.png")
    grid_pil.save(grid_path)
    print(f"  Saved grid: {grid_path} ({len(labels)} rows x {ncol} cols)")

    # Save individual frames
    for label, frames in frames_dict.items():
        label_clean = label.replace("=", "").replace(" ", "_").replace("(", "").replace(")", "")
        for fi in [0, frames.shape[0] // 2, frames.shape[0] - 1]:
            frame_path = os.path.join(output_dir, f"{sample_name}_{label_clean}_frame{fi:03d}.png")
            TF.to_pil_image(frames[fi]).save(frame_path)

    # Models stay on device


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=str, default="cuda:1")
    parser.add_argument("--model-size", type=str, default="14B", choices=["14B", "1.3B"])
    parser.add_argument("--viz-only", action="store_true")
    parser.add_argument("--sample-idx", type=int, default=0)
    args = parser.parse_args()

    device = torch.device(args.device)
    torch.cuda.set_device(device)
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Pick a sample
    with open(DATA_LIST, "r") as f:
        all_dirs = [line.strip() for line in f if line.strip()]
    video_dir = all_dirs[args.sample_idx]
    sample_name = os.path.basename(video_dir)
    save_dir = os.path.join(OUTPUT_DIR, sample_name)
    print(f"Sample: {sample_name}")
    print(f"Device: {device}")

    if args.viz_only:
        print("\n--viz-only: Loading saved ODE pairs...")
        path_save = torch.load(os.path.join(save_dir, "path.pth"), map_location="cpu", weights_only=True)
        x_0_save = torch.load(os.path.join(save_dir, "latent.pth"), map_location="cpu", weights_only=True)
        print(f"  path.pth: {path_save.shape}, latent.pth: {x_0_save.shape}")

        # Load pipeline just for VAE decode
        print(f"\nLoading {args.model_size} pipeline (VAE only)...")
        pipe = load_pipeline(args.model_size, device)
    else:
        # --- Load pipeline ---
        print(f"\nLoading {args.model_size} I2V pipeline...")
        t0 = time.time()
        pipe = load_pipeline(args.model_size, device)
        print(f"Pipeline loaded in {time.time() - t0:.1f}s, "
              f"VRAM: {torch.cuda.memory_allocated(device) / (1024**3):.1f} GB")

        # --- Load sample data ---
        print("\nLoading sample data...")

        # VAE-encode reference image to get img_lat (same as inference.py:270)
        vae_data = torch.load(os.path.join(video_dir, "vae_latents.pt"), map_location="cpu", weights_only=True)
        input_latents = vae_data["input_latents"]  # [16, T_lat, H_lat, W_lat]
        # img_lat = first frame only: [1, 16, 1, H, W]
        img_lat = input_latents[:, :1].unsqueeze(0).to(device, dtype=torch.bfloat16)

        # Audio embedding
        audio_data = torch.load(os.path.join(video_dir, "audio_emb_omniavatar.pt"), map_location="cpu", weights_only=True)
        audio_emb = audio_data["audio_emb"][:NUM_FRAMES]  # [T_video, 10752]
        audio_emb_tensor = audio_emb.unsqueeze(0).to(device, dtype=torch.bfloat16)

        # Read prompt
        prompt_path = os.path.join(video_dir, "prompt.txt")
        with open(prompt_path, "r") as f:
            prompt = f.read().strip()
        print(f"  Prompt: {prompt!r}")
        print(f"  img_lat: {img_lat.shape}, audio_emb: {audio_emb_tensor.shape}")

        # Expand img_lat to full T_lat frames (same as inference.py:306)
        # First frame = reference latent, rest = zeros
        T_lat = input_latents.shape[1]  # 21
        img_lat = torch.cat([
            img_lat,
            torch.zeros_like(img_lat[:, :, :1].repeat(1, 1, T_lat - 1, 1, 1)),
        ], dim=2)  # [1, 16, T_lat, H, W]
        print(f"  img_lat expanded: {img_lat.shape}")

        # --- Run ODE solve ---
        print(f"\nRunning ODE solve ({NUM_STEPS} steps, CFG={GUIDANCE_SCALE})...")
        t0 = time.time()
        x_0, path = generate_with_trajectory(
            pipe=pipe,
            img_lat=img_lat,
            prompt=prompt,
            audio_emb_tensor=audio_emb_tensor,
            target_timesteps=PATH_TIMESTEPS,
        )
        elapsed = time.time() - t0
        print(f"ODE solve done in {elapsed:.1f}s")
        print(f"  x_0: {x_0.shape}, path: {path.shape}")

        # --- Save ---
        os.makedirs(save_dir, exist_ok=True)
        x_0_save = x_0.squeeze(0).cpu().to(torch.bfloat16)
        path_save = path.cpu().to(torch.bfloat16)
        torch.save(path_save, os.path.join(save_dir, "path.pth"))
        torch.save(x_0_save, os.path.join(save_dir, "latent.pth"))
        print(f"\n  Saved path.pth: {path_save.shape}")
        print(f"  Saved latent.pth: {x_0_save.shape}")

        for i, t_val in enumerate(PATH_TIMESTEPS):
            p = path_save[i]
            print(f"  path[{i}] (t={t_val}): mean={p.float().mean():.4f}, std={p.float().std():.4f}")
        print(f"  x_0: mean={x_0_save.float().mean():.4f}, std={x_0_save.float().std():.4f}")

    # --- Visualize ---
    print("\nDecoding and visualizing...")
    vae_data = torch.load(os.path.join(video_dir, "vae_latents.pt"), map_location="cpu", weights_only=True)
    gt_latents = vae_data["input_latents"]

    decode_and_visualize(pipe, path_save, x_0_save, gt_latents, OUTPUT_DIR, sample_name, device)

    print(f"\nAll outputs saved to {OUTPUT_DIR}")
    print("Done!")


if __name__ == "__main__":
    main()

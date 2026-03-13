"""Test ODE pair generation on a single sample and visualize results.

Runs the FlowMatchScheduler-based ODE solve on cuda:1, saves path.pth + latent.pth,
then decodes all intermediates through VAE and saves as a visualization grid.

Usage:
    PYTHONPATH=$(pwd) python test_ode_pairs.py
"""

import os
import sys
import time

import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from PIL import Image

# Add FastGen and OmniAvatar to path
sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, "/home/work/.local/OmniAvatar")

DEVICE = torch.device("cuda:1")
CKPT_ROOT = "/home/work/.local/OmniAvatar/pretrained_models"
MASK_PATH = "/home/work/.local/OmniAvatar/OmniAvatar/utils/latentsync/mask.png"
VAE_PATH = f"{CKPT_ROOT}/Wan2.1-T2V-1.3B/Wan2.1_VAE.pth"
DATA_LIST = "/home/work/stableavatar_data/v2v_training_data/video_square_path.txt"
OUTPUT_DIR = "/home/work/.local/FastGen/tmp/ode_pairs_test"

T_LIST = [0.999, 0.937, 0.833, 0.624, 0.0]
PATH_TIMESTEPS = [t for t in T_LIST if t > 0]
LATENT_SHAPE = (16, 21, 64, 64)
NUM_FRAMES = 81
FLOW_SHIFT = 5.0
NUM_STEPS = 50
GUIDANCE_SCALE = 4.5


# ---------------------------------------------------------------------------
# FlowMatchScheduler (same as in generate_omniavatar_ode_pairs.py)
# ---------------------------------------------------------------------------
class FlowMatchScheduler:
    def __init__(self, shift=5.0, sigma_max=1.0, sigma_min=0.003 / 1.002, extra_one_step=True):
        self.num_train_timesteps = 1000
        self.shift = shift
        self.sigma_max = sigma_max
        self.sigma_min = sigma_min
        self.extra_one_step = extra_one_step

    def set_timesteps(self, num_inference_steps, device):
        sigma_start = self.sigma_min + (self.sigma_max - self.sigma_min)
        if self.extra_one_step:
            self.sigmas = torch.linspace(sigma_start, self.sigma_min, num_inference_steps + 1)[:-1]
        else:
            self.sigmas = torch.linspace(sigma_start, self.sigma_min, num_inference_steps)
        self.sigmas = self.shift * self.sigmas / (1 + (self.shift - 1) * self.sigmas)
        self.sigmas = self.sigmas.to(device)
        self.timesteps = self.sigmas * self.num_train_timesteps

    def step(self, velocity, step_idx, sample):
        sigma_cur = self.sigmas[step_idx]
        sigma_next = self.sigmas[step_idx + 1] if step_idx + 1 < len(self.sigmas) else 0.0
        return sample + velocity * (sigma_next - sigma_cur)


def sample_with_trajectory(teacher, noise, target_timesteps, guidance_scale, num_steps, shift, pos_condition, neg_condition, ref_latent=None):
    """Run ODE solve and capture intermediate states.

    Args:
        ref_latent: If provided, [B, 16, 1, H, W] reference latent to inject at each step
                    (replacing latents[:, :, :1] as in original OmniAvatar I2V inference).
    """
    scheduler = FlowMatchScheduler(shift=shift, extra_one_step=True)
    scheduler.set_timesteps(num_inference_steps=num_steps, device=noise.device)

    latents = scheduler.sigmas[0] * noise
    target_set = sorted(target_timesteps, reverse=True)
    captured = {}

    with torch.no_grad(), torch.cuda.amp.autocast(dtype=torch.bfloat16):
        for step_idx in range(len(scheduler.sigmas)):
            # Inject clean reference frame (original OmniAvatar I2V keeps frame 0 fixed)
            if ref_latent is not None:
                latents[:, :, :1] = ref_latent

            sigma = scheduler.sigmas[step_idx].item()

            for target_t in target_set:
                if target_t not in captured and sigma <= target_t:
                    captured[target_t] = latents.clone()
                    print(f"    Captured t={target_t} at step {step_idx} (sigma={sigma:.6f})")

            # Pass timestep (sigma * 1000) to the model, not raw sigma
            timestep = scheduler.timesteps[step_idx]
            t = timestep.unsqueeze(0).expand(latents.shape[0]).to(dtype=latents.dtype)
            velocity_pred = teacher(latents, t, condition=pos_condition)

            if guidance_scale > 1.0 and neg_condition is not None:
                velocity_uncond = teacher(latents, t, condition=neg_condition)
                velocity_pred = velocity_uncond + guidance_scale * (velocity_pred - velocity_uncond)

            latents = scheduler.step(velocity_pred, step_idx, latents)

    for target_t in target_set:
        if target_t not in captured:
            print(f"    WARNING: t={target_t} not captured, using final latents")
            captured[target_t] = latents.clone()

    # Final ref_latent injection after loop (matching OmniAvatar wan_video.py line 279-280)
    if ref_latent is not None:
        latents[:, :, :1] = ref_latent

    x_0 = latents
    path = torch.stack([captured[t].squeeze(0) for t in target_set], dim=0)
    return x_0, path


def decode_latents_to_frames(vae, latents, device):
    """Decode latent tensor [C, T, H, W] to video frames using Wan VAE."""
    # VAE expects [B, C, T, H, W]
    x = latents.unsqueeze(0).to(device=device, dtype=torch.float32)

    with torch.no_grad():
        # Wan VAE decode
        frames = vae.decode(x)  # [B, C, T_video, H_video, W_video]

    # Convert to [T, 3, H, W] in [0, 1]
    frames = frames.squeeze(0)  # [C, T, H, W]
    frames = (frames + 1.0) / 2.0
    frames = frames.clamp(0, 1)
    # frames is [3, T_video, H, W], transpose to [T_video, 3, H, W]
    frames = frames.permute(1, 0, 2, 3)
    return frames


def save_frame_grid(frames_dict, output_path, frame_indices=None):
    """Save a grid of frames from multiple ODE stages.

    Args:
        frames_dict: OrderedDict of {label: frames_tensor [T, 3, H, W]}
        output_path: Where to save the grid image
        frame_indices: Which frame indices to show (default: evenly spaced 5)
    """
    import torchvision

    labels = list(frames_dict.keys())
    all_frames = list(frames_dict.values())

    T = all_frames[0].shape[0]
    if frame_indices is None:
        # Pick 5 evenly spaced frames
        frame_indices = [0, T // 4, T // 2, 3 * T // 4, T - 1]

    # Build grid: rows = ODE stages, cols = frame indices
    grid_images = []
    for frames in all_frames:
        for fi in frame_indices:
            fi = min(fi, frames.shape[0] - 1)
            grid_images.append(frames[fi])

    ncol = len(frame_indices)
    grid = torchvision.utils.make_grid(grid_images, nrow=ncol, padding=2, normalize=False)

    # Convert to PIL and add labels
    grid_pil = TF.to_pil_image(grid.cpu())
    grid_pil.save(output_path)
    print(f"  Saved grid: {output_path} ({len(labels)} rows x {ncol} cols)")
    return grid_pil


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--viz-only", action="store_true", help="Skip ODE generation, just decode saved pairs")
    args = parser.parse_args()

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    torch.cuda.set_device(DEVICE)

    # --- Pick a sample ---
    with open(DATA_LIST, "r") as f:
        all_dirs = [line.strip() for line in f if line.strip()]
    video_dir = all_dirs[0]
    sample_name = os.path.basename(video_dir)
    save_dir = os.path.join(OUTPUT_DIR, sample_name)
    print(f"Sample: {sample_name}")
    print(f"Device: {DEVICE}")

    if args.viz_only:
        print("\n--viz-only: Loading saved ODE pairs...")
        path_save = torch.load(os.path.join(save_dir, "path.pth"), map_location="cpu", weights_only=True)
        x_0_save = torch.load(os.path.join(save_dir, "latent.pth"), map_location="cpu", weights_only=True)
        print(f"  path.pth: {path_save.shape}")
        print(f"  latent.pth: {x_0_save.shape}")
    else:
        # --- Load teacher ---
        print("\nLoading 14B I2V teacher...")
        t0 = time.time()

        os.environ["CKPT_ROOT_DIR"] = CKPT_ROOT
        os.environ["OMNIAVATAR_ROOT"] = CKPT_ROOT
        from fastgen.networks.OmniAvatar.network import OmniAvatarWan

        base_14b_paths = ",".join([
            f"{CKPT_ROOT}/Wan2.1-T2V-14B/diffusion_pytorch_model-0000{i}-of-00006.safetensors"
            for i in range(1, 7)
        ])
        teacher = OmniAvatarWan(
            in_dim=33,
            dim=5120, num_heads=40, ffn_dim=13824, num_layers=40,
            mode="i2v", use_audio=True, audio_hidden_size=32,
            has_image_input=False,
            base_model_paths=base_14b_paths,
            omniavatar_ckpt_path=f"{CKPT_ROOT}/OmniAvatar-14B/pytorch_model.pt",
            merge_lora=True, load_pretrained=True,
            net_pred_type="flow", schedule_type="rf",
        )
        teacher = teacher.to(device=DEVICE, dtype=torch.bfloat16)
        teacher.eval()
        for p in teacher.parameters():
            p.requires_grad_(False)
        print(f"Teacher loaded in {time.time() - t0:.1f}s, VRAM: {torch.cuda.memory_allocated(DEVICE) / (1024**3):.1f} GB")

        # --- Load sample data ---
        print("\nLoading sample data...")
        vae_data = torch.load(os.path.join(video_dir, "vae_latents.pt"), map_location="cpu", weights_only=True)
        input_latents = vae_data["input_latents"]

        audio_data = torch.load(os.path.join(video_dir, "audio_emb_omniavatar.pt"), map_location="cpu", weights_only=True)
        audio_emb = audio_data["audio_emb"][:NUM_FRAMES]

        text_emb_path = os.path.join(video_dir, "text_emb.pt")
        text_emb = torch.load(text_emb_path, map_location="cpu", weights_only=True)
        if isinstance(text_emb, dict):
            text_emb = text_emb.get("text_emb", text_emb.get("context", next(iter(text_emb.values()))))
        while text_emb.dim() > 2:
            text_emb = text_emb.squeeze(0)

        # For I2V mode: mask is all-ones for frames 1+ (generate everything)
        # The LatentSync spatial mask is only for V2V mode
        H_lat, W_lat = LATENT_SHAPE[2], LATENT_SHAPE[3]  # 64, 64
        mask = torch.ones(H_lat, W_lat, device=DEVICE, dtype=torch.float32)

        ref_latent = input_latents[:, :1].unsqueeze(0).to(DEVICE, dtype=torch.bfloat16)
        text_emb = text_emb.unsqueeze(0).to(DEVICE, dtype=torch.bfloat16)
        audio_emb = audio_emb.unsqueeze(0).to(DEVICE, dtype=torch.bfloat16)

        pos_cond = {
            "text_embeds": text_emb,
            "audio_emb": audio_emb,
            "ref_latent": ref_latent,
            "mask": mask,
        }
        neg_cond = {
            "text_embeds": torch.zeros_like(text_emb),
            "audio_emb": torch.zeros_like(audio_emb),
            "ref_latent": ref_latent,
            "mask": mask,
        }

        # --- Run ODE solve ---
        print(f"\nRunning ODE solve ({NUM_STEPS} Euler steps, CFG={GUIDANCE_SCALE}, shift={FLOW_SHIFT})...")
        noise = torch.randn(1, *LATENT_SHAPE, device=DEVICE, dtype=torch.bfloat16)

        t0 = time.time()
        x_0, path = sample_with_trajectory(
            teacher=teacher, noise=noise, target_timesteps=PATH_TIMESTEPS,
            guidance_scale=GUIDANCE_SCALE, num_steps=NUM_STEPS, shift=FLOW_SHIFT,
            pos_condition=pos_cond, neg_condition=neg_cond,
            ref_latent=ref_latent,  # Inject reference frame at each step (matching OmniAvatar I2V inference)
        )
        elapsed = time.time() - t0
        print(f"ODE solve done in {elapsed:.1f}s")
        print(f"  x_0: {x_0.shape}, path: {path.shape}")

        # --- Save path.pth and latent.pth ---
        os.makedirs(save_dir, exist_ok=True)

        x_0_save = x_0.squeeze(0).cpu().to(torch.bfloat16)
        path_save = path.cpu().to(torch.bfloat16)
        torch.save(path_save, os.path.join(save_dir, "path.pth"))
        torch.save(x_0_save, os.path.join(save_dir, "latent.pth"))
        print(f"\n  Saved path.pth: {path_save.shape}")
        print(f"  Saved latent.pth: {x_0_save.shape}")

        # Verify values are sane
        for i, t_val in enumerate(PATH_TIMESTEPS):
            p = path_save[i]
            print(f"  path[{i}] (t={t_val}): mean={p.float().mean():.4f}, std={p.float().std():.4f}, "
                  f"min={p.float().min():.4f}, max={p.float().max():.4f}")
        print(f"  x_0: mean={x_0_save.float().mean():.4f}, std={x_0_save.float().std():.4f}")

        del teacher
        torch.cuda.empty_cache()

    # --- Load GT latents for comparison ---
    vae_data = torch.load(os.path.join(video_dir, "vae_latents.pt"), map_location="cpu", weights_only=True)
    input_latents = vae_data["input_latents"]

    # --- Decode with VAE and visualize ---
    print("\nLoading VAE for visualization...")

    # Load Wan VAE using OmniAvatar's WanVideoVAE (DiffSynth-Studio)
    from OmniAvatar.models.wan_video_vae import WanVideoVAE, WanVideoVAEStateDictConverter

    vae = WanVideoVAE(z_dim=16)
    vae_sd = torch.load(VAE_PATH, map_location="cpu", weights_only=True)
    converter = WanVideoVAEStateDictConverter()
    vae_sd = converter.from_civitai(vae_sd)
    vae.load_state_dict(vae_sd)
    vae = vae.to(device=DEVICE, dtype=torch.float32)
    vae.eval()
    print(f"  VAE loaded, VRAM: {torch.cuda.memory_allocated(DEVICE) / (1024**3):.1f} GB")

    def decode_latent(latent_tensor):
        """Decode [C, T, H, W] latent to [T, 3, H, W] frames in [0, 1]."""
        with torch.no_grad():
            # WanVideoVAE.decode expects iterable of [C, T, H, W], returns [B, C, T_video, H_video, W_video]
            decoded = vae.decode([latent_tensor.float()], device=DEVICE)
        decoded = decoded.squeeze(0)  # [3, T_video, H, W]
        decoded = (decoded + 1.0) / 2.0
        decoded = decoded.clamp(0, 1)
        return decoded.permute(1, 0, 2, 3).cpu()  # [T, 3, H, W]

    print("Decoding ODE trajectory stages...")
    from collections import OrderedDict
    frames_dict = OrderedDict()

    # Decode each intermediate state
    for i, t_val in enumerate(PATH_TIMESTEPS):
        print(f"  Decoding path[{i}] (t={t_val})...")
        frames_dict[f"t={t_val}"] = decode_latent(path_save[i])

    # Decode clean x_0
    print("  Decoding x_0 (clean)...")
    frames_dict["x_0 (clean)"] = decode_latent(x_0_save)

    # Also decode the original GT latent for comparison
    print("  Decoding GT latent...")
    frames_dict["GT (original)"] = decode_latent(input_latents)

    # Save grid visualization
    print("\nSaving visualization grid...")
    grid_path = os.path.join(OUTPUT_DIR, f"{sample_name}_ode_grid.png")
    save_frame_grid(frames_dict, grid_path)

    # Also save individual frames for each stage
    for label, frames in frames_dict.items():
        label_clean = label.replace("=", "").replace(" ", "_").replace("(", "").replace(")", "")
        for fi in [0, frames.shape[0] // 2, frames.shape[0] - 1]:
            frame_path = os.path.join(OUTPUT_DIR, f"{sample_name}_{label_clean}_frame{fi:03d}.png")
            TF.to_pil_image(frames[fi]).save(frame_path)

    print(f"\nAll outputs saved to {OUTPUT_DIR}")
    print("Done!")


if __name__ == "__main__":
    main()

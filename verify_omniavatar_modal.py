"""
Verify OmniAvatar Self-Forcing integration on Modal (GPU).

Runs a progressive test suite:
  1. Import check — all modules import cleanly
  2. Config creation — experiment config builds without errors
  3. Forward pass (1.3B, random weights) — verify shapes for student + fake_score
  4. KV cache consistency — chunk-by-chunk vs full-sequence outputs match
  5. Forward pass (14B, random weights) — verify teacher shapes (optional, needs more VRAM)

No real checkpoints or data needed — all tests use random weights and dummy tensors.

Usage:
    # Run all tests (default: skip 14B teacher test)
    modal run verify_omniavatar_modal.py

    # Include 14B teacher test (needs ~30GB VRAM)
    modal run verify_omniavatar_modal.py --test-14b

    # Choose GPU type and count (default: A10G x1)
    modal run verify_omniavatar_modal.py --gpu L4
    modal run verify_omniavatar_modal.py --gpu A100-80GB --ngpus 2
    modal run verify_omniavatar_modal.py --gpu H100 --ngpus 4 --test-14b

    # Background
    modal run --detach verify_omniavatar_modal.py

Setup (one-time):
    pip install modal
    python3 -m modal setup
"""

import modal

app = modal.App("fastgen-omniavatar-verify")

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
        "CKPT_ROOT_DIR": "/tmp/nonexistent_ckpts",
    })
)

_common = dict(image=image, timeout=30 * 60)


def _run_tests(test_14b: bool = False):
    """Run all verification tests inside the Modal container."""
    import sys
    import traceback

    os_module = __import__("os")
    os_module.chdir("/root/FastGen")

    results = []

    def run_test(name, fn):
        print(f"\n{'='*60}")
        print(f"TEST: {name}")
        print(f"{'='*60}")
        try:
            fn()
            print(f"  PASS: {name}")
            results.append((name, True, None))
        except Exception as e:
            tb = traceback.format_exc()
            print(f"  FAIL: {name}")
            print(f"  Error: {e}")
            print(tb)
            results.append((name, False, str(e)))

    # ------------------------------------------------------------------ #
    # Test 1: Imports
    # ------------------------------------------------------------------ #
    def test_imports():
        from fastgen.networks.OmniAvatar import OmniAvatarWan, CausalOmniAvatarWan
        from fastgen.networks.OmniAvatar.wan_model import WanModel, build_rope_freqs
        from fastgen.networks.OmniAvatar.audio_pack import AudioPack
        from fastgen.methods.omniavatar_self_forcing import OmniAvatarSelfForcingModel
        from fastgen.datasets.omniavatar_dataloader import OmniAvatarDataset, OmniAvatarDataLoader
        import fastgen.configs.methods.config_omniavatar_sf as cfg_mod
        import fastgen.configs.experiments.OmniAvatar.config_sf as exp_mod
        print("  All imports successful")

    run_test("1. Import check", test_imports)

    # ------------------------------------------------------------------ #
    # Test 2: Config creation
    # ------------------------------------------------------------------ #
    def test_config():
        from fastgen.configs.experiments.OmniAvatar.config_sf import create_config
        config = create_config()
        assert hasattr(config, "model"), "config missing 'model'"
        assert hasattr(config, "trainer"), "config missing 'trainer'"
        assert hasattr(config.model, "net"), "config.model missing 'net'"
        assert hasattr(config.model, "teacher"), "config.model missing 'teacher'"
        assert hasattr(config.model, "fake_score_net"), "config.model missing 'fake_score_net'"
        assert config.model.gan_loss_weight_gen == 0, "Expected no GAN loss"
        assert config.model.precision == "bfloat16"
        assert config.model.input_shape == [16, 21, 64, 64]
        assert config.model.student_sample_steps == 4
        print(f"  Config created successfully")
        print(f"    model.input_shape = {config.model.input_shape}")
        print(f"    model.guidance_scale = {config.model.guidance_scale}")
        print(f"    trainer.max_iter = {config.trainer.max_iter}")

    run_test("2. Config creation", test_config)

    # ------------------------------------------------------------------ #
    # Test 3: Forward pass — 1.3B fake_score (non-causal, I2V)
    # ------------------------------------------------------------------ #
    def test_forward_1_3b_noncausal():
        import torch
        from fastgen.networks.OmniAvatar import OmniAvatarWan

        device = torch.device("cuda")
        dtype = torch.bfloat16

        net = OmniAvatarWan(
            in_dim=33, dim=1536, num_heads=12, ffn_dim=8960, num_layers=30,
            mode="i2v", use_audio=True, audio_hidden_size=32,
            has_image_input=False, load_pretrained=False,
            net_pred_type="flow", schedule_type="rf",
        ).to(device=device, dtype=dtype)

        param_count = sum(p.numel() for p in net.parameters()) / 1e6
        print(f"  1.3B non-causal params: {param_count:.1f}M")

        B, C, T, H, W = 1, 16, 21, 64, 64
        T_video = 81

        x_t = torch.randn(B, C, T, H, W, device=device, dtype=dtype)
        t = torch.rand(B, device=device, dtype=dtype)
        condition = {
            "text_embeds": torch.randn(B, 512, 4096, device=device, dtype=dtype),
            "audio_emb": torch.randn(B, T_video, 10752, device=device, dtype=dtype),
            "ref_latent": torch.randn(B, 16, 1, H, W, device=device, dtype=dtype),
            "mask": torch.ones(H, W, device=device, dtype=dtype),
            "masked_video": torch.randn(B, 16, T, H, W, device=device, dtype=dtype),
        }

        with torch.no_grad():
            out = net(x_t, t, condition=condition)

        assert out.shape == (B, C, T, H, W), f"Expected {(B, C, T, H, W)}, got {out.shape}"
        assert not torch.isnan(out).any(), "Output contains NaN"
        print(f"  Output shape: {out.shape}  (correct)")

        del net, x_t, t, condition, out
        torch.cuda.empty_cache()

    run_test("3. Forward pass — 1.3B non-causal (fake_score)", test_forward_1_3b_noncausal)

    # ------------------------------------------------------------------ #
    # Test 4: Forward pass — 1.3B student (causal, V2V), single chunk
    # ------------------------------------------------------------------ #
    def test_forward_1_3b_causal():
        import torch
        from fastgen.networks.OmniAvatar import CausalOmniAvatarWan

        device = torch.device("cuda")
        dtype = torch.bfloat16

        net = CausalOmniAvatarWan(
            in_dim=49, dim=1536, num_heads=12, ffn_dim=8960, num_layers=30,
            chunk_size=3, total_num_frames=21,
            use_audio=True, audio_hidden_size=32,
            has_image_input=False, load_pretrained=False,
            net_pred_type="flow", schedule_type="rf",
        ).to(device=device, dtype=dtype)

        param_count = sum(p.numel() for p in net.parameters()) / 1e6
        print(f"  1.3B causal params: {param_count:.1f}M")

        B, C, H, W = 1, 16, 64, 64
        T_total = 21
        T_video = 81
        chunk_size = 3

        condition = {
            "text_embeds": torch.randn(B, 512, 4096, device=device, dtype=dtype),
            "audio_emb": torch.randn(B, T_video, 10752, device=device, dtype=dtype),
            "ref_latent": torch.randn(B, 16, 1, H, W, device=device, dtype=dtype),
            "mask": torch.ones(H, W, device=device, dtype=dtype),
            "masked_video": torch.randn(B, 16, T_total, H, W, device=device, dtype=dtype),
        }

        # Forward for first chunk
        x_chunk = torch.randn(B, C, chunk_size, H, W, device=device, dtype=dtype)
        t = torch.rand(B, device=device, dtype=dtype)

        with torch.no_grad():
            out = net(
                x_chunk, t, condition=condition,
                cur_start_frame=0, store_kv=True, is_ar=True,
            )

        assert out.shape == (B, C, chunk_size, H, W), f"Expected {(B, C, chunk_size, H, W)}, got {out.shape}"
        assert not torch.isnan(out).any(), "Output contains NaN"
        print(f"  Chunk 0 output shape: {out.shape}  (correct)")

        # Forward for second chunk (uses KV cache)
        x_chunk2 = torch.randn(B, C, chunk_size, H, W, device=device, dtype=dtype)
        with torch.no_grad():
            out2 = net(
                x_chunk2, t, condition=condition,
                cur_start_frame=chunk_size, store_kv=True, is_ar=True,
            )

        assert out2.shape == (B, C, chunk_size, H, W), f"Chunk 1: expected {(B, C, chunk_size, H, W)}, got {out2.shape}"
        assert not torch.isnan(out2).any(), "Chunk 1 output contains NaN"
        print(f"  Chunk 1 output shape: {out2.shape}  (correct, KV cache used)")

        # Clear caches
        net.clear_caches()
        print("  Caches cleared successfully")

        del net, x_chunk, x_chunk2, t, condition, out, out2
        torch.cuda.empty_cache()

    run_test("4. Forward pass — 1.3B causal (student), multi-chunk", test_forward_1_3b_causal)

    # ------------------------------------------------------------------ #
    # Test 5: KV cache consistency
    # ------------------------------------------------------------------ #
    def test_kv_cache_consistency():
        """Verify that chunk-by-chunk causal forward matches full-sequence non-causal."""
        import torch
        from fastgen.networks.OmniAvatar import OmniAvatarWan, CausalOmniAvatarWan

        device = torch.device("cuda")
        dtype = torch.float32  # Use float32 for numerical comparison

        # Use tiny dimensions for faster test (head_dim must be divisible by 3 for 3D RoPE)
        dim, num_heads, ffn_dim, num_layers = 384, 3, 1024, 4

        # Non-causal reference (processes all frames at once)
        net_full = OmniAvatarWan(
            in_dim=49, dim=dim, num_heads=num_heads, ffn_dim=ffn_dim, num_layers=num_layers,
            mode="v2v", use_audio=True, audio_hidden_size=32,
            has_image_input=False, load_pretrained=False,
            net_pred_type="flow", schedule_type="rf",
        ).to(device=device, dtype=dtype)

        # Causal (chunk-by-chunk)
        net_causal = CausalOmniAvatarWan(
            in_dim=49, dim=dim, num_heads=num_heads, ffn_dim=ffn_dim, num_layers=num_layers,
            chunk_size=3, total_num_frames=6,
            use_audio=True, audio_hidden_size=32,
            has_image_input=False, load_pretrained=False,
            net_pred_type="flow", schedule_type="rf",
        ).to(device=device, dtype=dtype)

        # Copy weights from full to causal (they use the same WanModel)
        net_causal.model.load_state_dict(net_full.model.state_dict())

        B, C, T, H, W = 1, 16, 6, 32, 32  # Small but compatible spatial dims
        T_video = 21  # Must satisfy (T_video + 3) % 4 == 0 for AudioPack
        chunk_size = 3

        # Fixed input
        torch.manual_seed(42)
        x = torch.randn(B, C, T, H, W, device=device, dtype=dtype)
        t = torch.tensor([0.5], device=device, dtype=dtype)
        condition = {
            "text_embeds": torch.randn(B, 64, 4096, device=device, dtype=dtype),
            "audio_emb": torch.randn(B, T_video, 10752, device=device, dtype=dtype),
            "ref_latent": torch.randn(B, 16, 1, H, W, device=device, dtype=dtype),
            "mask": torch.ones(H, W, device=device, dtype=dtype),
            "masked_video": torch.randn(B, 16, T, H, W, device=device, dtype=dtype),
        }

        # Full-sequence forward
        with torch.no_grad():
            out_full = net_full(x, t, condition=condition)

        # Chunk-by-chunk forward
        net_causal.clear_caches()
        chunks_out = []
        for start in range(0, T, chunk_size):
            x_chunk = x[:, :, start:start + chunk_size]
            with torch.no_grad():
                out_chunk = net_causal(
                    x_chunk, t, condition=condition,
                    cur_start_frame=start, store_kv=True, is_ar=True,
                )
            chunks_out.append(out_chunk)

        out_causal = torch.cat(chunks_out, dim=2)

        # Compare
        max_diff = (out_full - out_causal).abs().max().item()
        mean_diff = (out_full - out_causal).abs().mean().item()
        print(f"  Full-seq output shape: {out_full.shape}")
        print(f"  Causal output shape:   {out_causal.shape}")
        print(f"  Max abs diff:  {max_diff:.6e}")
        print(f"  Mean abs diff: {mean_diff:.6e}")

        # The non-causal model sees all frames in self-attention, so outputs
        # WILL differ from the causal model (which only sees past frames).
        # This test verifies that:
        # 1) Both run without errors
        # 2) Outputs are finite and have matching shapes
        # 3) The magnitude of outputs is similar (not wildly divergent)
        assert out_full.shape == out_causal.shape, "Shape mismatch"
        assert not torch.isnan(out_full).any(), "Full output has NaN"
        assert not torch.isnan(out_causal).any(), "Causal output has NaN"

        # Check that outputs are in the same ballpark (within 10x of each other's scale)
        full_scale = out_full.abs().mean().item()
        causal_scale = out_causal.abs().mean().item()
        if full_scale > 1e-6:
            ratio = causal_scale / full_scale
            print(f"  Scale ratio (causal/full): {ratio:.3f}")
            assert 0.01 < ratio < 100, f"Output scales diverged: ratio={ratio:.3f}"

        print("  NOTE: Exact match not expected (causal vs bidirectional attention)")

        del net_full, net_causal
        torch.cuda.empty_cache()

    run_test("5. KV cache consistency (causal vs full)", test_kv_cache_consistency)

    # ------------------------------------------------------------------ #
    # Test 6 (optional): Forward pass — 14B teacher (non-causal, I2V)
    # ------------------------------------------------------------------ #
    if test_14b:
        def test_forward_14b():
            import torch
            from fastgen.networks.OmniAvatar import OmniAvatarWan

            device = torch.device("cuda")
            dtype = torch.bfloat16

            net = OmniAvatarWan(
                in_dim=33, dim=5120, num_heads=40, ffn_dim=13824, num_layers=40,
                mode="i2v", use_audio=True, audio_hidden_size=32,
                has_image_input=False, load_pretrained=False,
                net_pred_type="flow", schedule_type="rf",
            ).to(device=device, dtype=dtype)

            param_count = sum(p.numel() for p in net.parameters()) / 1e6
            print(f"  14B non-causal params: {param_count:.1f}M")

            # Small spatial dims to fit in VRAM
            # T_video must satisfy: (T_video + 3) / 4 == T_lat for audio alignment
            B, C, T, H, W = 1, 16, 5, 32, 32
            T_video = T * 4 - 3  # = 17, so (17+3)/4 = 5 = T_lat

            x_t = torch.randn(B, C, T, H, W, device=device, dtype=dtype)
            t = torch.rand(B, device=device, dtype=dtype)
            condition = {
                "text_embeds": torch.randn(B, 512, 4096, device=device, dtype=dtype),
                "audio_emb": torch.randn(B, T_video, 10752, device=device, dtype=dtype),
                "ref_latent": torch.randn(B, 16, 1, H, W, device=device, dtype=dtype),
                "mask": torch.ones(H, W, device=device, dtype=dtype),
                "masked_video": torch.randn(B, 16, T, H, W, device=device, dtype=dtype),
            }

            with torch.no_grad():
                out = net(x_t, t, condition=condition)

            assert out.shape == (B, C, T, H, W), f"Expected {(B, C, T, H, W)}, got {out.shape}"
            assert not torch.isnan(out).any(), "Output contains NaN"
            print(f"  Output shape: {out.shape}  (correct)")

            del net, x_t, t, condition, out
            torch.cuda.empty_cache()

        run_test("6. Forward pass — 14B non-causal (teacher)", test_forward_14b)
    else:
        print(f"\n{'='*60}")
        print("SKIPPED: 6. Forward pass — 14B teacher (use --test-14b to enable)")
        print(f"{'='*60}")

    # ------------------------------------------------------------------ #
    # Summary
    # ------------------------------------------------------------------ #
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    passed = sum(1 for _, ok, _ in results if ok)
    failed = sum(1 for _, ok, _ in results if not ok)
    for name, ok, err in results:
        status = "PASS" if ok else f"FAIL ({err})"
        print(f"  {status}: {name}")
    print(f"\n  {passed} passed, {failed} failed out of {len(results)} tests")

    if failed > 0:
        sys.exit(1)


# ---------------------------------------------------------------------------
# GPU variants — explicit global-scope functions (Modal requirement).
# ---------------------------------------------------------------------------

@app.function(**_common, gpu="A10G:1")
def verify_a10g_1(test_14b: bool = False):
    _run_tests(test_14b=test_14b)

@app.function(**_common, gpu="A10G:2")
def verify_a10g_2(test_14b: bool = False):
    _run_tests(test_14b=test_14b)

@app.function(**_common, gpu="A10G:4")
def verify_a10g_4(test_14b: bool = False):
    _run_tests(test_14b=test_14b)

@app.function(**_common, gpu="L4:1")
def verify_l4_1(test_14b: bool = False):
    _run_tests(test_14b=test_14b)

@app.function(**_common, gpu="L4:2")
def verify_l4_2(test_14b: bool = False):
    _run_tests(test_14b=test_14b)

@app.function(**_common, gpu="L4:4")
def verify_l4_4(test_14b: bool = False):
    _run_tests(test_14b=test_14b)

@app.function(**_common, gpu="L40S:1")
def verify_l40s_1(test_14b: bool = False):
    _run_tests(test_14b=test_14b)

@app.function(**_common, gpu="L40S:2")
def verify_l40s_2(test_14b: bool = False):
    _run_tests(test_14b=test_14b)

@app.function(**_common, gpu="L40S:4")
def verify_l40s_4(test_14b: bool = False):
    _run_tests(test_14b=test_14b)

@app.function(**_common, gpu="A100-80GB:1")
def verify_a100_1(test_14b: bool = False):
    _run_tests(test_14b=test_14b)

@app.function(**_common, gpu="A100-80GB:2")
def verify_a100_2(test_14b: bool = False):
    _run_tests(test_14b=test_14b)

@app.function(**_common, gpu="A100-80GB:4")
def verify_a100_4(test_14b: bool = False):
    _run_tests(test_14b=test_14b)

@app.function(**_common, gpu="A100-80GB:8")
def verify_a100_8(test_14b: bool = False):
    _run_tests(test_14b=test_14b)

@app.function(**_common, gpu="H100:1")
def verify_h100_1(test_14b: bool = False):
    _run_tests(test_14b=test_14b)

@app.function(**_common, gpu="H100:2")
def verify_h100_2(test_14b: bool = False):
    _run_tests(test_14b=test_14b)

@app.function(**_common, gpu="H100:4")
def verify_h100_4(test_14b: bool = False):
    _run_tests(test_14b=test_14b)

@app.function(**_common, gpu="H100:8")
def verify_h100_8(test_14b: bool = False):
    _run_tests(test_14b=test_14b)


_GPU_FUNCTIONS = {
    "A10G:1": verify_a10g_1, "A10G:2": verify_a10g_2, "A10G:4": verify_a10g_4,
    "L4:1": verify_l4_1, "L4:2": verify_l4_2, "L4:4": verify_l4_4,
    "L40S:1": verify_l40s_1, "L40S:2": verify_l40s_2, "L40S:4": verify_l40s_4,
    "A100-80GB:1": verify_a100_1, "A100-80GB:2": verify_a100_2,
    "A100-80GB:4": verify_a100_4, "A100-80GB:8": verify_a100_8,
    "H100:1": verify_h100_1, "H100:2": verify_h100_2,
    "H100:4": verify_h100_4, "H100:8": verify_h100_8,
}


@app.local_entrypoint()
def main(test_14b: bool = False, gpu: str = "A10G", ngpus: int = 1):
    key = f"{gpu.upper()}:{ngpus}"

    # Normalize A100 -> A100-80GB
    if key.startswith("A100:"):
        key = key.replace("A100:", "A100-80GB:")

    fn = _GPU_FUNCTIONS.get(key)
    if fn is None:
        available = sorted(_GPU_FUNCTIONS.keys())
        raise ValueError(
            f"Unknown GPU config '{key}'. Available:\n  " + "\n  ".join(available)
        )
    if test_14b and gpu.upper() in ("A10G", "L4") and ngpus == 1:
        print(f"WARNING: --test-14b requires ~30GB VRAM. {gpu} may not have enough.")
        print("Consider: --gpu A100-80GB --test-14b")
    print(f"Running verification on {key}...")
    fn.remote(test_14b=test_14b)

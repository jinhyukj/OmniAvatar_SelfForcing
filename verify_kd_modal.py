"""
Quick verification of OmniAvatar KD training pipeline on Modal.

Tests:
  1. OmniAvatarKDModel instantiation and config
  2. Forward pass + loss computation with dummy data
  3. Gradient flow (backward pass)

Uses random weights and dummy tensors — no real checkpoints needed.

Usage:
    modal run verify_kd_modal.py
    modal run verify_kd_modal.py --gpu A10G
"""

import modal

app = modal.App("fastgen-kd-verify")

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


def _run_kd_verification():
    """Run KD pipeline verification tests."""
    import os
    import sys
    import traceback

    os.chdir("/root/FastGen")

    results = []

    def run_test(name, fn):
        print(f"\n{'='*60}")
        print(f"TEST: {name}")
        print(f"{'='*60}")
        try:
            fn()
            print(f"PASS: {name}")
            results.append((name, True, ""))
        except Exception as e:
            tb = traceback.format_exc()
            print(f"FAIL: {name}\n{tb}")
            results.append((name, False, str(e)))

    # ----------------------------------------------------------------
    # Test 1: Imports and model class
    # ----------------------------------------------------------------
    def test_imports():
        from fastgen.methods.omniavatar_kd import OmniAvatarKDModel
        from fastgen.methods.knowledge_distillation.KD import KDModel
        assert issubclass(OmniAvatarKDModel, KDModel)
        print("  OmniAvatarKDModel imports and inherits from KDModel correctly")

    run_test("1. Imports", test_imports)

    # ----------------------------------------------------------------
    # Test 2: Config creation
    # ----------------------------------------------------------------
    def test_config():
        # Import the experiment config
        from fastgen.configs.experiments.OmniAvatar.config_kd import create_config
        config = create_config()

        assert config.model.student_sample_steps == 4
        assert config.model.sample_t_cfg.t_list == [0.999, 0.937, 0.833, 0.624, 0.0]
        assert config.model.precision == "bfloat16"
        print(f"  student_sample_steps: {config.model.student_sample_steps}")
        print(f"  t_list: {config.model.sample_t_cfg.t_list}")
        print(f"  input_shape: {config.model.input_shape}")
        print(f"  net_optimizer.lr: {config.model.net_optimizer.lr}")

    run_test("2. Config creation", test_config)

    # ----------------------------------------------------------------
    # Test 3: KD model forward + loss with dummy data (tiny network)
    # ----------------------------------------------------------------
    def test_kd_forward_and_loss():
        import torch
        from fastgen.methods.omniavatar_kd import OmniAvatarKDModel
        from fastgen.networks.OmniAvatar.network import OmniAvatarWan
        from fastgen.utils import LazyCall as L, instantiate
        from fastgen.configs.config import BaseModelConfig
        import attrs

        device = torch.device("cuda")

        # Create a tiny V2V student (random weights, no checkpoint loading)
        @attrs.define(slots=False)
        class TestModelConfig(BaseModelConfig):
            pass

        config = TestModelConfig()
        config.net = L(OmniAvatarWan)(
            in_dim=49, dim=384, num_heads=3, ffn_dim=1024, num_layers=4,
            mode="v2v", use_audio=True, audio_hidden_size=32,
            has_image_input=False,
            base_model_paths="", omniavatar_ckpt_path="",
            load_pretrained=False,
            net_pred_type="flow", schedule_type="rf",
        )
        config.input_shape = [16, 5, 8, 8]  # tiny: C, T_lat, H_lat, W_lat
        config.student_sample_steps = 4
        config.sample_t_cfg.t_list = [0.999, 0.937, 0.833, 0.624, 0.0]
        config.precision = "bfloat16"
        config.enable_preprocessors = False
        config.pretrained_model_path = ""
        config.load_student_weights = False
        config.pretrained_student_net_path = ""

        # Build model
        model = OmniAvatarKDModel(config)
        model.build_model()
        model.net.to(device=device, dtype=torch.bfloat16)
        print(f"  Model built, net params: {sum(p.numel() for p in model.net.parameters()):,}")

        # Create dummy data batch
        B, C, T, H, W = 2, 16, 5, 8, 8
        T_video = 17  # T_video = 4*T_lat - 3 for AudioPack alignment

        data = {
            "real": torch.randn(B, C, T, H, W, device=device, dtype=torch.bfloat16),
            "path": torch.randn(B, 4, C, T, H, W, device=device, dtype=torch.bfloat16),
            "condition": torch.randn(B, 512, 4096, device=device, dtype=torch.bfloat16),
            "neg_condition": torch.zeros(B, 512, 4096, device=device, dtype=torch.bfloat16),
            "audio_emb": torch.randn(B, T_video, 10752, device=device, dtype=torch.bfloat16),
            "ref_latent": torch.randn(B, C, 1, H, W, device=device, dtype=torch.bfloat16),
            "masked_video": torch.randn(B, C, T, H, W, device=device, dtype=torch.bfloat16),
            "mask": torch.ones(H, W, device=device, dtype=torch.bfloat16),
        }

        # Run single train step
        loss_map, outputs = model.single_train_step(data, iteration=0)

        total_loss = loss_map["total_loss"]
        print(f"  total_loss: {total_loss.item():.6f}")
        print(f"  recon_loss: {loss_map['recon_loss'].item():.6f}")
        assert total_loss.item() > 0, "Loss should be positive"
        assert torch.isfinite(total_loss), "Loss should be finite"

    run_test("3. KD forward + loss", test_kd_forward_and_loss)

    # ----------------------------------------------------------------
    # Test 4: Gradient flow (backward pass)
    # ----------------------------------------------------------------
    def test_gradient_flow():
        import torch
        from fastgen.methods.omniavatar_kd import OmniAvatarKDModel
        from fastgen.networks.OmniAvatar.network import OmniAvatarWan
        from fastgen.utils import LazyCall as L
        from fastgen.configs.config import BaseModelConfig
        import attrs

        device = torch.device("cuda")

        @attrs.define(slots=False)
        class TestModelConfig(BaseModelConfig):
            pass

        config = TestModelConfig()
        config.net = L(OmniAvatarWan)(
            in_dim=49, dim=384, num_heads=3, ffn_dim=1024, num_layers=4,
            mode="v2v", use_audio=True, audio_hidden_size=32,
            has_image_input=False,
            base_model_paths="", omniavatar_ckpt_path="",
            load_pretrained=False,
            net_pred_type="flow", schedule_type="rf",
        )
        config.input_shape = [16, 5, 8, 8]
        config.student_sample_steps = 4
        config.sample_t_cfg.t_list = [0.999, 0.937, 0.833, 0.624, 0.0]
        config.precision = "bfloat16"
        config.enable_preprocessors = False
        config.pretrained_model_path = ""
        config.load_student_weights = False
        config.pretrained_student_net_path = ""

        model = OmniAvatarKDModel(config)
        model.build_model()
        model.net.to(device=device, dtype=torch.bfloat16)

        B, C, T, H, W = 1, 16, 5, 8, 8
        T_video = 17  # T_video = 4*T_lat - 3 for AudioPack alignment

        data = {
            "real": torch.randn(B, C, T, H, W, device=device, dtype=torch.bfloat16),
            "path": torch.randn(B, 4, C, T, H, W, device=device, dtype=torch.bfloat16),
            "condition": torch.randn(B, 512, 4096, device=device, dtype=torch.bfloat16),
            "neg_condition": torch.zeros(B, 512, 4096, device=device, dtype=torch.bfloat16),
            "audio_emb": torch.randn(B, T_video, 10752, device=device, dtype=torch.bfloat16),
            "ref_latent": torch.randn(B, C, 1, H, W, device=device, dtype=torch.bfloat16),
            "masked_video": torch.randn(B, C, T, H, W, device=device, dtype=torch.bfloat16),
            "mask": torch.ones(H, W, device=device, dtype=torch.bfloat16),
        }

        # Zero grads
        for p in model.net.parameters():
            if p.grad is not None:
                p.grad.zero_()

        loss_map, outputs = model.single_train_step(data, iteration=0)
        loss = loss_map["total_loss"]
        loss.backward()

        # Check that gradients exist
        grad_params = sum(1 for p in model.net.parameters() if p.grad is not None and p.grad.abs().sum() > 0)
        total_params = sum(1 for p in model.net.parameters() if p.requires_grad)
        print(f"  Params with gradients: {grad_params}/{total_params}")
        assert grad_params > 0, "No gradients found!"

        # Check gradients are finite
        for name, p in model.net.named_parameters():
            if p.grad is not None:
                assert torch.isfinite(p.grad).all(), f"Non-finite gradient in {name}"
        print("  All gradients are finite")

    run_test("4. Gradient flow", test_gradient_flow)

    # ----------------------------------------------------------------
    # Test 5: Multi-step training (loss decreases)
    # ----------------------------------------------------------------
    def test_multi_step_training():
        import torch
        from fastgen.methods.omniavatar_kd import OmniAvatarKDModel
        from fastgen.networks.OmniAvatar.network import OmniAvatarWan
        from fastgen.utils import LazyCall as L
        from fastgen.configs.config import BaseModelConfig
        import attrs

        device = torch.device("cuda")

        @attrs.define(slots=False)
        class TestModelConfig(BaseModelConfig):
            pass

        config = TestModelConfig()
        config.net = L(OmniAvatarWan)(
            in_dim=49, dim=384, num_heads=3, ffn_dim=1024, num_layers=4,
            mode="v2v", use_audio=True, audio_hidden_size=32,
            has_image_input=False,
            base_model_paths="", omniavatar_ckpt_path="",
            load_pretrained=False,
            net_pred_type="flow", schedule_type="rf",
        )
        config.input_shape = [16, 5, 8, 8]
        config.student_sample_steps = 4
        config.sample_t_cfg.t_list = [0.999, 0.937, 0.833, 0.624, 0.0]
        config.precision = "bfloat16"
        config.enable_preprocessors = False
        config.pretrained_model_path = ""
        config.load_student_weights = False
        config.pretrained_student_net_path = ""

        model = OmniAvatarKDModel(config)
        model.build_model()
        model.net.to(device=device, dtype=torch.bfloat16)

        optimizer = torch.optim.Adam(model.net.parameters(), lr=1e-3)

        B, C, T, H, W = 2, 16, 5, 8, 8
        T_video = 17  # T_video = 4*T_lat - 3 for AudioPack alignment

        # Use fixed data so we can observe overfitting (loss decrease)
        torch.manual_seed(42)
        fixed_data = {
            "real": torch.randn(B, C, T, H, W, device=device, dtype=torch.bfloat16),
            "path": torch.randn(B, 4, C, T, H, W, device=device, dtype=torch.bfloat16),
            "condition": torch.randn(B, 512, 4096, device=device, dtype=torch.bfloat16),
            "neg_condition": torch.zeros(B, 512, 4096, device=device, dtype=torch.bfloat16),
            "audio_emb": torch.randn(B, T_video, 10752, device=device, dtype=torch.bfloat16),
            "ref_latent": torch.randn(B, C, 1, H, W, device=device, dtype=torch.bfloat16),
            "masked_video": torch.randn(B, C, T, H, W, device=device, dtype=torch.bfloat16),
            "mask": torch.ones(H, W, device=device, dtype=torch.bfloat16),
        }

        losses = []
        for step in range(20):
            optimizer.zero_grad()
            loss_map, _ = model.single_train_step(dict(fixed_data), iteration=step)
            loss = loss_map["total_loss"]
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
            if step % 5 == 0:
                print(f"  Step {step}: loss={loss.item():.6f}")

        print(f"  Initial loss: {losses[0]:.6f}")
        print(f"  Final loss:   {losses[-1]:.6f}")

        # Loss should decrease when overfitting on fixed data
        assert losses[-1] < losses[0], f"Loss did not decrease: {losses[0]:.6f} -> {losses[-1]:.6f}"
        print(f"  Loss decreased by {(1 - losses[-1]/losses[0])*100:.1f}%")

    run_test("5. Multi-step training (loss decrease)", test_multi_step_training)

    # ----------------------------------------------------------------
    # Summary
    # ----------------------------------------------------------------
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    passed = sum(1 for _, ok, _ in results if ok)
    total = len(results)
    for name, ok, err in results:
        status = "PASS" if ok else f"FAIL: {err}"
        print(f"  {status} - {name}")
    print(f"\n{passed}/{total} tests passed")

    if passed < total:
        sys.exit(1)


@app.function(image=image, gpu="L4", timeout=15 * 60)
def verify_kd():
    _run_kd_verification()


@app.local_entrypoint()
def main(gpu: str = "L4"):
    verify_kd.remote()

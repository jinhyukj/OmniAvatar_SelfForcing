"""
Launch FastGen T2V training on Modal.

Usage:
    # FSDP (recommended for 14B teacher)
    modal run train_modal_t2v.py --fsdp

    # DDP
    modal run train_modal_t2v.py

    # Background (survives terminal close)
    modal run --detach train_modal_t2v.py --fsdp

Setup (one-time):
    pip install modal
    python3 -m modal setup
    modal secret create wandb-secret WANDB_API_KEY=your_key_here
    modal volume create fastgen-data
    modal volume create fastgen-output

    # Upload data and model checkpoints
    modal volume put fastgen-data /path/to/t2v_wds_shards/ /t2v_wds_shards/
    modal volume put fastgen-data /path/to/fastgen_models/ /models/
"""

import modal

app = modal.App("fastgen-selfforcing-t2v")

# Persistent volumes
data_vol = modal.Volume.from_name("fastgen-data", create_if_missing=True)
output_vol = modal.Volume.from_name("fastgen-output", create_if_missing=True)

# Container image with FastGen installed
image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("git", "ffmpeg")
    .pip_install(
        "torch==2.10.0",
        "torchvision",
        "transformers==4.49.0",
        "diffusers==0.35.1",
        "accelerate",
        "wandb",
        "webdataset",
        "omegaconf",
        "loguru",
        "attrs",
        "einops",
        "numpy<2.0.0",
        "psutil",
        "pandas",
    )
    # Mount FastGen codebase (copy=True allows run_commands after)
    .add_local_dir(".", remote_path="/root/FastGen", copy=True, ignore=[
        ".git", "__pycache__", "*.pyc", "third_party",
        "FASTGEN_OUTPUT", "runs", "tmp", ".claude",
    ])
    .run_commands("cd /root/FastGen && pip install -e .")
    .env({
        "FASTGEN_OUTPUT_ROOT": "/mnt/output",
        "DATA_ROOT_DIR": "/mnt/data",
        "CKPT_ROOT_DIR": "/mnt/data/models",
        "WANDB_ENTITY": "jhjangbot-korea-advanced-institute-of-science-and-technology",
    })
)


def _run_training(ngpus: int, fsdp: bool):
    """Run torchrun inside the Modal container."""
    import os
    from torch.distributed.run import parse_args, run

    os.chdir("/root/FastGen")

    train_args = [
        "trainer.batch_size_global=64",
        "trainer.logging_iter=1",
        "dataloader_train.batch_size=1",
        'dataloader_train.datatags=["WDS:/mnt/data/v2v_wds_shards"]',
    ]

    if fsdp:
        train_args.append("trainer.fsdp=True")
        train_args.append("log_config.name=wan_sf_14b_teacher_fsdp_modal")
    else:
        train_args.append("log_config.name=wan_sf_14b_teacher_modal")

    args = [
        "--standalone",
        "--nnodes=1",
        f"--nproc-per-node={ngpus}",
        "train.py",
        "--config=fastgen/configs/experiments/WanT2V/config_sf_14b_teacher.py",
        "-",
        *train_args,
    ]
    run(parse_args(args))


# --- DDP variant ---
@app.function(
    image=image,
    gpu="H200:8",
    volumes={"/mnt/data": data_vol, "/mnt/output": output_vol},
    secrets=[modal.Secret.from_name("wandb-secret")],
    timeout=24 * 3600,
)
def train_ddp(ngpus: int = 8):
    _run_training(ngpus=ngpus, fsdp=False)


# --- FSDP variant ---
@app.function(
    image=image,
    gpu="H200:8",
    volumes={"/mnt/data": data_vol, "/mnt/output": output_vol},
    secrets=[modal.Secret.from_name("wandb-secret")],
    timeout=24 * 3600,
)
def train_fsdp(ngpus: int = 8):
    _run_training(ngpus=ngpus, fsdp=True)


@app.local_entrypoint()
def main(fsdp: bool = False, ngpus: int = 8):
    if fsdp:
        train_fsdp.remote(ngpus=ngpus)
    else:
        train_ddp.remote(ngpus=ngpus)

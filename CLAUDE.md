# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

FastGen is a PyTorch framework for building fast generative models using distillation and acceleration techniques. It supports large-scale training (≥10B params), multiple modalities (Text-to-Image, Image-to-Video, Video-to-Video), and methods including Consistency Models, Distribution Matching (DMD2), Self-Forcing, and Knowledge Distillation.

## Common Commands

```bash
make install           # Install linters (ruff, mypy, black)
make install-fastgen   # pip install -e .
make format            # Auto-format with ruff (excludes third_party/)
make lint              # Check style: ruff format --check + ruff check (excludes third_party/)
make mypy              # Type check: mypy --check-untyped-defs (excludes third_party/)
make pytest            # Run tests: pytest --ignore=FASTGEN_OUTPUT --ignore=runs --ignore=tmp --ignore=third_party
```

Run a single test: `python3 -m pytest tests/test_file.py::test_name`

Ruff line-length is 120 (configured in pyproject.toml). Commits must be signed with `-s` flag.

## Architecture

**Entry point**: `train.py` → instantiates model → initializes `Trainer` → runs training loop.

### Core layers (top-down):

1. **Trainer** (`fastgen/trainer.py`) — Orchestrates training lifecycle: checkpointing, auto-resume, callback management, distributed setup (DDP and FSDP2). Key method: `run()`.

2. **Methods** (`fastgen/methods/`) — Training algorithms, all inherit from `FastGenModel` (`model.py`). Methods are decoupled from network architectures.
   - `consistency_model/`: CM, sCM (JVP-based), TCM, MeanFlow
   - `distribution_matching/`: DMD2, f-Distill, LADD, CausVid, SelfForcing
   - `fine_tuning/`: SFT (standard + causal flow matching)
   - `knowledge_distillation/`: KD (standard + causal)

3. **Networks** (`fastgen/networks/`) — Architecture implementations, all inherit from `FastGenNetwork` (`network.py`). Image: EDM, EDM2, DiT, SD15, SDXL, Flux. Video: WAN, WanI2V, VaceWan, CogVideoX, Cosmos.
   - `noise_schedule.py`: Unified noise scheduling (edm, sd, sdxl, rf, cogvideox, trig)
   - `discriminators.py`: GAN discriminators for adversarial methods

4. **Callbacks** (`fastgen/callbacks/`) — Lifecycle hooks inheriting from `Callback` (`callback.py`): EMA, WandB, gradient clipping, GPU stats, profiling, weight normalization, curriculum schedules.

5. **Datasets** (`fastgen/datasets/`) — Class-conditional and WebDataset loaders with flexible key mapping. Supports precomputed latents/embeddings.

6. **Configs** (`fastgen/configs/`) — Hierarchical Python configs using attrs + Hydra + OmegaConf. Uses `LazyCall` pattern for deferred instantiation. Experiment configs live in `configs/experiments/`.

7. **Utils** (`fastgen/utils/`) — `instantiate()` for dynamic object creation from configs, `LazyCall` wrapper, distributed utilities (DDP/FSDP), checkpointing (local + S3), LR schedulers, logging (loguru).

### Key design patterns:
- **Methods are network-agnostic**: Same training algorithm works across EDM, SDXL, WAN, Cosmos, etc.
- **Configuration as code**: Python-based configs with attrs dataclasses, OmegaConf CLI overrides
- **Callback-driven lifecycle**: Training behavior (EMA, logging, profiling) decoupled via callbacks
- **`fastgen/third_party/`** is excluded from all linting, formatting, and type checks

## Environment Variables

| Variable | Purpose |
|----------|---------|
| `FASTGEN_OUTPUT_ROOT` | Output directory for checkpoints/logs |
| `DATA_ROOT_DIR` | Dataset root (default: `$FASTGEN_OUTPUT_ROOT/DATA`) |
| `CKPT_ROOT_DIR` | Pretrained checkpoint root (default: `$FASTGEN_OUTPUT_ROOT/MODEL`) |
| `HF_HOME` | HuggingFace cache (default: `$FASTGEN_OUTPUT_ROOT/.cache`) |
| `WANDB_API_KEY` | Weights & Biases API key |

## Inference

Inference scripts live in `scripts/inference/` for T2I, T2V, I2V, and V2V generation tasks.

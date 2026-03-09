# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import os
from datetime import datetime
from typing import TYPE_CHECKING, Any, Callable

import torch

from fastgen.callbacks.callback import Callback
from fastgen.utils.distributed import is_rank0

if TYPE_CHECKING:
    from fastgen.methods import FastGenModel


def _local_bytes(t: torch.Tensor) -> int:
    """Get actual local memory bytes of a tensor, handling FSDP2 DTensors.

    FSDP2 stores parameters as DTensor objects where .nelement() returns the
    global (unsharded) count. The actual local shard is in ._local_tensor.
    """
    if hasattr(t, "_local_tensor"):
        t = t._local_tensor
    return t.nelement() * t.element_size()


def _resolve_module(obj: Any) -> torch.nn.Module | None:
    """Extract the nn.Module from an object (handles preprocessor wrappers like WanVideoEncoder)."""
    if isinstance(obj, torch.nn.Module):
        return obj
    # Preprocessor wrappers store the actual nn.Module as an attribute
    for attr in ("vae", "text_encoder", "image_encoder", "model"):
        inner = getattr(obj, attr, None)
        if isinstance(inner, torch.nn.Module):
            return inner
    return None


def _module_param_bytes(module: Any, exclude_ids: set[int] | None = None) -> int:
    """Sum bytes of all parameters in a module, optionally excluding by param data_ptr id."""
    mod = _resolve_module(module)
    if mod is None:
        return 0
    total = 0
    for p in mod.parameters():
        if exclude_ids and id(p.data) in exclude_ids:
            continue
        total += _local_bytes(p.data)
    return total


def _optimizer_state_bytes(optimizer: torch.optim.Optimizer) -> int:
    """Sum bytes of all tensors stored in optimizer state (momentum, variance, etc.)."""
    total = 0
    for state in optimizer.state.values():
        for v in state.values():
            if isinstance(v, torch.Tensor):
                total += _local_bytes(v)
    return total


def _trainable_param_bytes(module: Any) -> int:
    """Sum bytes of parameters that have requires_grad=True."""
    mod = _resolve_module(module)
    if mod is None:
        return 0
    total = 0
    for p in mod.parameters():
        if p.requires_grad:
            total += _local_bytes(p.data)
    return total


def _module_buffer_bytes(module: Any, exclude_ids: set[int] | None = None) -> int:
    """Sum bytes of all registered buffers in a module (non-parameter persistent tensors)."""
    mod = _resolve_module(module)
    if mod is None:
        return 0
    total = 0
    for b in mod.buffers():
        if exclude_ids and id(b.data) in exclude_ids:
            continue
        if b.device.type == "cuda":
            total += _local_bytes(b)
    return total


def _gb(b: int | float) -> str:
    """Format bytes as GB string with 2 decimal places."""
    return f"{b / (1024**3):.2f} GB"


class VRAMReportCallback(Callback):
    """Produces a detailed VRAM breakdown report to a dedicated crash-safe log file.

    The report accounts for every byte on the GPU:
      Params + Optimizer States + Gradients + Activations = Peak Allocated
      Peak Allocated + Allocator Overhead = Peak Reserved

    Measurements are taken at multiple lifecycle points within each training step
    to capture both the steady-state and peak transient memory.
    """

    def __init__(self, every_n: int = 100):
        self.every_n = every_n
        self._file = None
        self._tracking = False
        self._after_forward: int = 0

    def on_train_begin(self, model: FastGenModel, iteration: int = 0) -> None:
        if not is_rank0():
            return
        if hasattr(self, "config"):
            self.every_n = self.config.trainer.logging_iter

        save_path = self.config.log_config.save_path
        os.makedirs(save_path, exist_ok=True)
        filepath = os.path.join(save_path, "vram_report.log")
        self._file = open(filepath, "a")  # noqa: SIM115
        self._write(
            f"VRAM Report initialized at {datetime.now().isoformat()}\n"
            f"  FSDP: {self.config.trainer.fsdp}\n"
            f"  Precision: {self.config.model.precision}\n"
            f"  Logging every {self.every_n} iterations\n"
        )

    def on_training_step_begin(self, model: FastGenModel, iteration: int = 0) -> None:
        if not is_rank0():
            return
        if iteration % self.every_n == 0:
            self._tracking = True
            self._after_forward = 0

    def on_backward_begin(
        self,
        model: FastGenModel,
        data_batch: dict[str, torch.Tensor],
        output_batch: dict[str, torch.Tensor | Callable],
        loss_dict: dict[str, torch.Tensor],
        iteration: int = 0,
        accum_iter: int = 0,
    ) -> None:
        if not is_rank0():
            return
        if self._tracking:
            self._after_forward = max(self._after_forward, torch.cuda.memory_allocated())

    def on_training_step_end(
        self,
        model: FastGenModel,
        data_batch: dict[str, torch.Tensor],
        output_batch: dict[str, torch.Tensor | Callable],
        loss_dict: dict[str, torch.Tensor],
        iteration: int = 0,
    ) -> None:
        del data_batch, output_batch, loss_dict
        if not is_rank0() or not self._tracking:
            return
        self._tracking = False

        # --- Peak and current memory from CUDA ---
        peak_backward = torch.cuda.max_memory_allocated()
        step_peak = max(self._after_forward, peak_backward)
        current_allocated = torch.cuda.memory_allocated()
        peak_reserved = torch.cuda.max_memory_reserved()

        # --- Determine training phase ---
        student_update_freq = getattr(getattr(model, "config", None), "student_update_freq", 1)
        is_student_phase = (iteration % student_update_freq == 0)
        phase = "Student Update" if is_student_phase else "Critic+Disc Update"

        # --- Discover components and collect preprocessor param ids to avoid double-counting ---
        preprocessor_param_ids: set[int] = set()
        preprocessors: dict[str, Any] = {}
        for prep_attr, prep_label in [("vae", "VAE"), ("text_encoder", "Text Encoder"), ("image_encoder", "Image Encoder")]:
            prep_obj = getattr(model.net, prep_attr, None)
            if prep_obj is None:
                continue
            prep_mod = _resolve_module(prep_obj)
            if prep_mod is not None:
                preprocessors[prep_label] = prep_obj
                preprocessor_param_ids.update(id(p.data) for p in prep_mod.parameters())

        # --- Discover all components (for params, buffers, grads) ---
        components: list[tuple[str, Any]] = []
        components.append(("Student (net)", model.net))
        if getattr(model, "teacher", None) is not None:
            components.append(("Teacher", model.teacher))
        if hasattr(model, "fake_score"):
            components.append(("Critic (fake_score)", model.fake_score))
        if hasattr(model, "discriminator"):
            components.append(("Discriminator", model.discriminator))
        for ema_name in getattr(model, "use_ema", []):
            ema_mod = getattr(model, ema_name, None)
            if ema_mod is not None:
                components.append((f"EMA ({ema_name})", ema_mod))
        for prep_name, prep_obj in preprocessors.items():
            components.append((prep_name, prep_obj))

        # --- Per-component parameter memory ---
        param_rows: list[tuple[str, int]] = []
        for comp_name, comp_obj in components:
            exclude = preprocessor_param_ids if comp_name == "Student (net)" else None
            param_rows.append((comp_name, _module_param_bytes(comp_obj, exclude_ids=exclude)))

        total_params = sum(b for _, b in param_rows)

        # --- Per-component buffer memory ---
        buffer_rows: list[tuple[str, int]] = []
        for comp_name, comp_obj in components:
            exclude = preprocessor_param_ids if comp_name == "Student (net)" else None
            buffer_rows.append((comp_name, _module_buffer_bytes(comp_obj, exclude_ids=exclude)))

        total_buffers = sum(b for _, b in buffer_rows)

        # --- Per-optimizer state memory ---
        optim_map: dict[str, str] = {
            "net": "Student (net)",
            "fake_score": "Critic (fake_score)",
            "discriminator": "Discriminator",
        }
        optim_rows: list[tuple[str, int, str]] = []  # (optimizer_name, bytes, component_label)
        optimizer_dict = model.optimizer_dict if hasattr(model, "optimizer_dict") else {}
        for opt_name, opt in optimizer_dict.items():
            label = optim_map.get(opt_name, opt_name)
            optim_rows.append((opt_name, _optimizer_state_bytes(opt), label))

        total_optim = sum(b for _, b, _ in optim_rows)

        # --- Analytical gradient memory for active phase ---
        grad_rows: list[tuple[str, int, str]] = []  # (component, bytes, annotation)
        if is_student_phase:
            gb = _trainable_param_bytes(model.net)
            grad_rows.append(("Student (net)", gb, "active (student update)"))
            if hasattr(model, "fake_score"):
                grad_rows.append(("Critic (fake_score)", 0, "frozen"))
            if hasattr(model, "discriminator"):
                grad_rows.append(("Discriminator", 0, "frozen"))
        else:
            grad_rows.append(("Student (net)", 0, "frozen"))
            if hasattr(model, "fake_score"):
                gb = _trainable_param_bytes(model.fake_score)
                grad_rows.append(("Critic (fake_score)", gb, "active (critic update)"))
            if hasattr(model, "discriminator"):
                gb = _trainable_param_bytes(model.discriminator)
                grad_rows.append(("Discriminator", gb, "active (disc update)"))

        total_grads = sum(b for _, b, _ in grad_rows)

        # --- "Other" = current_allocated minus everything we can account for ---
        accounted_steady = total_params + total_optim + total_buffers
        other = max(0, current_allocated - accounted_steady)

        # --- Compute activations as remainder ---
        steady_state = accounted_steady + other  # == current_allocated (by construction)
        activations = max(0, step_peak - steady_state - total_grads)
        allocator_overhead = max(0, peak_reserved - step_peak)

        # --- Build report ---
        sep = "=" * 80
        thin_sep = "-" * 58
        lines: list[str] = []
        lines.append("")
        lines.append(sep)
        lines.append(f"VRAM Report | Iteration {iteration} | Phase: {phase} | Rank 0")
        lines.append(sep)

        # Steady state table
        lines.append("")
        lines.append("STEADY STATE (always on GPU)")
        lines.append(
            f"  {'Component':<24} {'Params':>12} {'Buffers':>12} {'Optim States':>14} {'Total':>12}"
        )

        # Build lookups for optimizer and buffer bytes by component label
        optim_by_component: dict[str, int] = {}
        for _, ob, label in optim_rows:
            optim_by_component[label] = optim_by_component.get(label, 0) + ob
        buffer_by_component: dict[str, int] = dict(buffer_rows)

        for comp_name, pb in param_rows:
            ob = optim_by_component.get(comp_name, 0)
            bb = buffer_by_component.get(comp_name, 0)
            ob_str = _gb(ob) if ob > 0 else "\u2014"
            bb_str = _gb(bb) if bb > 0 else "\u2014"
            total = pb + bb + ob
            lines.append(
                f"  {comp_name:<24} {_gb(pb):>12} {bb_str:>12} {ob_str:>14} {_gb(total):>12}"
            )

        # "Other" row for FSDP internals, data tensors, misc
        dash = "\u2014"
        if other > 0:
            lines.append(
                f"  {'Other (FSDP/data/misc)':<24} {dash:>12} {dash:>12} {dash:>14} {_gb(other):>12}"
            )

        lines.append(f"  {thin_sep}")
        lines.append(
            f"  {'Subtotal':<24} {_gb(total_params):>12} {_gb(total_buffers):>12}"
            f" {_gb(total_optim):>14} {_gb(steady_state):>12}"
        )

        # Peak transient
        lines.append("")
        lines.append("PEAK TRANSIENT (created during forward/backward, freed after)")
        for comp_name, gb_val, annotation in grad_rows:
            arrow = "\u2190 " + annotation
            lines.append(f"  {'Gradients (' + comp_name + ')':<40} {_gb(gb_val):>12}  {arrow}")
        lines.append(
            f"  {'Activations+Temps':<40} {_gb(activations):>12}  "
            f"\u2190 forward pass intermediates for backward"
        )
        lines.append(f"  {thin_sep}")
        lines.append(f"  {'Subtotal':<40} {_gb(total_grads + activations):>12}")

        # Summary
        lines.append("")
        lines.append("SUMMARY")
        lines.append(f"  Params                : {_gb(total_params):>12}")
        lines.append(f"  + Buffers             : {_gb(total_buffers):>12}")
        lines.append(f"  + Optimizer States    : {_gb(total_optim):>12}")
        lines.append(f"  + Other (FSDP/data)   : {_gb(other):>12}")
        lines.append(f"  = Steady State        : {_gb(steady_state):>12}  (= Current Allocated)")
        lines.append(f"  + Gradients           : {_gb(total_grads):>12}  (backpropagation)")
        lines.append(f"  + Activations         : {_gb(activations):>12}  (forward pass intermediates)")
        lines.append(f"  = Peak Allocated      : {_gb(step_peak):>12}  (max during this step)")
        lines.append(f"  + Allocator Overhead  : {_gb(allocator_overhead):>12}  (CUDA caching/fragmentation)")
        lines.append(f"  = Peak Reserved       : {_gb(peak_reserved):>12}")
        lines.append(f"  Current Allocated     : {_gb(current_allocated):>12}  (measured at report time)")
        lines.append(sep)
        lines.append("")

        self._write("\n".join(lines))

    def on_app_end(self, model: FastGenModel, iteration: int = 0) -> None:
        if self._file is not None:
            self._write(f"Training ended at {datetime.now().isoformat()}, iteration {iteration}\n")
            self._file.close()
            self._file = None

    def _write(self, text: str) -> None:
        if self._file is not None:
            self._file.write(text)
            self._file.flush()
            os.fsync(self._file.fileno())

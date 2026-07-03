from __future__ import annotations

import json
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path  # noqa: TC003
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from torch.nn import Module

import torch


@dataclass
class ActivationStats:
    """Online running statistics for a tensor stream (Welford for mean/var)."""

    n: int = 0
    mean: float = 0.0
    m2: float = 0.0
    abs_min: float = float("inf")
    abs_max: float = 0.0
    near_zero_count: int = 0
    total_count: int = 0
    sparsity_threshold: float = 1e-6

    def update(self, x: torch.Tensor) -> None:
        x = x.detach().to(torch.float32).reshape(-1)
        n = x.numel()
        if n == 0:
            return

        batch_mean = x.mean().item()
        batch_var = x.var(unbiased=False).item() if n > 1 else 0.0

        new_n = self.n + n
        delta = batch_mean - self.mean
        self.mean += delta * n / new_n
        self.m2 += batch_var * n + delta * delta * self.n * n / new_n
        self.n = new_n

        self.abs_min = min(self.abs_min, x.abs().min().item())
        self.abs_max = max(self.abs_max, x.abs().max().item())
        self.near_zero_count += int((x.abs() < self.sparsity_threshold).sum().item())
        self.total_count += n

    def summary(self) -> dict[str, Any]:
        if self.n == 0:
            return {"n": 0}
        variance = self.m2 / self.n
        return {
            "n_samples": self.n,
            "mean": self.mean,
            "std": variance**0.5,
            "abs_min": self.abs_min,
            "abs_max": self.abs_max,
            "sparsity": self.near_zero_count / max(self.total_count, 1),
        }


@dataclass
class RoutingStats:
    """Per-expert routing decisions."""

    expert_hits: dict[int, int] = field(default_factory=lambda: defaultdict(int))
    token_top_k_unique_hist: dict[int, int] = field(default_factory=lambda: defaultdict(int))
    total_tokens: int = 0
    num_experts: int = 0

    def update(self, top_k_index: torch.Tensor, num_experts: int) -> None:
        self.num_experts = num_experts
        if top_k_index.numel() == 0:
            return

        flat = top_k_index.detach().reshape(-1).cpu().to(torch.long)
        self.total_tokens += int(flat.numel())

        for eid in flat.tolist():
            self.expert_hits[int(eid)] += 1

        if top_k_index.dim() == 2:
            sorted_, _ = torch.sort(top_k_index.detach().cpu().to(torch.long), dim=1)
            row_unique = (sorted_[:, 1:] != sorted_[:, :-1]).sum(dim=1) + 1
            for u in row_unique.tolist():
                self.token_top_k_unique_hist[int(u)] += 1

    def summary(self) -> dict[str, Any]:
        hits = [self.expert_hits.get(i, 0) for i in range(self.num_experts)]
        active = sum(1 for h in hits if h > 0)
        return {
            "total_routing_decisions": self.total_tokens,
            "num_experts": self.num_experts,
            "active_experts": active,
            "expert_hits": hits,
            "expert_hit_min": min(hits) if hits else 0,
            "expert_hit_max": max(hits) if hits else 0,
            "expert_hit_mean": (sum(hits) / len(hits)) if hits else 0.0,
            "top_k_unique_histogram": dict(self.token_top_k_unique_hist),
        }


class Measurements:
    """Aggregates routing and activation stats across forward passes."""

    def __init__(self, num_experts: int = 0):
        self.routing = RoutingStats(num_experts=num_experts)
        self.activations: dict[str, ActivationStats] = {}

    def update_routing(self, top_k_index: torch.Tensor, num_experts: int) -> None:
        self.routing.update(top_k_index, num_experts)

    def update_activation(self, name: str, tensor: torch.Tensor) -> None:
        if name not in self.activations:
            self.activations[name] = ActivationStats()
        self.activations[name].update(tensor)

    def summary(self) -> dict[str, Any]:
        return {
            "routing": self.routing.summary(),
            "activations": {
                name: stats.summary() for name, stats in sorted(self.activations.items())
            },
        }

    def dump_json(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.summary(), indent=2))


def _routing_pre_hook(module: Module, inputs: tuple[Any, ...], measurements: Measurements) -> None:
    """Pre-hook on CodebookExperts: captures top_k_index (second positional arg)."""
    if len(inputs) >= 2:
        top_k_index = inputs[1]
        if isinstance(top_k_index, torch.Tensor):
            measurements.update_routing(top_k_index, module.num_experts)


def _activation_forward_hook(
    module: Module, inputs: tuple[Any, ...], output: Any, name: str, measurements: Measurements
) -> None:
    """Forward hook: captures the module's output tensor."""
    tensor = output if isinstance(output, torch.Tensor) else None
    if tensor is not None:
        measurements.update_activation(name, tensor)


def install_measurement_hooks(
    model: Module, measurements: Measurements
) -> list[torch.utils.hooks.RemovableHook]:
    """Install forward hooks on every MoE layer's gate/up/intermediate/down.

    Returns the list of hook handles so callers can remove them later.
    """
    from skcq.eval_model import get_text_model

    handles: list[torch.utils.hooks.RemovableHook] = []
    text_model = get_text_model(model)

    for layer_idx, layer in enumerate(text_model.layers):
        experts = layer.mlp.experts

        targets = [
            ("gate", getattr(experts, "gate", None)),
            ("up", getattr(experts, "up", None)),
            ("intermediate", getattr(experts, "intermediate", None)),
            ("down", getattr(experts, "down", None)),
        ]
        for proj_name, module in targets:
            if module is None:
                continue
            handles.append(
                module.register_forward_hook(
                    lambda mod, inp, out, n=f"L{layer_idx}.{proj_name}": _activation_forward_hook(
                        mod, inp, out, n, measurements
                    )
                )
            )

        handles.append(
            experts.register_forward_pre_hook(
                lambda mod, inp: _routing_pre_hook(mod, inp, measurements)
            )
        )

    return handles


def remove_hooks(handles: list[torch.utils.hooks.RemovableHook]) -> None:
    for h in handles:
        h.remove()

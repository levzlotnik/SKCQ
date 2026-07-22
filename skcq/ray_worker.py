"""Ray remote worker actor for distributed codebook building.

Replaces the standalone TCP worker with a Ray remote actor that can be
launched with `ray start --head` and run tasks via `WorkerActor.remote(...)`.

Requires PYTHONPATH to include the project root for skcq.clustering import.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import ray
import torch
from huggingface_hub import snapshot_download
from safetensors import safe_open

from skcq.clustering import CodebookParams, CodebookResult
from skcq.experiment import CodebookConfig, CodebookExperiment

logger = logging.getLogger(__name__)

LayerShardMap = dict[int, dict[str, str]]
ProjectionRows = dict[str, torch.Tensor]


def _extract_layer_idx(key: str) -> int | None:
    """Extract layer index from a safetensors key like 'model.language_model.layers.5.mlp...'."""
    parts = key.split(".")
    for i, p in enumerate(parts):
        if p == "layers" and i + 1 < len(parts):
            try:
                return int(parts[i + 1])
            except ValueError:
                pass
    return None


def build_layer_shard_map(model_dir: Path) -> LayerShardMap:
    """Parse model.safetensors.index.json → {layer_idx: {gate_up: shard, down: shard}}."""
    index_path = model_dir / "model.safetensors.index.json"
    if not index_path.exists():
        raise FileNotFoundError(f"No safetensors index at {index_path}")

    with open(index_path) as f:
        idx = json.load(f)

    weight_map: dict[str, str] = idx["weight_map"]
    layer_shards: LayerShardMap = {}

    for key, shard in weight_map.items():
        if not key.startswith("model.language_model.layers."):
            continue
        if ".mlp.experts.gate_up_proj" in key:
            ln = _extract_layer_idx(key)
            if ln is not None:
                layer_shards.setdefault(ln, {})["gate_up"] = shard
        elif ".mlp.experts.down_proj" in key:
            ln = _extract_layer_idx(key)
            if ln is not None:
                layer_shards.setdefault(ln, {})["down"] = shard

    return layer_shards


@ray.remote(num_gpus=1, max_restarts=3, max_task_retries=2)
class WorkerActor:
    """Ray remote actor that builds codebooks for MoE layers.

    The actor holds the model directory and layer→shard map as persistent
    state. Each `process_layer` task extracts rows from safetensors shards
    and builds PQ codebooks for gate, up, and down projections.
    """

    def __init__(
        self,
        model_id: str,
        device: str = "auto",
        chunk_budget_mb: int = 2048,
    ) -> None:
        """Initialize the worker actor.

        Args:
            model_id: HuggingFace model identifier (e.g. Qwen/Qwen3.6-35B-A3B)
            device: Device to use (cuda, mps, cpu, or auto)
            chunk_budget_mb: Memory budget (MB) for k-means chunking
        """
        logger.info("WorkerActor initializing for model=%s, device=%s", model_id, device)

        model_dir = Path(snapshot_download(model_id))
        self.model_dir = model_dir
        logger.info("Model directory: %s", model_dir)

        self.layer_shards_map = build_layer_shard_map(model_dir)
        logger.info("Found %d layers in safetensors index", len(self.layer_shards_map))

        self.device = device
        self.chunk_budget_mb = chunk_budget_mb

    def process_layer(
        self,
        layer: int,
        params: dict[str, Any],
        num_experts: int,
        hidden_size: int,
        intermediate_size: int,
    ) -> dict[str, CodebookResult]:
        """Build codebooks for one layer's gate, up, and down projections.

        Args:
            layer: layer index
            params: CodebookParams kwargs (k_gate, k_up, k_down, n_blocks_gate_up,
                    n_blocks_down, n_codebooks, max_iters, norm_threshold, skip_zeros,
                    residual_k)
            num_experts: number of MoE experts
            hidden_size: model hidden dimension
            intermediate_size: MoE intermediate dimension

        Returns:
            Dict with "gate", "up", "down" keys, each containing a CodebookResult
            with all tensors moved to CPU.
        """
        logger.info("WorkerActor: processing layer %d", layer)

        shards = self.layer_shards_map[layer]
        proj_rows: ProjectionRows = {}

        gate_up_path = self.model_dir / shards["gate_up"]
        with safe_open(gate_up_path, framework="pt", device="cpu") as f:
            gu_key = f"model.language_model.layers.{layer}.mlp.experts.gate_up_proj"
            gate_up = f.get_tensor(gu_key)

        gate_rows = gate_up[:, :intermediate_size, :].reshape(-1, hidden_size)
        up_rows = gate_up[:, intermediate_size:, :].reshape(-1, hidden_size)
        proj_rows["gate"] = gate_rows
        proj_rows["up"] = up_rows

        down_path = self.model_dir / shards["down"]
        with safe_open(down_path, framework="pt", device="cpu") as f:
            dn_key = f"model.language_model.layers.{layer}.mlp.experts.down_proj"
            down = f.get_tensor(dn_key)

        down_rows = down.reshape(-1, intermediate_size)
        proj_rows["down"] = down_rows

        device = resolve_device(self.device)

        cb_params = CodebookParams(**params)
        cb_params.chunk_budget_mb = self.chunk_budget_mb

        results: dict[str, CodebookResult] = {}

        projections = [
            (
                "gate",
                proj_rows["gate"],
                params["k_gate"],
                params["n_blocks_gate_up"],
                intermediate_size,
            ),
            (
                "up",
                proj_rows["up"],
                params["k_up"],
                params["n_blocks_gate_up"],
                intermediate_size,
            ),
            (
                "down",
                proj_rows["down"],
                params["k_down"],
                params["n_blocks_down"],
                hidden_size,
            ),
        ]

        for name, rows, k, n_blocks, out_dim in projections:
            cb_name = f"L{layer}.{name}"
            logger.info(
                "Layer %d: building %s (k=%d, nb=%d, cb=%d)...",
                layer,
                name,
                k,
                n_blocks,
                cb_params.n_codebooks,
            )
            results[name] = CodebookExperiment(
                CodebookConfig(
                    params=cb_params,
                    k=k,
                    n_blocks=n_blocks,
                    n_codebooks=cb_params.n_codebooks,
                    num_experts=num_experts,
                    out_dim=out_dim,
                    device=device,
                    name=cb_name,
                )
            ).fit(rows=rows.to(device))

        logger.info("Layer %d: done.", layer)
        return results


def resolve_device(device: str) -> torch.device:
    if device == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(device)

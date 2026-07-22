"""Extract weight rows from model and build codebooks.

Delegates clustering to skcq.clustering, adds logging + CodebookParams integration.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping

import torch

from skcq.clustering import CodebookResult
from skcq.config import CodebookParams, LayerOverride
from skcq.eval_model import get_text_model
from skcq.experiment import CodebookConfig, CodebookExperiment
from skcq.rocm_client import RocmClient

logger = logging.getLogger(__name__)


def extract_and_build_codebooks(
    model: torch.nn.Module,
    params: CodebookParams | None = None,
    layer_overrides: Mapping[int, LayerOverride] | None = None,
    device: torch.device | None = None,
    progress: bool = True,
    cuda_worker: RocmClient | None = None,
) -> list[dict[str, CodebookResult]]:
    if params is None:
        params = CodebookParams()

    config = model.config
    if hasattr(config, "text_config"):
        config = config.text_config

    intermediate_size: int = config.moe_intermediate_size
    hidden_size: int = config.hidden_size
    num_experts: int = config.num_experts

    results: list[dict[str, CodebookResult]] = []

    text_model = get_text_model(model)
    layers = text_model.layers

    for layer_idx, layer in enumerate(layers):
        if progress:
            logger.info("Processing layer %d/%d", layer_idx, len(layers))

        layer_params = params
        if layer_overrides and layer_idx in layer_overrides:
            layer_params = params.model_copy(
                update={
                    k: v
                    for k, v in layer_overrides[layer_idx].model_dump().items()
                    if v is not None
                }
            )

        mlp = layer.mlp
        gate_up = mlp.experts.gate_up_proj.data
        down = mlp.experts.down_proj.data

        gate_rows = gate_up[:, :intermediate_size, :].reshape(-1, hidden_size)
        up_rows = gate_up[:, intermediate_size:, :].reshape(-1, hidden_size)
        down_rows = down.reshape(-1, intermediate_size)

        projections = [
            (
                "gate",
                gate_rows,
                layer_params.k_gate,
                layer_params.n_blocks_gate_up,
                intermediate_size,
            ),
            (
                "up",
                up_rows,
                layer_params.k_up,
                layer_params.n_blocks_gate_up,
                intermediate_size,
            ),
            (
                "down",
                down_rows,
                layer_params.k_down,
                layer_params.n_blocks_down,
                hidden_size,
            ),
        ]

        layer_result: dict[str, CodebookResult] = {}
        for name, rows, k, n_blocks, out_dim in projections:
            cb_name = f"L{layer_idx}.{name}"
            if cuda_worker is not None:
                layer_result[name] = cuda_worker.build_codebook(
                    rows,
                    params=layer_params,
                    k=k,
                    n_blocks=n_blocks,
                    n_codebooks=layer_params.n_codebooks,
                    num_experts=num_experts,
                    out_dim=out_dim,
                    name=cb_name,
                )
            else:
                layer_result[name] = CodebookExperiment(
                    CodebookConfig(
                        params=layer_params,
                        k=k,
                        n_blocks=n_blocks,
                        n_codebooks=layer_params.n_codebooks,
                        num_experts=num_experts,
                        out_dim=out_dim,
                        device=device,
                        name=cb_name,
                    )
                ).fit(rows)

        results.append(layer_result)

    return results

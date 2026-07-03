from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from torch.nn import Module

import torch
import torch.nn.functional as F  # noqa: N812

from skcq.eval_model import get_text_model

logger = logging.getLogger(__name__)


@dataclass
class RoutingCapture:
    """Per-layer routing decisions captured across one or more forward passes.

    Each entry in ``top_k_index`` / ``top_k_weights`` corresponds to one MoE
    forward call and has shape (total_tokens, top_k). ``layer_index`` records
    which layer produced each entry (a layer may appear multiple times if the
    experts module is called more than once per pass).
    """

    layer_index: list[int] = field(default_factory=list)
    top_k_index: list[torch.Tensor] = field(default_factory=list)
    top_k_weights: list[torch.Tensor] = field(default_factory=list)

    def reset(self) -> None:
        self.layer_index.clear()
        self.top_k_index.clear()
        self.top_k_weights.clear()

    def num_calls(self) -> int:
        return len(self.layer_index)

    def per_layer_index_tensors(self) -> dict[int, list[torch.Tensor]]:
        grouped: dict[int, list[torch.Tensor]] = {}
        for layer_idx, idx in zip(self.layer_index, self.top_k_index, strict=True):
            grouped.setdefault(layer_idx, []).append(idx)
        return grouped


def install_routing_capture(
    model: Module,
) -> tuple[RoutingCapture, list[torch.utils.hooks.RemovableHook]]:
    """Register pre-hooks on every MoE experts module to capture routing inputs.

    Works for both the original ``Qwen3_5MoeExperts`` and the codebook
    replacement, since both take ``(hidden_states, top_k_index, top_k_weights)``
    as positional arguments.
    """
    capture = RoutingCapture()
    handles: list[torch.utils.hooks.RemovableHook] = []
    text_model = get_text_model(model)

    for layer_idx, layer in enumerate(text_model.layers):
        experts = layer.mlp.experts

        def _hook(
            _mod: Module,
            inputs: tuple,
            _capture: RoutingCapture = capture,
            _layer_idx: int = layer_idx,
        ) -> None:
            if len(inputs) < 2 or not isinstance(inputs[1], torch.Tensor):
                return
            _capture.layer_index.append(_layer_idx)
            _capture.top_k_index.append(inputs[1].detach().cpu())
            if len(inputs) >= 3 and isinstance(inputs[2], torch.Tensor):
                _capture.top_k_weights.append(inputs[2].detach().cpu())
            else:
                _capture.top_k_weights.append(torch.empty(0))

        handles.append(experts.register_forward_pre_hook(_hook))

    return capture, handles


def remove_hooks(handles: list[torch.utils.hooks.RemovableHook]) -> None:
    for h in handles:
        h.remove()


def routing_agreement(base: RoutingCapture, quant: RoutingCapture) -> dict[str, object]:
    """Compare routing decisions between two captures.

    Aggregates per-layer agreement of the routed expert *sets* (order-independent).
    ``set_agreement`` is the fraction of tokens whose top-k expert set is identical.
    Also reports mean top-k weight cosine similarity and JSD of the per-expert
    routing-weight distributions.
    """
    base_by_layer = base.per_layer_index_tensors()
    quant_by_layer = quant.per_layer_index_tensors()

    common_layers = sorted(set(base_by_layer) & set(quant_by_layer))
    total_tokens = 0
    set_match_tokens = 0
    weight_cos_sum = 0.0
    weight_cos_count = 0
    jsd_sum = 0.0
    jsd_count = 0
    per_layer: list[dict[str, object]] = []

    for layer in common_layers:
        base_idx = torch.cat(base_by_layer[layer], dim=0)
        quant_idx = torch.cat(quant_by_layer[layer], dim=0)
        if base_idx.shape != quant_idx.shape:
            logger.warning(
                "routing layer %d: shape mismatch %s vs %s, skipping",
                layer,
                tuple(base_idx.shape),
                tuple(quant_idx.shape),
            )
            continue

        n = base_idx.shape[0]
        total_tokens += n

        # Order-independent set agreement: sort each row's expert indices and compare.
        base_sorted = base_idx.sort(dim=-1).values
        quant_sorted = quant_idx.sort(dim=-1).values
        match = (base_sorted == quant_sorted).all(dim=-1)
        set_match_tokens += int(match.sum().item())

        # Per-token weight cosine similarity (ordered by sorted expert ids).
        base_w = _gather_sorted_weights(base, layer, base_idx)
        quant_w = _gather_sorted_weights(quant, layer, quant_idx)
        if base_w is not None and quant_w is not None and base_w.shape == quant_w.shape:
            cos = F.cosine_similarity(base_w, quant_w, dim=-1)
            weight_cos_sum += float(cos.sum().item())
            weight_cos_count += int(cos.numel())

            # JSD between normalized weight distributions for this token's routed experts.
            jsd = _jsd_rows(base_w, quant_w)
            jsd_sum += float(jsd.sum().item())
            jsd_count += int(jsd.numel())

        per_layer.append(
            {
                "layer": layer,
                "tokens": n,
                "set_agreement": float(match.float().mean().item()) if n else 0.0,
            }
        )

    summary: dict[str, object] = {
        "common_layers": len(common_layers),
        "total_tokens": total_tokens,
        "set_agreement": (set_match_tokens / total_tokens) if total_tokens else 0.0,
        "weight_cosine": (weight_cos_sum / weight_cos_count) if weight_cos_count else 0.0,
        "routing_jsd": (jsd_sum / jsd_count) if jsd_count else 0.0,
        "per_layer": per_layer,
    }
    return summary


def _gather_sorted_weights(
    capture: RoutingCapture, layer: int, idx_sorted_by_row: torch.Tensor
) -> torch.Tensor | None:
    """Return weights aligned to the row-sorted expert order of ``idx_sorted_by_row``.

    Reconstructs the per-call weight tensors for ``layer`` and gathers them so
    that row i's weights are ordered to match ``idx_sorted_by_row[i]``.
    """
    weight_calls = [
        w for li, w in zip(capture.layer_index, capture.top_k_weights, strict=True) if li == layer
    ]
    if not weight_calls or weight_calls[0].numel() == 0:
        return None
    weights = torch.cat(weight_calls, dim=0)
    if weights.shape[0] != idx_sorted_by_row.shape[0]:
        return None

    sorted_idx = idx_sorted_by_row.argsort(dim=-1, stable=True)
    return weights.gather(dim=-1, index=sorted_idx)


def _jsd_rows(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Symmetric JSD per row for non-negative weight tensors."""
    a = a.clamp_min(0)
    b = b.clamp_min(0)
    a = a / a.sum(dim=-1, keepdim=True).clamp_min(1e-12)
    b = b / b.sum(dim=-1, keepdim=True).clamp_min(1e-12)
    m = 0.5 * (a + b)
    return 0.5 * (_kl_rows(a, m) + _kl_rows(b, m))


def _kl_rows(p: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
    return (p * ((p + 1e-12).log() - (q + 1e-12).log())).sum(dim=-1)


@torch.no_grad()
def capture_reference_logits(
    model: Module, input_ids: torch.Tensor, max_length: int = 2048
) -> torch.Tensor:
    """Return next-token logits for the first ``max_length`` positions.

    Shape: (min(max_length, seq_len), vocab). Stored in fp32 on the model device.
    """
    chunk = input_ids[:, :max_length]
    outputs = model(chunk)
    logits = outputs.logits[:, :-1].float()
    return logits.squeeze(0)


@torch.no_grad()
def kld_against_reference(
    model: Module, input_ids: torch.Tensor, reference_logits: torch.Tensor
) -> float:
    """Mean KL(reference || model) over tokens, computed on the reference chunk.

    Chunked over the token dimension to avoid OOM on low-VRAM GPUs.
    """
    chunk_len = reference_logits.shape[0] + 1
    chunk = input_ids[:, :chunk_len]
    outputs = model(chunk)
    logits = outputs.logits[:, :-1].float().squeeze(0)

    n = min(reference_logits.shape[0], logits.shape[0])
    ref = reference_logits[:n].to(logits.device)
    hyp = logits[:n].to(logits.device)

    total_kl = 0.0
    token_chunk = 256
    for i in range(0, n, token_chunk):
        j = min(i + token_chunk, n)
        log_p = F.log_softmax(ref[i:j], dim=-1)
        log_q = F.log_softmax(hyp[i:j], dim=-1)
        p = log_p.exp()
        kl = (p * (log_p - log_q)).sum(dim=-1)
        total_kl += float(kl.sum().item())

    return total_kl / n


@torch.no_grad()
def token_log_likelihoods(model: Module, input_ids: torch.Tensor) -> torch.Tensor:
    """Per-token log-likelihood under the model, summed over the sequence.

    Returns (seq_len-1,) tensor of log p(token_t | context) on CPU.
    """
    outputs = model(input_ids)
    logits = outputs.logits[:, :-1].float()
    targets = input_ids[:, 1:]
    log_probs = F.log_softmax(logits, dim=-1)
    token_lp = log_probs.gather(dim=-1, index=targets.unsqueeze(-1)).squeeze(-1)
    return token_lp.squeeze(0).cpu()

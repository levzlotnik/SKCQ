from __future__ import annotations

import logging

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

logger = logging.getLogger("skcq")


def get_text_model(model: torch.nn.Module) -> torch.nn.Module:
    """Get the text submodel, handling both CausalLM and ConditionalGeneration."""
    if hasattr(model, "model") and hasattr(model.model, "language_model"):
        return model.model.language_model  # type: ignore[union-attr]
    return model.model  # type: ignore[union-attr]


def load_model(model_id: str, device: str = "auto") -> tuple[torch.nn.Module, object]:
    """Load the Qwen3.5 MoE model.

    On UMA systems, loads to CPU first then transfers to GPU. The .to("cuda")
    call on UMA should just remap the same physical pages (zero-copy) rather
    than allocating new GPU memory through HIP's buggy allocator.
    """
    logger.info("Loading tokenizer for %s...", model_id)
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)

    logger.info("Loading model weights for %s (device=%s)...", model_id, device)
    if device == "auto" and torch.cuda.is_available():
        # Load to CPU first — avoids HIP allocator issues on UMA APUs
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            device_map="cpu",
            dtype=torch.bfloat16,
            trust_remote_code=True,
        )
        model.eval()
        # On UMA, .to("cuda") remaps existing physical pages — no clone
        logger.info("Transferring model to cuda...")
        model = model.to(device="cuda")
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            device_map=device,
            dtype=torch.bfloat16,
            trust_remote_code=True,
        )
        model.eval()

    logger.info("Model loaded.")
    return model, tokenizer


def compute_perplexity(
    model: torch.nn.Module,
    tokenizer: object,
    text: str,
    max_length: int = 2048,
    stride: int = 512,
) -> float:
    """Compute perplexity using a sliding window approach."""
    encodings = tokenizer(text, return_tensors="pt")  # type: ignore[operator]
    input_ids = encodings.input_ids.to(model.device)  # type: ignore[union-attr]
    seq_len = input_ids.size(1)

    nlls: list[torch.Tensor] = []
    num_tokens = 0
    num_windows = (seq_len + stride - 1) // stride
    logger.info(
        "Perplexity: %d tokens, %d windows (max_length=%d, stride=%d)",
        seq_len,
        num_windows,
        max_length,
        stride,
    )

    for begin_loc in range(0, seq_len, stride):
        end_loc = min(begin_loc + max_length, seq_len)
        input_chunk = input_ids[:, begin_loc:end_loc]
        target_chunk = input_chunk.clone()

        # Mask tokens outside the current window
        if begin_loc > 0:
            target_chunk[:, :-stride] = -100

        with torch.no_grad():
            outputs = model(input_chunk, labels=target_chunk)
            neg_log_likelihood = outputs.loss * input_chunk.size(1)

        nlls.append(neg_log_likelihood)
        num_tokens += input_chunk.size(1)
        window_idx = begin_loc // stride
        if window_idx % 10 == 0 or end_loc >= seq_len:
            logger.info(
                "Perplexity: window %d/%d (%.0f%%)",
                window_idx + 1,
                num_windows,
                100 * (window_idx + 1) / num_windows,
            )

        if end_loc >= seq_len:
            break

    avg_nll = torch.stack(nlls).sum() / num_tokens
    return avg_nll.exp().item()


def get_calibration_text(num_samples: int = 1000) -> str:
    """Load calibration text from C4 validation set."""
    from datasets import load_dataset  # type: ignore[import-untyped]

    logger.info("Loading %d calibration samples from C4...", num_samples)
    ds = load_dataset("allenai/c4", "en", split="validation", streaming=True)
    texts: list[str] = []
    for i, sample in enumerate(ds):
        if i >= num_samples:
            break
        texts.append(sample["text"])
        if (i + 1) % 200 == 0:
            logger.info("Calibration: %d/%d samples", i + 1, num_samples)

    logger.info("Calibration: %d samples loaded (%d chars)", len(texts), sum(len(t) for t in texts))
    return "\n".join(texts)

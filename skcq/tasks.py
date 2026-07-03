from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from torch.nn import Module

import torch
import torch.nn.functional as F  # noqa: N812

logger = logging.getLogger(__name__)


@dataclass
class TaskResult:
    task: str
    num_examples: int
    accuracy: float
    correct: int
    baseline_accuracy: float | None = None


@dataclass
class ChoiceExample:
    context: str
    choices: list[str]
    label: int


@torch.no_grad()
def score_continuation(
    model: Module,
    tokenizer: Any,
    context: str,
    continuation: str,
    max_length: int = 2048,
) -> float:
    """Summed log-likelihood of ``continuation`` tokens given ``context``.

    Tokens are computed by encoding context+continuation together and splitting
    at the context length, which is robust to tokenization boundary effects
    shared between context and continuation.
    """
    full = context + continuation
    full_ids = tokenizer(full, return_tensors="pt").input_ids.to(model.device)
    ctx_ids = tokenizer(context, return_tensors="pt").input_ids.to(model.device)

    ctx_len = ctx_ids.shape[1]
    if ctx_len >= full_ids.shape[1]:
        return float("-inf")

    full_ids = full_ids[:, :max_length]
    outputs = model(full_ids)
    logits = outputs.logits[:, :-1].float()
    targets = full_ids[:, 1:]

    log_probs = F.log_softmax(logits, dim=-1)
    cont_lp = log_probs[0, ctx_len - 1 :].gather(
        dim=-1, index=targets[0, ctx_len - 1 :].unsqueeze(-1)
    )
    return float(cont_lp.sum().item())


def evaluate_multiple_choice(
    model: Module,
    tokenizer: Any,
    examples: list[ChoiceExample],
    task: str,
    max_examples: int | None = None,
    max_length: int = 2048,
    progress_every: int = 50,
) -> TaskResult:
    """Accuracy over a multiple-choice task by per-choice continuation loglikelihood."""
    if max_examples is not None:
        examples = examples[:max_examples]
    correct = 0
    total = 0
    for i, ex in enumerate(examples):
        scores = [
            score_continuation(model, tokenizer, ex.context, c, max_length=max_length)
            for c in ex.choices
        ]
        pred = int(max(range(len(scores)), key=lambda j: scores[j]))
        if pred == ex.label:
            correct += 1
        total += 1
        if progress_every and (i + 1) % progress_every == 0:
            logger.info("[%s] %d/%d acc=%.4f", task, i + 1, total, correct / total)

    acc = correct / total if total else 0.0
    return TaskResult(task=task, num_examples=total, accuracy=acc, correct=correct)


def load_mmlu(num_examples: int | None = None, subject: str = "all") -> list[ChoiceExample]:
    """Load MMLU auxiliary_train-free test split (cais/mmlu)."""
    from datasets import load_dataset

    config = subject if subject != "all" else "all"
    logger.info("Loading MMLU (subject=%s, max=%s)...", config, num_examples)
    ds = load_dataset("cais/mmlu", config, split="test")

    examples: list[ChoiceExample] = []
    for sample in ds:
        question = sample["question"]
        choices = sample["choices"]
        label = int(sample["answer"])
        ctx = (
            "The following is a multiple choice question. "
            "Answer with the letter of the correct choice.\n\n"
            f"Question: {question}\n"
            "A. {0}\nB. {1}\nC. {2}\nD. {3}\n"
            "Answer:"
        ).format(*choices)
        conts = [f" {letter}" for letter in "ABCD"]
        examples.append(ChoiceExample(context=ctx, choices=conts, label=label))
        if num_examples is not None and len(examples) >= num_examples:
            break
    logger.info("MMLU: %d examples loaded", len(examples))
    return examples


def load_hellaswag(num_examples: int | None = None) -> list[ChoiceExample]:
    """Load HellaSwag validation split (Rowan/hellaswag)."""
    from datasets import load_dataset

    logger.info("Loading HellaSwag (max=%s)...", num_examples)
    ds = load_dataset("Rowan/hellaswag", split="validation")

    examples: list[ChoiceExample] = []
    for sample in ds:
        ctx = sample["ctx"]
        endings = sample["endings"]
        label = int(sample["label"])
        conts = [f" {e}" for e in endings]
        examples.append(ChoiceExample(context=ctx, choices=conts, label=label))
        if num_examples is not None and len(examples) >= num_examples:
            break
    logger.info("HellaSwag: %d examples loaded", len(examples))
    return examples


_TASK_LOADERS = {
    "mmlu": load_mmlu,
    "hellaswag": load_hellaswag,
}


def load_task(task: str, num_examples: int | None = None) -> list[ChoiceExample]:
    if task not in _TASK_LOADERS:
        raise ValueError(f"Unknown task {task!r}. Available: {list(_TASK_LOADERS)}")
    return _TASK_LOADERS[task](num_examples=num_examples)


def run_tasks(
    model: Module,
    tokenizer: Any,
    tasks: list[str],
    num_examples: int | None = None,
    max_length: int = 2048,
) -> list[TaskResult]:
    results: list[TaskResult] = []
    for task in tasks:
        logger.info("Loading task %s...", task)
        examples = load_task(task, num_examples=num_examples)
        logger.info("Evaluating %s on %d examples...", task, len(examples))
        result = evaluate_multiple_choice(
            model, tokenizer, examples, task=task, max_length=max_length
        )
        logger.info(
            "[%s] accuracy: %.4f (%d/%d)",
            task,
            result.accuracy,
            result.correct,
            result.num_examples,
        )
        results.append(result)
    return results

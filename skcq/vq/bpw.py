"""Bits-per-weight accounting for k-means codebook quantization.

Single source of truth — imported by both:

- ``skcq.vq.runner`` (the actual sweep executor, after a codebook is built)
- ``skcq.vq.hyperparams`` (the search-space enumerator, which needs bpw
  estimates to filter configs by the user's target bpw range *without*
  importing torch / huggingface)

Keeping this in a standalone pure-Python module avoids a circular import
(runner -> hyperparams for VQConfig, hyperparams -> runner for bpw) and
kills the previous duplicate-formula drift (the euclidean-no-scale fix
had to be applied in two places; now it's one).

Accounting per codebook ``c`` (block size ``bs_c``, codebook size ``K_c``):
  - codebook storage: ``n_cb * K_c * bs_c * codebook_bits`` where ``n_cb`` is
    1 for a shared codebook or ``n_blocks_c`` for per-block.
  - assignment indices: ``n_rows * n_blocks_c * ceil(log2(K_c))`` bits (1 bit
    when ``K_c <= 1``).
  - raw (bf16) remainder columns: ``16 * n_rows * (in_dim - n_blocks_c*bs_c)``.
  - sign bits (SSVQ): ``n_rows * cov_c`` when sign-split is on for this codebook.

Primary-only:
  - scale storage: ``n_rows * n_blocks_0 * scale_bits_per_elem`` bits, **only
    when the primary metric is ``cosine``**. Euclidean primary has no scale
    (centroids are the direct reconstruction), so it contributes 0 bits.
"""

from __future__ import annotations

import math


def bits_per_weight_kmeans(
    n_rows: int,
    in_dim: int,
    n_blocks: int,
    block_size: int,
    n_codebooks: int,
    k_per_codebook: list[int],
    shared_codebook: bool = False,
    sign_split: bool = False,
    scale_bits_per_elem: int = 16,
    bs_per_codebook: list[int] | None = None,
    codebook_bits: int = 16,
    residual_sign_split: bool | list[bool] | None = None,
    primary_metric: str = "cosine",
) -> float:
    """Compute effective bits per weight for k-means quantization.

    Accounts per codebook for: codebook storage, assignment indices, the raw
    (bf16) remainder columns (in_dim % bs_c), and — for any codebook with
    sign-split enabled — 1 sign bit per covered element (Σ_c n_rows·cov_c).
    Scales are primary-only and only charged when the primary metric is
    ``cosine`` (euclidean primary has no scale — centroids ARE the
    reconstruction). ``sign_split`` is the primary flag;
    ``residual_sign_split`` (bool or per-residual list) covers c>=1.
    """
    if bs_per_codebook is None:
        bs_per_codebook = [block_size] * n_codebooks

    # Per-codebook sign-split flags: primary + residuals.
    if residual_sign_split is None:
        res_ss = [False] * (n_codebooks - 1)
    elif isinstance(residual_sign_split, bool):
        res_ss = [residual_sign_split] * (n_codebooks - 1)
    else:
        res_ss = list(residual_sign_split)[: n_codebooks - 1]
        res_ss += [False] * (n_codebooks - 1 - len(res_ss))
    ss_per_codebook = [sign_split, *res_ss]

    codebook_bits_total = 0
    assign_bits = 0
    remainder_bits = 0
    sign_bits = 0
    for c, (bs_c, k_c) in enumerate(zip(bs_per_codebook, k_per_codebook, strict=True)):
        n_blocks_c = in_dim // bs_c
        cov_c = n_blocks_c * bs_c
        rem_c = in_dim - cov_c
        n_cb_c = 1 if shared_codebook else n_blocks_c
        codebook_bits_total += n_cb_c * k_c * bs_c * codebook_bits
        if k_c <= 1:
            assign_bits += n_rows * n_blocks_c * 1
        else:
            assign_bits += n_rows * n_blocks_c * math.ceil(math.log2(k_c))
        remainder_bits += n_rows * rem_c * 16  # bf16 raw remainder
        if ss_per_codebook[c]:
            sign_bits += n_rows * cov_c  # 1 bit per covered element

    # Scale bits: primary-only, and only for cosine (euclidean primary has
    # no scale — centroids are the direct reconstruction).
    scale_bits = n_rows * n_blocks * scale_bits_per_elem if primary_metric == "cosine" else 0

    total_bits = codebook_bits_total + assign_bits + scale_bits + sign_bits + remainder_bits
    total_weights = n_rows * in_dim
    return total_bits / total_weights

from __future__ import annotations

import logging
import sys
from pathlib import Path  # noqa: TC003


def setup_logging(log_file: Path | None = None, level: int = logging.INFO) -> logging.Logger:
    """Configure the 'skcq' logger with stderr + optional file handlers.

    Returns the configured logger so callers can log directly to it.
    Idempotent: removes any previously installed 'skcq' handlers first.
    """
    logger = logging.getLogger("skcq")
    logger.setLevel(level)
    logger.propagate = False

    for h in list(logger.handlers):
        logger.removeHandler(h)

    formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s")

    sh = logging.StreamHandler(sys.stderr)
    sh.setFormatter(formatter)
    logger.addHandler(sh)

    if log_file is not None:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        fh = logging.FileHandler(log_file)
        fh.setFormatter(formatter)
        logger.addHandler(fh)

    return logger

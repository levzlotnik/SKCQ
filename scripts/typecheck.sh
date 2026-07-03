#!/usr/bin/env bash
set -euo pipefail

uv run mypy --follow-imports=skip skcq/ main.py

#!/usr/bin/env bash
set -euo pipefail

uv run ruff check --fix skcq/ main.py
uv run ruff format skcq/ main.py

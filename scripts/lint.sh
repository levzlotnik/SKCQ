#!/usr/bin/env bash
set -euo pipefail

uv run ruff check skcq/ main.py
uv run ruff format --check skcq/ main.py

#!/usr/bin/env bash
set -euo pipefail

uv run ruff check --fix skcq/ build.py
uv run ruff format skcq/ build.py

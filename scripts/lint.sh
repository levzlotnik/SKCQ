#!/usr/bin/env bash
set -euo pipefail

uv run ruff check skcq/ build.py
uv run ruff format --check skcq/ build.py

#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
(cd rocm && uv sync)
(cd cuda && uv sync)

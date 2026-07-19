#!/usr/bin/env bash
# Create an opencode agent from a template by filling in the model id.
#
# Usage:
#   ./make_agent.sh <model id> <agent type = implementation|exploration|lean4>
#
# Example:
#   ./make_agent.sh tiger/GLM-5.2-IQ4 implementation
#
# The <model id> must be a fully-qualified opencode model id of the form
# "<provider>/<model>", e.g. "tiger/GLM-5.2-IQ4" or "serval/Qwen3.6-35B-A3B".
# The agent type selects which template to start from and becomes the agent's
# filename stem. The output file is written to .opencode/agents/.

set -euo pipefail

if [[ $# -lt 1 || $# -gt 2 ]]; then
  echo "Usage: $0 <model id> <agent type = implementation|exploration|lean4>" >&2
  exit 2
fi

MODEL_ID="$1"
AGENT_TYPE="${2:-implementation}"

if [[ ! "$MODEL_ID" =~ ^[A-Za-z0-9_.:-]+(/[A-Za-z0-9_.:-]+)+$ ]]; then
  echo "Error: model id must be of the form '<provider>/<model>[/<...>]' (got: '$MODEL_ID')" >&2
  exit 2
fi

case "$AGENT_TYPE" in
  implementation|exploration|lean4) ;;
  *)
    echo "Error: agent type must be 'implementation', 'exploration', or 'lean4' (got: '$AGENT_TYPE')" >&2
    exit 2
    ;;
esac

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TEMPLATE="$SCRIPT_DIR/.opencode/agents/${AGENT_TYPE}_agent_template.md"

if [[ ! -f "$TEMPLATE" ]]; then
  echo "Error: template not found at '$TEMPLATE'" >&2
  exit 1
fi

# Derive an agent name from the model id: "<provider>/<model>" -> "<provider>_<model>".
# Non-alphanumeric characters are replaced with underscores so the filename is safe.
AGENT_NAME="$(printf '%s' "$MODEL_ID" | tr '/' '_' | tr -c 'A-Za-z0-9_' '_')"
AGENT_NAME="${AGENT_TYPE}_${AGENT_NAME}"

OUTPUT="$SCRIPT_DIR/.opencode/agents/${AGENT_NAME}.md"

if [[ -f "$OUTPUT" ]]; then
  echo "Error: agent file already exists at '$OUTPUT'" >&2
  exit 1
fi

# Substitute the placeholder model line. The template's frontmatter has a line
# of the form 'model: FILL_MODEL'; replace FILL_MODEL with the provided id.
sed "s/^model: FILL_MODEL$/model: ${MODEL_ID//\//\\/}/" "$TEMPLATE" > "$OUTPUT"

echo "Created agent: $OUTPUT"
echo "  model: $MODEL_ID"
echo "  type:   $AGENT_TYPE"

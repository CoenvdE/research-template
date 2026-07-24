#!/usr/bin/env bash
# PostToolUse(Write|Edit) hook. Reads the edited file_path from the tool payload on
# stdin, matches it against .claude/knowledge-map.json, and emits a one-line nudge
# per matching entry so the docs/skills stay in sync with the code.
#
# Design: ZERO tokens on a non-match (emits nothing). Never blocks or fails a tool
# call — any error exits 0 silently. Deterministic; no LLM involved.
set -euo pipefail

DIR="${CLAUDE_PROJECT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
MAP="$DIR/.claude/knowledge-map.json"

command -v jq >/dev/null 2>&1 || exit 0
[ -f "$MAP" ] || exit 0

payload="$(cat)"
file_path="$(printf '%s' "$payload" | jq -r '.tool_input.file_path // ""' 2>/dev/null || true)"
[ -n "$file_path" ] || exit 0

# Collect nudges whose `match` regex hits the file_path.
nudges="$(jq -r --arg fp "$file_path" '
  .mappings[]
  | select(.match as $m | $fp | test($m))
  | "• " + .nudge
' "$MAP" 2>/dev/null || true)"

[ -n "$nudges" ] || exit 0

context="Knowledge-drift check for ${file_path}:
${nudges}
(Rules: .claude/knowledge-map.json. Update the doc now if the change makes it stale, and bump its last_validated.)"

jq -cn --arg ctx "$context" '{
  hookSpecificOutput: {
    hookEventName: "PostToolUse",
    additionalContext: $ctx
  }
}'

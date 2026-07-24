#!/usr/bin/env bash
# Deterministic docs/skills staleness audit. ZERO LLM cost — pure shell + git.
# For each skill: prints last-commit date, last_validated, age, and checks that every
# path in `sources:` still exists. Flags rows that need a human/Claude to re-validate.
#
# Usage: bash .claude/skills/audit-docs/audit.sh
set -uo pipefail

DIR="${CLAUDE_PROJECT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)}"
cd "$DIR" || exit 1
STALE_DAYS=90
now=$(date +%s)

flagged=0
printf '%-26s %-12s %-12s %-5s %s\n' "SKILL" "LASTCOMMIT" "VALIDATED" "AGE" "STATUS"
printf '%s\n' "----------------------------------------------------------------------------------------"

for f in .claude/skills/*/SKILL.md; do
  name=$(basename "$(dirname "$f")")
  lastcommit=$(git log -1 --format=%cs -- "$f" 2>/dev/null)
  validated=$(grep -m1 '^last_validated:' "$f" | sed 's/^last_validated:[[:space:]]*//')
  sources=$(grep -m1 '^sources:' "$f" | sed 's/^sources:[[:space:]]*//')

  reasons=""

  # Age since last_validated
  age="?"
  if [ -n "$validated" ]; then
    vsec=$(date -j -f "%Y-%m-%d" "$validated" +%s 2>/dev/null || date -d "$validated" +%s 2>/dev/null)
    if [ -n "${vsec:-}" ]; then
      age=$(( (now - vsec) / 86400 ))
      [ "$age" -gt "$STALE_DAYS" ] && reasons="${reasons}stale(${age}d) "
    fi
  else
    reasons="${reasons}no-last_validated "
  fi

  # Dead source references
  if [ -n "$sources" ]; then
    IFS=',' read -ra paths <<< "$sources"
    for p in "${paths[@]}"; do
      p="$(echo "$p" | xargs)"   # trim
      [ -z "$p" ] && continue
      if [ ! -e "$p" ] && ! compgen -G "$p*" >/dev/null 2>&1; then
        reasons="${reasons}missing:${p} "
      fi
    done
  fi

  # A source changed more recently than last_validated => content may have drifted
  if [ -n "$sources" ] && [ -n "$validated" ]; then
    IFS=',' read -ra paths <<< "$sources"
    for p in "${paths[@]}"; do
      p="$(echo "$p" | xargs)"; [ -z "$p" ] && continue
      srcdate=$(git log -1 --format=%cs -- "$p" 2>/dev/null)
      if [ -n "$srcdate" ] && [[ "$srcdate" > "$validated" ]]; then
        reasons="${reasons}src-changed:${p} "
      fi
    done
  fi

  if [ -z "$reasons" ]; then
    status="ok"
  else
    status="⚠ ${reasons}"
    flagged=$((flagged+1))
  fi
  printf '%-26s %-12s %-12s %-5s %s\n' "$name" "${lastcommit:-?}" "${validated:-?}" "$age" "$status"
done

echo
echo "Flagged: ${flagged}. Review each ⚠ skill against its sources, update if wrong, then bump last_validated to today."

---
name: audit-docs
description: Validate whether the repo's CLAUDE.md and .claude/skills/* are still accurate — flag stale, dead-referenced, or obsolete docs. Use when the user asks to audit/validate the docs or skills, check for doc drift, or after a large refactor. Run periodically to fight knowledge drift.
last_validated: 2026-07-22
sources: .claude/skills/audit-docs/audit.sh, .claude/knowledge-map.json
---

# Audit docs & skills for drift

Part of the knowledge framework (see the "Maintaining this knowledge" section of the root `CLAUDE.md`). The **hooks** catch drift as code changes; **this skill** catches drift that already accumulated and answers "are these skills even useful anymore?"

## Run it

```bash
bash .claude/skills/audit-docs/audit.sh
```

The script is deterministic (shell + git, no LLM cost). For each skill it reports:
- **LASTCOMMIT** — when the SKILL.md last changed.
- **VALIDATED** — the `last_validated` frontmatter date.
- **AGE** — days since `last_validated`; flagged `stale` past 90 days.
- **STATUS** — `ok`, or `⚠` with reasons: `stale(Nd)`, `missing:<path>` (a `sources:` path no longer exists), `src-changed:<path>` (a source changed after `last_validated`, so content may have drifted), `no-last_validated`.

## Then (the LLM part — only for ⚠ rows)

For each flagged skill:
1. Read the SKILL.md and skim its `sources:` files.
2. Decide: **accurate** (just bump `last_validated` to today) / **needs-update** (fix the content, then bump) / **obsolete** (propose deleting the skill to the user — don't delete silently).
3. Report a short per-skill verdict to the user; only edit after that.

## Keeping the map honest

`sources:` frontmatter and `.claude/knowledge-map.json` should point at the same areas. When a skill starts documenting a new part of the code, add both a `sources:` entry and a `knowledge-map.json` mapping so the hook nudges on future edits there.

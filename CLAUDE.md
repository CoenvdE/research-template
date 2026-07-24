# research-template

Lightning research template with correctness guardrails. This file is committed;
see README for the full tour.

## Commands

- `uv run pytest` runs the guardrail suite (CPU, seconds); `-m "not slow"` in CI.
- `uv run python -m src.train fit --config configs/base.yaml` trains on the synthetic set.
- Experiments stack overrides: `--config configs/base.yaml --config configs/experiment/<x>.yaml`.

## Conventions

- **Config discipline**: `configs/base.yaml` is the only base; experiments are
  small override files under `configs/experiment/`. Never copy-and-edit the
  base, never encode hyperparameters in filenames.
- **Every change to models/data/schedulers keeps the tests green.** When adding
  a model, port the guardrail tests to it (overfit-one-batch, unused-params,
  initial-loss, resume) rather than exempting it.
- **Stateful callbacks must implement `state_dict`/`load_state_dict`** (see
  `EMACallback`), or resume silently resets them; `tests/test_resume.py` is the
  proof and must stay green.
- **Resume**: keep `max_steps`/`max_epochs` unchanged across resumes (schedules
  are fraction-based). wandb resumes with the persisted id + `resume="must"`,
  never `"allow"` (see README "Resume").
- **EXPERIMENTS.md** is the run ledger: move TODO rows to DONE with a wandb link
  and a one-line conclusion when an experiment concludes. Keep entries to one line.
- **docs/DATASETS.md** describes every dataset; update it in the same change
  that alters data loading (dataset-overview skill).
- The synthetic dataset stays in the repo even after real data arrives; it is
  what keeps tests runnable everywhere.

## Adapting this template to a new project

Three tiers; when in doubt, keep.

- **KEEP always** (the point of the template): the guardrail tests and the rule
  that they get *ported* to the real model, never exempted; the synthetic
  dataset (it is what keeps tests runnable everywhere and CI green); config
  discipline (base + overrides); EXPERIMENTS.md flow; anti-drift wiring;
  provenance and resume rules.
- **REPLACE per project**: `src/models/mlp.py` and `src/data/` with the real
  model and data (added *next to* the synthetic pair, not instead of it);
  guard thresholds and `src/utils/checks.py` contracts tightened to the real
  data's units and normalization; `.claude/knowledge-map.json` regexes;
  DATASETS.md content; EXPERIMENTS.md rows.
- **DELETE if unused**: `tests/equivariance/` when no symmetry is claimed;
  `EMACallback` if not wanted; `configs/sweep.yaml` and
  `configs/experiment/ddp.yaml` until needed (they are templates-in-waiting,
  costless while unused).

Improvements made here that are project-agnostic get ported back to the
research-template repo in a separate commit.

## Maintaining this knowledge

The anti-drift system keeps docs honest: `.claude/hooks/knowledge-drift.sh`
(wired in `.claude/settings.json`) nudges when code covered by a doc changes,
per `.claude/knowledge-map.json`. Audit staleness with
`bash .claude/skills/audit-docs/audit.sh`.

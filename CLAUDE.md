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

## Validation integrity (non-negotiable)

Tests exist to detect defects, not to be green. When writing or fixing tests,
here or anywhere in a project based on this template:

- **Never weaken a test to make it pass.** A failing test is a finding to report,
  not a number to adjust. Do not loosen a tolerance, narrow an input range, add
  `skip`/`xfail`, wrap an assertion in try/except, or soften an assertion in
  order to reach green. Only the researcher decides whether the test or the code
  is wrong.
- **Expectations come from theory, a reference implementation, or a required
  property. Never from current output.** Legitimate: a closed-form value (MSE at
  init is Var(y), cross-entropy is ln(C)), an independent implementation, a
  property that must hold (equivariance, invariance, conservation), a physical
  bound. Recording what the model currently produces and asserting it keeps
  producing that is a *regression* test: legitimate only from an independently
  verified state, and it must be labeled as regression, not correctness.
- **Every new guardrail ships with a proof that it can fail**, either a planted
  defect it must catch or a written reason no control is possible. See
  `tests/test_guardrails_detect_bugs.py`; add new sensitivity proofs there or
  beside the test they belong to.
- **Tolerances are derived, not tuned.** Justify atol/rtol from dtype precision
  and accumulated operations; prefer float64 for property tests. A test that only
  passes at a loose tolerance is a finding.
- **Never assume a convention you did not read** (group action side, index order,
  channel layout, units). Read it from the code or detect it empirically. A wrong
  assumption that produces a pass is invisible.

The test to apply before committing any test: name the defect it would catch, and
say whether it would still pass if the model were subtly wrong. If you cannot,
delete it.

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

**Keep the map current.** It is not a one-time config. When a doc or skill starts
describing a new area of code, add a mapping for it in
`.claude/knowledge-map.json`; when a fact moves to a different doc, update that
entry's `doc` field so the nudge points at where the fact now lives. A map that
stops growing quietly stops catching drift.

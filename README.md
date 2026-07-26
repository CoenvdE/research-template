# research-template

A PyTorch Lightning template for research that is fast to iterate on and hard to
fool yourself with. Ships with correctness guardrails (the tests top labs are
paranoid about), performance callbacks (is the dataloader the bottleneck? what's
my MFU?), and a Claude-collaborative workflow (living docs kept honest by an
anti-drift hook).

I wrote up the reasoning behind it in [a blog post](https://coenvde.github.io/blogs/research-template/index.html).
The version I actually edit is [BLOG.md](BLOG.md) in this repo, so read it here
if you prefer.

Everything runs CPU-only on a synthetic dataset, so the full test suite works on
any machine with no data download.

## Quickstart

```bash
uv sync
uv run pytest                                   # the whole guardrail suite
uv run python -m src.train fit --config configs/base.yaml
```

## What's inside

**Callbacks** (`src/callbacks/`)
- `TimingCallback`: dataloader-wait vs forward vs backward time; `time/data_frac` tells you if the dataloader is the bottleneck
- `FlopsParamsCallback`: exact param count + measured forward FLOPs/sample
- `ThroughputCallback`: samples/sec, GPU memory, MFU (set `peak_flops_per_sec`)
- `InputStatsGuard`: raises when big numbers, NaNs, or unnormalized inputs reach the model (the degrees-instead-of-radians class of bug)
- `ActivationMonitor`: activation RMS per layer, catches explosions inside the net
- `GradNormCallback`: total (and top-k per-layer) gradient norms
- `EMACallback`: exponential moving average weights, checkpoint-safe
- `VisualizationCallback`: plots the same fixed val batch every epoch
- `ProvenanceCallback`: git SHA + dirty flag logged to every run

**Schedulers** (`src/schedulers/`): WSD and warmup-cosine, specified in
*fractions* of the run, working with `interval: step` or `epoch`.

**Tests** (`tests/`): overfit-one-batch, initial-loss sanity,
interrupt-and-resume equivalence (weights + EMA + LR), unused-parameter
detection, data-leakage asserts (overlap + temporal gap), dataloader
determinism (seed + num_workers), input-range guards, scheduler shapes, a
config smoke test, and a generic equivariance harness with examples.

**Tests of the tests** (`tests/test_guardrails_detect_bugs.py`): a suite that is
green because it *cannot fail* is worse than no suite. Every guardrail here is
paired with a planted defect it must catch, and CLAUDE.md states the validation
integrity rules: never weaken a test to reach green, derive expected values from
theory or a reference rather than from current output, and ship every new check
with a proof that it can fail.

**Agent** (`.claude/agents/model-sparring.md`): a design-review sparring partner
for model work. Reads the code, the metrics and `EXPERIMENTS.md`, then reports
correctness findings, a cost profile, and ranked proposals, each with a verdict
on whether it preserves the model's claimed invariances and a decisive ablation.
It writes tests and TODO rows, never the model.

## Config discipline

`configs/base.yaml` is the single base. Experiments are *small override files*
in `configs/experiment/` stacked on top:

```bash
uv run python -m src.train fit --config configs/base.yaml --config configs/experiment/wandb.yaml
```

Never copy-and-edit the base; hyperparameters belong in overrides and in wandb,
not in filenames.

## Beyond fit

- **Eval**: `uv run python -m src.eval --config configs/base.yaml --ckpt_path outputs/last.ckpt`
  validates a checkpoint and writes `outputs/eval/metrics.json` (which the
  anti-drift hook watches, nudging an EXPERIMENTS.md entry). `validate`,
  `test`, and `predict` subcommands also work directly on `src.train`.
- **Multi-GPU**: stack `configs/experiment/ddp.yaml` (Lightning launches the
  processes; see the file for batch-size and unused-params notes).
- **Sweeps**: `configs/sweep.yaml` is a ready wandb sweep spec; LightningCLI
  accepts the agent's `--model.lr=...` flags natively.

## Resume (training + wandb, crash-safe)

1. `ModelCheckpoint(save_last=True)` is in the base config; `last.ckpt` always exists.
2. Resume training: `... fit --config configs/base.yaml --ckpt_path outputs/last.ckpt`.
   Keep `max_epochs`/`max_steps` unchanged so fraction-based schedules line up.
3. Resume the *same wandb run*: the run id is persisted via
   `src.utils.provenance.get_or_create_wandb_id(run_dir)`; pass it with
   `resume="must"` (fails loudly instead of silently creating a new run):
   `--trainer.logger.init_args.id <id> --trainer.logger.init_args.resume must`.
4. `tests/test_resume.py` proves interrupted == uninterrupted. Any stateful
   custom callback must implement `state_dict`/`load_state_dict` or it silently
   resets on resume; `EMACallback` shows the pattern.

On SLURM, add `#SBATCH --signal=SIGUSR1@90` and `--requeue`; Lightning's
SLURMEnvironment checkpoints and requeues on preemption. Treat that as a bonus:
the periodic `last.ckpt` is the real safety net.

## Experiment tracking

`EXPERIMENTS.md` has two sections: **TODO** (hypothesis, config, status) and
**DONE** (one line per run: wandb link, key metric, conclusion). The anti-drift
hook nudges you to update it whenever training/eval code changes.

## Using this template

Click "Use this template" on GitHub, or run `gh repo create my-project --template
CoenvdE/research-template`.

The easiest way to tailor it is to hand it to Claude. Start a session in your new
clone and say something like:

> Read the README and CLAUDE.md, then set this up for my project. Ask me what you
> need to know about my model and data.

It should rename the project in `pyproject.toml`, put your goal into README and
`CLAUDE.md`, clear the example rows out of `EXPERIMENTS.md`, stub the real
dataset section in `docs/DATASETS.md`, tailor the regexes in
`.claude/knowledge-map.json` to your layout, and then run `uv run pytest` to
confirm the suite is still green before the first commit. What it should not do
is delete the synthetic dataset or exempt anything from the guardrail tests, and
`CLAUDE.md` tells it as much.

By hand it is the same list: rename in `pyproject.toml`, replace
`src/models/mlp.py` and `src/data/synthetic.py` with your model and data while
keeping the synthetic pair for smoke tests, tailor the knowledge map, and fill in
`docs/DATASETS.md`.

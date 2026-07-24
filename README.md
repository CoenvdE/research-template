# research-template

A PyTorch Lightning template for research that is fast to iterate on and hard to
fool yourself with. The full story behind it is in
[the blog post](https://coenvde.github.io/blogs/research-template/index.html),
whose source of truth lives here as [BLOG.md](BLOG.md). Ships with correctness guardrails (the tests top labs are
paranoid about), performance callbacks (is the dataloader the bottleneck? what's
my MFU?), and a Claude-collaborative workflow (living docs kept honest by an
anti-drift hook).

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
determinism (seed + num_workers), input-range guards, scheduler shapes, and a
generic equivariance harness with examples.

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

Click "Use this template" on GitHub (or `gh repo create my-project --template
CoenvdE/research-template`), then: rename the project in `pyproject.toml`,
replace `src/models/mlp.py` and `src/data/synthetic.py` with your model and
data (keep the synthetic set for smoke tests), tailor `.claude/knowledge-map.json`,
and fill `docs/DATASETS.md`. With Claude Code, the `new-research-project` skill
does this interactively.

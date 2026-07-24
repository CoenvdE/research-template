<!-- SOURCE OF TRUTH for this blog post. The website copy is generated from this file by import_claude_blogs.py in CoenvdE.github.io: edit here, then sync repo -> website. -->

# A solid research template
*By [Coen van den Elsen](https://coenvde.github.io/)*

---

### Why am I writing this?

Two bugs cost me the most GPU hours of my research career so far. The first: coordinates entered my model in degrees instead of radians. Nothing crashed; the model just trained worse, and it took embarrassingly long to find out why. The second: cluster jobs got preempted, and after rerunning them the loss would spike, forcing full reruns. Also silent, also expensive.

Neither bug is exotic. Every ML researcher I know has their own version. What struck me is that both are *mechanically detectable*, and that the standard academic repo detects neither. So I built a template that starts every project with the paranoia built in, and with the efficiency tooling (timing, throughput, schedulers, config discipline) that makes iteration fast rather than just safe: [github.com/CoenvdE/research-template](https://github.com/CoenvdE/research-template).

Two design rules shaped it. First, **guardrails are tests and callbacks, not discipline**: anything that relies on me remembering to check will eventually not get checked. Second, **everything runs CPU-only on a synthetic dataset shipped with the repo**, so the full suite passes on any laptop and in CI with no data download. The synthetic set never gets deleted, even after real data arrives; it is what keeps the tests runnable everywhere.

### The guardrail tests

**Overfit one batch.** Train on a single small batch and assert the loss collapses to ~0. The most diagnostic smoke test there is: a model that cannot memorize eight samples has a broken loss, data pipeline, or optimizer wiring. Seconds to run, catches entire categories of bugs.

**Initial loss sanity.** A freshly initialized model should score its theoretical know-nothing loss. For MSE the best constant prediction is the target mean, and the resulting error is by definition Var(y), which is ~1.0 for standardized targets; for classification it is ln(C). Assert the init loss is near that value: far above means broken normalization or init, and a *first epoch* far below should make you suspect leakage.

**Resume equals uninterrupted.** This is the loss-spike killer. The test trains N steps in one go, then separately trains, interrupts, resumes from the checkpoint, and asserts the two runs end with identical weights, identical EMA state, and identical learning rate. It catches weights-only checkpoints (Adam's moment estimates reset and the first resumed steps are badly scaled: the classic post-resume spike), schedulers that restart warmup, and any stateful callback that forgot to implement `state_dict`. This test caught a real bug during the template's own construction: EMA state restored from a checkpoint landed on CPU while the model trained on GPU.

**Unused parameters.** After one backward pass, every trainable parameter must have received a gradient; anything without one is a disconnected part of the graph. The suite also includes a *planted-bug* negative test: it deliberately attaches a layer that forward never calls and asserts the detector reports it. A test of the test, like a positive control in a lab assay: it proves the check cannot silently become vacuous. Bonus: a guaranteed fully-connected graph is what makes DDP's `find_unused_parameters=False` fast path valid.

**Data-leakage asserts.** Plain pytest over the split *definition*: train/val/test sample ids are disjoint; for time series, validation starts after training ends plus a gap at least as long as your input window and lead time (otherwise windows straddle the boundary); and normalization statistics were computed on the training range only. That last one is the sneakiest leak in practice.

**Dataloader determinism.** Same seed gives byte-identical batches, with `num_workers=0` and `num_workers=2`. Irreproducibility from seeding bugs is invisible until you need to reproduce a result, which is exactly when you cannot afford it.

**Input contracts.** One checker per physical convention the data must satisfy: `assert_radians` (the degrees bug, caught at load time), `assert_in_range` for hard physical bounds (sea-ice concentration in [0, 1]; Celsius sneaking into a Kelvin pipeline trips the [180, 340] K bound), `assert_standardized` (raw Pascals at ~1e5 fail instantly with "was normalization applied?"), and `assert_absmax` against gradient-destabilizing magnitudes.

**Config smoke test.** One `fast_dev_run` batch through the *committed* config, every callback included. The moment a rename breaks a `class_path`, CI goes red instead of your next cluster job.

### The callbacks

- **Timing**: splits step time into dataloader-wait vs forward vs backward. One logged number, `time/data_frac`, answers the eternal question "is my dataloader the bottleneck?"
- **FLOPs + params**: one generic callback measuring forward FLOPs on a real batch (I previously had five near-identical per-model copies; consolidation was overdue).
- **Throughput + MFU**: samples/sec, GPU memory, and Model FLOPs Utilization: the fraction of the card's peak math you actually use. Comparable across models and GPUs, so it directly answers "should I optimize or accept this?"
- **Input stats guard**: the runtime sibling of the input contracts. Checks the first three batches of every run (a broken pipeline dies in seconds, not after an hour) and periodically after, logging stats always and raising on violations.
- **Activation monitor**: activation RMS per layer via forward hooks. Inputs can be clean while a bad init grows activations layer by layer; this catches explosions *inside* the net.
- **Gradient norm**: total and top-k per-layer norms, the early-warning signal for instability.
- **EMA**: exponential moving average weights, swapped in for validation, checkpoint-safe by construction (the resume test enforces it).
- **Visualization**: plots the *same* fixed validation batch every epoch, so panels are comparable across a run.
- **Provenance**: logs git SHA and a dirty flag to every run. Any result traces back to exact code and config; no more archaeology through config files named `bigger_no_early_stop_lower_lr_good_ds.yaml` (a real filename from my past).

Schedulers round it out: WSD (warmup-stable-decay) and warmup-cosine, specified in *fractions* of the run so the same config works for any length, resolved through Lightning so step-based and epoch-based training both work and resumes stay consistent.

### Resume, done properly

The template's rules: `last.ckpt` is saved periodically and is a *full* checkpoint (optimizer moments, scheduler, precision state, callback states). Resume via `--ckpt_path` with unchanged `max_steps`. And wandb resumes into the *same run* using a persisted run id with `resume="must"`, which fails loudly rather than silently creating a duplicate run the way the default does. On SLURM, `#SBATCH --signal=SIGUSR1@90` gives Lightning a 90-second warning to checkpoint and requeue before the job is killed; the periodic checkpoint remains the real safety net.

### The Claude layer

The template is also configured as a collaborator-ready workspace for Claude Code (the setup itself is the subject of [the companion post](https://coenvde.github.io/blogs/claude-workflow/index.html)):

- **A committed CLAUDE.md** with the conventions *and* an adaptation guide: what to KEEP always (the guardrails, the synthetic data, config discipline), what to REPLACE per project (model, data, thresholds), what to DELETE if unused (the equivariance harness when no symmetry is claimed).
- **An anti-drift map**: editing data code nudges the agent to update `docs/DATASETS.md` in the same session; writing eval outputs nudges a one-line entry in the experiment ledger.
- **EXPERIMENTS.md as the human-agent interface**: a TODO table (hypothesis, config, status) and a DONE table (run, wandb link, key metric, one-line conclusion). Plans go in as TODO rows and come out as DONE rows.
- **A scaffolding skill**: "start a new research project called X" clones the template from GitHub, tailors it, and verifies the suite is green before the first commit. Improvements made in projects get ported back to the template, so it compounds instead of forking.

### Closing

None of the individual pieces is novel; most exist scattered across big-lab codebases and hard-won personal scripts. The value is having them *start present* in every project, verified by CI, so the paranoid checks run whether or not you remember to be paranoid. The repo is public and marked as a GitHub template: [CoenvdE/research-template](https://github.com/CoenvdE/research-template). Questions and improvements welcome.

<!-- SOURCE OF TRUTH for this blog post. The website copy is generated from this file by import_claude_blogs.py in CoenvdE.github.io: edit here, then sync repo -> website. -->

# A solid research template
*By [Coen van den Elsen](https://coenvde.github.io/)*

---

### Why am I writing this?

Two bugs have cost me more GPU hours than anything else in my research so far. In the first, coordinates went into my model in degrees instead of radians. Nothing crashed. The model just trained worse, and it took me embarrassingly long to work out why. In the second, cluster jobs got preempted, and when I reran them the loss would spike and I had to start over. Also silent, also expensive.

Neither bug is exotic. Every ML researcher I know has their own version of both. What struck me is that they are mechanically detectable, and that the standard academic repo detects neither. So I built a template that starts every project with the paranoia already in place, along with the efficiency tooling that makes iteration fast rather than only safe. It lives at [github.com/CoenvdE/research-template](https://github.com/CoenvdE/research-template).

Two design rules shaped it. The first is that **guardrails should be tests and callbacks, not discipline**, because anything relying on me to remember to check will eventually not get checked. The second is that **everything runs CPU-only on a synthetic dataset shipped with the repo**, so the full suite passes on any laptop and in CI without downloading anything. That synthetic set never gets deleted, even once real data arrives. It is what keeps the tests runnable everywhere.

### The guardrail tests

**Overfit one batch.** Train on a single small batch and assert the loss collapses to roughly zero. It is the most diagnostic smoke test I know of. A model that cannot memorize eight samples has a broken loss, a broken data pipeline, or broken optimizer wiring. It takes seconds and rules out whole categories of bug.

**Initial loss sanity.** A freshly initialized model should score its theoretical know-nothing loss. For MSE the best constant prediction is the target mean, and the error that leaves you with is by definition Var(y), which is about 1.0 for standardized targets. For classification it is ln(C). Assert the initial loss sits near that value. A first epoch far below it should make you suspect leakage.

It is worth being precise about what this actually catches, because I got it wrong once already. A model at init outputs near zero, so its loss is E[y²] = Var(y) + mean(y)², while the theoretical value is just Var(y). Targets left in raw physical units, so Kelvin or Pascals or raw counts, have a large mean, the ratio explodes, and the check fires hard. But rescaling *centered* targets moves Var(y) and E[y²] together, the ratio stays at 1, and the check is blind to it. That blind spot is covered by `assert_standardized` instead, and it is asserted in the test suite so it stays documented rather than turning into folklore.

**Resume equals uninterrupted.** This is the loss-spike killer. The test trains N steps in one go. Then it trains again, interrupts, resumes from the checkpoint, and asserts the two runs end with identical weights, identical EMA state and identical learning rate. It catches weights-only checkpoints, where Adam's moment estimates reset and the first resumed steps are badly scaled, which is the classic post-resume spike. It catches schedulers that restart warmup. And it catches any stateful callback that forgot to implement `state_dict`. It found a real bug while I was building the template, where EMA state restored from a checkpoint landed on CPU while the model trained on GPU.

**Unused parameters.** After one backward pass, every trainable parameter must have received a gradient. Anything without one is a disconnected part of the graph. The suite also includes a planted-bug negative test that deliberately attaches a layer forward never calls, then asserts the detector reports it. That is a test of the test, the same idea as a positive control in a lab assay, and it proves the check cannot quietly become vacuous. There is a side benefit too. A graph you know is fully connected is what makes DDP's `find_unused_parameters=False` fast path valid.

**Data-leakage asserts.** Plain pytest over the split definition. Train, val and test sample ids must be disjoint. For time series, validation has to start after training ends plus a gap at least as long as your input window and lead time, otherwise windows straddle the boundary. And normalization statistics must have been computed on the training range only. That last one is the sneakiest leak in practice.

**Dataloader determinism.** The same seed gives byte-identical batches, with `num_workers=0` and with `num_workers=2`. Irreproducibility from a seeding bug stays invisible until the moment you need to reproduce a result, which is exactly when you cannot afford it.

**Input contracts.** One checker per physical convention the data has to satisfy. `assert_radians` catches the degrees bug at load time. `assert_in_range` enforces hard physical bounds, so sea-ice concentration stays in [0, 1], and Celsius sneaking into a Kelvin pipeline trips the [180, 340] K bound. `assert_standardized` makes raw Pascals at 1e5 fail instantly. `assert_absmax` catches magnitudes large enough to destabilize gradients.

**Config smoke test.** One `fast_dev_run` batch through the committed config, with every callback included. The moment a rename breaks a `class_path`, CI goes red instead of your next cluster job.

### Tests of the tests

The failure mode that worries me more than any bug is a test suite that is green because it cannot fail. It happens quietly and in ordinary ways. An expected value gets recorded from the code under test, so the test asserts the bug. A tolerance gets loosened until things pass. Someone checks a property that a model ignoring its inputs would satisfy anyway. None of it announces itself, and afterwards you trust the model more than you did before, which is the expensive part.

The risk gets sharper when an agent writes the tests, because "make the tests pass" and "check the model is correct" are different goals that look identical from the outside, and the first one is much easier.

So the template treats it as a first-class concern, in three layers.

**A stated policy**, in CLAUDE.md and in the research rules, so it loads into every session that touches this code. Never weaken a test to make it pass. A failing test is a finding to report, not a number to adjust. Expected values come from theory, from a reference implementation, or from a property that must hold, never from recording current output. Tolerances get derived from dtype precision rather than tuned until green. And never assume a convention you did not read, because a wrong assumption that produces a pass is invisible.

**Paired negative controls.** Every guardrail ships with a demonstration that it can fail. The equivariance examples pair each symmetry with a transformation the function must not respect. The unused-parameter test plants a layer forward never calls. The input contracts plant degrees, raw Pascals and Celsius.

**A meta-test that plants defects**, in `tests/test_guardrails_detect_bugs.py`, which asserts the suite goes red when it should. It runs a model with a learning rate of zero and asserts that the overfit-one-batch criterion is not satisfied, which proves that criterion discriminates. It scales an init by 50x and asserts the initial-loss check raises. And it verifies the two premises the resume test silently rests on, that fixed seeds give bit-identical results, and that genuinely different runs do not compare equal at the tolerance used. Without those, the resume guarantee is unfalsifiable.

That last one is the pattern worth stealing. Ask of any check what defect it would catch, and whether it would still pass if the model were subtly wrong. If that question is uncomfortable to answer, the check is decoration.

### The callbacks

- **Timing** splits step time into dataloader-wait, forward and backward. One logged number, `time/data_frac`, answers the eternal question of whether the dataloader is the bottleneck.
- **FLOPs and params** is one generic callback measuring forward FLOPs on a real batch. I previously had five near-identical per-model copies of this, so consolidating was overdue.
- **Throughput and MFU** logs samples per second, GPU memory, and Model FLOPs Utilization, which is the fraction of the card's peak math you actually use. It is comparable across models and GPUs, so it answers whether optimizing is worth your time.
- **Input stats guard** is the runtime sibling of the input contracts. It checks the first three batches of every run, so a broken pipeline dies in seconds rather than after an hour, then keeps checking periodically. It logs stats always and raises on violations.
- **Activation monitor** logs activation RMS per layer through forward hooks. Inputs can be perfectly clean while a bad init grows activations layer by layer, and this is what catches explosions inside the net.
- **Gradient norm** logs total and top-k per-layer norms, which is the early-warning signal for instability.
- **EMA** keeps exponential moving average weights, swapped in for validation, checkpoint-safe by construction because the resume test enforces it.
- **Visualization** plots the same fixed validation batch every epoch, so panels stay comparable across a run.
- **Provenance** logs the git SHA and a dirty flag to every run, so any result traces back to the exact code and config. No more archaeology through config files named `bigger_no_early_stop_lower_lr_good_ds.yaml`, which is a real filename from my past.

Schedulers round it out. WSD (warmup-stable-decay) and warmup-cosine, both specified in fractions of the run so one config works at any length, and both resolved through Lightning so step-based and epoch-based training work and resumes stay consistent.

### Resume, done properly

The rules are simple. `last.ckpt` gets saved periodically and is a full checkpoint, meaning optimizer moments, scheduler, precision state and callback states. Resume with `--ckpt_path` and leave `max_steps` unchanged. And wandb resumes into the same run using a persisted run id with `resume="must"`, which fails loudly instead of silently creating a duplicate run the way the default does. On SLURM, `#SBATCH --signal=SIGUSR1@90` gives Lightning ninety seconds of warning to checkpoint and requeue before the job is killed, though the periodic checkpoint is still the real safety net.

### The Claude layer

The template is also set up as a collaborator-ready workspace for Claude Code. The setup itself is the subject of [the companion post](https://coenvde.github.io/blogs/claude-workflow/index.html).

- **A committed CLAUDE.md** with the conventions and an adaptation guide. What to keep always, meaning the guardrails, the synthetic data and the config discipline. What to replace per project, meaning model, data and thresholds. What to delete if unused, such as the equivariance harness when no symmetry is claimed.
- **An anti-drift map.** Editing data code nudges the agent to update `docs/DATASETS.md` in the same session, and writing eval outputs nudges a one-line entry in the experiment ledger.
- **EXPERIMENTS.md as the human-agent interface.** A TODO table with hypothesis, config and status, and a DONE table with the run, a wandb link, the key metric and a one-line conclusion. Plans go in as TODO rows and come out as DONE rows.
- **A scaffolding skill.** Saying "start a new research project called X" clones the template from GitHub, tailors it, and verifies the suite is green before the first commit. Improvements made inside projects get ported back to the template, so it compounds instead of forking.
- **A `model-sparring` agent**, which is the design review a good lab runs before anyone burns GPU hours. It reads the model code, the measured metrics and the experiment ledger, then reports correctness findings, a cost profile and ranked proposals. Its rules are what separate it from a listicle generator. No suggestion without a measurement or a code citation. Every proposal has to state whether it preserves the model's claimed invariants, because half the fashionable tricks break equivariance and a proposal that quietly does is worse than no proposal at all. No proposing an ablation that `EXPERIMENTS.md` already concluded. And it may write tests and TODO rows but never the model itself. It proposes, you decide.

### Closing

None of the individual pieces here is novel. Most of them exist scattered across big-lab codebases and hard-won personal scripts. The value is in having them present from the start of every project and verified by CI, so the paranoid checks run whether or not I remember to be paranoid that day. The repo is public and marked as a GitHub template at [CoenvdE/research-template](https://github.com/CoenvdE/research-template). Questions and improvements welcome.

"""Run provenance: git SHA + dirty flag + seed, and wandb run-id persistence for
crash-safe resume of the SAME wandb run."""

import os
import subprocess


def git_info(cwd: str | None = None) -> dict:
    def run(*args):
        try:
            return subprocess.run(
                ["git", *args], capture_output=True, text=True, cwd=cwd, timeout=10
            ).stdout.strip()
        except Exception:
            return ""

    sha = run("rev-parse", "HEAD")
    dirty = bool(run("status", "--porcelain"))
    return {"git_sha": sha or "unknown", "git_dirty": dirty}


def get_or_create_wandb_id(run_dir: str) -> str:
    """Stable wandb run id stored next to the checkpoints.

    First call creates and persists an id; later calls (resumes) return the same
    id. Pass it to WandbLogger(id=..., resume="must") on resume so metrics
    continue in the same run instead of silently starting a new one.
    """
    os.makedirs(run_dir, exist_ok=True)
    path = os.path.join(run_dir, "wandb_run_id.txt")
    if os.path.exists(path):
        with open(path) as f:
            return f.read().strip()
    import wandb

    run_id = wandb.util.generate_id()
    with open(path, "w") as f:
        f.write(run_id)
    return run_id

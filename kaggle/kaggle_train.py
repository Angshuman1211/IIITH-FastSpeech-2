"""Cell contents for the Kaggle training notebook.

Paste this into a Kaggle notebook with GPU enabled, after attaching the dataset
produced by scripts/package_for_kaggle.py.

Run it with Save Version -> Save & Run All (Commit), not interactively.
An interactive session idles out after about 20 minutes; a committed run goes
the full 12 hours and keeps /kaggle/working as the notebook output.

If a run is cut short, note the highest step in /kaggle/working/ckpt, attach
that output as an input to the next run, and set RESTORE_STEP below.
"""

import os
import subprocess
import sys
from pathlib import Path

LANGUAGE = "Telugu"          # or "English"
DATASET = "fs2-telugu"       # the attached dataset's folder under /kaggle/input
RESTORE_STEP = 0             # set to resume from a previous run's checkpoint
TOTAL_STEP = 80000
RESTORE_FROM = None          # e.g. "/kaggle/input/previous-run-output/ckpt"

IN = Path("/kaggle/input") / DATASET
WORK = Path("/kaggle/working")
CKPT = WORK / "ckpt"
LOGS = WORK / "logs"


def sh(cmd):
    print(f"$ {cmd}", flush=True)
    r = subprocess.run(cmd, shell=True)
    if r.returncode != 0:
        raise SystemExit(f"command failed: {cmd}")


# ---------------------------------------------------------------- environment
# Kaggle images ship torch already. These are the ones that are usually missing.
sh(f"{sys.executable} -m pip install -q pyworld tgt g2p-en inflect 'librosa>=0.10' 'setuptools<81'")

print("\n--- GPU ---", flush=True)
sh("nvidia-smi --query-gpu=name,memory.total --format=csv,noheader")

import torch  # noqa: E402

print("torch", torch.__version__, "| cuda available:", torch.cuda.is_available(), flush=True)
if not torch.cuda.is_available():
    raise SystemExit("No GPU. Turn on the accelerator in notebook settings.")

# ---------------------------------------------------------------- resume setup
CKPT.mkdir(parents=True, exist_ok=True)
if RESTORE_STEP and RESTORE_FROM:
    src = Path(RESTORE_FROM) / f"{RESTORE_STEP}.pth.tar"
    if not src.exists():
        raise SystemExit(f"checkpoint {src} not found")
    sh(f"cp '{src}' '{CKPT}/'")
    print(f"resuming from step {RESTORE_STEP}", flush=True)

# ---------------------------------------------------------------- train
entry = IN / "project" / LANGUAGE / "train.py"
pre = IN / "preprocessed" / LANGUAGE

cmd = (
    f"{sys.executable} '{entry}' "
    f"--restore_step {RESTORE_STEP} "
    f"--set preprocess.path.preprocessed_path={pre} "
    f"--set train.path.ckpt_path={CKPT} "
    f"--set train.path.log_path={LOGS} "
    f"--set train.path.result_path={WORK}/result "
    f"--set train.step.total_step={TOTAL_STEP}"
)
sh(cmd)

# ---------------------------------------------------------------- trim output
# Each full checkpoint is about 420 MB. Keep the two most recent plus every
# weights-only copy, which is what the repository actually ships.
full = sorted(CKPT.glob("[0-9]*.pth.tar"), key=lambda p: int(p.stem))
for p in full[:-2]:
    print(f"removing {p.name}", flush=True)
    p.unlink()

print("\n--- output ---", flush=True)
for p in sorted(WORK.rglob("*")):
    if p.is_file():
        print(f"{p.stat().st_size/1e6:9.1f} MB  {p.relative_to(WORK)}", flush=True)

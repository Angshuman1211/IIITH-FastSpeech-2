"""Kaggle script kernel that trains one FastSpeech 2 model.

Pushed and run headlessly by scripts/launch_kaggle.py, so the settings at the
top are rewritten by that script before each push. Nothing here is meant to be
edited by hand.

A Kaggle GPU session is killed at 12 h and the output is discarded with it, so
FS2_MAX_HOURS stops training short of that and saves on the way out. Resuming
attaches the previous run's output as an input and picks up from its checkpoint.
"""

import os
import subprocess
import sys
from pathlib import Path

# --- rewritten by the launcher ------------------------------------------------
LANGUAGE = "Telugu"
DATASET = "fs2-telugu"
TOTAL_STEP = 80000
MAX_HOURS = 11.0
RESTORE_STEP = 0
RESTORE_FROM = ""      # e.g. "/kaggle/input/fs2-telugu-train/ckpt"
# -----------------------------------------------------------------------------

IN = Path("/kaggle/input") / DATASET
WORK = Path("/kaggle/working")
CKPT = WORK / "ckpt"
LOGS = WORK / "logs"


def sh(cmd, check=True):
    print(f"$ {cmd}", flush=True)
    r = subprocess.run(cmd, shell=True)
    if check and r.returncode != 0:
        raise SystemExit(f"command failed ({r.returncode}): {cmd}")
    return r.returncode


# ------------------------------------------------------------------ environment
# Internet needs a phone-verified Kaggle account. If it is off, the image may
# still carry everything training needs, so warn rather than die here and let
# the import check below be the thing that decides.
# The list is every third-party module the package imports, not just the ones
# training touches, so inference in a later cell works too.
if sh(f"{sys.executable} -m pip install -q "
      f"unidecode inflect tgt pyworld g2p-en "
      f"'librosa>=0.10' 'setuptools<81'", check=False):
    print("WARNING: pip install failed, probably no internet on this session.",
          flush=True)

missing = []
for mod in ("torch", "numpy", "scipy", "librosa", "matplotlib", "tqdm", "yaml",
            "sklearn", "inflect", "unidecode", "tgt", "torch.utils.tensorboard"):
    try:
        __import__(mod)
    except Exception as e:
        missing.append(f"{mod} ({type(e).__name__})")
if missing:
    raise SystemExit("training cannot start, missing: " + ", ".join(missing) +
                     "\nEnable Internet in the notebook settings (needs a "
                     "phone-verified Kaggle account).")

print("\n--- GPU ---", flush=True)
sh("nvidia-smi --query-gpu=name,memory.total --format=csv,noheader", check=False)

import torch  # noqa: E402

print("torch", torch.__version__, "| cuda:", torch.cuda.is_available(), flush=True)
if not torch.cuda.is_available():
    raise SystemExit("No GPU on this session. Enable the accelerator.")

# ------------------------------------------------------------------ sanity
# Do not assume the mount name. Kaggle derives it from the dataset slug, and a
# dataset that was still processing when the kernel started is not mounted at
# all, so find the features and fail with the actual tree if they are absent.
print("\n--- /kaggle/input ---", flush=True)
root = Path("/kaggle/input")
for p in sorted(root.glob("*")):
    print("   ", p.name, flush=True)
    for q in sorted(p.glob("*"))[:10]:
        print("      ", q.name, flush=True)
        for r in sorted(q.glob("*"))[:10]:
            print("         ", r.name, flush=True)

# The mount depth is not fixed. Older images put a dataset at
# /kaggle/input/<slug>, this one uses /kaggle/input/datasets/<owner>/<slug>,
# so search for the features instead of assuming either.
hits = sorted(root.rglob(f"preprocessed/{LANGUAGE}/train.txt"))
if not hits:
    raise SystemExit(
        f"no preprocessed/{LANGUAGE}/train.txt anywhere under /kaggle/input. "
        f"Either the dataset is not attached, or it was still processing when "
        f"this run started. Check it reports 'ready' and push again."
    )
IN = hits[0].parent.parent.parent
pre = IN / "preprocessed" / LANGUAGE
print(f"\nusing dataset root {IN}", flush=True)

n_train = sum(1 for _ in open(pre / "train.txt", encoding="utf-8"))
n_val = sum(1 for _ in open(pre / "val.txt", encoding="utf-8"))
print(f"features OK: {n_train} train / {n_val} val utterances", flush=True)

entry = IN / "project" / LANGUAGE / "train.py"
if not entry.exists():
    raise SystemExit(f"training entry point missing at {entry}")

# ------------------------------------------------------------------ resume
CKPT.mkdir(parents=True, exist_ok=True)
if RESTORE_STEP and RESTORE_FROM:
    src = Path(RESTORE_FROM) / f"{RESTORE_STEP}.pth.tar"
    if not src.exists():
        raise SystemExit(f"checkpoint {src} not found")
    sh(f"cp '{src}' '{CKPT}/'")
    # carry the loss history forward so the curves stay continuous
    prev_logs = Path(RESTORE_FROM).parent / "logs"
    for name in ("losses_train.csv", "losses_val.csv"):
        if (prev_logs / name).exists():
            LOGS.mkdir(parents=True, exist_ok=True)
            sh(f"cp '{prev_logs / name}' '{LOGS}/'")
    print(f"resuming from step {RESTORE_STEP}", flush=True)

# ------------------------------------------------------------------ train
# Kaggle hands out two T4s, which would turn on nn.DataParallel. At batch 16
# that splits into 8 per device and gains little, while adding a failure mode
# to an unattended 11 h job, so train on one device.
env = dict(os.environ, FS2_MAX_HOURS=str(MAX_HOURS), PYTHONUNBUFFERED="1",
           CUDA_VISIBLE_DEVICES="0")
cmd = [
    sys.executable, str(entry),
    "--restore_step", str(RESTORE_STEP),
    "--set", f"preprocess.path.preprocessed_path={pre}",
    "--set", f"train.path.ckpt_path={CKPT}",
    "--set", f"train.path.log_path={LOGS}",
    "--set", f"train.path.result_path={WORK}/result",
    "--set", f"train.step.total_step={TOTAL_STEP}",
]
print("\n$ " + " ".join(cmd), flush=True)
rc = subprocess.run(cmd, env=env).returncode

# ------------------------------------------------------------------ keep output
# Each full checkpoint is about 420 MB, so keep only the most recent one plus
# every weights-only copy. Do this even if training failed, so a partial run
# still leaves something to resume from.
full = sorted(CKPT.glob("[0-9]*.pth.tar"), key=lambda p: int(p.stem))
for p in full[:-1]:
    print(f"removing {p.name}", flush=True)
    p.unlink()

print("\n--- output ---", flush=True)
for p in sorted(WORK.rglob("*")):
    if p.is_file():
        print(f"{p.stat().st_size/1e6:9.1f} MB  {p.relative_to(WORK)}", flush=True)

if rc != 0:
    raise SystemExit(f"training exited {rc}")

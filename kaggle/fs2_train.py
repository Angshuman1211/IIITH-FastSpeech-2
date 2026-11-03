"""Kaggle script kernel. Pushed by scripts/launch_kaggle.py, which rewrites
the settings below before each push."""

import os
import subprocess
import sys
from pathlib import Path

LANGUAGE = "Telugu"
DATASET = "fs2-telugu"
TOTAL_STEP = 80000
MAX_HOURS = 11.0
RESTORE_STEP = 0
RESTORE_FROM = ""

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


# internet needs a phone-verified account, so warn and let the imports decide
if sh(f"{sys.executable} -m pip install -q "
      f"unidecode inflect tgt pyworld g2p-en "
      f"'librosa>=0.10' 'setuptools<81'", check=False):
    print("WARNING: pip install failed, probably no internet.", flush=True)

missing = []
for mod in ("torch", "numpy", "scipy", "librosa", "matplotlib", "tqdm", "yaml",
            "sklearn", "inflect", "unidecode", "tgt", "torch.utils.tensorboard"):
    try:
        __import__(mod)
    except Exception as e:
        missing.append(f"{mod} ({type(e).__name__})")
if missing:
    raise SystemExit("missing: " + ", ".join(missing) + "\nEnable Internet.")

print("\n--- GPU ---", flush=True)
sh("nvidia-smi --query-gpu=name,memory.total --format=csv,noheader", check=False)

import torch  # noqa: E402

print("torch", torch.__version__, "| cuda:", torch.cuda.is_available(), flush=True)
if not torch.cuda.is_available():
    raise SystemExit("No GPU on this session. Enable the accelerator.")

print("\n--- /kaggle/input ---", flush=True)
root = Path("/kaggle/input")
for p in sorted(root.glob("*")):
    print("   ", p.name, flush=True)
    for q in sorted(p.glob("*"))[:10]:
        print("      ", q.name, flush=True)
        for r in sorted(q.glob("*"))[:10]:
            print("         ", r.name, flush=True)

# mount depth varies: /kaggle/input/<slug> on older images,
# /kaggle/input/datasets/<owner>/<slug> here, so search rather than assume
hits = sorted(root.rglob(f"preprocessed/{LANGUAGE}/train.txt"))
if not hits:
    raise SystemExit(
        f"no preprocessed/{LANGUAGE}/train.txt under /kaggle/input. Either the "
        f"dataset is not attached, or it was still processing when this started."
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

# one device: two T4s would turn on DataParallel, which at batch 16 gains
# little and adds a failure mode to an unattended run
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

# full checkpoints are 420 MB each, keep the newest plus the weights-only copies.
# match the digits exactly: "[0-9]*" also catches 50000_weights_only.pth.tar,
# and Path.stem only strips .tar off it
def step_of(p):
    return int(p.name.split(".")[0])


full = sorted((p for p in CKPT.glob("*.pth.tar") if p.name.split(".")[0].isdigit()),
              key=step_of)
for p in full[:-1]:
    print(f"removing {p.name}", flush=True)
    p.unlink()

print("\n--- output ---", flush=True)
for p in sorted(WORK.rglob("*")):
    if p.is_file():
        print(f"{p.stat().st_size/1e6:9.1f} MB  {p.relative_to(WORK)}", flush=True)

if rc != 0:
    raise SystemExit(f"training exited {rc}")

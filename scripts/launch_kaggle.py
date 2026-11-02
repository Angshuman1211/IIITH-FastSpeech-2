"""Push and run the training kernel on Kaggle, and report on it.

    python scripts/launch_kaggle.py --language Telugu --push
    python scripts/launch_kaggle.py --language Telugu --status
    python scripts/launch_kaggle.py --language Telugu --log
    python scripts/launch_kaggle.py --language Telugu --pull

A Kaggle session is capped at 12 h, so one push trains for at most MAX_HOURS
and saves a checkpoint. To continue, push again with --restore-step set to the
highest step in the previous run's output, which is attached automatically.
"""

import argparse
import json
import re
import shutil
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
USER = "angshuman12"


def api():
    from kaggle.api.kaggle_api_extended import KaggleApi

    a = KaggleApi()
    a.authenticate()
    return a


def slug(language):
    return f"{USER}/fs2-{language.lower()}-train"


def build(language, total_step, max_hours, restore_step, staging):
    """Write the kernel folder: the script with its settings baked in, plus the
    metadata Kaggle needs to run it on a GPU."""
    if staging.exists():
        shutil.rmtree(staging)
    staging.mkdir(parents=True)

    src = (ROOT / "kaggle" / "fs2_train.py").read_text(encoding="utf-8")
    kslug = slug(language)
    restore_from = f"/kaggle/input/fs2-{language.lower()}-train/ckpt" if restore_step else ""

    subs = {
        "LANGUAGE": f'"{language}"',
        "DATASET": f'"fs2-{language.lower()}"',
        "TOTAL_STEP": str(total_step),
        "MAX_HOURS": str(max_hours),
        "RESTORE_STEP": str(restore_step),
        "RESTORE_FROM": f'"{restore_from}"',
    }
    for key, val in subs.items():
        src, n = re.subn(rf"^{key} = .*?$", f"{key} = {val}", src, count=1, flags=re.M)
        if n != 1:
            sys.exit(f"could not set {key} in fs2_train.py")
    (staging / "fs2_train.py").write_text(src, encoding="utf-8")

    meta = {
        "id": kslug,
        "title": f"fs2 {language.lower()} train",
        "code_file": "fs2_train.py",
        "language": "python",
        "kernel_type": "script",
        "is_private": True,
        "enable_gpu": True,
        "enable_tpu": False,
        "enable_internet": True,
        "dataset_sources": [f"{USER}/fs2-{language.lower()}"],
        "competition_sources": [],
        "model_sources": [],
        # attaching the kernel to itself gives the next run the previous
        # run's checkpoints under /kaggle/input/<kernel-slug>/
        "kernel_sources": [kslug] if restore_step else [],
    }
    (staging / "kernel-metadata.json").write_text(json.dumps(meta, indent=1))
    print(f"built kernel {kslug}: {language}, total_step={total_step}, "
          f"max_hours={max_hours}, restore_step={restore_step}")
    return kslug


def status(a, kslug):
    r = a.kernels_status(kslug)
    st = getattr(r, "status", r)
    msg = getattr(r, "failure_message", None) or ""
    print(f"{kslug}: {st} {msg}".rstrip())
    return str(st)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--language", required=True, choices=["English", "Telugu"])
    ap.add_argument("--total-step", type=int, default=80000)
    ap.add_argument("--max-hours", type=float, default=11.0)
    ap.add_argument("--restore-step", type=int, default=0)
    ap.add_argument("--push", action="store_true")
    ap.add_argument("--status", action="store_true")
    ap.add_argument("--log", action="store_true")
    ap.add_argument("--pull", action="store_true", help="download the run output")
    ap.add_argument("--wait", action="store_true", help="poll until the run ends")
    args = ap.parse_args()

    kslug = slug(args.language)
    a = api()

    if args.push:
        staging = ROOT / "kaggle_pkg" / f"kernel_{args.language.lower()}"
        build(args.language, args.total_step, args.max_hours, args.restore_step, staging)
        res = a.kernels_push(str(staging))
        url = getattr(res, "url", None)
        err = getattr(res, "error", None)
        if err:
            sys.exit(f"push failed: {err}")
        print(f"pushed. running at {url or 'https://www.kaggle.com/' + kslug}")

    if args.status or args.wait:
        st = status(a, kslug)
        while args.wait and st.lower() in ("running", "queued"):
            time.sleep(120)
            st = status(a, kslug)

    if args.log or args.pull:
        dest = ROOT / "kaggle_pkg" / f"out_{args.language.lower()}"
        dest.mkdir(parents=True, exist_ok=True)
        # The run log is one of the output files, so --log and --pull differ
        # only in whether the big checkpoints come down too.
        pattern = None if args.pull else "*.log"
        files, _ = a.kernels_output(kslug, str(dest), file_pattern=pattern,
                                    force=True, quiet=not args.pull)
        print(f"output -> {dest}")
        for f in files:
            print("   ", f)


if __name__ == "__main__":
    main()

"""Draw training_loss.png and validation_loss.png from the trainer's CSV logs.

    python scripts/plot_losses.py --language Telugu

The trainer writes losses_train.csv and losses_val.csv under logs/<language>/.
Each has columns: step, total, mel, mel_postnet, pitch, energy, duration.
"""

import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
COMPONENTS = ["mel", "mel_postnet", "pitch", "energy", "duration"]


def read_csv(path):
    with path.open(encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        return None
    return {k: [float(r[k]) for r in rows] for k in rows[0]}


def smooth(ys, window):
    if window <= 1 or len(ys) < window:
        return ys
    out, run = [], 0.0
    for i, y in enumerate(ys):
        run += y
        if i >= window:
            run -= ys[i - window]
        out.append(run / min(i + 1, window))
    return out


def draw(data, title, out_path, smooth_window):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))

    ax1.plot(data["step"], data["total"], color="#bbbbbb", linewidth=0.8, label="total (raw)")
    ax1.plot(data["step"], smooth(data["total"], smooth_window), color="#C44E52",
             linewidth=1.8, label="total (smoothed)")
    ax1.set_title(f"{title}: total loss")
    ax1.set_xlabel("step")
    ax1.set_ylabel("loss")
    ax1.legend(fontsize=8)
    ax1.grid(alpha=0.25)

    for name in COMPONENTS:
        if name in data:
            ax2.plot(data["step"], smooth(data[name], smooth_window), linewidth=1.3, label=name)
    ax2.set_title(f"{title}: loss components")
    ax2.set_xlabel("step")
    ax2.set_ylabel("loss")
    ax2.legend(fontsize=8)
    ax2.grid(alpha=0.25)

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=160)
    plt.close(fig)
    print(f"wrote {out_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--language", required=True, choices=["English", "Telugu"])
    ap.add_argument("--logs", default=None)
    ap.add_argument("--results", default=None)
    ap.add_argument("--smooth", type=int, default=20)
    args = ap.parse_args()

    log_dir = Path(args.logs) if args.logs else ROOT / "logs" / args.language
    res_dir = Path(args.results) if args.results else ROOT / "project" / "Results" / args.language

    for csv_name, png_name, label, window in [
        ("losses_train.csv", "training_loss.png", f"{args.language} training", args.smooth),
        ("losses_val.csv", "validation_loss.png", f"{args.language} validation", 1),
    ]:
        path = log_dir / csv_name
        if not path.exists():
            print(f"missing {path}, skipping")
            continue
        data = read_csv(path)
        if data is None:
            print(f"{path} is empty, skipping")
            continue
        draw(data, label, res_dir / png_name, window)


if __name__ == "__main__":
    main()

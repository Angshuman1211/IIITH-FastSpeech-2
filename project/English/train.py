#!/usr/bin/env python
"""Train the English FastSpeech 2 model.

    python train.py                 # from scratch
    python train.py --restore_step 20000
"""

import argparse
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "fastspeech2"))

from config_utils import load_configs  # noqa: E402
from trainer import train  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(HERE / "config.yaml"))
    ap.add_argument("--restore_step", type=int, default=0)
    ap.add_argument("--set", action="append", default=[],
                    help="override a config value, e.g. --set train.path.ckpt_path=/kaggle/working/ckpt")
    args = ap.parse_args()

    configs = load_configs(args.config, args.set)
    train(args, configs)


if __name__ == "__main__":
    main()

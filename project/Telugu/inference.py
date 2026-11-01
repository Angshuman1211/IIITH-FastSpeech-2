#!/usr/bin/env python
"""Synthesise the Telugu test sentences and plot the variance adaptor output.

    python inference.py --restore_step 80000
"""

import argparse
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "fastspeech2"))

from inference_lib import run_inference  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(HERE / "config.yaml"))
    ap.add_argument("--restore_step", type=int, required=True)
    ap.add_argument(
        "--sentences", default=str(HERE.parent / "test_sentences" / "test_sentences.json")
    )
    ap.add_argument("--results", default=str(HERE.parent / "Results" / "Telugu"))
    ap.add_argument("--set", action="append", default=[],
                    help="override a config value")
    args = ap.parse_args()

    run_inference("telugu", args.config, args.restore_step, args.sentences, args.results, args.set)


if __name__ == "__main__":
    main()

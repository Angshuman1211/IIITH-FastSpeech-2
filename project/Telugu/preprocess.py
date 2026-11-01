#!/usr/bin/env python
"""Preprocess the Telugu corpus into FastSpeech 2 training features.

Expects the corpus and the MFA alignments to exist already:

    python ../../scripts/prepare_telugu.py
    python ../../scripts/build_telugu_lexicon.py --rewrite-labs
    mfa train ../../raw_data/Telugu ../../lexicon/telugu_lexicon.txt \
        ../../mfa_data/telugu_acoustic.zip \
        --output_directory ../../mfa_data/textgrids_telugu

Then:

    python preprocess.py
"""

import argparse
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "fastspeech2"))

from config_utils import load_configs  # noqa: E402
from preprocessor.preprocessor import Preprocessor  # noqa: E402


def link_textgrids(preprocess_config):
    """The preprocessor looks for TextGrids under
    <preprocessed_path>/TextGrid/<speaker>/. MFA writes them somewhere else, so
    point one at the other instead of copying thousands of small files."""
    src = Path(preprocess_config["path"]["textgrid_path"])
    dst = Path(preprocess_config["path"]["preprocessed_path"]) / "TextGrid"

    if not src.exists():
        sys.exit(f"MFA alignments not found at {src}. Run the mfa train step first.")

    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.is_symlink() or dst.exists():
        if dst.is_symlink():
            dst.unlink()
        else:
            sys.exit(f"{dst} exists and is not a symlink; remove it and rerun.")
    os.symlink(src, dst, target_is_directory=True)

    n = len(list(src.rglob("*.TextGrid")))
    print(f"linked {n} TextGrids: {dst} -> {src}")
    if n == 0:
        sys.exit("No TextGrids found. Alignment produced nothing.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(HERE / "config.yaml"))
    args = ap.parse_args()

    preprocess_config, _, _ = load_configs(args.config)
    os.makedirs(preprocess_config["path"]["preprocessed_path"], exist_ok=True)
    link_textgrids(preprocess_config)

    Preprocessor(preprocess_config).build_from_path()


if __name__ == "__main__":
    main()

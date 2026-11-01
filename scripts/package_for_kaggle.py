"""Assemble a Kaggle dataset containing everything needed to train on their GPU.

Training on Kaggle needs the code, the preprocessed features and the vocoder,
but none of the raw audio or the parquet shards. This stages exactly that and
can upload it with the Kaggle API.

    python scripts/package_for_kaggle.py --language Telugu
    python scripts/package_for_kaggle.py --language Telugu --upload

--upload needs ~/.kaggle/kaggle.json, from kaggle.com/settings -> API ->
Create New Token.
"""

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# Only the universal vocoder is used; the LJSpeech one and the zips are 100 MB
# of dead weight in the upload.
SKIP_NAMES = {
    "generator_LJSpeech.pth.tar",
    "generator_LJSpeech.pth.tar.zip",
    "generator_universal.pth.tar.zip",
    "__pycache__",
    ".DS_Store",
    "TextGrid",
}


def ignore(_dir, names):
    return [n for n in names if n in SKIP_NAMES or n.endswith(".pyc")]


def stage(language, staging):
    if staging.exists():
        shutil.rmtree(staging)
    staging.mkdir(parents=True)

    shutil.copytree(ROOT / "project" / "fastspeech2", staging / "project" / "fastspeech2",
                    ignore=ignore)
    shutil.copytree(ROOT / "project" / "test_sentences", staging / "project" / "test_sentences",
                    ignore=ignore)
    shutil.copytree(ROOT / "project" / language, staging / "project" / language, ignore=ignore)
    shutil.copytree(ROOT / "lexicon", staging / "lexicon", ignore=ignore)

    src = ROOT / "preprocessed" / language
    if not src.exists():
        sys.exit(f"{src} missing. Run the preprocess step first.")
    shutil.copytree(src, staging / "preprocessed" / language, ignore=ignore,
                    symlinks=False, ignore_dangling_symlinks=True)

    # sanity: the vocoder has to be there or training dies at step 0
    voc = staging / "project" / "fastspeech2" / "hifigan" / "generator_universal.pth.tar"
    if not voc.exists():
        sys.exit(f"vocoder missing at {voc}. Unzip it in project/fastspeech2/hifigan/ first.")

    n_mel = len(list((staging / "preprocessed" / language / "mel").glob("*.npy")))
    size_mb = sum(f.stat().st_size for f in staging.rglob("*") if f.is_file()) / 1e6
    print(f"staged {language}: {n_mel} mel files, {size_mb:.0f} MB -> {staging}")
    return staging


def write_metadata(staging, slug, title):
    meta = {"title": title, "id": slug, "licenses": [{"name": "CC0-1.0"}]}
    (staging / "dataset-metadata.json").write_text(json.dumps(meta, indent=1))
    print(f"wrote dataset-metadata.json (id={slug})")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--language", required=True, choices=["English", "Telugu"])
    ap.add_argument("--user", help="your Kaggle username, required for --upload")
    ap.add_argument("--upload", action="store_true")
    ap.add_argument("--staging", default=None)
    args = ap.parse_args()

    staging = Path(args.staging) if args.staging else ROOT / "kaggle_pkg" / args.language.lower()
    stage(args.language, staging)

    if not args.upload:
        print("\nnot uploading. Re-run with --upload --user <kaggle-username> when ready.")
        return

    if not args.user:
        sys.exit("--upload needs --user <kaggle-username>")
    if not (Path.home() / ".kaggle" / "kaggle.json").exists():
        sys.exit("~/.kaggle/kaggle.json not found. Create a token at kaggle.com/settings -> API.")

    slug = f"{args.user}/fs2-{args.language.lower()}"
    write_metadata(staging, slug, f"FastSpeech2 {args.language} preprocessed")

    cmd = ["kaggle", "datasets", "create", "-p", str(staging), "--dir-mode", "zip"]
    print("running:", " ".join(cmd))
    r = subprocess.run(cmd)
    if r.returncode != 0:
        print("\ncreate failed. If the dataset already exists, push a new version with:")
        print(f"  kaggle datasets version -p {staging} -m 'update' --dir-mode zip")


if __name__ == "__main__":
    main()

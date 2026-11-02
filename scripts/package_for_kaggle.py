"""Assemble a Kaggle dataset containing everything needed to train on their GPU.

Training on Kaggle needs the code, the preprocessed features and the vocoder,
but none of the raw audio or the parquet shards. This stages exactly that and
can upload it with the Kaggle API.

    python scripts/package_for_kaggle.py --language Telugu
    python scripts/package_for_kaggle.py --language Telugu --upload

--upload needs a Kaggle credential, which is either ~/.kaggle/kaggle.json or
the newer ~/.kaggle/access_token, both from kaggle.com/settings -> API.
"""

import argparse
import json
import shutil
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


def have_credentials():
    """The classic client reads kaggle.json, newer ones read access_token."""
    d = Path.home() / ".kaggle"
    return (d / "kaggle.json").exists() or (d / "access_token").exists()


def upload(staging, slug, new):
    """Create or version the dataset through the Python API, so this does not
    depend on the kaggle CLI being on PATH."""
    from kaggle.api.kaggle_api_extended import KaggleApi

    api = KaggleApi()
    api.authenticate()

    # Subdirectories are uploaded as zips; Kaggle expands them on its side.
    if new:
        print(f"creating {slug} (private) ...")
        api.dataset_create_new(str(staging), public=False, quiet=False, dir_mode="zip")
    else:
        print(f"pushing a new version of {slug} ...")
        api.dataset_create_version(str(staging), "update", quiet=False, dir_mode="zip")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--language", required=True, choices=["English", "Telugu"])
    ap.add_argument("--user", help="your Kaggle username, required for --upload")
    ap.add_argument("--upload", action="store_true")
    ap.add_argument("--version", action="store_true",
                    help="push a new version instead of creating the dataset")
    ap.add_argument("--staging", default=None)
    args = ap.parse_args()

    staging = Path(args.staging) if args.staging else ROOT / "kaggle_pkg" / args.language.lower()
    stage(args.language, staging)

    if not args.upload:
        print("\nnot uploading. Re-run with --upload --user <kaggle-username> when ready.")
        return

    if not args.user:
        sys.exit("--upload needs --user <kaggle-username>")
    if not have_credentials():
        sys.exit("no Kaggle credential in ~/.kaggle (kaggle.json or access_token). "
                 "Create one at kaggle.com/settings -> API.")

    slug = f"{args.user}/fs2-{args.language.lower()}"
    write_metadata(staging, slug, f"FastSpeech2 {args.language} preprocessed")
    upload(staging, slug, new=not args.version)


if __name__ == "__main__":
    main()

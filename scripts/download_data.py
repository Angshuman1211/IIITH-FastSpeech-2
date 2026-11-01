"""Download the two source corpora from HuggingFace.

Telugu (SPRINGLab/IndicTTS_Telugu) is open.
English (ai4bharat/Svarah) is auto-gated: accept the terms once at
https://huggingface.co/datasets/ai4bharat/Svarah then `hf auth login`.

Usage:
    python scripts/download_data.py telugu
    python scripts/download_data.py english
"""

import sys
from pathlib import Path

from huggingface_hub import snapshot_download

ROOT = Path(__file__).resolve().parent.parent

REPOS = {
    "telugu": ("SPRINGLab/IndicTTS_Telugu", ROOT / "data" / "indictts_telugu"),
    "english": ("ai4bharat/Svarah", ROOT / "data" / "svarah"),
}


def main(which):
    repo_id, dest = REPOS[which]
    dest.mkdir(parents=True, exist_ok=True)
    print(f"downloading {repo_id} -> {dest}", flush=True)
    path = snapshot_download(
        repo_id=repo_id,
        repo_type="dataset",
        local_dir=str(dest),
        max_workers=4,
    )
    print(f"done: {path}", flush=True)


if __name__ == "__main__":
    if len(sys.argv) != 2 or sys.argv[1] not in REPOS:
        sys.exit(f"usage: {sys.argv[0]} {{{'|'.join(REPOS)}}}")
    main(sys.argv[1])

"""Turn the Svarah parquet shards into a FastSpeech 2 raw_data folder.

Svarah is an ASR evaluation benchmark, not a TTS corpus: 9.6 h of read and
spontaneous Indian-accented English spread over roughly 117 speakers, published
as a single `test` split. Its schema carries demographics (gender, age-group,
native place) but **no speaker id column**, so speaker identity has to be
recovered before anything multi-speaker can be trained.

Run the inspection pass first, which downloads nothing extra and just reports
what identity signal exists:

    python scripts/prepare_svarah.py --inspect

Then build the corpus:

    python scripts/prepare_svarah.py
"""

import argparse
import collections
import io
import json
import random
import re
import sys
import wave
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import soundfile as sf
from scipy.signal import resample_poly

ROOT = Path(__file__).resolve().parent.parent
SHARD_DIR = ROOT / "data" / "svarah" / "data"

MIN_SEC, MAX_SEC = 1.0, 15.0
DST_SR = 22050
VAL_HOURS, TRAIN_HOURS = 1.0, 5.0
SEED = 1234

DEMOGRAPHIC_KEYS = [
    "gender",
    "age-group",
    "primary_language",
    "native_place_state",
    "native_place_district",
    "highest_qualification",
    "job_category",
    "occupation_domain",
]

_ABBREV = re.compile(r"[^a-z' ]+")


def clean_text(t):
    """Lowercase, expand digits, drop everything the ARPAbet lexicon cannot use."""
    import inflect

    p = inflect.engine()
    t = t.lower().strip()
    t = re.sub(r"(\d+)", lambda m: " " + p.number_to_words(m.group(1)) + " ", t)
    t = t.replace("-", " ")
    t = _ABBREV.sub(" ", t)
    return re.sub(r"\s+", " ", t).strip()


def speaker_from_path(path):
    """Svarah filenames look like <speaker>_<utt>.wav in the original manifest.
    Take everything before the last underscore-number group as the speaker."""
    stem = Path(path).stem
    m = re.match(r"^(.*?)[_-]\d+$", stem)
    return m.group(1) if m else stem


def load_rows(columns=None):
    shards = sorted(SHARD_DIR.glob("*.parquet"))
    if not shards:
        sys.exit(
            f"No parquet files under {SHARD_DIR}.\n"
            "Svarah is auto-gated: accept the terms at\n"
            "  https://huggingface.co/datasets/ai4bharat/Svarah\n"
            "then run `hf auth login` and `python scripts/download_data.py english`."
        )
    rows = []
    for p in shards:
        pf = pq.ParquetFile(p)
        names = set(pf.schema_arrow.names)
        want = [c for c in (columns or names) if c in names]
        for rg in range(pf.metadata.num_row_groups):
            tb = pf.read_row_group(rg, columns=want)
            rows.extend(tb.to_pylist())
    return rows


def audio_field(row):
    for k in ("audio_filepath", "audio"):
        if k in row and isinstance(row[k], dict):
            return row[k]
    return None


def inspect():
    rows = load_rows()
    print(f"rows: {len(rows)}")
    print(f"columns: {sorted(rows[0])}\n")

    a = audio_field(rows[0])
    print(f"sample audio path : {a.get('path') if a else 'n/a'}")
    print(f"sample text       : {rows[0].get('text')!r}\n")

    by_path = collections.Counter(speaker_from_path(audio_field(r)["path"]) for r in rows)
    print(f"distinct speakers from filename : {len(by_path)}")
    print(f"  most common: {by_path.most_common(5)}")

    present = [k for k in DEMOGRAPHIC_KEYS if k in rows[0]]
    by_demo = collections.Counter(tuple(str(r.get(k)) for k in present) for r in rows)
    print(f"distinct demographic tuples     : {len(by_demo)}  (from {present})")

    durs = [r["duration"] for r in rows if r.get("duration")]
    if durs:
        durs.sort()
        print(f"\ntotal duration: {sum(durs)/3600:.2f} h")
        for lo, hi in [(1, 10), (1, 12), (1, 15), (1, 20)]:
            sel = [d for d in durs if lo <= d <= hi]
            print(f"  band {lo}-{hi}s -> {len(sel)} utts, {sum(sel)/3600:.2f} h")


def build(out_dir, speaker_mode):
    rows = load_rows()
    present = [k for k in DEMOGRAPHIC_KEYS if k in rows[0]]

    def speaker_of(r):
        if speaker_mode == "filename":
            return speaker_from_path(audio_field(r)["path"])
        if speaker_mode == "demographic":
            return "spk_" + "_".join(
                re.sub(r"\W+", "", str(r.get(k)))[:8] for k in present
            )
        return "svarah"

    eligible = []
    for r in rows:
        a = audio_field(r)
        if not a or not a.get("bytes"):
            continue
        try:
            with wave.open(io.BytesIO(a["bytes"])) as w:
                dur = w.getnframes() / w.getframerate()
                sr = w.getframerate()
        except Exception:
            continue
        if not (MIN_SEC <= dur <= MAX_SEC):
            continue
        txt = clean_text(r.get("text") or "")
        if not txt:
            continue
        eligible.append(
            {"bytes": a["bytes"], "sr": sr, "dur": dur, "text": txt,
             "speaker": speaker_of(r), "stem": Path(a["path"]).stem}
        )

    print(f"eligible: {len(eligible)} utts, {sum(e['dur'] for e in eligible)/3600:.2f} h")
    print(f"speakers: {len(set(e['speaker'] for e in eligible))}")

    random.Random(SEED).shuffle(eligible)

    # Speaker-stratified: every speaker that appears must appear in both splits,
    # otherwise a multi-speaker model has validation speakers it never trained on.
    budget = {"val": VAL_HOURS * 3600, "train": TRAIN_HOURS * 3600}
    used = {"val": 0.0, "train": 0.0}
    counts = {"val": 0, "train": 0}
    seen_val = set()
    manifest = {}

    out_root = Path(out_dir)
    for e in eligible:
        if used["val"] >= budget["val"] and used["train"] >= budget["train"]:
            break

        x = np.frombuffer(e["bytes"], dtype=np.int16, offset=44).astype(np.float32) / 32768.0
        if e["sr"] != DST_SR:
            g = np.gcd(e["sr"], DST_SR)
            x = resample_poly(x, DST_SR // g, e["sr"] // g)
        import librosa

        x, _ = librosa.effects.trim(x, top_db=35)
        x = np.pad(x, (int(DST_SR * 0.05),) * 2)
        x = x / (np.abs(x).max() + 1e-9) * 0.95
        dur = len(x) / DST_SR

        # send a speaker's first utterance to val so val speakers are covered
        prefer = "val" if e["speaker"] not in seen_val else "train"
        order = (prefer, "train" if prefer == "val" else "val")
        split = next((s for s in order if used[s] + dur <= budget[s]), None)
        if split is None:
            continue
        if split == "val":
            seen_val.add(e["speaker"])

        d = out_root / e["speaker"]
        d.mkdir(parents=True, exist_ok=True)
        sf.write(d / f"{e['stem']}.wav", x, DST_SR, subtype="PCM_16")
        (d / f"{e['stem']}.lab").write_text(e["text"], encoding="utf-8")
        manifest[e["stem"]] = {"split": split, "speaker": e["speaker"],
                               "text": e["text"], "dur_out": round(dur, 3)}
        used[split] += dur
        counts[split] += 1

    for s in ("train", "val"):
        print(f"{s}: {counts[s]} utts, {used[s]/3600:.2f} h (after trimming)")
    print(f"speakers in val: {len(seen_val)}")

    meta = out_root / "metadata.json"
    meta.write_text(json.dumps(manifest, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"manifest -> {meta}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--inspect", action="store_true", help="report speaker signal and exit")
    ap.add_argument("--out", default=str(ROOT / "raw_data" / "English"))
    ap.add_argument("--speaker-mode", default="filename",
                    choices=["filename", "demographic", "single"])
    args = ap.parse_args()

    if args.inspect:
        inspect()
    else:
        build(args.out, args.speaker_mode)

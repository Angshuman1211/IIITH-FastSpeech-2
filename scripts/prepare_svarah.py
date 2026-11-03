"""Build raw_data/English from the Svarah parquet shards.

Svarah has no speaker id column, so speakers are recovered from the
demographic fields. Run --inspect first to see what is there.

    python scripts/prepare_svarah.py --inspect
    python scripts/prepare_svarah.py
"""

import argparse
import collections
import io
import json
import random
import re
import sys
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

def clean_text(t):
    # shared with inference, so the two cannot drift apart
    import sys

    sys.path.insert(0, str(ROOT / "project" / "fastspeech2"))
    from text.english_norm import normalise_for_training

    return normalise_for_training(t)


def speaker_from_path(path):
    """Leading recording id. Not a speaker: 2938 distinct values over 6656
    utterances against 117 speakers. Kept for comparison only."""
    stem = Path(path).stem
    return stem.split("_")[0]


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


def probe(blob):
    """Duration and rate without decoding. soundfile reads the real header,
    which a fixed 44-byte offset would get wrong."""
    info = sf.info(io.BytesIO(blob))
    return info.frames / info.samplerate, info.samplerate


def decode(blob):
    """Mono float32 samples at the file's own rate."""
    x, sr = sf.read(io.BytesIO(blob), dtype="float32", always_2d=True)
    return x.mean(axis=1), sr


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

    # count every drop reason: only 9.6 h available against 6 h needed
    eligible = []
    skipped = collections.Counter()
    for r in rows:
        a = audio_field(r)
        if not a or not a.get("bytes"):
            skipped["no audio bytes"] += 1
            continue
        try:
            dur, sr = probe(a["bytes"])
        except Exception as exc:
            skipped[f"unreadable ({type(exc).__name__})"] += 1
            continue
        if not (MIN_SEC <= dur <= MAX_SEC):
            skipped[f"outside {MIN_SEC}-{MAX_SEC}s"] += 1
            continue
        txt = clean_text(r.get("text") or "")
        if not txt:
            skipped["empty text"] += 1
            continue
        eligible.append(
            {"bytes": a["bytes"], "sr": sr, "dur": dur, "text": txt,
             "speaker": speaker_of(r), "stem": Path(a["path"]).stem}
        )

    print(f"eligible: {len(eligible)} utts, {sum(e['dur'] for e in eligible)/3600:.2f} h")
    print(f"speakers: {len(set(e['speaker'] for e in eligible))}")
    if skipped:
        print("skipped:")
        for reason, n in skipped.most_common():
            print(f"  {n:>6}  {reason}")

    random.Random(SEED).shuffle(eligible)

    # stratified, so val has no speaker the model never trained on
    budget = {"val": VAL_HOURS * 3600, "train": TRAIN_HOURS * 3600}
    used = {"val": 0.0, "train": 0.0}
    counts = {"val": 0, "train": 0}
    seen_val = set()
    manifest = {}

    out_root = Path(out_dir)
    for e in eligible:
        if used["val"] >= budget["val"] and used["train"] >= budget["train"]:
            break

        x, sr = decode(e["bytes"])
        if sr != DST_SR:
            g = np.gcd(sr, DST_SR)
            x = resample_poly(x, DST_SR // g, sr // g)
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
    # demographic: the only grouping near the 117 speakers in the paper
    ap.add_argument("--speaker-mode", default="demographic",
                    choices=["filename", "demographic", "single"])
    args = ap.parse_args()

    if args.inspect:
        inspect()
    else:
        build(args.out, args.speaker_mode)

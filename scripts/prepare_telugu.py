"""Turn the IndicTTS Telugu parquet shards into a FastSpeech 2 raw_data folder.

Data selection (justified in the report):
  * Only the male speaker (gender=1). The female speaker averages 21.3 s per
    utterance and yields only 2.31 h once filtered to a trainable length band,
    which is not enough for the required 5 h + 1 h. The male speaker yields
    6.86 h in the same band.
  * Only utterances between MIN_SEC and MAX_SEC. Very long utterances hurt
    forced alignment and blow up memory during training.
  * Seeded shuffle, then fill the validation hour first and the training hours
    after, so the two sets are disjoint by construction.

Audio is resampled 16 kHz -> 22.05 kHz to match the pretrained HiFi-GAN vocoder.
This adds no new information above 8 kHz and the report says so explicitly.

Output layout (what MFA and ming024/FastSpeech2 expect):
    raw_data/Telugu/telugumale/<basename>.wav
    raw_data/Telugu/telugumale/<basename>.lab
"""

import argparse
import io
import json
import random
import re
import wave
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import soundfile as sf
from scipy.signal import resample_poly

ROOT = Path(__file__).resolve().parent.parent
SHARDS = sorted((ROOT / "data" / "indictts_telugu" / "data").glob("*.parquet"))

SPEAKER = "telugumale"
GENDER = 1
MIN_SEC, MAX_SEC = 1.0, 15.0
SRC_SR, DST_SR = 16000, 22050
VAL_HOURS, TRAIN_HOURS = 1.0, 5.0
SEED = 1234

# Telugu block, digits, and the punctuation the corpus actually uses.
KEEP = re.compile(r"[^ఀ-౿0-9\s।\.,\?\!:\-]")


def clean_text(t):
    t = t.replace("‌", "").replace("‍", "")
    t = KEEP.sub(" ", t)
    return re.sub(r"\s+", " ", t).strip()


def trim_silence(x, sr, top_db=35, pad_ms=50):
    import librosa

    y, _ = librosa.effects.trim(x, top_db=top_db)
    pad = int(sr * pad_ms / 1000)
    return np.pad(y, (pad, pad), mode="constant")


def load_rows():
    rows = []
    for p in SHARDS:
        pf = pq.ParquetFile(p)
        for rg in range(pf.metadata.num_row_groups):
            tb = pf.read_row_group(rg, columns=["audio", "gender", "text"])
            for a, g, tx in zip(tb["audio"], tb["gender"], tb["text"]):
                if g.as_py() != GENDER:
                    continue
                a = a.as_py()
                b = a["bytes"]
                try:
                    with wave.open(io.BytesIO(b)) as w:
                        dur = w.getnframes() / w.getframerate()
                        sr = w.getframerate()
                except Exception:
                    continue
                if not (MIN_SEC <= dur <= MAX_SEC):
                    continue
                txt = clean_text(tx.as_py())
                if not txt:
                    continue
                rows.append({"path": a["path"], "bytes": b, "dur": dur, "sr": sr, "text": txt})
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(ROOT / "raw_data" / "Telugu"))
    args = ap.parse_args()

    out = Path(args.out) / SPEAKER
    out.mkdir(parents=True, exist_ok=True)

    rows = load_rows()
    print(f"eligible utterances: {len(rows)}  ({sum(r['dur'] for r in rows)/3600:.2f} h)")

    random.Random(SEED).shuffle(rows)

    # Budgets are filled using the duration AFTER trimming, not before. Trimming
    # removes roughly 14% of this corpus, so budgeting on the raw duration
    # undershoots the required 5 h + 1 h.
    budget = {"val": VAL_HOURS * 3600, "train": TRAIN_HOURS * 3600}
    used = {"val": 0.0, "train": 0.0}
    counts = {"val": 0, "train": 0}
    manifest = {}

    for r in rows:
        if used["val"] >= budget["val"] and used["train"] >= budget["train"]:
            break

        base = Path(r["path"]).stem
        with wave.open(io.BytesIO(r["bytes"])) as w:
            x = np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16)
        x = x.astype(np.float32) / 32768.0
        if r["sr"] != DST_SR:
            g = np.gcd(r["sr"], DST_SR)
            x = resample_poly(x, DST_SR // g, r["sr"] // g)
        x = trim_silence(x, DST_SR)
        x = x / (np.abs(x).max() + 1e-9) * 0.95
        dur = len(x) / DST_SR

        split = None
        for name in ("val", "train"):
            if used[name] + dur <= budget[name]:
                split = name
                break
        if split is None:
            continue

        sf.write(out / f"{base}.wav", x, DST_SR, subtype="PCM_16")
        (out / f"{base}.lab").write_text(r["text"], encoding="utf-8")
        manifest[base] = {
            "split": split,
            "text": r["text"],
            "dur_src": round(r["dur"], 3),
            "dur_out": round(dur, 3),
        }
        used[split] += dur
        counts[split] += 1

    for name in ("train", "val"):
        print(f"{name}: {counts[name]} utts, {used[name]/3600:.2f} h (after trimming)")

    meta = Path(args.out) / "metadata.json"
    meta.write_text(json.dumps(manifest, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"wrote {len(manifest)} wav/lab pairs -> {out}")
    print(f"manifest -> {meta}")


if __name__ == "__main__":
    main()

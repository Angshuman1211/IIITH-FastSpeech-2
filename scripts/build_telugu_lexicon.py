"""Build the MFA pronunciation dictionary for Telugu from the corpus transcripts.

MFA needs every word in the transcripts to have an entry. There is no published
Telugu dictionary, so this generates one with the rule-based G2P in
scripts/telugu_g2p.py.

It also reports any transcript token that is not pure Telugu script (digits,
Latin characters), because MFA will fail on those and they have to be either
normalised or dropped before alignment.
"""

import argparse
import collections
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from telugu_g2p import g2p_word, phone_inventory  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent

PURE_TELUGU = re.compile(r"^[ఀ-౿]+$")
HAS_DIGIT = re.compile(r"[0-9]")
HAS_LATIN = re.compile(r"[A-Za-z]")

# Punctuation is stripped before alignment. MFA aligns phones, and sentence
# breaks come back as sp/sil intervals in the TextGrid, so the marks themselves
# carry nothing into the phone sequence.
PUNCT = ".,?!:;।॥\"'()[]{}-–—…"
SPLIT_PUNCT = re.compile(f"[{re.escape(PUNCT)}]+")

# A token has to carry at least one consonant or independent vowel. Tokens that
# are nothing but a diacritic (a stray anusvara, say) are not pronounceable
# words and would otherwise become bogus lexicon entries.
BASE_CHAR = re.compile(r"[అ-ఔక-హౘ-ౚౠౡ]")


def normalise(tok):
    """Split on internal punctuation and drop anything unpronounceable.

    Abbreviations such as ఐ.టి.ఐ are written with internal periods. MFA needs
    bare words, so they are split into their parts rather than discarded.
    """
    out = []
    for part in SPLIT_PUNCT.split(tok):
        part = part.strip()
        if part and BASE_CHAR.search(part):
            out.append(part)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lab-dir", default=str(ROOT / "raw_data" / "Telugu" / "telugumale"))
    ap.add_argument("--out", default=str(ROOT / "lexicon" / "telugu_lexicon.txt"))
    ap.add_argument("--rewrite-labs", action="store_true",
                    help="rewrite each .lab with punctuation stripped, ready for MFA")
    args = ap.parse_args()

    lab_dir = Path(args.lab_dir)
    labs = sorted(lab_dir.glob("*.lab"))
    if not labs:
        sys.exit(f"no .lab files under {lab_dir}")

    counts = collections.Counter()
    for p in labs:
        toks = [sub for t in p.read_text(encoding="utf-8").split() for sub in normalise(t)]
        counts.update(toks)
        if args.rewrite_labs:
            p.write_text(" ".join(toks), encoding="utf-8")

    clean, digits, latin, other = {}, collections.Counter(), collections.Counter(), collections.Counter()
    for w, c in counts.items():
        if PURE_TELUGU.match(w):
            clean[w] = c
        elif HAS_DIGIT.search(w):
            digits[w] = c
        elif HAS_LATIN.search(w):
            latin[w] = c
        else:
            other[w] = c

    print(f"transcript files     : {len(labs)}")
    print(f"unique tokens        : {len(counts)}")
    print(f"  pure Telugu        : {len(clean)}")
    print(f"  containing digits  : {len(digits)}  (files affected below)")
    print(f"  containing Latin   : {len(latin)}")
    print(f"  other/punctuation  : {len(other)}")
    if digits:
        print(f"  digit examples     : {list(digits)[:10]}")
    if latin:
        print(f"  latin examples     : {list(latin)[:10]}")
    if other:
        print(f"  other examples     : {list(other)[:10]}")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with out.open("w", encoding="utf-8") as f:
        for w in sorted(clean):
            phones = g2p_word(w)
            if not phones:
                continue
            f.write(f"{w}\t{' '.join(phones)}\n")
            n += 1

    print(f"\nwrote {n} entries -> {out}")
    print(f"phone inventory ({len(phone_inventory())}): {' '.join(phone_inventory())}")

    # which files would break alignment
    bad = [w for w in list(digits) + list(latin)]
    if bad:
        bad_set = set(bad)
        affected = [p.name for p in labs if bad_set & set(p.read_text(encoding="utf-8").split())]
        print(f"\nfiles containing out-of-dictionary tokens: {len(affected)} / {len(labs)}")
        for name in affected[:10]:
            print(f"  {name}")


if __name__ == "__main__":
    main()

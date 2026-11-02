"""Extend the MFA english_us_arpa dictionary to cover Svarah's vocabulary.

Svarah is Indian-English over government services and digital payments, so a
US English dictionary misses a lot of it: Indian place names, scheme acronyms
such as EPFO and PRAN, and a few ordinary words the dictionary simply lacks.
Measured against the 208651-entry dictionary, 4.53% of tokens are
out-of-vocabulary and 33.1% of utterances contain at least one.

That matters because an out-of-vocabulary word either drops the utterance or
aligns against an unknown phone, and either way the phoneme durations the
variance adaptor trains on are wrong.

    python scripts/extend_english_lexicon.py --collect     # write the OOV list
    mfa g2p oov_words.txt english_us_arpa oov_pronunciations.dict
    python scripts/extend_english_lexicon.py --merge       # build the dictionary

Short tokens also get a spelled-out pronunciation as an alternative, because
the G2P treats every initialism as a word and gets them wrong: it gives otp as
"oh tee p" and upi as "uh pee" rather than naming the letters.

Spelling cannot tell an initialism from an ordinary word here, since the short
OOV list mixes epfo, cbse and gps with dosa, aloo and dahi. It does not need
to. MFA allows several pronunciations per word and the aligner picks whichever
fits the audio, so a candidate that is wrong is simply not chosen. Every OOV
token of 4 characters or fewer therefore gets the letter-by-letter form
alongside the G2P one, and so does any longer token with no vowel, which is an
initialism beyond reasonable doubt.
"""

import argparse
import collections
import glob
import pathlib

ROOT = pathlib.Path(__file__).resolve().parent.parent
BASE_DICT = (pathlib.Path.home() / "Documents" / "MFA" / "pretrained_models"
             / "dictionary" / "english_us_arpa.dict")
LAB_GLOB = str(ROOT / "raw_data" / "English" / "*" / "*.lab")
OOV_WORDS = ROOT / "lexicon" / "english_oov_words.txt"
OOV_DICT = ROOT / "lexicon" / "english_oov_pronunciations.dict"
OUT_DICT = ROOT / "lexicon" / "english_svarah.dict"

# ARPAbet for each letter said aloud, for the initialism alternatives.
LETTER = {
    "a": "EY1", "b": "B IY1", "c": "S IY1", "d": "D IY1", "e": "IY1",
    "f": "EH1 F", "g": "JH IY1", "h": "EY1 CH", "i": "AY1", "j": "JH EY1",
    "k": "K EY1", "l": "EH1 L", "m": "EH1 M", "n": "EH1 N", "o": "OW1",
    "p": "P IY1", "q": "K Y UW1", "r": "AA1 R", "s": "EH1 S", "t": "T IY1",
    "u": "Y UW1", "v": "V IY1", "w": "D AH1 B AH0 L Y UW0", "x": "EH1 K S",
    "y": "W AY1", "z": "Z IY1",
}
VOWELS = set("aeiou")


def base_words():
    return {l.split()[0].lower() for l in open(BASE_DICT, encoding="utf-8") if l.split()}


def find_oov():
    known = base_words()
    oov = collections.Counter()
    n_utt = n_tok = 0
    for f in glob.glob(LAB_GLOB):
        n_utt += 1
        for t in open(f, encoding="utf-8").read().split():
            n_tok += 1
            if t not in known:
                oov[t] += 1
    return oov, known, n_utt, n_tok


def spelled_out(word):
    """Letter-by-letter ARPAbet, or None if any character has no letter name."""
    out = []
    for ch in word:
        if ch not in LETTER:
            return None
        out.append(LETTER[ch])
    return " ".join(out)


def collect():
    oov, _, n_utt, n_tok = find_oov()
    OOV_WORDS.parent.mkdir(parents=True, exist_ok=True)
    with open(OOV_WORDS, "w", encoding="utf-8") as f:
        for w in sorted(oov):
            f.write(w + "\n")
    n_oov = sum(oov.values())
    print(f"utterances {n_utt}, tokens {n_tok}")
    print(f"OOV {n_oov} tokens ({100*n_oov/n_tok:.2f}%), {len(oov)} distinct")
    print(f"wrote {len(oov)} words -> {OOV_WORDS}")
    print(f"\nnow run:\n  mfa g2p {OOV_WORDS} english_us_arpa {OOV_DICT}")


def merge():
    if not OOV_DICT.exists():
        raise SystemExit(f"{OOV_DICT} missing. Run the mfa g2p step first.")

    lines = []
    with open(BASE_DICT, encoding="utf-8") as f:
        lines.extend(f.read().splitlines())
    n_base = len(lines)

    # The G2P output is "word PHONES", without the four probability columns the
    # pretrained dictionary carries. MFA accepts both shapes in one file.
    added = 0
    g2p_words = set()
    with open(OOV_DICT, encoding="utf-8") as f:
        for line in f:
            parts = line.split()
            if len(parts) < 2:
                continue
            g2p_words.add(parts[0].lower())
            lines.append(line.rstrip("\n"))
            added += 1

    initialisms = 0
    for w in sorted(g2p_words):
        if len(w) <= 4 or (len(w) <= 6 and not (set(w) & VOWELS)):
            spell = spelled_out(w)
            if spell:
                lines.append(f"{w}\t{spell}")
                initialisms += 1

    OUT_DICT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"base entries      : {n_base}")
    print(f"g2p entries added : {added}")
    print(f"spelled-out alts  : {initialisms}")
    print(f"total             : {len(lines)}  -> {OUT_DICT}")

    # Confirm the result actually covers the corpus.
    known = {l.split()[0].lower() for l in lines if l.split()}
    still = collections.Counter()
    for f in glob.glob(LAB_GLOB):
        for t in open(f, encoding="utf-8").read().split():
            if t not in known:
                still[t] += 1
    if still:
        print(f"\nSTILL OOV: {sum(still.values())} tokens, {len(still)} distinct")
        print("  ", still.most_common(10))
    else:
        print("\nzero out-of-vocabulary tokens remain")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--collect", action="store_true")
    ap.add_argument("--merge", action="store_true")
    a = ap.parse_args()
    if a.collect:
        collect()
    elif a.merge:
        merge()
    else:
        ap.error("pass --collect or --merge")

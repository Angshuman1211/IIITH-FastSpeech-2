"""English number and date expansion, shared by data preparation and inference.

This exists because the two paths were inconsistent. Every Svarah training
transcript was expanded before alignment, so the model only ever saw words,
but the reference's `preprocess_english` hands raw text to the lexicon and
`g2p_en`. Test sentence E4 contains "10:45 AM on 12 November 2026" and came
out as nonsense, with "10:45" becoming T ER0 F AY1 T IH0 V and 2026 becoming
S K AY1 Z.

So both paths now call the same expansion, and the Telugu side already works
this way through `text/telugu_norm.py`.

Two entry points, because training and inference want different things from
punctuation. Training strips it, since the `.lab` transcripts feed a forced
aligner that has no symbol for a comma. Inference keeps it, because the
reference turns punctuation into `sp` tokens and the model saw `sp` during
training wherever MFA measured a pause.
"""

import re

_INFLECT = None

_KEEP = re.compile(r"[^a-z' ]+")
_UNITS = ("rupees|rs|dollars|crore|lakh|km|kg|people|students|users|seats|"
          "metres|meters|feet|units|mah|watts")


def _engine():
    global _INFLECT
    if _INFLECT is None:
        import inflect

        _INFLECT = inflect.engine()
    return _INFLECT


def _digits(d):
    p = _engine()
    return " " + " ".join(p.number_to_words(c) for c in d) + " "


def expand_numbers(text):
    """Turn every digit run into words, leaving case and punctuation alone."""
    p = _engine()

    # Ordinals first, before the generic rule can split them. "7th July"
    # otherwise becomes "seven th july", because expanding the digits leaves
    # the suffix behind as its own token.
    text = re.sub(
        r"(\d+)\s*(?:st|nd|rd|th)\b",
        lambda m: " " + p.ordinal(p.number_to_words(m.group(1))) + " ",
        text,
        flags=re.I,
    )

    # Times, so the colon never survives to become an sp token. Without this
    # "10:45" expands as two separate numbers and leaves the colon between
    # them, which the inference front end turns into a pause in the middle of
    # the time. The training path strips punctuation and so never saw this.
    def say_time(m):
        h, mi = int(m.group(1)), int(m.group(2))
        if mi == 0:
            return f" {p.number_to_words(h)} o'clock "
        if mi < 10:
            return f" {p.number_to_words(h)} oh {p.number_to_words(mi)} "
        return f" {p.number_to_words(h)} {p.number_to_words(mi)} "

    text = re.sub(r"\b(\d{1,2}):(\d{2})\b", say_time, text)

    # A 4-digit run next to one of these words is an identifier read digit by
    # digit, not a quantity, and it is too short for the length rule below.
    text = re.sub(
        r"\b(m?pin|otp|code|cvv)\b([^0-9]{0,20}?)(\d{4})\b",
        lambda m: m.group(1) + m.group(2) + _digits(m.group(3)),
        text,
        flags=re.I,
    )

    # Years are said in two halves. 1984 is "nineteen eighty four", and on the
    # training corpus the long reading cost a median phone duration deviation
    # of 3.61 against 2.29 for everything else. Range-gated, and suppressed
    # when a unit follows so quantities stay quantities.
    def say_year(m):
        n = int(m.group(1))
        hi, lo = divmod(n, 100)
        if lo == 0:
            return f" {p.number_to_words(hi)} hundred "
        if 2000 <= n <= 2009:
            return f" two thousand and {p.number_to_words(lo)} "
        if lo < 10:
            return f" {p.number_to_words(hi)} oh {p.number_to_words(lo)} "
        return f" {p.number_to_words(hi)} {p.number_to_words(lo)} "

    text = re.sub(
        r"\b(1[1-9]\d{2}|20[0-9]\d)\b(?!\s*(?:" + _UNITS + r"))",
        say_year,
        text,
        flags=re.I,
    )

    # Everything else. A long run is an identifier, so it is spelled out,
    # unless it is round and therefore a real quantity however long.
    def say_number(m):
        d = m.group(1)
        if len(d) >= 5 and not d.endswith("000"):
            return _digits(d)
        return " " + p.number_to_words(d) + " "

    text = re.sub(r"(\d+)", say_number, text)
    return re.sub(r"\s+", " ", text).strip()


def normalise_for_training(text):
    """Lowercase words only, for the .lab transcripts MFA aligns against."""
    text = expand_numbers(text.lower().strip()).replace("-", " ")
    return re.sub(r"\s+", " ", _KEEP.sub(" ", text)).strip()


def normalise_for_inference(text):
    """Expand numbers but keep the punctuation the front end turns into sp."""
    return expand_numbers(text).replace("-", " ")

"""English number expansion, shared by prepare_svarah.py and inference.

Training transcripts were expanded before alignment, so the model only ever
saw words. Inference has to do the same or E4 comes out as nonsense.
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
    p = _engine()

    # before the generic rule, or "7th" splits into "seven th"
    text = re.sub(
        r"(\d+)\s*(?:st|nd|rd|th)\b",
        lambda m: " " + p.ordinal(p.number_to_words(m.group(1))) + " ",
        text,
        flags=re.I,
    )

    # keep the colon from surviving into an sp token
    def say_time(m):
        h, mi = int(m.group(1)), int(m.group(2))
        if mi == 0:
            return f" {p.number_to_words(h)} o'clock "
        if mi < 10:
            return f" {p.number_to_words(h)} oh {p.number_to_words(mi)} "
        return f" {p.number_to_words(h)} {p.number_to_words(mi)} "

    text = re.sub(r"\b(\d{1,2}):(\d{2})\b", say_time, text)

    # pins and OTPs are read digit by digit, and are too short for the rule below
    text = re.sub(
        r"\b(m?pin|otp|code|cvv)\b([^0-9]{0,20}?)(\d{4})\b",
        lambda m: m.group(1) + m.group(2) + _digits(m.group(3)),
        text,
        flags=re.I,
    )

    # 1984 is "nineteen eighty four", but 1500 rupees is still fifteen hundred
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

    # a long run is a phone number, unless it is round and so a real quantity
    def say_number(m):
        d = m.group(1)
        if len(d) >= 5 and not d.endswith("000"):
            return _digits(d)
        return " " + p.number_to_words(d) + " "

    text = re.sub(r"(\d+)", say_number, text)
    return re.sub(r"\s+", " ", text).strip()


def normalise_for_training(text):
    """Words only, for the .lab files MFA aligns against."""
    text = expand_numbers(text.lower().strip()).replace("-", " ")
    return re.sub(r"\s+", " ", _KEEP.sub(" ", text)).strip()


def normalise_for_inference(text):
    """Keeps punctuation, which the front end turns into sp."""
    return expand_numbers(text).replace("-", " ")

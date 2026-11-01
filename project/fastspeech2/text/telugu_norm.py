"""Telugu text normalisation for inference.

Test sentence T4 contains a year, an ordinal date and a clock time, none of
which exist in the training transcripts as digits, so they have to be expanded
into words before the G2P sees them. This covers 0-9999, which is everything
the test set needs.
"""

import re

UNITS = {
    0: "సున్నా", 1: "ఒకటి", 2: "రెండు", 3: "మూడు", 4: "నాలుగు",
    5: "ఐదు", 6: "ఆరు", 7: "ఏడు", 8: "ఎనిమిది", 9: "తొమ్మిది",
    10: "పది", 11: "పదకొండు", 12: "పన్నెండు", 13: "పదమూడు", 14: "పద్నాలుగు",
    15: "పదిహేను", 16: "పదహారు", 17: "పదిహేడు", 18: "పద్దెనిమిది", 19: "పందొమ్మిది",
}

TENS = {
    2: "ఇరవై", 3: "ముప్పై", 4: "నలభై", 5: "యాభై",
    6: "అరవై", 7: "డెబ్బై", 8: "ఎనభై", 9: "తొంభై",
}

# Ordinals are formed with the suffix -వ, written separately from the stem.
ORDINAL_SUFFIX = "వ"


def _below_hundred(n):
    if n < 20:
        return [UNITS[n]]
    t, u = divmod(n, 10)
    return [TENS[t]] if u == 0 else [TENS[t], UNITS[u]]


def number_to_words(n):
    """Expand an integer in 0..9999 into Telugu words."""
    if not 0 <= n <= 9999:
        raise ValueError(f"only 0-9999 supported, got {n}")
    if n < 100:
        return _below_hundred(n)

    out = []
    thousands, rest = divmod(n, 1000)
    if thousands:
        if thousands == 1:
            out.append("వెయ్యి")
        else:
            out += [UNITS[thousands], "వేల"]
    hundreds, rest = divmod(rest, 100)
    if hundreds:
        # a bare hundred is వంద, not "one hundreds"
        out += ["వంద"] if hundreds == 1 else [UNITS[hundreds], "వందల"]
    if rest:
        out += _below_hundred(rest)
    return out


def _expand_time(m):
    h, mm = int(m.group(1)), int(m.group(2))
    return " ".join(number_to_words(h) + number_to_words(mm))


def _expand_ordinal(m):
    """Telugu ordinals drop the cardinal's final -ు before -వ.

    పన్నెండు + వ -> పన్నెండవ, not పన్నెండువ.
    """
    n = int(m.group(1))
    if n == 1:
        return "మొదటి"  # irregular, not ఒకటి + వ
    words = number_to_words(n)
    stem = words[-1]
    if stem.endswith("ు"):  # vowel sign U
        stem = stem[:-1]
    return " ".join(words[:-1] + [stem + ORDINAL_SUFFIX])


def _expand_number(m):
    return " ".join(number_to_words(int(m.group(0))))


TIME_RE = re.compile(r"\b(\d{1,2}):(\d{2})\b")
ORDINAL_RE = re.compile(r"\b(\d+)" + ORDINAL_SUFFIX)
NUMBER_RE = re.compile(r"\b\d+\b")


def normalise(text):
    """Expand clock times, ordinals and bare numbers into Telugu words."""
    text = TIME_RE.sub(_expand_time, text)
    text = ORDINAL_RE.sub(_expand_ordinal, text)
    text = NUMBER_RE.sub(_expand_number, text)
    return re.sub(r"\s+", " ", text).strip()


if __name__ == "__main__":
    samples = [
        "ఈ సమావేశం 2026 నవంబర్ 12వ తేదీ ఉదయం 10:45 గంటలకు ప్రారంభమవుతుంది, సమయానికి సిద్ధంగా ఉంచండి.",
        "2026",
        "12వ",
        "10:45",
        "100",
        "1000",
        "45",
    ]
    for s in samples:
        print(f"{s}\n  -> {normalise(s)}\n")

"""Rule-based grapheme-to-phoneme for Telugu.

MFA has no Telugu model and there is no dictionary to download, but Telugu
orthography is close to phonemic so rules work. It is an abugida: a bare
consonant carries an inherent /a/, a matra replaces it, the virama U+0C4D
deletes it, and the anusvara U+0C02 takes the place of articulation of whatever
follows, so it needs lookahead. Phones are ASCII to go straight into an MFA
dictionary.
"""

import re
import sys

VIRAMA = "్"
ANUSVARA = "ం"
VISARGA = "ః"
CANDRABINDU = "ఁ"

CONSONANTS = {
    "క": "k",   "ఖ": "kh",  "గ": "g",   "ఘ": "gh",  "ఙ": "ng",
    "చ": "c",   "ఛ": "ch",  "జ": "j",   "ఝ": "jh",  "ఞ": "ny",
    "ట": "tt",  "ఠ": "tth", "డ": "dd",  "ఢ": "ddh", "ణ": "nn",
    "త": "t",   "థ": "th",  "ద": "d",   "ధ": "dh",  "న": "n",
    "ప": "p",   "ఫ": "ph",  "బ": "b",   "భ": "bh",  "మ": "m",
    "య": "y",   "ర": "r",   "ఱ": "rr",  "ల": "l",   "ళ": "ll",
    "వ": "v",   "శ": "sh",  "ష": "ss",  "స": "s",   "హ": "h",
    "ౘ": "k",   "ౙ": "kh",  "ౚ": "rr",
}

INDEPENDENT_VOWELS = {
    "అ": "a",  "ఆ": "aa", "ఇ": "i",  "ఈ": "ii",
    "ఉ": "u",  "ఊ": "uu", "ఋ": "ru", "ౠ": "ruu",
    "ఌ": "lu", "ౡ": "luu",
    "ఎ": "e",  "ఏ": "ee", "ఐ": "ai",
    "ఒ": "o",  "ఓ": "oo", "ఔ": "au",
}

VOWEL_SIGNS = {
    "ా": "aa", "ి": "i",  "ీ": "ii",
    "ు": "u",  "ూ": "uu", "ృ": "ru", "ౄ": "ruu",
    "ె": "e",  "ే": "ee", "ై": "ai",
    "ొ": "o",  "ో": "oo", "ౌ": "au",
    "ౢ": "lu", "ౣ": "luu",
}

# Anusvara assimilates to the place of articulation of whatever follows.
_VELAR = {"k", "kh", "g", "gh"}
_PALATAL = {"c", "ch", "j", "jh"}
_RETROFLEX = {"tt", "tth", "dd", "ddh"}
_DENTAL = {"t", "th", "d", "dh"}
_LABIAL = {"p", "ph", "b", "bh", "m"}


def _nasal_for(next_cons):
    if next_cons in _VELAR:
        return "ng"
    if next_cons in _PALATAL:
        return "ny"
    if next_cons in _RETROFLEX:
        return "nn"
    if next_cons in _DENTAL:
        return "n"
    if next_cons in _LABIAL:
        return "m"
    return "m"


def g2p_word(word):
    """Return a list of phones for one Telugu orthographic word."""
    chars = list(word)
    phones = []
    i = 0
    n = len(chars)

    while i < n:
        ch = chars[i]

        if ch in CONSONANTS:
            phones.append(CONSONANTS[ch])
            nxt = chars[i + 1] if i + 1 < n else ""
            if nxt == VIRAMA:
                # cluster: inherent vowel suppressed
                i += 2
                continue
            if nxt in VOWEL_SIGNS:
                phones.append(VOWEL_SIGNS[nxt])
                i += 2
                continue
            # no matra and no virama, so the inherent /a/ surfaces
            phones.append("a")
            i += 1
            continue

        if ch in INDEPENDENT_VOWELS:
            phones.append(INDEPENDENT_VOWELS[ch])
            i += 1
            continue

        if ch == ANUSVARA or ch == CANDRABINDU:
            # look ahead for the next consonant to pick the nasal's place
            nxt_cons = None
            for j in range(i + 1, n):
                if chars[j] in CONSONANTS:
                    nxt_cons = CONSONANTS[chars[j]]
                    break
            phones.append(_nasal_for(nxt_cons) if nxt_cons else "m")
            i += 1
            continue

        if ch == VISARGA:
            phones.append("h")
            i += 1
            continue

        if ch in VOWEL_SIGNS:
            # stray matra with no consonant carrier; keep the vowel
            phones.append(VOWEL_SIGNS[ch])
            i += 1
            continue

        # anything else (stray virama, punctuation) is skipped
        i += 1

    return phones


TELUGU_WORD = re.compile(r"[ఀ-౿]+")


def words_in(text):
    return TELUGU_WORD.findall(text)


def phone_inventory():
    inv = set(CONSONANTS.values()) | set(INDEPENDENT_VOWELS.values()) | set(VOWEL_SIGNS.values())
    inv |= {"a", "ng", "ny", "nn", "n", "m", "h"}
    return sorted(inv)


if __name__ == "__main__":
    tests = [
        "న్యూరోఫార్మకోలాజికల్",
        "పరిశోధనలో",
        "పట్టు",
        "అడవిలో",
        "నిశ్శబ్దంగా",
        "ప్రభావాన్ని",
        "విశ్లేషించారు",
        "సమావేశం",
        "ఉంది",
    ]
    for w in sys.argv[1:] or tests:
        print(f"{w}\t{' '.join(g2p_word(w))}")
    print(f"\nphone inventory ({len(phone_inventory())}): {' '.join(phone_inventory())}")

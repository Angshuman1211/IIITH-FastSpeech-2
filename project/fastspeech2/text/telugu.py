"""Telugu phone set used by the variance adaptor's text front end.

These are the phones produced by scripts/telugu_g2p.py and written into the MFA
dictionary, so the symbol table here has to stay in step with that inventory.
The reference implementation shipped ARPAbet for English and pinyin for Mandarin
and had nothing for an Indic language.
"""

valid_symbols = [
    # vowels
    "a", "aa", "i", "ii", "u", "uu",
    "ru", "ruu", "lu", "luu",
    "e", "ee", "ai",
    "o", "oo", "au",
    # stops and affricates
    "k", "kh", "g", "gh",
    "c", "ch", "j", "jh",
    "tt", "tth", "dd", "ddh",
    "t", "th", "d", "dh",
    "p", "ph", "b", "bh",
    # nasals
    "ng", "ny", "nn", "n", "m",
    # approximants, liquids, fricatives
    "y", "r", "rr", "l", "ll", "v",
    "sh", "ss", "s", "h",
]

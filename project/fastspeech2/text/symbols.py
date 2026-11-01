""" from https://github.com/keithito/tacotron """

"""
Defines the set of symbols used in text input to the model.

The default is a set of ASCII characters that works well for English or text that has been run through Unidecode. For other data, you can modify _characters. See TRAINING_DATA.md for details. """

from text import cmudict, pinyin, telugu

_pad = "_"
_punctuation = "!'(),.:;? "
_special = "-"
_letters = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"
_silences = ["@sp", "@spn", "@sil"]

# Prepend "@" to ARPAbet symbols to ensure uniqueness (some are the same as uppercase letters):
_arpabet = ["@" + s for s in cmudict.valid_symbols]
_pinyin = ["@" + s for s in pinyin.valid_symbols]
# Telugu phones are lowercase, so prefixing "@" keeps them distinct from the
# uppercase ARPAbet entries as well as from the plain letters.
_telugu = ["@" + s for s in telugu.valid_symbols]

# Export all symbols. Several Telugu phones spell the same as pinyin finals
# ("a", "ai", "o"), so the combined list is deduplicated while keeping order.
# Without this the table carries 18 unreachable embedding rows and two names
# for the same id.
_all_symbols = (
    [_pad]
    + list(_special)
    + list(_punctuation)
    + list(_letters)
    + _arpabet
    + _pinyin
    + _telugu
    + _silences
)

symbols = list(dict.fromkeys(_all_symbols))

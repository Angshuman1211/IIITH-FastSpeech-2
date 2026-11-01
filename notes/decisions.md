# Decisions log

Every choice that is not obvious from the code, with the reason and the number
that justified it. The technical report pulls from this file.

## D1. Deadline read as Friday 6 Nov, 23:59

The email says "07-11-2026 (Saturday) midnight 12 am". 12 am on Saturday is the
start of Saturday, so the safe reading is end of Friday 6 November. Planning
against Friday night, not Saturday night.

## D2. Telugu uses the male speaker only

The HuggingFace release has no speaker ID, only a `gender` field, which for
IndicTTS means one female and one male speaker. Measured from the audio headers:

| speaker | utts | total | mean len | yield at 1-12 s |
|---|---|---|---|---|
| female (gender=0) | 4043 | 23.92 h | 21.30 s | 2.31 h |
| male (gender=1) | 4533 | 12.79 h | 10.16 s | 6.86 h |

The assignment needs 5 h train + 1 h val = 6 h. After filtering to a length band
FastSpeech 2 can actually train on, the female speaker only yields 2.31 h. The
male speaker yields 6.86 h. So: male speaker, single-speaker model.

## D3. Utterances filtered to 1-15 seconds

The corpus is long-form narration. The female median is 18.5 s and the longest
file is 151.6 s. Long utterances hurt forced alignment and cost a lot of memory
during training.

Started at 1-12 s, which gave 3468 male utterances and 6.86 h. That turned out
not to be enough once trimming was taken into account (see D9), so the band was
widened to 1-15 s, giving 3806 utterances and 8.11 h before trimming.

## D4. Audio resampled to 22.05 kHz

This HuggingFace release is **16 kHz**, not the 48 kHz of the original IIT Madras
database. Resampling up to 22.05 kHz to match the pretrained HiFi-GAN vocoder.

An earlier version of this note said the result would be audibly band-limited.
That was overstated. The mel front-end uses `mel_fmax: 8000`, and a 16 kHz
recording already carries everything up to 8 kHz, so the mel representation the
model actually sees loses essentially nothing. The caveat worth keeping is
narrower: the vocoder reconstructs the 8-11 kHz band from a mel that never
contained it, for both languages equally.

## D5. Train/val split is disjoint by construction

Seeded shuffle (seed 1234), then fill the 1 h validation budget first and the
5 h training budget after. A file can only land in one of them, so there is no
overlap to check for afterwards.

## D6. Test sentence text verified against the rendered PDF, not the text layer

`pdftotext` inserts spurious spaces at Telugu glyph boundaries. The text layer
gives "విశ్ల ేషించారు" and "వచ్చ ింది", both of which put a vowel sign at the
start of a word, which is impossible. Rendering page 2 to an image and reading it
shows those words are contiguous.

The one real anomaly is **T3 "పులి అడ విలో"**, which has a visible gap in the
rendered page where అడవిలో should be one word. That mirrors **E3 "The rhyth m"**
in the English set. Both are deliberate. The assignment says to preserve the
sentence text as provided, so both stay broken and the report analyses what the
model does with them instead of quietly fixing the spelling.

## D7. Reference implementation needs porting

`ming024/FastSpeech2` was last pushed in October 2023 and pins torch 1.7.0,
numpy 1.19.0 and librosa 0.7.2. Running it on current torch needs fixes for the
`weight_norm` move, numpy 2.0 removals, and the librosa keyword-argument change.
Installing current versions and patching the call sites rather than trying to
recreate a 2023 environment.

## D8. No MFA Telugu acoustic model exists

The MFA model index ships English and Tamil, but not Telugu. English alignment
is a pretrained download. Telugu needs an acoustic model trained from scratch on
the corpus plus a pronunciation dictionary built from scratch. This is the
longest single task in the project.

## D9. Split budgets are filled by post-trim duration

The first build budgeted on the raw duration of each file and produced a corpus
MFA measured at 18464.7 s, which is 5.13 h, not the 6 h intended. Silence
trimming removes about 14% of this corpus, so budgeting before trimming
undershoots by roughly that much and the delivered split would have been 4.3 h
and 0.85 h.

The fix is to trim first, measure the trimmed audio, and only then assign a file
to the validation or training budget. The band widening in D3 was needed to
leave enough material to reach 5 h + 1 h after the loss.

## D10. Telugu lexicon is generated, and the transcripts are tokenised to match

MFA needs every transcript token to be in the dictionary. Two passes were needed:

  * Stripping only leading and trailing punctuation left 5315 bad tokens, because
    abbreviations such as ఐ.టి.ఐ carry punctuation inside the word.
  * Splitting on internal punctuation too, and dropping tokens that contain no
    consonant or independent vowel (a stray anusvara had become its own entry),
    gives 12459 entries with zero out-of-vocabulary tokens.

MFA validation confirms this: 3032 sound files, 3032 transcripts, 1 speaker, no
missing words, no feature-generation failures, no unreadable audio.

## D11. Training moves to Kaggle, because MPS is too slow

Measured on the M1 Pro: about 1.8 s/step in the first hundred steps and about
3.15 s/step sustained once the periodic validation passes are included. At the
configured 80000 steps that is roughly 70 h for one language and double that
for both, against roughly 115 h of wall clock before the deadline. It does not
fit alongside English preprocessing, alignment, inference and the report.

Kaggle gives a free P100 or dual T4, a 12 h session cap and about 30 GPU hours
a week. That should put 80000 steps at 5-8 h per language, so both fit inside
one week's quota with each run inside a single session.

Consequences for the code:
  * `load_configs` takes `--set section.key=value` overrides so one config file
    works both here and under /kaggle/input and /kaggle/working.
  * The trainer already checkpoints every 10000 steps and `--restore_step`
    resumes, so a dropped session costs at most 10000 steps.
  * `scripts/package_for_kaggle.py` stages only the code, the preprocessed
    features and the universal vocoder, which is 579 MB rather than the full
    6 GB corpus.

## D12. Inference and plotting were validated before any model finished training

Running `inference.py --restore_step 0` against a randomly initialised model
exercised the whole path: G2P, symbol lookup, model forward, vocoder, wav
writing and all three plots. It produced 5 wavs and 15 plots with correct
phoneme counts, which caught the path and API problems early instead of after
a long GPU run. The audio is meaningless at that point, deliberately so.

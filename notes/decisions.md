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
    gives zero out-of-vocabulary tokens.

MFA validation confirms this: 1 speaker, no missing words, no feature-generation
failures, no unreadable audio.

Final counts, on the 1-15 s corpus of D3 and the post-trim budgets of D9. An
earlier version of this note quoted 12459 lexicon entries and 3032 files, which
were measured on the narrower 1-12 s corpus and are superseded:

  * 14621 lexicon entries
  * 3244 wav files with 3244 matching .lab transcripts
  * 3242 TextGrids, so 2 utterances failed to align

Both failures are long run-on sentences of comma-separated vocatives, where the
aligner could not fit the transcript to the audio. They are simply absent from
the feature set rather than silently mismatched, which is why the splits are
2669 and 573 rather than the 2670 and 574 the manifest budgeted. That is a 0.06%
loss and the split durations are still 5.00 h and 1.00 h to the precision worth
quoting.

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

## D13. The 187 Hz pitch mean is correct, not an extraction fault

The Telugu pitch statistics come out at mean 187.2 Hz, std 45.5, which is high
for a speaker the dataset labels male and looks like the classic pyworld octave
doubling. It is not. Two checks rule that out.

First, the distribution is unimodal. Denormalising every stored phoneme-level
value gives a median of 185.7 Hz with quartiles at 154.5 and 219.0, and only
7% above 260 Hz. Octave doubling leaves a second mode near twice the first, and
there is no such mode.

Second, four independent estimators agree on a sample of twelve utterances:

| estimator | settings | median of per-file medians |
|---|---|---|
| dio + stonemask | floor 71, ceil 800 (the default) | 201.2 Hz |
| dio + stonemask | floor 60, ceil 300 | 198.0 Hz |
| harvest | floor 60, ceil 300 | 197.4 Hz |
| librosa yin | fmin 60, fmax 400 | 206.6 Hz |

Narrowing the ceiling to 300 Hz moves the estimate by 3 Hz, and harvest, which
is far more resistant to octave error than dio, agrees to within 4 Hz. So the
speaker really does sit near 190-200 Hz.

For the report: that is in the usual adult female range and
well above the usual adult male range, so the corpus's own `gender` label looks
questionable. The label mapping itself was verified and is not the problem,
`gender=1` does correspond to the files named `telugumale` across all 8576 rows.
Either way the feature pipeline is correct and no re-extraction is needed. D2
is unaffected, since that speaker was chosen on duration yield rather than on
gender.

## D14. Training is launched on Kaggle headlessly, with a wall-clock guard

Driving a notebook by hand wastes the deadline, so `scripts/launch_kaggle.py`
pushes a script kernel through the Kernels API and starts it. One push trains
one language. Resuming attaches the kernel's own previous output, so
`--restore-step` picks up from the last checkpoint.

The guard matters. A Kaggle GPU session is killed at 12 h
and the output is discarded with it, so an overrunning job loses every
checkpoint rather than just the last stretch. `FS2_MAX_HOURS` stops the loop at
a clean step, saves, and exits, and is set to 11 h against the 12 h cap. This
was tested by setting it to 22 seconds and confirming the run stopped at a step
boundary and wrote both the full and the weights-only checkpoint.

## D15. Svarah has exactly one distribution channel, and it is gated

Checked before treating English as blocked. The AI4Bharat GitHub repo, the
AI4Bharat dataset API and the AIKosh government portal all link to the same
HuggingFace dataset and host no copy of their own. Anonymous access reads the
metadata and the file list but `resolve` returns 401 GatedRepoError on all three
parquet shards. So English cannot start without the gate being accepted on an
account and a token for that account present on this machine. There is no
mirror to fall back to.

## D16. Svarah carries no speaker id, so speakers are recovered from demographics

The schema is `audio_filepath`, `duration`, `text` and eight demographic
fields. There is no speaker column, and the assignment needs a multi-speaker
English model, so speaker identity has to be recovered.

The filename looks like it should carry it. It does not. Measured over all
6656 utterances:

| grouping | distinct values |
|---|---|
| leading recording id | 2938 |
| the `fNNNN` field | 3232 |
| recording id + `fNNNN` | 3232 |
| demographic tuple | **115** |

The paper documents 117 speakers. Both filename fields are per-recording, not
per-speaker, at roughly two utterances each. The `f` prefix is not a gender
marker either, it appears on 3579 female and 3077 male rows alike.

So the eight demographic fields together are the speaker proxy. 115 against a
documented 117 means two pairs of speakers share an identical demographic
tuple and get merged, which I accept and record rather than hide. The grouping
is well behaved otherwise, a median of 47 utterances and 227 s per speaker,
with the largest five holding 15.7% of the audio between them.

`--speaker-mode demographic` is therefore the default. The filename mode is
kept only so the comparison above can be reproduced.

## D17. Long digit runs are read digit by digit, not as cardinals

Svarah covers digital payments and government services, so it is full of phone
numbers, OTPs and pincodes. 3.1% of transcripts contain a run of 5 or more
digits.

Expanding those as cardinals is simply wrong against the audio. A speaker
reading 79083174897 says "seven nine zero eight three...", not "seventy nine
billion eighty three million...". That mismatch either fails alignment or, far
worse, aligns badly and silently corrupts the phoneme durations that the
variance adaptor trains on.

So a run of 5 or more digits is spelled out digit by digit. The exception is a
round number, which is a genuine quantity at any length, so 10000 stays "ten
thousand" while 169804 becomes "one six nine eight zero four". Shorter runs
keep cardinal expansion, since years and small counts are read that way.

## D18. The English split is stratified by speaker

The Telugu split just fills the validation budget first, because it is one
speaker. English has 115, and a multi-speaker model validated on speakers it
never trained on measures the wrong thing.

Each speaker's first utterance is therefore sent to validation before anything
else, after which both budgets fill as before. All 115 speakers appear in
validation, so validation covers the full accent range rather than whichever
speakers happened to be shuffled to the front.

## D19. English text normalisation, fixed against measured alignment quality

Three faults in my own `clean_text`, found by reading the utterances that
failed to align rather than by guessing.

**Ordinals were being split.** "7th July" became "seven th july", because
expanding the digits leaves the suffix as its own token. Now
`p.ordinal(p.number_to_words(n))` gives "seventh". 0.7% of the corpus carries
a digit ordinal, and four of the eleven unaligned utterances were this.

**Short identifiers were read as quantities.** D17 only caught runs of 5 or
more digits, so a 4-digit MPIN still came out as "six thousand two hundred and
seventy nine". Now a 4-digit run next to pin, mpin, otp, code or cvv is spelled
out digit by digit. Same reasoning as D17, keyed on the neighbouring word
rather than on length.

**Years were read the long way.** MFA writes an `alignment_analysis.csv`, which
makes this measurable. Utterances containing a year expanded as "one thousand nine hundred and
eighty four" aligned with a median phone duration deviation of **3.61** against
**2.29** for everything else, where the whole corpus's 90th percentile is 3.72.
So those utterances were aligning about as badly as the worst tenth of the
corpus, and there were 103 of them.

A year is said in two halves, so 1984 is now "nineteen eighty four", 1900 is
"nineteen hundred", 1905 is "nineteen oh five" and 2008 is "two thousand and
eight". The rule is gated to 1100-2099 and suppressed when a unit word follows,
so "1500 rupees" and "1200 people" stay quantities.

Re-aligning afterwards confirms it, on the 3899 utterances common to both runs:

| group | n | before | after | change |
|---|---|---|---|---|
| carries a year | 99 | 3.609 | 2.499 | **-1.111** |
| everything else | 3800 | 2.291 | 2.303 | +0.013 |

70 of the 99 improved individually. The affected group moved from worse than
the corpus 90th percentile to roughly its median, and the rest of the corpus did
not move, which is the shape the fix should have. Corpus-wide the median barely
changes, 2.315 to 2.308, because 99 utterances out of 3899 cannot move it. The
per-group split is the number that matters, and quoting only the corpus median
here would hide the result rather than report it.

Alignment also recovered 3 more utterances, 3911 against 3908.
## D20. The reference discards the pronunciation that test sentence E2 asks for

E2 is a minimal pair, "the orchestra's conduct" against "will conduct an
experiment". English moves the stress between the noun and the verb, and the
assignment is clearly probing whether the model reproduces that.

The reference cannot, but not for the reason I first assumed. The information
is there and gets thrown away.

`lexicon/librispeech-lexicon.txt` holds both pronunciations:

```
CONDUCT  K AA1 N D AH0 K T     <- noun, stress on the first syllable
CONDUCT  K AH0 N D AH1 K T     <- verb, stress on the second
```

`read_lexicon` builds its dictionary with `setdefault`, so the second line is
dropped and every "conduct" gets the noun form.

`g2p_en` would get it right. It tags part of speech with NLTK and
disambiguates homographs, and on the two actual test sentences it returns
`K AA1 N D AH0 K T` for the noun and `K AA0 N D AH1 K T` for the verb. But
`preprocess_english` looks the word up in the lexicon first and only falls
through to `g2p_en` when that misses, so for "conduct" the part-of-speech aware
path is never reached.

So E2a and E2b will be phonetically identical, both with noun stress, and E2b
is simply wrong. Reporting it this way, with the phone strings, rather than as
a vague architectural limitation. The fix is a front end that defers to
`g2p_en` when the lexicon offers more than one pronunciation.

## D21. g2p_en needs NLTK tagger data, or English inference crashes

Not a design choice, a trap worth recording. `g2p_en` imports cleanly and only
fails when called, because the NLTK `averaged_perceptron_tagger_eng` resource
is fetched at run time rather than packaged. Two of the four English test
sentences contain words that miss the lexicon, "antidisestablishmentarianism"
and the deliberately broken "rhyth", so inference reaches `g2p_en` and would
have died there.

Found before training rather than after, by testing the front end directly.
`requirements.txt` and the README both need the download step.

## D22. Silence is removed three times, so report selected and effective hours

D9 fixed budgeting against post-trim duration. There is a third trim after it,
and it took measuring the finished features to notice.

The preprocessor's `get_alignment` skips every `sil`, `sp` and `spn` interval
before the first real phone and cuts the sequence at the last one, so the mel
frames the model trains on exclude leading and trailing silence as MFA marked
it. Internal pauses survive as phones. So the chain is: pick within the length
band, trim with librosa and budget on that, then lose more to the aligner.

| | selected | after alignment | lost |
|---|---|---|---|
| Telugu train | 5.000 h | 4.263 h | 15% |
| Telugu val | 1.000 h | 0.853 h | 15% |
| English train | 5.000 h | 3.949 h | 21% |
| English val | 1.000 h | 0.797 h | 20% |

Not fixing this. It is not a bug. Removing leading and trailing silence is what the reference
preprocessor does and what any FastSpeech 2 implementation does, because
training a duration predictor on arbitrary head and tail silence is harmful.
The selection is exactly the 5 h and 1 h the assignment asks for, and the
reduction happens inside the standard pipeline.

Topping the budgets up to land on 5 h after alignment would mean about 6.3 h
of training audio per language. Telugu has room. English does not: Svarah
offers 7.97 h in the 1-15 s band, so train plus validation would need 7.6 h of
it and leave almost no margin, on a corpus that is already the whole usable
dataset.

Matching the two languages beats hitting a round number, since section 6 of the
report compares them. Both are selected identically at 5 h and
1 h, both lose silence to the same code, and the difference between 15% and 21%
is a property of the corpora, since Svarah is ASR material with more head and
tail silence than studio narration. Quoting only "5 hours" would be the
misleading choice here, so the report carries both columns.

## D23. English training and inference were expanding numbers differently

Caught by dry-running English inference against an untrained model, the same
trick as D12, before the GPU run finished.

Every Svarah training transcript goes through `clean_text`, so the model only
ever saw number words. The reference's `preprocess_english` hands raw text
straight to the lexicon and `g2p_en`. Test sentence E4 is "10:45 AM on
12 November 2026", so the two paths disagreed exactly where the assignment
probes.

What E4 produced before the fix, against after:

| token | before | after |
|---|---|---|
| 10:45 | `T EH1 N T ER0 F AY1 T IH0 V` | `T EH1 N F AO1 R T IY0 F AY1 V` |
| 2026 | `T W EH1 N T IY0 T W EH2 N T IY0 S K AY1 Z` | `T W EH1 N IY0 T W EH1 N IY0 S IH1 K S` |

The first was not even words, and the second ends in "sixes" rather than "six".

The rules now live once, in `project/fastspeech2/text/english_norm.py`, and
both paths call them, which mirrors how Telugu already worked through
`text/telugu_norm.py`. Two entry points, because training strips punctuation
for the aligner while inference keeps it, since the reference turns punctuation
into `sp` and the model saw `sp` wherever MFA measured a pause.
`normalise_for_training` was checked to reproduce the old `clean_text` on every
case, so the aligned corpus still matches the code that built it.

One rule is new rather than moved. A time needed handling as a time, because
expanding "10:45" as two numbers leaves the colon between them and inference
would turn that into a pause in the middle of the time. Only 1 of 6656 source
texts contains a time, and for that one, "11:28", the old path already gave the
right words, so the training corpus is unaffected.

## D24. A zero-length mel is a crash, so guard the collapsed case

The English dry run died inside the decoder with "cannot reshape tensor of 0
elements", which is what the variance adaptor produces when every phoneme is
predicted zero frames. An untrained model does precisely that, since
`exp(0) - 1` is 0.

Per-phoneme zero is deliberate in the reference and worth keeping, because it
is what lets a trained model drop a phoneme. The degenerate case is when the
whole sequence collapses, and then the output is not merely wrong but
impossible, and the error names a tensor dimension rather than the real
problem.

So the guard is scoped to the whole-sequence case: if a sequence's durations
sum to zero across its real tokens, its first phoneme gets one frame. It cannot
fire once durations are sensible, so it cannot change a trained model's output,
and it makes the inference path testable before a model exists, which is the
point.

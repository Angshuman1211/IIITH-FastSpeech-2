# FastSpeech 2 for English and Telugu

Submission for the IIIT Hyderabad SLFI Speech Processing Lab assignment.

Two separate FastSpeech 2 models, one per language, adapted from
[ming024/FastSpeech2](https://github.com/ming024/FastSpeech2) and trained on
5 hours of audio each with 1 hour held out. Both models are trained, all ten
test sentences are synthesised, and the variance adaptor output is analysed per
sentence.

**The full write-up is [project/Report/Technical_Report.pdf](project/Report/Technical_Report.pdf).**

## What was built

| | English | Telugu |
|---|---|---|
| corpus | [Svarah](https://huggingface.co/datasets/ai4bharat/Svarah) | [IndicTTS Telugu](https://huggingface.co/datasets/SPRINGLab/IndicTTS_Telugu) |
| speakers | 115 | 1 |
| train / validation | 3262 / 648 utterances | 2669 / 573 utterances |
| grapheme to phoneme | `g2p_en` with a lexicon | rule-based, written for this project |
| alignment | pretrained MFA, dictionary extended by 3382 entries | acoustic model trained from scratch |
| training | 73100 of 80000 steps | 63675 of 80000 steps |
| checkpoint released | step 73100 | step 60000 |

Telugu needed the most work, because MFA publishes no Telugu acoustic model and
no Telugu pronunciation dictionary exists. Both were built from scratch, along
with a 51-phone rule-based G2P. English needed its dictionary extended, because
4.53% of Svarah tokens are outside `english_us_arpa` and 33.1% of utterances
contain at least one such token.

## Results

Neither run reached 80000 steps, and section 3.3 of the report shows this
changed nothing. Both models reached their best validation loss within about
2000 steps and spent the rest of training overfitting. Synthesising the test
sentences from the 10000-step and 60000-step Telugu checkpoints gives durations
within 3% and mean F0 within 9% of each other.

**Telugu is the better model.** Every Telugu test sentence is intelligible as
well formed, separable words. English is patchy: the three short sentences hold
up, and both long ones start cleanly and then degrade. Mean spectral flatness
is 0.0084 for Telugu against 0.0226 for English, so the measurement agrees with
listening.

The clearest single finding is that **English validation duration loss sits at
0.2297 and never improves, while Telugu reaches 0.075**. One speaker at one
speaking rate is a learnable duration target. 115 speakers at different rates,
with no speaker identity given at inference, is not. That is consistent with
the English failures beginning part way through a sentence rather than
affecting it evenly.

Test sentence E2 cannot work, for a reason in the front end rather than the
model. The LibriSpeech lexicon holds both stress patterns for "conduct", but
`read_lexicon` keeps only the first, so both sentences get the noun form. ARPAbet
encodes stress in the phone symbol, so no amount of training could fix it.
Section 4.3 has the detail.

## Where the deliverables are

| | |
|---|---|
| [project/Report/Technical_Report.pdf](project/Report/Technical_Report.pdf) | the technical report, 20 pages |
| [project/Results/English/](project/Results/English/) | loss curves, 5 generated wavs, 15 variance-adaptor plots |
| [project/Results/Telugu/](project/Results/Telugu/) | the same for Telugu |
| [project/English/](project/English/), [project/Telugu/](project/Telugu/) | per-language entry points and config |
| [project/fastspeech2/](project/fastspeech2/) | the shared model code, with every modification to the reference |
| [project/README.md](project/README.md) | setup and the full command sequence |
| [notes/decisions.md](notes/decisions.md) | every design decision with the measurement behind it, D1 to D24 |

Checkpoints are too large for GitHub and are linked from
[project/README.md](project/README.md).

## Reproducing a result

The quickest check is to regenerate the test sentences from a released
checkpoint:

```bash
conda create -n fs2 python=3.11 && conda activate fs2
pip install -r project/requirements.txt
python -c "import nltk; nltk.download('averaged_perceptron_tagger_eng'); nltk.download('cmudict')"

# download telugu_60000.pth.tar from the link in project/README.md
mv telugu_60000.pth.tar project/Telugu/checkpoints/60000.pth.tar
cd project/Telugu && python inference.py --restore_step 60000
```

That writes the wavs and all three variance-adaptor plots. Rebuilding the
corpora and alignments from scratch is documented in
[project/README.md](project/README.md), and needs Montreal Forced Aligner and
access to the two datasets. Svarah is gated on HuggingFace and has no mirror,
so it needs an account that has accepted its terms.

## Known limitations

- Both models overfit. The variance predictors end 36 to 63 times worse on
  validation than on training, so the pitch and energy contours should be read
  as plausible rather than correct for unseen text.
- English degrades part way through long sentences. Its two longest test
  sentences are the two that fail.
- E2b carries the wrong stress, as described above.
- The vocoder reconstructs the 8 to 11 kHz band from a mel that never contained
  it, since `mel_fmax` is 8000 for both languages.
- 80000 steps was chosen to fit the time available rather than from this data,
  and the loss curves show it was the wrong target.

## Notes on the reference implementation

It was last updated in October 2023 and pins torch 1.7.0, numpy 1.19.0 and
librosa 0.7.2, none of which install on a current machine, so the call sites
were patched and `project/requirements.txt` carries current versions.

Three defects in the reference were also found and fixed. Two of them degrade
results silently rather than failing:

- the pitch and energy scalers were fitted outside the branch that produced the
  values, so statistics from skipped utterances shifted the normalisation of
  every training target
- the train and validation split was built by an unseeded shuffle at
  preprocessing time, so it could not be reproduced or audited
- `nn.DataParallel` was applied whether or not more than one device existed,
  which is pointless on one GPU and fails on Apple MPS

Section 1.4 of the report has the detail.

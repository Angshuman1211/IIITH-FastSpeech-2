# FastSpeech 2 for English and Telugu

Two separate FastSpeech 2 text-to-speech models, one per language, built from
[ming024/FastSpeech2](https://github.com/ming024/FastSpeech2) and trained on
5 hours of audio each with 1 hour held out for validation.

Submission for the IIIT Hyderabad SLFI Speech Processing Lab assignment. The
task, the test sentences and the required deliverables are in
[FastSpeech2_Assignment.pdf](FastSpeech2_Assignment.pdf).

| | English | Telugu |
|---|---|---|
| corpus | [Svarah](https://huggingface.co/datasets/ai4bharat/Svarah) | [IndicTTS Telugu](https://huggingface.co/datasets/SPRINGLab/IndicTTS_Telugu) |
| speakers | 115 | 1 |
| train | 3262 utterances | 2669 utterances |
| validation | 648 utterances | 573 utterances |
| alignment | pretrained MFA model, dictionary extended | acoustic model trained from scratch |
| grapheme to phoneme | `g2p_en` and a lexicon | rule-based, written for this project |

## Start here

- **[project/README.md](project/README.md)** is the full guide: setup, every
  command, and how to run the pipeline end to end.
- **[project/Report/Technical_Report.pdf](project/Report/)** is the write-up.
  It covers the architecture, every modification made to the reference, the
  data selection and preprocessing for both languages, the loss curves, and the
  variance-adaptor analysis for each test sentence.
- **[notes/decisions.md](notes/decisions.md)** is the working log. Every
  non-obvious choice, with the measurement that justified it, numbered D1 to
  D24. If you want to know why something is the way it is, it is in there.

## What is in this repository

```
project/           the submission: per-language entry points, shared model code,
                   results, report, README and requirements
scripts/           data preparation, lexicon building, Kaggle orchestration
lexicon/           generated pronunciation dictionaries
notes/             the decisions log
kaggle/            the training kernel that runs on Kaggle's GPU
```

The corpora, the extracted features, the alignments and the checkpoints are not
in the repository, because they are large and in one case not redistributable.
`project/README.md` explains how to regenerate each of them, and where to
download the checkpoints.

## Two things worth knowing before you start

**Svarah is gated and has no mirror.** Accept the terms once at
<https://huggingface.co/datasets/ai4bharat/Svarah>, then `hf auth login`. The
AI4Bharat GitHub repository, the AI4Bharat dataset API and the AIKosh portal
all point back to that same entry, so this step cannot be skipped.

**The reference implementation does not install as pinned.** It was last
updated in October 2023 and pins torch 1.7.0, numpy 1.19.0 and librosa 0.7.2,
none of which build on a current machine. The call sites were patched instead,
and `project/requirements.txt` carries current versions. Section 1.3 of the
report lists each change.

## A note on the reference implementation

Three defects were found in the upstream code and fixed. They are worth knowing
about if you work from the same starting point, because two of them degrade
results silently rather than failing:

- the pitch and energy scalers were fitted outside the branch that produced the
  values, so statistics from skipped utterances shifted the normalisation of
  every training target
- the train and validation split was built by an unseeded shuffle at
  preprocessing time, so it could not be reproduced or audited
- `nn.DataParallel` was applied whether or not more than one device existed,
  which is pointless on one GPU and fails outright on Apple MPS

Section 1.4 of the report has the detail.

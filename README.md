# FastSpeech 2 for English and Telugu

Two separate FastSpeech 2 models, one per language, built from
[ming024/FastSpeech2](https://github.com/ming024/FastSpeech2) and trained on
5 hours of audio each with 1 hour held out.

Submission for the IIIT Hyderabad SLFI Speech Processing Lab assignment.

| | English | Telugu |
|---|---|---|
| corpus | [Svarah](https://huggingface.co/datasets/ai4bharat/Svarah) | [IndicTTS Telugu](https://huggingface.co/datasets/SPRINGLab/IndicTTS_Telugu) |
| speakers | 115 | 1 |
| train | 3262 utterances | 2669 utterances |
| validation | 648 utterances | 573 utterances |
| alignment | pretrained MFA, dictionary extended | acoustic model trained from scratch |
| G2P | `g2p_en` and a lexicon | rule-based, written for this |

## Where things are

- [project/README.md](project/README.md) has the setup and every command.
- [project/Report/](project/Report/) has the technical report.
- [notes/decisions.md](notes/decisions.md) is the working log, D1 to D24. If
  you want to know why something is the way it is, it is in there.

```
project/     the submission: entry points, model code, results, report
scripts/     data prep, lexicon building, Kaggle orchestration
lexicon/     generated pronunciation dictionaries
notes/       decisions log
kaggle/      the training kernel
```

Corpora, features, alignments and checkpoints are not in the repo. See
`project/README.md` for how to regenerate them and where to get the
checkpoints.

## Two gotchas

Svarah is gated and has no mirror. Accept the terms at
<https://huggingface.co/datasets/ai4bharat/Svarah>, then `hf auth login`. The
GitHub repo, the AI4Bharat API and AIKosh all point back to that same entry.

The reference does not install as pinned. It wants torch 1.7.0, numpy 1.19.0
and librosa 0.7.2, none of which build now, so the call sites were patched and
`project/requirements.txt` carries current versions instead.

## Bugs found in the reference

Worth knowing if you start from the same code, since two of these fail
silently rather than loudly:

- the pitch and energy scalers were fitted outside the branch that produced the
  values, so skipped utterances shifted the normalisation of every target
- the train/validation split came from an unseeded shuffle at preprocessing
  time, so it could not be reproduced
- `nn.DataParallel` was applied unconditionally, which is pointless on one GPU
  and breaks on MPS

Section 1.4 of the report has the detail.

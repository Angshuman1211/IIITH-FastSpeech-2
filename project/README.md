# FastSpeech 2 for English and Telugu

Two separate FastSpeech 2 models, one per language, built from
[ming024/FastSpeech2](https://github.com/ming024/FastSpeech2) and trained on
5 hours of audio with 1 hour held out for validation.

- English: [Svarah](https://huggingface.co/datasets/ai4bharat/Svarah),
  Indian-accented English, 115 speakers
- Telugu: [IndicTTS Telugu](https://huggingface.co/datasets/SPRINGLab/IndicTTS_Telugu),
  single speaker

The full write-up is `Report/Technical_Report.pdf`. It covers the architecture,
every modification made to the reference implementation, the data selection and
preprocessing for both languages, the loss curves, and the variance-adaptor
analysis for each test sentence.

## Layout

```
project/
├── English/            train.py, preprocess.py, inference.py, config.yaml, checkpoints/
├── Telugu/             the same four, with a Telugu config
├── fastspeech2/        the shared model code, ported from the reference
├── test_sentences/     the assignment's test sentences, verified against the PDF
├── Results/
│   ├── English/        loss plots, generated audio, variance-adaptor plots
│   └── Telugu/         the same
├── Report/
│   └── Technical_Report.pdf
├── README.md
└── requirements.txt
```

`English/` and `Telugu/` hold thin entry points. The model, preprocessor,
vocoder and text front ends are shared in `fastspeech2/`, because the two
languages differ in their data and their text handling rather than in the
network.

## Setup

```bash
conda create -n fs2 python=3.11
conda activate fs2
pip install -r requirements.txt
```

Two things pip cannot do.

`g2p_en` downloads its part-of-speech tagger from NLTK the first time it is
called, not at install time, so it imports cleanly and then fails during
English inference. Run once:

```bash
python -c "import nltk; nltk.download('averaged_perceptron_tagger_eng'); nltk.download('cmudict')"
```

Montreal Forced Aligner is a conda package. It is only needed to regenerate
alignments from scratch, not to train or to run inference:

```bash
conda create -n mfa -c conda-forge montreal-forced-aligner=3.4.3
```

## Running it

### 1. Data

```bash
python scripts/download_data.py telugu
python scripts/download_data.py english
```

Svarah is gated. Accept the terms once at
<https://huggingface.co/datasets/ai4bharat/Svarah>, then `hf auth login`. There
is no mirror, so this step cannot be skipped.

```bash
python scripts/prepare_telugu.py
python scripts/prepare_svarah.py --speaker-mode demographic
```

Both write a `raw_data/<language>/` tree of 22.05 kHz wavs with matching `.lab`
transcripts, plus a `metadata.json` that fixes the train/validation split. The
split is decided here rather than at preprocessing time so that it is
reproducible and auditable.

`prepare_svarah.py --inspect` reports what speaker signal the release carries
before committing to a grouping, which is worth running first if the dataset is
ever republished.

### 2. Alignment

Telugu has no pretrained MFA model, so one is trained on the corpus:

```bash
python scripts/build_telugu_lexicon.py --rewrite-labs
mfa validate raw_data/Telugu lexicon/telugu_lexicon.txt -j 8
mfa train raw_data/Telugu lexicon/telugu_lexicon.txt \
    mfa_data/telugu_acoustic.zip --output_directory mfa_data/textgrids_telugu --clean -j 8
```

English uses the pretrained model, but its dictionary has to be extended first.
Against `english_us_arpa`, 4.53% of Svarah tokens are out of vocabulary and
33.1% of utterances contain at least one, mostly Indian place names and scheme
acronyms:

```bash
mfa model download acoustic english_us_arpa
mfa model download dictionary english_us_arpa
mfa model download g2p english_us_arpa

python scripts/extend_english_lexicon.py --collect
mfa g2p lexicon/english_oov_words.txt english_us_arpa \
    lexicon/english_oov_pronunciations.dict --clean
python scripts/extend_english_lexicon.py --merge

mfa align raw_data/English lexicon/english_svarah.dict english_us_arpa \
    mfa_data/textgrids_english --clean -j 8
```

### 3. Features, training, inference

From `English/` or `Telugu/`:

```bash
python preprocess.py
python train.py --restore_step 0
python inference.py --restore_step 80000
```

Any config value can be overridden without editing the file, which is how the
same config runs locally and on a hosted GPU:

```bash
python train.py --restore_step 0 \
    --set train.path.ckpt_path=/kaggle/working/ckpt \
    --set train.step.total_step=80000
```

Training picks CUDA, then Apple MPS, then CPU. `FS2_DEVICE` forces one.
`FS2_MAX_HOURS` stops the loop at a clean step and saves before a hosted
session's hard limit, which otherwise kills the job and discards every
checkpoint with it.

Loss figures:

```bash
python scripts/plot_losses.py --language Telugu
```

### 4. Training on Kaggle

Local training on MPS runs at about 3.15 s/step, which does not reach 80000
steps in any useful time, so training runs on a Kaggle GPU. It is driven from
the command line rather than the browser:

```bash
python scripts/package_for_kaggle.py --language Telugu --upload --user <kaggle-user>
python scripts/launch_kaggle.py --language Telugu --push
python scripts/launch_kaggle.py --language Telugu --status
python scripts/launch_kaggle.py --language Telugu --pull
```

The package step uploads only the code, the preprocessed features and the
universal vocoder, which is about 580 MB rather than the full corpus. A session
is capped at 12 hours, so one push trains for at most 11 and saves. To continue,
push again with `--restore-step` set to the last checkpoint, which is attached
automatically.

## Checkpoints

Each save writes a full checkpoint with optimiser state, about 420 MB, and a
weights-only copy, about 140 MB. Inference needs only the weights-only copy.

These exceed GitHub's file size limit, so they are not in the repository.
Download links are listed below and are kept live for evaluation.

| model | step | file | link |
|---|---|---|---|
| Telugu | 80000 | `80000_weights_only.pth.tar` | *(to be filled once training finishes)* |
| English | 80000 | `80000_weights_only.pth.tar` | *(to be filled once training finishes)* |

Place the file in `English/checkpoints/` or `Telugu/checkpoints/` and run
inference with the matching `--restore_step`.

The HiFi-GAN vocoder is the universal checkpoint that ships with the reference
repository, unzipped into `fastspeech2/hifigan/`.

## Notes

The reference implementation was last updated in October 2023 and pins
torch 1.7.0, numpy 1.19.0 and librosa 0.7.2, none of which install on a current
machine. Rather than recreate a 2023 environment, the call sites were patched
and the pinned versions in `requirements.txt` are current. Section 1.3 of the
report lists each change.

Three defects in the reference itself were also fixed, and are worth knowing
about if you work from the same upstream: the pitch and energy scalers were
fitted outside the branch that produced the values, the train/validation split
was built by an unseeded shuffle at preprocessing time, and `nn.DataParallel`
was applied whether or not more than one device existed. Section 1.4 of the
report has the detail.

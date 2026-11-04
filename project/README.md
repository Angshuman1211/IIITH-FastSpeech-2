# FastSpeech 2 for English and Telugu

Two separate models, built from
[ming024/FastSpeech2](https://github.com/ming024/FastSpeech2). English uses
[Svarah](https://huggingface.co/datasets/ai4bharat/Svarah) (115 speakers),
Telugu uses [IndicTTS Telugu](https://huggingface.co/datasets/SPRINGLab/IndicTTS_Telugu)
(one speaker). 5 h train and 1 h validation each.

`Report/Technical_Report.pdf` has the architecture, the modifications, the loss
curves and the variance-adaptor analysis.

## Layout

```
project/
├── English/            train.py, preprocess.py, inference.py, config.yaml, checkpoints/
├── Telugu/             the same, with a Telugu config
├── fastspeech2/        shared model code, ported from the reference
├── test_sentences/     the assignment's test sentences
├── Results/
│   ├── English/        loss plots, generated audio, variance-adaptor plots
│   └── Telugu/
├── Report/
├── README.md
└── requirements.txt
```

The two language folders are thin entry points. Everything else is shared,
since the languages differ in their data and text handling, not in the network.

## Setup

```bash
conda create -n fs2 python=3.11
conda activate fs2
pip install -r requirements.txt
```

Two things pip will not do for you. `g2p_en` fetches its tagger from NLTK the
first time it is called, so it imports fine and then fails during English
inference:

```bash
python -c "import nltk; nltk.download('averaged_perceptron_tagger_eng'); nltk.download('cmudict')"
```

MFA is conda-only, and only needed to redo alignments:

```bash
conda create -n mfa -c conda-forge montreal-forced-aligner=3.4.3
```

## Running it

### Data

```bash
python scripts/download_data.py telugu
python scripts/download_data.py english
```

Svarah is gated, so accept the terms at
<https://huggingface.co/datasets/ai4bharat/Svarah> and run `hf auth login`
first. There is no mirror.

```bash
python scripts/prepare_telugu.py
python scripts/prepare_svarah.py --speaker-mode demographic
```

Both write `raw_data/<language>/` with 22.05 kHz wavs, `.lab` transcripts and a
`metadata.json` holding the split, which is fixed here rather than at
preprocessing time so it stays reproducible. `prepare_svarah.py --inspect`
reports what speaker signal the release actually carries.

### Alignment

Telugu has no pretrained MFA model, so one gets trained:

```bash
python scripts/build_telugu_lexicon.py --rewrite-labs
mfa validate raw_data/Telugu lexicon/telugu_lexicon.txt -j 8
mfa train raw_data/Telugu lexicon/telugu_lexicon.txt \
    mfa_data/telugu_acoustic.zip --output_directory mfa_data/textgrids_telugu --clean -j 8
```

English uses the pretrained model, but the dictionary needs extending first:
4.53% of Svarah tokens are out of vocabulary and 33.1% of utterances contain at
least one.

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

### Features, training, inference

From `English/` or `Telugu/`:

```bash
python preprocess.py
python train.py --restore_step 0
python inference.py --restore_step 60000
```

Config values can be overridden without editing the file, which is how the same
config runs locally and on a GPU box:

```bash
python train.py --restore_step 0 \
    --set train.path.ckpt_path=/kaggle/working/ckpt \
    --set train.step.total_step=80000
```

Training picks CUDA, then MPS, then CPU. `FS2_DEVICE` forces one.
`FS2_MAX_HOURS` stops at a clean step and saves before a hosted session's hard
limit, which otherwise kills the job and discards every checkpoint.

```bash
python scripts/plot_losses.py --language Telugu
```

### Kaggle

MPS runs at about 3.15 s/step, which does not get anywhere near 80000 steps, so
training runs on a Kaggle GPU, driven from the command line:

```bash
python scripts/package_for_kaggle.py --language Telugu --upload --user <kaggle-user>
python scripts/launch_kaggle.py --language Telugu --push
python scripts/launch_kaggle.py --language Telugu --status
python scripts/launch_kaggle.py --language Telugu --pull
```

The package step uploads code, features and the vocoder, about 580 MB rather
than the whole corpus. Sessions are capped at 12 h, so one push trains for at
most 11 and saves. Push again with `--restore-step` to continue, and the last
run's output is attached automatically.

Note that `package_for_kaggle.py` is what ships the training code, so a change
to `fastspeech2/` needs a re-upload, not just a `launch_kaggle.py --push`.

## Checkpoints

Each save writes a full checkpoint with optimiser state (420 MB) and a
weights-only copy (140 MB). Inference needs only the weights-only one. Both are
too big for git, so they are linked below.

[Both are in one Google Drive folder.](https://drive.google.com/drive/folders/1NdqrYxD7Aa2KroXQ8ad7mIBCqE0oMkWw)

| model | step | reached | file | link |
|---|---|---|---|---|
| Telugu | 60000 | stopped at 63675 of 80000 | `telugu_60000.pth.tar` | [download](https://drive.google.com/file/d/1Qg88v7XsNzVJduXm4rOUvhTccaSdRVRA/view?usp=sharing) |
| English | 73100 | stopped at 73100 of 80000 | `english_73100.pth.tar` | [download](https://drive.google.com/file/d/1qfxqT4mpIB9dzPzJsSCK4zOY7prU7fik/view?usp=sharing) |

Neither run finished 80000 steps. Section 3.3 of the report shows this changed
nothing: both models converged inside about 10000 steps and the rest was
overfitting.

To use one, rename it to `<step>.pth.tar`, drop it in the matching
`checkpoints/` folder and pass that step:

```bash
mv telugu_60000.pth.tar Telugu/checkpoints/60000.pth.tar
cd Telugu && python inference.py --restore_step 60000

mv english_73100.pth.tar English/checkpoints/73100.pth.tar
cd English && python inference.py --restore_step 73100
```

Both files are over 100 MB, so Drive shows a virus-scan warning instead of
downloading directly. In a browser just click through it. On the command line
that page is what `wget` saves, so use:

```bash
curl -L -o telugu_60000.pth.tar \
  "https://drive.usercontent.google.com/download?id=1Qg88v7XsNzVJduXm4rOUvhTccaSdRVRA&export=download&confirm=t"
curl -L -o english_73100.pth.tar \
  "https://drive.usercontent.google.com/download?id=1qfxqT4mpIB9dzPzJsSCK4zOY7prU7fik&export=download&confirm=t"
```

The vocoder is the universal HiFi-GAN checkpoint from the reference repo,
unzipped into `fastspeech2/hifigan/`.

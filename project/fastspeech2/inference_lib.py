"""Synthesis and variance-adaptor analysis for the assignment test sentences.

Produces, per sentence and per language:
  * a wav in Results/<lang>/test_generated_audios/
  * three plots in Results/<lang>/variance_adaptor_plots/{duration,pitch,energy}/

The variance adaptor predicts a duration, a pitch and an energy value for every
input phoneme. Those are plotted against the phoneme axis so the predictions can
be read against the text, which is what the report needs for sentences like the
noun/verb "conduct" pair.
"""

import json
import os
import re
from string import punctuation

import matplotlib
import numpy as np
import torch

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from device_utils import device  # noqa: E402
from text import text_to_sequence  # noqa: E402
from text.english_norm import normalise_for_inference as english_normalise  # noqa: E402
from text.telugu_g2p import g2p_word  # noqa: E402
from text.telugu_norm import normalise as telugu_normalise  # noqa: E402
from utils.model import get_model, get_vocoder  # noqa: E402
from utils.tools import to_device  # noqa: E402


def read_lexicon(lex_path):
    lexicon = {}
    with open(lex_path, encoding="utf-8") as f:
        for line in f:
            parts = re.split(r"\s+", line.strip("\n"))
            if len(parts) < 2:
                continue
            word, phones = parts[0], parts[1:]
            lexicon.setdefault(word.lower(), phones)
    return lexicon


def preprocess_english(text, preprocess_config):
    """English front end.

    Numbers are expanded with the same code that built every training
    transcript, which the reference does not do. Without it E4's "10:45 AM on
    12 November 2026" reaches the lexicon as digits and comes out as nonsense,
    because the model only ever saw number words.
    """
    from g2p_en import G2p

    text = english_normalise(text).rstrip(punctuation)
    lexicon = read_lexicon(preprocess_config["path"]["lexicon_path"])

    g2p = G2p()
    phones = []
    words = re.split(r"([,;.\-\?\!\s+])", text)
    for w in words:
        if w.lower() in lexicon:
            phones += lexicon[w.lower()]
        else:
            phones += list(filter(lambda p: p != " ", g2p(w)))
    phones = "{" + "}{".join(phones) + "}"
    phones = re.sub(r"\{[^\w\s]?\}", "{sp}", phones)
    phones = phones.replace("}{", " ")
    return phones


def preprocess_telugu(text, preprocess_config):
    """Telugu front end. Numbers are expanded first, then each word is looked up
    in the generated lexicon, falling back to the rule-based G2P for anything
    unseen (which is what happens to the broken word in T3)."""
    text = telugu_normalise(text)
    lexicon = read_lexicon(preprocess_config["path"]["lexicon_path"])

    phones = []
    for w in text.split():
        w = w.strip(punctuation + "।॥")
        if not w:
            continue
        if w.lower() in lexicon:
            phones += lexicon[w.lower()]
        else:
            got = g2p_word(w)
            if got:
                phones += got
    return "{" + " ".join(phones) + "}"


def phones_to_sequence(phones, preprocess_config):
    cleaners = preprocess_config["preprocessing"]["text"]["text_cleaners"]
    return np.array(text_to_sequence(phones, cleaners))


def phone_labels(phones):
    """Strip the curly braces and return the phone list for plot tick labels."""
    return phones.strip("{}").replace("}{", " ").split()


def load_stats(preprocess_config):
    path = os.path.join(preprocess_config["path"]["preprocessed_path"], "stats.json")
    with open(path, encoding="utf-8") as f:
        stats = json.load(f)
    return stats


def denormalise(values, mean, std, enabled):
    return values * std + mean if enabled else values


def plot_variance(values, labels, title, ylabel, out_path, kind):
    n = len(values)
    fig, ax = plt.subplots(figsize=(max(6, n * 0.28), 3.2))
    x = np.arange(n)

    if kind == "duration":
        ax.bar(x, values, color="#4C72B0", width=0.8)
    else:
        ax.plot(x, values, color="#C44E52", linewidth=1.4, marker="o", markersize=2.5)
        ax.axhline(0, color="grey", linewidth=0.6, linestyle=":")

    ax.set_title(title, fontsize=10)
    ax.set_ylabel(ylabel, fontsize=9)
    ax.set_xlabel("phoneme", fontsize=9)
    ax.set_xlim(-0.5, n - 0.5)

    # label every phoneme when there is room, otherwise thin them out
    stride = 1 if n <= 60 else int(np.ceil(n / 60))
    ax.set_xticks(x[::stride])
    ax.set_xticklabels([labels[i] for i in range(0, n, stride)], rotation=90, fontsize=5)
    ax.tick_params(axis="y", labelsize=7)
    fig.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def synthesise_one(model, vocoder, configs, phones, sentence_id):
    preprocess_config, model_config, train_config = configs

    sequence = phones_to_sequence(phones, preprocess_config)
    texts = np.array([sequence])
    text_lens = np.array([len(sequence)])
    speakers = np.array([0])
    batch = (
        [sentence_id],
        [""],
        torch.from_numpy(speakers).long().to(device),
        torch.from_numpy(texts).long().to(device),
        torch.from_numpy(text_lens).to(device),
        int(max(text_lens)),
    )

    with torch.no_grad():
        output = model(*(batch[2:]))

    mel_postnet = output[1]
    p_pred = output[2][0].detach().cpu().numpy()
    e_pred = output[3][0].detach().cpu().numpy()
    d_rounded = output[5][0].detach().cpu().numpy()
    mel_len = int(output[9][0].item())

    from utils.model import vocoder_infer

    wav = vocoder_infer(
        mel_postnet.transpose(1, 2),
        vocoder,
        model_config,
        preprocess_config,
        lengths=[mel_len * preprocess_config["preprocessing"]["stft"]["hop_length"]],
    )[0]

    return wav, p_pred, e_pred, d_rounded


def run_inference(language, config_path, restore_step, sentences_path, results_root,
                  overrides=None):
    """Synthesise every test sentence for one language and write the wavs and
    the three variance-adaptor plots."""
    from scipy.io import wavfile

    from config_utils import load_configs

    configs = load_configs(config_path, overrides)
    preprocess_config, model_config, train_config = configs

    with open(sentences_path, encoding="utf-8") as f:
        sentences = json.load(f)[language.lower()]

    class _Args:
        pass

    args = _Args()
    args.restore_step = restore_step
    model = get_model(args, configs, device, train=False)
    model.eval()
    vocoder = get_vocoder(model_config, device)

    stats = load_stats(preprocess_config)
    pitch_min, pitch_max, pitch_mean, pitch_std = stats["pitch"]
    energy_min, energy_max, energy_mean, energy_std = stats["energy"]
    pitch_norm = preprocess_config["preprocessing"]["pitch"]["normalization"]
    energy_norm = preprocess_config["preprocessing"]["energy"]["normalization"]

    sr = preprocess_config["preprocessing"]["audio"]["sampling_rate"]
    hop = preprocess_config["preprocessing"]["stft"]["hop_length"]

    audio_dir = os.path.join(results_root, "test_generated_audios")
    plot_root = os.path.join(results_root, "variance_adaptor_plots")
    os.makedirs(audio_dir, exist_ok=True)

    front_end = preprocess_telugu if language.lower() == "telugu" else preprocess_english
    summary = {}

    for s in sentences:
        sid, text = s["id"], s["text"]
        phones = front_end(text, preprocess_config)
        labels = phone_labels(phones)

        wav, p_pred, e_pred, d_rounded = synthesise_one(
            model, vocoder, configs, phones, sid
        )

        wav_path = os.path.join(audio_dir, f"{sid}.wav")
        wavfile.write(wav_path, sr, wav)

        n = min(len(labels), len(d_rounded))
        labels, d_rounded = labels[:n], d_rounded[:n]
        p_pred, e_pred = p_pred[:n], e_pred[:n]

        dur_ms = d_rounded * hop / sr * 1000.0
        pitch_hz = denormalise(p_pred, pitch_mean, pitch_std, pitch_norm)
        energy_v = denormalise(e_pred, energy_mean, energy_std, energy_norm)

        plot_variance(
            dur_ms, labels, f"{sid} predicted duration", "ms",
            os.path.join(plot_root, "duration", f"{sid}.png"), "duration",
        )
        plot_variance(
            pitch_hz, labels, f"{sid} predicted pitch (F0)",
            "Hz" if pitch_norm else "normalised",
            os.path.join(plot_root, "pitch", f"{sid}.png"), "pitch",
        )
        plot_variance(
            energy_v, labels, f"{sid} predicted energy", "energy",
            os.path.join(plot_root, "energy", f"{sid}.png"), "energy",
        )

        summary[sid] = {
            "text": text,
            "phones": " ".join(labels),
            "n_phones": int(n),
            "total_duration_s": round(float(dur_ms.sum()) / 1000.0, 3),
            "pitch_hz_mean": round(float(np.mean(pitch_hz)), 2),
            "energy_mean": round(float(np.mean(energy_v)), 4),
        }
        print(f"{sid}: {n} phones, {summary[sid]['total_duration_s']:.2f}s -> {wav_path}")

    out = os.path.join(results_root, "inference_summary.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=1)
    print(f"\nsummary -> {out}")
    return summary

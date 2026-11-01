"""Load the single per-language config.yaml and hand back the three dicts the
reference implementation expects.

The assignment asks for one config.yaml per language. ming024/FastSpeech2 wants
preprocess_config, model_config and train_config separately, so the merged file
is split here. Relative paths are resolved against the config file's own
directory, which keeps the config readable and lets the entry points be run
from anywhere.
"""

from pathlib import Path

import yaml

# keys whose values are filesystem paths and therefore need resolving
_PATH_KEYS = {
    "corpus_path",
    "lexicon_path",
    "raw_path",
    "textgrid_path",
    "preprocessed_path",
    "ckpt_path",
    "log_path",
    "result_path",
}


def _resolve(section, base):
    if not isinstance(section, dict):
        return
    for k, v in section.items():
        if k in _PATH_KEYS and isinstance(v, str):
            section[k] = str((base / v).resolve())
        else:
            _resolve(v, base)


def _override(cfg, dotted, value):
    """Set a nested key such as 'train.path.ckpt_path' to an absolute path."""
    section, *rest = dotted.split(".")
    if section not in cfg:
        raise KeyError(f"unknown config section {section!r} in override {dotted!r}")
    node = cfg[section]
    for k in rest[:-1]:
        if k not in node:
            raise KeyError(f"unknown config key {dotted!r}")
        node = node[k]
    leaf = rest[-1]
    if leaf not in node:
        raise KeyError(f"unknown config key {dotted!r}")
    node[leaf] = str(Path(value).expanduser().resolve()) if leaf in _PATH_KEYS else value


def load_configs(config_path, overrides=None):
    """Load the merged config.

    overrides is a list of "section.key.subkey=value" strings. This exists so
    the same config file works unchanged on this machine and on Kaggle, where
    the data lives under /kaggle/input and output must go to /kaggle/working.
    """
    config_path = Path(config_path).resolve()
    with config_path.open(encoding="utf-8") as f:
        cfg = yaml.load(f, Loader=yaml.FullLoader)

    missing = {"preprocess", "model", "train"} - set(cfg)
    if missing:
        raise KeyError(f"{config_path} is missing top-level section(s): {sorted(missing)}")

    base = config_path.parent
    for name in ("preprocess", "model", "train"):
        _resolve(cfg[name], base)

    for item in overrides or []:
        if "=" not in item:
            raise ValueError(f"override must look like section.key=value, got {item!r}")
        dotted, value = item.split("=", 1)
        if value.isdigit():
            value = int(value)
        _override(cfg, dotted.strip(), value)

    return cfg["preprocess"], cfg["model"], cfg["train"]

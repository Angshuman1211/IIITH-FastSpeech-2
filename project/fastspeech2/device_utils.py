"""Single place that decides what hardware to run on.

The reference implementation hardcoded `cuda if available else cpu` in six
files. On Apple silicon that silently falls back to CPU, so this adds MPS and
keeps the choice in one place. Set FS2_DEVICE to override, which is what the
benchmark script uses to compare backends.
"""

import os

import torch


def get_device():
    forced = os.environ.get("FS2_DEVICE")
    if forced:
        return torch.device(forced)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


device = get_device()

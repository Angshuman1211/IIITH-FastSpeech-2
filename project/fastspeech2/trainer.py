"""Training loop, adapted from ming024/FastSpeech2 train.py.

Changes from the reference:
  * device comes from device_utils, so MPS is used when present
  * nn.DataParallel is only applied when there is genuinely more than one CUDA
    device. The reference wrapped the model unconditionally, which is pointless
    on a single device and breaks on MPS.
  * every logged step is appended to losses_train.csv / losses_val.csv, because
    the assignment wants training and validation loss curves and the reference
    only wrote them to TensorBoard
  * a weights-only copy of each checkpoint is saved alongside the full one, so
    the repository can ship a checkpoint without the optimizer state
  * FS2_MAX_HOURS stops the run cleanly and saves a checkpoint before a hosted
    session's hard limit, which would otherwise kill the job and discard the
    output
"""

import csv
import os
import time

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from dataset import Dataset
from device_utils import device
from evaluate import evaluate
from model import FastSpeech2Loss
from utils.model import get_model, get_param_num, get_vocoder
from utils.tools import log, synth_one_sample, to_device

LOSS_FIELDS = ["step", "total", "mel", "mel_postnet", "pitch", "energy", "duration"]


def _init_csv(path):
    if not os.path.exists(path):
        with open(path, "w", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow(LOSS_FIELDS)


def _append_csv(path, step, losses):
    with open(path, "a", newline="", encoding="utf-8") as f:
        csv.writer(f).writerow([step] + [f"{v:.6f}" for v in losses])


def train(args, configs):
    preprocess_config, model_config, train_config = configs
    print(f"Training on device: {device}")

    dataset = Dataset("train.txt", preprocess_config, train_config, sort=True, drop_last=True)
    batch_size = train_config["optimizer"]["batch_size"]
    group_size = 4
    assert batch_size * group_size < len(dataset), (
        f"batch_size*group_size ({batch_size * group_size}) must be < dataset size ({len(dataset)})"
    )
    loader = DataLoader(
        dataset,
        batch_size=batch_size * group_size,
        shuffle=True,
        collate_fn=dataset.collate_fn,
    )

    model, optimizer = get_model(args, configs, device, train=True)
    # Only worth it with several CUDA devices; on MPS this wrapper breaks.
    if device.type == "cuda" and torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)
        bare_model = model.module
    else:
        bare_model = model

    num_param = get_param_num(model)
    Loss = FastSpeech2Loss(preprocess_config, model_config).to(device)
    print("Number of FastSpeech2 Parameters:", num_param)

    vocoder = get_vocoder(model_config, device)

    for p in train_config["path"].values():
        os.makedirs(p, exist_ok=True)
    train_log_path = os.path.join(train_config["path"]["log_path"], "train")
    val_log_path = os.path.join(train_config["path"]["log_path"], "val")
    os.makedirs(train_log_path, exist_ok=True)
    os.makedirs(val_log_path, exist_ok=True)
    train_logger = SummaryWriter(train_log_path)
    val_logger = SummaryWriter(val_log_path)

    train_csv = os.path.join(train_config["path"]["log_path"], "losses_train.csv")
    val_csv = os.path.join(train_config["path"]["log_path"], "losses_val.csv")
    _init_csv(train_csv)
    _init_csv(val_csv)

    step = args.restore_step + 1
    epoch = 1
    grad_acc_step = train_config["optimizer"]["grad_acc_step"]
    grad_clip_thresh = train_config["optimizer"]["grad_clip_thresh"]
    total_step = train_config["step"]["total_step"]
    log_step = train_config["step"]["log_step"]
    save_step = train_config["step"]["save_step"]
    synth_step = train_config["step"]["synth_step"]
    val_step = train_config["step"]["val_step"]

    outer_bar = tqdm(total=total_step, desc="Training", position=0)
    outer_bar.n = args.restore_step
    outer_bar.update()

    # Hosted sessions are killed at a hard wall clock and the output is lost
    # with them, so stop a little short of it and save on the way out.
    max_hours = float(os.environ.get("FS2_MAX_HOURS", "0"))
    started = time.time()

    def save_ckpt(at_step):
        ckpt_dir = train_config["path"]["ckpt_path"]
        torch.save(
            {
                "model": bare_model.state_dict(),
                "optimizer": optimizer._optimizer.state_dict(),
            },
            os.path.join(ckpt_dir, f"{at_step}.pth.tar"),
        )
        # weights only, for shipping in the repository
        torch.save(
            {"model": bare_model.state_dict()},
            os.path.join(ckpt_dir, f"{at_step}_weights_only.pth.tar"),
        )

    done = False
    out_of_time = False
    while not done:
        inner_bar = tqdm(total=len(loader), desc=f"Epoch {epoch}", position=1)
        for batchs in loader:
            for batch in batchs:
                batch = to_device(batch, device)
                output = model(*(batch[2:]))

                losses = Loss(batch, output)
                total_loss = losses[0]

                total_loss = total_loss / grad_acc_step
                total_loss.backward()
                if step % grad_acc_step == 0:
                    nn.utils.clip_grad_norm_(model.parameters(), grad_clip_thresh)
                    optimizer.step_and_update_lr()
                    optimizer.zero_grad()

                if step % log_step == 0:
                    values = [l.item() for l in losses]
                    message1 = f"Step {step}/{total_step}, "
                    message2 = (
                        "Total Loss: {:.4f}, Mel Loss: {:.4f}, Mel PostNet Loss: {:.4f}, "
                        "Pitch Loss: {:.4f}, Energy Loss: {:.4f}, Duration Loss: {:.4f}"
                    ).format(*values)
                    with open(os.path.join(train_log_path, "log.txt"), "a") as f:
                        f.write(message1 + message2 + "\n")
                    outer_bar.write(message1 + message2)
                    log(train_logger, step, losses=values)
                    _append_csv(train_csv, step, values)

                if step % synth_step == 0:
                    fig, wav_reconstruction, wav_prediction, tag = synth_one_sample(
                        batch, output, vocoder, model_config, preprocess_config
                    )
                    log(train_logger, fig=fig, tag=f"Training/step_{step}_{tag}")
                    sampling_rate = preprocess_config["preprocessing"]["audio"]["sampling_rate"]
                    log(
                        train_logger,
                        audio=wav_reconstruction,
                        sampling_rate=sampling_rate,
                        tag=f"Training/step_{step}_{tag}_reconstructed",
                    )
                    log(
                        train_logger,
                        audio=wav_prediction,
                        sampling_rate=sampling_rate,
                        tag=f"Training/step_{step}_{tag}_synthesized",
                    )

                if step % val_step == 0:
                    model.eval()
                    message, val_losses = evaluate(model, step, configs, val_logger, vocoder)
                    with open(os.path.join(val_log_path, "log.txt"), "a") as f:
                        f.write(message + "\n")
                    outer_bar.write(message)
                    _append_csv(val_csv, step, val_losses)
                    model.train()

                if step % save_step == 0 or step == total_step:
                    save_ckpt(step)

                if max_hours and step % log_step == 0:
                    if time.time() - started > max_hours * 3600:
                        out_of_time = True

                if step >= total_step or out_of_time:
                    done = True
                    break
                step += 1
                outer_bar.update(1)
            if done:
                break
            inner_bar.update(1)
        epoch += 1

    if out_of_time:
        save_ckpt(step)
        hrs = (time.time() - started) / 3600
        print(f"Stopped at step {step} after {hrs:.2f} h, short of {total_step}, "
              f"because FS2_MAX_HOURS={max_hours} was reached. Checkpoint saved.")
    else:
        print(f"Training finished at step {step}")

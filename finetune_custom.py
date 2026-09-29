#!/usr/bin/env python
"""
Fine-tune a released checkpoint on your own labelled audio.

    python finetune_custom.py --csv mydata.csv --out ckpts/custom/my_run

Starts from LT seed 1 in ckpts/ (DEFAULT_PRETRAINED); --pretrained takes any
other run directory or .pth.

The CSV needs two columns, `path` and `labels`:

    path,labels
    clips/robin_01.wav,Erithacus rubecula
    clips/mixed_02.wav,Erithacus rubecula;Turdus merula
    clips/quiet_03.wav,

Relative paths resolve against the CSV's directory (--audio-root overrides).
`labels` holds zero or more names separated by `;` (--label-sep); an empty cell
is a valid negative example. Classes are collected from the file unless
--classes is given.

Training is multi-label (BCE), matching the architecture: the scores are
per-class presence, not a distribution summing to one. The backbone transfers
and the head is rebuilt, since the attention convolutions are sized by the
class count -- --freeze-backbone trains only that head.

Writes <out>/models/model.pth as {"state_dict", "config"}, the same layout as
the released checkpoints, so export_onnx.py and sa4birds.evaluation.load_model
accept it unchanged.
"""
import argparse
import copy
import csv
import json
import os
import random
import sys

import numpy as np
import torch
import torch.nn as nn
from omegaconf import OmegaConf
from torch.utils.data import DataLoader, Dataset
from torchaudio.transforms import MelScale, Spectrogram

from sa4birds.models.dsa import DSA
from sa4birds.models.ssa import SSA
from sa4birds.utils.metric import TopKAccuracy, calculate_auc, calculate_map
from sa4birds.utils.power_to_db import PowerToDB

import librosa


# LT seed 1, as listed in registry.py: trained across the full XCL label space,
# so the most general of the three regimes. Relative to the project root.
DEFAULT_PRETRAINED = "ckpts/LT/dsa_LT_seed1"

ARCHITECTURES = {"DSA": DSA, "SSA": SSA}


# ---------------------------------------------------------------- data

class CsvClipDataset(Dataset):
    """Rows of (path, label-set) -> (log-mel spectrogram, multi-hot target).

    The frontend mirrors sa4birds/utils/transform.py: power spectrogram, mel scale, dB,
    then (x - mean) / (std * 2). The factor 2 is part of the convention the
    checkpoints were trained with. Unlike transform.py, the 80 dB floor is taken
    per clip rather than per batch.
    """

    def __init__(self, rows, label_map, cfg, train):
        self.rows = rows
        self.label_map = label_map
        self.train = train

        fe = cfg.frontend
        self.sample_rate = int(fe.sample_rate)
        self.target_length = int(fe.val_target_length)
        self.mean, self.std = float(fe.mean), float(fe.std)
        # Yields exactly target_length frames, so no spectrogram padding is needed.
        self.num_samples = int(fe.hop_length) * (self.target_length - 1)

        self.to_spec = Spectrogram(n_fft=int(fe.n_fft), hop_length=int(fe.hop_length),
                                   power=float(fe.power))
        self.to_mel = MelScale(n_mels=int(fe.n_mels), sample_rate=self.sample_rate,
                               n_stft=int(fe.n_stft))
        self.to_db = PowerToDB()

    def __len__(self):
        return len(self.rows)

    def _window(self, wav):
        """Random window during training, centred during validation so the
        metric is deterministic across epochs."""
        n = self.num_samples
        if len(wav) < n:
            return np.pad(wav, (0, n - len(wav)))
        if len(wav) == n:
            return wav
        start = random.randint(0, len(wav) - n) if self.train else (len(wav) - n) // 2
        return wav[start:start + n]

    def __getitem__(self, idx):
        path, labels = self.rows[idx]
        wav, _ = librosa.load(path, sr=self.sample_rate, mono=True)
        wav = self._window(np.asarray(wav, dtype=np.float32))

        spec = self.to_db(self.to_mel(self.to_spec(torch.from_numpy(wav).unsqueeze(0))))
        spec = (spec - self.mean) / (self.std * 2)

        target = torch.zeros(len(self.label_map), dtype=torch.float32)
        for name in labels:
            target[self.label_map[name]] = 1.0
        return spec, target


def read_csv(path, label_sep, audio_root):
    """CSV -> [(absolute audio path, [label, ...]), ...]. Every referenced file
    is checked up front rather than failing mid-run."""
    rows, missing = [], []
    with open(path, newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        cols = {c.strip().lower(): c for c in (reader.fieldnames or [])}
        if "path" not in cols:
            sys.exit(f"{path}: needs a 'path' column, found {reader.fieldnames}")
        if "labels" not in cols:
            sys.exit(f"{path}: needs a 'labels' column, found {reader.fieldnames}")
        for i, row in enumerate(reader, start=2):
            raw = (row[cols["path"]] or "").strip()
            if not raw:
                continue
            clip = raw if os.path.isabs(raw) else os.path.join(audio_root, raw)
            if not os.path.exists(clip):
                missing.append(f"line {i}: {clip}")
            labels = [l.strip() for l in (row[cols["labels"]] or "").split(label_sep)]
            rows.append((clip, [l for l in labels if l]))
    if missing:
        sys.exit("audio files referenced by the CSV do not exist:\n  " +
                 "\n  ".join(missing[:20]) +
                 (f"\n  ... and {len(missing) - 20} more" if len(missing) > 20 else ""))
    if not rows:
        sys.exit(f"{path}: no usable rows")
    return rows


# ---------------------------------------------------------------- model

def load_pretrained(model, ckpt_path, verbose=True):
    """Transfer every tensor whose shape still fits.

    The attention block's att/cla convolutions are sized by num_classes, so they
    never match a new label set and keep their fresh init. Matching by shape
    rather than by name stays correct if more class-dependent parameters appear.
    """
    blob = torch.load(ckpt_path, map_location="cpu")
    src = blob["state_dict"] if "state_dict" in blob else blob
    tgt = model.state_dict()

    keep, skipped = {}, []
    for k, v in src.items():
        if k in tgt and tgt[k].shape == v.shape:
            keep[k] = v
        else:
            skipped.append(k)
    model.load_state_dict(keep, strict=False)
    if verbose:
        print(f"  transferred {len(keep)}/{len(tgt)} tensors from the pretrained checkpoint")
        if skipped:
            print(f"  re-initialised {len(skipped)} class-dependent tensor(s): "
                  f"{', '.join(skipped[:4])}{' ...' if len(skipped) > 4 else ''}")
    return blob.get("config")


def resolve_checkpoint(arg):
    """Accept either a run directory (as listed in registry.py) or a .pth."""
    if os.path.isfile(arg):
        return arg
    p = os.path.join(arg, "models", "model.pth")
    if os.path.isfile(p):
        return p
    msg = (f"no checkpoint found at {arg}\n"
           f"(expected a .pth, or a directory containing models/model.pth)")
    if os.path.abspath(arg) == os.path.abspath(DEFAULT_PRETRAINED):
        # No weights ship with the repository; point at the release instead.
        msg += ("\n\nThis is the default starting point, and checkpoints are not\n"
                "part of the repository. Fetch the LT archive from the v1.0 release:\n"
                "    BASE=https://github.com/jab2026/sa4birds/releases/download/v1.0\n"
                "    curl -LO $BASE/sa4birds-LT-DSA.tar\n"
                "    tar -xf sa4birds-LT-DSA.tar -C .\n"
                "run this from the project root, or pass --pretrained explicitly.")
    sys.exit(msg)


# ---------------------------------------------------------------- metrics

def evaluate_metrics(targets, preds):
    """AUROC / cMAP / top-1 via sa4birds/utils/metric.py, the same functions
    validate_birdset.py reports with.

    Aggregated from their per-class output, not their mean: a class with no
    positive is marked -1 and excluded, but an all-positive class yields nan from
    sklearn, and np.mean would spread that nan over the whole average. Both are
    dropped and the scored-class count reported.

    targets is cloned because calculate_auc binarises its argument in place.
    """
    out, n_classes = {}, targets.shape[1]
    for name, fn in (("auroc", calculate_auc), ("cmap", calculate_map)):
        _, per_class = fn(targets.clone(), preds)
        scored = [v for v in per_class if v != -1 and not np.isnan(v)]
        out[name] = float(np.mean(scored)) if scored else float("nan")
        out[f"{name}_classes"] = len(scored)
        if len(scored) < n_classes:
            out.setdefault("notes", []).append(
                f"{name} averaged over {len(scored)}/{n_classes} classes "
                f"(the rest are all-positive or all-negative in this split)")

    topk = TopKAccuracy(topk=1)
    topk.update(preds, targets)
    # All-negative ("no-call") rows are excluded from the denominator, so a split
    # of only those would leave it at zero.
    out["top1"] = float(topk.compute()) if topk.total > 0 else float("nan")
    return out


# ---------------------------------------------------------------- main

def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--csv", required=True, help="CSV with 'path' and 'labels' columns")
    ap.add_argument("--pretrained", default=DEFAULT_PRETRAINED,
                    help=f"checkpoint to start from: a run directory or a .pth "
                         f"(default: {DEFAULT_PRETRAINED})")
    ap.add_argument("--out", required=True, help="output directory for the new checkpoint")
    ap.add_argument("--audio-root", default=None,
                    help="root for relative paths in the CSV (default: the CSV's directory)")
    ap.add_argument("--classes", default=None,
                    help="text file with one class name per line, fixing the label order "
                         "(default: every name seen in the CSV, sorted)")
    ap.add_argument("--label-sep", default=";", help="separator between labels (default ';')")
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--weight-decay", type=float, default=1e-2)
    ap.add_argument("--val-split", type=float, default=0.2,
                    help="fraction held out for validation (0 disables validation)")
    ap.add_argument("--freeze-backbone", action="store_true",
                    help="train only the new head -- usually right for small datasets")
    ap.add_argument("--num-workers", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", choices=["cuda", "cpu"], default=None)
    args = ap.parse_args()

    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))

    audio_root = args.audio_root or os.path.dirname(os.path.abspath(args.csv))
    rows = read_csv(args.csv, args.label_sep, audio_root)

    if args.classes:
        with open(args.classes, encoding="utf-8") as fh:
            names = [l.strip() for l in fh if l.strip()]
        unknown = {l for _, ls in rows for l in ls} - set(names)
        if unknown:
            sys.exit(f"labels in the CSV are missing from {args.classes}: "
                     f"{', '.join(sorted(unknown)[:10])}")
    else:
        names = sorted({l for _, ls in rows for l in ls})
    if not names:
        sys.exit("no labels found -- every 'labels' cell is empty")
    label_map = {n: i for i, n in enumerate(names)}

    # Build the model from the pretrained run's own config, so the frontend and
    # backbone settings match the weights exactly; only the label space changes.
    ckpt_path = resolve_checkpoint(args.pretrained)
    cfg = torch.load(ckpt_path, map_location="cpu")["config"]
    # Deep copy so the loaded config is not mutated; struct mode is relaxed so
    # num_classes and label_map can be replaced.
    cfg = copy.deepcopy(cfg)
    OmegaConf.set_struct(cfg, False)
    cfg.train.num_classes = len(names)
    cfg.train.label_map = dict(label_map)   # export_onnx.py reads class names from here

    print(f"device      : {device}")
    print(f"clips       : {len(rows)}")
    print(f"classes     : {len(names)} ({', '.join(names[:6])}{' ...' if len(names) > 6 else ''})")
    print(f"pretrained  : {ckpt_path}")

    arch = str(cfg.network.classifier)
    if arch not in ARCHITECTURES:
        sys.exit(f"unsupported classifier {arch!r} in {ckpt_path} "
                 f"(expected {' or '.join(ARCHITECTURES)})")
    model = ARCHITECTURES[arch](cfg)
    load_pretrained(model, ckpt_path)
    if args.freeze_backbone:
        for p in model.backbone.parameters():
            p.requires_grad = False
        n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"  backbone frozen; {n_train:,} trainable parameters remain")
    model.to(device)

    idx = list(range(len(rows)))
    random.Random(args.seed).shuffle(idx)
    n_val = int(len(idx) * args.val_split)
    val_rows = [rows[i] for i in idx[:n_val]]
    train_rows = [rows[i] for i in idx[n_val:]]
    print(f"split       : {len(train_rows)} train / {len(val_rows)} val")

    train_loader = DataLoader(
        CsvClipDataset(train_rows, label_map, cfg, train=True),
        batch_size=args.batch_size, shuffle=True, drop_last=False,
        num_workers=args.num_workers)
    val_loader = DataLoader(
        CsvClipDataset(val_rows, label_map, cfg, train=False),
        batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers) if val_rows else None

    # DSA/SSA already output probabilities (sigmoid inside the attention block)
    criterion = nn.BCELoss()
    params = [p for p in model.parameters() if p.requires_grad]
    optimiser = torch.optim.AdamW(params, lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimiser, T_max=args.epochs)

    os.makedirs(os.path.join(args.out, "models"), exist_ok=True)
    out_pth = os.path.join(args.out, "models", "model.pth")
    best, best_epoch, history = -float("inf"), None, []

    for epoch in range(1, args.epochs + 1):
        model.train()
        total, seen = 0.0, 0
        for spec, target in train_loader:
            spec, target = spec.to(device), target.to(device)
            # Training mode returns (weak, *attention maps); only weak is used.
            probs = model(spec, center_5s=False)[0]
            loss = criterion(probs, target)
            optimiser.zero_grad(set_to_none=True)
            loss.backward()
            optimiser.step()
            total += loss.item() * len(spec); seen += len(spec)
        scheduler.step()
        train_loss = total / max(seen, 1)

        line = f"epoch {epoch:3d}/{args.epochs}  train_loss {train_loss:.4f}"
        val_auc, metrics = float("nan"), {}
        if val_loader:
            model.eval()
            ys, ps, vtotal, vseen = [], [], 0.0, 0
            with torch.no_grad():
                for spec, target in val_loader:
                    spec = spec.to(device)
                    probs = model(spec, center_5s=False)
                    vtotal += criterion(probs, target.to(device)).item() * len(spec)
                    vseen += len(spec)
                    ys.append(target)
                    ps.append(probs.cpu())
            metrics = evaluate_metrics(torch.cat(ys), torch.cat(ps))
            val_auc = metrics["auroc"]
            line += (f"  val_loss {vtotal / max(vseen, 1):.4f}"
                     f"  auroc {metrics['auroc']:.4f}"
                     f"  cmap {metrics['cmap']:.4f}"
                     f"  top1 {metrics['top1']:.4f}")
            for note in metrics.get("notes", []):
                line += f"\n    {note}"
        print(line)
        history.append({"epoch": epoch, "train_loss": train_loss,
                        **{k: v for k, v in metrics.items() if k != "notes"}})

        # With no validation split, keep the latest epoch instead of a best.
        # Keep the epoch with the best validation AUROC; without a validation split,
        # keep the latest epoch. An epoch whose AUROC is undefined (too few
        # validation clips) never replaces a scored one.
        if val_loader is None:
            best_epoch = epoch
            torch.save({"state_dict": model.state_dict(), "config": cfg}, out_pth)
        elif not np.isnan(val_auc) and val_auc > best:
            best, best_epoch = val_auc, epoch
            torch.save({"state_dict": model.state_dict(), "config": cfg}, out_pth)

    if best_epoch is None:
        # Validation never produced a usable AUROC: keep the last epoch.
        torch.save({"state_dict": model.state_dict(), "config": cfg}, out_pth)

    with open(os.path.join(args.out, "metrics.json"), "w", encoding="utf-8") as fh:
        json.dump(history, fh, indent=2)

    print(f"\nwrote {out_pth}")
    if val_loader is None:
        print(f"  {len(names)} classes, last epoch kept (no validation split)")
    elif best_epoch is None:
        print(f"  {len(names)} classes, last epoch kept (validation AUROC was never "
              f"defined -- the split is too small)")
    else:
        print(f"  {len(names)} classes, best validation AUROC {best:.4f} (epoch {best_epoch})")
    print("Serve it with:")
    print(f"  python export_onnx.py --ckpt {out_pth} --out {os.path.join(args.out, 'onnx')}")


if __name__ == "__main__":
    main()

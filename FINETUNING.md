# Fine-tuning on your own data

`finetune_custom.py` adapts a released checkpoint to your own labelled audio —
your own species, your own recordings, your own label set.

```bash
python finetune_custom.py --csv mydata.csv --out ckpts/custom/my_run --freeze-backbone
```

---

## 1. Describe the data in a CSV

Two columns, `path` and `labels`:

```text
path,labels
clips/robin_01.wav,Erithacus rubecula
clips/mixed_02.wav,Erithacus rubecula;Turdus merula
clips/quiet_03.wav,
```

| | |
|---|---|
| `path` | A single audio file. Relative paths resolve against **the CSV's own directory** (`--audio-root` overrides). Anything `librosa` can read works — wav, flac, ogg, mp3. |
| `labels` | Zero or more class names separated by `;` (`--label-sep`). An **empty cell is a valid negative example** — a clip with no species present. |

The class list is collected from the file and sorted, unless you pass
`--classes` with a text file of one name per line to fix the order. Every
referenced file is checked before training starts, so a typo fails immediately
rather than in epoch 3.

## 2. Get a starting checkpoint

The default is **LT seed 1**:

```
ckpts/LT/dsa_LT_seed1
```

LT is trained across the full Xeno-Canto label space, which makes it the most
general of the three regimes — a DT checkpoint is specialised to a single
BirdSet task and is the wrong prior for arbitrary audio. No weights ship with
this repository, so fetch it first (see [Checkpoints](README.md#checkpoints)):

```bash
BASE=https://github.com/jab2026/sa4birds/releases/download/v1.0
curl -LO $BASE/sa4birds-LT-DSA.tar && tar -xf sa4birds-LT-DSA.tar -C .
```

`--pretrained` takes any other run directory or `.pth`. Both `DSA` and `SSA`
checkpoints work; the architecture is read from the checkpoint's own config.

## 3. Train

```bash
python finetune_custom.py \
    --csv mydata.csv \
    --out ckpts/custom/my_run \
    --freeze-backbone \
    --epochs 20 --batch-size 16 --lr 1e-4
```

| Option | Default | |
|---|---|---|
| `--csv` | *required* | the CSV above |
| `--out` | *required* | output directory |
| `--pretrained` | LT seed 1 | run directory or `.pth` to start from |
| `--freeze-backbone` | off | train only the new head |
| `--epochs` | 20 | |
| `--batch-size` | 16 | |
| `--lr` | 1e-4 | AdamW, cosine schedule |
| `--weight-decay` | 1e-2 | |
| `--val-split` | 0.2 | fraction held out; the epoch with the best validation AUROC is kept. `0` disables validation and keeps the last epoch |
| `--classes` | from CSV | text file fixing the label order |
| `--label-sep` | `;` | separator inside the `labels` cell |
| `--audio-root` | CSV's directory | root for relative paths |
| `--num-workers` | 4 | |
| `--seed` | 0 | |
| `--device` | auto | `cuda` or `cpu` |

**Use `--freeze-backbone` for small datasets.** With a few hundred clips there
is not enough signal to move a backbone trained on 9,736 species without
degrading it; training only the new head is usually both faster and better.

## 4. Serve the result

The checkpoint lands at `<out>/models/model.pth` in the same
`{"state_dict", "config"}` layout as every released checkpoint, so it works
unchanged with the export path and therefore with both applications:

```bash
python export_onnx.py --ckpt ckpts/custom/my_run/models/model.pth \
                      --out ckpts/onnx/my_run
```

`cfg.train.label_map` is written with your class names, so the exported
`metadata.json` carries them and the apps display real labels.

---

## What actually happens

**Training is multi-label**, because the architecture is: the model emits one
independent score per class, trained with binary cross-entropy. A single-label
dataset is simply the case where every row names one class — no special handling
needed, but read the outputs as *how much of each class is present*, not as a
distribution summing to one.

**The backbone transfers; the head is rebuilt.** The attention block's two 1×1
convolutions are sized by the class count, so they cannot carry over to a
different label set and start from scratch. Everything else is reused — the
transfer is decided by tensor *shape*, not by name, so it stays correct if the
architecture gains class-dependent parameters later. A typical run reports:

```
transferred 347/351 tensors from the pretrained checkpoint
re-initialised 4 class-dependent tensor(s): attention.att.weight, attention.att.bias,
                                            attention.cla.weight, attention.cla.bias
```

**Audio is windowed, not resampled in time.** Each clip contributes one
fixed-length window matching the model's native input: random during training
(cheap augmentation on long recordings), centred during validation so the metric
is deterministic across epochs. Shorter clips are zero-padded.

**Validation reports AUROC, cMAP and top-1** through
[`sa4birds/utils/metric.py`](sa4birds/utils/metric.py) — the same functions
`validate_birdset.py` uses, so the metric definitions are the same; the test
protocol (clip windows, dB scaling) is not.

They are aggregated from those functions' per-class output rather than their
mean. A class with no positive example is already excluded, but a class whose
validation rows are *all* positive is a second degenerate case: `sklearn`
returns `nan` for it, and `np.mean` would spread that `nan` across the whole
average, reporting the entire epoch as `nan`. Both are dropped and the count of
scored classes printed, so a thin split is visible rather than silent:

```
epoch   1/20  train_loss 0.7018  val_loss 0.7035  auroc 0.7500  cmap 0.9167  top1 0.3333
    auroc averaged over 2/3 classes (the rest are all-positive or all-negative in this split)
```

If you see that note often, the validation split is too small for the number of
classes — raise `--val-split`, or add clips for the classes it names.

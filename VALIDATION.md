# Validation

Reproducing the paper's numbers on the **BirdSet** and **BEANS** benchmarks.

```bash
python validate_birdset.py --mode=DT --down_task=HSN
```

Needs the dependencies from `requirements.txt` and the checkpoints downloaded
into `ckpts/` (see [Checkpoints](README.md#checkpoints)).

---

## BirdSet

```bash
python validate_birdset.py --mode=DT --down_task=HSN    # one task
python validate_birdset.py --mode=DT --down_task=ALL    # every task
python validate_birdset.py --mode=LT --down_task=ALL
python validate_birdset.py --mode=LT_SSA --down_task=ALL
```

| Option | Default | |
|---|---|---|
| `--mode` | `DT` | training regime: `DT`, `MT`, `LT` or `LT_SSA` |
| `--down_task` | `HSN` | one of the eight tasks below, or `ALL` |
| `--cpu` | off | force CPU instead of GPU |
| `--num_workers` | 6 | dataloader workers; overrides the checkpoint value |

The eight downstream tasks are **HSN, NBP, NES, PER, POW, SNE, SSW, UHH**.

### What the regimes mean

| Regime | Models |
|---|---|
| **DT** | Dedicated training — one model per task, so `--down_task` selects which |
| **MT** | Medium training — a single model evaluated on every task |
| **LT** | Large training — a single model over the full Xeno-Canto label space |
| **LT_SSA** | The LT models with the single-branch SSA head |

Each entry in `registry.py` holds **three independently trained runs**.
`validate_birdset.py` evaluates all three and reports the **mean** of each
metric across them — the reported figure is never a single run.

### Metrics

**AUROC**, **cMAP** and **top-1 accuracy**, computed by
[`sa4birds/utils/metric.py`](sa4birds/utils/metric.py). Per-class AUROC and
average precision skip classes with no positive example in the test split, and
top-1 excludes all-negative ("no-call") rows from its denominator.

The same metric functions back `finetune_custom.py`, although its validation
protocol differs — see [FINETUNING.md](FINETUNING.md).

### Notebooks

The same evaluation, with cell outputs committed as the expected results:

| Notebook | |
|---|---|
| [`notebooks/evaluation_birdset.ipynb`](notebooks/evaluation_birdset.ipynb) | evaluation across all BirdSet tasks |
| [`notebooks/evaluation_ablation_study.ipynb`](notebooks/evaluation_ablation_study.ipynb) | the ablation study experiments |

Each notebook downloads the checkpoints it needs from the release (the ablation
notebook uses the `sa4birds-ablation-*.tar` archives); see
[Checkpoints](README.md#checkpoints).

---

## BEANS

Transfer-learning experiments on the [BEANS
benchmark](https://github.com/earthspecies/beans) — various animal sounds, not
only birds. Follow that repository's instructions to download the data, set
`BEANS_ROOT` to its location, then use (it downloads the
`sa4birds-BEANS-<task>.tar` checkpoints itself):

| Notebook | |
|---|---|
| [`notebooks/evaluation_beans.ipynb`](notebooks/evaluation_beans.ipynb) | all BEANS results for our trained models |

---

## Before you start

**Disk.** BirdSet across all tasks needs roughly **160 GB** for the datasets;
BEANS needs about **320 GB**. Checkpoints add ~8.4 GB for BirdSet, ~6.4 GB for
BEANS and ~9.9 GB for the ablations. Datasets are fetched
through the HuggingFace `datasets` library on first use and cached in
`~/.cache/huggingface/`.

**GPU.** Measured on an RTX A5000 against the real HSN test set: **5.4 GB** for a DT model (21 classes) and **6.3 GB** for
LT (9,736 classes), so an 8 GB card is sufficient.

The two regimes land close together because memory is dominated by backbone
activations rather than by the classifier head or the 0.44 GB of weights —
despite LT's head being two orders of magnitude larger.

`--cpu` works, but a full evaluation run on CPU is impractically slow.

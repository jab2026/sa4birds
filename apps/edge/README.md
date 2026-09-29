# 🎙️ apps/edge

Runs the same bird recognition as [`../web`](../web) —
**9,736 bird species**, the full Xeno-Canto label space — but on a **Raspberry
Pi**, from the command line, and **fully offline**. Once a model and the four
Python packages are in it ([Setting up a device](#️-setting-up-a-device)), this
folder needs no internet, no cloud and nothing else from this repo — copy it
onto a device and it works in the field.

It's built on **Spectrogram Attention**, the method introduced in the paper
*Automated Recognition of Bird Species in Environmental Soundscapes Using
Spectrogram Attention*. See the [main project README](../../README.md) for the
research side.

✅ **Verified on real hardware** — a Raspberry Pi (Debian 11, Python 3.9.2,
4 cores, 1.8GB RAM, no internet) with an AudioMoth USB microphone. Its output
matches the workstation's to within float32 rounding noise (~1e-4): same
species, same ranking, every segment.

---

## ▶️ Analyzing recordings

Run these inside the folder's virtual environment (`source venv/bin/activate`,
see [Setting up a device](#️-setting-up-a-device)):

```bash
python3 edge_predict.py --audio recording.wav
python3 edge_predict.py --audio recordings/ --format csv --out detections.csv
```

`--audio` takes a single file or a folder (searched recursively for
`.wav` / `.flac` / `.mp3` / `.ogg` / `.m4a`). Long recordings are split into
7-second clips, each scored on its own:

```
$ python3 edge_predict.py --audio XC973025.mp3 --top-k 3 --threshold 0.5 --format csv
loaded 9,736 classes on CPUExecutionProvider in 23.5s
XC973025.mp3: 15 detection(s) in 16.4s

file,start_s,end_s,class,label,confidence
XC973025.mp3,0.0,7.0,redkit1,Red Kite (redkit1),0.9631
XC973025.mp3,0.0,7.0,blakit1,Black Kite (blakit1),0.7667
XC973025.mp3,0.0,7.0,cowpig1,Common Wood-Pigeon (cowpig1),0.7619
XC973025.mp3,7.0,14.0,coatit2,Coal Tit (coatit2),0.9776
XC973025.mp3,14.0,21.0,redkit1,Red Kite (redkit1),0.972
...
```

## 🔴 Listening live

Plug in a USB microphone and `--live` scores it continuously, in the same
7-second windows. It finds an **AudioMoth** automatically:

```bash
python3 edge_predict.py --live                        # until Ctrl+C
python3 edge_predict.py --live --duration 60 --format csv --out session.csv
```

```
$ python3 edge_predict.py --live --duration 30 --threshold 0.05 --top-k 3
loaded 9,736 classes on CPUExecutionProvider in 22.8s
listening on plughw:1,0 (Ctrl+C to stop), stopping after 30s
timestamp                 elapsed_s confidence  label
-----------------------------------------------------
2025-10-09T11:28:39+01:00     0.00s     0.7271  Hooded Crow (hoocro1)
2025-10-09T11:28:39+01:00     0.00s     0.6734  Crested Caracara (y00678)
2025-10-09T11:28:39+01:00     0.00s     0.6115  Mallard (mallar3)
2025-10-09T11:28:46+01:00     7.00s     0.6277  White-throated Crake (whtcra1)
2025-10-09T11:28:53+01:00    14.00s     0.7089  Long-billed Hermit (lobher)
2025-10-09T11:29:00+01:00    21.00s     0.1214  Hooded Crow (hoocro1)
2025-10-09T11:29:07+01:00    28.00s     0.5975  Yellow-chinned Spinetail (yecspi2)
...
```

That session was recorded indoors with nothing actually singing — read those
labels as the noise floor, not as detections. Room noise alone can score above
the 0.3 default threshold, so set `--threshold` to suit your site. (The
timestamps come from the device clock, which on a Pi without network time may
be wrong.)

📌 **It keeps up with real time** — a 7-second window scores in about **2s** on
the test Pi, so the timestamps stay 7s apart indefinitely. Only the first
segment is slower, by roughly 2s, while onnxruntime warms up; **no audio is
dropped** meanwhile.

Live output replaces file mode's `file`/`start_s`/`end_s` with `timestamp` and
`elapsed_s`. With `--format json` it emits one JSON object per line, not one
big array — the stream has no end to close an array at.

## 🎛️ Options

| Flag | What it does |
|---|---|
| `--audio` | A file or folder to analyze (not with `--live`) |
| `--live` | Score a microphone continuously instead |
| `--top-k` | Max detections per 7s segment (default 5) |
| `--threshold` | Minimum confidence to report (default 0.3) |
| `--format` | `table` (default), `csv`, or `json` |
| `--out` | Write to a file instead of the screen |
| `--ckpt` | A different ONNX export (default: `./model`) |
| `--alsa-device` | `--live` only: e.g. `plughw:3,0` if auto-detect misses |
| `--duration` | `--live` only: stop after N seconds of audio |
| `--temperature` | Attention sharpness (default: the checkpoint's own) |
| `--device` | `cpu` or `cuda` — a Pi is always CPU |

---

## 🛠️ Setting up a device

Two things are missing from this folder and both are too large to ship in a git
repository: **the model** and **the Python packages**. Add the model (and, for
an offline device, the package wheels) on a normal computer, copy the folder to
the device, then install the packages there.

**1. The model.** `edge_predict.py` loads an ONNX export from `./model` by
default. The release has a ready-made one (LT with the SSA head, already at IR version 9,
which matters on a Pi for the reason in the gotchas below):

```bash
curl -LO https://github.com/jab2026/sa4birds/releases/download/v1.0/sa4birds-LT-SSA-onnx.tar
tar -xf sa4birds-LT-SSA-onnx.tar && mv ckpts/onnx/ssa_LT_seed1 model && rm -r ckpts sa4birds-LT-SSA-onnx.tar
```

For a different checkpoint, download it ([main README](../../README.md#checkpoints))
and convert it with the project root's `export_onnx.py`; keep `--ir-version 9`:

```bash
cd ../..
python export_onnx.py --ckpt ckpts/LT/dsa_LT_seed1/models/model.pth \
                      --out "apps/edge/model" --ir-version 9
```

**2. Copy the folder to the device:**

```bash
rsync -avz "apps/edge/" user@device:~/sa4birds-edge/
```

**3. The packages** — Python 3 and four of them, installed **on the device**
into a virtual environment. With internet on the device:

```bash
python3 -m venv venv
./venv/bin/pip install -r requirements.txt
```

**Without** internet, the normal case for a field device, build `wheels/` for
the device's platform before step 2 (see below), then install from it:

```bash
python3 -m venv venv
./venv/bin/pip install --no-index --find-links=wheels -r requirements.txt
```

<details>
<summary><b>Building <code>wheels/</code> for offline install</b></summary>

Wheels must match the **device's** platform, not the machine that downloads
them. For **aarch64 + Python 3.9** (the test Pi's exact platform), from an
internet-connected x86_64 machine:

```bash
pip download --dest wheels \
  --platform manylinux2014_aarch64 --python-version 3.9 \
  --implementation cp --abi cp39 --only-binary=:all: \
  -r requirements.txt
```

Change `--platform` / `--python-version` / `--abi` to match your target (e.g.
`manylinux2014_armv7l` for 32-bit Pi OS, `cp311` for Python 3.11), then run
the offline install **on the device**.
</details>

<details>
<summary><b>Two gotchas that will bite on old ARM builds</b></summary>

**`numpy<2` is required, not a preference.** The only `onnxruntime` wheel PyPI
publishes for linux/aarch64 + Python 3.9 is `1.16.3` (~Sept 2023), built
against numpy's 1.x ABI. Left unpinned, pip resolves numpy 2.x and imports
fail with `AttributeError: _ARRAY_API not found`. Already pinned in
`requirements.txt`.

**ONNX IR version 10 won't load.** That same `onnxruntime==1.16.3` supports IR
version up to **9**, while a recent torch/onnx toolchain exports **10**:

```
Unsupported model IR version: 10, max supported IR version: 9
```

IR version is a container/format marker — *not* the opset, which is what
decides which operators a runtime can actually execute. Downgrading it is a
metadata-only change that leaves the predictions unchanged. Export with it
already applied:

```bash
cd ../..     # project root
python export_onnx.py --ckpt ckpts/LT/ssa_LT_seed1/models/model.pth \
                      --out "apps/edge/model" --ir-version 9
```

This is why the export command in step 1 passes `--ir-version 9`. Newer onnxruntime
builds handle IR 10 fine and need none of it — an already-exported model can
also be converted in place with `onnx.load` / `m.ir_version = 9` / `onnx.save`.
</details>

<details>
<summary><b>What's in this folder</b></summary>

```
apps/edge/
├── edge_predict.py     # the CLI -- onnxruntime + librosa, no torch
├── requirements.txt    # onnxruntime, librosa, numpy<2, soundfile
├── reference/          # eBird taxonomy (code -> common name), from apps/web
├── model/              # YOU ADD: the ONNX export (model.onnx + metadata.json)
└── wheels/             # YOU ADD: prebuilt wheels, for an offline install
```

The model lives **inside this folder** rather than being loaded from a shared
path, because a device in the field can't assume any storage but its own is
reachable. Point `--ckpt` elsewhere to override.

The spectrogram frontend and the numpy-side prediction rebuild are **copied
from `apps/web/app.py`, not imported** — keeping this folder standalone is
worth more here than sharing a module. `edge_predict.py`'s docstring names
the parts kept in sync by hand.
</details>

## ⚡ Performance

Measured on the test Pi with the released LT export (SSA head, 9,736 classes).
Loading the model and its class names takes about **23s** (43s on a cold
start), almost all of it ONNX graph loading and very sensitive to what the page
cache still holds. Scoring then settles at about **2s per 7-second segment** —
five segments of a 32.6s clip in 11.9s — so live capture stays ahead of the
microphone indefinitely.
`intra_op_num_threads` is pinned to the core count, since a 4-core Pi has
little scheduling headroom to spare.

Two one-off costs sit on top of that steady state and dominate any short
recording, which is why end-to-end file timings vary far more than the
per-segment figure:

| | |
|---|---|
| **Warm-up** | The first segment costs about **2s extra** while onnxruntime allocates its arenas. |
| **Decoding** | `librosa` has no `libsndfile` mp3 support here and falls back to `audioread`, which adds about **4.5s** on a 32.6s mp3. A `.wav` of the same audio decodes in well under a second. |

So the same 32.6s clip takes **16.4s** as an mp3 but **11.9s** as a wav once
warm. **Convert long recordings to wav** before a batch run if the decode is
worth avoiding.

#!/usr/bin/env python
"""
Gradio demo for a BIRD-ONLY DSA or SSA export -- torch-free serving.

    ./run_app.sh
    ./run_app.sh --ckpt /path/to/onnx_export_dir --device cpu --share

Self-contained: reference/ebird_taxonomy_v2022.csv is this directory's own copy.
The log-mel frontend is plain librosa/numpy, matching the torchaudio pipeline the
model was trained with; inference is onnxruntime, which uses CUDA when available
and falls back to CPU (--device forces one). Neither torch nor a GPU is needed.

--ckpt points at an ONNX EXPORT DIRECTORY (model.onnx + metadata.json, plus
model.onnx.data if the exporter wrote one), not a .pth -- produce one with the project root's export_onnx.py.
DEFAULT_CKPT is where the release's sa4birds-LT-SSA-onnx.tar extracts to.

Expects a bird-only checkpoint over XCL's 9,736 eBird classes, which is why there
are no taxon tabs or filters.

Scores full 7s windows (center_5s=False), not the center-5s crop
validate_birdset uses, so the model attends over the whole window. A recording is
cut into non-overlapping 7s segments, each scored independently.
"""
import argparse
import json
import os
import tempfile

import gradio as gr
import librosa
import numpy as np
import onnxruntime as ort
import pandas as pd
from PIL import Image, ImageDraw

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_CKPT = os.path.join(os.path.dirname(os.path.dirname(HERE)), "ckpts", "onnx",
                            "ssa_LT_seed1")
TAXONOMY_CSV = os.path.join(HERE, "reference", "ebird_taxonomy_v2022.csv")

SEGMENT_S = 7.0    # scored window == the model's native input length (center_5s=False)

SUMMARY_TOP_N = 25  # rows in the whole-recording summary

# Inferno-like anchor stops (dark -> purple -> orange -> pale yellow); a small LUT
# instead of a matplotlib dependency.
_CMAP_STOPS = np.array([
    [0, 0, 4], [40, 11, 84], [101, 21, 110], [159, 42, 99],
    [212, 72, 66], [245, 125, 21], [250, 193, 39], [252, 255, 164],
], dtype=np.float32)


def colorize(db):
    """(mel, time) dB array -> (mel, time, 3) uint8 image, low frequency at the bottom."""
    lo, hi = float(db.min()), float(db.max())
    norm = (db - lo) / max(hi - lo, 1e-6)
    pos = norm * (len(_CMAP_STOPS) - 1)
    i0 = np.clip(pos.astype(int), 0, len(_CMAP_STOPS) - 2)
    frac = (pos - i0)[..., None]
    rgb = _CMAP_STOPS[i0] * (1 - frac) + _CMAP_STOPS[i0 + 1] * frac
    return np.flipud(rgb).astype(np.uint8)


def bilinear_resize(arr, size):
    """2D float32 array -> resized via PIL bilinear (mode 'F' carries float32).
    Used only for attention-map upsampling, never for model input. ``size`` is
    (height, width), numpy order rather than PIL's."""
    height, width = size
    img = Image.fromarray(arr.astype(np.float32), mode="F")
    return np.array(img.resize((width, height), Image.BILINEAR))


def load_common_names(path):
    if not os.path.exists(path):
        return {}
    df = pd.read_csv(path, usecols=["SPECIES_CODE", "PRIMARY_COM_NAME"])
    return dict(zip(df["SPECIES_CODE"], df["PRIMARY_COM_NAME"]))


def draw_time_freq_axes(img, duration, sample_rate, step):
    """White time ticks + kHz labels drawn in place onto a spectrogram-like image, so
    a viewer can read off when-in-the-window and which-frequency-band without cross-
    referencing a separate figure."""
    draw = ImageDraw.Draw(img)
    width, height = img.size
    t = step
    while t < duration:
        x_px = int(t / duration * width)
        label = f"{int(t) // 60}:{int(t) % 60:02d}" if duration > 60 else f"{t:g}s"
        draw.line([(x_px, 0), (x_px, height)], fill=(255, 255, 255), width=1)
        draw.text((x_px + 3, height - 12), label, fill=(255, 255, 255))
        t += step
    draw.text((3, 2), f"{sample_rate / 2 / 1000:g} kHz", fill=(255, 255, 255))
    draw.text((3, height - 12), "0 kHz", fill=(255, 255, 255))


def display_name(code, common_names):
    if code in common_names:
        return f"{common_names[code]} ({code})"
    return code.replace("_", " ")


def fmt_time(t):
    return f"{int(t) // 60}:{int(t) % 60:02d}"


def to_csv_file(df, suffix):
    path = tempfile.NamedTemporaryFile(delete=False, suffix=suffix).name
    df.to_csv(path, index=False)
    return path


def select_providers(device, available):
    """device: None (auto -- prefer CUDA, silently fall back to CPU if it can't
    actually initialize: no GPU, no onnxruntime-gpu, a CUDA/cuDNN version mismatch,
    ...), 'cuda' (fail loudly if unusable, rather than silently serving on CPU when
    the caller asked for GPU specifically), or 'cpu'."""
    if device == "cpu":
        return ["CPUExecutionProvider"]
    if device == "cuda":
        if "CUDAExecutionProvider" not in available:
            raise RuntimeError(
                "CUDAExecutionProvider is not available in this onnxruntime install "
                "(plain `onnxruntime` is CPU-only -- pip install onnxruntime-gpu, "
                "matched to your CUDA/cuDNN; see README.md). Use --device cpu, or "
                "omit --device to auto-fall-back instead of failing.")
        return ["CUDAExecutionProvider", "CPUExecutionProvider"]
    if "CUDAExecutionProvider" in available:
        return ["CUDAExecutionProvider", "CPUExecutionProvider"]
    return ["CPUExecutionProvider"]


class Predictor:
    def __init__(self, ckpt_dir, device=None):
        self.ckpt_dir = ckpt_dir
        onnx_path = os.path.join(ckpt_dir, "model.onnx")
        meta_path = os.path.join(ckpt_dir, "metadata.json")
        if not os.path.exists(onnx_path) or not os.path.exists(meta_path):
            raise FileNotFoundError(
                f"{ckpt_dir} doesn't look like an ONNX export directory (expected "
                f"model.onnx and metadata.json -- run ../../export_onnx.py on a "
                f"checkpoint first, see README.md)")

        providers = select_providers(device, ort.get_available_providers())
        self.session = ort.InferenceSession(onnx_path, providers=providers)
        self.device = self.session.get_providers()[0]
        if device == "cuda" and self.device != "CUDAExecutionProvider":
            raise RuntimeError(
                "--device cuda was requested but onnxruntime could not initialize CUDA "
                f"and fell back to {self.device} (see the onnxruntime error above).")

        with open(meta_path, encoding="utf-8") as f:
            meta = json.load(f)
        self.class_names = meta["class_names"]
        self.default_temperature = float(meta["default_temperature"])
        self.fuse_weight = float(meta["fuse_weight"])

        fe = meta["frontend"]
        self.sample_rate = int(fe["sample_rate"])
        self.n_fft = int(fe["n_fft"])
        self.hop_length = int(fe["hop_length"])
        self.power = float(fe["power"])
        self.n_mels = int(fe["n_mels"])
        self.mean, self.std = float(fe["mean"]), float(fe["std"])
        self.target_length = int(fe["val_target_length"])
        # htk=True, norm=None are torchaudio.transforms.MelScale's defaults, not
        # librosa's -- librosa defaults to Slaney-style filters, which disagree with
        # what this model was trained on by tens of dB.
        self.mel_fb = librosa.filters.mel(sr=self.sample_rate, n_fft=self.n_fft,
                                          n_mels=self.n_mels, htk=True,
                                          norm=None).astype(np.float32)
        self.common_names = load_common_names(TAXONOMY_CSV)

    def _mel_db(self, wave):
        """Raw (non-normalized) log-mel dB spectrogram of a 1-D float32 wave at
        self.sample_rate. ``ref=1.0, amin=1e-10, top_db=80.0`` below reproduce
        sa4birds/utils/power_to_db.py's PowerToDB exactly (that module's defaults)."""
        S = librosa.stft(wave, n_fft=self.n_fft, hop_length=self.hop_length,
                         win_length=self.n_fft, window="hann", center=True,
                         pad_mode="reflect")
        power_spec = np.abs(S) ** self.power
        mel_spec = self.mel_fb @ power_spec
        log_spec = 10.0 * np.log10(np.maximum(mel_spec, 1e-10))
        log_spec -= 10.0 * np.log10(max(1.0, 1e-10))
        peak = log_spec.max()
        log_spec = np.maximum(log_spec, peak - 80.0)
        return log_spec.astype(np.float32)

    def _to_spectrogram(self, wave):
        """wave: 1-D float32 numpy at self.sample_rate, length == SEGMENT_S seconds.
        Returns (1, 1, mel, target_length) float32, normalized -- the model's input."""
        spec = self._mel_db(wave)
        diff = self.target_length - spec.shape[-1]
        if diff > 0:
            spec = np.pad(spec, ((0, 0), (0, diff)), constant_values=spec.min())
        elif diff < 0:
            spec = spec[:, :self.target_length]
        spec = (spec - self.mean) / (self.std * 2)
        return spec[None, None].astype(np.float32)

    @staticmethod
    def _to_mono(wave):
        if wave.ndim == 1:
            return wave
        # gradio's numpy audio can come back as (samples, channels) or (channels,
        # samples); the channel axis is always the shorter one.
        axis = 1 if wave.shape[1] < wave.shape[0] else 0
        return wave.mean(axis=axis)

    def _prepare_wave(self, wave, sr):
        wave = np.asarray(wave)
        if wave.dtype.kind in "iu":
            wave = wave.astype(np.float32) / np.iinfo(wave.dtype).max
        else:
            wave = wave.astype(np.float32)
        wave = self._to_mono(wave)
        if sr != self.sample_rate:
            wave = librosa.resample(wave, orig_sr=sr, target_sr=self.sample_rate)
        return wave

    def spectrogram_image(self, wave, row_scale=4, max_width=2000):
        """Log-mel dB spectrogram of the whole (already resampled/mono) recording, the
        same frontend the model scores with, minus the model-input mean/std normalize.
        Raw dB is what's conventionally shown to a person; the normalized version is
        only meaningful as the model's input tensor.

        One pixel per STFT frame would make a long recording wider than formats (WebP,
        at 16383px) or browsers tolerate, so time is block-max-pooled down to
        ``max_width`` columns first -- max, not mean, so brief calls survive pooling
        instead of being blurred into the background.
        """
        db = self._mel_db(wave)
        duration = db.shape[1] / (self.sample_rate / self.hop_length)

        if db.shape[1] > max_width:
            factor = int(np.ceil(db.shape[1] / max_width))
            pad = (-db.shape[1]) % factor
            if pad:
                db = np.pad(db, ((0, 0), (0, pad)), mode="edge")
            db = db.reshape(db.shape[0], -1, factor).max(axis=2)

        img = Image.fromarray(colorize(db)).resize(
            (max(db.shape[1], 512), db.shape[0] * row_scale), Image.NEAREST)

        # A tick per 7s segment on short recordings ties the image to the per-segment
        # table; the step grows on long ones to keep the ticks legible.
        if duration <= 12 * SEGMENT_S:
            step = SEGMENT_S
        else:
            step = min((s for s in (5, 10, 15, 30, 60, 120, 300, 600, 1800, 3600)
                       if s >= duration / 15), default=3600)
        draw_time_freq_axes(img, duration, self.sample_rate, step)
        return np.array(img)

    @staticmethod
    def _branch(logits, cla, temperature):
        """logits, cla: (num_classes, Fq, T) float32, the raw pre-temperature,
        pre-softmax outputs export_onnx.py's ExportWrapper exports. Reproduces
        MultiHeadSABlock.forward's temperature-divide + softmax-over-space +
        weighted-sum against cla in plain numpy (see export_onnx.py's docstring for
        why head_weights doesn't need to be here: heads=1 throughout this model).
        Returns (weighted (num_classes,), norm_att (num_classes, Fq, T)) -- the
        clip-level score per class, and the spatial attention map the UI visualizes.
        """
        att = logits / temperature
        flat = att.reshape(att.shape[0], -1)
        flat = flat - flat.max(axis=-1, keepdims=True)  # numerically stable softmax
        e = np.exp(flat)
        norm_att = (e / e.sum(axis=-1, keepdims=True)).reshape(att.shape)
        weighted = (norm_att * cla).sum(axis=(-2, -1))
        return weighted, norm_att

    def _infer(self, spec, temperature):
        """spec: (1,1,mel,T) float32 model input (see _to_spectrogram). Returns
        (probs (num_classes,), local_attention, global_attention), each attention
        map (num_classes, Fq, T) at its own branch's native resolution."""
        outs = self.session.run(None, {"spectrogram": spec})
        # A DSA export has four outputs, an SSA export only the local pair.
        l_weak, l_att = self._branch(outs[0][0], outs[1][0], temperature)
        if len(outs) == 2:
            return l_weak, l_att, None
        g_weak, g_att = self._branch(outs[2][0], outs[3][0], temperature)
        probs = l_weak * self.fuse_weight + g_weak * (1 - self.fuse_weight)
        return probs, l_att, g_att

    def predict(self, wave, top_k=5, threshold=0.1, temperature=None):
        """wave: 1-D float32 numpy, mono, already at self.sample_rate."""
        temperature = self.default_temperature if temperature is None else float(temperature)

        n = len(wave)
        seg_len = int(round(SEGMENT_S * self.sample_rate))
        n_segments = max(1, int(np.ceil(n / seg_len)))

        tail_pad = max(0, n_segments * seg_len - n)
        padded = np.pad(wave, (0, tail_pad))

        # detail_meta parallels detail_rows 1:1 with (class_idx, segment_index), so a
        # click on a table row in the UI knows exactly what to re-run attention on.
        detail_rows, detail_meta, best = [], [], {}
        for i in range(n_segments):
            chunk = padded[i * seg_len:(i + 1) * seg_len]

            spec = self._to_spectrogram(chunk)
            probs, _, _ = self._infer(spec, temperature)

            t0 = i * SEGMENT_S
            t1 = min(n / self.sample_rate, (i + 1) * SEGMENT_S)
            for idx in np.argsort(-probs)[:top_k]:
                p = float(probs[idx])
                if p < threshold:
                    continue
                name = self.class_names[idx]
                detail_rows.append({"start": fmt_time(t0), "end": fmt_time(t1),
                                    "label": display_name(name, self.common_names),
                                    "confidence": round(p, 3)})
                detail_meta.append((int(idx), i))
                if p > best.get(name, (0.0, 0, 0))[0]:
                    best[name] = (p, i, int(idx))

        detail_df = (pd.DataFrame(detail_rows) if detail_rows else
                    pd.DataFrame(columns=["start", "end", "label", "confidence"]))
        summary_df = self._summary(best)
        overall_best = max(best.items(), key=lambda kv: kv[1][0], default=None)
        top1 = (overall_best[1][2], overall_best[1][1]) if overall_best else None
        return summary_df, detail_df, detail_meta, top1

    def _summary(self, best):
        """best: name -> (max_confidence, segment_index, class_idx) -> one table of the
        strongest species across the whole recording, as a single ranked list."""
        rows = sorted(((n, p) for n, (p, _, _) in best.items()), key=lambda kv: -kv[1])
        rows = rows[:SUMMARY_TOP_N]
        return (pd.DataFrame([{"label": display_name(n, self.common_names),
                               "max_confidence": round(p, 3)} for n, p in rows])
                if rows else pd.DataFrame(columns=["label", "max_confidence"]))

    def attention_overlay(self, wave, segment_index, class_idx, temperature=None, row_scale=6):
        """Per-class spatial attention for one 7s segment, blended over its
        spectrogram in cyan. Renders the local (fine-grained) branch's map only. For
        a DSA export the global branch's map is essentially uniform, so it would
        localize nothing, although the prediction still fuses both branches with
        the learned fuse_weight; an SSA export has only the local branch. Also
        returns the raw 7s chunk so the UI can play back exactly what is shown."""

        seg_len = int(round(SEGMENT_S * self.sample_rate))
        start = segment_index * seg_len
        chunk = wave[start:start + seg_len]
        if len(chunk) < seg_len:
            chunk = np.pad(chunk, (0, seg_len - len(chunk)))

        db = self._mel_db(chunk)
        spec = self._to_spectrogram(chunk)
        _, l_att, _ = self._infer(spec, temperature)

        att = l_att[class_idx]
        att = (att - att.min()) / max(att.ptp(), 1e-8)

        att_full = bilinear_resize(att, db.shape)
        att_full = np.flipud(att_full)  # match colorize()'s vertical flip

        base = colorize(db).astype(np.float32)
        # Cyan, not red/yellow: _CMAP_STOPS is red-dominant throughout, so a warm
        # overlay would vanish exactly where the spectrogram is already bright.
        heat = np.zeros_like(base)
        heat[..., 1] = 255
        heat[..., 2] = 255
        # gamma < 1 lifts the mid-range, so a broad, low-contrast map is still
        # visible, not just its peak.
        alpha = (att_full[..., None] ** 0.5) * 0.85
        blended = (base * (1 - alpha) + heat * alpha).clip(0, 255).astype(np.uint8)

        img = Image.fromarray(blended).resize(
            (max(blended.shape[1] * 6, 512), blended.shape[0] * row_scale), Image.NEAREST)
        draw_time_freq_axes(img, SEGMENT_S, self.sample_rate, step=1.0)
        return np.array(img), (self.sample_rate, chunk)


def build_ui(predictor):
    with gr.Blocks(title="🐦 sa4birds — Acoustic Bird Recognition") as demo:
        gr.Markdown(
            f"# 🐦 sa4birds — Acoustic Bird Recognition\n"
            f"\n"
            f"Listens to a recording and tells you which **bird species** it hears — "
            f"**{len(predictor.class_names):,} of them**, the full Xeno-Canto label "
            f"space — with a confidence score for each.\n"
            f"\n"
            f"🎧 &nbsp;**Upload or record** a clip on the left, then press "
            f"**Analyze**. Long recordings are split into {SEGMENT_S:g}-second "
            f"clips, each checked separately."
        )
        with gr.Row():
            with gr.Column(scale=1):
                audio_in = gr.Audio(sources=["upload", "microphone"], type="numpy",
                                    label="Recording")
                top_k = gr.Slider(1, 10, value=3, step=1, label="Top-K per segment")
                threshold = gr.Slider(0.0, 1.0, value=0.4, step=0.01,
                                      label="Confidence threshold")
                temperature = gr.Slider(
                    0.1, 5.0, value=predictor.default_temperature, step=0.1,
                    label="Attention temperature",
                    info=f"Softmax temperature over time/frequency attention. Lower = "
                        f"peakier (more localized) attention, higher = smoother. "
                        f"Trained at {predictor.default_temperature:g}.")
                run_btn = gr.Button("Analyze", variant="primary")
            with gr.Column(scale=2):
                spec_out = gr.Image(label="Spectrogram (log-mel, dB)", interactive=False)
                gr.Markdown("**Top detections (whole recording)**:")
                summary_out = gr.Dataframe(interactive=False)
                summary_csv_out = gr.DownloadButton("Download summary CSV",
                                                    size="sm", interactive=False)
                detail_out = gr.Dataframe(
                    label="Per-segment detections (click a row for its attention map)",
                    interactive=False)
                detail_csv_out = gr.DownloadButton("Download per-segment CSV", size="sm",
                                                   interactive=False)
                with gr.Row():
                    with gr.Column():
                        attn_caption = gr.Markdown()
                        attn_out = gr.Image(label="Attention (cyan = where the model looked)",
                                            interactive=False)
                    with gr.Column():
                        compare_dropdown = gr.Dropdown(
                            label="Compare against another class detected in this segment",
                            choices=[])
                        compare_caption = gr.Markdown()
                        compare_out = gr.Image(label="Comparison attention", interactive=False)
                segment_audio_out = gr.Audio(
                    label="Segment audio (7s) -- shared by both sides, same segment",
                    interactive=False)

        wave_state = gr.State()
        meta_state = gr.State()
        full_detail_df_state = gr.State()
        full_detail_meta_state = gr.State()
        current_segment_state = gr.State()

        def _attention_for(wave, class_idx, seg_idx, temp):
            img, audio = predictor.attention_overlay(wave, seg_idx, class_idx,
                                                     temperature=float(temp))
            name = predictor.class_names[class_idx]
            t0, t1 = seg_idx * SEGMENT_S, (seg_idx + 1) * SEGMENT_S
            caption = (f"**{display_name(name, predictor.common_names)}** — segment "
                      f"{fmt_time(t0)}–{fmt_time(t1)} (local-branch attention)")
            return img, caption, audio

        def _compare_choices(meta, seg_idx, exclude_idx):
            """(label, class_idx) pairs for every other class detected in this same
            segment -- Gradio hands back class_idx directly on selection, so the
            compare handler needs no separate name -> index lookup."""
            return sorted({
                (display_name(predictor.class_names[ci], predictor.common_names), ci)
                for ci, si in meta if si == seg_idx and ci != exclude_idx})

        def _run(audio, k, thr, temp):
            if audio is None:
                empty = pd.DataFrame()
                no_file = gr.update(value=None, interactive=False)
                return (None, empty, empty, no_file, no_file,
                       None, None, None, "", empty, None,
                       gr.update(choices=[], value=None), None, "", None, None)
            sr, wave = audio
            wave = predictor._prepare_wave(wave, sr)
            spec = predictor.spectrogram_image(wave)
            summary_df, detail_df, detail_meta, top1 = predictor.predict(
                wave, top_k=int(k), threshold=float(thr), temperature=float(temp))
            if top1 is not None:
                attn_img, attn_cap, seg_audio = _attention_for(wave, top1[0], top1[1], temp)
                compare_update = gr.update(
                    choices=_compare_choices(detail_meta, top1[1], top1[0]), value=None)
                compare_segment = top1[1]
            else:
                attn_img, attn_cap, seg_audio = None, "No detections above threshold.", None
                compare_update = gr.update(choices=[], value=None)
                compare_segment = None
            summary_csv = gr.update(
                value=to_csv_file(summary_df, "_summary.csv"),
                interactive=True)
            detail_csv = gr.update(value=to_csv_file(detail_df, "_detail.csv"),
                                   interactive=True)
            # The compare dropdown/segment audio are seeded from top1's own segment,
            # same as the primary attention map, so they're usable immediately without
            # a row click.
            return (spec, summary_df, detail_df, summary_csv,
                   detail_csv, wave, detail_meta, attn_img, attn_cap,
                   detail_df, detail_meta,
                   compare_update, None, "", compare_segment, seg_audio)

        run_btn.click(_run, inputs=[audio_in, top_k, threshold, temperature],
                      outputs=[spec_out, summary_out, detail_out,
                               summary_csv_out, detail_csv_out, wave_state, meta_state,
                               attn_out, attn_caption, full_detail_df_state,
                               full_detail_meta_state,
                               compare_dropdown, compare_out, compare_caption,
                               current_segment_state, segment_audio_out])

        def _on_detail_select(evt: gr.SelectData, wave, meta, temp):
            no_change = (gr.update(), gr.update(), gr.update(), None, "", None, gr.update())
            if wave is None or not meta or evt.index is None:
                return no_change
            row = evt.index[0]
            if row >= len(meta):
                return no_change
            class_idx, seg_idx = meta[row]
            img, cap, seg_audio = _attention_for(wave, class_idx, seg_idx, temp)
            choices = _compare_choices(meta, seg_idx, class_idx)
            return (img, cap, gr.update(choices=choices, value=None), None, "",
                   seg_idx, seg_audio)

        detail_out.select(
            _on_detail_select, inputs=[wave_state, meta_state, temperature],
            outputs=[attn_out, attn_caption, compare_dropdown, compare_out,
                     compare_caption, current_segment_state, segment_audio_out])

        def _on_compare_select(class_idx, wave, seg_idx, temp):
            if class_idx is None or wave is None or seg_idx is None:
                return None, ""
            img, cap, _seg_audio = _attention_for(wave, class_idx, seg_idx, temp)
            return img, cap

        compare_dropdown.change(
            _on_compare_select,
            inputs=[compare_dropdown, wave_state, current_segment_state, temperature],
            outputs=[compare_out, compare_caption])
    return demo


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ckpt", default=DEFAULT_CKPT,
                    help="ONNX export directory (model.onnx + metadata.json), "
                         "from export_onnx.py -- not a .pth checkpoint")
    ap.add_argument("--device", choices=["cuda", "cpu"], default=None,
                    help="force an onnxruntime execution provider; default auto-"
                         "selects CUDA if available, falling back to CPU")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=7860)
    ap.add_argument("--share", action="store_true")
    args = ap.parse_args()

    predictor = Predictor(args.ckpt, device=args.device)
    print(f"loaded {len(predictor.class_names):,} classes, running on {predictor.device}",
          flush=True)
    demo = build_ui(predictor)
    demo.launch(server_name=args.host, server_port=args.port, share=args.share)


if __name__ == "__main__":
    main()

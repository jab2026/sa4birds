#!/usr/bin/env python
"""
apps/edge -- run a BIRD-ONLY ONNX model on-device (Raspberry Pi and similar
ARM SBCs), with no network dependency at inference time.

    python3 edge_predict.py --audio recording.wav
    python3 edge_predict.py --audio recordings/ --format csv --out detections.csv
    python3 edge_predict.py --live                    # score a live mic in real time
    python3 edge_predict.py --live --format csv --out live.csv

Leaner than apps/web/app.py: no Gradio, PIL, pandas or attention maps, which
belong to interactive inspection on a workstation. What it shares with that app --
the librosa/numpy log-mel frontend and the numpy reconstruction of predictions
from the ONNX graph's raw attention logits -- is COPIED, not imported, and kept in
sync by hand, so this directory depends on nothing else in the repo.

--ckpt points at an ONNX export directory (model.onnx + metadata.json, plus
model.onnx.data if the exporter wrote one; see README.md), defaulting to ./model, which is meant to
be copied onto the device alongside this script.

--live captures from an ALSA input through `arecord` as a subprocess emitting raw
PCM, rather than a Python audio library: a deployed device cannot fetch new
wheels, and PyAudio/sounddevice need a portaudio build matching its exact
arch+libc, while `arecord` ships with Raspberry Pi OS. The default device is
auto-detected from `arecord -l` by matching "AudioMoth".
"""
import argparse
import collections
import csv
import datetime
import json
import os
import re
import subprocess
import sys
import threading
import time

import librosa
import numpy as np
import onnxruntime as ort

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_CKPT = os.path.join(HERE, "model")
TAXONOMY_CSV = os.path.join(HERE, "reference", "ebird_taxonomy_v2022.csv")

SEGMENT_S = 7.0

# Expects a BIRD-ONLY model (9,736 eBird classes), so every prediction is a bird and
# there is no taxon column or class lookup to carry.


def load_common_names(path):
    if not os.path.exists(path):
        return {}
    names = {}
    with open(path, encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            code = row.get("SPECIES_CODE")
            name = row.get("PRIMARY_COM_NAME")
            if code and name:
                names[code] = name
    return names


def display_name(code, common_names):
    if code in common_names:
        return f"{common_names[code]} ({code})"
    return code.replace("_", " ")


def select_providers(device, available):
    """See apps/web/app.py's select_providers. On a Raspberry Pi this will
    always resolve to CPUExecutionProvider (no CUDA-capable GPU exists here),
    but keeping the same auto/cuda/cpu selection logic costs nothing and means
    this script also runs unmodified on a GPU-capable ARM board (e.g. a Jetson)
    without a code change -- only --device or the packages installed differ."""
    if device == "cpu":
        return ["CPUExecutionProvider"]
    if device == "cuda":
        if "CUDAExecutionProvider" not in available:
            raise RuntimeError(
                "CUDAExecutionProvider is not available in this onnxruntime "
                "install. Use --device cpu, or omit --device to auto-fall-back.")
        return ["CUDAExecutionProvider", "CPUExecutionProvider"]
    if "CUDAExecutionProvider" in available:
        return ["CUDAExecutionProvider", "CPUExecutionProvider"]
    return ["CPUExecutionProvider"]


def list_alsa_capture_devices():
    """Raw text of `arecord -l`, for error messages -- not parsed here."""
    try:
        return subprocess.run(["arecord", "-l"], capture_output=True, text=True,
                              timeout=5).stdout
    except (OSError, subprocess.TimeoutExpired) as e:
        return f"(couldn't run `arecord -l`: {e})"


def find_alsa_device(name_substr="AudioMoth"):
    """Auto-detect an ALSA capture device by matching `name_substr` against
        `arecord -l`, returning a string like "plughw:3,0". Returns None if nothing
        matches.

        `plughw` rather than `hw` is load-bearing: the AudioMoth's USB audio device
        exposes a single fixed native rate (384kHz on the tested unit). `hw:` records
        at that rate whatever you ask for, while `plughw:` lets ALSA resample to the
        rate requested -- here the model's 32kHz.
        """
    out = list_alsa_capture_devices()
    for m in re.finditer(r"^card (\d+): \S+ \[([^\]]+)\], device (\d+):",
                         out, re.MULTILINE):
        card, desc, device = m.group(1), m.group(2), m.group(3)
        if name_substr.lower() in desc.lower():
            return f"plughw:{card},{device}"
    return None


class _PipeDrain:
    """Continuously drains a subprocess pipe on a background thread, so the
        pipe's own OS buffer (~64KB, under 1.1s of audio here) can never fill and
        block the writer.

        Scoring one 7s segment takes several seconds on a Pi. If nothing reads
        `arecord`'s stdout meanwhile, the pipe fills, `arecord` blocks in write(),
        and it stops draining ALSA's much smaller capture ring -- an overrun that
        silently drops audio on essentially every segment. Draining here means a slow
        segment costs latency instead of audio.
        """
    def __init__(self, stream):
        self._chunks = collections.deque()
        self._nbytes = 0
        self._cv = threading.Condition()
        self._eof = False
        self._err = None
        self._thread = threading.Thread(
            target=self._run, args=(stream,), daemon=True)
        self._thread.start()

    def _run(self, stream):
        try:
            while True:
                piece = stream.read(65536)
                if not piece:
                    return
                with self._cv:
                    self._chunks.append(piece)
                    self._nbytes += len(piece)
                    self._cv.notify_all()
        except Exception as e:
            with self._cv:
                self._err = e
                self._cv.notify_all()
        finally:
            with self._cv:
                self._eof = True
                self._cv.notify_all()

    def _take_locked(self, take):
        """Caller must hold self._cv. Removes and returns up to `take` bytes
        (fewer only if the buffer doesn't have that much)."""
        take = min(take, self._nbytes)
        out = bytearray()
        while len(out) < take:
            piece = self._chunks[0]
            need = take - len(out)
            if len(piece) <= need:
                out.extend(piece)
                self._chunks.popleft()
            else:
                out.extend(piece[:need])
                self._chunks[0] = piece[need:]
        self._nbytes -= len(out)
        return bytes(out)

    def read_exact(self, n):
        """Blocks until `n` bytes are available and returns exactly that
        many, or returns fewer (down to b"") once EOF is reached with less
        than `n` left. Re-raises any exception the reader thread hit. Used
        for the main streaming path, where blocking indefinitely is correct
        -- there's always more real-time audio eventually."""
        with self._cv:
            while self._nbytes < n and not self._eof:
                self._cv.wait(timeout=1.0)
            if self._err is not None:
                raise self._err
            return self._take_locked(n)

    def read_available(self, timeout=2.0):
        """Returns whatever's currently buffered, waiting up to `timeout`
        seconds for at least one byte if the buffer is empty -- but not for
        EOF. For best-effort diagnostics (e.g. grabbing partial stderr after
        a crash) where an unbounded wait would be its own bug."""
        with self._cv:
            if self._nbytes == 0 and not self._eof:
                self._cv.wait(timeout=timeout)
            return self._take_locked(self._nbytes)


def iter_live_chunks(device, sample_rate, seg_len_samples):
    """Spawns `arecord` as a subprocess streaming raw signed-16-bit mono PCM
    at `sample_rate`, and yields one waveform per `seg_len_samples`-sample
    chunk, forever: a float32 array in [-1, 1], the same convention
    librosa.load uses. No Python audio library involved -- see the module
    docstring for why."""
    cmd = ["arecord", "-D", device, "-f", "S16_LE", "-r", str(sample_rate),
          "-c", "1", "-t", "raw", "-q", "-"]
    try:
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    except FileNotFoundError:
        raise RuntimeError(
            "`arecord` not found -- install alsa-utils (it's expected to "
            "already be present on Raspberry Pi OS)") from None

    stdout_drain = _PipeDrain(proc.stdout)
    stderr_drain = _PipeDrain(proc.stderr)
    nbytes = seg_len_samples * 2
    try:
        while True:
            buf = stdout_drain.read_exact(nbytes)
            if len(buf) < nbytes:
                err = stderr_drain.read_available().decode(errors="replace").strip()
                raise RuntimeError(
                    f"arecord exited unexpectedly (device={device!r})"
                    + (f": {err}" if err else " (no stderr output)"))
            yield np.frombuffer(buf, dtype="<i2").astype(np.float32) / 32768.0
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=2)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait()


class EdgePredictor:
    def __init__(self, ckpt_dir, device=None):
        onnx_path = os.path.join(ckpt_dir, "model.onnx")
        meta_path = os.path.join(ckpt_dir, "metadata.json")
        if not os.path.exists(onnx_path) or not os.path.exists(meta_path):
            raise FileNotFoundError(
                f"{ckpt_dir} doesn't look like an ONNX export directory "
                f"(expected model.onnx and metadata.json -- see README.md)")

        providers = select_providers(device, ort.get_available_providers())
        sess_options = ort.SessionOptions()
        # Pinned explicitly rather than left to an onnxruntime default that may
        # change between versions.
        sess_options.intra_op_num_threads = max(1, os.cpu_count() or 1)
        self.session = ort.InferenceSession(onnx_path, sess_options=sess_options,
                                            providers=providers)
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
        # htk=True, norm=None are torchaudio's MelScale defaults, not librosa's;
        # see apps/web/app.py's Predictor.__init__.
        self.mel_fb = librosa.filters.mel(sr=self.sample_rate, n_fft=self.n_fft,
                                          n_mels=self.n_mels, htk=True,
                                          norm=None).astype(np.float32)
        self.common_names = load_common_names(TAXONOMY_CSV)

    def _mel_db(self, wave):
        S = librosa.stft(wave, n_fft=self.n_fft, hop_length=self.hop_length,
                         win_length=self.n_fft, window="hann", center=True,
                         pad_mode="reflect")
        power_spec = np.abs(S) ** self.power
        mel_spec = self.mel_fb @ power_spec
        log_spec = 10.0 * np.log10(np.maximum(mel_spec, 1e-10))
        log_spec -= 10.0 * np.log10(max(1.0, 1e-10))
        peak = log_spec.max()
        return np.maximum(log_spec, peak - 80.0).astype(np.float32)

    def _to_spectrogram(self, wave):
        spec = self._mel_db(wave)
        diff = self.target_length - spec.shape[-1]
        if diff > 0:
            spec = np.pad(spec, ((0, 0), (0, diff)), constant_values=spec.min())
        elif diff < 0:
            spec = spec[:, :self.target_length]
        spec = (spec - self.mean) / (self.std * 2)
        return spec[None, None].astype(np.float32)

    @staticmethod
    def _branch(logits, cla, temperature):
        """See apps/web/app.py's Predictor._branch -- reproduces
        MultiHeadSABlock.forward's temperature-divide + softmax-over-space +
        weighted-sum against cla in numpy. heads=1 throughout this model, so
        head_weights doesn't need to be here (see export_onnx.py)."""
        att = logits / temperature
        flat = att.reshape(att.shape[0], -1)
        flat = flat - flat.max(axis=-1, keepdims=True)
        e = np.exp(flat)
        norm_att = (e / e.sum(axis=-1, keepdims=True)).reshape(att.shape)
        return (norm_att * cla).sum(axis=(-2, -1))

    def _infer(self, spec, temperature):
        outs = self.session.run(
            None, {"spectrogram": spec})
        # A DSA export has four outputs, an SSA export only the local pair.
        l_weak = self._branch(outs[0][0], outs[1][0], temperature)
        if len(outs) == 2:
            return l_weak
        g_weak = self._branch(outs[2][0], outs[3][0], temperature)
        return l_weak * self.fuse_weight + g_weak * (1 - self.fuse_weight)

    @staticmethod
    def _load_wave(path, sample_rate):
        wave, sr = librosa.load(path, sr=sample_rate, mono=True)
        return wave.astype(np.float32)

    def _score_chunk(self, chunk, top_k, threshold, temperature):
        """Scores one seg_len-sample waveform chunk, yielding (class_name,
        confidence) pairs above `threshold`, highest confidence first,
        capped at `top_k`. Shared by predict_file and predict_live -- they
        differ only in where the chunk came from and what location metadata
        (file+offset vs timestamp) gets attached to each detection."""
        spec = self._to_spectrogram(chunk)
        probs = self._infer(spec, temperature)
        for idx in np.argsort(-probs)[:top_k]:
            p = float(probs[idx])
            if p < threshold:
                continue
            yield self.class_names[idx], p

    def predict_file(self, path, top_k=5, threshold=0.1, temperature=None):
        """Yields one dict per (segment, class-above-threshold) detection, in
        the same non-overlapping-7s-window scheme as apps/web/app.py."""
        temperature = self.default_temperature if temperature is None else temperature
        wave = self._load_wave(path, self.sample_rate)

        n = len(wave)
        seg_len = int(round(SEGMENT_S * self.sample_rate))
        n_segments = max(1, int(np.ceil(n / seg_len)))
        padded = np.pad(wave, (0, max(0, n_segments * seg_len - n)))

        for i in range(n_segments):
            chunk = padded[i * seg_len:(i + 1) * seg_len]
            t0 = i * SEGMENT_S
            t1 = min(n / self.sample_rate, (i + 1) * SEGMENT_S)
            for name, p in self._score_chunk(chunk, top_k, threshold, temperature):
                yield {
                    "file": os.path.basename(path),
                    "start_s": round(t0, 2), "end_s": round(t1, 2),
                    "class": name,
                    "label": display_name(name, self.common_names),
                    "confidence": round(p, 4),
                }

    def predict_live(self, device=None, top_k=5, threshold=0.1, temperature=None,
                     duration=None):
        """Continuously scores non-overlapping SEGMENT_S windows from an ALSA
                input, with the same windowing as predict_file. Runs until `duration`
                seconds of capture time have elapsed, or forever if None; the `finally`
                closes iter_live_chunks, which tears down the arecord subprocess.

                The duration check happens BEFORE requesting the next chunk, so a run
                stops promptly rather than after capturing one more full segment.
                Detection latency is up to SEGMENT_S plus the time to score one
                segment.
                """
        temperature = self.default_temperature if temperature is None else temperature
        if device is None:
            device = find_alsa_device()
            if device is None:
                raise RuntimeError(
                    "couldn't auto-detect an AudioMoth via `arecord -l`; pass "
                    "--alsa-device explicitly. Available capture devices:\n"
                    + list_alsa_capture_devices())

        seg_len = int(round(SEGMENT_S * self.sample_rate))
        chunks = iter_live_chunks(device, self.sample_rate, seg_len)
        try:
            i = 0
            while duration is None or i * SEGMENT_S < duration:
                chunk = next(chunks)
                t0 = i * SEGMENT_S
                now = datetime.datetime.now().astimezone().isoformat(timespec="seconds")
                for name, p in self._score_chunk(chunk, top_k, threshold, temperature):
                    yield {
                        "timestamp": now,
                        "elapsed_s": round(t0, 2),
                        "class": name,
                        "label": display_name(name, self.common_names),
                        "confidence": round(p, 4),
                    }
                i += 1
        finally:
            chunks.close()


def iter_audio_files(path):
    exts = (".wav", ".flac", ".mp3", ".ogg", ".m4a")
    if os.path.isdir(path):
        for root, _, files in os.walk(path):
            for fn in sorted(files):
                if fn.lower().endswith(exts):
                    yield os.path.join(root, fn)
    else:
        yield path


def _run_files(predictor, args, out):
    rows = []
    for path in iter_audio_files(args.audio):
        t0 = time.time()
        file_rows = list(predictor.predict_file(
            path, top_k=args.top_k, threshold=args.threshold,
            temperature=args.temperature))
        rows.extend(file_rows)
        print(f"{path}: {len(file_rows)} detection(s) in {time.time() - t0:.1f}s",
             file=sys.stderr)

    if args.format == "json":
        json.dump(rows, out, indent=2)
        out.write("\n")
    elif args.format == "csv":
        writer = csv.DictWriter(out, fieldnames=[
            "file", "start_s", "end_s", "class", "label", "confidence"])
        writer.writeheader()
        writer.writerows(rows)
    else:
        if rows:
            widths = {k: max(len(k), max(len(str(r[k])) for r in rows))
                     for k in rows[0]}
            header = "  ".join(k.ljust(widths[k]) for k in rows[0])
            print(header, file=out)
            print("-" * len(header), file=out)
            for r in rows:
                print("  ".join(str(r[k]).ljust(widths[k]) for k in r), file=out)
        else:
            print("(no detections above threshold)", file=out)


# One format string for both the header and the rows of --live's table, so the two
# cannot drift apart. `label` is last and unpadded as the only variable-width
# column; `class` is omitted since `label` already contains it -- csv/json keep both.
LIVE_ROW = "{ts:<25} {el:>9} {conf:>10}  {label}"


def _run_live(predictor, args, out):
    """Streams detections as they're scored, rather than buffering like
    _run_files -- a live capture may never end, so nothing here waits to
    collect "all" rows first. --format json emits newline-delimited JSON
    (one object per line) instead of a single JSON array for the same
    reason: the array could only be closed once the stream ends."""
    device = args.alsa_device or find_alsa_device()
    if device is None:
        raise RuntimeError(
            "couldn't auto-detect an AudioMoth via `arecord -l`; pass "
            "--alsa-device explicitly. Available capture devices:\n"
            + list_alsa_capture_devices())

    fieldnames = ["timestamp", "elapsed_s", "class", "label", "confidence"]
    print(f"listening on {device} (Ctrl+C to stop)"
         + (f", stopping after {args.duration:.0f}s" if args.duration else ""),
         file=sys.stderr)

    writer = None
    if args.format == "csv":
        writer = csv.DictWriter(out, fieldnames=fieldnames)
        writer.writeheader()
    elif args.format == "json":
        print("(newline-delimited JSON: one detection per line, not a single "
             "JSON array, since this stream doesn't end on its own)",
             file=sys.stderr)
    else:
        header = LIVE_ROW.format(ts="timestamp", el="elapsed_s",
                                 conf="confidence", label="label")
        print(header, file=out)
        print("-" * len(header), file=out)
    out.flush()

    n = 0
    t_start = time.time()
    try:
        for row in predictor.predict_live(
                device=device, top_k=args.top_k, threshold=args.threshold,
                temperature=args.temperature, duration=args.duration):
            n += 1
            if args.format == "csv":
                writer.writerow(row)
            elif args.format == "json":
                json.dump(row, out)
                out.write("\n")
            else:
                print(LIVE_ROW.format(
                    ts=row["timestamp"],
                    el=f"{row['elapsed_s']:.2f}s",
                    conf=f"{row['confidence']:.4f}",
                    label=row["label"]), file=out)
            out.flush()
    except KeyboardInterrupt:
        print(f"\nstopped after {time.time() - t_start:.1f}s, {n} detection(s)",
             file=sys.stderr)
    # Anything else (arecord dying, device unplugged) propagates as a traceback.


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ckpt", default=DEFAULT_CKPT,
                    help="ONNX export directory (default: ./model)")
    ap.add_argument("--audio", default=None,
                    help="an audio file, or a directory to scan recursively "
                         "(mutually exclusive with --live)")
    ap.add_argument("--live", action="store_true",
                    help="continuously score a live ALSA input (e.g. an "
                         "AudioMoth USB microphone) instead of a file")
    ap.add_argument("--alsa-device", default=None,
                    help="--live only: ALSA device string, e.g. plughw:3,0 "
                         "(default: auto-detect a capture device whose name "
                         "contains \"AudioMoth\")")
    ap.add_argument("--duration", type=float, default=None,
                    help="--live only: stop after this many seconds of "
                         "captured audio (default: run until Ctrl+C)")
    ap.add_argument("--top-k", type=int, default=5)
    ap.add_argument("--threshold", type=float, default=0.3)
    ap.add_argument("--temperature", type=float, default=None,
                    help="default: the checkpoint's own trained temperature")
    ap.add_argument("--device", choices=["cuda", "cpu"], default=None)
    ap.add_argument("--format", choices=["csv", "json", "table"], default="table")
    ap.add_argument("--out", default=None,
                    help="write to this file instead of stdout")
    args = ap.parse_args()

    if args.live and args.audio:
        ap.error("--live and --audio are mutually exclusive")
    if not args.live and not args.audio:
        ap.error("either --audio or --live is required")

    t_load = time.time()
    predictor = EdgePredictor(args.ckpt, device=args.device)
    print(f"loaded {len(predictor.class_names):,} classes on {predictor.device} "
         f"in {time.time() - t_load:.1f}s", file=sys.stderr)

    out = open(args.out, "w", newline="") if args.out else sys.stdout
    try:
        if args.live:
            _run_live(predictor, args, out)
        else:
            _run_files(predictor, args, out)
    finally:
        if args.out:
            out.close()


if __name__ == "__main__":
    main()

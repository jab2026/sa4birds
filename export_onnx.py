#!/usr/bin/env python
"""
Export a sa4birds DSA or SSA checkpoint to ONNX for apps/web and
apps/edge, neither of which needs torch to serve the result.

A DSA export has four outputs (local and global attention logits plus their
class maps); SSA has no global branch, so it exports only the local pair and
records fuse_weight 1.0. metadata.json names the architecture either way.

Both accept forward(..., return_attention=True), which is what the cross-check
below compares against.

    python export_onnx.py --ckpt ckpts/LT/ssa_LT_seed1/models/model.pth \
                          --out ckpts/onnx/ssa_LT_seed1 --ir-version 9
    ./apps/web/run_app.sh --ckpt ../../ckpts/onnx/ssa_LT_seed1   # relative to apps/web

It lives at the project root because it needs torch + timm + omegaconf to load a
.pth, which the two serving tools deliberately do not.

Requires the `onnx` package (in requirements.txt). The default opset 17 is the
highest the pinned torch 2.2 can export.

Writes model.onnx (newer torch exporters also write the weights to a separate
model.onnx.data) and metadata.json -- class names, spectrogram parameters, and
the checkpoint's trained temperature and fuse weight. That metadata is what makes
the graph usable without the original OmegaConf config.

Exports the attention head's raw pre-temperature, pre-softmax logits and class
feature maps instead of baking the temperature into the graph, so the serving
code can recompute predictions at any temperature. See ExportWrapper below; the
reformulation is cross-checked against the eager model at export time, which
--no-verify skips.
"""
import argparse
import json
import os

import torch
import torch.nn as nn
import torch.nn.functional as F

from sa4birds.models.dsa import DSA
from sa4birds.models.ssa import SSA

ARCHITECTURES = {"DSA": DSA, "SSA": SSA}


class ExportWrapper(nn.Module):
    """Everything the model's forward does up to (but not including) the
    temperature division + softmax + weighted-sum-over-space that
    MultiHeadSABlock.forward does internally -- so temperature stays adjustable
    after export instead of being fixed at whatever the checkpoint trained with.

    Both DSA and SSA hardcode heads=1, so the head_weights-weighted average
    MultiHeadSABlock applies after the weighted sum is a no-op with a single
    head, and is skipped; the numpy reconstruction does not need head_weights.
    """
    def __init__(self, model, dual):
        super().__init__()
        self.model = model
        self.dual = dual        # DSA fuses a global branch; SSA has only the local one

    def forward(self, spec):
        m = self.model

        def normalize_and_project(feat, proj_layer):
            feat = m.norm(feat.transpose(1, -1)).transpose(1, -1)
            return F.relu(proj_layer(feat))

        x = m.backbone(spec)
        l_x = normalize_and_project(x, m.l_proj)
        # SSA always adds the residual; DSA drops it when proj_dim narrows the
        # channel count, which is what use_residual records.
        if getattr(m, "use_residual", True):
            l_x = x + l_x

        l_att_logits = m.attention.att(l_x)
        l_cla = m.attention.nonlinearity(m.attention.cla(l_x))
        if not self.dual:
            return l_att_logits, l_cla

        g_x = normalize_and_project(m.avg_pool(x), m.g_proj)
        g_att_logits = m.attention.att(g_x)
        g_cla = m.attention.nonlinearity(m.attention.cla(g_x))
        return l_att_logits, l_cla, g_att_logits, g_cla


@torch.no_grad()
def _numpy_reconstruction_matches(wrapper, model, dummy, fuse_w, temperature, dual,
                                  atol=1e-4):
    """Cross-check: rebuild probs from the exported outputs the way apps/web and
    apps/edge do, in plain numpy, and compare against the eager torch model's own
    forward (which computes them differently -- inside MultiHeadSABlock, in
    torch). Disagreement here means the two really would predict differently,
    not just a numerical wrinkle -- see the module docstring's --no-verify."""
    import numpy as np

    outs = [o.numpy() for o in wrapper(dummy)]

    def branch(logits, cla):
        att = logits / temperature
        flat = att.reshape(*att.shape[:2], -1)
        flat = flat - flat.max(axis=-1, keepdims=True)
        e = np.exp(flat)
        norm_att = (e / e.sum(axis=-1, keepdims=True)).reshape(att.shape)
        return (norm_att * cla).sum(axis=(-2, -1))

    recon = branch(outs[0], outs[1])
    if dual:
        recon = recon * fuse_w + branch(outs[2], outs[3]) * (1 - fuse_w)
    torch_probs = model(dummy, center_5s=False, return_attention=True)[0]
    diff = float(np.abs(recon - torch_probs.numpy()).max())
    return diff <= atol, diff


def _source_name(ckpt_path):
    """Identify the checkpoint without recording the exporting machine's paths.

    metadata.json ships inside every deployed export, so an absolute path here
    would travel with it. Released checkpoints are <RUN>/models/model.pth with
    RUN named <head>_<task>_<seed> (dsa_HSN_seed1, ssa_LT_seed1), so the run
    directory alone identifies one.
    """
    parts = os.path.normpath(os.path.abspath(ckpt_path)).split(os.sep)
    if len(parts) >= 3 and parts[-2] == "models":
        return parts[-3]
    return os.path.basename(ckpt_path)


def _strip_debug_info(graph):
    """Drop per-node doc strings and metadata, recursing into subgraphs."""
    graph.doc_string = ""
    for node in graph.node:
        node.doc_string = ""
        del node.metadata_props[:]
        for attr in node.attribute:
            if attr.HasField("g"):
                _strip_debug_info(attr.g)
            for sub in attr.graphs:
                _strip_debug_info(sub)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ckpt", required=True, help="sa4birds checkpoint (.pth)")
    ap.add_argument("--out", required=True,
                    help="output directory; writes model.onnx and metadata.json here")
    ap.add_argument("--opset", type=int, default=17,
                    help="ONNX opset (17 is the highest the pinned torch 2.2 exports)")
    ap.add_argument("--ir-version", type=int, default=None,
                    help="downgrade the file's ONNX IR version after export (a format "
                         "marker, not the opset; predictions are unchanged). Use 9 for "
                         "older onnxruntime builds such as 1.16 on a Raspberry Pi -- "
                         "see apps/edge/README.md")
    ap.add_argument("--source-name", default=None,
                    help="name stored as metadata.json's source_checkpoint; defaults "
                         "to the checkpoint's run directory, e.g. ssa_LT_seed1. metadata.json "
                         "ships with the export, so set this when those directories "
                         "carry a name you do not want to publish")
    ap.add_argument("--no-verify", action="store_true",
                    help="skip the numpy-reconstruction-vs-eager-torch cross-check")
    args = ap.parse_args()

    obj = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    state_dict, cfg = obj["state_dict"], obj["config"]
    arch = str(cfg.network.classifier)
    if arch not in ARCHITECTURES:
        raise ValueError(f"export_onnx.py supports classifier "
                         f"{' or '.join(ARCHITECTURES)}, got {arch!r}")
    dual = arch == "DSA"

    class_names = [n for n, _ in sorted(cfg.train.label_map.items(), key=lambda kv: kv[1])]

    model = ARCHITECTURES[arch](cfg)
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing or unexpected:
        raise RuntimeError(f"checkpoint did not load cleanly: {len(missing)} missing, "
                           f"{len(unexpected)} unexpected keys")
    model.eval()

    wrapper = ExportWrapper(model, dual)
    wrapper.eval()
    fe = cfg.frontend
    dummy = torch.randn(1, fe.in_chans, fe.n_mels, fe.val_target_length)
    # SSA has no second branch to fuse, so its local scores are the whole answer.
    fuse_w = float(torch.clamp(model.fuse_weight, 0, 1)) if dual else 1.0
    temperature = float(cfg.network.temperature)

    out_names = ["local_att_logits", "local_cla"]
    if dual:
        out_names += ["global_att_logits", "global_cla"]

    if not args.no_verify:
        ok, diff = _numpy_reconstruction_matches(wrapper, model, dummy, fuse_w,
                                                temperature, dual)
        print(f"numpy-reconstruction cross-check: max abs diff {diff:.2e} "
             f"({'OK' if ok else 'FAILED -- see module docstring'})")
        if not ok:
            raise RuntimeError(
                "numpy reconstruction disagrees with the eager torch model by more "
                f"than 1e-4 (got {diff:.4g}); the exported graph would not "
                "reproduce this checkpoint's predictions. Re-run with --no-verify "
                "only if you understand why (see export_onnx.py's docstring).")

    os.makedirs(args.out, exist_ok=True)
    onnx_path = os.path.join(args.out, "model.onnx")
    with torch.no_grad():
        torch.onnx.export(
            wrapper, (dummy,), onnx_path,
            input_names=["spectrogram"],
            output_names=out_names,
            dynamic_axes={n: {0: "batch"} for n in ["spectrogram"] + out_names},
            opset_version=args.opset,
        )

    # Newer torch exporters attach each node's Python stack trace as metadata,
    # which records this machine's absolute paths ~1,400 times and would ship
    # inside every deployed model.onnx. It is debug-only; inference ignores it.
    import onnx
    m = onnx.load(onnx_path, load_external_data=False)
    _strip_debug_info(m.graph)
    m.doc_string = ""
    if args.ir_version is not None:
        m.ir_version = args.ir_version
    onnx.save(m, onnx_path)

    metadata = {
        "class_names": class_names,
        "frontend": {
            "sample_rate": int(fe.sample_rate), "n_fft": int(fe.n_fft),
            "n_stft": int(fe.n_stft), "hop_length": int(fe.hop_length),
            "power": float(fe.power), "n_mels": int(fe.n_mels),
            "mean": float(fe.mean), "std": float(fe.std),
            "val_target_length": int(fe.val_target_length),
        },
        "default_temperature": temperature,
        "classifier": arch,
        "fuse_weight": fuse_w,
        "source_checkpoint": args.source_name or _source_name(args.ckpt),
    }
    with open(os.path.join(args.out, "metadata.json"), "w") as f:
        json.dump(metadata, f, indent=2)

    onnx_size = os.path.getsize(onnx_path)
    data_path = onnx_path + ".data"
    data_size = os.path.getsize(data_path) if os.path.exists(data_path) else 0
    print(f"\nwrote {args.out}/")
    print(f"  model.onnx       {onnx_size / 1e6:.1f} MB")
    if data_size:
        print(f"  model.onnx.data  {data_size / 1e6:.1f} MB")
    print(f"  metadata.json    {len(class_names):,} classes, "
         f"temperature={temperature:g}, fuse_weight={fuse_w:.3f}")
    print(f"\nServe it with:\n  apps/web/run_app.sh --ckpt {os.path.abspath(args.out)}")


if __name__ == "__main__":
    main()

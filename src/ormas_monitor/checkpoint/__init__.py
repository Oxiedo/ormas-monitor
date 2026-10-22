"""Health check for a saved model: no training, no data, weights only.

    ormas-monitor check model.pt                     # one checkpoint
    ormas-monitor check ./my-hf-model                # a Hugging Face folder (sharded or not)
    ormas-monitor check after.pt --against before.pt # what changed between two

Reads .pt/.pth/.bin/.ckpt (PyTorch), .safetensors and Hugging Face model folders, one tensor at
a time. PyTorch files are loaded with weights_only=True, so a checkpoint cannot run code on your
machine unless you pass trust=True.

`run()` returns the documented result dict (brief section 4.5): `file, tensors, parameters,
batchnorm_layers, findings, layers`, plus `against`/`diff` with a second checkpoint. Added keys:
`format, files, wrapper, streamed, dtypes, model_type, wiring, notes, norm_layers,
severity_counts`; `layers[*]` gains `feeds_relu, wiring`; `diff` gains `dtype_changed` and
`embedding_rows_unchanged`; `wiring` gains `model_batchnorms`.

With `arch`, an architecture that holds under half of the file's BatchNorms (or the file under
half of its) raises CheckpointError; a partial match above that adds an `arch_partial_match`
warning finding.
"""
from __future__ import annotations

from collections import OrderedDict
from pathlib import Path
from typing import Mapping

import torch

from .analysis import (CheckConfig, _batchnorm_pass, _embedding_rows_unchanged, _Reader, analyse,
                       batchnorm_names, check, compare, diff_findings, is_embedding, sort_findings)
from .sources import CheckpointError, TensorSource, open_source
from .wiring import Wiring, arch_relu_followers, resolve

__all__ = ["run", "check", "diff", "load_tensors", "open_source", "arch_relu_followers",
           "CheckpointError", "CheckConfig", "TensorSource", "Wiring"]


def load_tensors(path: str | Path, trust: bool = False) -> "OrderedDict[str, torch.Tensor]":
    """Every tensor in a checkpoint file or folder, in memory. Prefer `run`, which streams."""
    with open_source(path, trust) as src:
        return OrderedDict(src.items())


def _diff_pass(new: _Reader, old: _Reader) -> tuple[dict, dict]:
    """One streamed pass over both checkpoints: per-tensor stats of `new` and the diff dict."""
    cfg, ns, os_ = new.cfg, new.src, old.src
    stats, rows, frozen, added, reshaped, dtype_changed, emb = {}, [], [], [], [], [], []
    for name, info in ns.info.items():
        t = ns.get(name)
        if info.floating:
            stats[name] = new.measure(name, t)
        if name not in os_:
            added.append(name)
            continue
        oinfo = os_.info[name]
        if oinfo.shape != info.shape:
            reshaped.append(name)
            continue
        if not (info.floating and oinfo.floating):
            continue
        o = os_.get(name)
        if oinfo.dtype != info.dtype:
            dtype_changed.append({"tensor": name, "old": oinfo.dtype, "new": info.dtype})
        if len(info.shape) >= 2:
            old.remember_rows(name, o)    # for the newly-dead check, without a second read
        row = compare(name, t, o, cfg)
        rows.append(row)
        if row["abs_change"] == 0:
            frozen.append(name)
        elif is_embedding(name, ns):
            same = _embedding_rows_unchanged(t, o, cfg)
            if same.numel():
                emb.append({"tensor": name, "count": int(same.numel()), "size": info.shape[0],
                            "rows": same[:cfg.max_listed].tolist()})
    rows.sort(key=lambda r: -r["change"])
    removed = [k for k in os_.info if k not in ns]
    return stats, {"changed": rows, "unchanged": frozen, "added": added, "removed": removed,
                   "reshaped": reshaped, "dtype_changed": dtype_changed, "embedding_rows_unchanged": emb}


def diff(new: "TensorSource | Mapping[str, torch.Tensor]",
         old: "TensorSource | Mapping[str, torch.Tensor]", config: CheckConfig | None = None) -> dict:
    """What changed between two sets of weights (streamed when given TensorSources)."""
    from .sources import as_source
    cfg = config or CheckConfig()
    return _diff_pass(_Reader(as_source(new), cfg), _Reader(as_source(old), cfg))[1]


def run(path: str | Path, against: str | Path | None = None, trust: bool = False,
        arch: str | None = None, config: CheckConfig | None = None) -> dict:
    """Check the checkpoint at `path` (file or HF folder); with `against`, also diff the two."""
    cfg = config or CheckConfig()
    with open_source(path, trust) as src:
        bns = batchnorm_names(src)
        wiring = resolve(arch, src, bns, cfg.arch_min_match)
        follow = set(wiring.relu_after) if wiring.relu_after is not None else None
        known = set(wiring.known) if wiring.known is not None else None
        reader = _Reader(src, cfg)
        if against is None:
            stats = {k: reader.measure(k, t) for k, t in src.items() if src.info[k].floating}
            result = {"file": str(path), **analyse(reader, stats, follow, known)}
        else:
            with open_source(against, trust) as osrc:
                old = _Reader(osrc, cfg)
                stats, d = _diff_pass(reader, old)
                result = {"file": str(path), **analyse(reader, stats, follow, known)}
                result["against"] = str(against)
                old_known = known & set(batchnorm_names(osrc)) if known is not None else None
                before = {(l["layer"], l["batchnorm"]): set(l["certified"])
                          for l in _batchnorm_pass(old, follow, old_known, cfg)[0]}
                newly = []
                for l in result["layers"]:
                    gained = set(l["certified"]) - before.get((l["layer"], l["batchnorm"]), set())
                    if gained:
                        newly.append({"layer": l["layer"], "batchnorm": l["batchnorm"], "count": len(gained)})
                d["newly_dead"] = newly
                result["diff"] = d
                if newly:
                    result["findings"].append({
                        "severity": "warning", "kind": "newly_dead",
                        "title": f"{sum(n['count'] for n in newly)} channels died between the two checkpoints",
                        "detail": ", ".join(f"{n['layer']}: {n['count']}" for n in newly)})
                result["findings"] += diff_findings(d, Path(str(against)).name)
                sort_findings(result["findings"])
        if wiring.warning:
            result["findings"].append({
                "severity": "warning", "kind": "arch_partial_match", "title": wiring.warning,
                "detail": f"{wiring.note}. Findings for the matched layers use the traced wiring; check "
                          "that --arch names the architecture this file was saved from."})
            sort_findings(result["findings"])
        model_types = [src.config.get("model_type")] if src.config else []
        result.update({
            "format": src.format, "files": list(src.files), "wrapper": src.wrapper, "streamed": src.streamed,
            "model_type": model_types[0] if model_types else None,
            "wiring": wiring.as_dict(bns)})
    if wiring.note:
        result["notes"].insert(0, wiring.note)
    elif wiring.relu_after is None and bns:
        result["notes"].insert(0, "Wiring not known, so BatchNorm findings say \"if a ReLU directly follows\". Pass "
                                  "--arch torchvision:<name> or --arch hf for exact wiring.")
    if not result["streamed"]:
        result["notes"].append("This file uses PyTorch's legacy (pre-1.6) format, which cannot be memory-mapped; "
                               "it was read into memory whole.")
    counts = {s: 0 for s in ("critical", "warning", "info")}
    for f in result["findings"]:
        counts[f["severity"]] += 1
    result["severity_counts"] = counts
    return result

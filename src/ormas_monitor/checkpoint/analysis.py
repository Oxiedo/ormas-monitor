"""Weights-only findings for one checkpoint, and the diff between two.

Every tensor is read once, in chunks converted to float32, so fp16/bf16 checkpoints are
measured at full precision and memory stays near the size of the largest tensor. What can be
read from weights alone, and how sure each finding is:

  * NaN/Inf values (critical) and all-zero weight tensors (warning): facts about the file.
  * BatchNorm channels whose output is a constant <= 0 (warning): dead for good when a ReLU
    follows directly (Theorem 2); the wording says "if" when the wiring is not known.
  * LayerNorm/RMSNorm/GroupNorm scale entries ~0 (warning): that feature is switched off now.
    This is not a permanence proof: gradient can still reach the scale.
  * With a second checkpoint: tensors and embedding rows that did not change at all (info).

Nothing here claims dead units behind GELU/SiLU: weights cannot prove that.
"""
from __future__ import annotations

import math
import re
from collections import OrderedDict
from dataclasses import dataclass
from typing import Mapping

import torch

from ..structure import scan_channels
from .sources import TensorSource, as_source

BN_SUFFIXES = ("weight", "bias", "running_mean", "running_var")
NORM_SCALE_SUFFIXES = ("weight", "scale", "gamma")
# Model families whose norms multiply by (1 + weight): a stored 0 there is a healthy scale of 1.
ZERO_CENTRED_FAMILIES = ("gemma", "nemotron", "qwen3_next")
SEVERITY_RANK = {"critical": 0, "warning": 1, "info": 2}
KIND_RANK = ["nonfinite", "missing_tensors", "arch_partial_match", "identical_weights", "newly_dead",
             "dead_channels", "decayed_channels", "zero_tensor", "norm_scale_zero"]
LAYER_KINDS = {2: "Linear", 3: "Conv1d", 4: "Conv2d", 5: "Conv3d"}    # weight rank -> module kind


@dataclass(frozen=True)
class CheckConfig:
    """Thresholds for the weights-only check."""
    rel_tol: float = 1e-5            # "~0" means below rel_tol x the tensor's median |value|
    chunk_elems: int = 1 << 22       # elements converted to float32 at a time (16 MB)
    max_listed: int = 64             # longest list of rows/entries kept per tensor in new findings
    collapsed_var: float = 1e-12     # BatchNorm running variance treated as "never varies"
    arch_min_match: float = 0.5      # --arch must find this share of BatchNorms on both sides


@dataclass(frozen=True)
class TensorStats:
    """Counts over one floating-point tensor."""
    numel: int
    nan: int
    posinf: int
    neginf: int
    nonzero: int

    @property
    def nonfinite(self) -> int:
        return self.nan + self.posinf + self.neginf


def _rows(t: torch.Tensor, cfg: CheckConfig) -> list[torch.Tensor]:
    """Split a non-empty `t`, viewed as 2-D (rows = output units), into blocks of ~chunk_elems.

    A single row longer than chunk_elems is split into 1-row pieces of the flattened tensor.
    """
    flat = t.reshape(t.shape[0], -1) if t.dim() >= 2 else t.reshape(1, -1)
    if flat.shape[1] > cfg.chunk_elems:
        return [c.reshape(1, -1) for c in t.reshape(-1).split(cfg.chunk_elems)]
    return list(flat.split(max(1, cfg.chunk_elems // max(1, flat.shape[1])), dim=0))


class _Reader:
    """Measures tensors of one source and caches per-row max |w| for 2-D+ weights."""

    def __init__(self, src: TensorSource, cfg: CheckConfig) -> None:
        self.src, self.cfg = src, cfg
        self.rowmax: dict[str, torch.Tensor] = {}

    def measure(self, name: str, t: torch.Tensor) -> TensorStats:
        info = self.src.info[name]
        acc = torch.zeros(4, dtype=torch.int64)
        maxes = []
        wide = t.dim() >= 2 and t.shape[0] > 0 and t[0].numel() <= self.cfg.chunk_elems
        for c in _rows(t, self.cfg) if t.numel() else []:
            c = c.float()
            acc += torch.stack([torch.isnan(c).sum(), torch.isposinf(c).sum(), torch.isneginf(c).sum(),
                                torch.count_nonzero(c)])
            if wide:
                maxes.append(c.abs().amax(dim=1))
        if maxes:
            self.rowmax[name] = torch.cat(maxes)
        nan, pos, neg, nz = acc.tolist()
        return TensorStats(info.numel, nan, pos, neg, nz)

    def remember_rows(self, name: str, t: torch.Tensor) -> None:
        """Cache max |w| per output unit of `t` without the other statistics."""
        if name not in self.rowmax and t.dim() >= 2 and t.numel() and t[0].numel() <= self.cfg.chunk_elems:
            self.rowmax[name] = torch.cat([c.float().abs().amax(dim=1) for c in _rows(t, self.cfg)])

    def row_max(self, name: str) -> torch.Tensor:
        """max |w| per output unit, as an [n, 1] tensor (what scan_channels reads of a filter)."""
        if name not in self.rowmax:
            t = self.src.get(name)
            self.rowmax[name] = torch.stack([r.float().abs().amax() for r in t.reshape(t.shape[0], -1)]) \
                if t[0].numel() > self.cfg.chunk_elems else \
                torch.cat([c.float().abs().amax(dim=1) for c in _rows(t, self.cfg)])
        return self.rowmax[name].unsqueeze(1)


def _groups(src: TensorSource) -> "OrderedDict[str, dict[str, str]]":
    """prefix -> {suffix: full tensor name}, in file order."""
    groups: "OrderedDict[str, dict[str, str]]" = OrderedDict()
    for name in src.info:
        prefix, _, suffix = name.rpartition(".")
        groups.setdefault(prefix, {})[suffix] = name
    return groups


def _layer_kind(src: TensorSource, prefix: str) -> str | None:
    """Conv2d, Linear, ... inferred from the rank of `prefix`.weight; None when unknown."""
    info = src.info.get(f"{prefix}.weight")
    return LAYER_KINDS.get(len(info.shape)) if info is not None else None


def _where(src: TensorSource, prod: str | None, bn: str) -> str:
    """'layer `3` (Conv2d, BatchNorm `4`)', or 'BatchNorm `4`' when no producer was found."""
    if prod is None:
        return f"BatchNorm `{bn}`"
    kind = _layer_kind(src, prod)
    return f"layer `{prod}` ({kind + ', ' if kind else ''}BatchNorm `{bn}`)"


def batchnorm_names(src: TensorSource) -> list[str]:
    """BatchNorm layers in the file: groups with weight, bias, running_mean and running_var."""
    return [p for p, g in _groups(src).items() if all(s in g for s in BN_SUFFIXES)]


def _producer_for(bn: str, size: int, groups: "OrderedDict[str, dict[str, str]]", src: TensorSource) -> str | None:
    """Find the Conv/Linear feeding BatchNorm `bn`: by name first, then by file order.

    Safetensors stores names sorted, so for it the file-order fallback is replaced by "the only
    sibling with a matching weight".
    """
    def ok(p: str) -> bool:
        g = groups.get(p, {})
        w = src.info.get(g.get("weight", ""))
        return (p != bn and w is not None and w.floating and len(w.shape) >= 2 and w.shape[0] == size
                and "running_mean" not in g)
    parent, _, child = bn.rpartition(".")
    join = (parent + ".") if parent else ""
    candidates = []
    m = re.fullmatch(r"(.*?)(bn|norm|batchnorm|batch_norm)(\d*)", child)
    if m:
        for stem in ("conv", "fc", "linear", "proj"):
            candidates.append(join + m.group(1) + stem + m.group(3))
    if child.isdigit():
        candidates.append(join + str(int(child) - 1))
    for c in candidates:
        if ok(c):
            return c
    if not src.ordered:
        siblings = [p for p in groups if p.rpartition(".")[0] == parent and ok(p)]
        return siblings[0] if len(siblings) == 1 else None
    order = list(groups)
    i = order.index(bn)
    for p in reversed(order[:i]):
        if ok(p):
            return p
        if "running_mean" in groups[p]:
            break
    return None


def _is_norm_scale(prefix: str, suffix: str, group: dict[str, str], src: TensorSource) -> bool:
    info = src.info[group[suffix]]
    leaf = prefix.rpartition(".")[2].lower()
    return (suffix in NORM_SCALE_SUFFIXES and info.floating and len(info.shape) == 1 and info.numel > 1
            and "running_mean" not in group
            and ("norm" in leaf or re.match(r"ln(_|\d|f|$)", leaf) is not None))


def is_embedding(name: str, src: TensorSource) -> bool:
    """A 2-D `.weight` whose module name says it is a token or position embedding table."""
    prefix, _, suffix = name.rpartition(".")
    leaf = prefix.rpartition(".")[2].lower()
    info = src.info[name]
    return (suffix == "weight" and len(info.shape) == 2 and info.floating
            and ("embed" in leaf or leaf in ("wte", "wpe", "shared")))


def _model_types(cfg: object) -> list[str]:
    found: list[str] = []
    if isinstance(cfg, dict):
        for k, v in cfg.items():
            if k == "model_type" and isinstance(v, str):
                found.append(v)
            else:
                found += _model_types(v)
    return found


def zero_centred(src: TensorSource) -> str | None:
    """The model_type from config.json if that family stores norm scales as 1 + weight."""
    for mt in _model_types(src.config):
        if any(f in mt for f in ZERO_CENTRED_FAMILIES):
            return mt
    return None


def _tensor_findings(src: TensorSource, stats: dict[str, TensorStats], norm_names: set[str]) -> list[dict]:
    findings = []
    nonfinite = [k for k, s in stats.items() if s.nonfinite]
    if nonfinite:
        findings.append({"severity": "critical", "kind": "nonfinite",
                         "title": f"{len(nonfinite)} tensors contain NaN or Inf",
                         "detail": ", ".join(nonfinite[:12]), "tensors": nonfinite,
                         "values": {k: {"nan": stats[k].nan, "inf": stats[k].posinf + stats[k].neginf}
                                    for k in nonfinite}})
    zero = [k for k, s in stats.items() if s.numel > 1 and s.nonzero == 0 and k not in norm_names
            and not k.endswith(("running_mean", "bias"))]
    if zero:
        findings.append({"severity": "warning", "kind": "zero_tensor",
                         "title": f"{len(zero)} weight tensors are entirely zero",
                         "detail": ", ".join(zero[:12]), "tensors": zero})
    for p in src.problems:
        findings.append({"severity": "critical", "kind": p["kind"],
                         "title": f"{len(p['tensors'])} tensors are missing from the shards",
                         "detail": f"{p['detail']}: {', '.join(p['tensors'][:12])}", "tensors": p["tensors"]})
    return findings


def _norm_findings(src: TensorSource, norm_names: list[str], cfg: CheckConfig) -> list[dict]:
    hits = []
    for name in norm_names:
        a = src.get(name).float().abs()
        nz = a[a > 0]
        thresh = cfg.rel_tol * float(nz.median()) if nz.numel() else 0.0
        off = ((a == 0) | (a <= thresh)).nonzero().flatten()
        if off.numel():
            hits.append({"tensor": name, "count": int(off.numel()), "size": int(a.numel()),
                         "entries": off[:cfg.max_listed].tolist()})
    if not hits:
        return []
    total = sum(h["count"] for h in hits)
    where = hits[0]["tensor"] if len(hits) == 1 else f"{len(hits)} norm layers"
    return [{
        "severity": "warning", "kind": "norm_scale_zero",
        "title": f"{total} normalisation scale entries of {where} are ~0 (features switched off)",
        "detail": "A LayerNorm/RMSNorm/GroupNorm scale of ~0 multiplies its normalised feature by "
                  "~zero, so that feature leaves the norm as a constant (its bias, or 0) for every "
                  "input. Unlike a dead ReLU channel this is not proven permanent: gradient can "
                  "still reach the scale. Applies to norms that multiply by the stored weight; "
                  "families that store 1 + weight (e.g. Gemma) are recognised from config.json. "
                  + ", ".join(f"{h['tensor']} ({h['count']}/{h['size']})" for h in hits[:12]),
        "tensors": hits}]


def _batchnorm_pass(reader: _Reader, relu_after: set[str] | None, known: set[str] | None,
                    cfg: CheckConfig) -> tuple[list[dict], list[dict], set[str]]:
    src = reader.src
    groups = _groups(src)
    findings: list[dict] = []
    layers: list[dict] = []
    paired: set[str] = set()
    for bn, g in groups.items():
        if not all(s in g for s in BN_SUFFIXES):
            continue
        size = src.info[g["weight"]].shape[0]
        prod = _producer_for(bn, size, groups, src)
        traced = relu_after is not None and (known is None or bn in known)
        if traced:
            feeds = bn in relu_after
        else:   # wiring unknown: skip-connection branches never feed a ReLU directly
            feeds = not re.search(r"downsample|shortcut|skip|residual", bn)
        gamma, beta = src.get(g["weight"]).float(), src.get(g["bias"]).float()
        w = reader.row_max(groups[prod]["weight"]) if prod else None
        pb = groups[prod].get("bias") if prod else None
        scan = scan_channels(w, src.get(pb).float() if pb else None, gamma, beta, rel_tol=cfg.rel_tol)
        if prod:
            paired.add(prod)
        var = src.get(g["running_var"]).float()
        collapsed = int((var < cfg.collapsed_var).sum())
        # Dormant: the shift sits more than 3 scales below zero, so with roughly normal
        # normalised inputs the channel is positive on under ~0.13% of positions.
        exact = set(scan.certified) | set(scan.flat) | set(scan.decayed)
        dormant = [c for c in ((beta + 3 * gamma.abs()) <= 0).nonzero().flatten().tolist() if c not in exact]
        if not feeds:                      # output joins a skip connection first: no ReLU argument
            # No ReLU argument: dead-looking and decayed channels are reported as (near-)constant.
            scan.flat, scan.certified, scan.decayed, dormant = sorted(set(scan.flat) | set(scan.certified) | set(scan.decayed)), [], [], []
        layers.append({"layer": prod or "(none found)", "batchnorm": bn, "channels": size,
                       "certified": scan.certified, "flat": scan.flat, "decayed": scan.decayed,
                       "dormant": dormant, "collapsed_var": collapsed,
                       "feeds_relu": feeds, "wiring": "traced" if traced else "heuristic"})
        where = _where(src, prod, bn)
        cond = "" if traced else f" if a ReLU directly follows `{bn}`"
        if scan.certified:
            findings.append({
                "severity": "warning", "kind": "dead_channels", "layer": prod, "batchnorm": bn,
                "title": f"{len(scan.certified)} of {size} channels in {where} are dead for good{cond}",
                "detail": "Their incoming filter or BatchNorm scale is exactly zero and the BatchNorm "
                          "shift is <= 0, so the output is a constant <= 0. A ReLU then closes for "
                          "every input and no gradient can ever reach them again (Theorem 2). "
                          "Does not apply where the BatchNorm output is added to a skip connection "
                          "before the ReLU.",
                "channels": scan.certified})
        if scan.decayed:
            findings.append({
                "severity": "warning", "kind": "decayed_channels", "layer": prod, "batchnorm": bn,
                "title": f"{len(scan.decayed)} of {size} channels in {where} are effectively dead{cond}",
                "detail": "Their filter or BatchNorm scale has shrunk to under 1e-5 of the layer's typical "
                          "size (or the filter is zero behind a nonzero bias, which BatchNorm cancels only up "
                          "to rounding) and their shift is <= 0, so they output ~0 for every input. This is "
                          "what a dead channel looks like after weight decay has been shrinking it for a while.",
                "channels": scan.decayed})
        if scan.flat:
            findings.append({
                "severity": "info", "kind": "flat_channels", "layer": prod, "batchnorm": bn,
                "title": f"{len(scan.flat)} of {size} channels in {where} output a constant",
                "detail": "Their incoming filter or BatchNorm scale is exactly zero, so they output "
                          "the same value (up to rounding) for every input and carry no information.",
                "channels": scan.flat})
        if dormant:
            findings.append({
                "severity": "info", "kind": "dormant_channels", "layer": prod, "batchnorm": bn,
                "title": f"{len(dormant)} of {size} channels in {where} are probably dormant{cond}",
                "detail": "BatchNorm shift <= -3 x |scale|: if a ReLU follows, these fire on under "
                          "~0.13% of positions when normalised inputs are roughly normal. An estimate, "
                          "not a proof.", "channels": dormant})
        if collapsed:
            findings.append({
                "severity": "info", "kind": "collapsed_variance", "batchnorm": bn,
                "title": f"{collapsed} of {size} channels of BatchNorm `{bn}` never vary (running variance ~0)",
                "detail": "Whatever reaches these channels is the same for every input, so they carry no "
                          "information. Usually a dead filter upstream.",
                "channels": (var < cfg.collapsed_var).nonzero().flatten().tolist()})
    return layers, findings, paired


def _unit_findings(reader: _Reader, paired: set[str]) -> list[dict]:
    src, findings = reader.src, []
    for prefix, g in _groups(src).items():
        w = g.get("weight")
        info = src.info.get(w or "")
        if (prefix in paired or info is None or not info.floating or len(info.shape) < 2
                or "running_mean" in g or is_embedding(w, src)):
            continue
        b = g.get("bias")
        scan = scan_channels(reader.row_max(w), src.get(b).float() if b else None, None, None)
        dead = scan.certified + scan.flat
        if dead and len(dead) < info.shape[0]:
            findings.append({
                "severity": "info", "kind": "zero_filters", "layer": prefix,
                "title": f"{len(dead)} of {info.shape[0]} output units of layer `{prefix}`"
                         + (f" ({LAYER_KINDS[len(info.shape)]})" if len(info.shape) in LAYER_KINDS else "")
                         + " have all-zero weights",
                "detail": "Each outputs only its bias. Followed by a ReLU, those with bias <= 0 "
                          "never fire.", "channels": dead})
    return findings


def sort_findings(findings: list[dict]) -> None:
    """Most severe first; within a severity, the kinds that matter most first (stable)."""
    findings.sort(key=lambda f: (SEVERITY_RANK[f["severity"]],
                                 KIND_RANK.index(f["kind"]) if f["kind"] in KIND_RANK else len(KIND_RANK)))


def analyse(reader: _Reader, stats: dict[str, TensorStats], relu_after: set[str] | None = None,
            known: set[str] | None = None) -> dict:
    """Assemble the check result from per-tensor stats (already measured) and a BatchNorm pass."""
    src, cfg = reader.src, reader.cfg
    groups = _groups(src)
    centred = zero_centred(src)
    norm_names = [g[s] for p, g in groups.items() for s in NORM_SCALE_SUFFIXES
                  if s in g and _is_norm_scale(p, s, g, src)]
    notes = []
    if centred and norm_names:
        notes.append(f"Norm scales not checked: {centred} stores them as 1 + weight, so 0 is a healthy value.")
    findings = _tensor_findings(src, stats, set(norm_names))
    if not centred:
        findings += _norm_findings(src, norm_names, cfg)
    layers, bn_findings, paired = _batchnorm_pass(reader, relu_after, known, cfg)
    findings += bn_findings + _unit_findings(reader, paired)
    skipped = [k for k, i in src.info.items() if not i.floating and len(i.shape) >= 2]
    if skipped:
        notes.append(f"{len(skipped)} integer weight tensors (quantised?) were not analysed.")
    sort_findings(findings)
    dtypes: dict[str, int] = {}
    for i in src.info.values():
        dtypes[i.dtype] = dtypes.get(i.dtype, 0) + 1
    return {"tensors": len(src.info), "parameters": sum(i.numel for i in src.info.values() if i.floating),
            "batchnorm_layers": len(layers), "findings": findings, "layers": layers,
            "norm_layers": 0 if centred else len(norm_names), "dtypes": dtypes, "notes": notes}


def check(tensors: "TensorSource | Mapping[str, torch.Tensor]", relu_after: set[str] | None = None,
          *, known: set[str] | None = None, config: CheckConfig | None = None) -> dict:
    """Weights-only health check of a state dict or an open TensorSource.

    `relu_after`, if known (see structure.relu_followers), limits dead/dormant findings to
    BatchNorms that feed a ReLU directly. `known` is the set of BatchNorm names that wiring
    covers; BatchNorms outside it fall back to the name heuristic (default: all covered).
    """
    src = as_source(tensors)
    reader = _Reader(src, config or CheckConfig())
    stats = {k: reader.measure(k, t) for k, t in src.items() if src.info[k].floating}
    return analyse(reader, stats, relu_after, known)


def _embedding_rows_unchanged(new: torch.Tensor, old: torch.Tensor, cfg: CheckConfig) -> torch.Tensor:
    """Indices of rows that are bit-for-bit equal (compared in float32)."""
    same = [(a.float() == b.float()).all(dim=1) for a, b in zip(_rows(new, cfg), _rows(old, cfg))]
    return torch.cat(same).nonzero().flatten()


def compare(name: str, new: torch.Tensor, old: torch.Tensor, cfg: CheckConfig) -> dict:
    """||new - old|| and ||old||, accumulated over float32 chunks (no full-size copies)."""
    if new.numel() == 0:
        return {"tensor": name, "change": 0.0, "abs_change": 0.0}
    sq = []
    for a, b in zip(_rows(new, cfg), _rows(old, cfg)):
        a, b = a.float(), b.float()
        sq.append(torch.stack([torch.linalg.vector_norm(a - b, dtype=torch.float64),
                               torch.linalg.vector_norm(b, dtype=torch.float64)]) ** 2)
    dn, on = (math.sqrt(x) for x in torch.stack(sq).sum(0).tolist())
    return {"tensor": name, "change": dn / on if on > 0 else (math.inf if dn else 0.0), "abs_change": dn}


def diff_findings(d: dict, against_name: str) -> list[dict]:
    """Info/warning findings from a diff dict (frozen tensors, unmoved embedding rows)."""
    out = []
    n_cmp = len(d["changed"])
    if n_cmp and len(d["unchanged"]) == n_cmp and not d["added"] and not d["removed"]:
        out.append({"severity": "warning", "kind": "identical_weights",
                    "title": f"The weights are identical to {against_name}",
                    "detail": "Every tensor is bit-for-bit unchanged: no training step changed this model "
                              "(or the two paths point to the same weights)."})
    elif d["unchanged"]:
        out.append({"severity": "info", "kind": "frozen_tensors",
                    "title": f"{len(d['unchanged'])} of {n_cmp} tensors are exactly unchanged",
                    "detail": "Bit-for-bit identical to the earlier checkpoint. Expected for layers that were "
                              "frozen during fine-tuning (a frozen backbone, adapter-only training); if you "
                              "meant to train them, they received no updates. BatchNorm running statistics "
                              "also stay fixed in eval mode. " + ", ".join(d["unchanged"][:12]),
                    "tensors": d["unchanged"]})
    for e in d["embedding_rows_unchanged"]:
        out.append({"severity": "info", "kind": "embedding_rows_unchanged", "layer": e["tensor"],
                    "title": f"{e['count']} of {e['size']} rows of `{e['tensor']}` never moved",
                    "detail": "Each unchanged row is a token (or position) whose embedding received no update "
                              "between the checkpoints: typically tokens absent from the fine-tuning data. "
                              "Decoupled weight decay (AdamW) would move every row, so this also suggests it "
                              "was off for this tensor. Usually harmless.",
                    "rows": e["rows"]})
    return out

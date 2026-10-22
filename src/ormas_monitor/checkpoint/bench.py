"""Measure time and peak memory of the checkpoint check (streamed vs. fully loaded).

    python -m ormas_monitor.checkpoint.bench                       # ResNet-50 + synthetic 2 GB folder
    python -m ormas_monitor.checkpoint.bench --gb 0.5 --resnet ""  # smaller, skip ResNet-50

Each measurement runs in a fresh subprocess. "streamed" is `checkpoint.run`; "materialised"
reproduces the v0.1 loader (every tensor in memory, then check) for comparison. Two memory
numbers are reported:

  * peak RSS (``ru_maxrss``): includes file pages the OS mapped in while reading. Those pages
    are clean and the OS can drop them under pressure, so this overstates what the check needs;
  * peak footprint (macOS ``lifetime_max_phys_footprint``, what Activity Monitor shows as
    "Memory"): memory the process actually owns. Not available on other platforms (None).

The synthetic folder is a fp16, transformer-shaped, sharded safetensors model written to a
temporary directory and deleted afterwards. A second copy (hard links for unchanged shards, one
shard rewritten) gives a streamed `--against` diff.
"""
from __future__ import annotations

import argparse
import ctypes
import glob
import json
import os
import resource
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path

import torch


@dataclass(frozen=True)
class SyntheticModel:
    """Shape of the synthetic transformer: about 2 bytes x parameters on disk (fp16)."""
    vocab: int = 32000
    d: int = 2048
    ffn: int = 5632
    layers: int = 16
    shards: int = 4


def _peak_footprint() -> int | None:
    """Lifetime peak physical footprint in bytes on macOS, else None."""
    if sys.platform != "darwin":
        return None
    names = ("user_time", "system_time", "pkg_idle_wkups", "interrupt_wkups", "pageins", "wired_size",
             "resident_size", "phys_footprint", "proc_start_abstime", "proc_exit_abstime",
             "child_user_time", "child_system_time", "child_pkg_idle_wkups", "child_interrupt_wkups",
             "child_pageins", "child_elapsed_abstime", "diskio_bytesread", "diskio_byteswritten",
             "cpu_time_qos_default", "cpu_time_qos_maintenance", "cpu_time_qos_background",
             "cpu_time_qos_utility", "cpu_time_qos_legacy", "cpu_time_qos_user_initiated",
             "cpu_time_qos_user_interactive", "billed_system_time", "serviced_system_time",
             "logical_writes", "lifetime_max_phys_footprint", "instructions", "cycles", "billed_energy",
             "serviced_energy", "interval_max_phys_footprint", "runnable_time")

    class RusageInfoV4(ctypes.Structure):
        _fields_ = [("uuid", ctypes.c_uint8 * 16)] + [(n, ctypes.c_uint64) for n in names]
    info = RusageInfoV4()
    lib = ctypes.CDLL("/usr/lib/libproc.dylib")
    if lib.proc_pid_rusage(os.getpid(), 4, ctypes.byref(info)) != 0:     # RUSAGE_INFO_V4
        return None
    return int(info.lifetime_max_phys_footprint)


def _child(mode: str, path: str, against: str | None) -> None:
    """Run one measurement in this (fresh) process and print a JSON line."""
    from . import check, run
    t0 = time.perf_counter()
    if mode == "streamed":
        res = run(path, against=against)
    else:
        res = check(_materialise(Path(path)))
    seconds = time.perf_counter() - t0
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    rss = rss if sys.platform == "darwin" else rss * 1024
    fp = _peak_footprint()
    print(json.dumps({"mode": mode, "seconds": round(seconds, 2), "peak_rss_mb": round(rss / 2**20),
                      "peak_footprint_mb": None if fp is None else round(fp / 2**20),
                      "parameters": res["parameters"], "tensors": res["tensors"],
                      "findings": {f["kind"]: f["severity"] for f in res["findings"]}}))


def _materialise(path: Path) -> dict:
    """The v0.1 path: torch.load without mmap, safetensors.load_file; every tensor in memory."""
    if path.is_dir():
        files = sorted(path.glob("*.safetensors"))
        from safetensors.torch import load_file
        out: dict = {}
        for f in files:
            out.update({k: v.clone() for k, v in load_file(str(f)).items()})
        return out
    return torch.load(path, map_location="cpu", weights_only=True)


def measure(mode: str, path: Path, against: Path | None = None) -> dict:
    code = (f"from ormas_monitor.checkpoint.bench import _child; "
            f"_child({mode!r}, {str(path)!r}, {None if against is None else str(against)!r})")
    out = subprocess.run([sys.executable, "-c", code], check=True, capture_output=True, text=True).stdout
    return json.loads(out.strip().splitlines()[-1])


def baseline() -> dict:
    """Peak memory of a process that only imports torch and the package."""
    code = ("import ormas_monitor.checkpoint, resource, sys, json; "
            "from ormas_monitor.checkpoint.bench import _peak_footprint as f; "
            "r = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss; "
            "r = r if sys.platform == 'darwin' else r * 1024; fp = f(); "
            "print(json.dumps({'peak_rss_mb': round(r / 2**20), "
            "'peak_footprint_mb': None if fp is None else round(fp / 2**20)}))")
    return json.loads(subprocess.run([sys.executable, "-c", code], check=True, capture_output=True,
                                     text=True).stdout)


def write_synthetic(folder: Path, spec: SyntheticModel, scale: float = 1.0) -> int:
    """Write a fp16 sharded safetensors model; returns total bytes. Two planted defects: NaN, norm zeros."""
    from safetensors.torch import save_file
    torch.manual_seed(0)
    d, f = spec.d, spec.ffn
    names: list[tuple[str, tuple[int, ...]]] = [("model.embed_tokens.weight", (spec.vocab, d))]
    for i in range(spec.layers):
        p = f"model.layers.{i}."
        names += [(p + "input_layernorm.weight", (d,)), (p + "post_attention_layernorm.weight", (d,))]
        names += [(p + f"self_attn.{n}_proj.weight", (d, d)) for n in "qkvo"]
        names += [(p + "mlp.gate_proj.weight", (f, d)), (p + "mlp.up_proj.weight", (f, d)),
                  (p + "mlp.down_proj.weight", (d, f))]
    names += [("model.norm.weight", (d,)), ("lm_head.weight", (spec.vocab, d))]
    folder.mkdir(parents=True)
    per = -(-len(names) // spec.shards)
    weight_map, total = {}, 0
    for s in range(spec.shards):
        shard = f"model-{s + 1:05d}-of-{spec.shards:05d}.safetensors"
        part = {}
        for name, shape in names[s * per:(s + 1) * per]:
            t = (torch.ones(shape) if name.endswith("norm.weight") else torch.randn(shape) * 0.02 * scale).half()
            if name == "model.layers.3.input_layernorm.weight":
                t[:4] = 0                               # planted: 4 features switched off
            if name == "model.layers.7.mlp.up_proj.weight":
                t[0, 0] = float("nan")                  # planted: one NaN
            part[name] = t
            total += t.numel() * 2
        save_file(part, str(folder / shard))
        weight_map.update({k: shard for k in part})
        del part
    (folder / "model.safetensors.index.json").write_text(json.dumps({"metadata": {"total_size": total},
                                                                    "weight_map": weight_map}))
    (folder / "config.json").write_text(json.dumps({"model_type": "llama"}))
    return total


def fine_tuned_copy(src: Path, dst: Path) -> None:
    """Hard-link every shard but the first; rewrite the first with 1,000 embedding rows moved."""
    from safetensors.torch import load_file, save_file
    dst.mkdir()
    for p in src.iterdir():
        if p.name != "model-00001-of-00004.safetensors":
            os.link(p, dst / p.name)
    part = load_file(str(src / "model-00001-of-00004.safetensors"))
    part["model.embed_tokens.weight"][:1000] += 0.01
    save_file(part, str(dst / "model-00001-of-00004.safetensors"))


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    cache = Path(torch.hub.get_dir()) / "checkpoints"
    default = next(iter(sorted(glob.glob(str(cache / "resnet50-*.pth")))), "")
    ap.add_argument("--resnet", default=default, help="ResNet-50 .pth to measure ('' to skip)")
    ap.add_argument("--gb", type=float, default=2.0, help="approximate size of the synthetic folder")
    ap.add_argument("--no-materialised", action="store_true", help="skip the load-everything comparison")
    a = ap.parse_args(argv)
    print(json.dumps({"baseline (import only)": baseline()}))
    if a.resnet:
        size = os.path.getsize(a.resnet) / 2**20
        for mode in ("streamed",) + (() if a.no_materialised else ("materialised",)):
            print(json.dumps({f"resnet50 {size:.0f} MB": measure(mode, Path(a.resnet))}))
    if a.gb > 0:
        base = SyntheticModel()
        layers = max(1, round(base.layers * a.gb / 2.0))
        spec = SyntheticModel(layers=layers, shards=4)
        with tempfile.TemporaryDirectory(prefix="ormas-bench-") as tmp:
            folder = Path(tmp) / "model"
            total = write_synthetic(folder, spec)
            label = f"synthetic fp16 {total / 2**30:.2f} GiB, {spec.shards} shards"
            print(json.dumps({label: measure("streamed", folder)}))
            if not a.no_materialised:
                print(json.dumps({label: measure("materialised", folder)}))
            fine_tuned_copy(folder, Path(tmp) / "tuned")
            print(json.dumps({f"{label}, --against": measure("streamed", Path(tmp) / "tuned", folder)}))
        print(f"deleted {tmp}")


if __name__ == "__main__":
    main()

"""Which BatchNorms feed a ReLU directly, traced from a known architecture.

    --arch torchvision:<name>   build the torchvision model (no weights) and trace it
    --arch hf                   build a Hugging Face model from the folder's config.json
                                (random init, no download) and trace it with a dummy image

Tracing runs one forward pass with `structure.relu_followers`, which leaves out layers whose
output is added to a skip connection before the ReLU. Model and file names are aligned by a
common prefix (e.g. ``resnet.`` or Lightning's ``model.``). Without `--arch` the check keeps a
conservative name heuristic and words its findings conditionally.
"""
from __future__ import annotations

import inspect
from dataclasses import dataclass
from pathlib import Path

import torch

from ..structure import BN_KINDS, relu_followers
from .sources import CheckpointError, TensorSource


@dataclass(frozen=True)
class Wiring:
    """Traced wiring, mapped to the file's tensor names. `relu_after` None means unknown."""
    source: str
    relu_after: frozenset | None = None
    known: frozenset | None = None
    note: str | None = None

    def as_dict(self, batchnorms: list[str]) -> dict:
        matched = len(set(batchnorms) & self.known) if self.known is not None else 0
        return {"source": self.source, "traced": self.relu_after is not None,
                "batchnorms": len(batchnorms), "matched": matched, "note": self.note}


HEURISTIC = Wiring("heuristic")


def _torchvision_model(name: str) -> tuple[torch.nn.Module, torch.Tensor]:
    try:
        import torchvision
    except ImportError as e:
        raise CheckpointError("--arch torchvision:<name> needs `pip install torchvision`") from e
    try:
        model = torchvision.models.get_model(name, weights=None)
    except ValueError as e:
        raise CheckpointError(f"torchvision has no model called {name!r}") from e
    size = 299 if name.startswith("inception") else 224
    return model, torch.randn(1, 3, size, size)


def arch_relu_followers(arch: str) -> set[str]:
    """Wiring for a known architecture, e.g. 'torchvision:resnet50' (needs torchvision)."""
    src, _, name = arch.partition(":")
    if src != "torchvision" or not name:
        raise CheckpointError("--arch takes torchvision:<model name> (e.g. torchvision:resnet50) or hf")
    model, example = _torchvision_model(name)
    return relu_followers(model, example)


def _hf_model(folder: Path) -> tuple[torch.nn.Module | None, torch.Tensor | None, str]:
    """Build the model in `folder`/config.json with random weights; (None, None, why) if impossible."""
    if not (folder / "config.json").is_file():
        raise CheckpointError(f"--arch hf needs a config.json next to the weights (looked in {folder})")
    try:
        import transformers
    except ImportError as e:
        raise CheckpointError("--arch hf needs `pip install transformers`") from e
    verbosity = transformers.logging.get_verbosity()
    transformers.logging.set_verbosity_error()
    try:
        try:
            cfg = transformers.AutoConfig.from_pretrained(str(folder), local_files_only=True)
        except (ValueError, KeyError, OSError) as e:
            return None, None, f"transformers cannot read this config.json ({e})"
        cls = next((getattr(transformers, a) for a in (cfg.architectures or [])
                    if hasattr(transformers, a)), None)
        if cls is None:
            try:
                cls = transformers.AutoModel._model_mapping[type(cfg)]
            except KeyError:
                return None, None, f"transformers has no model class for model_type {cfg.model_type!r}"
        if "pixel_values" not in inspect.signature(cls.forward).parameters:
            return None, None, (f"{cls.__name__} does not take an image input, so its ReLU wiring "
                                "cannot be traced with a dummy image")
        model = cls(cfg) if cfg.architectures else transformers.AutoModel.from_config(cfg)
        size = getattr(cfg, "image_size", None) or 224
        h, w = (size, size) if isinstance(size, int) else tuple(size)[-2:]
        return model, torch.randn(1, int(getattr(cfg, "num_channels", 3)), int(h), int(w)), cls.__name__
    finally:
        transformers.logging.set_verbosity(verbosity)


def _align(model_keys: list[str], file_keys: set[str]) -> tuple[str, str]:
    """(strip, add): file name = add + model name without `strip`, chosen to match the most keys."""
    candidates = {("", "")}
    for m in model_keys[:50]:
        for f in file_keys:
            if f != m and f.endswith("." + m):
                candidates.add(("", f[: -len(m)]))
            elif m != f and m.endswith("." + f):
                candidates.add((m[: -len(f)], ""))

    def score(c: tuple[str, str]) -> int:
        strip, add = c
        return sum(1 for m in model_keys if m.startswith(strip) and add + m[len(strip):] in file_keys)
    return max(sorted(candidates), key=score)


def _trace(model: torch.nn.Module, example: torch.Tensor, src: TensorSource, label: str) -> Wiring:
    model_bns = [n for n, m in model.named_modules() if type(m).__name__ in BN_KINDS]
    try:
        relu = relu_followers(model, example)
    except Exception as e:     # the model's own forward failed on a dummy input: report, don't guess
        return Wiring("heuristic", note=f"{label}: tracing failed ({type(e).__name__}: {e}); "
                                        "using the name-based heuristic")
    strip, add = _align(list(model.state_dict()), set(src.info))

    def to_file(n: str) -> str:
        return add + n[len(strip):] if n.startswith(strip) else n
    return Wiring(label, frozenset(to_file(n) for n in relu), frozenset(to_file(n) for n in model_bns))


def resolve(arch: str | None, src: TensorSource, batchnorms: list[str]) -> Wiring:
    """Wiring for `src` from `--arch`, or the heuristic when no arch is given or none applies."""
    if not arch:
        return HEURISTIC
    if arch != "hf" and not arch.startswith("torchvision:"):
        raise CheckpointError("--arch takes torchvision:<model name> (e.g. torchvision:resnet50) or hf")
    if not batchnorms:
        return Wiring("heuristic", note=f"--arch {arch}: the file has no BatchNorm layers, so wiring is not needed")
    if arch == "hf":
        first = Path(src.files[0])
        model, example, label = _hf_model(first.parent)
        if model is None:
            return Wiring("heuristic", note=f"--arch hf: {label}; using the name-based heuristic instead")
        wiring = _trace(model, example, src, f"hf:{label}")
    else:
        model, example = _torchvision_model(arch.partition(":")[2])
        wiring = _trace(model, example, src, arch)
    if wiring.known is None:
        return wiring
    matched = set(batchnorms) & wiring.known
    if not matched:
        raise CheckpointError(f"--arch {arch} does not match this file: none of its {len(batchnorms)} "
                              "BatchNorm layers exist in that model")
    if len(matched) < len(batchnorms):
        return Wiring(wiring.source, wiring.relu_after, wiring.known,
                      f"{len(batchnorms) - len(matched)} of {len(batchnorms)} BatchNorm layers are not in "
                      f"the --arch model; the name-based heuristic is used for those")
    return wiring

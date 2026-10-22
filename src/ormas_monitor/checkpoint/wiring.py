"""Which BatchNorms feed a ReLU directly, traced from a known architecture.

    --arch torchvision:<name>   build the torchvision model (no weights) and trace it
    --arch hf                   build a Hugging Face model from the folder's config.json
                                (random init, no download) and trace it with a dummy image

Tracing runs one forward pass with `structure.relu_followers`, which leaves out layers whose
output is added to a skip connection before the ReLU. The model is built and traced on PyTorch's
``meta`` device (shapes only: no weight initialisation, no arithmetic), which is what makes
``--arch`` cheap; a model whose forward needs real values falls back to the CPU. Model and file
names are aligned by a common prefix (e.g. ``resnet.`` or Lightning's ``model.``).

The architecture must match the file: if fewer than `min_match` (default half) of the file's
BatchNorms are in the model, or of the model's BatchNorms are in the file, `resolve` raises
`CheckpointError`; any smaller mismatch is reported as a warning. Without `--arch` the check
keeps a conservative name heuristic and words its findings conditionally.
"""
from __future__ import annotations

import inspect
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import torch

from ..structure import BN_KINDS, relu_followers
from .sources import CheckpointError, TensorSource

ARCH_USAGE = "--arch takes torchvision:<model name> (e.g. torchvision:resnet50) or hf"

Factory = Callable[[], torch.nn.Module]


@dataclass(frozen=True)
class Wiring:
    """Traced wiring, mapped to the file's tensor names. `relu_after` None means unknown.

    `model_batchnorms` is how many BatchNorms the traced model has (None when not traced);
    `warning` is set when the model matches the file only in part.
    """
    source: str
    relu_after: frozenset | None = None
    known: frozenset | None = None
    note: str | None = None
    model_batchnorms: int | None = None
    warning: str | None = None

    def as_dict(self, batchnorms: list[str]) -> dict:
        matched = len(set(batchnorms) & self.known) if self.known is not None else 0
        return {"source": self.source, "traced": self.relu_after is not None,
                "batchnorms": len(batchnorms), "matched": matched,
                "model_batchnorms": self.model_batchnorms, "note": self.note}


HEURISTIC = Wiring("heuristic")


def _torchvision_factory(name: str) -> tuple[Factory, tuple[int, ...]]:
    """A builder for torchvision model `name` (no weights) and the input shape to trace it with."""
    try:
        import torchvision
    except ImportError as e:
        raise CheckpointError("--arch torchvision:<name> needs `pip install torchvision`") from e
    if name not in torchvision.models.list_models():
        raise CheckpointError(f"torchvision has no model called {name!r}")

    def build() -> torch.nn.Module:
        with warnings.catch_warnings():    # inception/googlenet warn about their default init
            warnings.simplefilter("ignore", FutureWarning)
            return torchvision.models.get_model(name, weights=None)
    size = 299 if name.startswith("inception") else 224
    return build, (1, 3, size, size)


def _traced_on(build: Factory, shape: tuple[int, ...], device: str) -> tuple[torch.nn.Module, set[str]]:
    with torch.device(device):
        model = build()
    return model, relu_followers(model, torch.randn(shape, device=device))


def _traced(build: Factory, shape: tuple[int, ...]) -> tuple[torch.nn.Module, set[str]]:
    """Build and trace on the meta device; on the CPU if the meta trace fails (its error is final)."""
    try:
        return _traced_on(build, shape, "meta")
    except Exception:          # a forward that needs real values (.item(), data-dependent shapes)
        return _traced_on(build, shape, "cpu")


def arch_relu_followers(arch: str) -> set[str]:
    """Wiring for a known architecture, e.g. 'torchvision:resnet50' (needs torchvision)."""
    src, _, name = arch.partition(":")
    if src != "torchvision" or not name:
        raise CheckpointError(ARCH_USAGE)
    return _traced(*_torchvision_factory(name))[1]


def _hf_factory(folder: Path) -> tuple[Factory | None, tuple[int, ...] | None, str]:
    """A builder for the model in `folder`/config.json (random init); (None, None, why) if impossible."""
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
    finally:
        transformers.logging.set_verbosity(verbosity)

    def build() -> torch.nn.Module:
        transformers.logging.set_verbosity_error()
        try:
            return cls(cfg) if cfg.architectures else transformers.AutoModel.from_config(cfg)
        finally:
            transformers.logging.set_verbosity(verbosity)
    size = getattr(cfg, "image_size", None) or 224
    h, w = (size, size) if isinstance(size, int) else tuple(size)[-2:]
    return build, (1, int(getattr(cfg, "num_channels", 3)), int(h), int(w)), cls.__name__


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


def _trace(build: Factory, shape: tuple[int, ...], src: TensorSource, label: str) -> Wiring:
    try:
        model, relu = _traced(build, shape)
    except Exception as e:     # the model's own forward failed on a dummy input: report, don't guess
        return Wiring("heuristic", note=f"{label}: tracing failed ({type(e).__name__}: {e}); "
                                        "using the name-based heuristic")
    model_bns = [n for n, m in model.named_modules() if type(m).__name__ in BN_KINDS]
    strip, add = _align(list(model.state_dict()), set(src.info))

    def to_file(n: str) -> str:
        return add + n[len(strip):] if n.startswith(strip) else n
    return Wiring(label, frozenset(to_file(n) for n in relu), frozenset(to_file(n) for n in model_bns),
                  model_batchnorms=len(model_bns))


def resolve(arch: str | None, src: TensorSource, batchnorms: list[str], min_match: float = 0.5) -> Wiring:
    """Wiring for `src` from `--arch`, or the heuristic when no arch is given or none applies.

    Raises CheckpointError when the architecture does not match the file (under `min_match` of
    either side's BatchNorms found in the other); a partial match above it sets `warning`.
    """
    if not arch:
        return HEURISTIC
    if arch != "hf" and not arch.startswith("torchvision:"):
        raise CheckpointError(ARCH_USAGE)
    if not batchnorms:
        return Wiring("heuristic", note=f"--arch {arch}: the file has no BatchNorm layers, so wiring is not needed")
    if arch == "hf":
        build, shape, label = _hf_factory(Path(src.files[0]).parent)
        if build is None:
            return Wiring("heuristic", note=f"--arch hf: {label}; using the name-based heuristic instead")
        wiring = _trace(build, shape, src, f"hf:{label}")
    else:
        wiring = _trace(*_torchvision_factory(arch.partition(":")[2]), src, arch)
    if wiring.known is None:
        return wiring
    n_file, n_model = len(batchnorms), len(wiring.known)
    found = len(set(batchnorms) & wiring.known)
    if found < min_match * n_file or found < min_match * n_model:
        raise CheckpointError(
            f"--arch {arch} does not match this file: {found} of its {n_file} BatchNorm layers found "
            f"in that model ({n_model} BatchNorm layers). Pass the architecture the file was saved "
            "from, or leave out --arch to use the name-based heuristic.")
    if found == n_file and found == n_model:
        return wiring
    parts = []
    if found < n_file:
        parts.append(f"{n_file - found} of the file's {n_file} BatchNorm layers are not in that model "
                     "(the name-based heuristic is used for them)")
    if found < n_model:
        parts.append(f"{n_model - found} of the model's {n_model} BatchNorm layers are not in the file")
    return Wiring(wiring.source, wiring.relu_after, wiring.known,
                  note="; ".join(parts), model_batchnorms=n_model,
                  warning=f"--arch {arch} matches this file only in part: {found} of its {n_file} "
                          f"BatchNorm layers found in that model ({n_model} BatchNorm layers)")

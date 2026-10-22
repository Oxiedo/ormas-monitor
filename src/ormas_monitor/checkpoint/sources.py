"""Read the tensors of a saved model one at a time, without loading the whole model.

`open_source(path)` accepts:

  * a single file: .pt/.pth/.bin/.ckpt (PyTorch, read with ``torch.load(mmap=True,
    weights_only=True)``) or .safetensors (read with ``safe_open``, tensor by tensor);
  * a wrapper dict around the weights: ``state_dict``, ``model``, ``ema``, a Lightning ``.ckpt``;
  * a Hugging Face model folder: ``model.safetensors``, sharded ``model-0000x-of-0000N.safetensors``
    with ``model.safetensors.index.json``, ``pytorch_model.bin`` or its sharded index;
  * a ``*.index.json`` file directly.

Files named by a folder or an index (shards, a symlinked ``model.safetensors``) must resolve
inside that folder; ``../``, absolute paths and symlinks pointing elsewhere are refused. The one
exception is a Hugging Face cache snapshot, whose files are symlinks into the cache's ``blobs``.

``module.`` (DataParallel) and ``_orig_mod.`` (torch.compile) prefixes are removed. PyTorch files
that contain Python objects rather than plain tensors are refused unless ``trust=True``, because
unpickling them can run code.
"""
from __future__ import annotations

import json
import pickle
import zipfile
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Mapping

import torch

WRAPPER_KEYS = ("state_dict", "model_state_dict", "model", "net", "module", "ema")
STRIP_PREFIXES = ("module.", "_orig_mod.")
FOLDER_FILES = ("model.safetensors.index.json", "model.safetensors",
                "pytorch_model.bin.index.json", "pytorch_model.bin")
SAFETENSORS_DTYPES = {
    "F64": "float64", "F32": "float32", "F16": "float16", "BF16": "bfloat16",
    "F8_E4M3": "float8_e4m3fn", "F8_E5M2": "float8_e5m2",
    "I64": "int64", "I32": "int32", "I16": "int16", "I8": "int8", "U8": "uint8", "BOOL": "bool",
}


class CheckpointError(Exception):
    """The file cannot be read, or reading it would not be safe."""


@dataclass(frozen=True)
class TensorInfo:
    """What is known about a tensor before its data is read."""
    name: str
    shape: tuple[int, ...]
    dtype: str          # torch dtype name without the "torch." prefix, e.g. "bfloat16"
    floating: bool
    numel: int


def _clean(name: str) -> str:
    changed = True
    while changed:
        changed = False
        for p in STRIP_PREFIXES:
            if name.startswith(p):
                name, changed = name[len(p):], True
    return name


def _dtype_floating(dtype: str) -> bool:
    t = getattr(torch, dtype, None)
    return isinstance(t, torch.dtype) and t.is_floating_point


class TensorSource:
    """An ordered, read-only view of a checkpoint's tensors, read one at a time.

    Attributes: `info` (name -> TensorInfo, in file order), `format`, `files`, `ordered` (True
    when the order is the model's module order, False for safetensors, which stores names
    sorted), `streamed` (False when the whole file had to be read into memory), `wrapper` (the
    dict key the weights were found under, or None), `config` (the folder's config.json, or
    None), `problems` (structural defects such as shards missing from an index).
    """

    format = "tensors"

    def __init__(self) -> None:
        self.info: "OrderedDict[str, TensorInfo]" = OrderedDict()
        self.files: list[str] = []
        self.ordered = True
        self.streamed = True
        self.wrapper: str | None = None
        self.config: dict | None = None
        self.problems: list[dict] = []

    def get(self, name: str) -> torch.Tensor:
        """The tensor `name`, detached, on the CPU, in its stored dtype."""
        raise NotImplementedError

    def names(self) -> list[str]:
        return list(self.info)

    def __contains__(self, name: object) -> bool:
        return name in self.info

    def __len__(self) -> int:
        return len(self.info)

    def items(self) -> Iterator[tuple[str, torch.Tensor]]:
        for name in self.info:
            yield name, self.get(name)

    def close(self) -> None:
        """Release file handles and memory maps."""

    def __enter__(self) -> "TensorSource":
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    def _add(self, name: str, shape: tuple[int, ...], dtype: str) -> str | None:
        """Register a tensor under its cleaned name; returns the name, or None for a duplicate."""
        clean = _clean(name)
        if clean in self.info:
            return None
        numel = 1
        for s in shape:
            numel *= int(s)
        self.info[clean] = TensorInfo(clean, tuple(int(s) for s in shape), dtype, _dtype_floating(dtype), numel)
        return clean


class DictSource(TensorSource):
    """Tensors already in memory, e.g. `model.state_dict()`."""

    format = "state_dict"

    def __init__(self, tensors: Mapping[str, object]) -> None:
        super().__init__()
        self._t: dict[str, torch.Tensor] = {}
        for k, v in tensors.items():
            if isinstance(v, torch.Tensor):
                name = self._add(k, tuple(v.shape), str(v.dtype).replace("torch.", ""))
                if name is not None:
                    self._t[name] = v.detach()

    def get(self, name: str) -> torch.Tensor:
        return self._t[name]

    def close(self) -> None:
        self._t = {}


class SafetensorsSource(TensorSource):
    """One or more .safetensors files, read tensor by tensor through `safe_open`."""

    format = "safetensors"

    def __init__(self, files: list[Path]) -> None:
        super().__init__()
        try:
            from safetensors import safe_open
        except ImportError as e:
            raise CheckpointError("reading .safetensors needs `pip install safetensors`") from e
        self.ordered = False
        self.files = [str(f) for f in files]
        self._handles: list = []
        self._where: dict[str, tuple[int, str]] = {}
        for i, f in enumerate(files):
            try:
                h = safe_open(str(f), framework="pt", device="cpu")
            except Exception as e:      # safetensors raises its own error types
                raise CheckpointError(f"could not read {f.name}: {e}") from e
            self._handles.append(h)
            for raw in h.keys():
                sl = h.get_slice(raw)
                dtype = SAFETENSORS_DTYPES.get(sl.get_dtype(), sl.get_dtype().lower())
                name = self._add(raw, tuple(sl.get_shape()), dtype)
                if name is not None:
                    self._where[name] = (i, raw)

    def get(self, name: str) -> torch.Tensor:
        i, raw = self._where[name]
        return self._handles[i].get_tensor(raw)

    def close(self) -> None:
        self._handles = []


def _unwrap(obj: object, label: str) -> tuple[dict, str | None]:
    """Find the state dict inside a checkpoint object; returns it and the wrapper key used."""
    wrapper = None
    if isinstance(obj, torch.nn.Module):
        obj = obj.state_dict()
    for _ in range(3):
        if not isinstance(obj, dict):
            break
        key = next((k for k in WRAPPER_KEYS if k in obj
                    and isinstance(obj[k], (dict, torch.nn.Module))), None)
        if key is None:
            break
        inner = obj[key]
        obj = inner.state_dict() if isinstance(inner, torch.nn.Module) else inner
        wrapper = key if wrapper is None else f"{wrapper}.{key}"
    if not isinstance(obj, dict):
        raise CheckpointError(f"{label} does not hold a state dict")
    return obj, wrapper


def _torch_load(path: Path, trust: bool) -> tuple[object, bool]:
    """torch.load with weights_only=True first; returns (object, memory-mapped?)."""
    mmap = zipfile.is_zipfile(path)    # mmap needs the zip format torch has saved since 1.6
    try:
        return torch.load(path, map_location="cpu", weights_only=True, mmap=mmap), mmap
    except pickle.UnpicklingError as e:
        if "weights only" not in str(e).lower():
            raise CheckpointError(f"could not read {path.name}: it is not a PyTorch checkpoint ({e})") from e
        if not trust:
            raise CheckpointError(
                f"{path.name} contains Python objects, not just weights, and loading it could run "
                "code. Re-run with --trust only if you trust where the file came from, or save "
                "model.state_dict() instead.") from e
    except Exception as e:
        raise CheckpointError(f"could not read {path.name}: {e}") from e
    try:
        return torch.load(path, map_location="cpu", weights_only=False, mmap=mmap), mmap
    except Exception as e:
        raise CheckpointError(f"could not read {path.name}: {e}") from e


class TorchSource(TensorSource):
    """One or more PyTorch files, memory-mapped when the file format allows it."""

    format = "pytorch"

    def __init__(self, files: list[Path], trust: bool = False) -> None:
        super().__init__()
        self.files = [str(f) for f in files]
        self._t: dict[str, torch.Tensor] = {}
        for f in files:
            obj, mapped = _torch_load(f, trust)
            self.streamed = self.streamed and mapped
            sd, wrapper = _unwrap(obj, f.name)
            if wrapper is not None:
                self.wrapper = wrapper
            for k, v in sd.items():
                if isinstance(v, torch.Tensor):
                    name = self._add(k, tuple(v.shape), str(v.dtype).replace("torch.", ""))
                    if name is not None:
                        self._t[name] = v.detach()
            del obj, sd

    def get(self, name: str) -> torch.Tensor:
        return self._t[name]

    def close(self) -> None:
        self._t = {}


def _allowed_roots(folder: Path) -> list[Path]:
    """Where files named by `folder` may live: the folder itself, plus the shared blob store
    when `folder` is a Hugging Face cache snapshot (``snapshots/<rev>/x -> ../../blobs/<hash>``)."""
    roots = [folder.resolve()]
    if folder.parent.name == "snapshots":
        roots.append((folder.parent.parent / "blobs").resolve())
    return roots


def _inside(path: Path, folder: Path, what: str) -> Path:
    """`path`, after checking that it (and any symlink it is) resolves inside `folder`."""
    real = path.resolve()
    if not any(real == r or r in real.parents for r in _allowed_roots(folder)):
        raise CheckpointError(f"{what} resolves to {real}, outside {folder}; refusing to read it "
                              "(a model folder may only point at files inside itself)")
    return real


def _shard_path(index: Path, shard: object) -> Path:
    """The file a weight_map entry names, refused unless it is a plain relative path inside the folder."""
    if not isinstance(shard, str) or not shard.strip():
        raise CheckpointError(f"{index.name} is not a valid shard index: shard names must be "
                              f"non-empty strings, found {shard!r}")
    rel = Path(shard)
    if rel.is_absolute() or shard.startswith(("/", "\\")) or ".." in rel.parts or ".." in shard.split("\\"):
        raise CheckpointError(f"{index.name} names the shard {shard!r}, which is outside the folder; "
                              "shard paths must be relative and stay inside it")
    path = index.parent / rel
    _inside(path, index.parent, f"{index.name} shard {shard!r}")
    return path


def _from_index(index: Path, trust: bool) -> TensorSource:
    try:
        weight_map = json.loads(index.read_text())["weight_map"]
    except (ValueError, KeyError, TypeError) as e:
        raise CheckpointError(f"{index.name} is not a valid shard index (needs a weight_map)") from e
    if not isinstance(weight_map, dict):
        raise CheckpointError(f"{index.name} is not a valid shard index: weight_map must map tensor "
                              "names to shard file names")
    shards: list[str] = []
    for shard in weight_map.values():
        if shard not in shards:
            shards.append(shard)
    paths = [_shard_path(index, s) for s in shards]
    absent = [p.name for p in paths if not p.is_file()]
    if absent:
        raise CheckpointError(f"{index.name} lists shard files that are missing: {', '.join(absent[:5])}")
    src: TensorSource = (SafetensorsSource(paths) if index.name.endswith(".safetensors.index.json")
                         else TorchSource(paths, trust))
    src.files.insert(0, str(index))
    missing = sorted({_clean(k) for k in weight_map} - set(src.info))
    if missing:
        src.problems.append({"kind": "missing_tensors", "tensors": missing,
                             "detail": f"{index.name} lists them, but no shard contains them"})
    return src


def _folder_files(folder: Path) -> Path:
    for name in FOLDER_FILES:
        if (folder / name).is_file():
            return folder / name
    for pattern in ("*.safetensors.index.json", "*.bin.index.json", "*.safetensors", "*.bin", "*.pt", "*.pth"):
        found = sorted(folder.glob(pattern))
        if len(found) == 1:
            return found[0]
        if len(found) > 1:
            raise CheckpointError(f"{folder} holds several weight files ({', '.join(p.name for p in found[:4])}) "
                                  "and no index; pass one of them")
    raise CheckpointError(f"{folder} holds no model weights (looked for {', '.join(FOLDER_FILES)})")


def _read_config(folder: Path) -> dict | None:
    cfg = folder / "config.json"
    if not cfg.is_file():
        return None
    try:
        data = json.loads(cfg.read_text())
    except ValueError:
        return None
    return data if isinstance(data, dict) else None


def open_source(path: str | Path, trust: bool = False) -> TensorSource:
    """Open a checkpoint file or Hugging Face folder for tensor-by-tensor reading."""
    path = Path(path)
    if not path.exists():
        raise CheckpointError(f"{path} does not exist")
    folder = path.is_dir()
    target = _folder_files(path) if folder else path
    if folder:
        _inside(target, path, target.name)
    if target.name.endswith(".index.json"):
        src = _from_index(target, trust)
        src.format = f"{src.format}-sharded"
    elif target.suffix == ".safetensors":
        src = SafetensorsSource([target])
    else:
        src = TorchSource([target], trust)
    if folder:
        src.format = f"hf-folder/{src.format}"
    src.config = _read_config(target.parent)
    if not src.info:
        raise CheckpointError(f"no tensors found in {path.name}")
    return src


def as_source(tensors: "TensorSource | Mapping[str, object]") -> TensorSource:
    """Accept either a TensorSource or an in-memory mapping of name -> tensor."""
    return tensors if isinstance(tensors, TensorSource) else DictSource(tensors)

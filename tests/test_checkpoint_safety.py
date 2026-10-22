"""Folder and index safety: every file a folder or *.index.json names must stay inside the folder."""
from __future__ import annotations

import json
import os

import pytest
import torch
import torch.nn as nn

from ormas_monitor.checkpoint import CheckpointError, open_source, run
from ormas_monitor.cli import main as cli

safetensors = pytest.importorskip("safetensors.torch")


def weights(path) -> None:
    safetensors.save_file({"secret.weight": torch.randn(3, 3)}, str(path))


def index(folder, weight_map, kind: str = "safetensors") -> None:
    folder.mkdir(exist_ok=True)
    name = "model.safetensors.index.json" if kind == "safetensors" else "pytorch_model.bin.index.json"
    (folder / name).write_text(json.dumps({"weight_map": weight_map}))


@pytest.fixture
def outside(tmp_path):
    (tmp_path / "outside").mkdir()
    weights(tmp_path / "outside" / "private.safetensors")
    torch.save({"x": torch.ones(2)}, tmp_path / "outside" / "other.bin")
    return tmp_path / "outside"


def test_absolute_shard_path_refused(tmp_path, outside):
    index(tmp_path / "m", {"secret.weight": str(outside / "private.safetensors")})
    with pytest.raises(CheckpointError, match="outside the folder"):
        open_source(tmp_path / "m")


@pytest.mark.parametrize("shard", ["../outside/other.bin", "sub/../../outside/other.bin", "..\\outside\\other.bin"])
def test_parent_shard_path_refused(tmp_path, outside, shard):
    index(tmp_path / "m", {"x": shard}, kind="bin")
    with pytest.raises(CheckpointError, match="outside the folder"):
        open_source(tmp_path / "m")


@pytest.mark.parametrize("shard", [5, None, ["a.bin"], ""])
def test_non_string_shard_name_is_a_checkpoint_error(tmp_path, shard):
    index(tmp_path / "m", {"x": shard}, kind="bin")
    with pytest.raises(CheckpointError, match="shard names must be non-empty strings"):
        open_source(tmp_path / "m")


@pytest.mark.parametrize("weight_map", [["a.bin"], "a.bin", 3])
def test_weight_map_must_be_a_mapping(tmp_path, weight_map):
    index(tmp_path / "m", weight_map, kind="bin")
    with pytest.raises(CheckpointError, match="weight_map must map"):
        open_source(tmp_path / "m")


def test_symlinked_shard_escaping_folder_refused(tmp_path, outside):
    index(tmp_path / "m", {"x": "shard.bin"}, kind="bin")
    os.symlink(outside / "other.bin", tmp_path / "m" / "shard.bin")
    with pytest.raises(CheckpointError, match="outside"):
        open_source(tmp_path / "m")


def test_symlinked_model_file_escaping_folder_refused(tmp_path, outside):
    (tmp_path / "m").mkdir()
    os.symlink(outside / "private.safetensors", tmp_path / "m" / "model.safetensors")
    with pytest.raises(CheckpointError, match="outside"):
        open_source(tmp_path / "m")


def test_symlinks_inside_folder_and_hf_cache_blobs_allowed(tmp_path):
    folder = tmp_path / "m"
    folder.mkdir()
    weights(folder / "real.safetensors")
    os.symlink(folder / "real.safetensors", folder / "model.safetensors")
    with open_source(folder) as src:
        assert "secret.weight" in src
    # Hugging Face cache layout: snapshots/<rev>/model.safetensors -> ../../blobs/<hash>
    repo = tmp_path / "models--org--name"
    (repo / "blobs").mkdir(parents=True)
    (repo / "snapshots" / "abc123").mkdir(parents=True)
    weights(repo / "blobs" / "f00d")
    os.symlink("../../blobs/f00d", repo / "snapshots" / "abc123" / "model.safetensors")
    with open_source(repo / "snapshots" / "abc123") as src:
        assert "secret.weight" in src


def test_index_given_directly_is_held_to_its_folder(tmp_path, outside):
    index(tmp_path / "m", {"secret.weight": "../outside/private.safetensors"})
    with pytest.raises(CheckpointError, match="outside the folder"):
        run(tmp_path / "m" / "model.safetensors.index.json")


def test_cli_exits_2_with_a_clear_message(tmp_path, outside, capsys):
    index(tmp_path / "m", {"x": 5}, kind="bin")
    assert cli(["check", str(tmp_path / "m")]) == 2
    err = capsys.readouterr().err
    assert err.startswith("error: ") and "TypeError" not in err


def test_valid_relative_shards_still_work(tmp_path):
    folder = tmp_path / "m"
    folder.mkdir()
    (folder / "shards").mkdir()
    sd = nn.Linear(3, 2).state_dict()
    torch.save(sd, folder / "shards" / "a.bin")
    index(folder, {k: "shards/a.bin" for k in sd}, kind="bin")
    with open_source(folder) as src:
        assert set(src.info) == set(sd)

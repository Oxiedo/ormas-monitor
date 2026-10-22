"""Checkpoint formats: single files, wrappers, prefixes, sharded folders, fp16, pickled refusal."""
from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch
import torch.nn as nn

from ormas_monitor.checkpoint import CheckpointError, load_tensors, open_source, run


def small_cnn() -> nn.Sequential:
    torch.manual_seed(0)
    return nn.Sequential(
        nn.Conv2d(3, 8, 3, padding=1), nn.BatchNorm2d(8), nn.ReLU(),
        nn.Conv2d(8, 16, 3, padding=1), nn.BatchNorm2d(16), nn.ReLU(),
        nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(16, 4))


def lesioned() -> dict:
    """small_cnn with conv '3' zeroed and its BatchNorm shift <= 0: 16 certified-dead channels."""
    m = small_cnn()
    with torch.no_grad():
        m[3].weight.zero_()
        m[4].bias.fill_(-0.1)
    return m.state_dict()


def dead_count(res: dict) -> int:
    return sum(len(f["channels"]) for f in res["findings"] if f["kind"] == "dead_channels")


def write_sharded_safetensors(sd: dict, folder: Path, n: int = 3) -> None:
    safetensors = pytest.importorskip("safetensors.torch")
    folder.mkdir()
    keys = list(sd)
    weight_map = {}
    for i in range(n):
        shard = f"model-{i + 1:05d}-of-{n:05d}.safetensors"
        part = {k: sd[k].contiguous() for k in keys[i::n]}
        safetensors.save_file(part, str(folder / shard))
        weight_map.update({k: shard for k in part})
    (folder / "model.safetensors.index.json").write_text(json.dumps({"metadata": {}, "weight_map": weight_map}))


@pytest.mark.parametrize("suffix", [".pt", ".pth", ".bin", ".ckpt"])
def test_torch_single_file(tmp_path, suffix):
    torch.save(lesioned(), tmp_path / f"m{suffix}")
    res = run(tmp_path / f"m{suffix}")
    assert dead_count(res) == 16 and res["format"] == "pytorch" and res["streamed"]


@pytest.mark.parametrize("key", ["state_dict", "model", "ema", "model_state_dict"])
def test_wrapper_dicts(tmp_path, key):
    torch.save({key: lesioned(), "epoch": 3, "optimizer": {"state": {}}}, tmp_path / "w.pt")
    res = run(tmp_path / "w.pt")
    assert dead_count(res) == 16 and res["wrapper"] == key


def test_lightning_style_checkpoint(tmp_path):
    sd = {f"model.{k}": v for k, v in lesioned().items()}
    torch.save({"epoch": 1, "global_step": 10, "pytorch-lightning_version": "2.4.0", "state_dict": sd,
                "optimizer_states": [{"state": {0: {"exp_avg": torch.zeros(3)}}, "param_groups": []}],
                "lr_schedulers": [], "callbacks": {}, "hyper_parameters": {"lr": 0.1}}, tmp_path / "x.ckpt")
    res = run(tmp_path / "x.ckpt")
    assert res["wrapper"] == "state_dict" and dead_count(res) == 16


@pytest.mark.filterwarnings("ignore")
def test_real_lightning_checkpoint(tmp_path):
    L = pytest.importorskip("lightning")

    class Module(L.LightningModule):
        def __init__(self) -> None:
            super().__init__()
            self.net = small_cnn()

        def training_step(self, batch, _):
            return self.net(batch[0]).sum()

        def configure_optimizers(self):
            return torch.optim.SGD(self.parameters(), lr=0.0)

    data = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(torch.randn(4, 3, 8, 8)), batch_size=4)
    trainer = L.Trainer(max_steps=1, logger=False, enable_checkpointing=False, enable_progress_bar=False,
                        enable_model_summary=False, accelerator="cpu", default_root_dir=tmp_path)
    module = Module()
    trainer.fit(module, data)
    trainer.save_checkpoint(tmp_path / "real.ckpt")
    res = run(tmp_path / "real.ckpt")
    assert res["wrapper"] == "state_dict" and res["tensors"] == len(module.state_dict())


@pytest.mark.parametrize("prefix", ["module.", "_orig_mod.", "module._orig_mod."])
def test_prefixes_are_stripped(tmp_path, prefix):
    torch.save({prefix + k: v for k, v in lesioned().items()}, tmp_path / "p.pt")
    res = run(tmp_path / "p.pt")
    assert dead_count(res) == 16 and all(not l["batchnorm"].startswith(("module", "_orig")) for l in res["layers"])


def test_single_safetensors_fp16(tmp_path):
    safetensors = pytest.importorskip("safetensors.torch")
    sd = {k: (v.half() if v.is_floating_point() else v) for k, v in lesioned().items()}
    safetensors.save_file(sd, str(tmp_path / "m.safetensors"))
    res = run(tmp_path / "m.safetensors")
    assert res["format"] == "safetensors" and dead_count(res) == 16 and "float16" in res["dtypes"]
    assert res["layers"][1]["layer"] == "3"          # paired by name order despite sorted storage


def test_bf16_statistics_in_float32(tmp_path):
    sd = small_cnn().state_dict()
    sd["0.weight"] = sd["0.weight"].bfloat16()
    sd["0.weight"][0, 0, 0, 0] = float("nan")
    torch.save(sd, tmp_path / "b.pt")
    res = run(tmp_path / "b.pt")
    nf = [f for f in res["findings"] if f["kind"] == "nonfinite"]
    assert nf and nf[0]["values"]["0.weight"] == {"nan": 1, "inf": 0}


def test_hf_folder_sharded_safetensors(tmp_path):
    write_sharded_safetensors(lesioned(), tmp_path / "hf")
    res = run(tmp_path / "hf")
    assert res["format"] == "hf-folder/safetensors-sharded" and len(res["files"]) == 4
    assert dead_count(res) == 16


def test_index_file_directly_and_missing_tensor(tmp_path):
    write_sharded_safetensors(lesioned(), tmp_path / "hf")
    index = tmp_path / "hf" / "model.safetensors.index.json"
    data = json.loads(index.read_text())
    data["weight_map"]["9.weight"] = "model-00001-of-00003.safetensors"     # listed but absent
    index.write_text(json.dumps(data))
    res = run(index)
    missing = [f for f in res["findings"] if f["kind"] == "missing_tensors"]
    assert missing and missing[0]["severity"] == "critical" and missing[0]["tensors"] == ["9.weight"]


def test_missing_shard_is_an_error(tmp_path):
    write_sharded_safetensors(lesioned(), tmp_path / "hf")
    (tmp_path / "hf" / "model-00002-of-00003.safetensors").unlink()
    with pytest.raises(CheckpointError, match="missing"):
        run(tmp_path / "hf")


def test_hf_folder_pytorch_bin_and_sharded_bin(tmp_path):
    sd = lesioned()
    (tmp_path / "one").mkdir()
    torch.save(sd, tmp_path / "one" / "pytorch_model.bin")
    assert dead_count(run(tmp_path / "one")) == 16
    folder = tmp_path / "sharded"
    folder.mkdir()
    keys, weight_map = list(sd), {}
    for i in range(2):
        shard = f"pytorch_model-{i + 1:05d}-of-00002.bin"
        torch.save({k: sd[k] for k in keys[i::2]}, folder / shard)
        weight_map.update({k: shard for k in keys[i::2]})
    (folder / "pytorch_model.bin.index.json").write_text(json.dumps({"weight_map": weight_map}))
    res = run(folder)
    assert res["format"] == "hf-folder/pytorch-sharded" and dead_count(res) == 16


def test_hf_folder_without_weights(tmp_path):
    (tmp_path / "config.json").write_text("{}")
    with pytest.raises(CheckpointError, match="no model weights"):
        run(tmp_path)


def test_pickled_objects_refused_unless_trusted(tmp_path):
    torch.save({"model": small_cnn()}, tmp_path / "full.pt")
    with pytest.raises(CheckpointError, match="--trust"):
        run(tmp_path / "full.pt")
    assert run(tmp_path / "full.pt", trust=True)["tensors"] == len(small_cnn().state_dict())


def test_not_a_checkpoint(tmp_path):
    (tmp_path / "junk.pt").write_bytes(b"this is not a checkpoint")
    with pytest.raises(CheckpointError, match="could not read"):
        run(tmp_path / "junk.pt")


def test_legacy_format_is_read_and_flagged(tmp_path):
    torch.save(lesioned(), tmp_path / "old.pt", _use_new_zipfile_serialization=False)
    res = run(tmp_path / "old.pt")
    assert not res["streamed"] and dead_count(res) == 16 and any("legacy" in n for n in res["notes"])


def test_load_tensors_compat(tmp_path):
    torch.save({"state_dict": {"module." + k: v for k, v in lesioned().items()}}, tmp_path / "c.pt")
    tensors = load_tensors(tmp_path / "c.pt")
    assert list(tensors) == list(lesioned())
    with open_source(tmp_path / "c.pt") as src:
        assert src.info["3.weight"].shape == (16, 8, 3, 3)

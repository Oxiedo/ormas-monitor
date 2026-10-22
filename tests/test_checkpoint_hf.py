"""`--arch hf`: wiring traced from a local config.json (tiny random models, no download)."""
from __future__ import annotations

import json

import pytest
import torch

from ormas_monitor.checkpoint import CheckpointError, run

transformers = pytest.importorskip("transformers")


def tiny_resnet(folder, **save_kwargs):
    config = transformers.ResNetConfig(embedding_size=8, hidden_sizes=[8, 16], depths=[1, 1],
                                       layer_type="basic", num_labels=3)
    torch.manual_seed(0)
    model = transformers.ResNetForImageClassification(config)
    stem_conv = model.resnet.embedder.embedder.convolution
    stem_bn = model.resnet.embedder.embedder.normalization         # feeds a ReLU directly
    end_bn = model.resnet.encoder.stages[1].layers[0].layer[1].normalization   # residual add, then ReLU
    with torch.no_grad():
        stem_conv.weight[:2] = 0
        stem_bn.bias[:2] = -0.5
        end_bn.weight[:4] = 0
        end_bn.bias[:4] = -0.5
    model.save_pretrained(folder, **save_kwargs)
    return model


def by_kind(res: dict, kind: str) -> list[dict]:
    return [f for f in res["findings"] if f["kind"] == kind]


def test_hf_resnet_folder_traced(tmp_path):
    tiny_resnet(tmp_path / "m")
    res = run(tmp_path / "m", arch="hf")
    w = res["wiring"]
    assert w["traced"] and w["source"] == "hf:ResNetForImageClassification" and w["matched"] == w["batchnorms"]
    dead = by_kind(res, "dead_channels")
    assert [(f["batchnorm"], f["channels"]) for f in dead] == [("resnet.embedder.embedder.normalization", [0, 1])]
    assert "if a ReLU follows" not in dead[0]["title"]
    end = "resnet.encoder.stages.1.layers.0.layer.1.normalization"
    flat = {f["batchnorm"]: f["channels"] for f in by_kind(res, "flat_channels")}
    assert flat[end] == [0, 1, 2, 3]                 # residual add before the ReLU: no certificate


def test_hf_resnet_without_arch_uses_conditional_wording(tmp_path):
    tiny_resnet(tmp_path / "m")
    res = run(tmp_path / "m")
    assert not res["wiring"]["traced"]
    assert all("if a ReLU follows" in f["title"] for f in by_kind(res, "dead_channels"))


@pytest.mark.parametrize("kwargs", [{"max_shard_size": "20KB"}, {"safe_serialization": False}])
def test_hf_resnet_sharded_and_bin(tmp_path, kwargs):
    tiny_resnet(tmp_path / "m", **kwargs)
    res = run(tmp_path / "m", arch="hf")
    assert res["wiring"]["traced"] and len(by_kind(res, "dead_channels")) == 1
    if "max_shard_size" in kwargs:
        assert res["format"] == "hf-folder/safetensors-sharded" and len(res["files"]) > 2


def test_hf_text_model_without_batchnorm(tmp_path):
    config = transformers.BertConfig(vocab_size=40, hidden_size=16, num_hidden_layers=1,
                                     num_attention_heads=2, intermediate_size=32)
    transformers.BertModel(config).save_pretrained(tmp_path / "bert")
    res = run(tmp_path / "bert", arch="hf")
    assert "no BatchNorm" in res["notes"][0] and res["norm_layers"] > 0
    assert [f for f in res["findings"] if f["severity"] != "info"] == []


def test_hf_non_image_model_with_batchnorm_falls_back(tmp_path):
    folder = tmp_path / "odd"
    folder.mkdir()
    conv, bn = torch.nn.Conv1d(4, 4, 1), torch.nn.BatchNorm1d(4)
    torch.save({"conv.weight": conv.weight, "conv.bias": conv.bias,
                **{f"bn.{k}": v for k, v in bn.state_dict().items()}}, folder / "pytorch_model.bin")
    (folder / "config.json").write_text(json.dumps(transformers.BertConfig(
        vocab_size=40, hidden_size=16, num_hidden_layers=1, num_attention_heads=2).to_dict()))
    res = run(folder, arch="hf")
    assert not res["wiring"]["traced"] and "does not take an image input" in res["notes"][0]


def test_arch_that_does_not_match_is_an_error(tmp_path):
    tiny_resnet(tmp_path / "m")
    pytest.importorskip("torchvision")
    with pytest.raises(CheckpointError, match="does not match"):
        run(tmp_path / "m", arch="torchvision:resnet18")


def test_arch_hf_needs_config(tmp_path):
    torch.save(torch.nn.BatchNorm1d(4).state_dict(), tmp_path / "m.pt")
    with pytest.raises(CheckpointError, match="config.json"):
        run(tmp_path / "m.pt", arch="hf")

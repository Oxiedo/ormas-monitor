"""Weights-only findings: healthy models stay quiet; transformer norms; streaming diff."""
from __future__ import annotations

import json

import pytest
import torch
import torch.nn as nn

from ormas_monitor.checkpoint import CheckConfig, check, diff, open_source, run


class TinyLM(nn.Module):
    """A pre-norm transformer-shaped model: embeddings, LayerNorm, RMSNorm-style scale, MLP."""

    def __init__(self, vocab: int = 50, d: int = 16) -> None:
        super().__init__()
        self.embed_tokens = nn.Embedding(vocab, d)
        self.layers = nn.ModuleList([nn.ModuleDict({
            "input_layernorm": nn.LayerNorm(d), "q_proj": nn.Linear(d, d), "mlp_up": nn.Linear(d, 4 * d),
            "mlp_down": nn.Linear(4 * d, d)}) for _ in range(2)])
        self.norm = nn.LayerNorm(d, bias=False)
        self.lm_head = nn.Linear(d, vocab, bias=False)


def warnings_or_worse(res: dict) -> list[dict]:
    return [f for f in res["findings"] if f["severity"] != "info"]


def test_healthy_torchvision_model_has_no_warnings(tmp_path):
    torchvision = pytest.importorskip("torchvision")
    torch.save(torchvision.models.resnet18(weights=None).state_dict(), tmp_path / "r18.pt")
    assert warnings_or_worse(run(tmp_path / "r18.pt")) == []
    assert warnings_or_worse(run(tmp_path / "r18.pt", arch="torchvision:resnet18")) == []


def test_healthy_transformer_has_no_warnings():
    res = check(TinyLM().state_dict())
    assert warnings_or_worse(res) == [] and res["norm_layers"] == 3


def test_norm_scale_near_zero_is_a_warning():
    sd = TinyLM().state_dict()
    sd["layers.1.input_layernorm.weight"][[2, 5]] = 0.0
    sd["norm.weight"][7] = 1e-9
    f = [x for x in check(sd)["findings"] if x["kind"] == "norm_scale_zero"]
    assert len(f) == 1 and f[0]["severity"] == "warning"
    assert {h["tensor"]: h["entries"] for h in f[0]["tensors"]} == {
        "layers.1.input_layernorm.weight": [2, 5], "norm.weight": [7]}


def test_zero_centred_family_is_not_flagged(tmp_path):
    safetensors = pytest.importorskip("safetensors.torch")
    sd = TinyLM().state_dict()
    for k in sd:
        if "norm" in k:
            sd[k] = torch.zeros_like(sd[k])          # Gemma stores scale - 1: zero is healthy
    (tmp_path / "hf").mkdir()
    safetensors.save_file(sd, str(tmp_path / "hf" / "model.safetensors"))
    (tmp_path / "hf" / "config.json").write_text(json.dumps({"model_type": "gemma2"}))
    res = run(tmp_path / "hf")
    assert warnings_or_worse(res) == [] and res["norm_layers"] == 0
    assert any("1 + weight" in n for n in res["notes"])


def test_nonfinite_and_zero_tensor():
    sd = TinyLM().state_dict()
    sd["layers.0.q_proj.weight"][0, 0] = float("inf")
    sd["layers.1.mlp_up.weight"].zero_()
    kinds = {f["kind"]: f["severity"] for f in check(sd)["findings"]}
    assert kinds["nonfinite"] == "critical" and kinds["zero_tensor"] == "warning"


def test_no_dead_neuron_claims_for_mlp_rows():
    sd = TinyLM().state_dict()
    sd["layers.0.mlp_up.weight"][:3] = 0.0           # zero rows in a GELU/SiLU MLP
    res = check(sd)
    assert not [f for f in res["findings"] if f["kind"] in ("dead_channels", "decayed_channels")]
    assert all(f["severity"] == "info" for f in res["findings"])


def test_embedding_padding_row_is_not_reported():
    m = TinyLM()
    with torch.no_grad():
        m.embed_tokens.weight[0].zero_()
    assert check(m.state_dict())["findings"] == []


def test_chunking_does_not_change_results():
    sd = TinyLM().state_dict()
    sd["layers.0.mlp_up.weight"][1, 3] = float("nan")
    a = check(sd)
    b = check(sd, config=CheckConfig(chunk_elems=7))
    assert a["findings"] == b["findings"]


def test_streaming_diff_frozen_and_embedding_rows(tmp_path):
    safetensors = pytest.importorskip("safetensors.torch")
    before = TinyLM().state_dict()
    after = {k: v.clone() for k, v in before.items()}
    after["layers.1.mlp_down.weight"] += 0.01         # trained
    after["lm_head.weight"] += 0.01
    after["embed_tokens.weight"][:10] += 0.1          # only 10 of 50 tokens seen
    safetensors.save_file(before, str(tmp_path / "before.safetensors"))
    safetensors.save_file(after, str(tmp_path / "after.safetensors"))
    res = run(tmp_path / "after.safetensors", against=tmp_path / "before.safetensors")
    d = res["diff"]
    assert {r["tensor"] for r in d["changed"]} == set(before)
    assert set(before) - set(d["unchanged"]) == {"layers.1.mlp_down.weight", "lm_head.weight", "embed_tokens.weight"}
    emb = d["embedding_rows_unchanged"]
    assert emb == [{"tensor": "embed_tokens.weight", "count": 40, "size": 50, "rows": list(range(10, 50))}]
    kinds = {f["kind"]: f["severity"] for f in res["findings"]}
    assert kinds == {"frozen_tensors": "info", "embedding_rows_unchanged": "info"}
    assert d["newly_dead"] == []


def test_identical_weights_warning(tmp_path):
    torch.save(TinyLM().state_dict(), tmp_path / "a.pt")
    res = run(tmp_path / "a.pt", against=tmp_path / "a.pt")
    assert [f["kind"] for f in res["findings"]] == ["identical_weights"]


def test_diff_matches_dense_computation():
    a, b = TinyLM().state_dict(), TinyLM().state_dict()
    d = diff(a, b, CheckConfig(chunk_elems=5))
    row = next(r for r in d["changed"] if r["tensor"] == "layers.0.mlp_up.weight")
    dense = float((a["layers.0.mlp_up.weight"] - b["layers.0.mlp_up.weight"]).norm())
    assert row["abs_change"] == pytest.approx(dense, rel=1e-6)


def test_newly_dead_and_dtype_change(tmp_path):
    torch.manual_seed(0)
    m = nn.Sequential(nn.Conv2d(3, 8, 3), nn.BatchNorm2d(8), nn.ReLU())
    torch.save(m.state_dict(), tmp_path / "before.pt")
    with torch.no_grad():
        m[0].weight[:3] = 0
        m[0].bias[:3] = 0
        m[1].bias[:3] = -1
    torch.save({k: (v.half() if v.is_floating_point() else v) for k, v in m.state_dict().items()},
               tmp_path / "after.pt")
    res = run(tmp_path / "after.pt", against=tmp_path / "before.pt")
    assert res["diff"]["newly_dead"] == [{"layer": "0", "batchnorm": "1", "count": 3}]
    assert {x["tensor"] for x in res["diff"]["dtype_changed"]} >= {"0.weight", "1.bias"}
    assert res["findings"][0]["kind"] == "newly_dead"


def test_result_keeps_documented_keys(tmp_path):
    torch.save(TinyLM().state_dict(), tmp_path / "a.pt")
    res = run(tmp_path / "a.pt", against=tmp_path / "a.pt")
    assert {"file", "tensors", "parameters", "batchnorm_layers", "findings", "layers", "against", "diff"} <= set(res)
    assert {"changed", "unchanged", "added", "removed", "reshaped", "newly_dead"} <= set(res["diff"])
    json.dumps(res)                                    # serialisable as is (no tensors)


def test_open_source_reads_lazily(tmp_path):
    torch.save(TinyLM().state_dict(), tmp_path / "a.pt")
    with open_source(tmp_path / "a.pt") as src:
        assert src.streamed and src.info["embed_tokens.weight"].shape == (50, 16)


def test_titles_name_layer_kind_from_weight_rank():
    torch.manual_seed(0)
    m = nn.Sequential(nn.Linear(6, 8), nn.BatchNorm1d(8), nn.ReLU(), nn.Linear(8, 4))
    sd = m.state_dict()
    sd["0.weight"][:2] = 0
    sd["0.bias"][:2] = 0
    sd["1.bias"][:2] = -1
    sd["3.weight"][1] = 0
    titles = {f["kind"]: f["title"] for f in check(sd)["findings"]}
    assert titles["dead_channels"] == ("2 of 8 channels in layer `0` (Linear, BatchNorm `1`) are dead (zero filter or scale, shift <= 0) "
                                       "if a ReLU directly follows `1`")
    assert titles["zero_filters"] == "1 of 4 output units of layer `3` (Linear) have all-zero weights"
    traced = {f["kind"]: f["title"] for f in check(sd, relu_after={"1"})["findings"]}
    assert traced["dead_channels"] == "2 of 8 channels in layer `0` (Linear, BatchNorm `1`) are dead (zero filter or scale, shift <= 0)"


def test_title_without_a_producer_names_the_batchnorm():
    bn = nn.BatchNorm2d(4)
    with torch.no_grad():
        bn.weight[0] = 0
        bn.bias[0] = -1
    f = next(f for f in check({f"stem.bn.{k}": v for k, v in bn.state_dict().items()})["findings"]
             if f["kind"] == "dead_channels")
    assert f["title"].startswith("1 of 4 channels in BatchNorm `stem.bn` is dead (zero filter or scale, shift <= 0)")


def zero_filter_with_bias(bias: float, beta: float, gamma: float = 1.0) -> dict:
    """Linear(bias) -> BatchNorm1d with unit 0's filter zeroed (repro r5: BN of a constant column)."""
    torch.manual_seed(0)
    m = nn.Sequential(nn.Linear(5, 2), nn.BatchNorm1d(2), nn.ReLU())
    with torch.no_grad():
        m[0].weight[0] = 0
        m[0].bias[0] = bias
        m[1].weight[0] = gamma
        m[1].bias[0] = beta
    return m.state_dict()


def certified(sd: dict, tmp_path, name: str) -> list[int]:
    torch.save(sd, tmp_path / f"{name}.pt")
    res = run(tmp_path / f"{name}.pt")
    in_findings = sorted(c for f in res["findings"] if f["kind"] == "dead_channels" for c in f["channels"])
    in_layers = sorted(c for l in res["layers"] for c in l["certified"])
    assert in_findings == in_layers                    # findings and layers agree
    return in_layers


def test_zero_filter_with_nonzero_producer_bias_is_not_certified(tmp_path):
    # In floats, BatchNorm of a constant nonzero column is not exactly beta: up to ~1e-3 above it,
    # so with beta = 0 the ReLU opens on some inputs and the channel is not provably dead.
    assert certified(zero_filter_with_bias(bias=0.7, beta=0.0), tmp_path, "biased") == []
    assert certified(zero_filter_with_bias(bias=0.0, beta=0.0), tmp_path, "unbiased") == [0]
    assert certified(zero_filter_with_bias(bias=0.7, beta=0.0, gamma=0.0), tmp_path, "gamma0") == [0]
    torch.save(zero_filter_with_bias(bias=0.7, beta=-1.0), tmp_path / "shut.pt")
    res = run(tmp_path / "shut.pt")                    # nearly constant and shut: decayed, a warning
    assert res["layers"][0]["certified"] == [] and res["layers"][0]["decayed"] == [0]
    assert [f["kind"] for f in res["findings"] if f["severity"] == "warning"] == ["decayed_channels"]


def test_newly_dead_agrees_with_certificate_rule(tmp_path):
    torch.save(zero_filter_with_bias(bias=0.0, beta=0.5), tmp_path / "before.pt")
    torch.save(zero_filter_with_bias(bias=0.7, beta=0.0), tmp_path / "after.pt")
    assert run(tmp_path / "after.pt", against=tmp_path / "before.pt")["diff"]["newly_dead"] == []

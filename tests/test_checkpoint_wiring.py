"""`--arch` wiring: matching the file, partial-match warnings, and cheap tracing on the meta device."""
from __future__ import annotations

import pytest
import torch
import torch.nn as nn

from ormas_monitor.checkpoint import CheckpointError, arch_relu_followers, run
from ormas_monitor.checkpoint.wiring import _torchvision_factory, _traced, _traced_on
from ormas_monitor.cli import main as cli

torchvision = pytest.importorskip("torchvision")


@pytest.fixture(scope="module")
def r18_sd() -> dict:
    torch.manual_seed(0)
    return torchvision.models.resnet18(weights=None).state_dict()


def bn_prefixes(sd: dict) -> list[str]:
    return [k[: -len(".running_mean")] for k in sd if k.endswith(".running_mean")]


def without(sd: dict, prefixes: list[str]) -> dict:
    return {k: v for k, v in sd.items() if not any(k.startswith(p + ".") for p in prefixes)}


def test_full_match_has_no_warning(tmp_path, r18_sd):
    torch.save(r18_sd, tmp_path / "r18.pt")
    res = run(tmp_path / "r18.pt", arch="torchvision:resnet18")
    w = res["wiring"]
    assert w["traced"] and w["matched"] == w["batchnorms"] == w["model_batchnorms"] == 20
    assert not [f for f in res["findings"] if f["kind"] == "arch_partial_match"]


def test_poor_match_is_an_error(tmp_path):
    torch.manual_seed(0)
    m = nn.Sequential(nn.Conv2d(3, 8, 3), nn.BatchNorm2d(8), nn.ReLU(), nn.Conv2d(8, 8, 3), nn.BatchNorm2d(8))
    torch.save(m.state_dict(), tmp_path / "small.pt")
    with pytest.raises(CheckpointError, match=r"does not match this file: \d+ of its 2 BatchNorm layers found "
                                              r"in that model \(53 BatchNorm layers\)"):
        run(tmp_path / "small.pt", arch="torchvision:resnet50")


def test_under_half_of_the_file_matched_is_an_error(tmp_path, r18_sd):
    sd = dict(r18_sd)
    for p in bn_prefixes(r18_sd)[:11]:                 # 11 of 20 BatchNorms renamed: 9 found
        for s in ("weight", "bias", "running_mean", "running_var", "num_batches_tracked"):
            sd[f"extra.{p}.{s}"] = sd.pop(f"{p}.{s}")
    torch.save(sd, tmp_path / "x.pt")
    with pytest.raises(CheckpointError, match="9 of its 20 BatchNorm layers"):
        run(tmp_path / "x.pt", arch="torchvision:resnet18")


def test_partial_match_warns_in_findings_and_cli(tmp_path, r18_sd, capsys):
    sd = without(r18_sd, bn_prefixes(r18_sd)[-4:])      # 16 of the model's 20 BatchNorms in the file
    torch.save(sd, tmp_path / "part.pt")
    res = run(tmp_path / "part.pt", arch="torchvision:resnet18")
    partial = [f for f in res["findings"] if f["kind"] == "arch_partial_match"]
    assert len(partial) == 1 and partial[0]["severity"] == "warning"
    assert "only in part: 16 of its 16 BatchNorm layers found in that model (20 BatchNorm layers)" \
        in partial[0]["title"]
    assert "4 of the model's 20 BatchNorm layers are not in the file" in partial[0]["detail"]
    assert res["findings"][0]["kind"] == "arch_partial_match"
    assert cli(["check", str(tmp_path / "part.pt"), "--arch", "torchvision:resnet18",
                "--out", str(tmp_path / "r.html")]) == 0
    out = capsys.readouterr().out
    assert "[warning] --arch torchvision:resnet18 matches this file only in part" in out
    assert "--fail-on warning to fail on them" in out


def test_extra_batchnorms_in_file_use_heuristic_and_warn(tmp_path, r18_sd):
    sd = dict(r18_sd)
    bn = nn.BatchNorm2d(4).state_dict()
    sd.update({f"head.bn.{k}": v for k, v in bn.items()})
    torch.save(sd, tmp_path / "extra.pt")
    res = run(tmp_path / "extra.pt", arch="torchvision:resnet18")
    f = next(f for f in res["findings"] if f["kind"] == "arch_partial_match")
    assert "20 of its 21" in f["title"] and "heuristic is used for them" in f["detail"]
    layer = next(l for l in res["layers"] if l["batchnorm"] == "head.bn")
    assert layer["wiring"] == "heuristic"


def test_unknown_torchvision_model_is_an_error():
    with pytest.raises(CheckpointError, match="no model called"):
        arch_relu_followers("torchvision:not_a_model")


@pytest.mark.parametrize("name", ["resnet18", "densenet121", "mobilenet_v2"])
def test_meta_trace_matches_cpu_trace(name):
    build, shape = _torchvision_factory(name)
    model, meta = _traced_on(build, shape, "meta")
    assert next(model.parameters()).is_meta
    assert meta and meta == _traced_on(build, (1, 3, 64, 64), "cpu")[1]


def test_arch_builds_on_meta_device():
    model, relu = _traced(*_torchvision_factory("vgg19_bn"))     # ~3 s to initialise on the CPU
    assert next(model.parameters()).is_meta and len(relu) == 18   # 16 BatchNorms + 2 Linear


class NeedsValues(nn.Module):
    """A forward that reads a value (impossible on the meta device), so tracing falls back to the CPU."""

    def __init__(self) -> None:
        super().__init__()
        self.conv, self.bn, self.relu = nn.Conv2d(3, 4, 3), nn.BatchNorm2d(4), nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if float(x.abs().sum()) < 0:
            return x
        return self.relu(self.bn(self.conv(x)))


def test_meta_failure_falls_back_to_cpu():
    model, relu = _traced(NeedsValues, (1, 3, 8, 8))
    assert not next(model.parameters()).is_meta and relu == {"bn"}

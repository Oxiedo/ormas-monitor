"""Run diff: two real Monitor runs on a tiny CNN, one healthy and one with a zeroed layer."""
import argparse
import json
import shutil
from pathlib import Path

import pytest
import torch
import torch.nn as nn

from ormas_monitor import Monitor
from ormas_monitor.commands import diff as diff_cmd
from ormas_monitor.record import diff_runs

SAMPLE = Path(__file__).resolve().parents[1] / "sample_reports" / "cifar10_with_repair.diary.jsonl"
LAYER_KEYS = {"layer", "in_a", "in_b", "alarms_a", "alarms_b", "peak_stuck_a", "peak_stuck_b",
              "repairs_a", "repairs_b", "certified_a", "certified_b"}


def small_cnn():
    torch.manual_seed(0)
    return nn.Sequential(
        nn.Conv2d(3, 8, 3, padding=1), nn.BatchNorm2d(8), nn.ReLU(),
        nn.Conv2d(8, 16, 3, padding=1), nn.BatchNorm2d(16), nn.ReLU(),
        nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(16, 4))


def make_run(out, zero=False, lr=0.05, steps=30):
    model = small_cnn()
    opt = torch.optim.SGD(model.parameters(), lr=lr)
    if zero:            # the paper's lesion: zero filter and bias, negative BatchNorm shift
        with torch.no_grad():
            model[3].weight.zero_()
            model[3].bias.zero_()
            model[4].bias.fill_(-0.1)
    mon = Monitor(model, opt, out=out, check_every=2, warmup=3, verbose=False)
    g = torch.Generator().manual_seed(1)
    for _ in range(steps):
        x, y = torch.randn(16, 3, 8, 8, generator=g), torch.randint(0, 4, (16,), generator=g)
        loss = nn.functional.cross_entropy(model(x), y)
        opt.zero_grad()
        loss.backward()
        mon.step(loss)
        opt.step()
    mon.close(report=False)
    return Path(out)


@pytest.fixture(scope="module")
def runs(tmp_path_factory):
    root = tmp_path_factory.mktemp("diff")
    return {"healthy": make_run(root / "healthy"), "zeroed": make_run(root / "zeroed", zero=True),
            "lr": make_run(root / "lr", lr=0.01)}


def test_result_shape_matches_the_contract(runs):
    r = diff_runs(runs["healthy"], runs["zeroed"])
    for side in ("a", "b"):
        assert {"path", "name", "model", "architecture_sha256", "steps", "intact"} <= set(r[side])
    assert {"same_architecture", "config_changes", "events", "layers", "verdict"} <= set(r)
    assert all(LAYER_KEYS <= set(row) for row in r["layers"])
    json.dumps(r)                                   # D renders it; it must serialise


def test_zeroed_layer_is_named_as_worse(runs):
    r = diff_runs(runs["healthy"], runs["zeroed"])
    assert r["a"]["intact"] and r["b"]["intact"] and r["same_architecture"]
    assert r["config_changes"] == []
    assert r["worse"] == ["3"] and r["better"] == []
    row = next(x for x in r["layers"] if x["layer"] == "3")
    assert row["certified_a"] == 0 and row["certified_b"] == 16 and row["peak_stuck_b"] == 16
    assert r["events"]["certificate"] == {"a": 0, "b": 1}
    assert "less healthy" in r["verdict"] and "layer 3: certified dead channels 0 -> 16" in r["verdict"]
    bn = next(x for x in r["layers"] if x["layer"] == "4")
    assert bn["peak_stuck_a"] is None             # BatchNorm is not scanned: absent, not zero


def test_reverse_diff_reports_improvement(runs):
    r = diff_runs(runs["zeroed"], runs["healthy"])
    assert r["better"] == ["3"] and r["worse"] == []
    assert "healthier" in r["verdict"]


def test_same_run_has_no_changes(runs):
    r = diff_runs(runs["healthy"], runs["healthy"])
    assert r["config_changes"] == [] and r["worse"] == r["better"] == r["mixed"] == []
    assert not any(row["changed"] for row in r["layers"])
    assert r["verdict"] == ("Same architecture. No optimizer or monitor settings changed. "
                            "No layer's alarms, stuck channels or certified dead channels changed.")


def test_config_change_is_reported(runs):
    r = diff_runs(runs["healthy"], runs["lr"])
    assert {"key": "optimizer.lr", "a": 0.05, "b": 0.01} in r["config_changes"]
    assert "1 setting changed: optimizer.lr 0.05 -> 0.01" in r["verdict"]


def test_tampered_record_leads_the_verdict(runs, tmp_path):
    bad = Path(shutil.copytree(runs["zeroed"], tmp_path / "bad"))
    rows = (bad / "diary.jsonl").read_text().splitlines()
    (bad / "diary.jsonl").write_text("\n".join(rows[:-1]) + "\n")
    r = diff_runs(runs["healthy"], bad)
    assert r["b"]["intact"] is False and "bookends" in r["b"]["failed_checks"]
    assert r["verdict"].startswith("Run B's record failed verification")


def test_diary_only_records_can_be_compared():
    r = diff_runs(SAMPLE, SAMPLE)
    assert r["a"]["metrics"] is False and r["a"]["steps"] == 5083
    assert r["worse"] == [] and r["events"]["repair"] == {"a": 2, "b": 2}


def test_cli_diff_text_and_json(runs, capsys):
    ap = argparse.ArgumentParser()
    diff_cmd.register(ap.add_subparsers(dest="cmd", required=True))
    a = ap.parse_args(["diff", str(runs["healthy"]), str(runs["zeroed"])])
    assert a.func(a) == 0
    out = capsys.readouterr().out
    assert "less healthy" in out and "0->16" in out
    a = ap.parse_args(["diff", str(runs["healthy"]), str(runs["zeroed"]), "--json"])
    assert a.func(a) == 0
    assert json.loads(capsys.readouterr().out)["worse"] == ["3"]


def test_cli_diff_out_uses_the_html_writer_when_present(runs, tmp_path, monkeypatch, capsys):
    from ormas_monitor import report
    seen = {}

    def fake_writer(result, path):
        seen["verdict"] = result["verdict"]
        Path(path).write_text("<html></html>")
        return Path(path)
    monkeypatch.setattr(report, "write_diff_report", fake_writer, raising=False)
    ap = argparse.ArgumentParser()
    diff_cmd.register(ap.add_subparsers(dest="cmd", required=True))
    a = ap.parse_args(["diff", str(runs["healthy"]), str(runs["zeroed"]), "--out", str(tmp_path / "d.html")])
    assert a.func(a) == 0
    assert "less healthy" in seen["verdict"] and (tmp_path / "d.html").exists()

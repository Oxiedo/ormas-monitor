"""Run diff: two real Monitor runs on a tiny CNN, one healthy and one with a zeroed layer."""
import argparse
import copy
import json
import shutil
from pathlib import Path

import pytest
import torch
import torch.nn as nn

from ormas_monitor import Monitor
from ormas_monitor.commands import diff as diff_cmd
from ormas_monitor.record import diff_runs

SAMPLE = Path(__file__).resolve().parent / "fixtures" / "v0_1_cifar10_with_repair.diary.jsonl"
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
    assert r["verdict"] == ("Run B is less healthy in 1 layer (layer 3: proven-stuck 0 to 16, "
                            "certified dead 0 to 16).")
    assert row["health"] == "worse" and row["proven_at_end_b"] == 16
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
    assert r["verdict"] == "No layer's alarms, stuck channels or certified dead channels changed."
    assert r["verdict_detail"] == "Same architecture. No optimizer or monitor settings changed."


def test_config_change_is_reported(runs):
    r = diff_runs(runs["healthy"], runs["lr"])
    assert {"key": "optimizer.lr", "a": 0.05, "b": 0.01} in r["config_changes"]
    assert "1 setting changed: optimizer.lr 0.05 -> 0.01" in r["verdict_detail"]


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
    assert "less healthy" in out and "0->16" in out and "Same architecture." in out
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


# ------------------------------------------------------------------ watch-only vs repair (V2 fresh-user item 3)
class DemoCNN(nn.Module):
    """A miniature of the demo's StandardCNN: conv1-3 blocks and a two-layer classifier."""

    def __init__(self):
        super().__init__()

        def block(i, o):
            return nn.Sequential(nn.Conv2d(i, o, 3, padding=1), nn.BatchNorm2d(o), nn.ReLU(), nn.MaxPool2d(2))
        self.conv1, self.conv2, self.conv3 = block(3, 8), block(8, 16), block(16, 32)
        self.classifier = nn.Sequential(nn.Linear(32, 64), nn.ReLU(), nn.Linear(64, 4))

    def forward(self, x):
        return self.classifier(self.conv3(self.conv2(self.conv1(x))).flatten(1))


def _batch(g):
    x = torch.randn(32, 3, 8, 8, generator=g)
    return x, (x[:, 0].mean((1, 2)) > 0).long() + 2 * (x[:, 1].mean((1, 2)) > 0).long()


@pytest.fixture(scope="module")
def demo_arms(tmp_path_factory):
    """The demo's experiment in miniature: train, then from the same state zero conv2 and conv3 and
    clear the optimizer, once watch-only and once with repair."""
    root = tmp_path_factory.mktemp("arms")
    torch.manual_seed(0)
    model = DemoCNN()
    opt = torch.optim.AdamW(model.parameters(), lr=3e-3, weight_decay=1e-2)
    g = torch.Generator().manual_seed(0)
    for _ in range(40):
        x, y = _batch(g)
        loss = nn.functional.cross_entropy(model(x), y)
        opt.zero_grad()
        loss.backward()
        opt.step()
    snap = copy.deepcopy(model.state_dict()), copy.deepcopy(opt.state_dict())
    for arm, repair in (("watch", False), ("repair", True)):
        m = DemoCNN()
        m.load_state_dict(snap[0])
        o = torch.optim.AdamW(m.parameters(), lr=3e-3, weight_decay=1e-2)
        o.load_state_dict(snap[1])
        g = torch.Generator().manual_seed(1)
        mon = Monitor(m, o, out=root / arm, repair=repair, check_every=2, warmup=3, verbose=False)
        for step in range(60):
            if step == 10:
                with torch.no_grad():
                    for blk in (m.conv2, m.conv3):
                        blk[0].weight.zero_()
                        blk[0].bias.zero_()
                o.state.clear()
            x, y = _batch(g)
            loss = nn.functional.cross_entropy(m(x), y)
            o.zero_grad()
            loss.backward()
            mon.step(loss)
            o.step()
        mon.close(report=False)
    return root / "watch", root / "repair"


def test_repair_run_is_healthier_even_when_a_layer_has_more_silent_units(demo_arms):
    r = diff_runs(*demo_arms)
    assert r["better"] == ["conv2.0", "conv3.0"] and r["worse"] == [] and r["mixed"] == []
    assert r["more_silent"] == ["classifier.0"] and r["fewer_silent"] == []
    rows = {row["layer"]: row for row in r["layers"]}
    for name in ("conv2.0", "conv3.0"):
        row = rows[name]
        assert row["health"] == "better" and row["proven_at_end_a"] > 0 and row["proven_at_end_b"] == 0
        assert row["channels_repaired_a"] == 0 and row["channels_repaired_b"] > 0
    cls = rows["classifier.0"]
    assert cls["health"] == "same" and cls["silent_change"] == "more"
    assert cls["silent_at_end_b"] > cls["silent_at_end_a"] and cls["proven_at_end_a"] == cls["proven_at_end_b"] == 0
    c2, c3 = rows["conv2.0"], rows["conv3.0"]
    assert r["verdict"] == (
        f"Run B is healthier in 2 layers (conv2.0: proven-stuck {c2['proven_at_end_a']} to 0, "
        f"{c2['channels_repaired_b']} repaired; conv3.0: proven-stuck {c3['proven_at_end_a']} to 0, "
        f"{c3['channels_repaired_b']} repaired) and has more silent channels in 1 (classifier.0: silent "
        f"{cls['silent_at_end_a']} to {cls['silent_at_end_b']}, unproven).")
    assert "less healthy" not in r["verdict"]
    assert "config.repair False -> True" in r["verdict_detail"] and "Repairs: 0 (0 channels) in run A" in r["verdict_detail"]
    assert "never make a layer less healthy" in r["verdict_detail"]


def test_reverse_watch_vs_repair_names_the_proven_layers_as_worse(demo_arms):
    r = diff_runs(demo_arms[1], demo_arms[0])
    assert r["worse"] == ["conv2.0", "conv3.0"] and r["fewer_silent"] == ["classifier.0"]
    assert r["verdict"].startswith("Run B is less healthy in 2 layers (conv2.0: proven-stuck 0 to ")
    assert "and has fewer silent channels in 1 (classifier.0" in r["verdict"]


def test_silent_only_change_is_not_called_less_healthy(demo_arms, tmp_path):
    """Equal proven evidence, more silent units: a note, never 'less healthy'."""
    watch = Path(shutil.copytree(demo_arms[0], tmp_path / "w"))
    rows = [json.loads(t) for t in (watch / "metrics.jsonl").read_text().splitlines()]
    silent = rows[-1]["stuck"]["classifier.0"][2]
    rows[-1]["stuck"]["classifier.0"][2] += 5
    (watch / "metrics.jsonl").write_text("".join(json.dumps(x) + "\n" for x in rows))
    r = diff_runs(demo_arms[0], watch)
    assert r["worse"] == r["better"] == r["mixed"] == [] and r["more_silent"] == ["classifier.0"]
    assert r["verdict"] == ("Run B's record failed verification (metrics), so this comparison cannot be trusted. "
                            "No layer changed on proven evidence; run B has more silent channels in 1 layer "
                            f"(classifier.0: silent {silent} to {silent + 5}, unproven).")


def test_malformed_metrics_rows_do_not_crash_the_diff(runs, tmp_path):
    bad = Path(shutil.copytree(runs["zeroed"], tmp_path / "bad"))
    with open(bad / "metrics.jsonl", "a") as fh:
        for row in ({"grad": [1, 2], "stuck": "x"}, {"stuck": {"3": ["a", None]}}, {"grad": {"3": 1}, "stuck": [[]]},
                    {"stuck": {"3": {"c": 10 ** 400}}}):
            fh.write(json.dumps(row) + "\n")
        fh.write("[" * 100_000 + "\n")
    r = diff_runs(runs["healthy"], bad)
    assert r["b"]["intact"] is False and r["verdict"].startswith("Run B's record failed verification")

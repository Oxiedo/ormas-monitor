"""Report rendering from records and results that were not made by today's Monitor: the v0.1
sample diary (no metrics file), synthetic old-format rows, unknown events, a synthetic run diff,
a real `diff_runs` output, and checkpoint results with unknown finding kinds."""
from __future__ import annotations

import json
import shutil
from html import escape
from pathlib import Path

import pytest
import torch
import torch.nn as nn

from ormas_monitor import Monitor
from ormas_monitor.diary import Diary
from ormas_monitor.report import analyse, write_all, write_check_report, write_diff_report

SAMPLE = Path(__file__).resolve().parents[1] / "sample_reports" / "cifar10_with_repair.diary.jsonl"


@pytest.mark.skipif(not SAMPLE.exists(), reason="sample record not in this checkout")
def test_v01_sample_diary_without_metrics(tmp_path):
    run = tmp_path / "sample"
    run.mkdir()
    shutil.copy(SAMPLE, run / "diary.jsonl")
    paths = write_all(run)
    f = analyse(run)
    assert not f.has_metrics and f.trust.intact and f.total_steps == 5083
    # stuck from the certificate check (400) until the repair check (450), rebuilt from events alone
    assert f.cost.stuck_steps == 50 and f.cost.spans == ((400, 450),)
    assert f.cost.stuck_seconds is not None and 0 < f.cost.stuck_seconds < 10
    lead = f.leads[0]
    assert lead.spike["step"] == 400 and lead.signal["step"] == 150 and lead.steps_earlier == 250
    assert [ld.signal for ld in f.leads[1:]] == [None, None]          # later spikes: nothing earlier
    md = paths["summary_md"].read_text(encoding="utf-8")
    assert "PROBLEM FOUND AND REPAIRED" in md
    assert "192 channels were proven dead" in md and "repaired 192 channels in 2 layers (at step 450)" in md
    assert "jumped 3 times, first at step 400" in md
    assert "the measurements file that belongs with it is missing" in md
    html = paths["report"].read_text(encoding="utf-8")
    assert "Lead time before loss spikes" in html and "250 steps" in html
    assert "No per-check loss values" in html and "no <span class='mono'>manifest</span> event" in html


def _diary(run: Path, events: list[tuple[str, dict]]) -> None:
    d = Diary(run / "diary.jsonl")
    for name, fields in events:
        d.write(name, **fields)
    d.close()


def test_old_rows_unknown_events_and_no_run_end(tmp_path):
    run = tmp_path / "old"
    run.mkdir()
    _diary(run, [
        ("run_start", {"name": "old", "model": "Net", "config": {"check_every": 10, "repair": False}}),
        ("structure", {"step": 0, "pairs": {"c": "b"}, "relu_after": ["b"]}),
        ("channels", {"step": 10, "severity": "warning", "layer": "c", "batchnorm": "b", "channels": 4,
                      "level": "some", "certified": [0], "flat": [], "observed": []}),   # no n_* fields
        ("sink_error", {"sink": "<webhook>", "error": "ConnectionError: refused"}),
        ("skipped_step", {"step": 12}), ("skipped_step", {"step": 13}),
        ("quantum_flux", {"step": 15, "severity": "warning", "layer": "c", "flux": [1, 2]}),
    ])
    rows = [{"step": 0, "loss": 1.0, "grad": {"c": 0.5}, "stuck": {"c": [0, 0, 0]}},
            {"step": 10, "loss": 0.9, "grad": {"c": 0.4}, "stuck": {"c": [1, 0, 0]}},   # 3-element v0.1 rows
            {"step": 20, "loss": 0.8, "grad": {"c": 0.4}, "stuck": {"c": [1, 0, 2]}}]
    (run / "metrics.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    paths = write_all(run)
    f = analyse(run)
    assert not f.finished and f.total_steps == 21
    assert f.cost.stuck_steps == 11 and f.layer_peaks["c"].silent == 2 and f.layer_peaks["c"].decayed == 0
    html = paths["report"].read_text(encoding="utf-8")
    assert "quantum flux: flux=[1, 2]" in html                    # unknown event, rendered generically
    assert "2 optimizer steps skipped" in html and "ConnectionError" in html
    assert "no <span class='mono'>run_end</span> event" in html
    md = paths["summary_md"].read_text(encoding="utf-8")
    assert "The record has no end-of-run entry" in md and "1 output plug-in failed" in md


def test_diff_report_synthetic(tmp_path):
    d = {"a": {"path": "runs/a", "name": "approved", "model": "Net", "architecture_sha256": "ab" * 32,
               "steps": 100, "intact": True, "complete": True, "sealed": True, "signed": False},
         "b": {"path": "runs/b", "name": "proposed", "model": "Net", "architecture_sha256": "ab" * 32,
               "steps": 120, "intact": False, "failed_checks": ["chain"]},
         "same_architecture": True,
         "config_changes": [{"key": "optimizer.lr", "a": 0.001, "b": 0.01}],
         "environment_changes": [{"key": "torch", "a": "2.10", "b": "2.11"}],
         "events": {"alarm": {"a": 0, "b": 3}, "repair": {"a": 1, "b": 1}},
         "layers": [{"layer": "conv1", "in_a": True, "in_b": True, "alarms_a": 0, "alarms_b": 3,
                     "peak_stuck_a": None, "peak_stuck_b": None, "certified_a": None, "certified_b": None,
                     "repairs_a": 0, "repairs_b": 0, "changed": True},
                    {"layer": "fc", "in_a": True, "in_b": False, "alarms_a": 0, "alarms_b": None}],
         "worse": ["conv1"], "better": [], "mixed": [],
         "verdict": "Run B's record failed verification.", "brand_new_key": {"x": 1}}
    html = write_diff_report(d, tmp_path / "diff.html").read_text(encoding="utf-8")
    assert "Record failed verification" in html and "FAILED: chain" in html
    assert "optimizer.lr" in html and "0.01" in html and "torch" in html
    assert "<b>0 → 3</b>" in html and "worse in B" in html and "only in A" in html
    assert "brand_new_key" in html                                  # unknown keys are shown, not dropped
    assert "None" not in html.split("<h2>Layers</h2>")[1].split("<h2>Events")[0]   # None renders as –


def _tiny_run(out: Path, repair: bool) -> Path:
    torch.manual_seed(0)
    model = nn.Sequential(nn.Conv2d(3, 4, 3, padding=1), nn.BatchNorm2d(4), nn.ReLU(),
                          nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(4, 2))
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    mon = Monitor(model, opt, out=out, repair=repair, check_every=2, patience=2, verbose=False)
    g = torch.Generator().manual_seed(1)
    for i in range(30):
        if i == 10:
            with torch.no_grad():
                model[0].weight.zero_()
                model[0].bias.zero_()
                model[1].bias.fill_(-0.1)
            opt.state.clear()
        loss = nn.functional.cross_entropy(model(torch.randn(8, 3, 6, 6, generator=g)),
                                           torch.randint(0, 2, (8,), generator=g))
        opt.zero_grad()
        loss.backward()
        mon.step(loss)
        opt.step()
    mon.close(report=False)
    return out


def test_diff_report_from_real_diff_runs(tmp_path):
    diff = pytest.importorskip("ormas_monitor.record.diff")
    a = _tiny_run(tmp_path / "watch", repair=False)
    b = _tiny_run(tmp_path / "repair", repair=True)
    result = diff.diff_runs(a, b)
    html = write_diff_report(result, tmp_path / "diff.html").read_text(encoding="utf-8")
    assert result["verdict"].split(".")[0] in html
    assert "config.repair" in html and "Run A" in html and "Run B" in html
    assert "<span class='mono'>0</span>" in html and "Repairs" in html


def test_check_report_known_and_unknown_kinds(tmp_path):
    result = {
        "file": "/x/model.safetensors", "tensors": 10, "parameters": 1234, "batchnorm_layers": 1,
        "findings": [
            {"severity": "warning", "kind": "dead_channels", "layer": "conv", "batchnorm": "bn",
             "title": "2/8 channels of conv are dead for good", "detail": "Theorem 2."},
            {"severity": "info", "kind": "attention_entropy_collapse", "title": "head 3 attends to one token",
             "detail": "an estimate"},
            {"severity": "notice", "kind": "brand_new"},                  # unknown severity and no title
            {"severity": "warning", "kind": "norm_scale_zero", "title": "LayerNorm scale is zero",
             "detail": "x", "tensors": [{"tensor": "ln.weight", "zero": 3}, {"tensor": "ln2.weight", "zero": 1}]},
        ],
        "layers": [{"layer": "conv", "batchnorm": "bn", "channels": 8, "certified": [0, 1], "flat": [],
                    "decayed": [3], "dormant": [4, 5], "collapsed_var": 0, "frozen": [6]}],
        "against": "/x/before.safetensors",
        "diff": {"changed": [{"tensor": "conv.weight", "change": 0.4, "abs_change": 1.0},
                             {"tensor": "bn.bias", "change": 0.0, "abs_change": 0.0}],
                 "unchanged": ["bn.bias"], "added": [], "removed": [], "reshaped": [],
                 "newly_dead": [{"layer": "conv", "batchnorm": "bn", "count": 2}]},
    }
    html = write_check_report(result, tmp_path / "c.html").read_text(encoding="utf-8")
    assert "2 problems found in the weights" in html and "ln2.weight" in html
    assert "head 3 attends to one token" in html and "brand new" in html and "Notice" in html
    assert "Decayed to ~0" in html and "Dormant" in html and "frozen" in html
    body = html.split("Largest relative changes")[1]
    assert "conv.weight" in body and "bn.bias" not in body      # unchanged tensors are not "changes"
    assert "Channels that died between the two files" in html


def test_check_report_from_real_checkpoint(tmp_path):
    from ormas_monitor.checkpoint import run as check
    torch.manual_seed(0)
    model = nn.Sequential(nn.Conv2d(3, 8, 3), nn.BatchNorm2d(8), nn.ReLU())
    torch.save(model.state_dict(), tmp_path / "before.pt")
    with torch.no_grad():
        model[0].weight[:3] = 0
        model[0].bias[:3] = 0
        model[1].bias[:3] = -0.5
    torch.save(model.state_dict(), tmp_path / "after.pt")
    alone = check(tmp_path / "after.pt")
    html = write_check_report(alone, tmp_path / "alone.html").read_text(encoding="utf-8")
    assert "Checkpoint health report" in html and "What changed" not in html
    assert "3/8 channels" in html
    if "wiring" in alone:
        assert "How the wiring was determined" in html and "ReLU follows" in html
    for note in alone.get("notes") or []:
        assert escape(note) in html
    both = check(tmp_path / "after.pt", tmp_path / "before.pt")
    html = write_check_report(both, tmp_path / "both.html").read_text(encoding="utf-8")
    assert "What changed since before.pt" in html and "Channels that died between the two files" in html

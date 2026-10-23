"""Reports never claim more than the record shows: a run whose checks saw no gradients is NOT
MONITORED, a run with no check is UNKNOWN, a repaired run with silent classifier units says so,
new core events (`no_gradients`, `certificate_refused`) are rendered, layers are named with their
module type, the diff banner is the diff's own verdict, and each page's footer names its source."""
from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch
import torch.nn as nn

from ormas_monitor import Monitor
from ormas_monitor.diary import Diary
from ormas_monitor.report import analyse, write_all, write_check_report, write_diff_report

STEPS, EVERY, LESION_AT = 60, 5, 30


def _net() -> nn.Sequential:
    torch.manual_seed(0)
    return nn.Sequential(nn.Conv2d(3, 8, 3, padding=1), nn.BatchNorm2d(8), nn.ReLU(),
                         nn.Conv2d(8, 16, 3, padding=1), nn.BatchNorm2d(16), nn.ReLU(),
                         nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(16, 32), nn.ReLU(),
                         nn.Linear(32, 4))


def _train(out: Path, *, repair: bool = False, lesion: bool = False, step_first: bool = False,
           steps: int = STEPS) -> Path:
    model = _net()
    with torch.no_grad():                   # classifier hidden units that never fire: silent, not proven
        model[8].bias[:12] = -50.0
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    mon = Monitor(model, opt, out=out, repair=repair, check_every=EVERY, patience=2, verbose=False,
                  name=out.name)
    g = torch.Generator().manual_seed(1)
    for i in range(steps):
        if lesion and i == LESION_AT:
            with torch.no_grad():
                model[3].weight.zero_()
                model[3].bias.zero_()
                model[4].bias.fill_(-0.1)
            opt.state.clear()
        loss = nn.functional.cross_entropy(model(torch.randn(16, 3, 8, 8, generator=g)),
                                           torch.randint(0, 4, (16,), generator=g))
        opt.zero_grad()
        if step_first:                      # the fresh-user mistake: step() before backward()
            mon.step(loss)
            loss.backward()
        else:
            loss.backward()
            mon.step(loss)
        opt.step()
    mon.close()
    return out


@pytest.fixture(scope="module")
def runs(tmp_path_factory) -> dict[str, Path]:
    root = tmp_path_factory.mktemp("coverage")
    return {"mistake": _train(root / "mistake", step_first=True),
            "empty": _train(root / "empty", steps=0),
            "watch": _train(root / "watch", lesion=True),
            "repair": _train(root / "repair", repair=True, lesion=True)}


def _read(run: Path, name: str) -> str:
    return (run / name).read_text(encoding="utf-8")


def _diary(run: Path, events: list[tuple[str, dict]]) -> None:
    run.mkdir(parents=True, exist_ok=True)
    d = Diary(run / "diary.jsonl")
    for name, fields in events:
        d.write(name, **fields)
    d.close()


def test_step_before_backward_is_not_monitored(runs):
    f = analyse(runs["mistake"])
    assert f.coverage.state == "blind" and f.coverage.gradients_seen is False
    md = _read(runs["mistake"], "summary.md")
    n = STEPS // EVERY
    assert f"**NOT MONITORED: No gradients were seen at any of the {n} checks; call mon.step() after " \
           f"loss.backward().**" in md
    assert "HEALTHY" not in md and "Nothing measurable" not in md
    assert "Every layer's learning signal stayed within its own normal range" not in md
    assert "The usual cause is calling mon.step() before loss.backward()" in md
    html = _read(runs["mistake"], "report.html")
    assert "<span class='status'>Not monitored</span>" in html and ">Healthy<" not in html.split("</div>")[0]


def test_zero_checks_is_unknown(runs):
    f = analyse(runs["empty"])
    assert f.coverage.state == "unknown"
    md = _read(runs["empty"], "summary.md")
    assert "**UNKNOWN: The monitor made no check during this run" in md and "HEALTHY" not in md
    assert "<span class='status'>Unknown</span>" in _read(runs["empty"], "report.html")


def test_repaired_run_names_silent_units_and_claims_only_proven(runs):
    f = analyse(runs["repair"])
    reps = f.of("repair")
    assert reps and not any(ep.open_at_end for ep in f.episodes)
    assert f.silent_at_end.get("8", 0) >= 12          # the 12 units forced off, maybe a few more
    md = _read(runs["repair"], "summary.md")
    last = max(ep.end for ep in f.episodes)
    silent = sum(f.silent_at_end.values())
    assert (f"the monitor repaired {sum(e['count'] for e in reps)} channels by step "
            f"{max(e['step'] for e in reps)}. After step {last}, no check found a channel proven dead, "
            f"constant or decayed to ~0. {silent} channels were still silent at the end (likely unused, "
            f"not proven dead).**") in md
    assert (f"At the last check (step {f.last_check}), {f.silent_at_end['8']} channels were silent in "
            f"layer `8` (Linear)") in md
    assert "no part was stuck" not in md and "no layer was stuck" not in md.lower()
    assert "**Silent**: " in md                                     # glossary entry
    html = _read(runs["repair"], "report.html")
    assert "channels silent at the last check (unproven)" in html
    assert "<span class='mono'>8</span> <span class='sub small'>(Linear)</span>" in html
    assert "<code>8</code> (Linear)" in _read(runs["repair"], "summary.html")


def test_partly_monitored(tmp_path):
    run = tmp_path / "partial"
    _diary(run, [("run_start", {"name": "partial", "model": "Net", "config": {"check_every": 10}}),
                 ("run_end", {"steps": 30, "checks": 3, "counts": {}, "metrics_sha256": None})])
    rows = [{"step": 0, "loss": 1.0, "grad": {}, "stuck": {}},
            {"step": 10, "loss": 0.9, "grad": {"c": 0.4}, "stuck": {}},
            {"step": 20, "loss": 0.8, "grad": {"c": 0.4}, "stuck": {}}]
    (run / "metrics.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    md = write_all(run)["summary_md"].read_text(encoding="utf-8")
    assert "**PARTLY MONITORED: No problems found, but 1 of 3 checks saw no gradients" in md
    assert "(first at step 0)" in md


def test_new_core_events_without_metrics(tmp_path):
    run = tmp_path / "events"
    _diary(run, [
        ("run_start", {"name": "events", "model": "Net", "config": {"check_every": 10}}),
        ("manifest", {"step": 0, "layers": {"monitored": 2, "total": 2, "filter": None},
                      "producers": [{"layer": "3", "kind": "Conv2d", "batchnorm": "4"}]}),
        ("no_gradients", {"step": 0, "severity": "warning"}),
        ("certificate_refused", {"step": 10, "severity": "info", "layer": "3", "batchnorm": "4", "count": 5,
                                 "reason": "the optimizer holds momentum for these channels"}),
        ("certificate_refused", {"step": 20, "severity": "info", "layer": "3", "batchnorm": "4", "count": 6,
                                 "reason": "the optimizer holds momentum for these channels"}),
        ("warp_drive", {"step": 20, "severity": "info", "layer": "3", "factor": 9}),
        ("run_end", {"steps": 30, "checks": 3, "counts": {"checks_without_gradients": 3},
                     "metrics_sha256": None}),
    ])
    paths = write_all(run)
    f = analyse(run)
    assert f.coverage.state == "blind" and f.kind("3") == "Conv2d"
    md = paths["summary_md"].read_text(encoding="utf-8")
    assert "NOT MONITORED" in md and "HEALTHY" not in md
    assert "In layer `3` (Conv2d), channels looked dead from their weights, but a condition of the proof" in md
    html = paths["report"].read_text(encoding="utf-8")
    section = html.split("Why some layers could not be certified</h2>")[1].split("<h2>")[0]
    assert "10–20" in section and "the optimizer holds momentum for these channels" in section
    assert "<td class='num'>6</td>" in section                      # the most at one check
    assert "2 certificates refused in 1 layer" in html               # folded into one event row
    assert "no watched layer had a gradient at this check" in html
    assert "warp drive: factor=9" in html                            # unknown events stay generic


def test_footer_names_each_page_source(runs, tmp_path):
    assert "from the run&#x27;s own record" in _read(runs["watch"], "report.html")
    result = {"file": "/x/model.safetensors", "findings": [], "layers": []}
    check = write_check_report(result, tmp_path / "c.html").read_text(encoding="utf-8")
    assert "own record" not in check and "no training record is involved" in check
    diff = write_diff_report({"a": {}, "b": {}, "verdict": "x"}, tmp_path / "d.html").read_text(encoding="utf-8")
    assert "own record" not in diff and "the two runs&#x27; records" in diff


def test_diff_banner_is_the_diffs_own_verdict(tmp_path):
    d = {"a": {"name": "watch", "intact": True}, "b": {"name": "repair", "intact": True},
         "layers": [{"layer": "3", "in_a": True, "in_b": True, "health": "better", "silent_change": None,
                     "peak_stuck_a": 16, "peak_stuck_b": 16, "proven_at_end_a": 16, "proven_at_end_b": 0,
                     "changed": True},
                    {"layer": "8", "in_a": True, "in_b": True, "health": "same", "silent_change": "more",
                     "silent_at_end_a": 18, "silent_at_end_b": 19, "changed": True}],
         "worse": [], "better": ["3"], "mixed": [], "more_silent": ["8"], "fewer_silent": [],
         "verdict": "Run B is healthier in 1 layer.", "verdict_detail": "Same architecture. Silent never counts."}
    html = write_diff_report(d, tmp_path / "d.html").read_text(encoding="utf-8")
    assert "<span class='status'>Comparison</span><b class='head'>Run B is healthier in 1 layer.</b>" in html
    assert "<p>Same architecture. Silent never counts.</p>" in html
    assert "better in B" in html and "more silent in B" in html and "Silent at end" in html
    assert "<b>18 → 19</b>" in html and "more_silent" not in html     # known keys are not dumped as extras
    broken = dict(d, b={"name": "repair", "intact": False, "failed_checks": ["chain"]})
    html = write_diff_report(broken, tmp_path / "e.html").read_text(encoding="utf-8")
    assert "Record failed verification" in html

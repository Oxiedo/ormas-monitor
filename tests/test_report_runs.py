"""Reports built from real monitored runs: a healthy run, a run with a zeroed layer in watch-only
mode, and the same lesion with repair. Asserts the executive summary's key sentences and numbers."""
from __future__ import annotations

import json
import re
from pathlib import Path

import pytest
import torch
import torch.nn as nn

from ormas_monitor import Monitor
from ormas_monitor.cli import main as cli
from ormas_monitor.report import analyse, write_all

STEPS, LESION_AT, EVERY = 120, 60, 5


def _cnn() -> nn.Sequential:
    torch.manual_seed(0)
    return nn.Sequential(
        nn.Conv2d(3, 8, 3, padding=1), nn.BatchNorm2d(8), nn.ReLU(),
        nn.Conv2d(8, 16, 3, padding=1), nn.BatchNorm2d(16), nn.ReLU(),
        nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(16, 4))


def _train(out: Path, repair: bool, lesion: bool) -> Path:
    model = _cnn()
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    mon = Monitor(model, opt, out=out, repair=repair, check_every=EVERY, patience=2, verbose=False,
                  name=out.name)
    g = torch.Generator().manual_seed(1)
    for i in range(STEPS):
        if lesion and i == LESION_AT:          # the paper's lesion: layer zeroed, optimizer state cleared
            with torch.no_grad():
                model[3].weight.zero_()
                model[3].bias.zero_()
                model[4].bias.fill_(-0.1)
            opt.state.clear()
        x = torch.randn(16, 3, 8, 8, generator=g)
        y = torch.randint(0, 4, (16,), generator=g)
        loss = nn.functional.cross_entropy(model(x), y)
        opt.zero_grad()
        loss.backward()
        mon.step(loss)
        opt.step()
    mon.close()
    return out


@pytest.fixture(scope="module")
def runs(tmp_path_factory) -> dict[str, Path]:
    root = tmp_path_factory.mktemp("runs")
    return {"healthy": _train(root / "healthy", False, False),
            "watch": _train(root / "watch", False, True),
            "repair": _train(root / "repair", True, True)}


def _md(run: Path) -> str:
    return (run / "summary.md").read_text(encoding="utf-8")


def test_close_writes_all_reports(runs):
    for run in runs.values():
        for name in ("report.html", "summary.html", "summary.md"):
            assert (run / name).stat().st_size > 1000, name


def test_healthy_summary(runs):
    md = _md(runs["healthy"])
    assert "**HEALTHY: No problems found in 120 training steps.**" in md
    assert "Nothing measurable: at no check was any part of the model stuck." in md
    assert "Nothing needed doing. The monitor ran in watch-only mode" in md
    assert "Partly: all 4 entries of the record check out" in md     # run_start, structure, manifest, run_end
    assert "it is not signed" in md
    assert "$" not in md                                              # no price given: no dollars


def test_broken_watch_only_summary(runs):
    md = _md(runs["watch"])
    f = analyse(runs["watch"])
    assert "PROBLEM, NOT REPAIRED" in md
    assert f"Part of the model stopped learning (detected at step {LESION_AT}) and was still stuck" in md
    assert "At step 60 it found stuck channels in layer `3` (Conv2d); 16 channels were proven dead" in md
    assert "<code>3</code> (Conv2d)" in (runs["watch"] / "summary.html").read_text(encoding="utf-8")
    assert "They were still stuck when the run ended." in md
    # the zeroed layer is stuck from the check that saw it until the last step
    assert f.cost.stuck_steps == STEPS - LESION_AT and f.cost.proven_steps == STEPS - LESION_AT
    assert f"{STEPS - LESION_AT} of {STEPS} training steps (50%) ran with part of the model stuck." in md
    assert f.cost.stuck_seconds is not None and 0 < f.cost.stuck_seconds <= f.cost.total_seconds
    assert "Nothing. The monitor ran in watch-only mode: it recorded the problem but did not change" in md
    ep = f.episodes[0]
    assert (ep.layer, ep.start, ep.end, ep.open_at_end, ep.peak.certified) == ("3", 60, 120, True, 16)


def test_broken_repair_summary(runs):
    md = _md(runs["repair"])
    f = analyse(runs["repair"])
    rep = next(e for e in f.events if e["event"] == "repair")
    assert "PROBLEM FOUND AND REPAIRED" in md
    reps = [e for e in f.events if e["event"] == "repair"]
    last = max(ep.end for ep in f.episodes)
    assert (f"the monitor repaired {sum(e['count'] for e in reps)} channels by step "
            f"{max(e['step'] for e in reps)}. After step {last}, no check found a channel proven dead, "
            f"constant or decayed to ~0.") in md
    assert "no part was stuck" not in md and "no layer was stuck" not in md.lower()
    assert f"The monitor repaired {rep['count']} channels in 1 layer (at step {rep['step']})" in md
    # stuck from detection until the repair at a later check
    assert f.cost.stuck_steps == rep["step"] - LESION_AT > 0
    assert not any(ep.open_at_end for ep in f.episodes)
    html = (runs["repair"] / "report.html").read_text(encoding="utf-8")
    if "bound" in rep:                                 # v0.2 records: the change and its (absent) bound
        assert "No limit was enforced on how much a repair could change the layer." in md
        assert "no bound on the size of the change was enforced" in html
    if rep.get("relative_change") is not None:
        assert "layer weights changed by" in html


def test_dollar_estimate_only_with_price(runs, tmp_path):
    paths = write_all(runs["watch"], gpu_hour_cost=3.0, gpus=4)
    assert set(paths) >= {"report", "summary", "summary_md"}
    md = paths["summary_md"].read_text(encoding="utf-8")
    f = analyse(runs["watch"])
    hours = f.cost.stuck_seconds / 3600
    assert "**ESTIMATE:** " in md and "× 4 GPUs × $3.00 per GPU-hour" in md and f"{hours:.3f} h stuck" in md
    assert "not measured" in md
    html = paths["summary"].read_text(encoding="utf-8")
    assert ">ESTIMATE<" in html
    write_all(runs["watch"])                     # back to no price: the dollar line disappears
    assert "ESTIMATE" not in _md(runs["watch"])


def test_cli_cost_flags(runs):
    assert cli(["report", str(runs["watch"]), "--gpu-hour-cost", "2", "--gpus", "1"]) == 0
    assert "× 1 GPU × $2.00 per GPU-hour" in _md(runs["watch"])
    with pytest.raises(SystemExit):
        cli(["report", str(runs["watch"]), "--gpu-hour-cost", "-1"])
    assert cli(["report", str(runs["watch"])]) == 0


def test_no_compliance_claim_and_not_list(runs):
    for run in runs.values():
        md = _md(run)
        html = (run / "summary.html").read_text(encoding="utf-8")
        assert "It does not make anyone compliant with the EU AI Act" in md
        assert "What this report does not tell you" in html
        lowered = md.lower().replace("does not make anyone compliant", "")
        assert "compliant" not in lowered and "certified compliant" not in lowered


def test_engineer_report_sections(runs):
    html = (runs["watch"] / "report.html").read_text(encoding="utf-8")
    for section in ("Time spent stuck", "What happened", "Layers", "How the monitor was set up", "The record"):
        assert section in html, section
    assert "certified dead" in html and "still stuck at end" in html


def test_tampered_record_says_so(runs, tmp_path):
    run = tmp_path / "tampered"
    run.mkdir()
    lines = (runs["watch"] / "diary.jsonl").read_text().splitlines()
    rec = json.loads(lines[2])
    rec["time"] = rec["time"] + 1
    lines[2] = json.dumps(rec)
    (run / "diary.jsonl").write_text("\n".join(lines) + "\n")
    (run / "metrics.jsonl").write_bytes((runs["watch"] / "metrics.jsonl").read_bytes())
    write_all(run)
    md = _md(run)
    assert "RECORD ALTERED" in md and md.count("No: the record has been altered or damaged") == 1


def test_pages_are_offline_and_printable(runs):
    for run in runs.values():
        for name in ("report.html", "summary.html"):
            html = (run / name).read_text(encoding="utf-8")
            assert not re.search(r"(src|href)=['\"]?https?://(?!oxiedo\.com)", html)
            assert "<script" not in html and "@import" not in html
            assert "@media print" in html and "max-width:480px" in html
            assert not re.search(r"=[^'\"\s>]+/>", html)     # every self-closing attribute is quoted

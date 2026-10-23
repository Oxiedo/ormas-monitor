"""`ormas-monitor check` CLI: exit codes, --fail-on, --json, --quiet, --out."""
from __future__ import annotations

import json

import pytest
import torch
import torch.nn as nn

from ormas_monitor.cli import main as cli


def model_file(tmp_path, *, dead: bool = False, nan: bool = False):
    torch.manual_seed(0)
    m = nn.Sequential(nn.Conv2d(3, 8, 3), nn.BatchNorm2d(8), nn.ReLU(), nn.Flatten(), nn.LazyLinear(2))
    m(torch.randn(1, 3, 6, 6))
    sd = m.state_dict()
    if dead:
        sd["0.weight"][:2] = 0
        sd["0.bias"][:2] = 0
        sd["1.bias"][:2] = -1
    if nan:
        sd["4.weight"][0, 0] = float("nan")
    path = tmp_path / f"m{int(dead)}{int(nan)}.pt"
    torch.save(sd, path)
    return path


@pytest.mark.parametrize("dead,nan,fail_on,code", [
    (False, False, "critical", 0), (False, False, "warning", 0),
    (True, False, "critical", 0), (True, False, "warning", 1), (True, False, "never", 0),
    (False, True, "critical", 1), (False, True, "never", 0),
])
def test_fail_on_exit_codes(tmp_path, capsys, dead, nan, fail_on, code):
    path = model_file(tmp_path, dead=dead, nan=nan)
    assert cli(["check", str(path), "--fail-on", fail_on, "--quiet"]) == code
    assert capsys.readouterr().out == ""


def test_default_fail_on_is_critical(tmp_path):
    assert cli(["check", str(model_file(tmp_path, dead=True)), "--quiet"]) == 0
    assert cli(["check", str(model_file(tmp_path, nan=True)), "--quiet"]) == 1


def test_unreadable_and_refused_exit_2(tmp_path, capsys):
    assert cli(["check", str(tmp_path / "missing.pt")]) == 2
    torch.save({"model": nn.Linear(2, 2)}, tmp_path / "full.pt")
    assert cli(["check", str(tmp_path / "full.pt"), "--quiet"]) == 2
    assert "--trust" in capsys.readouterr().err
    assert cli(["check", str(tmp_path / "full.pt"), "--trust", "--quiet"]) == 0


def test_json_is_pure_stdout_and_valid(tmp_path, capsys):
    path = model_file(tmp_path, dead=True)
    sd = torch.load(path)
    sd["4.bias"] = torch.zeros_like(sd["4.bias"])      # zero before, nonzero after: change is inf
    torch.save(sd, tmp_path / "before.pt")
    assert cli(["check", str(path), "--json", "--against", str(tmp_path / "before.pt")]) == 0
    captured = capsys.readouterr()
    res = json.loads(captured.out)                     # strict JSON: no bare Infinity
    assert res["file"] == str(path)
    assert next(r for r in res["diff"]["changed"] if r["tensor"] == "4.bias")["change"] == "inf"
    assert "report:" in captured.err                    # human summary goes to stderr with --json


def test_out_path_and_summary(tmp_path, capsys):
    out = tmp_path / "custom.html"
    assert cli(["check", str(model_file(tmp_path, dead=True)), "--out", str(out)]) == 0
    text = capsys.readouterr().out
    assert out.exists() and "[warning]" in text and "wiring: name-based heuristic" in text


def test_folder_default_report_next_to_folder(tmp_path):
    folder = tmp_path / "hf"
    folder.mkdir()
    torch.save(nn.Linear(3, 3).state_dict(), folder / "pytorch_model.bin")
    assert cli(["check", str(folder) + "/", "--quiet"]) == 0
    assert (tmp_path / "hf.health.html").exists()


def test_bad_arch_exits_2(tmp_path):
    assert cli(["check", str(model_file(tmp_path)), "--arch", "keras:resnet", "--quiet"]) == 2


@pytest.mark.parametrize("dead,nan,fail_on,line", [
    (True, False, "critical", "exit 0: no critical findings (1 warning; use --fail-on warning to fail on them)"),
    (False, False, "critical", "exit 0: no critical findings"),
    (False, False, "warning", "exit 0: no critical or warning findings"),
    (True, False, "warning", "exit 1: 1 warning at or above --fail-on warning"),
    (True, True, "warning", "exit 1: 1 critical, 1 warning at or above --fail-on warning"),
    (True, True, "critical", "exit 1: 1 critical at or above --fail-on critical"),
    (True, True, "never", "exit 0: --fail-on never (1 critical, 1 warning)"),
])
def test_last_line_explains_exit_code(tmp_path, capsys, dead, nan, fail_on, line):
    path = model_file(tmp_path, dead=dead, nan=nan)
    code = cli(["check", str(path), "--fail-on", fail_on])
    last = capsys.readouterr().out.rstrip("\n").splitlines()[-1]
    assert last == line and last.startswith(f"exit {code}:")


def test_exit_line_goes_to_stderr_with_json(tmp_path, capsys):
    cli(["check", str(model_file(tmp_path, dead=True)), "--json"])
    captured = capsys.readouterr()
    assert json.loads(captured.out) and captured.err.rstrip().splitlines()[-1].startswith("exit 0: ")


def test_sequential_layer_names_are_readable(tmp_path, capsys):
    torch.manual_seed(0)
    m = nn.Sequential(nn.Conv2d(3, 8, 3), nn.BatchNorm2d(8), nn.ReLU(), nn.Conv2d(8, 16, 3),
                      nn.BatchNorm2d(16), nn.ReLU())
    sd = m.state_dict()
    sd["3.weight"][:5] = 0
    sd["3.bias"][:5] = 0
    sd["4.bias"][:5] = -1
    torch.save(sd, tmp_path / "seq.pt")
    cli(["check", str(tmp_path / "seq.pt")])
    assert ("[warning] 5 of 16 channels in layer `3` (Conv2d, BatchNorm `4`) are dead (zero filter or scale, shift <= 0) "
            "if a ReLU directly follows `4`") in capsys.readouterr().out

"""Record file safety: sign/seal/keygen never write through a link, and verify never crashes on a
hostile run directory (symlinked files, FIFOs, seals naming paths outside the run, malformed JSON)."""
import argparse
import json
import os
import shutil
from pathlib import Path

import pytest
import torch
import torch.nn as nn

from ormas_monitor import Monitor
from ormas_monitor.commands import sign as sign_cmd
from ormas_monitor.commands import verify as verify_cmd
from ormas_monitor.record import RecordError, keygen, seal, sign, verify_run
from ormas_monitor.record.signing import _replace_all, seal_bytes

pytest.importorskip("cryptography")
needs_symlinks = pytest.mark.skipif(not hasattr(os, "symlink") or os.name == "nt",
                                    reason="symbolic links need a POSIX file system")


def make_run(out):
    torch.manual_seed(0)
    m = nn.Sequential(nn.Linear(4, 4), nn.ReLU(), nn.Linear(4, 1))
    opt = torch.optim.SGD(m.parameters(), 0.1)
    mon = Monitor(m, opt, out=out, check_every=1, verbose=False)
    for _ in range(3):
        loss = m(torch.randn(4, 4)).pow(2).mean()
        opt.zero_grad()
        loss.backward()
        mon.step(loss)
        opt.step()
    mon.close(report=False)
    return Path(out)


@pytest.fixture(scope="module")
def base_run(tmp_path_factory):
    return make_run(tmp_path_factory.mktemp("safety") / "run")


@pytest.fixture
def run(base_run, tmp_path):
    return Path(shutil.copytree(base_run, tmp_path / "run"))


@pytest.fixture(scope="module")
def keys(tmp_path_factory):
    d = tmp_path_factory.mktemp("keys")
    keygen(d / "k.pem", d / "k.pub.pem")
    return d / "k.pem", d / "k.pub.pem"


def cli(module, argv):
    ap = argparse.ArgumentParser()
    module.register(ap.add_subparsers(dest="cmd", required=True))
    a = ap.parse_args(argv)
    return a.func(a)


def no_temp_files(run):
    return not [p for p in os.listdir(run) if p.endswith(".tmp")]


# ------------------------------------------------------------------ sign / seal: the r11 attacks
@needs_symlinks
def test_sign_refuses_a_symlinked_signature_and_leaves_its_target_alone(run, keys, tmp_path, capsys):
    victim = tmp_path / "victim_notes.txt"
    victim.write_text("important reviewer notes\n")
    os.symlink(victim, run / "record.sig")
    with pytest.raises(RecordError, match=r"record\.sig is a symbolic link"):
        sign(run, keys[0])
    assert cli(sign_cmd, ["sign", str(run), "--key", str(keys[0])]) == 2
    assert "symbolic link" in capsys.readouterr().err
    assert victim.read_text() == "important reviewer notes\n"
    assert not (run / "record.json").exists() and no_temp_files(run)


@needs_symlinks
def test_seal_refuses_a_dangling_record_json_link_and_creates_nothing(run, tmp_path):
    target = tmp_path / "created_by_seal.json"
    os.symlink(target, run / "record.json")
    with pytest.raises(RecordError, match=r"record\.json is a symbolic link"):
        seal(run)
    assert cli(sign_cmd, ["sign", str(run), "--seal-only"]) == 2
    assert not target.exists() and not os.path.lexists(target)


@needs_symlinks
@pytest.mark.parametrize("name", ["diary.jsonl", "metrics.jsonl"])
def test_seal_refuses_symlinked_evidence(run, tmp_path, name):
    real = tmp_path / name
    shutil.move(run / name, real)
    os.symlink(real, run / name)
    with pytest.raises(RecordError, match="symbolic link"):
        seal(run)
    assert not (run / "record.json").exists()


@needs_symlinks
def test_replace_swaps_a_link_planted_after_the_checks_instead_of_writing_through_it(tmp_path):
    run = tmp_path / "run"
    run.mkdir()
    victim = tmp_path / "victim"
    victim.write_text("keep")
    os.symlink(victim, run / "record.json")
    (path,) = _replace_all(run, [("record.json", b"{}")])
    assert victim.read_text() == "keep"
    assert not path.is_symlink() and path.read_bytes() == b"{}" and no_temp_files(run)


def test_seal_and_sign_leave_no_temp_files_and_verify(run, keys):
    seal(run)
    assert no_temp_files(run)
    sign(run, keys[0])
    assert no_temp_files(run) and verify_run(run, public_key=keys[1]).complete


@needs_symlinks
def test_keygen_never_writes_through_a_link(tmp_path):
    target = tmp_path / "elsewhere.pem"
    os.symlink(target, tmp_path / "dangling.pem")
    with pytest.raises(FileExistsError):
        keygen(tmp_path / "dangling.pem", tmp_path / "k.pub.pem")
    assert not os.path.lexists(target) and not (tmp_path / "k.pub.pem").exists()
    os.symlink(target, tmp_path / "dangling.pub.pem")
    with pytest.raises(FileExistsError):
        keygen(tmp_path / "k2.pem", tmp_path / "dangling.pub.pem")
    assert not os.path.lexists(target) and not (tmp_path / "k2.pem").exists()


# ------------------------------------------------------------------ verify: symlinked or odd files
@needs_symlinks
def test_symlinked_record_files_fail_their_check(run, keys, tmp_path):
    sign(run, keys[0])
    for name, check in (("metrics.jsonl", "metrics"), ("record.json", "seal"), ("record.sig", "signature")):
        case = Path(shutil.copytree(run, tmp_path / f"case_{check}"))
        real = tmp_path / f"real_{name}"
        shutil.move(case / name, real)
        os.symlink(real, case / name)
        r = verify_run(case, public_key=keys[1])
        assert r.check(check).status == "fail", check
        assert "symbolic link" in r.check(check).message
        assert not r.ok


@needs_symlinks
def test_symlinked_diary_fails_without_being_read(run, tmp_path, capsys):
    real = tmp_path / "real.jsonl"
    shutil.move(run / "diary.jsonl", real)
    os.symlink(real, run / "diary.jsonl")
    r = verify_run(run)
    assert r.check("parse").status == "fail" and "symbolic link" in r.check("parse").message
    assert {c.status for c in r.checks[1:]} == {"skipped"} and r.events == 0 and not r.ok
    assert cli(verify_cmd, ["verify", str(run)]) == 1
    assert verify_run(real).check("chain").status == "pass"      # named directly, a lone diary is read


@pytest.mark.skipif(not hasattr(os, "mkfifo"), reason="needs FIFOs")
def test_a_fifo_in_place_of_metrics_fails_instead_of_hanging(run):
    (run / "metrics.jsonl").unlink()
    os.mkfifo(run / "metrics.jsonl")
    r = verify_run(run)
    assert r.check("metrics").status == "fail" and "not a regular file" in r.check("metrics").message


# ------------------------------------------------------------------ verify: hostile seals (fuzz)
def sealed(run):
    seal(run)
    return json.loads((run / "record.json").read_bytes())


def write_seal(run, obj):
    (run / "record.json").write_bytes(seal_bytes(obj))


HOSTILE_FILES = [
    {"..": "0" * 64}, {".": "0" * 64}, {"": "0" * 64}, {"/etc/passwd": "0" * 64}, {"../x": "0" * 64},
    {"a/b": "0" * 64}, {"a\\b": "0" * 64}, {"x\x00y": "0" * 64}, {"diary.jsonl": None},
    {"diary.jsonl": ["list"]}, {"diary.jsonl": 10 ** 300}, {"\x1b[31mred": "0" * 64}, {"subdir": "0" * 64},
]


@pytest.mark.parametrize("files", HOSTILE_FILES, ids=[repr(sorted(f))[:20] for f in HOSTILE_FILES])
def test_hostile_seal_file_names_fail_the_seal_check(run, files):
    (run / "subdir").mkdir()
    obj = sealed(run)
    obj["files"].update(files)
    write_seal(run, obj)
    r = verify_run(run)
    c = r.check("seal")
    assert c.status == "fail" and c.message
    assert "\x1b" not in c.message and "\x00" not in c.message


@pytest.mark.parametrize("field, value", [
    ("files", None), ("files", "diary.jsonl"), ("files", ["diary.jsonl"]), ("files", 7),
    ("events", 10 ** 300), ("events", 1e308), ("events", True), ("events", "5"), ("events", [1] * 50),
    ("diary_head", {"x": 1}), ("format", ["x"] * 1000), ("signer", "me"), ("signer", [1, 2]),
    ("signer", {"public_key_sha256": ["not", "hex"]}), ("created", "\x1b]0;pwned\x07"),
])
def test_hostile_seal_values_never_raise(run, field, value):
    obj = sealed(run)
    obj[field] = value
    write_seal(run, obj)
    r = verify_run(run)
    assert r.check("seal").status in ("fail", "pass")
    assert all("\x1b" not in c.message and "\x07" not in c.message for c in r.checks)
    if field != "created" and field != "signer":
        assert r.check("seal").status == "fail"


@pytest.mark.parametrize("raw", [
    b"", b"{", b"[]", b"null", b"42", b'"text"', b"\xff\xfe\x00garbage", b"[" * 100_000,
    b'{"events":' + b"9" * 5000 + b"}", b'{"files":{"a":' * 3000 + b"1" + b"}}" * 3000,
], ids=lambda raw: f"raw{len(raw)}")
def test_malformed_seal_json_fails_the_seal_check(run, raw):
    (run / "record.json").write_bytes(raw)
    r = verify_run(run)
    assert r.check("seal").status == "fail"


def test_oversized_seal_is_not_read(run):
    (run / "record.json").write_bytes(b" " * ((1 << 20) + 1))
    c = verify_run(run).check("seal")
    assert c.status == "fail" and "larger than any seal" in c.message


@pytest.mark.parametrize("make", ["dir", "short", "long"])
def test_bad_record_sig_fails_the_signature_check(run, keys, make):
    sign(run, keys[0])
    (run / "record.sig").unlink()
    if make == "dir":
        (run / "record.sig").mkdir()
    else:
        (run / "record.sig").write_bytes(b"x" * (10 if make == "short" else 100_000))
    c = verify_run(run, public_key=keys[1]).check("signature")
    assert c.status == "fail" and c.message


def test_record_json_as_a_directory_fails(run):
    (run / "record.json").mkdir()
    assert verify_run(run).check("seal").status == "fail"


def test_cli_verify_reports_a_dotdot_seal_as_a_failed_check(run, capsys):
    obj = sealed(run)
    obj["files"][".."] = "0" * 64
    write_seal(run, obj)
    assert cli(verify_cmd, ["verify", str(run)]) == 1
    out = capsys.readouterr().out
    assert "FAIL" in out and "not a file in the run directory" in out


@pytest.mark.parametrize("line", [
    "[" * 100_000, '{"seq":' + "1" * 5000 + "}", '{"seq": 0, "event": ["x"], "severity": {"a": 1}}',
    '{"seq": {}, "time": [], "event": "alarm", "diagnosis": [1], "layer": {}, "prev": 5, "hash": null}',
    '{"seq": 0, "event": "channels", "level": {"x": []}, "step": "s"}',
], ids=lambda line: f"line{len(line)}")
def test_hostile_diary_lines_fail_checks_without_raising(run, line):
    rows = (run / "diary.jsonl").read_text().splitlines()
    rows.insert(1, line)
    (run / "diary.jsonl").write_text("\n".join(rows) + "\n")
    r = verify_run(run)
    assert not r.ok

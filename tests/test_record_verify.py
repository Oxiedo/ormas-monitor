"""Record verification, sealing and signing: tamper tests, round trips, keys and the CLI."""
import argparse
import json
import os
import shutil
import stat
from pathlib import Path

import pytest
import torch
import torch.nn as nn

from ormas_monitor import Monitor
from ormas_monitor.commands import keygen as keygen_cmd
from ormas_monitor.commands import sign as sign_cmd
from ormas_monitor.commands import verify as verify_cmd
from ormas_monitor.diary import GENESIS, Diary, file_sha256, read, verify
from ormas_monitor.record import RecordError, seal, verify_run
from ormas_monitor.record.chain import body_of, digest

SAMPLE = Path(__file__).resolve().parents[1] / "sample_reports" / "cifar10_with_repair.diary.jsonl"


def small_cnn():
    torch.manual_seed(0)
    return nn.Sequential(
        nn.Conv2d(3, 8, 3, padding=1), nn.BatchNorm2d(8), nn.ReLU(),
        nn.Conv2d(8, 16, 3, padding=1), nn.BatchNorm2d(16), nn.ReLU(),
        nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(16, 4))


def make_run(out, steps=24, close=True):
    model = small_cnn()
    opt = torch.optim.SGD(model.parameters(), lr=0.05)
    mon = Monitor(model, opt, out=out, check_every=2, warmup=3, verbose=False)
    g = torch.Generator().manual_seed(1)
    for _ in range(steps):
        x, y = torch.randn(16, 3, 8, 8, generator=g), torch.randint(0, 4, (16,), generator=g)
        loss = nn.functional.cross_entropy(model(x), y)
        opt.zero_grad()
        loss.backward()
        mon.step(loss)
        opt.step()
    if close:
        mon.close(report=False)
    return Path(out)


@pytest.fixture(scope="module")
def base_run(tmp_path_factory):
    return make_run(tmp_path_factory.mktemp("base") / "run")


@pytest.fixture
def run(base_run, tmp_path):
    return Path(shutil.copytree(base_run, tmp_path / "run"))


def lines(run):
    return (run / "diary.jsonl").read_text().splitlines()


def write_lines(run, rows):
    (run / "diary.jsonl").write_text("\n".join(rows) + "\n")


def status(result):
    return {c.name: c.status for c in result.checks}


def cli(module, argv):
    ap = argparse.ArgumentParser()
    module.register(ap.add_subparsers(dest="cmd", required=True))
    a = ap.parse_args(argv)
    return a.func(a)


# ------------------------------------------------------------------ compatibility
def test_diary_module_still_exports_v01_names(tmp_path):
    d = Diary(tmp_path / "d.jsonl")
    d.write("run_start", name="x")
    d.close()
    assert read(tmp_path / "d.jsonl")[0]["prev"] == GENESIS
    assert verify(tmp_path / "d.jsonl") == (True, 1, "all 1 events intact")
    assert len(file_sha256(tmp_path / "d.jsonl")) == 64


def test_sample_diary_only_reports_chain_intact_and_parts_absent():
    r = verify_run(SAMPLE)
    s = status(r)
    assert r.ok and not r.complete and r.events == 36 and r.run_dir is None
    assert all(s[k] == "pass" for k in ("parse", "genesis", "chain", "seq", "bookends"))
    assert s["metrics"] == s["seal"] == s["signature"] == "absent"
    assert "only a diary file was given" in r.check("metrics").message


def test_fresh_run_passes(run):
    r = verify_run(run)
    s = status(r)
    assert r.ok and all(s[k] == "pass" for k in ("parse", "genesis", "chain", "seq", "bookends", "metrics"))
    assert s["seal"] == s["signature"] == "absent"
    assert s["schema"] in ("pass", "skipped")
    assert verify_run(run / "diary.jsonl").run_dir == str(run)   # diary.jsonl belongs to its run dir


# ------------------------------------------------------------------ tamper tests
def test_edited_field_breaks_chain_at_that_event(run):
    rows = lines(run)
    obj = json.loads(rows[1])
    obj["step"] = 999
    rows[1] = json.dumps(obj)
    write_lines(run, rows)
    r = verify_run(run)
    assert not r.ok and r.check("chain").status == "fail"
    assert "event 1" in r.check("chain").message and "edited" in r.check("chain").message


def test_deleted_line_breaks_link_and_sequence(run):
    rows = lines(run)
    del rows[1]
    write_lines(run, rows)
    s = status(verify_run(run))
    assert s["chain"] == "fail" and s["seq"] == "fail"
    assert "deleted, inserted or reordered" in verify_run(run).check("chain").message


def test_deleted_first_line_breaks_genesis(run):
    write_lines(run, lines(run)[1:])
    s = status(verify_run(run))
    assert s["genesis"] == "fail" and s["bookends"] == "fail"


def test_reordered_lines_break_chain(run):
    rows = lines(run)
    rows[1], rows[2] = rows[2], rows[1]
    write_lines(run, rows)
    assert verify_run(run).check("chain").status == "fail"


def test_truncated_tail_is_reported_as_incomplete(run):
    write_lines(run, lines(run)[:-1])
    r = verify_run(run)
    assert r.check("chain").status == "pass"     # a chain alone cannot see a cut tail
    b = r.check("bookends")
    assert b.status == "fail" and "incomplete or truncated" in b.message and "cannot detect" in b.message
    assert r.check("metrics").status == "skipped"


def test_crashed_run_without_run_end_is_incomplete(tmp_path):
    run = make_run(tmp_path / "crashed", steps=6, close=False)
    r = verify_run(run)
    assert not r.ok and "incomplete or truncated" in r.check("bookends").message


def test_partial_last_line_is_reported(run):
    with open(run / "diary.jsonl", "a") as fh:
        fh.write('{"seq": 99, "event": "al')
    r = verify_run(run)
    assert r.check("parse").status == "fail" and r.check("chain").status == "fail"


def test_edited_metrics_fails_metrics_check(run):
    with open(run / "metrics.jsonl", "a") as fh:
        fh.write('{"step": 1}\n')
    c = verify_run(run).check("metrics")
    assert c.status == "fail" and "changed after the run ended" in c.message


def test_missing_metrics_fails_in_a_run_dir(run):
    (run / "metrics.jsonl").unlink()
    assert verify_run(run).check("metrics").status == "fail"


def test_missing_diary_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        verify_run(tmp_path)


def rechain(run, edit):
    """Rewrite the diary with `edit(obj)` applied to every line and every hash recomputed."""
    prev, rows = GENESIS, []
    for text in lines(run):
        obj = json.loads(text)
        edit(obj)
        obj["prev"] = prev
        obj["hash"] = prev = digest(prev, body_of(obj))
        rows.append(json.dumps(obj))
    write_lines(run, rows)


def test_schema_is_checked_strictly_when_the_record_declares_a_version(run):
    schema = pytest.importorskip("ormas_monitor.schema")

    def declare(obj):
        if obj["event"] == "run_start":
            obj["record_version"] = schema.RECORD_VERSION
    rechain(run, declare)
    assert verify_run(run).check("schema").status == "pass"

    def drop(obj):
        if obj["event"] == "structure":
            del obj["pairs"]
    rechain(run, drop)
    c = verify_run(run).check("schema")
    assert c.status == "fail" and "pairs" in c.message


def test_pre_schema_record_says_why_schema_was_not_checked():
    if verify_run(SAMPLE).check("schema").status != "skipped":
        pytest.fail("the v0.1 sample has no record_version and must not be failed by the v1 schema")
    msg = verify_run(SAMPLE).check("schema").message
    assert "predates the event schema" in msg or "no ormas_monitor.schema" in msg


# ------------------------------------------------------------------ seal (no cryptography needed)
def test_seal_detects_truncation_the_chain_cannot(run):
    seal(run)
    assert verify_run(run).check("seal").status == "pass"
    rows = lines(run)
    write_lines(run, rows[:-1])
    c = verify_run(run).check("seal")
    assert c.status == "fail" and "sealed head" in c.message


def test_seal_format(run):
    path = seal(run)
    raw = path.read_bytes()
    obj = json.loads(raw)
    assert raw == json.dumps(obj, sort_keys=True, separators=(",", ":")).encode()
    assert obj["format"] == "ormas-monitor-record" and obj["signer"] is None
    assert obj["files"] == {"diary.jsonl": file_sha256(run / "diary.jsonl"),
                            "metrics.jsonl": file_sha256(run / "metrics.jsonl")}
    assert obj["diary_head"] == json.loads(lines(run)[-1])["hash"]
    assert obj["events"] == len(lines(run))
    assert set(obj) == {"format", "record_version", "tool", "created", "files", "diary_head", "events", "signer"}


def test_seal_refuses_incomplete_or_tampered_records(run, tmp_path):
    write_lines(run, lines(run)[:-1])
    with pytest.raises(RecordError, match="does not verify"):
        seal(run)
    assert not (run / "record.json").exists()
    crashed = make_run(tmp_path / "crashed", steps=4, close=False)
    with pytest.raises(RecordError):
        seal(crashed)


def test_reseal_refused_after_files_changed(run):
    seal(run)
    data = json.loads(lines(run)[-1])
    with open(run / "metrics.jsonl", "a") as fh:      # tamper, then fix run_end's hash to match
        fh.write('{"step": 1}\n')
    data["metrics_sha256"] = file_sha256(run / "metrics.jsonl")
    data["hash"] = digest(data["prev"], body_of(data))
    write_lines(run, lines(run)[:-1] + [json.dumps(data)])
    assert verify_run(run).check("metrics").status == "pass"
    with pytest.raises(RecordError, match="no longer matches"):
        seal(run)


def test_noncanonical_seal_edit_is_detected(run):
    path = seal(run)
    obj = json.loads(path.read_bytes())
    path.write_text(json.dumps(obj, indent=1))
    assert "canonical" in verify_run(run).check("seal").message


# ------------------------------------------------------------------ signing
@pytest.fixture
def keys(tmp_path):
    pytest.importorskip("cryptography")
    from ormas_monitor.record import keygen
    return keygen(tmp_path / "k.pem", tmp_path / "k.pub.pem")


@pytest.fixture
def other_key(tmp_path):
    from ormas_monitor.record import keygen
    return keygen(tmp_path / "o.pem", tmp_path / "o.pub.pem")


def test_keygen_permissions_and_no_overwrite(keys, tmp_path):
    assert stat.S_IMODE(os.stat(keys.private_path).st_mode) == 0o600
    assert len(keys.fingerprint) == 64
    from ormas_monitor.record import keygen
    before = keys.private_path.read_bytes()
    with pytest.raises(FileExistsError):
        keygen(keys.private_path, tmp_path / "new.pub.pem")
    with pytest.raises(FileExistsError):
        keygen(tmp_path / "new.pem", keys.public_path)
    assert keys.private_path.read_bytes() == before
    assert not (tmp_path / "new.pem").exists() and not (tmp_path / "new.pub.pem").exists()


def test_sign_verify_round_trip(run, keys):
    from ormas_monitor.record import sign
    sig = sign(run, keys.private_path)
    assert sig.name == "record.sig" and len(sig.read_bytes()) == 64
    seal_obj = json.loads((run / "record.json").read_bytes())
    assert seal_obj["signer"] == {"algorithm": "Ed25519", "public_key_sha256": keys.fingerprint}
    r = verify_run(run, public_key=keys.public_path)
    assert r.ok and r.check("signature").status == "pass" and r.check("seal").status == "pass"
    assert verify_run(run).check("signature").status == "unchecked"     # no key: never "pass"
    sign(run, keys.private_path)                                         # re-signing an intact run is allowed
    assert verify_run(run, public_key=keys.public_path).ok


def test_wrong_key_fails(run, keys, other_key):
    from ormas_monitor.record import sign
    sign(run, keys.private_path)
    c = verify_run(run, public_key=other_key.public_path).check("signature")
    assert c.status == "fail" and "not by the key given" in c.message


def test_edited_seal_fails_signature(run, keys):
    from ormas_monitor.record import sign
    from ormas_monitor.record.signing import seal_bytes
    sign(run, keys.private_path)
    obj = json.loads((run / "record.json").read_bytes())
    obj["created"] = "2020-01-01T00:00:00Z"           # still canonical, files still match
    (run / "record.json").write_bytes(seal_bytes(obj))
    r = verify_run(run, public_key=keys.public_path)
    assert r.check("seal").status == "pass" and r.check("signature").status == "fail"


def test_missing_signature_fails(run, keys):
    from ormas_monitor.record import sign
    sign(run, keys.private_path)
    (run / "record.sig").unlink()
    c = verify_run(run).check("signature")
    assert c.status == "fail" and "record.sig is missing" in c.message
    unsigned = run.parent / "unsigned"
    shutil.copytree(run, unsigned)
    (unsigned / "record.json").unlink()
    assert verify_run(unsigned, public_key=keys.public_path).check("signature").status == "fail"


def test_full_rewrite_passes_the_chain_but_not_the_signature(run, keys):
    """The threat model in one test: recomputing every hash defeats the chain, not the key."""
    from ormas_monitor.record import sign
    sign(run, keys.private_path)

    def forge(obj):
        if obj["event"] == "run_start":
            obj["parameters"] = 1
    rechain(run, forge)
    r = verify_run(run, public_key=keys.public_path)
    assert r.check("chain").status == "pass" and r.check("seal").status == "fail"
    with pytest.raises(RecordError):
        seal(run)                                     # the old seal no longer matches: refused
    (run / "record.json").unlink()
    (run / "record.sig").unlink()
    seal(run)                                         # a forger can always write a fresh seal...
    assert verify_run(run, public_key=keys.public_path).check("signature").status == "fail"   # ...not a signature


def test_sign_refuses_a_broken_record(run, keys):
    from ormas_monitor.record import sign
    write_lines(run, lines(run)[:-1])
    with pytest.raises(RecordError):
        sign(run, keys.private_path)
    assert not (run / "record.sig").exists()


def test_missing_cryptography_gives_install_hint(run, monkeypatch, tmp_path):
    import sys
    from ormas_monitor.record import keygen
    from ormas_monitor.record.signing import SigningUnavailable
    monkeypatch.setitem(sys.modules, "cryptography.hazmat.primitives", None)
    with pytest.raises(SigningUnavailable, match=r"ormas-monitor\[signing\]"):
        keygen(tmp_path / "a.pem", tmp_path / "b.pem")


# ------------------------------------------------------------------ CLI
def test_cli_verify_exit_codes_and_json(run, capsys):
    assert cli(verify_cmd, ["verify", str(run)]) == 0
    out = capsys.readouterr().out
    assert "[ok  ] hash chain" in out and "not a compliance certificate" in out
    assert cli(verify_cmd, ["verify", str(run), "--strict"]) == 1    # unsealed, unsigned
    capsys.readouterr()
    write_lines(run, lines(run)[:-1])
    assert cli(verify_cmd, ["verify", str(run), "--json"]) == 1
    data = json.loads(capsys.readouterr().out)
    assert data["ok"] is False and {c["name"]: c["status"] for c in data["checks"]}["bookends"] == "fail"
    assert cli(verify_cmd, ["verify", str(SAMPLE)]) == 0


def test_cli_keygen_sign_verify(run, tmp_path, capsys):
    pytest.importorskip("cryptography")
    priv, pub = tmp_path / "c.pem", tmp_path / "c.pub.pem"
    assert cli(keygen_cmd, ["keygen", "--private", str(priv), "--public", str(pub)]) == 0
    assert cli(keygen_cmd, ["keygen", "--private", str(priv), "--public", str(pub)]) == 2
    assert cli(sign_cmd, ["sign", str(run), "--key", str(priv)]) == 0
    assert cli(verify_cmd, ["verify", str(run), "--key", str(pub), "--strict"]) in (0, 1)
    r = verify_run(run, public_key=pub)
    assert r.ok and r.check("signature").status == "pass"
    assert cli(verify_cmd, ["verify", str(run), "--key", str(pub)]) == 0
    capsys.readouterr()
    assert cli(sign_cmd, ["sign", str(run), "--seal-only"]) == 2      # already signed: refuse to re-seal
    assert "re-sealing would invalidate" in capsys.readouterr().err

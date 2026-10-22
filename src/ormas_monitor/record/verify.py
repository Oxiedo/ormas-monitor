"""Full verification of a run record: chain, bookends, schema, metrics, seal and signature.

    result = verify_run("ormas_run", public_key="reviewer.pub.pem")
    result.ok            # no check failed
    result.complete      # every check passed (nothing absent or unchecked)
    for c in result.checks: print(c.status, c.label, c.message)

Each check has one of five statuses:
    pass       checked and correct
    fail       checked and wrong
    absent     the part does not exist (an unsealed run, or a lone diary file with no metrics)
    unchecked  the part exists but could not be checked as asked (a signature with no key given)
    skipped    the check could not run (no run_end to compare metrics with, or no schema module)

What verification can and cannot establish is spelled out in `record.chain` and `record.signing`.

A run directory may come from someone else, so every value read from it is treated as hostile:
verification never raises on a malformed record, it fails the check concerned. Record files that
are symbolic links (or FIFOs, devices, directories) fail their check without being followed,
because `sign` refuses to seal such a record and a passing verdict must mean the directory itself
holds the evidence; following a link could also read a file elsewhere on the reviewer's machine or
hang on a FIFO. Values echoed in messages are clipped and stripped of control characters.
In short: a passing record has not been edited since it was written (or since it was signed, if a
trusted key verified it). It is evidence of what the monitor recorded, not a compliance certificate.
"""
from __future__ import annotations

import importlib
import inspect
import json
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType
from typing import Any

from .chain import GENESIS, body_of, digest, file_sha256, parse_lines
from .layout import FORMAT, LAYOUT, Location, RecordLayout, file_problem, locate
from .signing import fingerprint, load_public_key, seal_bytes, verify_signature

STATUSES = ("pass", "fail", "absent", "unchecked", "skipped")
MAX_SEAL_BYTES = 1 << 20          # a real record.json is well under 1 KiB
SIGNATURE_BYTES = 64              # a raw Ed25519 signature

# name -> label, in the order the checklist is printed
CHECKS = {
    "parse": "JSON lines",
    "genesis": "genesis link",
    "chain": "hash chain",
    "seq": "sequence numbers",
    "bookends": "run_start / run_end",
    "schema": "event schema",
    "metrics": "metrics hash",
    "seal": "seal (record.json)",
    "signature": "signature (record.sig)",
}


@dataclass(frozen=True)
class Check:
    name: str
    status: str
    message: str

    def __post_init__(self) -> None:
        assert self.status in STATUSES, self.status
        assert self.name in CHECKS, self.name

    @property
    def label(self) -> str:
        return CHECKS[self.name]


@dataclass(frozen=True)
class VerifyResult:
    path: str               # what the user passed
    diary: str              # the diary file that was read
    run_dir: str | None     # None when a lone diary file was verified
    events: int
    head: str | None        # hash of the last line, None for an empty diary
    checks: tuple[Check, ...]

    @property
    def ok(self) -> bool:
        """No check failed. Absent, unchecked and skipped parts are not failures; see `complete`."""
        return all(c.status != "fail" for c in self.checks)

    @property
    def complete(self) -> bool:
        """Every check ran and passed: the run is finished, sealed and its signature verified."""
        return all(c.status == "pass" for c in self.checks)

    def check(self, name: str) -> Check:
        for c in self.checks:
            if c.name == name:
                return c
        raise KeyError(name)

    def to_dict(self) -> dict:
        return {"path": self.path, "diary": self.diary, "run_dir": self.run_dir, "events": self.events,
                "head": self.head, "ok": self.ok, "complete": self.complete,
                "checks": [{"name": c.name, "label": c.label, "status": c.status, "message": c.message}
                           for c in self.checks]}


def load_schema() -> ModuleType | None:
    """`ormas_monitor.schema` if this install has it, else None."""
    try:
        return importlib.import_module("ormas_monitor.schema")
    except ModuleNotFoundError as e:
        if e.name != "ormas_monitor.schema":
            raise
        return None


def _clip(value: Any, limit: int = 60) -> str:
    """A value read from the record, made safe to print: control characters replaced, length capped."""
    text = "".join(ch if ch.isprintable() else "?" for ch in str(value)[:limit + 1])
    return text if len(text) <= limit else text[:limit - 3] + "..."


def _short(h: Any) -> str:
    return f"{_clip(h, 12)}..."


def _plain_name(name: Any) -> bool:
    """A bare file name inside the run directory: no separators, not "." or "..", not absolute."""
    return (isinstance(name, str) and name not in ("", ".", "..") and "/" not in name and "\\" not in name
            and "\x00" not in name and Path(name).name == name)


def _check_chain(events: list[dict], bad_lines: list[int]) -> list[Check]:
    out: list[Check] = []
    n = len(events)
    if bad_lines:
        more = f" (and {len(bad_lines) - 1} more)" if len(bad_lines) > 1 else ""
        out.append(Check("parse", "fail", f"line {bad_lines[0]}{more} is not a JSON object; a write was cut "
                                          "short or the file was edited"))
    else:
        out.append(Check("parse", "pass", f"{n} events parsed"))

    if not events:
        out.append(Check("genesis", "fail", "the diary has no events"))
    elif events[0].get("prev") != GENESIS:
        out.append(Check("genesis", "fail", "the first event does not link to the genesis hash: "
                                            "earlier lines were removed"))
    else:
        out.append(Check("genesis", "pass", "first event links to the genesis hash"))

    broken = None
    for i, line in enumerate(events):
        prev = line.get("prev")
        if digest(str(prev), body_of(line)) != line.get("hash"):
            broken = f"event {i} (seq {_clip(line.get('seq'))}, {_clip(line.get('event'))}): its hash does not " \
                     "match its contents, so it was edited after it was written"
            break
        if i and prev != events[i - 1].get("hash"):
            broken = f"event {i} (seq {_clip(line.get('seq'))}, {_clip(line.get('event'))}) does not link to " \
                     "the event before it: lines were deleted, inserted or reordered"
            break
    if bad_lines:
        out.append(Check("chain", "fail", f"not checkable past the unreadable line {bad_lines[0]}"
                                          + (f"; also {broken}" if broken else "")))
    elif broken:
        out.append(Check("chain", "fail", broken))
    else:
        out.append(Check("chain", "pass", f"all {n} hashes recompute and link"))

    seqs = [line.get("seq") for line in events]
    gap = next((i for i, s in enumerate(seqs) if s != i), None)
    if gap is None:
        out.append(Check("seq", "pass", f"seq runs 0..{n - 1}" if n else "no events"))
    else:
        out.append(Check("seq", "fail", f"event {gap} has seq {_clip(repr(seqs[gap]))}, expected {gap}"))

    kinds = [line.get("event") for line in events]
    problems = []
    if not kinds or kinds[0] != "run_start":
        problems.append("the first event is not run_start")
    if kinds.count("run_start") > 1:
        problems.append(f"run_start appears {kinds.count('run_start')} times")
    if kinds.count("run_end") > 1 or ("run_end" in kinds and kinds[-1] != "run_end"):
        problems.append("events follow run_end, or run_end appears more than once")
    if problems:
        out.append(Check("bookends", "fail", "; ".join(problems)))
    elif kinds[-1] != "run_end":
        out.append(Check("bookends", "fail",
                         "no run_end: the record is incomplete or truncated. A run that crashed and a diary "
                         "whose last lines were deleted look the same, because a hash chain alone cannot "
                         "detect lines removed from the end; a seal (record.json) can"))
    else:
        out.append(Check("bookends", "pass", "run_start is first and run_end is last: the run finished"))
    return out


def _check_schema(events: list[dict]) -> Check:
    schema = load_schema()
    if schema is None:
        return Check("schema", "skipped", "this install has no ormas_monitor.schema; event fields not checked")
    start = events[0] if events and events[0].get("event") == "run_start" else {}
    if "record_version" not in start:
        return Check("schema", "skipped", f"this record predates the event schema (no record_version in run_start; "
                                          f"written by {_clip(start.get('tool', 'an unknown tool'))}), so event "
                                          f"fields "
                                          f"were not checked against schema version "
                                          f"{getattr(schema, 'RECORD_VERSION', '?')}")
    validate = schema.validate_event
    kwargs = {"strict": True} if "strict" in inspect.signature(validate).parameters else {}
    bad = []
    for i, line in enumerate(events):
        try:
            errs = [_clip(e, 200) for e in validate(line, **kwargs)]
        except (TypeError, ValueError, RecursionError) as e:     # a field of a type the schema never expects
            errs = [f"fields of unexpected types ({type(e).__name__})"]
        if errs:
            bad.append(f"event {i} ({_clip(line.get('event'))}): {'; '.join(errs)}")
    if bad:
        more = f" (+{len(bad) - 3} more events)" if len(bad) > 3 else ""
        return Check("schema", "fail", " | ".join(bad[:3]) + more)
    return Check("schema", "pass", f"all {len(events)} events match record schema "
                                   f"{getattr(schema, 'RECORD_VERSION', '?')}")


def _check_metrics(loc: Location, events: list[dict], layout: RecordLayout) -> Check:
    path = loc.file(layout.metrics)
    if path is None:
        return Check("metrics", "absent", "only a diary file was given, so metrics.jsonl was not compared "
                                          "with the hash recorded in run_end")
    end = events[-1] if events and events[-1].get("event") == "run_end" else None
    if end is None:
        return Check("metrics", "skipped", "no run_end, so there is no recorded metrics hash to compare")
    expected = end.get("metrics_sha256")
    if not expected:
        return Check("metrics", "fail", "run_end carries no metrics_sha256")
    problem = file_problem(path)
    if problem is not None:
        return Check("metrics", "fail", f"{layout.metrics} is {problem}")
    actual = file_sha256(path)
    if actual != expected:
        return Check("metrics", "fail", f"{layout.metrics} changed after the run ended (sha256 {_short(actual)} "
                                        f"vs {_short(expected)} recorded in run_end)")
    return Check("metrics", "pass", f"sha256 {_short(actual)} as recorded in run_end")


def _read_seal(path: Path) -> tuple[dict | None, bytes, str | None]:
    size = path.stat().st_size
    if size > MAX_SEAL_BYTES:
        return None, b"", f"record.json is {size:,} bytes, far larger than any seal; not read"
    raw = path.read_bytes()
    try:
        obj = json.loads(raw)
    except (ValueError, RecursionError):       # not JSON or not UTF-8, an over-long integer, nested too deep
        return None, raw, "record.json is not valid JSON"
    if not isinstance(obj, dict):
        return None, raw, "record.json is not a JSON object"
    return obj, raw, None


def _check_seal(loc: Location, events: list[dict], layout: RecordLayout) -> tuple[Check, dict | None, bytes]:
    if loc.run_dir is None:
        return Check("seal", "absent", "only a diary file was given; there is no seal to check"), None, b""
    path = loc.run_dir / layout.seal
    problem = file_problem(path)
    if problem == "missing":
        return Check("seal", "absent", "not sealed (no record.json). Without a seal, lines cut from the end of "
                                       "the diary show only as a missing run_end; `ormas-monitor sign` seals "
                                       "it"), None, b""
    if problem is not None:
        return Check("seal", "fail", f"record.json is {problem}"), None, b""
    seal, raw, err = _read_seal(path)
    if seal is None:
        return Check("seal", "fail", str(err)), None, raw
    problems = []
    if raw != seal_bytes(seal):
        problems.append("record.json is not in canonical form, so it was rewritten after sealing")
    if seal.get("format") != FORMAT:
        problems.append(f"format is {_clip(repr(seal.get('format')))}, not {FORMAT!r}")
    files = seal.get("files")
    if not isinstance(files, dict):
        problems.append("record.json lists no files")
        files = {}
    for name in layout.sealed_files:
        if name not in files:
            problems.append(f"the seal does not cover {name}")
    for name, expected in sorted(files.items()):
        if not _plain_name(name):
            problems.append(f"the seal names {_clip(repr(name))}, which is not a file in the run directory")
            continue
        f = loc.run_dir / name
        problem = file_problem(f)
        if problem is not None:
            problems.append(f"{_clip(name)} is {problem}")
        elif not isinstance(expected, str) or file_sha256(f) != expected:
            problems.append(f"{_clip(name)} changed after sealing")
    head = events[-1].get("hash") if events else None
    if seal.get("diary_head") != head:
        problems.append("the diary's last hash differs from the sealed head: lines were added, removed or "
                        "rewritten after sealing")
    count = seal.get("events")
    if not (type(count) is int and count == len(events)):
        problems.append(f"the diary has {len(events)} events, the seal says {_clip(repr(count))}")
    if problems:
        more = f" (+{len(problems) - 5} more)" if len(problems) > 5 else ""
        return Check("seal", "fail", "; ".join(problems[:5]) + more), seal, raw
    return Check("seal", "pass", f"{', '.join(_clip(n) for n in sorted(files))} and the diary head match the "
                                 f"seal of {_clip(seal.get('created'))}"), seal, raw


def _check_signature(loc: Location, seal: dict | None, raw: bytes, public_key: str | Path | None,
                     layout: RecordLayout) -> Check:
    if loc.run_dir is None:
        return Check("signature", "absent", "only a diary file was given; there is no signature to check")
    sig_path = loc.run_dir / layout.signature
    signer = seal.get("signer") if seal else None
    claimed = signer.get("public_key_sha256") if isinstance(signer, dict) else None
    problem = file_problem(sig_path)
    if problem not in (None, "missing"):
        return Check("signature", "fail", f"record.sig is {problem}")
    if problem == "missing":
        if claimed:
            return Check("signature", "fail", f"record.json says it was signed by key {_short(claimed)} but "
                                              "record.sig is missing")
        if public_key is not None:
            return Check("signature", "fail", "a key was given but the record is not signed (no record.sig)")
        return Check("signature", "absent", "not signed (no record.sig)")
    if seal is None:
        return Check("signature", "fail", "record.sig exists but record.json is missing or unreadable")
    size = sig_path.stat().st_size
    if size != SIGNATURE_BYTES:
        return Check("signature", "fail", f"record.sig is {size:,} bytes, not a {SIGNATURE_BYTES}-byte Ed25519 "
                                          "signature")
    if public_key is None:
        who = f"key {_short(claimed)}" if claimed else "an unnamed key"
        return Check("signature", "unchecked", f"record.sig is present, made by {who}; pass --key with the "
                                               "signer's public key, obtained from them rather than from this "
                                               "directory, to check it")
    key = load_public_key(public_key)
    fp = fingerprint(key)
    if claimed != fp:
        return Check("signature", "fail", f"signed by key {_short(claimed)}, not by the key given ({_short(fp)})")
    if not verify_signature(raw, sig_path.read_bytes(), key):
        return Check("signature", "fail", "the signature does not match record.json: the seal or the signature "
                                          "was altered after signing")
    return Check("signature", "pass", f"valid Ed25519 signature over record.json by key {_short(fp)}")


def verify_run(run_dir_or_diary: str | Path, public_key: str | Path | None = None,
               layout: RecordLayout = LAYOUT) -> VerifyResult:
    """Verify a run directory (or a lone diary file) and return every check's result.

    `public_key` is a PEM file. When given, a valid signature by exactly that key is required;
    when omitted, a signature that exists is reported as `unchecked`, never as passed.
    Raises FileNotFoundError if there is no diary, and SigningUnavailable if a key is given but
    the optional `cryptography` package is missing.
    """
    loc = locate(run_dir_or_diary, layout)
    # Inside a run directory the diary must be the directory's own regular file. A lone diary file
    # is whatever the user named, so it is opened as given.
    problem = file_problem(loc.diary) if loc.run_dir is not None else None
    if problem == "missing" or (loc.run_dir is None and not loc.diary.is_file()):
        raise FileNotFoundError(f"no diary at {loc.diary}")
    if problem is not None:
        return _unread(run_dir_or_diary, loc, f"{loc.diary.name} is {problem}")
    events, bad = parse_lines(loc.diary)
    checks = _check_chain(events, bad)
    checks.append(_check_schema(events))
    checks.append(_check_metrics(loc, events, layout))
    seal_check, seal, raw = _check_seal(loc, events, layout)
    checks.append(seal_check)
    checks.append(_check_signature(loc, seal, raw, public_key, layout))
    assert [c.name for c in checks] == list(CHECKS)
    return VerifyResult(path=str(run_dir_or_diary), diary=str(loc.diary),
                        run_dir=None if loc.run_dir is None else str(loc.run_dir), events=len(events),
                        head=events[-1].get("hash") if events else None, checks=tuple(checks))


def _unread(path: str | Path, loc: Location, why: str) -> VerifyResult:
    """The result for a diary that was deliberately not read: the parse check fails, the rest skip."""
    checks = [Check("parse", "fail", why)]
    checks += [Check(name, "skipped", "the diary was not read") for name in list(CHECKS)[1:]]
    return VerifyResult(path=str(path), diary=str(loc.diary),
                        run_dir=None if loc.run_dir is None else str(loc.run_dir), events=0, head=None,
                        checks=tuple(checks))

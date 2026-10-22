"""The hash chain: an append-only diary of what happened during training.

Each line is one JSON event. It carries the SHA-256 of the previous line, so editing,
deleting or reordering any line breaks the link at that point, and verification finds where.

    hash = SHA-256( prev + canonical_json(body) )     body = the line without `prev` and `hash`

A chain on its own has two blind spots, stated so nobody over-reads it:
- removing lines from the END leaves a shorter chain that is still internally consistent
  (verification reports the missing `run_end`; a seal over the head hash closes the gap);
- anyone who can write the file can rewrite it whole and recompute every hash. Only a
  signature by a key the writer does not hold detects that (see `record.signing`).
"""
from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path
from typing import Any

GENESIS = "0" * 64


def canonical_json(obj: Any) -> str:
    """The one serialisation every hash and signature in the record is computed over."""
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), default=str)


def digest(prev: str, body: dict) -> str:
    """Hash of one diary line, given the previous line's hash and this line's body."""
    return hashlib.sha256((prev + canonical_json(body)).encode()).hexdigest()


def body_of(line: dict) -> dict:
    """The hashed part of a diary line: everything except `prev` and `hash`."""
    return {k: v for k, v in line.items() if k not in ("prev", "hash")}


class Diary:
    """Writer for `diary.jsonl`. Every `write` is flushed, so a crash loses at most one line."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._fh = open(self.path, "w")
        self._prev = GENESIS
        self._seq = 0

    def write(self, event: str, **fields: Any) -> dict:
        """Append one event and return the full line, including `seq`, `prev` and `hash`."""
        body = {"seq": self._seq, "time": round(time.time(), 3), "event": event, **fields}
        line = {**body, "prev": self._prev, "hash": digest(self._prev, body)}
        self._fh.write(json.dumps(line, default=str) + "\n")
        self._fh.flush()
        self._prev, self._seq = line["hash"], self._seq + 1
        return line

    def close(self) -> None:
        if not self._fh.closed:
            self._fh.close()


def read(path: str | Path) -> list[dict]:
    """All events of a diary. Raises on a line that is not JSON; see `parse_lines` to tolerate it."""
    with open(path) as fh:
        return [json.loads(line) for line in fh if line.strip()]


def parse_lines(path: str | Path) -> tuple[list[dict], list[int]]:
    """Events that parse as JSON objects, and the 1-based line numbers of those that do not."""
    events: list[dict] = []
    bad: list[int] = []
    with open(path, encoding="utf-8", errors="replace") as fh:
        for no, text in enumerate(fh, 1):
            if not text.strip():
                continue
            try:
                obj = json.loads(text)
            except (ValueError, RecursionError):     # not JSON, an over-long integer, or nested too deep
                bad.append(no)
                continue
            if isinstance(obj, dict):
                events.append(obj)
            else:
                bad.append(no)
    return events, bad


def verify(path: str | Path) -> tuple[bool, int, str]:
    """Chain-only check, kept for v0.1 callers: (intact, number of events, message).

    `record.verify.verify_run` is the full check (bookends, metrics, schema, seal, signature).
    """
    events, bad = parse_lines(path)
    if bad:
        return False, len(events), f"record broken: line {bad[0]} is not a JSON event"
    prev = GENESIS
    for i, line in enumerate(events):
        if line.get("prev") != prev or line.get("seq") != i or digest(prev, body_of(line)) != line.get("hash"):
            return False, len(events), f"record broken at event {i} (seq {line.get('seq')})"
        prev = line["hash"]
    return True, len(events), f"all {len(events)} events intact"


def file_sha256(path: str | Path) -> str:
    """SHA-256 of a file's bytes, streamed."""
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()

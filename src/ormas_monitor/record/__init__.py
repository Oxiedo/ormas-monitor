"""The training record: a hash-chained diary that a third party can verify, seal, sign and compare.

    from ormas_monitor.record import verify_run, seal, sign, keygen, diff_runs

- `record.chain`    the diary writer and the hash chain (also importable as `ormas_monitor.diary`)
- `record.verify`   full verification with a per-check result
- `record.signing`  record.json seal, Ed25519 detached signature, key generation
- `record.diff`     "what changed since the run I approved?"

The record is evidence for technical documentation. It is never a compliance certificate.
"""
from __future__ import annotations

from .chain import GENESIS, Diary, canonical_json, file_sha256, read
from .diff import diff_runs
from .layout import LAYOUT, RecordLayout
from .signing import KeyInfo, RecordError, SigningUnavailable, keygen, seal, sign
from .verify import Check, VerifyResult, verify_run

__all__ = ["GENESIS", "LAYOUT", "Check", "Diary", "KeyInfo", "RecordError", "RecordLayout",
           "SigningUnavailable", "VerifyResult", "canonical_json", "diff_runs", "file_sha256", "keygen",
           "read", "seal", "sign", "verify_run"]

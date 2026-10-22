"""The Diary: an append-only, hash-chained record of what happened during training.

The implementation lives in `ormas_monitor.record.chain`; this module keeps the v0.1 import
path (`from ormas_monitor.diary import Diary, read, verify, file_sha256, GENESIS`) working.
"""
from __future__ import annotations

from .record.chain import GENESIS, Diary, canonical_json, file_sha256, read, verify

__all__ = ["GENESIS", "Diary", "canonical_json", "file_sha256", "read", "verify"]

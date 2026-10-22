"""`ormas-monitor sign`: seal a finished run (record.json) and sign it (record.sig)."""
from __future__ import annotations

import sys


def register(sub) -> None:
    s = sub.add_parser("sign", help="seal a finished run and sign the seal",
                       description="Write record.json (SHA-256 of diary.jsonl and metrics.jsonl, the diary "
                                   "head and length) and record.sig (Ed25519 over record.json's exact bytes). "
                                   "Refuses if the run does not verify.")
    s.add_argument("run", help="run directory")
    how = s.add_mutually_exclusive_group(required=True)
    how.add_argument("--key", metavar="PRIVATE.pem", help="Ed25519 private key (from `ormas-monitor keygen`)")
    how.add_argument("--seal-only", action="store_true",
                     help="write record.json without a signature (a checksum, not proof of who sealed it)")
    s.set_defaults(func=run)


def run(a) -> int:
    from ..record.signing import RecordError, SigningUnavailable, seal, sign

    try:
        if a.seal_only:
            path = seal(a.run)
            print(f"sealed: {path} (unsigned: anyone who can write this directory can write a new seal)")
            return 0
        sig = sign(a.run, a.key)
    except (RecordError, SigningUnavailable, FileNotFoundError, ValueError) as e:
        print(f"error: {e}", file=sys.stderr)
        return 2
    print(f"sealed: {sig.parent / 'record.json'}")
    print(f"signed: {sig}")
    print(f"check:  ormas-monitor verify {a.run} --key PUBLIC.pem")
    return 0

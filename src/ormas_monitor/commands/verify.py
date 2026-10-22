"""`ormas-monitor verify`: check that a run's record is complete and has not been edited."""
from __future__ import annotations

import json
import sys
from pathlib import Path

MARKS = {"pass": "ok  ", "fail": "FAIL", "absent": " -- ", "unchecked": " ?? ", "skipped": "skip"}


def register(sub) -> None:
    v = sub.add_parser("verify", help="check that a run's record is complete and has not been edited",
                       description="Verify a run directory (or a lone diary.jsonl): hash chain, sequence, "
                                   "run_start/run_end, event schema, metrics hash, seal and signature. "
                                   "Exit code 0 means no check failed.")
    v.add_argument("run", help="run directory or diary file")
    v.add_argument("--key", metavar="PUBLIC.pem",
                   help="the signer's public key; a valid signature by this key is then required")
    v.add_argument("--strict", action="store_true",
                   help="also fail when a part is absent or unchecked (unsealed, unsigned, no metrics)")
    v.add_argument("--json", action="store_true", help="print the result as JSON")
    v.set_defaults(func=run)


def run(a) -> int:
    from ..record.signing import SigningUnavailable
    from ..record.verify import verify_run

    try:
        result = verify_run(a.run, public_key=a.key)
    except (FileNotFoundError, SigningUnavailable, ValueError) as e:
        print(f"error: {e}", file=sys.stderr)
        return 2
    passed = result.complete if a.strict else result.ok
    if a.json:
        print(json.dumps({**result.to_dict(), "strict": a.strict, "passed": passed}, indent=2))
        return 0 if passed else 1
    target = result.run_dir or result.diary
    print(f"ormas-monitor verify {Path(target)}  ({result.events} events)")
    for c in result.checks:
        print(f"  [{MARKS[c.status]}] {c.label:<23} {c.message}")
    counts = {s: sum(c.status == s for c in result.checks) for s in MARKS}
    tail = ", ".join(f"{n} {s}" for s, n in counts.items() if n and s != "pass")
    verdict = "PASSED" if passed else "FAILED"
    print(f"{verdict}: {counts['pass']} of {len(result.checks)} checks passed" + (f" ({tail})" if tail else "")
          + (" [strict]" if a.strict else ""))
    print("This record is evidence of what the monitor wrote, not a compliance certificate.")
    return 0 if passed else 1

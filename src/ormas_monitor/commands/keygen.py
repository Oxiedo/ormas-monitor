"""`ormas-monitor keygen`: create an Ed25519 key pair for signing run records."""
from __future__ import annotations

import sys


def register(sub) -> None:
    k = sub.add_parser("keygen", help="create an Ed25519 key pair for signing run records",
                       description="Write an unencrypted Ed25519 private key (mode 0600) and its public key "
                                   "as PEM. Never overwrites. A key held by the reviewer, not the training "
                                   "machine, makes a signature mean more.")
    k.add_argument("--private", default="ormas-signing.pem", metavar="PATH",
                   help="private key file (default: ormas-signing.pem)")
    k.add_argument("--public", default="ormas-signing.pub.pem", metavar="PATH",
                   help="public key file (default: ormas-signing.pub.pem)")
    k.set_defaults(func=run)


def run(a) -> int:
    from ..record.signing import SigningUnavailable, keygen

    try:
        info = keygen(a.private, a.public)
    except (FileExistsError, SigningUnavailable, ValueError) as e:
        print(f"error: {e}", file=sys.stderr)
        return 2
    print(f"private key: {info.private_path} (mode 0600; keep it secret, ideally off the training machine)")
    print(f"public key:  {info.public_path} (give this to whoever verifies your runs)")
    print(f"fingerprint: {info.fingerprint}")
    return 0

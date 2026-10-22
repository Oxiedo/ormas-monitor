"""Sealing and signing a run record (Ed25519, via the optional `cryptography` package).

    seal(run_dir)                  writes record.json: canonical JSON naming the SHA-256 of
                                   diary.jsonl and metrics.jsonl, the diary's head hash and length
    sign(run_dir, private.pem)     re-seals with the signer's key fingerprint, then writes
                                   record.sig: a detached Ed25519 signature over the exact bytes
                                   of record.json (64 raw bytes)
    keygen(private.pem, public.pem)

record.json format (canonical JSON: sorted keys, no whitespace, ASCII, no trailing newline):

    {"created": "2026-10-22T10:00:00Z", "diary_head": "<hash of the last diary line>",
     "events": 36, "files": {"diary.jsonl": "<sha256>", "metrics.jsonl": "<sha256>"},
     "format": "ormas-monitor-record", "record_version": <run_start.record_version, null if absent>,
     "signer": null | {"algorithm": "Ed25519", "public_key_sha256": "<sha256 of the raw 32-byte key>"},
     "tool": "ormas-monitor 0.x"}

Threat model, stated plainly:
- The hash chain detects editing, deleting or reordering a line. It does NOT detect someone who
  rewrites the whole diary and recomputes every hash, and alone it cannot see lines cut from the end.
- The seal pins the diary head, its length and the metrics file, so truncation and metrics edits are
  detectable while the seal itself is trusted. Anyone who can write the run directory can also write
  a new seal; an unsigned seal is a checksum, not evidence of who made it.
- A signature binds the seal to whoever holds the private key. It proves the holder signed these
  bytes; it does not prove the training happened as recorded. A key kept on the same machine as the
  trainer proves little more than the chain does, because whoever can edit the record can also use
  the key. A key held by a reviewer, who signs after inspecting the run, proves much more.
- The record is evidence for technical documentation. It is never a compliance certificate.

The format can be checked without this package: `shasum -a 256` for the files, and
`openssl pkeyutl -verify -pubin -inkey public.pem -rawin -in record.json -sigfile record.sig`
for the signature (OpenSSL 3).
"""
from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .chain import file_sha256, parse_lines
from .layout import FORMAT, LAYOUT, RecordLayout

ALGORITHM = "Ed25519"
INSTALL_HINT = "signing needs the optional 'cryptography' package: pip install 'ormas-monitor[signing]'"


class RecordError(Exception):
    """A record that cannot be sealed or signed as asked. Nothing is written when this is raised."""


class SigningUnavailable(ImportError):
    """The optional `cryptography` dependency is not installed."""


@dataclass(frozen=True)
class KeyInfo:
    private_path: Path
    public_path: Path
    fingerprint: str        # SHA-256 of the raw 32-byte public key, hex


def _crypto() -> Any:
    try:
        from cryptography.hazmat.primitives import serialization
        from cryptography.hazmat.primitives.asymmetric import ed25519
    except ImportError as e:
        raise SigningUnavailable(INSTALL_HINT) from e
    return serialization, ed25519


def seal_bytes(seal: dict) -> bytes:
    """The exact bytes written to record.json and signed."""
    return json.dumps(seal, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("ascii")


def fingerprint(public_key: Any) -> str:
    """SHA-256 of an Ed25519 public key's raw 32 bytes, hex."""
    serialization, _ = _crypto()
    raw = public_key.public_bytes(serialization.Encoding.Raw, serialization.PublicFormat.Raw)
    return hashlib.sha256(raw).hexdigest()


def load_public_key(path: str | Path) -> Any:
    """Read an Ed25519 public key from a PEM file. Raises ValueError if it is another key type."""
    serialization, ed25519 = _crypto()
    key = serialization.load_pem_public_key(Path(path).read_bytes())
    if not isinstance(key, ed25519.Ed25519PublicKey):
        raise ValueError(f"{path} is not an Ed25519 public key")
    return key


def load_private_key(path: str | Path) -> Any:
    """Read an unencrypted Ed25519 private key from a PEM file."""
    serialization, ed25519 = _crypto()
    key = serialization.load_pem_private_key(Path(path).read_bytes(), password=None)
    if not isinstance(key, ed25519.Ed25519PrivateKey):
        raise ValueError(f"{path} is not an Ed25519 private key")
    return key


def verify_signature(data: bytes, signature: bytes, public_key: Any) -> bool:
    """True if `signature` is a valid Ed25519 signature of `data` under `public_key`."""
    _crypto()
    from cryptography.exceptions import InvalidSignature
    try:
        public_key.verify(signature, data)
    except InvalidSignature:
        return False
    return True


def _write_exclusive(path: Path, data: bytes, mode: int) -> None:
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, mode)
    with os.fdopen(fd, "wb") as fh:
        fh.write(data)
    os.chmod(path, mode)        # os.open's mode is filtered by the umask; set it exactly


def keygen(private_path: str | Path, public_path: str | Path) -> KeyInfo:
    """Create an Ed25519 key pair as PEM files. The private key is written unencrypted with mode
    0600. Refuses to overwrite either file. For signatures that mean something, keep the private
    key with the person who reviews runs, not on the machine that trains them."""
    serialization, ed25519 = _crypto()
    priv_p, pub_p = Path(private_path), Path(public_path)
    if priv_p.resolve() == pub_p.resolve():
        raise ValueError("private and public key paths must differ")
    for p in (priv_p, pub_p):
        if p.exists():
            raise FileExistsError(f"{p} already exists; refusing to overwrite a key")
    key = ed25519.Ed25519PrivateKey.generate()
    priv = key.private_bytes(serialization.Encoding.PEM, serialization.PrivateFormat.PKCS8,
                             serialization.NoEncryption())
    pub = key.public_key().public_bytes(serialization.Encoding.PEM,
                                        serialization.PublicFormat.SubjectPublicKeyInfo)
    _write_exclusive(priv_p, priv, 0o600)
    try:
        _write_exclusive(pub_p, pub, 0o644)
    except BaseException:
        priv_p.unlink()
        raise
    return KeyInfo(priv_p, pub_p, fingerprint(key.public_key()))


def build_seal(run_dir: str | Path, signer: str | None, layout: RecordLayout = LAYOUT) -> dict:
    """The seal for a run directory as it is now. Does not validate; `seal` and `sign` do."""
    from .. import __version__

    run = Path(run_dir)
    events, _ = parse_lines(run / layout.diary)
    start = events[0] if events else {}
    version = start.get("record_version")     # None: a record that predates the event schema
    return {
        "format": FORMAT,
        "record_version": version,
        "tool": f"ormas-monitor {__version__}",
        "created": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "files": {name: file_sha256(run / name) for name in layout.sealed_files},
        "diary_head": events[-1]["hash"] if events else None,
        "events": len(events),
        "signer": None if signer is None else {"algorithm": ALGORITHM, "public_key_sha256": signer},
    }


def _check_sealable(run: Path, layout: RecordLayout, resigning: bool) -> None:
    """Fail closed: a record that does not verify completely is not sealed."""
    from .verify import verify_run

    if not run.is_dir():
        raise RecordError(f"{run} is not a run directory")
    result = verify_run(run)
    problems = [c for c in result.checks if c.status == "fail" and c.name not in ("seal", "signature")]
    problems += [c for c in result.checks if c.status == "absent" and c.name not in ("seal", "signature")]
    if problems:
        raise RecordError("refusing to seal a record that does not verify: "
                          + "; ".join(f"{c.label}: {c.message}" for c in problems))
    if result.check("seal").status == "fail":
        raise RecordError("refusing to re-seal: the existing record.json no longer matches the files "
                          f"({result.check('seal').message})")
    if not resigning and (run / layout.signature).exists():
        raise RecordError(f"{run / layout.signature} exists; re-sealing would invalidate it. Use sign().")


def seal(run_dir: str | Path, layout: RecordLayout = LAYOUT) -> Path:
    """Write record.json for a finished, intact run and return its path.

    Refuses (RecordError, nothing written) if any verification check fails, if the run has no
    `run_end`, if an existing seal no longer matches the files, or if the run is already signed.
    """
    run = Path(run_dir)
    _check_sealable(run, layout, resigning=False)
    path = run / layout.seal
    path.write_bytes(seal_bytes(build_seal(run, None, layout)))
    return path


def sign(run_dir: str | Path, private_key_path: str | Path, layout: RecordLayout = LAYOUT) -> Path:
    """Seal the run with the signer's public-key fingerprint, sign record.json's exact bytes with
    Ed25519, write the detached signature to record.sig, and return its path.

    Same refusals as `seal`. Re-signing a run that is already signed replaces the old signature,
    but only while the existing seal still matches the files.
    """
    key = load_private_key(private_key_path)
    run = Path(run_dir)
    _check_sealable(run, layout, resigning=True)
    data = seal_bytes(build_seal(run, fingerprint(key.public_key()), layout))
    sig = key.sign(data)
    (run / layout.seal).write_bytes(data)
    out = run / layout.signature
    out.write_bytes(sig)
    return out
